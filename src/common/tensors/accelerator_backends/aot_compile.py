"""General AOT compilation access, for anything (including the torture test)
that wants to compile a real Python function ahead-of-time through the
precompiler, instead of walking a captured tape.

This module is a Python compiler frontend.  Its input contract is Python
source plus an entrypoint; the ``FusedProgram`` objects exposed on the result
are internal products made *after* AST ingestion, ProcessGraph construction,
control planning, and specialization.  They are not an alternate application
entrypoint and must not be mistaken for the scope of Python the compiler
understands.  Consumers compiling an application enter here (or through a
higher-level source/bundle compiler), never by manufacturing a numerical
``FusedProgram`` and calling a backend emitter directly.

This is the real, existing pipeline -- not a new one:

    ast.parse(source)
    -> ProcessGraph.build_from_ast          (transmogrifier/graph/graph_express2.py)
    -> reduce_abstract_tensor_topology       (common/tensors/topological_reducer.py)
    -> strategize_shell_deployment(backend=) (compiler/glsl_deployment_strategy.py)
    -> deployment.compile_process_graph()
    -> deployment.capture_fused_programs(feeds)
    -> deployment.execute_named(feeds)

``strategize_shell_deployment`` (renamed from ``strategize_glsl_deployment``,
which named the function after the one backend it happened to be written
against instead of what it does) is the compilation choke point: every
backend -- c, python, glsl, fortran, webgl, webgpu -- funnels its
ProcessGraph through this one control-planning stage before any
backend-specific emission diverges.  ``shell_language`` is a real, validated
constructor argument -- but as of this writing only ``"glsl"`` has an actual
distinct emission path.  ``emit_glsl`` (the flag
that decides whether GLSL source is produced) is gated purely by
``precompile_only`` inside ``capture_fused_programs``, never by
``shell_language``; ``shell_language`` itself is read in exactly one other
place (a ``device_resident`` hint).  So ``"c"``/``"python"`` would validate
and run but produce the *same* GLSL-emitted, GLSL-executed output as
``"glsl"`` -- there is no separate C or Python code generator behind them.
Picking a shell "kind" by name here would be fake, so this module no longer
does it: ``shell_language`` is always ``"glsl"``, the one real path.  What a
caller actually wants out of a run -- which language happened to serve it,
how long it took, whether it logged anything, how it composed with other
runs -- is what ``dual_ir_shell.DualIRShell`` (``.shell`` below) describes;
it doesn't need a language pick to be meaningful.

Real Fortran AOT exists, but through a different route: get
``compiled_shell_program``/``shell_control_program`` from a
``precompile_only=True`` run (backend-agnostic -- a ``FusedProgram``/
``ControlProgram``, no GLSL involved) and feed those into
``precompile_to_ssa.lower_precompile_and_control_to_ssa`` then
``ssa_fortran_backend.emit_module``/``compile_module``, exactly as
``program_order.order_program`` already does for ``ast=``/
``sympy_expression=``.  See ``docs/PIPELINE_STAGE_DISAMBIGUATION.md`` for
how this fits against the tape-walking JIT backends.

``unroll_limit`` bounds which loops ``remove_loops`` may flatten. The
default of 8 is low for a target that needs a flat program: a loop above
the limit is retained rather than refused, so the caller gets a program
that compiles and runs fewer iterations than it asked for. Set it to the
trip count you expect.

``source`` must be real ``def`` functions, not a lambda -- and, following
the one verified working shape (``tests/test_glsl_fused_network.py``'s
``affine``/``render_value`` pair), the function whose result you want must
call at least one other function defined in the same source rather than
compute its result inline; a single bare top-level function hits an
unrelated, unverified failure in the graph coordinator.

=======================================================================
DO NOT USE SIMULTANEOUS TUPLE ASSIGNMENT FOR A LOOP-CARRIED VALUE.
=======================================================================

    for _ in range(n):
        zx, zy = zx * zx - zy * zy + cx, 2.0 * zx * zy + cy   # DOES NOT LOWER
        zx = zx.minimum(limit)                                 # (rebinding too)

    for _ in range(n):
        next_zx = zx * zx - zy * zy + cx                       # LOWERS
        next_zy = 2.0 * zx * zy + cy
        zx = next_zx.minimum(limit)
        zy = next_zy.minimum(limit)

A tuple assignment binds each name to a tuple temporary, so the loop's
carried update names that temporary rather than a value any region
produces.  ``lower_precompile_and_control_to_ssa`` then reports

    control:loop_carried at root.sequence[N].body;
    carried update value N has no producer inside the loop body

and the lowering is incomplete.  The failure is easy to misread, because
the loop *is* discovered and the control program *does* contain a real
``LoopBlock`` -- only the carried-value wiring is missing.  Assign each
carried name once per iteration, from a value computed in the body.  This
is a restriction of the loop-carried binding analysis, not of Python: the
two forms are the same computation.
"""

import ast
import contextlib
import inspect
import io
import traceback
import types
from pathlib import Path
from dataclasses import dataclass, field, replace
from collections.abc import Iterable
from typing import Any, Callable, Mapping

from ....compiler.glsl_deployment_strategy import (
    ProcessGraphGLSLDeployment,
    _build_hierarchical_glsl_artifact,
    _control_partition_keys,
    _find_nested_loop_node_ids_in_block,
    _loop_reduction_nesting_hints,
    _structural_region_program_from_subgraph,
    _walk_planned_shells,
    propagate_bound_planner_specializations,
    strategize_shell_deployment,
)
from ....compiler.control_source import overlay_scheduled_control
from ....compiler.hierarchical_control import (
    _present_loop_ids,
    compose_hierarchical_control,
)
from ....compiler.hierarchical_plan import (
    assign_hierarchy_ids,
    reduce_hierarchy_identities,
)
from ....compiler.shell_reference_tables import (
    build_class_navigation_table,
    build_map_dependency_regions,
)
from ....compiler.precompile_to_ssa import lower_class_navigation_to_ssa
from ....transmogrifier.graph.graph_express2 import ProcessGraph
from ..abstraction import AbstractTensor
from ..topological_reducer import (
    _normalize_lexical_values,
    reduce_abstract_tensor_topology,
)
from ..fused_ir import FusedProgram
from .dual_ir_shell import DualIRShell, compose_dual_ir_shell
from .aot_checkpoint import (
    AOTCheckpointStore,
    ReductionArtifactStore,
    callable_digest,
)


AOT_BAKE_MODES = frozenset({"one_shot", "whole_program"})
AOT_SCHEDULE_PREFERENCES = frozenset({"asap", "alap"})


def normalize_aot_bake_mode(value: str) -> str:
    """Validate the public choice between trace and retained-control baking."""

    mode = str(value).strip().lower().replace("-", "_")
    if mode not in AOT_BAKE_MODES:
        raise ValueError(
            "bake_mode must be 'one_shot' or 'whole_program', "
            f"not {value!r}"
        )
    return mode


def normalize_aot_schedule_preference(value: str) -> str:
    """Validate the pinned placement preference used by every deployment."""

    preference = str(value).strip().lower()
    if preference not in AOT_SCHEDULE_PREFERENCES:
        raise ValueError(
            "schedule_preference must be 'asap' or 'alap', "
            f"not {value!r}"
        )
    return preference


def _source_dependency_is_not_tensor_primitive(value: Any) -> bool:
    """Keep the source linker above the established tensor-op boundary.

    ``AbstractTensor`` methods are the compiler's numerical vocabulary.  If
    dependency discovery recursively ingests their Python dispatch wrappers,
    an ordinary ``AT.tensor`` call expands into backend selection, autograd,
    and tape object construction instead of remaining the tensor operation
    the reducer already knows how to lower.
    """

    target = value.__func__ if inspect.ismethod(value) else value
    owner = str(getattr(target, "__qualname__", "")).split(".", 1)[0]
    module = str(getattr(target, "__module__", ""))
    if not module.startswith("src."):
        return False
    if module.endswith(".debug"):
        return False
    for name in dir(AbstractTensor):
        candidate = getattr(AbstractTensor, name, None)
        candidate = (
            candidate.__func__ if inspect.ismethod(candidate) else candidate
        )
        if target is candidate:
            return False
    return not (
        owner == "AbstractTensor"
        and module == "src.common.tensors.abstraction"
    )


def _expand_python_static_bindings(
    bindings: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve referenced globals of supplied functions and classes."""

    expanded = dict(bindings or {})
    queue = list(expanded.values())
    visited: set[int] = set()
    while queue:
        value = queue.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        targets = (
            tuple(vars(value).values())
            if inspect.isclass(value)
            else (value.__func__ if inspect.ismethod(value) else value,)
        )
        for target in targets:
            code = getattr(target, "__code__", None)
            namespace = getattr(target, "__globals__", None)
            if code is None or not isinstance(namespace, dict):
                continue
            pending_codes = [code]
            seen_codes: set[int] = set()
            while pending_codes:
                nested_code = pending_codes.pop()
                if id(nested_code) in seen_codes:
                    continue
                seen_codes.add(id(nested_code))
                pending_codes.extend(
                    constant
                    for constant in nested_code.co_consts
                    if isinstance(constant, types.CodeType)
                )
                for name in nested_code.co_names:
                    if name not in namespace:
                        continue
                    dependency = namespace[name]
                    if name not in expanded:
                        expanded[name] = dependency
                        queue.append(dependency)
    return expanded


@dataclass(frozen=True)
class AOTCompilation:
    entrypoint: str
    outputs: Mapping[str, Any]
    # INTERNAL COMPILATION PRODUCT.  This is exposed so later compiler stages
    # can assemble control/numeric SSA and emit targets.  Its flat numerical
    # shape does not describe or limit the Python source accepted above it.
    compiled_shell_program: Any
    shell_control_program: Any
    deployment: Any
    shell: DualIRShell
    map_ir: Mapping[str, Any] | None = None
    # The numeric program for each ``__scheduled_region_N__`` the control
    # program references. ``lower_precompile_and_control_to_ssa`` needs these
    # to find the producers of a loop's carried updates; without them a
    # program whose control shell has regions cannot be lowered.
    region_programs: Mapping[int, Any] = field(default_factory=dict)
    # ProcessGraph-owned operator order with optional implementation evidence
    # for each region. Tensor kernels remain FusedPrograms in
    # ``region_programs``; plain and structural operators remain graph lines.
    planned_operator_implementations: Mapping[int, Any] = field(
        default_factory=dict
    )
    hierarchy_plan: Any = None
    region_feed_values: Mapping[int, Any] = field(default_factory=dict)
    # Retained by the reduced function ProcessGraph: source name -> complete
    # canonical value-ID history. ``function_outputs`` selects public names.
    identity_table: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    function_outputs: tuple[str, ...] = ()
    function_parameters: tuple[str, ...] = ()
    # Global value identities authored by hierarchical composition.  These
    # bridge source names (including aggregate paths such as ``state.u``) to
    # the captured region/control namespace; local ProcessGraph node IDs are
    # not interchangeable with these values.
    public_input_value_ids: Mapping[str, int] = field(default_factory=dict)
    public_output_value_ids: Mapping[str, int] = field(default_factory=dict)
    hierarchical_value_diagnostics: Mapping[int, Any] = field(
        default_factory=dict
    )
    hierarchical_value_aliases: Mapping[int, int] = field(
        default_factory=dict
    )
    # Planner-owned control that could not be represented by the exported
    # ControlProgram.  A whole-program backend must reject these rather than
    # silently baking the numerical trace observed during discovery.
    control_shortfalls: tuple[Mapping[str, Any], ...] = ()
    # ``one_shot`` packages the numerical trace produced by the discovery
    # execution. ``whole_program`` requires consumers to retain the planned
    # ControlProgram and its real region kernels.
    bake_mode: str = "whole_program"
    schedule_preference: str = "alap"
    # The full source record remains authoritative. A configured record fixes
    # selected public parameters to literals before ProcessGraph reduction.
    program_record_mode: str = "full"
    constant_map: Mapping[str, Any] = field(default_factory=dict)
    mutable_parameters: tuple[str, ...] = ()
    # The real MapDependencyRegions/ClassNavigationTable objects
    # (shell_reference_tables.py), not the flattened/buried copies that
    # otherwise only survive inside map_ir. See
    # GRAPH_DESCRIPTION_LAYER_SURVEY.md for why these are kept as typed
    # fields instead of being reduced to another dict entry.
    class_navigation: Any = None
    dependency_regions: Any = None


class _BakeDeployment:
    """The only thing the whole-program bake reads off a full deployment: the
    process graph (for the inspection-panel summary).

    The planning deployment that produced a program is 2.8-3.5 GB -- millions of
    control-IR objects -- and the old resume path deserialized all of it purely
    to hand the bake ``deployment.process_graph`` (~10 MB). Standing in this
    light holder lets a precompile bake resume from the 79 MB captured program
    plus the cheap source-graph checkpoint, never touching the multi-GB plan.
    That is the difference between a resume that swaps the whole time and one
    that does not.
    """

    __slots__ = ("process_graph",)

    def __init__(self, process_graph):
        self.process_graph = process_graph


# Manual versions for checkpoint implementation digests. These exist so that
# editing a phase's orchestration function (logging, resume plumbing, loop
# shape) does NOT invalidate its multi-GB checkpoint -- only a deliberate bump,
# or a change to one of the genuine content functions still listed in the
# recipe, does. Bump when that phase's produced content actually changes.
_PREPARED_RECIPE_VERSION = 1
_CAPTURE_RECIPE_VERSION = 1


def _apply_parameter_constant_map(
    module: ast.Module,
    entrypoint: str,
    constant_map: Mapping[str, Any],
    mutable_parameters: tuple[str, ...] = (),
) -> ast.Module:
    """Replace configured parameter reads before topology reduction."""

    if not constant_map:
        return module
    function = next(
        (
            node for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == entrypoint
        ),
        None,
    )
    if function is None:
        raise ValueError(f"configured entrypoint {entrypoint!r} is absent")
    parameters = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }
    unknown = set(map(str, constant_map)) - parameters
    if unknown:
        raise ValueError(
            "configured constants name unknown parameters: "
            + ", ".join(sorted(unknown))
        )
    forbidden = set(map(str, constant_map)) & set(map(str, mutable_parameters))
    if forbidden:
        raise ValueError(
            "configured constants cannot freeze mutable parameters: "
            + ", ".join(sorted(forbidden))
        )
    replacements: dict[str, ast.expr] = {}
    normalized: dict[str, Any] = {}
    for name, value in constant_map.items():
        try:
            expression = ast.parse(repr(value), mode="eval").body
            literal = ast.literal_eval(expression)
        except (SyntaxError, ValueError, TypeError) as error:
            raise ValueError(
                f"configured constant {name!r} must be a Python literal"
            ) from error
        replacements[str(name)] = expression
        normalized[str(name)] = literal

    class ReplaceConfiguredReads(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id in replacements:
                return ast.copy_location(
                    ast.fix_missing_locations(
                        ast.parse(repr(normalized[node.id]), mode="eval").body
                    ),
                    node,
                )
            return node

    ReplaceConfiguredReads().visit(function)
    ast.fix_missing_locations(module)
    return module


def project_public_numerical_program(compilation: AOTCompilation) -> Any:
    """Project one numeric region through the retained source identity map.

    Discovery regions may retain values needed by surrounding control and
    therefore expose more numeric terminals than the source function returns.
    Backend APIs must describe the public function, not those compiler-private
    boundaries. When one region contains every declared public output, select
    and name those values from ``function_outputs``/``identity_table`` and
    restore source parameter names on its feeds.

    Multi-region/control programs remain unchanged; their public ABI is
    assembled by the SSA/control path rather than fabricated by flattening.
    """

    outputs = {
        name: int(compilation.identity_table[name][-1])
        for name in compilation.function_outputs
        if compilation.identity_table.get(name)
    }
    fallback = getattr(
        compilation.compiled_shell_program,
        "program",
        compilation.compiled_shell_program,
    )
    if not outputs or len(outputs) != len(compilation.function_outputs):
        return fallback

    candidates = [*compilation.region_programs.values(), fallback]
    selected = []
    for candidate in candidates:
        available = {
            *map(int, candidate.feeds),
            *(int(step.result_id) for step in candidate.steps),
            *map(int, candidate.outputs.values()),
        }
        if set(outputs.values()) <= available:
            selected.append(candidate)
    if len(selected) != 1:
        return fallback

    program = selected[0]
    # Region assembly can encounter the same captured literal through more
    # than one returned expression.  Both references deliberately retain the
    # same identity, so collapse only byte-for-byte-equivalent definitions;
    # a differing redefinition remains visible to downstream validation.
    steps = []
    definitions: dict[int, Any] = {}
    for step in program.steps:
        result_id = int(step.result_id)
        previous = definitions.get(result_id)
        if previous is not None and previous == step:
            continue
        definitions[result_id] = step
        steps.append(step)
    extras = dict(program.extras or {})
    origins = dict(extras.get("capture_feed_origins", {}) or {})
    for name in compilation.function_parameters:
        history = tuple(compilation.identity_table.get(name, ()))
        if history and int(history[0]) in program.feeds:
            origins[int(history[0])] = {"binding_name": name}
    extras["capture_feed_origins"] = origins
    return FusedProgram(
        version=program.version,
        feeds=set(program.feeds),
        steps=steps,
        outputs=outputs,
        state_in=None if program.state_in is None else set(program.state_in),
        meta=None if program.meta is None else dict(program.meta),
        extras=extras,
    )


def _mutable_feed_signature(value: Any) -> Mapping[str, Any]:
    """Describe a runtime feed without fingerprinting its sample value."""

    signature: dict[str, Any] = {
        "runtime_type": (
            f"{type(value).__module__}.{type(value).__qualname__}"
        ),
    }
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            signature["shape"] = tuple(map(int, shape))
        except (TypeError, ValueError):
            pass
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        signature["dtype"] = str(dtype)
    device = getattr(value, "device", None)
    if device is not None:
        signature["device"] = str(device)
    return signature


def prepare_aot_checkpoint_store(
    source: str,
    entrypoint: str,
    feeds: Mapping[str, Any],
    *,
    backend: str = "c",
    remove_loops: bool = False,
    unroll_limit: int = 8,
    bake_mode: str = "whole_program",
    schedule_preference: str = "alap",
    constant_map: Mapping[str, Any] | None = None,
    mutable_parameters: tuple[str, ...] | list[str] | set[str] = (),
    retain: Any = (),
    python_bindings: Mapping[str, Any] | None = None,
    checkpoint: bool | str | Path = False,
) -> tuple[Any, str, str, str, str, Mapping[str, Any]]:
    """Build the checkpoint store and phase digests ``compile_ast_aot`` resumes
    from, so a caller that already built its own ``ProcessGraph`` (site_bundle's
    ``build_program_bundle``) can store it under the exact identity
    ``compile_ast_aot`` will look for moments later -- instead of
    ``compile_ast_aot`` silently rebuilding a second, independent one from the
    same source.

    Returns ``(checkpoint_store, frontend_implementation,
    source_graph_implementation, bake_mode, schedule_preference,
    checkpoint_feeds)``. ``bake_mode``/``schedule_preference`` are the
    normalized forms: the checkpoint identity is keyed off them, so every
    caller must normalize the same way or the identity silently diverges and
    the store is never resumed.
    """

    bake_mode = normalize_aot_bake_mode(bake_mode)
    schedule_preference = normalize_aot_schedule_preference(
        schedule_preference
    )
    constant_map = dict(constant_map or {})
    mutable_parameters = tuple(dict.fromkeys(map(str, mutable_parameters)))

    frontend_implementation = callable_digest(
        ProcessGraph.build_from_ast,
        reduce_abstract_tensor_topology,
        _normalize_lexical_values,
        propagate_bound_planner_specializations,
        build_map_dependency_regions,
        build_class_navigation_table,
    )
    source_graph_implementation = callable_digest(
        ProcessGraph.build_from_ast,
        _apply_parameter_constant_map,
    )
    checkpoint_feeds = {
        str(name): (
            _mutable_feed_signature(value)
            if str(name) in mutable_parameters
            else value
        )
        for name, value in feeds.items()
    }
    if not checkpoint:
        return (
            None,
            frontend_implementation,
            source_graph_implementation,
            bake_mode,
            schedule_preference,
            checkpoint_feeds,
        )

    binding_values = tuple(
        value
        for _name, value in sorted(
            (python_bindings or {}).items(), key=lambda pair: str(pair[0])
        )
        if callable(value)
    )
    checkpoint_store = AOTCheckpointStore(
        {
            "source": source,
            "entrypoint": entrypoint,
            "feeds": checkpoint_feeds,
            "backend": backend,
            "remove_loops": bool(remove_loops),
            "unroll_limit": int(unroll_limit),
            "bake_mode": bake_mode,
            "schedule_preference": schedule_preference,
            "constant_map": constant_map,
            "mutable_parameters": mutable_parameters,
            "retain": retain,
            "python_binding_sources": callable_digest(*binding_values),
        },
        None if checkpoint is True else checkpoint,
    )
    return (
        checkpoint_store,
        frontend_implementation,
        source_graph_implementation,
        bake_mode,
        schedule_preference,
        checkpoint_feeds,
    )


def compile_ast_aot(
    source: str,
    entrypoint: str,
    feeds: Mapping[str, Any],
    *,
    backend: str = "c",
    remove_loops: bool = False,
    unroll_limit: int = 8,
    profiling: bool = False,
    precompile_only: bool = False,
    python_bindings: Mapping[str, Any] | None = None,
    python_package: str | None = None,
    bake_mode: str = "whole_program",
    schedule_preference: str = "alap",
    constant_map: Mapping[str, Any] | None = None,
    mutable_parameters: tuple[str, ...] | list[str] | set[str] = (),
    retain: Any = (),
    progress: "Callable[[str], None] | None" = None,
    checkpoint: bool | str | Path = False,
    resume: bool = True,
) -> AOTCompilation:
    """Compile ``entrypoint`` in ``source`` ahead-of-time and execute it once.

    ``backend`` tags the loop-composition capability profile
    (``LoopBackendCapabilities``) used while planning -- a real choice, since
    it decides whether loops stay native or get removed/kpn-composed.  It no
    longer picks a ``shell_language``: that is always ``"glsl"`` now (see
    this module's docstring), since ``"c"``/``"python"`` never had a
    distinct emission path to pick.  ``compiled_shell_program``/
    ``shell_control_program`` (the ``FusedProgram``/``ControlProgram`` this
    run captured) are returned regardless of backend, so a caller can feed
    them into ``precompile_to_ssa.lower_precompile_and_control_to_ssa`` for a
    backend that only consumes SSA (Fortran today, via
    ``ssa_fortran_backend``).  The returned ``.shell`` is the
    ``DualIRShell`` describing the same numeric/control pair.
    """

    def _report(message: str) -> None:
        if progress is not None:
            progress(message)

    constant_map = dict(constant_map or {})
    mutable_parameters = tuple(dict.fromkeys(map(str, mutable_parameters)))
    expanded_python_bindings = _expand_python_static_bindings(
        python_bindings
    )

    (
        checkpoint_store,
        frontend_implementation,
        source_graph_implementation,
        bake_mode,
        schedule_preference,
        checkpoint_feeds,
    ) = prepare_aot_checkpoint_store(
        source,
        entrypoint,
        feeds,
        backend=backend,
        remove_loops=remove_loops,
        unroll_limit=unroll_limit,
        bake_mode=bake_mode,
        schedule_preference=schedule_preference,
        constant_map=constant_map,
        mutable_parameters=mutable_parameters,
        retain=retain,
        python_bindings=python_bindings,
        checkpoint=checkpoint,
    )
    # A compiled plan contains the frontend's resolved call identities and
    # reduced topology, not merely the scheduler's output.  Any frontend IR
    # change must therefore invalidate the downstream plan checkpoint; a
    # capture/coordinator-only change may continue to reuse it.
    planning_implementation = callable_digest(
        strategize_shell_deployment,
        _control_partition_keys,
        frontend_implementation,
    )
    prepared_implementation = callable_digest(
        # A manual version plus the real region builder, NOT
        # ``prepare_graph_precompile``'s source. The prepared plan's content is
        # the region programs ``_structural_region_program_from_subgraph``
        # produces; ``prepare_graph_precompile`` around it is orchestration
        # (the per-region loop, logging, the reduction-cache plumbing) whose
        # edits must not invalidate a 3.5 GB checkpoint. Bump the version when
        # the prepare-phase content genuinely changes.
        _PREPARED_RECIPE_VERSION,
        _structural_region_program_from_subgraph,
        _loop_reduction_nesting_hints,
        overlay_scheduled_control,
        planning_implementation,
    )
    capture_implementation = callable_digest(
        # A manual version, NOT ``compile_ast_aot``'s source. This digest must
        # invalidate when the captured program's *content* changes, and the
        # content-affecting logic lives in the helpers listed below plus the
        # graph recipes folded in through ``prepared_implementation`` -- while
        # ``compile_ast_aot`` itself is orchestration (resume order, feed
        # marshalling, checkpoint plumbing) whose inputs are already pinned by
        # the checkpoint identity (source/entrypoint/feeds/constants). Hashing
        # ``compile_ast_aot``'s source made the function that *loads* this
        # checkpoint also *invalidate* it: adding a resume fast-path silently
        # dropped a 79 MB captured program and forced a 3.5 GB plan reload.
        # Bump this when a genuine capture-content change lands here.
        _CAPTURE_RECIPE_VERSION,
        _build_hierarchical_glsl_artifact,
        # ``_build_hierarchical_glsl_artifact``'s own source is what
        # ``callable_digest`` sees -- it does not walk call targets. These
        # are the real hierarchy-composition dependencies whose behavior
        # this phase's cached output actually depends on; a checkpoint must
        # invalidate when any of them changes, not only when the one
        # function whose name appears in this list is edited.
        compose_hierarchical_control,
        _present_loop_ids,
        reduce_hierarchy_identities,
        assign_hierarchy_ids,
        _find_nested_loop_node_ids_in_block,
        prepared_implementation,
    )

    deployment = None
    deployment_prepared = False
    # Bake fast-path. A precompile bake consumes the captured program (79 MB:
    # region_programs, shell_control_program, control_shortfalls) plus the
    # process graph, and nothing else off the planning deployment. If the
    # captured-program checkpoint is already on disk, resume straight from it
    # and pair it with the cheap source-graph checkpoint for the process graph
    # -- so the resume loads ~89 MB instead of deserializing the 2.8-3.5 GB
    # plan it would otherwise pull in only to read one ~10 MB attribute.
    if checkpoint_store is not None and resume and precompile_only:
        _report("aot: probing captured-program checkpoint (bake fast-path)")
        captured = checkpoint_store.load(
            "captured_program", capture_implementation
        )
        if isinstance(captured, AOTCompilation):
            bake_graph = getattr(captured.deployment, "process_graph", None)
            if bake_graph is None:
                bake_graph = checkpoint_store.load(
                    "source_graph", source_graph_implementation
                )
            if bake_graph is not None:
                _report(
                    "aot: resumed captured-program checkpoint via bake "
                    "fast-path (skipped multi-GB plan load)"
                )
                return replace(
                    captured, deployment=_BakeDeployment(bake_graph)
                )
            _report(
                "aot: bake fast-path unavailable (no process graph on disk); "
                "falling back to full plan resume"
            )
        else:
            _report(
                "aot: captured-program checkpoint unavailable for fast-path "
                f"({checkpoint_store.last_load_status})"
            )
    if checkpoint_store is not None and resume:
        _report("aot: loading prepared-plan checkpoint")
        deployment = checkpoint_store.load(
            "prepared_plan",
            prepared_implementation,
        )
        if deployment is not None:
            deployment_prepared = True
            _report("aot: resumed prepared-plan checkpoint")
            graph = deployment.process_graph
            map_ir = dict(graph.G.graph.get("map_ir") or {})
        else:
            _report(
                "aot: prepared-plan checkpoint unavailable "
                f"({checkpoint_store.last_load_status})"
            )
    if deployment is None and checkpoint_store is not None and resume:
        _report("aot: loading compiled-plan checkpoint")
        deployment = checkpoint_store.load(
            "compiled_plan",
            planning_implementation,
        )
        if deployment is not None:
            _report("aot: resumed compiled-plan checkpoint")
            graph = deployment.process_graph
            map_ir = dict(graph.G.graph.get("map_ir") or {})
        else:
            _report(
                "aot: compiled-plan checkpoint unavailable "
                f"({checkpoint_store.last_load_status})"
            )

    graph = None if deployment is None else graph
    frontend_ready = graph is not None
    # Only computed fresh below (not on a resumed frontend/compiled-plan
    # checkpoint) -- a resumed compile carries these as None rather than
    # re-deriving them, which is a known gap, not a silent wrong answer.
    class_navigation = None
    dependency_regions = None
    # Defensive default only -- every traced path assigns a real map_ir
    # before use (compiled-plan/frontend checkpoint resume, or fresh build
    # inside _lower_process_graph_to_compilation). Kept for the same reason
    # as the two defaults above: crossing a function boundary now, so an
    # untraced edge case fails loud (None reaching somewhere that expects a
    # mapping) rather than as an UnboundLocalError with a confusing frame.
    map_ir = None
    # The entrypoint's own FunctionDef node, found unambiguously in this
    # call's own freshly-parsed source below (module.body has exactly one
    # top-level function named `entrypoint`, by construction). Used later to
    # look the entrypoint's reference up by node identity
    # (FunctionTable.reference_by_source_node) instead of by bare name
    # (FunctionTable.reference) -- see that call site for why bare-name
    # lookup is unsafe here. None on a resumed frontend/compiled-plan
    # checkpoint, where this call never re-parses its own source; the
    # bare-name lookup remains the fallback for that path specifically.
    entrypoint_node_id = None
    if deployment is None and checkpoint_store is not None and resume:
        _report("aot: loading frontend checkpoint")
        graph = checkpoint_store.load(
            "frontend",
            frontend_implementation,
        )
        if graph is not None:
            _report("aot: resumed frontend checkpoint")
            map_ir = dict(graph.G.graph.get("map_ir") or {})
            frontend_ready = True
        else:
            _report(
                "aot: frontend checkpoint unavailable "
                f"({checkpoint_store.last_load_status})"
            )

    if graph is None and checkpoint_store is not None and resume:
        _report("aot: loading source-graph checkpoint")
        graph = checkpoint_store.load(
            "source_graph",
            source_graph_implementation,
        )
        if graph is not None:
            _report("aot: resumed source-graph checkpoint")
        else:
            _report(
                "aot: source-graph checkpoint unavailable "
                f"({checkpoint_store.last_load_status})"
            )

    if graph is None:
        _report("aot: applying constant map")
        module = _apply_parameter_constant_map(
            ast.parse(source), entrypoint, constant_map, mutable_parameters
        )
        entrypoint_function_def = next(
            (
                node for node in module.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == entrypoint
            ),
            None,
        )
        if entrypoint_function_def is not None:
            entrypoint_node_id = id(entrypoint_function_def)
    # This graph is a second, independent ProcessGraph build -- the caller
    # (site_bundle.build_program_bundle) already built and reduced one of
    # its own moments ago for telemetry/summarize_process_graph. They are
    # not the same object and this does not reuse that work; real, currently
    # unavoidable duplicate compute, not just an unlogged phase.
        graph = ProcessGraph(materialize_memory=False)
    # AOT compilation may target a function from a live module.  Its resolved
    # globals are static closure values, not runtime tensor feeds.  Capturing
    # them here lets the reducer retain computed constants and imported
    # references without executing or reinterpreting their source expressions.
        graph.python_bindings = dict(expanded_python_bindings)
        # Real import resolution (_import_ast_bindings -> importlib.
        # import_module) inside build_from_ast needs this to resolve a
        # relative import (``from .machine_path_forest import ...``) at
        # all -- without it, importlib silently fails (a relative import
        # requires ``package``) and a name the source itself imports is
        # never discovered, indistinguishable from one that truly is
        # external and must be supplied.  site_bundle.py's own, separate,
        # first ProcessGraph build already sets this; this second,
        # independent graph build never did.
        graph.python_package = python_package
        _report("aot: building process graph (second, independent build)")
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(
                module,
                resolve_unresolved_parents=True,
                parent_include=_source_dependency_is_not_tensor_primitive,
                retain=retain,
                progress=_report,
            )
        if checkpoint_store is not None:
            _report("aot: saving source-graph checkpoint")
            try:
                checkpoint_store.store(
                    "source_graph",
                    source_graph_implementation,
                    graph,
                )
            except Exception as error:
                _report(
                    "aot: source-graph checkpoint skipped "
                    f"({type(error).__name__}: {error})"
                )

    # Everything from here on operates on ``graph`` (a built ProcessGraph)
    # and ``entrypoint_node_id`` (the entrypoint's own node identity)
    # generically -- nothing below re-touches Python syntax. This is the one
    # permitted higher (ProcessGraph) -> lower (SSA/FusedProgram/
    # ControlProgram/DualIRShell) crossing point every language's ingestion
    # is meant to converge on (see PROCESS_GRAPH_LOWERING_SEAM_HANDOFF.md and
    # GRAPH_DESCRIPTION_LAYER_SURVEY.md's "two convergence layers" section).
    # Lateral transformations at the same level (SSA rewriting, binary
    # relifting) are unaffected by this rule and continue to happen wherever
    # they already do. Raising (lower -> higher, e.g. a hypothetical
    # SSA->ProcessGraph decompilation) is a deliberately separate, deferred
    # question -- do not fold it into this function.
    return _lower_process_graph_to_compilation(
        graph, entrypoint_node_id, entrypoint, feeds,
        backend=backend,
        remove_loops=remove_loops,
        unroll_limit=unroll_limit,
        profiling=profiling,
        precompile_only=precompile_only,
        expanded_python_bindings=expanded_python_bindings,
        bake_mode=bake_mode,
        schedule_preference=schedule_preference,
        constant_map=constant_map,
        mutable_parameters=mutable_parameters,
        progress=progress,
        checkpoint_store=checkpoint_store,
        checkpoint_feeds=checkpoint_feeds,
        frontend_implementation=frontend_implementation,
        source_graph_implementation=source_graph_implementation,
        planning_implementation=planning_implementation,
        prepared_implementation=prepared_implementation,
        capture_implementation=capture_implementation,
        deployment=deployment,
        deployment_prepared=deployment_prepared,
        frontend_ready=frontend_ready,
        class_navigation=class_navigation,
        dependency_regions=dependency_regions,
        map_ir=map_ir,
        resume=resume,
    )


def _lower_process_graph_to_compilation(
    graph: Any,
    entrypoint_node_id: int | None,
    entrypoint: str,
    feeds: Mapping[str, Any],
    *,
    backend: str,
    remove_loops: bool,
    unroll_limit: int,
    profiling: bool,
    precompile_only: bool,
    expanded_python_bindings: Mapping[str, Any],
    bake_mode: str,
    schedule_preference: str,
    constant_map: Mapping[str, Any],
    mutable_parameters: tuple[str, ...],
    progress: "Callable[[str], None] | None",
    checkpoint_store: Any,
    checkpoint_feeds: Mapping[str, Any],
    frontend_implementation: str,
    source_graph_implementation: str,
    planning_implementation: str,
    prepared_implementation: str,
    capture_implementation: str,
    deployment: Any,
    deployment_prepared: bool,
    frontend_ready: bool,
    class_navigation: Any,
    dependency_regions: Any,
    map_ir: Mapping[str, Any] | None,
    resume: bool,
) -> AOTCompilation:
    """Lower an already-built ``ProcessGraph`` into a real ``AOTCompilation``.

    This is the shared "higher -> lower" seam every language's ingestion
    converges on -- see the comment at this function's one call site in
    ``compile_ast_aot`` for the full rule. Nothing here is Python-specific:
    ``graph`` is a built ``ProcessGraph`` regardless of the language that
    produced it, and ``entrypoint_node_id`` is the entrypoint's own node
    identity in that graph (an ``ast.FunctionDef`` for Python, a
    ``pycparser`` ``FuncDef`` for the C++-like shell, resolved by whichever
    ingestion wrapper called this function -- never re-derived here by
    name, which is exactly the bug ``FunctionTable.reference_by_source_node``
    exists to route around).
    """

    def _report(message: str) -> None:
        if progress is not None:
            progress(message)

    if not frontend_ready:
        _report("aot: reducing abstract tensor topology")
        reduce_abstract_tensor_topology(graph)
        _report("aot: propagating bound planner specializations")
        propagate_bound_planner_specializations(
            graph,
            entrypoint,
            feeds,
            mutable_parameters=mutable_parameters,
        )
        _report("aot: building map dependency regions")
        dependency_regions = build_map_dependency_regions(graph, entrypoint)
        map_ir = dict(graph.G.graph.get("map_ir") or {})
        map_ir["dependency_regions"] = {
            "runtime": dependency_regions.runtime,
            "mapped": dependency_regions.mapped,
            "retained": dependency_regions.retained,
            "map_only": dependency_regions.map_only,
            "bindings": dependency_regions.bindings,
        }
        _report("aot: building class navigation table")
        class_navigation = build_class_navigation_table(graph)
        map_ir["class_navigation"] = class_navigation
        navigation_ssa = lower_class_navigation_to_ssa(class_navigation)
        map_ir["semantic_methods"] = tuple(
            {
                "function": function.name,
                "operations": tuple(dict.fromkeys(
                    instruction.op
                    for instruction in function.blocks["entry"].instrs
                )),
            }
            for function in navigation_ssa.functions.values()
        )
        graph.G.graph["map_ir"] = map_ir
        if checkpoint_store is not None:
            _report("aot: saving frontend checkpoint")
            try:
                checkpoint_store.store(
                    "frontend",
                    frontend_implementation,
                    graph,
                )
            except Exception as error:
                _report(
                    "aot: frontend checkpoint skipped "
                    f"({type(error).__name__}: {error})"
                )

    if deployment is None:
        _report("aot: strategizing glsl deployment (scheduling/planning pass)")
        deployment_type = strategize_shell_deployment(
            graph,
            backend=backend,
            remove_loops=remove_loops,
            unroll_limit=unroll_limit,
            schedule_preference=schedule_preference,
        )
        _report("aot: glsl deployment strategy selected")
    # shell_language is fixed at "glsl", the one path that actually emits
    # something distinct -- see this module's docstring. It is not derived
    # from backend= any more; there is no shell "kind" left to pick.
        _report("aot: constructing glsl deployment plan")
        deployment = deployment_type(profiling=profiling, shell_language="glsl")
        _report("aot: glsl deployment plan constructed")
    try:
        planned_shells = tuple(_walk_planned_shells(deployment))

        def store_deployment_phase(phase: str, implementation: str) -> None:
            saved_static_bindings = []
            try:
                for planned_shell in planned_shells:
                    if "static_python_bindings" not in planned_shell.__dict__:
                        continue
                    saved_static_bindings.append((
                        planned_shell,
                        planned_shell.__dict__.pop("static_python_bindings"),
                    ))
                checkpoint_store.store(phase, implementation, deployment)
            finally:
                for planned_shell, bindings in saved_static_bindings:
                    planned_shell.static_python_bindings = bindings

        for planned_shell in planned_shells:
            planned_shell.static_python_bindings = {
                **expanded_python_bindings,
                **planned_shell.static_python_bindings,
            }
        if not getattr(deployment, "whole_program_compiled", False):
            _report("aot: compile_process_graph (usually the largest phase)")
            deployment.compile_process_graph()
            if checkpoint_store is not None:
                _report("aot: saving compiled-plan checkpoint")
                try:
                    store_deployment_phase(
                        "compiled_plan",
                        planning_implementation,
                    )
                except Exception as error:
                    _report(
                        "aot: compiled-plan checkpoint skipped "
                        f"({type(error).__name__}: {error})"
                    )
        if checkpoint_store is not None and resume and precompile_only:
            _report("aot: loading captured-program checkpoint")
            captured_compilation = checkpoint_store.load(
                "captured_program",
                capture_implementation,
            )
            if isinstance(captured_compilation, AOTCompilation):
                _report("aot: resumed captured-program checkpoint")
                return replace(
                    captured_compilation,
                    deployment=deployment,
                )
            _report(
                "aot: captured-program checkpoint unavailable "
                f"({checkpoint_store.last_load_status})"
            )
        # FunctionTable.declare() binds every function under its own bare,
        # unqualified name (FunctionTable._bindings), last writer wins --
        # there is no collision detection there at all, only the qualified
        # name (FunctionTable._qualified) is disambiguated. A whole-program
        # discovery trace routinely declares more than one function sharing
        # the entrypoint's bare name (a same-named method on a class the
        # trace happens to reach), so looking the entrypoint up by name here
        # silently returns whichever same-named function was declared last --
        # not necessarily this compile's actual entrypoint. Resolve by the
        # entrypoint's own source node identity instead (collision-proof by
        # construction: two functions can share a name, never a source
        # node); the bare-name lookup remains the fallback only for a
        # resumed frontend/compiled-plan checkpoint, which never re-parses
        # its own source and so has no node id to resolve by.
        reference = (
            graph.function_table.reference_by_source_node(entrypoint_node_id)
            if entrypoint_node_id is not None
            else None
        ) or graph.function_table.reference(entrypoint)
        if reference is None:
            raise ValueError(
                f"{entrypoint!r} is not a defined function in source"
            )
        # deployment.function_shells indexes ProcessGraphGLSLDeployment's own
        # per-function sub-deployments by address -- an unrelated, internal
        # sense of "shell" from glsl_deployment_strategy.py, not this
        # module's DualIRShell.
        function_shell = deployment.function_shells.get(reference.address, deployment)
        feeds = dict(feeds)
        missing_mutable_parameters = tuple(
            name for name in mutable_parameters if name not in feeds
        )
        if precompile_only and missing_mutable_parameters:
            if deployment_prepared:
                _report("aot: graph-only precompile already prepared")
            else:
                _report(
                    "aot: preparing graph-only precompile; runtime parameters "
                    + repr(missing_mutable_parameters)
                )
                # Per-region logging makes this otherwise-silent portion
                # visible. The content-addressed catalogue that would also make
                # it an incremental backup is intentionally OFF for now: a
                # region's key would have to be stable across builds to ever be
                # reused, but the build is not deterministic (node ids shift
                # under hash randomization, and node data carries opaque objects
                # that do not serialize stably even with a fixed seed). Writing
                # keys that never match would only add pickling overhead and
                # unbounded cache growth. The plumbing stays so this switches on
                # once the build is made deterministic and regions get a sound
                # canonical key. See the design note for the load/unload plan.
                region_catalogue = None
                function_shell.prepare_graph_precompile(
                    reduction_cache=region_catalogue,
                    progress=_report,
                )
                if checkpoint_store is not None:
                    _report("aot: saving prepared-plan checkpoint")
                    try:
                        store_deployment_phase(
                            "prepared_plan",
                            prepared_implementation,
                        )
                    except Exception as error:
                        _report(
                            "aot: prepared-plan checkpoint skipped "
                            f"({type(error).__name__}: {error})"
                        )
        else:
            _report("aot: capturing fused programs")
            function_shell.capture_fused_programs(
                feeds, precompile_only=precompile_only
            )
        # Planning composes once before this explicit AOT capture has filled
        # every local region.  Recompose bottom-up now that the numerical
        # programs exist; otherwise callers see local regions but an empty
        # hierarchy and cannot link nested methods/loops into a whole program.
        hierarchy_exception = False
        for planned_shell in tuple(dict.fromkeys((
            function_shell,
            deployment,
        ))):
            if not getattr(planned_shell, "callsite_function_shells", None):
                continue
            try:
                hierarchical_artifact = _build_hierarchical_glsl_artifact(
                    planned_shell
                )
            except Exception as error:
                hierarchy_exception = True
                failure_traceback = traceback.format_exc()
                planned_shell.hierarchical_compose_failure = {
                    **dict(
                        getattr(
                            planned_shell,
                            "hierarchical_compose_failure",
                            None,
                        )
                        or {}
                    ),
                    "reason": "hierarchy-recomposition-exception",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": failure_traceback,
                }
                _report(
                    "aot: hierarchy recomposition traceback\n"
                    + failure_traceback
                )
                if not (
                    getattr(
                        planned_shell,
                        "hierarchical_control_program",
                        None,
                    )
                    and getattr(
                        planned_shell,
                        "hierarchical_captured_region_programs",
                        {},
                    )
                ):
                    _report(
                        "aot: hierarchy recomposition skipped "
                        f"({type(error).__name__}: {str(error)[:320]})"
                    )
                    continue
                _report(
                    "aot: optional GLSL hierarchy emission skipped "
                    f"({type(error).__name__}: {str(error)[:320]})"
                )
                hierarchical_artifact = None
            if hierarchical_artifact is not None:
                planned_shell.composed_shell_artifact = hierarchical_artifact
                planned_shell.hierarchical_shell_composed = True
        if precompile_only:
            # No GLSL was emitted, so there is nothing to execute here; the
            # caller wants the backend-agnostic FusedProgram/ControlProgram
            # pair to hand to its own lowering (Fortran via
            # precompile_to_ssa, for example).
            outputs = {}
        else:
            outputs = {
                name: value.numpy() if hasattr(value, "numpy") else value
                for name, value in function_shell.execute_named(feeds).items()
            }
        # The entrypoint's own shell is frequently a thin wrapper whose
        # control program has no regions -- the convention this module
        # documents is that the entrypoint calls another function, and the
        # loop lives in the callee's shell. Taking the wrapper's IR silently
        # loses that loop, so the shell that actually carries scheduled
        # regions is preferred when one exists.
        source_shell = function_shell
        if not getattr(
            getattr(function_shell, "shell_control_program", None),
            "region_indices",
            (),
        ):
            for candidate in _walk_planned_shells(deployment):
                control = getattr(candidate, "shell_control_program", None)
                if control is not None and control.region_indices:
                    source_shell = candidate
                    break
        control_shortfalls = tuple(
            {
                "function": str(
                    candidate.process_graph.G.graph.get("function_name")
                    or "?"
                ),
                "loop_node_id": int(reduction.loop_node_id),
                "source_type": type(
                    candidate.process_graph.G.nodes[
                        int(reduction.loop_node_id)
                    ].get("expr_obj")
                ).__name__,
                "condition_nodes": tuple(
                    int(parent)
                    for parent, role in (
                        candidate.process_graph.G.nodes[
                            int(reduction.loop_node_id)
                        ].get("parents") or ()
                    )
                    if str(role) in {"test", "ifs"}
                ),
                "blockers": tuple(map(str, reduction.blockers)),
                "captured_regions": tuple(
                    int(key if not isinstance(key, tuple) else key[-2])
                    for key in (
                        candidate.captured_region_programs or {}
                    )
                ),
            }
            for candidate in _walk_planned_shells(deployment)
            if candidate.captured_region_programs
            for reduction in candidate.loop_shader_reductions
            if reduction.region_indices
            and reduction.control_program is None
        )
        if control_shortfalls:
            _report(
                "aot: control lowering left "
                f"{len(control_shortfalls)} loop(s) as fill-later spot(s)"
            )
            for item in control_shortfalls:
                _report(
                    "aot: control-hole "
                    f"{item['function']} loop {item['loop_node_id']} "
                    f"[{', '.join(item['blockers'])}]"
                )
        entry_output_names = tuple(map(
            str,
            function_shell.process_graph.G.graph.get("function_outputs")
            or (),
        ))
        hierarchy_candidates = tuple(dict.fromkeys((
            function_shell,
            deployment,
            *_walk_planned_shells(deployment),
        )))
        _report(
            "aot: hierarchy candidates "
            + repr(tuple(
                (
                    candidate.process_graph.G.graph.get("function_name"),
                    len(getattr(
                        candidate,
                        "hierarchical_captured_region_programs",
                        {},
                    ) or {}),
                    tuple((
                        getattr(
                            getattr(candidate, "composed_shell_artifact", None),
                            "terminal_outputs",
                            {},
                        )
                        or getattr(
                            candidate,
                            "hierarchical_terminal_outputs",
                            {},
                        )
                        or {}
                    ).keys()),
                    getattr(candidate, "hierarchical_compose_failure", None),
                )
                for candidate in hierarchy_candidates
                if getattr(
                    candidate,
                    "hierarchical_captured_region_programs",
                    {},
                )
                or getattr(candidate, "hierarchical_compose_failure", None)
            ))
        )
        hierarchy_owner = max(
            hierarchy_candidates,
            key=lambda candidate: (
                len(
                    set(entry_output_names)
                    & set(
                        getattr(
                            getattr(candidate, "composed_shell_artifact", None),
                            "terminal_outputs",
                            {},
                        )
                        or getattr(
                            candidate,
                            "hierarchical_terminal_outputs",
                            {},
                        )
                        or {}
                    )
                ),
                bool(getattr(
                    candidate,
                    "hierarchical_captured_region_programs",
                    {},
                )),
            ),
        )
        hierarchical_control = getattr(
            hierarchy_owner, "hierarchical_control_program", None
        )
        hierarchical_regions = getattr(
            hierarchy_owner,
            "hierarchical_captured_region_programs",
            {},
        )
        if hierarchical_control is not None and hierarchical_regions:
            source_shell = hierarchy_owner
            shell_control_program = hierarchical_control
            selected_regions = hierarchical_regions
        else:
            shell_control_program = source_shell.shell_control_program
            selected_regions = getattr(
                source_shell, "captured_region_programs", {}
            ) or {}
        # A nested shell may own the retained loop/control regions, but it is
        # never the public numerical function the caller requested.  Keeping
        # these coupled replaced an entrypoint such as ``page`` with the
        # first nested loop owner (for example ``_project``), exporting that
        # method's hundreds of internal terminals instead of ``page``'s
        # declared returns.  Control ownership and public ABI ownership are
        # independent: regions may come from ``source_shell`` while the
        # numerical program and source names remain on ``function_shell``.
        compiled_shell_program = function_shell.compiled_shell_program
        region_programs = {
            int(index): getattr(program, "program", program)
            for index, program in selected_regions.items()
        }
        region_feed_values = {
            int(value_id): value
            for captured in selected_regions.values()
            for value_id, value in (
                getattr(captured, "feeds", {}) or {}
            ).items()
        }
        capture_feed_values = dict(region_feed_values)
        for candidate in _walk_planned_shells(deployment):
            captured_programs = (
                *(
                    getattr(candidate, "captured_region_programs", {})
                    or {}
                ).values(),
                *(
                    getattr(
                        candidate,
                        "hierarchical_captured_region_programs",
                        {},
                    )
                    or {}
                ).values(),
            )
            whole_captured = getattr(
                candidate, "compiled_shell_program", None
            )
            if whole_captured is not None:
                captured_programs = (*captured_programs, whole_captured)
            for captured in captured_programs:
                capture_feed_values.update({
                    int(value_id): value
                    for value_id, value in (
                        getattr(captured, "feeds", {}) or {}
                    ).items()
                })
        for global_id, capture_id in dict(
            getattr(
                hierarchy_owner,
                "hierarchical_capture_value_ids",
                {},
            )
            or {}
        ).items():
            if int(capture_id) in capture_feed_values:
                region_feed_values[int(global_id)] = (
                    capture_feed_values[int(capture_id)]
                )
        region_feed_values.update({
            int(value_id): value
            for value_id, value in dict(
                getattr(
                    hierarchy_owner,
                    "hierarchical_specialized_values",
                    {},
                )
                or {}
            ).items()
        })
        source_graph_metadata = function_shell.process_graph.G.graph
        root_value_ids = dict(
            getattr(function_shell, "hierarchical_root_value_ids", {}) or {}
        )
        identity_table = {
            str(name): tuple(
                int(root_value_ids.get(int(value_id), int(value_id)))
                for value_id in value_ids
            )
            for name, value_ids in (
                source_graph_metadata.get("identity_table") or {}
            ).items()
        }
        function_outputs = tuple(map(
            str, source_graph_metadata.get("function_outputs") or ()
        ))
        function_parameters = tuple(map(
            str, source_graph_metadata.get("function_parameters") or ()
        ))
        composed_artifact = getattr(
            hierarchy_owner, "composed_shell_artifact", None
        )
        composed_outputs = dict(
            getattr(composed_artifact, "terminal_outputs", {})
            or getattr(
                hierarchy_owner,
                "hierarchical_terminal_outputs",
                {},
            )
            or {}
        )
        public_output_value_ids = {
            name: int(composed_outputs[name])
            for name in function_outputs
            if name in composed_outputs
        }
        public_input_value_ids = dict(
            getattr(hierarchy_owner, "hierarchical_root_field_value_ids", {})
            or {}
        )
        hierarchical_value_diagnostics = dict(
            getattr(
                hierarchy_owner,
                "hierarchical_endpoint_details",
                {},
            )
            or {}
        )
        hierarchical_value_aliases = dict(
            getattr(
                hierarchy_owner,
                "hierarchical_value_aliases",
                {},
            )
            or {}
        )
        for local_id, global_id in dict(
            getattr(hierarchy_owner, "hierarchical_root_value_ids", {}) or {}
        ).items():
            if int(local_id) not in hierarchy_owner.process_graph.G:
                continue
            data = hierarchy_owner.process_graph.G.nodes[int(local_id)]
            attributes = data.get("attributes") or {}
            if (
                data.get("type") == "Input"
                and attributes.get("binding_kind") == "parameter"
            ):
                name = attributes.get("binding_name") or data.get("label")
                if name is not None:
                    public_input_value_ids.setdefault(
                        str(name), int(global_id)
                    )
    finally:
        # Matches the release-in-finally discipline every existing caller of
        # this deployment class already follows (tests/test_glsl_fused_network.py).
        deployment.release()
    _report("aot: rebuilding region program feed provenance")
    # Captured control regions are intentionally thin numerical graphs, but
    # their external feed identities still belong to the enclosing Python
    # function.  Carry that provenance onto every region so backends that
    # publish the regions directly can expose the source ABI instead of
    # inventing positional ``input_N`` names.
    named_feed_ids = {
        int(history[0]): name
        for name in function_parameters
        if (history := tuple(identity_table.get(name, ())))
    }
    region_programs = {
        index: FusedProgram(
            version=program.version,
            feeds=set(program.feeds),
            steps=list(program.steps),
            outputs=dict(program.outputs),
            state_in=None if program.state_in is None else set(program.state_in),
            meta=None if program.meta is None else dict(program.meta),
            extras={
                **dict(program.extras or {}),
                "capture_feed_origins": {
                    **dict(
                        (program.extras or {}).get("capture_feed_origins", {})
                        or {}
                    ),
                    **{
                        value_id: {"binding_name": name}
                        for value_id, name in named_feed_ids.items()
                        if value_id in program.feeds
                    },
                },
            },
        )
        for index, program in region_programs.items()
    }
    executable_value_ids = {
        int(value_id)
        for program in region_programs.values()
        for value_id in (
            *tuple(program.feeds),
            *tuple(step.result_id for step in program.steps),
            *tuple(program.outputs.values()),
        )
    }
    def collect_hierarchy_values(closure: Any) -> None:
        executable_value_ids.update(
            map(int, getattr(closure, "captures", ()))
        )
        for item in getattr(closure, "items", ()):
            executable_value_ids.update(
                map(int, getattr(item, "inputs", ()))
            )
            executable_value_ids.update(
                map(int, getattr(item, "outputs", ()))
            )
            if hasattr(item, "items"):
                collect_hierarchy_values(item)
            callee = getattr(item, "callee", None)
            if callee is not None:
                collect_hierarchy_values(callee)

    collect_hierarchy_values(
        getattr(source_shell, "hierarchy_plan", None)
    )
    for parameter in function_parameters:
        history = tuple(map(int, identity_table.get(parameter, ())))
        live = tuple(
            value_id
            for value_id in history
            if value_id in executable_value_ids
        )
        if live:
            public_input_value_ids.setdefault(parameter, live[0])
    elided_mutable_parameters = tuple(
        parameter
        for parameter in mutable_parameters
        if parameter in function_parameters
        and parameter not in public_input_value_ids
        and not any(
            str(name).startswith(f"{parameter}.")
            for name in public_input_value_ids
        )
    )
    if elided_mutable_parameters:
        raise RuntimeError(
            "mutable runtime parameters were specialized out of the "
            "executable AST program: "
            f"{elided_mutable_parameters!r}; their ProcessGraph identities "
            "exist, but no retained control/numerical SSA value consumes "
            "them"
        )
    shell = DualIRShell(
        compiled_shell_program=compiled_shell_program,
        shell_control_program=shell_control_program,
        map_ir=map_ir,
        name=entrypoint,
        class_navigation=class_navigation,
        dependency_regions=dependency_regions,
        reference_tables=getattr(deployment, "reference_tables", None),
        hierarchy_plan=getattr(source_shell, "hierarchy_plan", None),
    )
    compilation = AOTCompilation(
        entrypoint=entrypoint,
        outputs=outputs,
        compiled_shell_program=compiled_shell_program,
        shell_control_program=shell_control_program,
        deployment=deployment,
        shell=shell,
        map_ir=map_ir,
        class_navigation=class_navigation,
        dependency_regions=dependency_regions,
        region_programs=region_programs,
        planned_operator_implementations=dict(
            getattr(
                source_shell,
                "planned_operator_implementations",
                {},
            )
            or {}
        ),
        hierarchy_plan=getattr(source_shell, "hierarchy_plan", None),
        region_feed_values=region_feed_values,
        identity_table=identity_table,
        function_outputs=function_outputs,
        function_parameters=function_parameters,
        public_input_value_ids=public_input_value_ids,
        public_output_value_ids=public_output_value_ids,
        hierarchical_value_diagnostics=hierarchical_value_diagnostics,
        hierarchical_value_aliases=hierarchical_value_aliases,
        control_shortfalls=control_shortfalls,
        bake_mode=bake_mode,
        schedule_preference=schedule_preference,
        program_record_mode=("configured" if constant_map else "full"),
        constant_map=constant_map,
        mutable_parameters=mutable_parameters,
    )
    if (
        checkpoint_store is not None
        and precompile_only
        and not hierarchy_exception
    ):
        _report("aot: saving captured-program checkpoint")
        try:
            checkpoint_store.store(
                "captured_program",
                capture_implementation,
                replace(compilation, deployment=None),
            )
        except Exception as error:
            _report(
                "aot: captured-program checkpoint skipped "
                f"({type(error).__name__}: {error})"
            )
    elif checkpoint_store is not None and precompile_only:
        _report(
            "aot: captured-program checkpoint skipped "
            "(hierarchy recomposition raised; prepared-plan remains resumable)"
        )
    return compilation


def compile_cpp_shell_aot(
    source: str,
    entrypoint: str,
    feeds: Mapping[str, Any],
    *,
    backend: str = "c",
    remove_loops: bool = False,
    unroll_limit: int = 8,
    profiling: bool = False,
    precompile_only: bool = False,
    python_bindings: Mapping[str, Any] | None = None,
    bake_mode: str = "whole_program",
    schedule_preference: str = "alap",
    constant_map: Mapping[str, Any] | None = None,
    mutable_parameters: tuple[str, ...] | list[str] | set[str] = (),
    progress: "Callable[[str], None] | None" = None,
) -> AOTCompilation:
    """Compile a narrow C++-like shell entrypoint via the same shared
    lowering path Python's ``compile_ast_aot`` uses.

    The second, proving half of ``PROCESS_GRAPH_LOWERING_SEAM_HANDOFF.md``:
    this wrapper's only job is language-specific ingestion (``desugar_cpp_shell``
    -> ``pycparser`` -> ``role_schemas`` -> a built ``ProcessGraph``, plus the
    entrypoint's own node identity) before handing off to the exact same
    ``_lower_process_graph_to_compilation`` Python uses. No checkpoint
    support yet (unlike ``compile_ast_aot``) -- this is deliberately the
    minimal wrapper needed to prove the join, not full parity.
    """

    from pycparser import c_ast, c_parser

    from ....compiler.cpp_shell_desugar import desugar_cpp_shell
    from ....transmogrifier.graph.oop_language_translations import (
        install_c_role_schemas,
    )

    bake_mode = normalize_aot_bake_mode(bake_mode)
    schedule_preference = normalize_aot_schedule_preference(
        schedule_preference
    )
    constant_map = dict(constant_map or {})
    mutable_parameters = tuple(dict.fromkeys(map(str, mutable_parameters)))
    expanded_python_bindings = _expand_python_static_bindings(
        python_bindings
    )

    desugared = desugar_cpp_shell(source)
    tree = c_parser.CParser().parse(desugared)
    entrypoint_function_def = next(
        (
            node for node in tree.ext
            if isinstance(node, c_ast.FuncDef)
            and node.decl.name == entrypoint
        ),
        None,
    )
    entrypoint_node_id = (
        id(entrypoint_function_def)
        if entrypoint_function_def is not None
        else None
    )

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = dict(expanded_python_bindings)
    install_c_role_schemas(graph)
    graph.build_graph(tree)

    return _lower_process_graph_to_compilation(
        graph, entrypoint_node_id, entrypoint, feeds,
        backend=backend,
        remove_loops=remove_loops,
        unroll_limit=unroll_limit,
        profiling=profiling,
        precompile_only=precompile_only,
        expanded_python_bindings=expanded_python_bindings,
        bake_mode=bake_mode,
        schedule_preference=schedule_preference,
        constant_map=constant_map,
        mutable_parameters=mutable_parameters,
        progress=progress,
        checkpoint_store=None,
        checkpoint_feeds={},
        frontend_implementation="",
        source_graph_implementation="",
        planning_implementation="",
        prepared_implementation="",
        capture_implementation="",
        deployment=None,
        deployment_prepared=False,
        frontend_ready=False,
        class_navigation=None,
        dependency_regions=None,
        map_ir=None,
        resume=False,
    )


__all__ = [
    "AOT_BAKE_MODES",
    "AOT_SCHEDULE_PREFERENCES",
    "AOTCompilation",
    "compile_ast_aot",
    "compile_cpp_shell_aot",
    "normalize_aot_bake_mode",
    "normalize_aot_schedule_preference",
    "prepare_aot_checkpoint_store",
    "project_public_numerical_program",
    "_expand_python_static_bindings",
]
