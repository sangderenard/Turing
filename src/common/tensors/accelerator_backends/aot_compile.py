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
    -> strategize_glsl_deployment(backend=)  (compiler/glsl_deployment_strategy.py)
    -> deployment.compile_process_graph()
    -> deployment.capture_fused_programs(feeds)
    -> deployment.execute_named(feeds)

``strategize_glsl_deployment``'s name is historical, and ``shell_language``
is a real, validated constructor argument -- but as of this writing only
``"glsl"`` has an actual distinct emission path.  ``emit_glsl`` (the flag
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

from __future__ import annotations

import ast
import contextlib
import inspect
import io
from dataclasses import dataclass, field
from typing import Any, Mapping

from ....compiler.glsl_deployment_strategy import (
    _walk_planned_shells,
    propagate_bound_planner_specializations,
    strategize_glsl_deployment,
)
from ....compiler.shell_reference_tables import (
    build_class_navigation_table,
    build_map_dependency_regions,
)
from ....compiler.precompile_to_ssa import lower_class_navigation_to_ssa
from ....transmogrifier.graph.graph_express2 import ProcessGraph
from ..abstraction import AbstractTensor
from ..topological_reducer import reduce_abstract_tensor_topology
from ..fused_ir import FusedProgram
from .dual_ir_shell import DualIRShell, compose_dual_ir_shell


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
    # Retained by the reduced function ProcessGraph: source name -> complete
    # canonical value-ID history. ``function_outputs`` selects public names.
    identity_table: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    function_outputs: tuple[str, ...] = ()
    function_parameters: tuple[str, ...] = ()
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


def _apply_parameter_constant_map(
    module: ast.Module,
    entrypoint: str,
    constant_map: Mapping[str, Any],
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
    bake_mode: str = "whole_program",
    schedule_preference: str = "alap",
    constant_map: Mapping[str, Any] | None = None,
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

    bake_mode = normalize_aot_bake_mode(bake_mode)
    schedule_preference = normalize_aot_schedule_preference(
        schedule_preference
    )
    constant_map = dict(constant_map or {})
    module = _apply_parameter_constant_map(
        ast.parse(source), entrypoint, constant_map
    )
    graph = ProcessGraph(materialize_memory=False)
    # AOT compilation may target a function from a live module.  Its resolved
    # globals are static closure values, not runtime tensor feeds.  Capturing
    # them here lets the reducer retain computed constants and imported
    # references without executing or reinterpreting their source expressions.
    graph.python_bindings = dict(python_bindings or {})
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            module,
            resolve_unresolved_parents=True,
            parent_include=_source_dependency_is_not_tensor_primitive,
        )
    reduce_abstract_tensor_topology(graph)
    propagate_bound_planner_specializations(
        graph, entrypoint, feeds
    )
    dependency_regions = build_map_dependency_regions(graph, entrypoint)
    map_ir = dict(graph.G.graph.get("map_ir") or {})
    map_ir["dependency_regions"] = {
        "runtime": dependency_regions.runtime,
        "mapped": dependency_regions.mapped,
        "retained": dependency_regions.retained,
        "map_only": dependency_regions.map_only,
        "bindings": dependency_regions.bindings,
    }
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

    deployment_type = strategize_glsl_deployment(
        graph,
        backend=backend,
        remove_loops=remove_loops,
        unroll_limit=unroll_limit,
        schedule_preference=schedule_preference,
    )
    # shell_language is fixed at "glsl", the one path that actually emits
    # something distinct -- see this module's docstring. It is not derived
    # from backend= any more; there is no shell "kind" left to pick.
    deployment = deployment_type(profiling=profiling, shell_language="glsl")
    try:
        deployment.compile_process_graph()
        reference = graph.function_table.reference(entrypoint)
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
        function_shell.capture_fused_programs(
            feeds, precompile_only=precompile_only
        )
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
        hierarchical_control = getattr(
            function_shell, "hierarchical_control_program", None
        )
        hierarchical_regions = getattr(
            function_shell,
            "hierarchical_captured_region_programs",
            {},
        )
        if hierarchical_control is not None and hierarchical_regions:
            source_shell = function_shell
            shell_control_program = hierarchical_control
            selected_regions = hierarchical_regions
        else:
            shell_control_program = source_shell.shell_control_program
            selected_regions = getattr(
                source_shell, "captured_region_programs", {}
            ) or {}
        compiled_shell_program = source_shell.compiled_shell_program
        region_programs = {
            int(index): getattr(program, "program", program)
            for index, program in selected_regions.items()
        }
        source_graph_metadata = source_shell.process_graph.G.graph
        root_value_ids = dict(
            getattr(source_shell, "hierarchical_root_value_ids", {}) or {}
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
    finally:
        # Matches the release-in-finally discipline every existing caller of
        # this deployment class already follows (tests/test_glsl_fused_network.py).
        deployment.release()
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
    shell = DualIRShell(
        compiled_shell_program=compiled_shell_program,
        shell_control_program=shell_control_program,
        map_ir=map_ir,
        name=entrypoint,
    )
    return AOTCompilation(
        entrypoint=entrypoint,
        outputs=outputs,
        compiled_shell_program=compiled_shell_program,
        shell_control_program=shell_control_program,
        deployment=deployment,
        shell=shell,
        map_ir=map_ir,
        region_programs=region_programs,
        identity_table=identity_table,
        function_outputs=function_outputs,
        function_parameters=function_parameters,
        control_shortfalls=control_shortfalls,
        bake_mode=bake_mode,
        schedule_preference=schedule_preference,
        program_record_mode=("configured" if constant_map else "full"),
        constant_map=constant_map,
    )


__all__ = [
    "AOT_BAKE_MODES",
    "AOT_SCHEDULE_PREFERENCES",
    "AOTCompilation",
    "compile_ast_aot",
    "normalize_aot_bake_mode",
    "normalize_aot_schedule_preference",
    "project_public_numerical_program",
]
