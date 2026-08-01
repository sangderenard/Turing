"""General AOT compilation access, for anything (including the torture test)
that wants to compile a real Python function ahead-of-time through the
precompiler, instead of walking a captured tape.

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
import io
from dataclasses import dataclass, field
from typing import Any, Mapping

from ....compiler.glsl_deployment_strategy import (
    _walk_planned_shells,
    strategize_glsl_deployment,
)
from ....transmogrifier.graph.graph_express2 import ProcessGraph
from ..topological_reducer import reduce_abstract_tensor_topology
from .dual_ir_shell import DualIRShell, compose_dual_ir_shell


@dataclass(frozen=True)
class AOTCompilation:
    entrypoint: str
    outputs: Mapping[str, Any]
    compiled_shell_program: Any
    shell_control_program: Any
    deployment: Any
    shell: DualIRShell
    # The numeric program for each ``__scheduled_region_N__`` the control
    # program references. ``lower_precompile_and_control_to_ssa`` needs these
    # to find the producers of a loop's carried updates; without them a
    # program whose control shell has regions cannot be lowered.
    region_programs: Mapping[int, Any] = field(default_factory=dict)


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

    module = ast.parse(source)
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    deployment_type = strategize_glsl_deployment(
        graph,
        backend=backend,
        remove_loops=remove_loops,
        unroll_limit=unroll_limit,
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
        compiled_shell_program = source_shell.compiled_shell_program
        shell_control_program = source_shell.shell_control_program
        region_programs = {
            int(index): getattr(program, "program", program)
            for index, program in (
                getattr(source_shell, "captured_region_programs", {}) or {}
            ).items()
        }
    finally:
        # Matches the release-in-finally discipline every existing caller of
        # this deployment class already follows (tests/test_glsl_fused_network.py).
        deployment.release()
    shell = DualIRShell(
        compiled_shell_program=compiled_shell_program,
        shell_control_program=shell_control_program,
        name=entrypoint,
    )
    return AOTCompilation(
        entrypoint=entrypoint,
        outputs=outputs,
        compiled_shell_program=compiled_shell_program,
        shell_control_program=shell_control_program,
        deployment=deployment,
        shell=shell,
        region_programs=region_programs,
    )


__all__ = ["AOTCompilation", "compile_ast_aot"]
