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

``source`` must be real ``def`` functions, not a lambda -- and, following
the one verified working shape (``tests/test_glsl_fused_network.py``'s
``affine``/``render_value`` pair), the function whose result you want must
call at least one other function defined in the same source rather than
compute its result inline; a single bare top-level function hits an
unrelated, unverified failure in the graph coordinator.
"""

from __future__ import annotations

import ast
import contextlib
import io
from dataclasses import dataclass
from typing import Any, Mapping

from ....compiler.glsl_deployment_strategy import strategize_glsl_deployment
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


def compile_ast_aot(
    source: str,
    entrypoint: str,
    feeds: Mapping[str, Any],
    *,
    backend: str = "c",
    remove_loops: bool = False,
    profiling: bool = False,
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
        graph, backend=backend, remove_loops=remove_loops
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
        function_shell.capture_fused_programs(feeds)
        outputs = {
            name: value.numpy() if hasattr(value, "numpy") else value
            for name, value in function_shell.execute_named(feeds).items()
        }
        compiled_shell_program = function_shell.compiled_shell_program
        shell_control_program = function_shell.shell_control_program
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
    )


__all__ = ["AOTCompilation", "compile_ast_aot"]
