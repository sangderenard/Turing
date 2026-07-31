"""Front end for ordering a compiled program: JIT (tape) or AOT (precompiler).

These are two different products, not one pipeline wearing two hats:

* ``tape=`` -- JIT. Calls the existing
  :func:`c_primitive_program.compile_elementwise_tape`. Returns a
  :class:`CapturedFusedProgram` (a ``FusedProgram`` is its whole product: a
  one-shot numeric section, nothing further).

* ``ast=`` / ``sympy_expression=`` -- AOT. Builds a ``ProcessGraph`` via the
  existing ``ProcessGraph.build_from_ast`` / ``build_from_expression``,
  plans it into regions with the existing
  ``process_graph_fusion.plan_process_graph_dispatches`` /
  ``dispatch_region_to_fused_program``, and unifies those regions into one
  real SSA ``IRModule`` via the existing
  ``precompile_to_ssa.lower_precompile_and_control_to_ssa``. Returns a
  :class:`PrecompileSSALoweringResult` -- basic blocks, real control edges,
  named functions calling each other. Not a ``FusedProgram``.

Every step below is a call to a function that already exists elsewhere in
the repo. See ``docs/PIPELINE_STAGE_DISAMBIGUATION.md`` for the stages this
front end is choosing between.
"""

from __future__ import annotations

from typing import Any

from ....compiler.control_source import ControlProgram, SequenceBlock, StatementBlock
from ....compiler.precompile_to_ssa import (
    PrecompileSSALoweringResult,
    lower_precompile_and_control_to_ssa,
)
from ....compiler.process_graph_fusion import (
    BackendFusionProfile,
    dispatch_region_to_fused_program,
    plan_process_graph_dispatches,
)
from ....transmogrifier.graph.graph_express2 import ProcessGraph
from ..fused_ir import ELEMENTWISE_BINARY, ELEMENTWISE_UNARY, FusedProgram
from .c_primitive_program import CapturedFusedProgram, compile_elementwise_tape


def order_program(
    *,
    tape: Any = None,
    output: Any = None,
    ast: Any = None,
    sympy_expression: Any = None,
    backend: str = "generic",
) -> CapturedFusedProgram | PrecompileSSALoweringResult:
    """Return the compiled product for whichever source kwarg was supplied.

    ``tape=`` -> ``CapturedFusedProgram`` (JIT).
    ``ast=`` / ``sympy_expression=`` -> ``PrecompileSSALoweringResult`` (AOT).
    """

    if tape is not None:
        return compile_elementwise_tape(tape, output)

    if ast is None and sympy_expression is None:
        raise ValueError(
            "order_program requires tape= (JIT) or ast=/sympy_expression= "
            "(AOT)"
        )

    graph = ProcessGraph(materialize_memory=False)
    if ast is not None:
        graph.build_from_ast(ast)
    else:
        graph.build_from_expression(sympy_expression)

    profile = BackendFusionProfile(
        backend,
        frozenset(ELEMENTWISE_UNARY | ELEMENTWISE_BINARY),
        max_bindings=1 << 20,
    )
    plan = plan_process_graph_dispatches(graph, profile)
    region_indices = tuple(range(len(plan.regions)))
    region_programs = {
        index: dispatch_region_to_fused_program(graph, region)
        for index, region in zip(region_indices, plan.regions)
    }

    # There is no separate top-level numeric precompile here -- planning
    # already places every fusible op into a region. The control shell is
    # the minimal one that actually matches that: call each region once, in
    # order, using the existing __scheduled_region_N__ marker convention
    # (control_source.py / precompile_to_ssa.py both already parse it).
    control = ControlProgram(
        root=SequenceBlock(tuple(
            StatementBlock((f"__scheduled_region_{index}__",))
            for index in region_indices
        )),
        region_indices=region_indices,
    )
    empty_artifact = FusedProgram(version=1, feeds=set(), steps=[], outputs={})

    return lower_precompile_and_control_to_ssa(
        empty_artifact,
        control,
        region_programs=region_programs,
    )


__all__ = ["order_program"]
