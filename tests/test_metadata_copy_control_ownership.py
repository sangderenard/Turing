"""Predicate computation precedes an If; it is not that If's arm body."""
import ast
from types import SimpleNamespace

import networkx as nx
import pytest

from src.compiler.control_source import ConditionalBlock, ControlProgram, StatementBlock
from src.compiler.glsl_deployment_strategy import _source_control_nesting_hints


@pytest.mark.parametrize("in_predicate", [True, False])
def test_comprehension_ownership_uses_conditional_arm_slots(in_predicate):
    source = (
        "if all(item is not None for item in values):\n    result = 1\n"
        if in_predicate else
        "if enabled:\n    result = all(item is not None for item in values)\n"
    )
    conditional = ast.parse(source).body[0]
    clause = next(node for node in ast.walk(conditional) if isinstance(node, ast.comprehension))
    G = nx.DiGraph()
    G.add_node(1, expr_obj=conditional)
    G.add_node(2, expr_obj=clause)
    graph = SimpleNamespace(G=G)
    reduction = SimpleNamespace(loop_node_id=2, control_program=None, region_indices=(0,))
    control = ControlProgram(
        ConditionalBlock(10, StatementBlock(("__scheduled_region_4__",)), source_node_id=1),
        region_indices=(4,),
    )

    hints = _source_control_nesting_hints((reduction,), (control,), (), graph)

    assert hints == ({} if in_predicate else {1: frozenset({0})})
