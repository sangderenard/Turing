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


@pytest.mark.parametrize(
    ("source", "child_selector", "expected_slot"),
    [
        (
            "result = [item for item in (value for value in values)]",
            lambda tree: next(
                node for node in ast.walk(tree.generators[0].iter)
                if isinstance(node, ast.comprehension)
            ),
            "iter",
        ),
        (
            "result = [item for item in values if any(value for value in item)]",
            lambda tree: next(
                node for node in ast.walk(tree.generators[0].ifs[0])
                if isinstance(node, ast.comprehension)
            ),
            "ifs",
        ),
        (
            "result = [(value for value in item) for item in values]",
            lambda tree: next(
                node for node in ast.walk(tree.elt)
                if isinstance(node, ast.comprehension)
            ),
            "elt",
        ),
    ],
)
def test_comprehension_ownership_preserves_exact_authored_site(
    source, child_selector, expected_slot
):
    aggregate = ast.parse(source).body[0].value
    parent = aggregate.generators[0]
    child = child_selector(aggregate)
    graph_data = nx.DiGraph()
    graph_data.add_node(1, expr_obj=parent)
    graph_data.add_node(2, expr_obj=child)
    graph_data.add_node(3, expr_obj=aggregate, parents=((1, "generators"),))
    graph_data.add_edge(1, 3)
    graph = SimpleNamespace(G=graph_data)
    reductions = (
        SimpleNamespace(loop_node_id=1, control_program=None, region_indices=(0,)),
        SimpleNamespace(loop_node_id=2, control_program=None, region_indices=()),
    )
    loop_plans = (SimpleNamespace(
        loop=SimpleNamespace(node_id=1, body_nodes=(2,)),
    ),)
    sites = {}

    hints = _source_control_nesting_hints(
        reductions,
        (),
        loop_plans,
        graph,
        authored_control_site_sink=sites,
    )

    assert hints == {0: frozenset({1})}
    assert sites[(0, 1)].parent_source_node_id == 1
    assert sites[(0, 1)].child_source_node_id == 2
    assert sites[(0, 1)].slot == expected_slot


def test_comprehension_element_conditional_uses_aggregate_owner_edge():
    """The clause and element meet through their exact aggregate graph node."""

    aggregate = ast.parse(
        "result = {(value.dtype, tuple(value.shape or ())) if value "
        "else (None, ()) for value in values}"
    ).body[0].value
    parent = aggregate.generators[0]
    child = aggregate.elt
    graph_data = nx.DiGraph()
    graph_data.add_node(1, expr_obj=parent)
    graph_data.add_node(2, expr_obj=child)
    graph_data.add_node(3, expr_obj=aggregate, parents=((1, "generators"),))
    graph_data.add_edge(1, 3)
    graph = SimpleNamespace(G=graph_data)
    reductions = (
        SimpleNamespace(loop_node_id=1, control_program=None, region_indices=(0, 1)),
    )
    control = ControlProgram(
        ConditionalBlock(
            10, StatementBlock(("__scheduled_region_0__",)), source_node_id=2,
        ),
        region_indices=(0, 1),
    )
    sites = {}

    hints = _source_control_nesting_hints(
        reductions, (control,), (), graph, authored_control_site_sink=sites,
    )

    assert hints == {0: frozenset({1})}
    assert sites[(0, 1)].slot == "elt"


def test_comprehension_generator_ownership_is_left_to_right():
    aggregate = ast.parse(
        "result = [(left, right) for left in rows for right in left]"
    ).body[0].value
    outer, inner = aggregate.generators
    graph_data = nx.DiGraph()
    graph_data.add_node(1, expr_obj=outer)
    graph_data.add_node(2, expr_obj=inner)
    graph_data.add_node(3, expr_obj=aggregate, parents=(
        (1, "generators"), (2, "generators"),
    ))
    graph_data.add_edges_from(((1, 3), (2, 3)))
    reductions = (
        SimpleNamespace(loop_node_id=1, control_program=None, region_indices=(0,)),
        SimpleNamespace(loop_node_id=2, control_program=None, region_indices=(1,)),
    )
    sites = {}

    hints = _source_control_nesting_hints(
        reductions, (), (), SimpleNamespace(G=graph_data),
        authored_control_site_sink=sites,
    )

    assert hints == {0: frozenset({1})}
    assert sites[(0, 1)].slot == "generators"
    assert sites[(0, 1)].ordinal == 1
