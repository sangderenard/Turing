from __future__ import annotations

from src.compiler.process_graph_fusion import (
    serialize_scheduled_operator_dispatches,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _payload(op, parents=()):
    parents = list(parents)
    return {
        "label": op,
        "type": op,
        "op": op,
        "parents": parents,
        "children": [],
        "attributes": {},
        "extra_args": {},
        "expr_obj": None,
        "store_id": None,
    }


def _add(graph, node_id, op, parents=()):
    parent_items = tuple(
        (parent, f"arg{index}")
        for index, parent in enumerate(parents)
    )
    graph.G.add_node(node_id, **_payload(op, parent_items))
    for parent, role in parent_items:
        graph.G.add_edge(parent, node_id, role=role)
        graph.G.nodes[parent]["children"].append((node_id, role))


def test_schedule_batches_isolated_dependency_columns_as_forward_records():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "input")
    _add(graph, 2, "input")
    _add(graph, 3, "add", (1,))
    _add(graph, 4, "add", (2,))
    _add(graph, 5, "sin", (3, 4))
    _add(graph, 6, "cos", (5,))
    _add(graph, 7, "return", (6,))

    plan = serialize_scheduled_operator_dispatches(
        graph,
        max_nodes_per_dispatch=8,
    )

    assert [
        (pattern.level, pattern.operator, pattern.node_ids)
        for pattern in plan.patterns
    ] == [
        (0, "input", (1, 2)),
        (1, "add", (3, 4)),
        (2, "sin", (5,)),
        (3, "cos", (6,)),
        (4, "return", (7,)),
    ]
    assert [dispatch.kind for dispatch in plan.dispatches] == [
        "forward_record",
        "forward_record",
    ]
    assert plan.dispatches[0].operator_pattern == ("input", "add")
    assert plan.dispatches[0].dependency_columns == ((1, 3), (2, 4))
    assert plan.dispatches[1].operator_pattern == (
        "sin",
        "cos",
        "return",
    )
    assert plan.dispatches[1].dependency_columns == ((5, 6, 7),)
    assert plan.node_locations[6] == (1, 1)


def test_same_operator_level_is_split_only_at_the_dispatch_cap():
    graph = ProcessGraph(materialize_memory=False)
    for node_id in range(5):
        _add(graph, node_id, "input")

    plan = serialize_scheduled_operator_dispatches(
        graph,
        max_nodes_per_dispatch=2,
    )

    assert [pattern.node_ids for pattern in plan.patterns] == [
        (0, 1),
        (2, 3),
        (4,),
    ]
    assert [pattern.batch_index for pattern in plan.patterns] == [0, 1, 2]
    assert [pattern.batch_count for pattern in plan.patterns] == [3, 3, 3]
