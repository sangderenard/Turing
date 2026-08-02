from __future__ import annotations

import ast

from src.compiler.process_graph_fusion import (
    extract_clean_process_subgraph,
    reduce_scheduled_shader_regions,
    serialize_scheduled_operator_dispatches,
)
from src.compiler.glsl_deployment_strategy import (
    _control_partition_keys,
    _dispatch_subgraph,
    _is_dispatch_metadata_node,
    _inert_routing_nodes,
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


def test_recursion_control_projection_still_reduces_numeric_interior():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "input")
    _add(graph, 2, "add", (1,))
    _add(graph, 3, "mul", (2,))
    _add(graph, 4, "LoopResult", (3,))
    graph.G.add_edge(4, 2, role="carried")
    graph.G.nodes[2]["parents"].append((4, "carried"))
    graph.G.nodes[4]["children"].append((2, "carried"))
    graph.levels = {1: 0, 2: 1, 3: 1, 4: 1}

    plan = reduce_scheduled_shader_regions(
        graph,
        (2, 3, 4),
        control_node_ids=(4,),
        max_nodes_per_region=8,
    )

    assert any(set(dispatch.node_ids) == {2, 3} for dispatch in plan.dispatches)
    assert all(4 not in dispatch.node_ids for dispatch in plan.dispatches)


def test_static_python_call_name_collision_is_not_numerical_dispatch():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "sub")
    graph.G.nodes[1]["attributes"]["static_python_reference"] = "re.sub"

    assert _is_dispatch_metadata_node(graph, 1)


def test_canonical_static_tensor_operator_remains_numerical_dispatch():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "stack")
    graph.G.nodes[1]["attributes"].update({
        "static_python_reference": "AbstractTensor.stack",
        "operator_reference_node": 99,
    })

    assert not _is_dispatch_metadata_node(graph, 1)


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


def test_shader_region_reducer_applies_all_three_identities_to_fixed_point():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Input")
    _add(graph, 2, "Add", (0,))
    _add(graph, 3, "Add", (1,))
    _add(graph, 4, "Sin", (2,))
    _add(graph, 5, "Cos", (3,))
    _add(graph, 6, "Neg", (0,))

    plan = reduce_scheduled_shader_regions(
        graph,
        executable_node_ids=(2, 3, 4, 5, 6),
    )

    assert len(plan.dispatches) == 1
    dispatch = plan.dispatches[0]
    assert set(dispatch.node_ids) == {2, 3, 4, 5, 6}
    assert "horizontal-batch" in dispatch.rewrite_history
    assert "vertical-fusion" in dispatch.rewrite_history
    assert "horizontal-shader-pack" in dispatch.rewrite_history


def test_shader_region_reducer_never_crosses_a_planner_partition():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Add", (0,))
    _add(graph, 2, "Mul", (1,))

    plan = reduce_scheduled_shader_regions(
        graph,
        executable_node_ids=(1, 2),
        partition_keys={1: ("outside",), 2: ("loop-7",)},
    )

    assert [dispatch.node_ids for dispatch in plan.dispatches] == [(1,), (2,)]


def test_shader_region_reducer_preserves_hidden_structural_dependency():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Add", (0,))
    _add(graph, 2, "coordinator", (1,))
    _add(graph, 3, "Mul", (2,))

    plan = reduce_scheduled_shader_regions(
        graph,
        executable_node_ids=(1, 3),
    )

    assert [dispatch.node_ids for dispatch in plan.dispatches] == [(1,), (3,)]


def test_tensor_accessor_and_its_scalar_comparison_stay_with_coordinator():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "ndims", (0,))
    _add(graph, 2, "Constant")
    _add(graph, 3, "not_equal", (1, 2))
    graph.G.nodes[1]["expr_obj"] = ast.parse("value.ndims()").body[0].value
    graph.G.nodes[3]["expr_obj"] = ast.parse("rank != 1").body[0].value
    graph.G.nodes[2]["constant"] = 1

    assert _is_dispatch_metadata_node(graph, 1)
    assert _is_dispatch_metadata_node(graph, 3)


def test_clean_subgraph_drops_obligations_to_excluded_parents():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "input")
    _add(graph, 2, "add", (1,))
    _add(graph, 3, "sin", (2,))
    graph.levels = {1: 0, 2: 1, 3: 2}
    graph.roots = [3]

    extracted = extract_clean_process_subgraph(graph, (2, 3))

    assert set(extracted.G) == {2, 3}
    assert extracted.G.nodes[2]["parents"] == []
    assert extracted.G.nodes[2]["children"] == [(3, "arg0")]
    assert extracted.G.nodes[3]["parents"] == [(2, "arg0")]
    assert extracted.levels == {2: 1, 3: 2}
    assert extracted.roots == [3]
    assert graph.G.nodes[2]["parents"] == [(1, "arg0")]


def test_shader_reducer_consumes_existing_process_graph_schedule():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Add", (0,))
    _add(graph, 2, "Mul", (0,))
    graph.levels = {0: 0, 1: 2, 2: 1}

    def reject_reschedule(*_args, **_kwargs):
        raise AssertionError("reducer replaced the planner schedule")

    graph.compute_levels = reject_reschedule
    plan = reduce_scheduled_shader_regions(
        graph,
        executable_node_ids=(1, 2),
    )

    assert plan.levels == ((0, (0,)), (1, (2,)), (2, (1,)))


def test_shader_region_reducer_leaves_an_unfusible_operation_alone():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Add", (0,))
    _add(graph, 2, "sum", (1,))
    _add(graph, 3, "Mul", (2,))

    plan = reduce_scheduled_shader_regions(
        graph,
        executable_node_ids=(1, 2, 3),
        fusible_node_ids=(1, 3),
    )

    assert [dispatch.node_ids for dispatch in plan.dispatches] == [
        (1,),
        (2,),
        (3,),
    ]


def test_shader_region_reducer_keeps_multiple_published_values_fused():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Add", (0,))
    _add(graph, 2, "Mul", (1,))
    _add(graph, 3, "observer", (1,))
    _add(graph, 4, "observer", (2,))

    plan = reduce_scheduled_shader_regions(
        graph,
        executable_node_ids=(1, 2),
    )
    assert [dispatch.node_ids for dispatch in plan.dispatches] == [(1, 2)]


def test_shader_region_reducer_honours_a_declared_hidden_dependency():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Add", (0,))
    _add(graph, 2, "coordinator")
    _add(graph, 3, "Mul", (2,))

    fused = reduce_scheduled_shader_regions(
        graph,
        executable_node_ids=(1, 3),
    )
    assert [dispatch.node_ids for dispatch in fused.dispatches] == [(1, 3)]

    # Node 1 is routed into the coordinator by name, so node 3 transitively
    # depends on it and the two cannot share one region.
    split = reduce_scheduled_shader_regions(
        graph,
        executable_node_ids=(1, 3),
        extra_dependency_edges=((1, 2),),
    )
    assert [dispatch.node_ids for dispatch in split.dispatches] == [(1,), (3,)]


def test_region_planning_never_spans_a_conditional_branch():
    tree = ast.parse("if flag:\n    a = x + 1\nelse:\n    a = x - 1\n")
    conditional = tree.body[0]

    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Add", (0,))
    _add(graph, 2, "Add", (0,))
    _add(graph, 3, "If", (1, 2))
    graph.G.nodes[1]["expr_obj"] = conditional.body[0].value
    graph.G.nodes[2]["expr_obj"] = conditional.orelse[0].value
    graph.G.nodes[3]["expr_obj"] = conditional

    keys = _control_partition_keys(graph, (), (1, 2))
    assert keys[1] != keys[2]

    plan = reduce_scheduled_shader_regions(
        graph,
        executable_node_ids=(1, 2),
        partition_keys=keys,
    )
    assert [dispatch.node_ids for dispatch in plan.dispatches] == [(1,), (2,)]


def test_region_planning_never_spans_a_nested_callsite():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Add", (0,))
    _add(graph, 2, "Call", (1,))
    graph.G.nodes[2]["attributes"]["callee_ref"] = 99
    _add(graph, 3, "Mul", (2,))
    _add(graph, 4, "Cos", (3,))

    keys = _control_partition_keys(graph, (), (1, 3, 4))
    assert keys[1] != keys[3]
    assert keys[3] == keys[4]

    plan = reduce_scheduled_shader_regions(
        graph,
        executable_node_ids=(1, 3, 4),
        partition_keys=keys,
    )
    assert [dispatch.node_ids for dispatch in plan.dispatches] == [
        (1,),
        (3, 4),
    ]


def test_region_reduction_is_open_to_every_executable_tensor_operation():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Mul", (0,))
    _add(graph, 2, "reshape", (1,))
    _add(graph, 3, "sum", (2,))

    plan = reduce_scheduled_shader_regions(
        graph,
        executable_node_ids=(1, 2, 3),
    )

    assert [dispatch.node_ids for dispatch in plan.dispatches] == [(1, 2, 3)]


def test_a_lookup_nobody_reads_is_not_a_region_boundary():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Add", (0,))
    _add(graph, 2, "Attribute", (1,))
    _add(graph, 3, "Mul", (1,))
    graph.G.nodes[2]["expr_obj"] = ast.parse("value.sqrt").body[0].value
    graph.compute_levels(method="asap", order="dependency")

    inert = _inert_routing_nodes(graph)
    assert inert == frozenset({2})

    observed = _dispatch_subgraph(graph, (1, 3))
    assert observed.G.graph["deployment_outputs"] == (1, 3)

    ignored = _dispatch_subgraph(graph, (1, 3), inert_nodes=inert)
    assert ignored.G.graph["deployment_outputs"] == (3,)


def test_an_attribute_assignment_is_never_treated_as_inert():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Attribute", (0,))
    graph.G.nodes[1]["expr_obj"] = ast.parse("obj.field = 1").body[0].targets[0]

    assert _inert_routing_nodes(graph) == frozenset()


def test_dispatch_exports_a_value_routed_by_identity():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Add", (0,))
    _add(graph, 2, "Mul", (1,))
    graph.compute_levels(method="asap", order="dependency")

    dispatch = _dispatch_subgraph(
        graph,
        (1, 2),
        required_outputs=frozenset({1}),
    )

    assert dispatch.G.graph["deployment_outputs"] == (1, 2)


def test_dispatch_exports_an_internal_value_used_outside_the_region():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 0, "Input")
    _add(graph, 1, "Add", (0,))
    _add(graph, 2, "Mul", (1,))
    _add(graph, 3, "external", (1,))
    graph.compute_levels(method="asap", order="dependency")

    dispatch = _dispatch_subgraph(graph, (1, 2))

    assert dispatch.G.graph["deployment_outputs"] == (1, 2)
    assert len(dispatch.G.graph["deployment_store_nodes"]) == 2
