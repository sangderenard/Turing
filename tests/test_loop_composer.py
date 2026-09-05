from __future__ import annotations

import ast
import contextlib
import io
import inspect
import textwrap
from dataclasses import replace

from src.common.tensors.accelerator_backends.glsl_backend import (
    emit_native_for_loop,
)
from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.compiler.loop_composer import (
    LoopBackendCapabilities,
    LoopComposer,
    LoopStrategy,
    _rebuild_graph_edges,
    analyze_shader_loop_reductions,
    bind_control_deployments_to_regions,
    evaporate_unrolled_loops,
    materialize_retained_loop_ports,
    planned_collection_bindings,
)
from src.compiler.glsl_deployment_strategy import (
    _fold_callsite_structural_values,
    _resolve_grounded_method_references,
    propagate_bound_planner_specializations,
    strategize_shell_deployment,
)
from src.compiler.precompile_to_ssa import lower_control_sections_to_ssa
from src.compiler.loop_ir import (
    IterableAccess,
    LoopStateEffect,
    LoopStateEffectMode,
)
from src.compiler.control_source import (
    ControlTarget,
    LoopBlock,
    LoopControlBlock,
    SequenceBlock,
    StreamPublishBlock,
    ValidationBlock,
    WhileBlock,
    render_control_program,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph
import src.transmogrifier.graph.graph_express2 as graph_express2


def _function_graph(source: str, name: str):
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    return graph.function_table.entry(name).graph


def _glsl_composer(*, unroll_limit: int = 8) -> LoopComposer:
    return LoopComposer(
        LoopBackendCapabilities(
            backend="glsl",
            native_for=True,
            native_while=True,
            dynamic_bounds=True,
            unroll_limit=unroll_limit,
        )
    )


def test_rebuild_preserves_irreducible_recursion_for_loop_lowering():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.add_node(0, parents=[(2, "carried")], children=[])
    graph.G.add_node(1, parents=[(0, "body")], children=[])
    graph.G.add_node(2, parents=[(1, "next")], children=[])

    _rebuild_graph_edges(graph)

    region = graph.G.graph["recursion_table"][0]
    assert region["kind"] == "irreducible_recursion"
    assert region["lower_as"] == "while"
    assert set(region["members"]) == {0, 1, 2}
    assert set(region["feedback"]) == {
        (0, 1, "body"),
        (1, 2, "next"),
        (2, 0, "carried"),
    }
    assert set(graph.levels) == {0, 1, 2}
    assert len(set(graph.levels.values())) == 1
    assert graph.G.has_edge(2, 0)
    assert graph.G.nodes[0]["attributes"]["recursion_region_id"] == 0


def test_loop_composer_unrolls_small_static_range():
    graph = _function_graph(
        "def kernel(x):\n"
        "    for index in range(4):\n"
        "        x = x + index\n"
        "    return x\n",
        "kernel",
    )

    plan, = _glsl_composer().compose(graph)

    assert plan.strategy is LoopStrategy.UNROLL
    assert plan.loop.target == "index"
    assert plan.loop.trip_count == 4
    assert plan.loop.body_nodes


def test_unroll_does_not_evaporate_explicit_sequence_mutation():
    graph = _function_graph(
        "def kernel(x):\n"
        "    values = []\n"
        "    for index in range(3):\n"
        "        x = x + index\n"
        "        values.append(x)\n"
        "    return AbstractTensor.stack(values, dim=0)\n",
        "kernel",
    )
    composer = _glsl_composer()
    plans = composer.discover(graph)

    assert all(plan.semantic is None for plan in plans)

    evaporated = evaporate_unrolled_loops(graph, plans)

    assert evaporated == ()
    assert any(
        isinstance(data.get("expr_obj"), ast.For)
        for _node_id, data in graph.G.nodes(data=True)
    )


def test_evaporation_ignores_stale_ids_from_retained_loop_plan():
    graph = _function_graph(
        "def kernel(x, keep_running):\n"
        "    for index in range(2):\n"
        "        x = x + index\n"
        "    while keep_running:\n"
        "        x = x + 1\n"
        "    return x\n",
        "kernel",
    )
    composer = _glsl_composer()
    plans = composer.discover(graph)
    unrolled = [plan for plan in plans if plan.strategy is LoopStrategy.UNROLL]
    retained = [plan for plan in plans if plan.strategy is not LoopStrategy.UNROLL]
    assert unrolled and retained
    stale_id = int(retained[0].loop.node_id)
    graph.G.remove_node(stale_id)

    evaporated = evaporate_unrolled_loops(graph, plans)

    assert evaporated
    assert stale_id not in graph.G


def test_static_mapping_generator_reduction_preserves_destructured_outputs():
    graph = _function_graph(
        "def kernel(mapping):\n"
        "    return any(\n"
        "        value > 0 for name, value in mapping.items()\n"
        "    )\n",
        "kernel",
    )
    graph.G.graph["planner_specializations"] = {
        "mapping": {"quiet": 0, "active": 2},
    }
    graph.G.graph["deployment_schedule_preference"] = "asap"
    composer = _glsl_composer()
    plans = composer.discover(graph)

    plan, = plans
    assert plan.loop.target_bindings
    assert plan.loop.iteration_outputs

    evaporate_unrolled_loops(graph, plans)

    generator_id = next(
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.GeneratorExp)
    )
    generator = graph.G.nodes[generator_id]
    assert generator["attributes"]["materialization_kind"] == (
        "unrolled_loop"
    )
    generated_values = tuple(
        parent
        for parent, role in generator["parents"]
        if str(role).startswith("arg")
    )
    assert len(generated_values) == 2
    assert all(
        graph.G.nodes[value_id]["type"] == "greater"
        for value_id in generated_values
    )
    assert not any(
        data.get("label") in {"name", "value"}
        for _node_id, data in graph.G.nodes(data=True)
    )
    deployment, = graph.G.graph["control_deployment_regions"]
    assert deployment.origin == "unrolled_loop"
    assert deployment.schedule == "independent_lanes"
    assert deployment.schedule_preference == "asap"
    assert len(deployment.lanes) == 2
    for lane in deployment.lanes:
        assert lane.source_node_ids
        assert all(
            (deployment.region_id, lane.index)
            in graph.G.nodes[value_id]["attributes"][
                "deployment_memberships"
            ]
            for value_id in lane.source_node_ids
        )
    mapped, = bind_control_deployments_to_regions(
        (deployment,),
        tuple(lane.source_node_ids for lane in deployment.lanes),
    )
    assert mapped.schedule_preference == "asap"
    assert [lane.region_indices for lane in mapped.lanes] == [(0,), (1,)]


def test_literal_callsite_specialization_precedes_single_loop_reduction():
    module = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        module.build_from_ast(ast.parse(
            "def child(x, amounts):\n"
            "    for amount in amounts:\n"
            "        x = x + amount\n"
            "    return x\n"
            "\n"
            "def root(x):\n"
            "    return child(x, (1, 2, 3))\n"
        ))
    reduce_abstract_tensor_topology(module)

    deployment = strategize_shell_deployment(module)
    child = next(
        shell
        for shell in deployment.function_shell_types.values()
        if shell.process_graph.G.graph.get("function_name") == "child"
    )

    assert child.process_graph.G.graph["planner_specializations"] == {
        "amounts": (1, 2, 3),
    }
    assert child.loop_plans == ()
    evaporated, = child.process_graph.G.graph["evaporated_loop_plans"]
    assert evaporated.loop.iterable_constant == (1, 2, 3)


def test_first_class_function_parameter_becomes_a_parametric_callsite_edge():
    module = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        module.build_from_ast(ast.parse(
            "def apply(value, operation):\n"
            "    return operation(value)\n"
            "\n"
            "def increment(value):\n"
            "    return value + 1\n"
            "\n"
            "def root(value):\n"
            "    return apply(value, increment)\n"
        ))
    reduce_abstract_tensor_topology(module)

    deployment_type = strategize_shell_deployment(module)
    deployment = deployment_type()
    root = next(
        shell
        for shell in deployment.function_shells.values()
        if shell.process_graph.G.graph.get("function_name") == "root"
    )
    apply = next(
        shell
        for shell in root.callsite_function_shells.values()
        if shell.process_graph.G.graph.get("function_name") == "apply"
    )
    callback = next(iter(apply.callsite_function_shells.values()))

    assert callback.process_graph.G.graph.get("function_name") == "increment"
    assert any(
        (data.get("attributes") or {}).get("callee_resolution")
        == "bound-function-parameter"
        for _node_id, data in apply.process_graph.G.nodes(data=True)
    )


def test_mutable_public_parameter_is_not_a_planner_specialization():
    module = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        module.build_from_ast(ast.parse(
            "def parse_and_run(source):\n"
            "    return consume(source)\n"
            "\n"
            "def consume(source):\n"
            "    return len(source)\n"
        ))
    reduce_abstract_tensor_topology(module)

    propagate_bound_planner_specializations(
        module,
        "parse_and_run",
        {"source": "sample only"},
        mutable_parameters=("source",),
    )

    entry = module.function_table.entry("parse_and_run").graph
    child = module.function_table.entry("consume").graph
    assert "source" not in entry.G.graph.get(
        "planner_specializations", {}
    )
    assert "source" not in child.G.graph.get(
        "planner_specializations", {}
    )


def test_structural_fold_does_not_specialize_a_public_parameter_default():
    graph = _function_graph(
        "def map_block_bytes(capacity: int = 8) -> int:\n"
        "    return 16 + int(capacity) * 24\n",
        "map_block_bytes",
    )
    input_id = next(
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_name") == "capacity"
    )

    _fold_callsite_structural_values(graph)

    assert input_id in graph.G
    assert graph.G.nodes[input_id]["type"] == "Input"
    assert graph.G.out_degree(input_id) > 0


def test_structural_fold_treats_specialized_scalar_tensor_as_non_none():
    graph = _function_graph(
        "def select_reference(dt, ref=None):\n"
        "    ref = dt if ref is None else ref\n"
        "    return ref\n",
        "select_reference",
    )
    ref_input_id = next(
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_name") == "ref"
    )
    graph.G.graph["parameter_value_abi"] = {
        "ref": {
            "storage": "scalar",
            "dtype": "float64",
            "rank": 0,
            "python_type": "builtins.float",
        },
    }
    if_exp_id = next(
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.IfExp)
    )
    graph.G.nodes[if_exp_id].setdefault("attributes", {})[
        "loop_carried_bindings"
    ] = {"ref": (ref_input_id, ref_input_id)}

    _fold_callsite_structural_values(graph)

    assert not any(
        isinstance(data.get("expr_obj"), ast.IfExp)
        for _node_id, data in graph.G.nodes(data=True)
    )
    assert ref_input_id in graph.G


def test_grounded_method_resolution_uses_declared_parameter_record_identity():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse("""
class CodeBuilder:
    def to_body(self):
        return 1

def run(body: CodeBuilder):
    return body.to_body()
"""))
    reduce_abstract_tensor_topology(graph)
    run_graph = graph.function_table.entry("run").graph
    method_ref = graph.G.graph["class_table"]["CodeBuilder"]["methods"][
        "to_body"
    ]
    run_graph.G.graph["parameter_record_abi"] = {
        "body": {"identity": "CodeBuilder", "fields": {}},
    }
    for _node_id, data in run_graph.G.nodes(data=True):
        attributes = dict(data.get("attributes") or {})
        if isinstance(data.get("expr_obj"), ast.Call):
            attributes.pop("method_ref", None)
            attributes.pop("callee_ref", None)
        if (
            data.get("type") == "Input"
            and attributes.get("binding_name") == "body"
        ):
            attributes.pop("class_ref", None)
        data["attributes"] = attributes

    _resolve_grounded_method_references(run_graph)

    call = next(
        data
        for _node_id, data in run_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
    )
    assert call["attributes"]["method_ref"] == method_ref
    assert call["attributes"]["method_resolution"] == "receiver-class-ref"


def test_structural_fold_follows_declared_nested_record_schema():
    graph = _function_graph(
        "def read_flag(graph):\n"
        "    if isinstance(graph.G.enabled, bool):\n"
        "        return graph.G.enabled\n"
        "    return False\n",
        "read_flag",
    )
    graph.G.graph["parameter_record_abi"] = {
        "graph": {
            "identity": "CompilerProcessGraph",
            "fields": {
                "G": {
                    "storage": "record", "dtype": None, "rank": 0,
                    "mutable": False, "record": "CompilerDiGraph",
                },
            },
        },
    }
    graph.G.graph["program_abi"] = {
        "records": {
            "CompilerProcessGraph": graph.G.graph[
                "parameter_record_abi"
            ]["graph"],
            "CompilerDiGraph": {
                "identity": "CompilerDiGraph",
                "fields": {
                    "enabled": {
                        "storage": "scalar", "dtype": "bool", "rank": 0,
                        "mutable": False,
                    },
                },
            },
        },
        "bindings": [],
        "values": [],
    }

    _fold_callsite_structural_values(graph)

    assert not any(
        data.get("type") == "If" for _node_id, data in graph.G.nodes(data=True)
    )
    assert len(graph.G.graph[
        "structurally_specialized_conditional_node_ids"
    ]) == 1
    assert any(
        (data.get("attributes") or {}).get("attribute") == "enabled"
        for _node_id, data in graph.G.nodes(data=True)
    )


def test_structural_fold_follows_keyed_record_row_schema():
    graph = _function_graph(
        "def read_kind(graph, node_id):\n"
        "    data = graph.G.nodes[node_id]\n"
        "    if isinstance(data.get('kind'), int):\n"
        "        return data.get('kind')\n"
        "    return -1\n",
        "read_kind",
    )
    graph.G.graph["parameter_record_abi"] = {
        "graph": {
            "identity": "CompilerProcessGraph",
            "fields": {
                "G": {
                    "storage": "record", "dtype": None, "rank": 0,
                    "mutable": False, "record": "CompilerDiGraph",
                },
            },
        },
    }
    graph.G.graph["program_abi"] = {
        "records": {
            "CompilerProcessGraph": graph.G.graph[
                "parameter_record_abi"
            ]["graph"],
            "CompilerDiGraph": {
                "identity": "CompilerDiGraph",
                "fields": {
                    "nodes": {
                        "storage": "keyed", "dtype": "int64", "rank": 1,
                        "key_encoding": "integer_identity",
                        "value_record": "CompilerNode",
                    },
                },
            },
            "CompilerNode": {
                "identity": "CompilerNode",
                "fields": {
                    "kind": {
                        "storage": "scalar", "dtype": "int64", "rank": 0,
                        "mutable": False,
                    },
                },
            },
        },
        "bindings": [],
        "values": [],
    }

    _fold_callsite_structural_values(graph)

    assert not any(
        data.get("type") == "If" for _node_id, data in graph.G.nodes(data=True)
    )
    assert any(
        str(data.get("op") or data.get("type")).casefold() == "get"
        for _node_id, data in graph.G.nodes(data=True)
    )


def test_structural_fold_does_not_treat_row_reference_fact_as_its_value():
    graph = _function_graph(
        "def inspect_row(graph, node_id):\n"
        "    data = graph.G.nodes[node_id]\n"
        "    if data.get('type') == 'Constant':\n"
        "        return True\n"
        "    if isinstance(data.get('expr_obj'), tuple):\n"
        "        return True\n"
        "    return False\n",
        "inspect_row",
    )
    graph.G.graph["parameter_record_abi"] = {
        "graph": {
            "identity": "CompilerProcessGraph",
            "fields": {
                "G": {
                    "storage": "record", "record": "CompilerDiGraph",
                },
            },
        },
    }
    graph.G.graph["program_abi"] = {
        "records": {
            "CompilerProcessGraph": graph.G.graph[
                "parameter_record_abi"
            ]["graph"],
            "CompilerDiGraph": {
                "identity": "CompilerDiGraph",
                "fields": {
                    "nodes": {
                        "storage": "keyed", "dtype": "int64", "rank": 1,
                        "value_record": "CompilerNode",
                    },
                },
            },
            "CompilerNode": {
                "identity": "CompilerNode",
                "fields": {
                    "type": {"storage": "reference"},
                    "expr_obj": {"storage": "reference"},
                },
            },
        },
        "bindings": [],
        "values": [],
    }
    authored_if_nodes = sum(
        data.get("type") == "If"
        for _node_id, data in graph.G.nodes(data=True)
    )

    _fold_callsite_structural_values(graph)

    assert sum(
        data.get("type") == "If"
        for _node_id, data in graph.G.nodes(data=True)
    ) == authored_if_nodes


def test_structural_fold_retains_singleton_aggregate_leaf_identity():
    graph = _function_graph(
        "def packet(payload: bytes) -> bytes:\n"
        "    return payload + bytes([11])\n",
        "packet",
    )
    aggregate_id = next(
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("aggregate_kind") == "list"
    )
    leaf_ids = tuple(
        (graph.G.nodes[aggregate_id].get("attributes") or {}).get(
            "aggregate_leaf_value_ids", ()
        )
    )

    _fold_callsite_structural_values(graph)

    attributes = graph.G.nodes[aggregate_id]["attributes"]
    assert graph.G.nodes[aggregate_id]["constant"] == [11]
    assert attributes["structural_specialization"] is True
    assert attributes["aggregate_kind"] == "list"
    assert tuple(attributes["aggregate_leaf_value_ids"]) == leaf_ids


def test_loop_composer_keeps_larger_range_in_glsl_source():
    graph = _function_graph(
        "def kernel(x):\n"
        "    for index in range(64):\n"
        "        x = x + index\n"
        "    return x\n",
        "kernel",
    )

    plan, = _glsl_composer().compose(graph)

    assert plan.strategy is LoopStrategy.NATIVE_SOURCE
    assert plan.loop.trip_count == 64
    assert (plan.loop.start, plan.loop.stop, plan.loop.step) == (0, 64, 1)


def test_unroll_limit_never_shortens_source_loop_domain():
    graph = _function_graph(
        "def kernel(x):\n"
        "    for index in range(4097):\n"
        "        x = x + index\n"
        "    return x\n",
        "kernel",
    )

    plan, = _glsl_composer(unroll_limit=8).compose(graph)

    assert plan.strategy is LoopStrategy.NATIVE_SOURCE
    assert plan.loop.trip_count == 4097
    assert (plan.loop.start, plan.loop.stop, plan.loop.step) == (0, 4097, 1)


def test_reused_loop_spelling_keeps_one_identity_per_lexical_binding():
    graph = _function_graph(
        "def kernel(packets):\n"
        "    widths = tuple(packet.width for packet in packets)\n"
        "    counts = tuple(packet.count for packet in packets)\n"
        "    total = 0\n"
        "    for packet in packets:\n"
        "        total = total + packet.count\n"
        "    return widths, counts, total\n",
        "kernel",
    )

    plans = _glsl_composer().compose(graph)
    bindings = tuple(
        plan.loop.target_bindings[0][1]
        for plan in plans
        if plan.loop.target == "packet"
    )

    assert len(bindings) == 3
    assert len(set(bindings)) == 3
    assert all(graph.G.out_degree(binding) for binding in bindings)


def test_iterable_loop_discards_observed_numeric_capture_bound():
    graph = _function_graph(
        "def kernel(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total = total + item\n"
        "    return total\n",
        "kernel",
    )
    loop_id = next(
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    )
    graph.G.nodes[loop_id].setdefault("attributes", {})["stop"] = 1
    graph.G.nodes[loop_id]["attributes"][
        "iterator_kind"
    ] = "arithmetic_sequence"
    observed_bound = max(graph.G.nodes) + 1
    graph.G.add_node(
        observed_bound,
        expr_obj=ast.Constant(value=1),
        constant=1,
    )
    graph.G.nodes[loop_id]["parents"] = (
        *tuple(graph.G.nodes[loop_id].get("parents") or ()),
        (observed_bound, "stop"),
    )

    plan, = _glsl_composer(unroll_limit=0).compose(graph)

    assert plan.loop.stop is None
    assert plan.loop.stop_node is None
    assert plan.loop.trip_count is None
    assert plan.loop.iterable_node is not None


def test_iterable_loop_recovers_authored_attribute_after_parent_edge_loss():
    graph = _function_graph(
        "def kernel(owner):\n"
        "    total = 0\n"
        "    for item in owner.items:\n"
        "        total = total + item\n"
        "    return total\n",
        "kernel",
    )
    loop_id = next(
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    )
    iterable_parent = next(
        parent
        for parent, role in graph.G.nodes[loop_id]["parents"]
        if str(role) in {"iterable", "iter"}
    )
    graph.G.nodes[loop_id]["parents"] = tuple(
        (parent, role)
        for parent, role in graph.G.nodes[loop_id]["parents"]
        if int(parent) != int(iterable_parent)
        or str(role) not in {"iterable", "iter"}
    )
    if graph.G.has_edge(iterable_parent, loop_id):
        graph.G.remove_edge(iterable_parent, loop_id)
    graph.G.nodes[iterable_parent]["expr_obj"] = None

    plan, = _glsl_composer(unroll_limit=0).compose(graph)

    assert plan.loop.iterable_node == iterable_parent
    assert graph.G.nodes[plan.loop.iterable_node]["type"] == "GetAttr"


def test_parameter_default_does_not_evaporate_runtime_iterable():
    graph = _function_graph(
        "def kernel(items=()):\n"
        "    return tuple(item for item in items)\n",
        "kernel",
    )

    plan, = _glsl_composer().discover(graph)

    assert plan.loop.iterable_constant is None
    assert plan.loop.trip_count is None
    assert plan.strategy not in {LoopStrategy.CONSTANT, LoopStrategy.UNROLL}


def test_reducer_does_not_create_loop_ports_before_planning():
    graph = _function_graph(
        "def kernel(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total = total + item\n"
        "    return total\n",
        "kernel",
    )

    assert not any(
        data.get("type") in {
            "LoopExit",
            "LoopStateTransition",
            "LoopResult",
            "LoopStatePort",
        }
        for _node_id, data in graph.G.nodes(data=True)
    )

    composer = _glsl_composer()
    plans = composer.discover(graph)
    assert all(plan.semantic is None for plan in plans)
    plans = composer.materialize_semantic_ir(graph, plans)
    materialize_retained_loop_ports(graph, plans)
    root, = graph.roots

    assert graph.G.nodes[root]["type"] == "LoopResult"
    assert {role for _parent, role in graph.G.nodes[root]["parents"]} == {
        "control",
        "value",
    }


def test_dynamic_range_bound_is_a_control_dependency_not_none_uniform():
    graph = _function_graph(
        "def kernel(x, count):\n"
        "    for index in range(1, count + 1):\n"
        "        x = x + index\n"
        "    return x\n",
        "kernel",
    )
    plans = _glsl_composer().compose(graph)
    plan, = plans
    assert plan.loop.iterator_kind == "arithmetic_sequence"
    assert plan.loop.iterable_node is None
    assert plan.loop.stop_node is not None
    body = plan.loop.body_nodes
    reduction, = analyze_shader_loop_reductions(graph, plans, (body,))

    assert reduction.collapsible
    assert reduction.control_program is not None
    rendered = render_control_program(
        reduction.control_program,
        ControlTarget.GLSL,
    )
    assert "u_control_None" not in rendered
    assert reduction.preferred_shell == "glsl"
    assert any(
        uniform.name.startswith("u_control_")
        for uniform in reduction.control_program.uniforms
    )
    assert "u_control_" in reduction.control_program.root.stop


def test_raise_guard_inside_retained_loop_becomes_lexical_validation():
    graph = _function_graph(
        "def kernel(values, count):\n"
        "    value = 0\n"
        "    for index in range(count):\n"
        "        value = values[index]\n"
        "        if value < 0:\n"
        "            raise ValueError('negative')\n"
        "        value = value + 1\n"
        "    return value\n",
        "kernel",
    )
    plan, = _glsl_composer().compose(graph)
    reduction, = analyze_shader_loop_reductions(
        graph, (plan,), (plan.loop.body_nodes,)
    )

    assert reduction.collapsible
    assert "Raise" not in reduction.blockers
    assert reduction.control_program is not None
    assert reduction.control_program.root.source_loop_node_id == (
        plans[0].loop.node_id
    )
    root = reduction.control_program.root
    assert isinstance(root, LoopBlock)
    assert isinstance(root.body, SequenceBlock)
    validation = next(
        block for block in root.body.blocks
        if isinstance(block, ValidationBlock)
    )
    assert validation.expect_true is False


def test_shape_index_range_is_recovered_as_bound_not_range_object():
    graph = _function_graph(
        "def kernel(frames):\n"
        "    total = frames[0]\n"
        "    for index in range(frames.shape[0]):\n"
        "        total = total + frames[index]\n"
        "    return total\n",
        "kernel",
    )
    plan, = _glsl_composer().compose(graph)

    assert plan.loop.iterator_kind == "arithmetic_sequence"
    assert plan.loop.stop_node is not None
    assert plan.semantic.domain.stop.value_id == plan.loop.stop_node


def test_static_iterable_is_an_explicit_constant_loop_definition():
    graph = _function_graph(
        "def kernel(x):\n"
        "    for value in (1, 3, 5):\n"
        "        x = x + value\n"
        "    return x\n",
        "kernel",
    )
    plans = _glsl_composer().compose(graph)
    plan, = plans
    reduction, = analyze_shader_loop_reductions(
        graph, plans, (plan.loop.body_nodes,)
    )

    assert plan.semantic.domain.iterable.literal == (1, 3, 5)
    assert plan.semantic.body_closure is not None
    assert reduction.control_program.static_iterable_bindings
    assert not reduction.control_program.iterable_bindings


def test_enumerate_resident_iterable_exports_projected_bindings():
    graph = _function_graph(
        "def kernel(field):\n"
        "    for index, value in enumerate(field):\n"
        "        field[index] = value + 1\n"
        "    return field\n",
        "kernel",
    )
    plan, = _glsl_composer().compose(graph)
    reduction, = analyze_shader_loop_reductions(
        graph, (plan,), (plan.loop.body_nodes,)
    )

    assert reduction.collapsible
    assert reduction.control_program is not None
    bindings = reduction.control_program.projected_iterable_bindings
    assert len(bindings) == 2
    assert bindings[0][3] == "induction"
    assert bindings[1][3] is None
    assert reduction.control_program.root.stop == (
        f"__iterable_extent_{bindings[0][0]}__"
    )


def test_materialized_closure_iterable_binds_resident_source_identities():
    graph = _function_graph(
        "def kernel(values):\n"
        "    frozen = tuple(values)\n"
        "    return tuple(value + 1 for value in frozen)\n",
        "kernel",
    )
    plans = _glsl_composer().compose(graph)
    aggregate = next(
        plan
        for plan in plans
        if plan.semantic.domain_kind.value == "iterable"
        and plan.semantic.domain.access
        is IterableAccess.CLOSURE_AGGREGATE
    )
    reduction, = analyze_shader_loop_reductions(
        graph,
        (aggregate,),
        (aggregate.loop.body_nodes,),
    )

    assert aggregate.semantic.domain.source_value_ids
    assert reduction.collapsible
    assert reduction.control_program is not None
    binding, = reduction.control_program.closure_iterable_bindings
    assert binding[3] == aggregate.semantic.domain.source_value_ids


def test_tuple_generator_materialization_owns_publication_storage():
    graph = _function_graph(
        "def kernel(values):\n"
        "    return AbstractTensor.stack(\n"
        "        tuple(value + 1 for value in values), dim=0\n"
        "    )\n",
        "kernel",
    )
    plan, = _glsl_composer().compose(graph)
    plan, = materialize_retained_loop_ports(graph, (plan,))
    reduction, = analyze_shader_loop_reductions(
        graph,
        (plan,),
        (plan.loop.body_nodes,),
    )
    tuple_id = next(
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if (
            isinstance(data.get("expr_obj"), ast.Call)
            and isinstance(data["expr_obj"].func, ast.Name)
            and data["expr_obj"].func.id == "tuple"
        )
    )

    assert reduction.collapsible
    assert reduction.control_program is not None
    binding, = reduction.control_program.collection_bindings
    collection_id = binding[1]
    assert collection_id != tuple_id
    assert graph.G.nodes[collection_id]["type"] == "LoopResult"
    assert graph.G.nodes[tuple_id]["attributes"]["collection_owner_id"] == (
        collection_id
    )


def test_structural_tuple_generator_does_not_claim_tensor_storage():
    graph = _function_graph(
        "def kernel(values):\n"
        "    return tuple((value, value + 1) for value in values)\n",
        "kernel",
    )
    plan, = _glsl_composer().compose(graph)

    assert planned_collection_bindings(graph, plan.loop) == ()


def test_append_mutation_is_retained_as_explicit_sequence_effect():
    graph = _function_graph(
        "def kernel(values):\n"
        "    results = []\n"
        "    for value in values:\n"
        "        results.append(value + 1)\n"
        "    return results\n",
        "kernel",
    )
    composer = _glsl_composer()
    plan, = composer.compose(graph)
    assert planned_collection_bindings(graph, plan.loop) == ()
    plan, = materialize_retained_loop_ports(graph, (plan,))
    assert planned_collection_bindings(graph, plan.loop) == ()

    assert plan.loop.publication_nodes == ()
    assert plan.loop.backpressured_output is False
    effect, = plan.semantic.state_effects
    assert effect.operator == "append"
    assert effect.mode is LoopStateEffectMode.SEQUENCE_MUTATION
    assert effect.sequence_policy == "duplicates"
    # The descriptor's arena/length/status cells are mutated in place.  A
    # synthetic LoopResult would incorrectly require a value producer inside
    # the loop body for this memory effect.
    assert effect.state_output_id is None
    assert effect.loop_result_id is None

    reductions = analyze_shader_loop_reductions(
        graph,
        (plan,),
        (plan.loop.body_nodes,),
    )
    control = reductions[0].control_program
    assert control is not None
    _module, shortfalls, _ = lower_control_sections_to_ssa(
        control,
        identity_table=graph.G.graph.get("identity_table") or {},
    )
    assert not any(item.name == "loop_carried" for item in shortfalls)


def test_comprehension_result_is_resident_sequence_for_following_loop():
    graph = _function_graph(
        "def kernel(values):\n"
        "    definitions = [value + 1 for value in values]\n"
        "    total = 0\n"
        "    for definition in definitions:\n"
        "        total += definition\n"
        "    return total\n",
        "kernel",
    )
    plans = _glsl_composer().compose(graph)
    consumer = next(
        plan
        for plan in plans
        if plan.loop.source_type == "For"
        and (graph.G.nodes[plan.loop.node_id].get("attributes") or {}).get(
            "target"
        ) == "definition"
    )

    assert consumer.semantic is not None
    domain = consumer.semantic.domain
    assert domain.access is IterableAccess.RESIDENT
    assert (
        graph.G.nodes[domain.iterable.value_id].get("attributes") or {}
    ).get("producer_kind") == "sequence_materialization"


def test_comprehension_clause_has_no_statement_body_for_return_analysis():
    graph = _function_graph(
        "def kernel(values):\n"
        "    return [value for value in values if value]\n",
        "kernel",
    )
    composer = _glsl_composer()
    comprehension_ids = tuple(
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.comprehension)
    )

    assert comprehension_ids
    assert all(
        composer.describe(graph, node_id).source_type == "comprehension"
        for node_id in comprehension_ids
    )


def test_indexed_table_store_is_memory_effect_not_value_carried_phi():
    graph = _function_graph(
        "def kernel(values):\n"
        "    table = {}\n"
        "    for value in values:\n"
        "        table[value] = value + 1\n"
        "    return table\n",
        "kernel",
    )
    plans = _glsl_composer().compose(graph)
    plan, = plans
    assert plan.loop.carried_bindings
    assert {
        graph.G.nodes[updated].get("type")
        for _name, _initial, updated in plan.loop.carried_bindings
    } == {"IndexedStore"}

    plan, = materialize_retained_loop_ports(graph, plans)
    reduction, = analyze_shader_loop_reductions(
        graph,
        (plan,),
        (plan.loop.body_nodes,),
    )
    control = reduction.control_program
    assert control is not None
    assert control.root.carried_aliases == ()
    _module, shortfalls, _ = lower_control_sections_to_ssa(
        control,
        identity_table=graph.G.graph.get("identity_table") or {},
    )
    assert not any(item.name == "loop_carried" for item in shortfalls)


def test_sequence_mutation_no_longer_uses_index_publication_shortcut():
    graph = _function_graph(
        "def kernel(values):\n"
        "    results = []\n"
        "    for value in values:\n"
        "        results.append(value + 1)\n"
        "    return results\n",
        "kernel",
    )
    composer = _glsl_composer()
    plan, = composer.discover(graph)
    plan, = composer.materialize_semantic_ir(graph, (plan,))
    plan, = materialize_retained_loop_ports(graph, (plan,))
    effect, = plan.loop.state_effects
    renamed = replace(effect, operator="source_method_name_is_irrelevant")
    loop = replace(plan.loop, state_effects=(renamed,))

    assert renamed.mode is LoopStateEffectMode.SEQUENCE_MUTATION
    assert planned_collection_bindings(graph, loop) == ()


def test_mapping_update_becomes_deterministic_table_item_mutation():
    graph = _function_graph(
        "def kernel(rows):\n"
        "    table = {}\n"
        "    for key, value in rows:\n"
        "        table.update({key: value})\n"
        "    return table\n",
        "kernel",
    )
    plan, = _glsl_composer().compose(graph)
    effect, = plan.loop.state_effects

    assert effect.mode is LoopStateEffectMode.MAPPING_MUTATION
    assert plan.strategy is not LoopStrategy.UNROLL

    plan, = materialize_retained_loop_ports(graph, (plan,))
    reduction, = analyze_shader_loop_reductions(
        graph, (plan,), (plan.loop.body_nodes,),
    )
    mutation, = reduction.control_program.root.sequence_mutations

    assert mutation.operator == "update"
    assert mutation.argument_kind == "mapping_items"
    assert mutation.policy == "unique"
    assert mutation.argument_value_ids == (0, 2)


def test_mapping_pop_none_preserves_optional_result_identity():
    graph = _function_graph(
        "def kernel(rows):\n"
        "    table = {}\n"
        "    for key, value in rows:\n"
        "        previous = table.pop(key, None)\n"
        "    return table\n",
        "kernel",
    )
    plan, = _glsl_composer().compose(graph)
    effect, = plan.loop.state_effects

    assert effect.mode is LoopStateEffectMode.MAPPING_MUTATION

    plan, = materialize_retained_loop_ports(graph, (plan,))
    reduction, = analyze_shader_loop_reductions(
        graph, (plan,), (plan.loop.body_nodes,),
    )
    mutation, = reduction.control_program.root.sequence_mutations

    assert mutation.operator == "pop"
    assert mutation.argument_kind == "mapping_pop_default_none"
    assert len(mutation.argument_value_ids) == 2


def test_zero_argument_list_pop_is_a_resident_loop_state_transition():
    graph = _function_graph(
        "def kernel(values):\n"
        "    pending = list(values)\n"
        "    total = 0\n"
        "    while pending:\n"
        "        total = total + pending.pop()\n"
        "    return total\n",
        "kernel",
    )
    plan, = _glsl_composer().compose(graph)
    effect, = plan.loop.state_effects

    assert effect.operator == "pop"
    assert effect.mode is LoopStateEffectMode.SEQUENCE_MUTATION
    assert effect.argument_value_ids == ()

    plan, = materialize_retained_loop_ports(graph, (plan,))
    reduction, = analyze_shader_loop_reductions(
        graph, (plan,), (plan.loop.body_nodes,),
    )
    mutation, = reduction.control_program.root.sequence_mutations
    assert mutation.operator == "pop"
    assert mutation.policy == "duplicates"


def test_sequence_mutation_expression_resolves_deterministic_value_identity():
    graph = _function_graph(
        "def kernel(values):\n"
        "    results = []\n"
        "    for value in values:\n"
        "        results.append(value + 1)\n"
        "    return results\n",
        "kernel",
    )
    plan, = _glsl_composer().compose(graph)
    plan, = materialize_retained_loop_ports(graph, (plan,))
    effect, = plan.loop.state_effects
    original_argument, = effect.argument_value_ids
    deterministic_value_id = max(map(int, graph.G.nodes)) + 1000
    assert deterministic_value_id not in graph.G
    graph.G.nodes[int(original_argument)]["value_id"] = deterministic_value_id
    renamed_effect = replace(
        effect, argument_value_ids=(deterministic_value_id,),
    )
    plan = replace(
        plan,
        loop=replace(plan.loop, state_effects=(renamed_effect,)),
    )

    reduction, = analyze_shader_loop_reductions(
        graph, (plan,), (plan.loop.body_nodes,),
    )
    mutation, = reduction.control_program.root.sequence_mutations

    assert mutation.argument_value_ids == (deterministic_value_id,)
    assert mutation.argument_expressions[0].value_id == deterministic_value_id


def test_extend_mutation_is_explicit_sequence_effect_not_index_publication():
    graph = _function_graph(
        "def kernel(values):\n"
        "    results = []\n"
        "    for index in range(2):\n"
        "        results.extend((values[index],))\n"
        "    return results\n",
        "kernel",
    )
    composer = _glsl_composer()
    plan, = composer.compose(graph)
    effect, = plan.loop.state_effects

    assert effect.mode is LoopStateEffectMode.SEQUENCE_MUTATION
    assert plan.strategy is not LoopStrategy.UNROLL
    assert planned_collection_bindings(graph, plan.loop) == ()


def test_method_spelling_does_not_turn_unknown_receiver_into_sequence():
    effect = LoopStateEffect(
        state_name="sink",
        operator="append",
        state_input_id=1,
        effect_node_id=2,
        mode=LoopStateEffectMode.OPAQUE,
    )

    assert effect.operator == "append"
    assert effect.mode is LoopStateEffectMode.OPAQUE
    assert effect.sequence_policy is None


def test_bootstrap_collection_mutation_occurrence_ledger_is_not_collapsed():
    expected = {
        "_attach_external_methods": [("add", "unique")],
        "_expand_unresolved_ast_parents": [
            ("append", "duplicates"),
            ("add", "unique"),
            ("extend", "duplicates"),
            ("append", "duplicates"),
            ("append", "duplicates"),
        ],
        "build_from_ast": [
            ("append", "duplicates"),
            ("add", "unique"),
        ],
    }
    targets = (
        graph_express2._attach_external_methods,
        graph_express2._expand_unresolved_ast_parents,
        ProcessGraph.build_from_ast,
    )
    for target in targets:
        source = textwrap.dedent(inspect.getsource(target))
        graph = _function_graph(source, target.__name__)
        plans = _glsl_composer().compose(graph)
        actual = [
            (effect.operator, effect.sequence_policy)
            for plan in plans
            for effect in plan.loop.state_effects
            if effect.mode is LoopStateEffectMode.SEQUENCE_MUTATION
        ]
        # This is an authored-operation ledger, not a count of compiler
        # worklist visits. Binding-revision requeues must never fabricate
        # a second runtime mutation for the same source Call node.
        assert actual == expected[target.__name__]


def test_ifexp_list_merge_preserves_each_args_extend_and_append_occurrence():
    source = textwrap.dedent(inspect.getsource(ProcessGraph.build_graph))
    graph = _function_graph(source, "build_graph")
    plans = _glsl_composer().compose(graph)
    args_effects = [
        effect
        for plan in plans
        for effect in plan.loop.state_effects
        if effect.state_name == "args"
    ]

    assert [effect.operator for effect in args_effects] == ["extend", "append"]
    assert all(
        effect.mode is LoopStateEffectMode.SEQUENCE_MUTATION
        and effect.sequence_policy == "duplicates"
        for effect in args_effects
    )


def test_annotated_sequence_parameter_carries_policy_into_loop_mutation():
    graph = _function_graph(
        "def kernel(values: list, seen: set):\n"
        "    for value in values:\n"
        "        seen.add(value)\n",
        "kernel",
    )
    plan, = _glsl_composer().compose(graph)
    effect, = plan.loop.state_effects

    assert effect.mode is LoopStateEffectMode.SEQUENCE_MUTATION
    assert effect.sequence_policy == "unique"


def test_extensive_range_collection_defaults_to_c_dispatch_shell():
    graph = _function_graph(
        "def kernel(value, count):\n"
        "    results = []\n"
        "    for index in range(count):\n"
        "        results.append(value + index)\n"
        "    return results\n",
        "kernel",
    )
    plan, = _glsl_composer().compose(graph)
    plan, = materialize_retained_loop_ports(graph, (plan,))
    reduction, = analyze_shader_loop_reductions(
        graph,
        (plan,),
        (plan.loop.body_nodes,),
    )

    assert reduction.collapsible
    assert reduction.preferred_shell == "c"
    assert reduction.dispatch_closure_count == 1
    assert reduction.control_program.root.dispatch_shell == "c"
    mutation, = reduction.control_program.root.sequence_mutations
    assert mutation.operator == "append"
    assert mutation.policy == "duplicates"


def test_semantic_loop_body_is_dependency_ordered_before_effect_publication():
    graph = _function_graph(
        "def kernel(values):\n"
        "    results = []\n"
        "    for value in values:\n"
        "        computed = value + 1\n"
        "        results.append(computed)\n"
        "    return results\n",
        "kernel",
    )

    plan, = _glsl_composer().compose(graph)
    lines = plan.semantic.body_closure.items
    positions = {
        int(output): position
        for position, line in enumerate(lines)
        for output in line.outputs
    }

    for position, line in enumerate(lines):
        for value_id in line.inputs:
            if int(value_id) in positions:
                assert positions[int(value_id)] < position
    effect, = plan.semantic.state_effects
    assert positions[int(effect.argument_value_ids[0])] < positions[
        int(effect.effect_node_id)
    ]


def test_generator_yield_is_planner_owned_backpressured_loop_output():
    graph = _function_graph(
        "def chunks(values):\n"
        "    for value in values:\n"
        "        yield value\n",
        "chunks",
    )

    plan, = _glsl_composer().compose(graph)

    stream = graph.G.graph["generator_stream"]
    assert stream["execution_owner"] == "planner_shell"
    assert stream["flow_control"] == "downstream_capacity"
    assert plan.loop.yield_nodes == stream["yield_nodes"]
    assert plan.loop.backpressured_output
    reduction, = analyze_shader_loop_reductions(
        graph,
        (plan,),
        (plan.loop.body_nodes,),
    )
    assert reduction.collapsible
    assert "backpressured-yield" not in reduction.blockers
    assert reduction.control_program is not None
    loop = reduction.control_program.root
    assert isinstance(loop, LoopBlock)
    assert isinstance(loop.body, SequenceBlock)
    publications = tuple(
        block
        for block in loop.body.blocks
        if isinstance(block, StreamPublishBlock)
    )
    assert len(publications) == 1
    publication = publications[0]
    yield_id = plan.loop.yield_nodes[0]
    payload_id = next(
        parent
        for parent, role in graph.G.nodes[yield_id]["parents"]
        if role == "value"
    )
    assert publication.value_id == payload_id


def test_generator_extend_routes_yields_and_filter_into_destination_insert():
    graph = _function_graph(
        "def kernel(batches):\n"
        "    out = []\n"
        "    for batch in batches:\n"
        "        out.extend(x for x in batch if x > 0)\n"
        "    return out\n",
        "kernel",
    )
    plans = _glsl_composer().compose(graph)
    plans = materialize_retained_loop_ports(graph, plans)
    reductions = analyze_shader_loop_reductions(
        graph,
        plans,
        tuple(plan.loop.body_nodes for plan in plans),
    )
    generator_index = next(
        index
        for index, plan in enumerate(plans)
        if plan.loop.source_type == "comprehension"
    )
    outer_index = next(
        index
        for index, plan in enumerate(plans)
        if plan.loop.source_type == "For"
    )
    generator_control = reductions[generator_index].control_program
    outer_control = reductions[outer_index].control_program

    assert generator_control is not None
    assert outer_control is not None
    assert generator_control.collection_bindings == ()
    mutation, = generator_control.root.sequence_mutations
    assert mutation.operator == "append"
    assert mutation.argument_kind == "value"
    assert mutation.predicate_expression is not None
    assert outer_control.root.sequence_mutations == ()

    module, shortfalls, _ = lower_control_sections_to_ssa(
        generator_control,
        identity_table=graph.G.graph.get("identity_table") or {},
    )
    assert shortfalls == ()
    assert (
        f"ssa_sequence_{mutation.sequence_value_id}_append"
        in module.functions
    )
    assert {
        "sequence_mutation_selected",
        "sequence_mutation_skipped",
        "sequence_mutation_merge",
    } <= set(module.functions["planned_control"].blocks)


def test_list_comprehension_extend_retains_eager_materialized_source():
    graph = _function_graph(
        "def kernel(batches):\n"
        "    out = []\n"
        "    for batch in batches:\n"
        "        out.extend([x for x in batch])\n"
        "    return out\n",
        "kernel",
    )
    plans = _glsl_composer().compose(graph)
    plans = materialize_retained_loop_ports(graph, plans)
    reductions = analyze_shader_loop_reductions(
        graph,
        plans,
        tuple(plan.loop.body_nodes for plan in plans),
    )
    comprehension_index = next(
        index
        for index, plan in enumerate(plans)
        if plan.loop.source_type == "comprehension"
    )
    outer_index = next(
        index
        for index, plan in enumerate(plans)
        if plan.loop.source_type == "For"
    )
    comprehension_control = reductions[comprehension_index].control_program
    outer_control = reductions[outer_index].control_program

    assert comprehension_control is not None
    assert comprehension_control.collection_bindings == ()
    comprehension_mutation, = comprehension_control.root.sequence_mutations
    assert comprehension_mutation.operator == "append"
    assert comprehension_mutation.sequence_value_id == (
        plans[comprehension_index].loop.iteration_outputs[0].result_value_id
    )
    mutation, = outer_control.root.sequence_mutations
    assert mutation.operator == "extend"
    assert mutation.argument_kind == "sequence"


def test_glsl_native_loop_wraps_an_already_lowered_region():
    source = emit_native_for_loop(
        ("float next = state + delta;", "state = next;"),
        induction="iteration",
        start=0,
        stop=64,
        step=1,
    )

    assert source == (
        "    for (int iteration = int(0); iteration < int(64); "
        "iteration += int(1)) {",
        "        float next = state + delta;",
        "        state = next;",
        "    }",
    )


def test_loop_shader_reduction_runs_after_region_compartmentalization():
    graph = _function_graph(
        "def kernel(x):\n"
        "    for index in range(64):\n"
        "        x = x * x + 1\n"
        "    return x\n",
        "kernel",
    )
    plans = _glsl_composer().compose(graph)
    body = plans[0].loop.body_nodes

    reduction, = analyze_shader_loop_reductions(
        graph,
        plans,
        (body,),
    )

    assert reduction.collapsible
    assert reduction.region_indices == (0,)
    assert reduction.estimated_dispatches_removed == 63
    assert reduction.preferred_shell == "glsl"
    assert reduction.dispatch_closure_count == 1
    assert reduction.control_program is not None
    from src.compiler.control_source import ControlTarget
    rendered = render_control_program(
        reduction.control_program,
        ControlTarget.GLSL,
    )
    assert "__scheduled_region_0__" in rendered
    assert rendered.startswith(
        f"for (int iteration_{plans[0].loop.node_id} = 0; "
        f"iteration_{plans[0].loop.node_id} < 64;"
    )


def test_while_condition_and_break_become_planner_control_edges():
    graph = _function_graph(
        "def kernel(x):\n"
        "    while x < 8:\n"
        "        x = x + 1\n"
        "        if x > 4:\n"
        "            break\n"
        "    return x\n",
        "kernel",
    )
    plan, = _glsl_composer().compose(graph)

    assert plan.loop.break_nodes
    reduction, = analyze_shader_loop_reductions(
        graph,
        (plan,),
        (plan.loop.condition_nodes, plan.loop.body_nodes),
    )

    assert reduction.collapsible
    assert reduction.blockers == ()
    assert reduction.control_program is not None
    loop = reduction.control_program.root
    assert isinstance(loop, WhileBlock)
    assert loop.predicate_value_id == plan.loop.condition_nodes[0]
    assert any(
        isinstance(block, LoopControlBlock) and block.action == "break"
        for block in loop.body.blocks
    )


def test_while_sequence_condition_is_explicit_length_predicate():
    graph = _function_graph(
        "def kernel(items):\n"
        "    pending = list(items)\n"
        "    while pending:\n"
        "        break\n"
        "    return pending\n",
        "kernel",
    )
    plan, = _glsl_composer().compose(graph)

    reduction, = analyze_shader_loop_reductions(
        graph,
        (plan,),
        (plan.loop.condition_nodes, plan.loop.body_nodes),
    )

    assert reduction.collapsible
    loop = reduction.control_program.root
    assert isinstance(loop, WhileBlock)
    assert loop.predicate_expression.op == "sequence_nonempty"
    assert loop.predicate_expression.value_id == plan.loop.condition_nodes[0]


def test_while_ternary_assignment_is_explicit_loop_carried_state():
    graph = _function_graph(
        "def infer_shape(data):\n"
        "    shape = []\n"
        "    while isinstance(data, list):\n"
        "        shape.append(len(data))\n"
        "        data = data[0] if data else []\n"
        "    return tuple(shape)\n",
        "infer_shape",
    )

    plan, = _glsl_composer().compose(graph)

    assert plan.strategy is LoopStrategy.NATIVE_SOURCE
    carried = {
        name: (initial, updated)
        for name, initial, updated in plan.loop.carried_bindings
    }
    assert "data" in carried
    assert carried["data"][0] != carried["data"][1]
    assert plan.loop.condition_nodes
    assert plan.loop.body_nodes


def test_multi_carried_recurrence_preservation_outranks_unrolling():
    graph = _function_graph(
        "def kernel(a, b):\n"
        "    for index in range(4):\n"
        "        a = a + index\n"
        "        b = b - index\n"
        "    return a + b\n",
        "kernel",
    )

    plan, = _glsl_composer().compose(graph)

    assert len(plan.loop.carried_bindings) == 2
    assert plan.strategy is LoopStrategy.NATIVE_SOURCE
    assert "outranks loop unrolling" in plan.reason


def test_sequential_loop_body_does_not_own_captured_predecessor_values():
    graph = _function_graph(
        "def kernel(xs, ys):\n"
        "    total = 0\n"
        "    for x in xs:\n"
        "        total = total + x\n"
        "    outputs = []\n"
        "    for y in ys:\n"
        "        outputs.append(total * y)\n"
        "    return outputs\n",
        "kernel",
    )

    plans = sorted(
        _glsl_composer().discover(graph),
        key=lambda plan: graph.G.nodes[plan.loop.node_id]["source_span"]["line"],
    )
    first, second = plans
    first_result = next(
        node_id for node_id in first.loop.body_nodes
        if (graph.G.nodes[node_id].get("source_span") or {}).get("line") == 4
    )

    assert first_result not in second.loop.body_nodes
    assert all(
        node_id in dict(second.loop.target_bindings).values()
        or (graph.G.nodes[node_id].get("source_span") or {}).get("line") == 7
        for node_id in second.loop.body_nodes
    )
