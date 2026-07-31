from __future__ import annotations

import ast
import contextlib
import io
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
    analyze_shader_loop_reductions,
    evaporate_unrolled_loops,
    materialize_retained_loop_ports,
    planned_collection_bindings,
)
from src.compiler.glsl_deployment_strategy import (
    strategize_glsl_deployment,
)
from src.compiler.loop_ir import (
    IterableAccess,
    LoopStateEffectMode,
)
from src.compiler.control_source import (
    ControlTarget,
    LoopBlock,
    SequenceBlock,
    StreamPublishBlock,
    render_control_program,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


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


def test_unroll_strategy_evaporates_to_straight_line_value_graph():
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

    assert len(evaporated) == 1
    assert composer.compose(graph) == ()
    assert not any(
        isinstance(
            data.get("expr_obj"),
            (ast.For, ast.While, ast.comprehension),
        )
        for _node_id, data in graph.G.nodes(data=True)
    )
    assert not any(
        data.get("type") in {"LoopExit", "LoopStateTransition"}
        for _node_id, data in graph.G.nodes(data=True)
    )
    root, = graph.roots
    stack_inputs = tuple(
        parent
        for parent, role in graph.G.nodes[root]["parents"]
        if str(role).startswith("arg")
    )
    assert len(stack_inputs) == 3
    assert all(graph.G.nodes[value_id]["type"] == "Add"
               for value_id in stack_inputs)
    assert len(set(stack_inputs)) == 3


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

    deployment = strategize_glsl_deployment(module)
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


def test_append_publication_storage_is_post_loop_owner_not_initializer():
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
    binding, = planned_collection_bindings(graph, plan.loop)
    owner_id = binding[1]
    initializer_id = next(
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.List)
    )

    assert plan.loop.publication_nodes == ()
    assert plan.loop.backpressured_output is False
    effect, = plan.semantic.state_effects
    assert effect.operator == "append"
    assert effect.mode is LoopStateEffectMode.INDEXED_PUBLICATION
    assert effect.argument_value_ids[0] == binding[0]
    assert effect.loop_result_id == binding[1]
    assert graph.G.nodes[owner_id].get("type") == "LoopResult"
    assert owner_id != initializer_id


def test_collection_lowering_uses_effect_semantics_not_method_spelling():
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

    binding, = planned_collection_bindings(graph, loop)

    assert binding[0] == renamed.argument_value_ids[0]
    assert binding[1] == renamed.loop_result_id


def test_unknown_mutation_is_not_misclassified_as_collection_publication():
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

    assert effect.mode is LoopStateEffectMode.OPAQUE
    assert plan.strategy is not LoopStrategy.UNROLL
    assert planned_collection_bindings(graph, plan.loop) == ()


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
