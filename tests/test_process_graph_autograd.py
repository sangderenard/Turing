from __future__ import annotations

import ctypes
import re

import networkx as nx
import numpy as np
import pytest

from src.compiler.process_graph_autograd import (
    abstract_tensor_program_to_process_graph,
    ConditionalAdjointContract,
    compile_process_graph_backward,
    ProcessGraphAutogradError,
    LoopAdjointContract,
    differentiate_control_program,
    differentiate_process_graph,
    differentiate_process_program,
    fuse_forward_loss_backward,
    isolate_process_program_adjoint_regions,
    lower_training_motion_to_repository_ssa,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _add(graph, node_id, op, parents=(), *, label=None, shape=(1,)):
    parent_items = [(parent, f"arg{index}") for index, parent in enumerate(parents)]
    graph.G.add_node(
        node_id,
        op=op,
        type=op,
        label=label or op,
        parents=parent_items,
        children=[],
        attributes={},
        extra_args={},
        tensor={"shape": tuple(shape), "dtype": "float64"},
        control={},
        constant=None,
        expr_obj=None,
        store_id=None,
    )
    for parent, role in parent_items:
        graph.G.add_edge(parent, node_id, role=role)
        graph.G.nodes[parent]["children"].append((node_id, role))


def test_process_graph_adjoint_is_parametric_and_accumulates_shared_uses():
    # loss = (w*x) + (w*x): dw has two independently generated contributions.
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "input", label="w")
    _add(graph, 2, "input", label="x")
    _add(graph, 3, "mul", (1, 2))
    _add(graph, 4, "mul", (1, 2))
    _add(graph, 5, "add", (3, 4))
    graph.roots = [5]

    adjoint = differentiate_process_graph(graph, wrt=(1, 2))
    backward = adjoint.backward

    assert backward.G.graph["graph_kind"] == "parametric_backward"
    assert backward.G.graph["python_backward_callbacks"] is False
    assert backward.G.graph["backward_rule_registry"].endswith("BACKWARD_RULES")
    assert set(adjoint.seed_value_ids) == {5}
    # Multiplication retains its numeric operands; addition also retains the
    # two result descriptors required by the authored unbroadcast identity.
    assert {
        value_id for value_id, contract in adjoint.saved_value_contracts.items()
        if contract.storage == "resident"
    } == {1, 2}
    assert {
        value_id for value_id, contract in adjoint.saved_value_contracts.items()
        if contract.storage == "descriptor"
    } == {3, 4}
    assert set(adjoint.gradient_value_ids) == {1, 2}
    # Both requested gradients are explicit graph outputs, and each shared
    # input's contributions meet at an ordinary add node.
    assert backward.roots == [
        adjoint.gradient_value_ids[1], adjoint.gradient_value_ids[2]
    ]
    for source_id in (1, 2):
        root = adjoint.gradient_value_ids[source_id]
        assert backward.G.nodes[root]["op"] == "add"
        assert backward.G.in_degree(root) == 2


def test_saved_values_are_named_runtime_inputs_not_embedded_python_objects():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 10, "input", label="left")
    _add(graph, 11, "input", label="right")
    _add(graph, 12, "mul", (10, 11))
    graph.roots = [12]

    adjoint = differentiate_process_graph(graph)
    for forward_id, backward_id in adjoint.saved_value_ids.items():
        data = adjoint.backward.G.nodes[backward_id]
        assert data["op"] == "input"
        assert data["attributes"] == {
            "name": f"saved_{forward_id}",
            "binding_kind": "saved_forward",
            "source_forward_id": forward_id,
        }
        assert not any(callable(value) for value in data.values())


@pytest.mark.parametrize("packaging", ("independent", "combined"))
def test_every_backward_request_returns_the_adjoint_binding_graph(packaging):
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "input", label="left", shape=(2, 2))
    _add(graph, 2, "input", label="right", shape=(2, 2))
    _add(graph, 3, "mul", (1, 2), shape=(2, 2))
    graph.roots = [3]

    product = compile_process_graph_backward(
        graph, wrt=(1, 2), packaging=packaging,
    )

    assert product.binding_graph is product.adjoint.binding_graph
    assert set(product.binding_graph.graph) == {1, 2}
    if packaging == "independent":
        assert product.motion is None
        assert product.graph is product.adjoint.backward
    else:
        assert product.motion is not None
        assert product.binding_graph is product.motion.binding_graph
        assert product.graph is product.motion.graph


def test_abstract_tensor_ingestion_retains_a_complete_multi_output_surface():
    from src.common.tensors.accelerator_backends.ssa_backend import (
        SSATensorOperations,
        SSATensorProgram,
    )

    program = SSATensorProgram("two_output_surface")
    left = SSATensorOperations.input(program, (2,))
    right = SSATensorOperations.input(program, (2,))
    added, multiplied = left + right, left * right

    graph = abstract_tensor_program_to_process_graph(
        (added, multiplied), bindings={"left": left, "right": right},
    )

    assert graph.roots == [added.data.value.id, multiplied.data.value.id]
    assert {graph.G.nodes[root]["op"] for root in graph.roots} == {"add", "mul"}


def test_backward_compilation_obeys_the_execution_contract_source():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["execution_contract"] = {
        "backward_source": "authored_python",
    }
    _add(graph, 1, "input", label="x")
    _add(graph, 2, "sin", (1,))
    graph.roots = [2]

    with pytest.raises(ProcessGraphAutogradError, match="authored_python"):
        compile_process_graph_backward(graph)


def test_backward_graph_inherits_python_host_opportunistic_dispatch_contract():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["execution_contract"] = {
        "host_runtime": "python",
        "dependency_search": "reachable",
        "native_lowering": "opportunistic",
        "dispatch_unit": "isolated_numeric_subgraph",
        "unlowered_behavior": "execute_in_python",
        "require_full_native": False,
        "backward_source": "process_graph",
        "python_callbacks": "contract_only",
        "numeric_semantics": "abstract_tensor",
        "scalar_promotion": "all_numeric",
    }
    _add(graph, 1, "input", label="x")
    _add(graph, 2, "sin", (1,))
    graph.roots = [2]

    adjoint = differentiate_process_graph(graph)

    assert adjoint.backward.G.graph["execution_contract"] == (
        graph.G.graph["execution_contract"]
    )
    assert (
        adjoint.backward.G.graph["deployment_role"]
        == "opportunistic_numeric_dispatch"
    )


def test_unknown_operator_fails_closed_instead_of_storing_backward_callable():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "input", label="x")
    _add(graph, 2, "opaque_python", (1,))
    graph.roots = [2]

    with pytest.raises(ProcessGraphAutogradError, match="2:opaque_python"):
        differentiate_process_graph(graph)


def test_cyclic_control_refuses_observed_traversal_differentiation():
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "input", label="state")
    _add(graph, 2, "mul", (1,))
    graph.G.add_edge(2, 1, role="carried")
    graph.G.nodes[1]["parents"].append((2, "carried"))
    graph.G.nodes[2]["children"].append((1, "carried"))
    graph.roots = [2]

    with pytest.raises(ProcessGraphAutogradError, match="ControlProgram"):
        differentiate_process_graph(graph)


def test_mse_numeric_subgraph_derives_graph_backed_adjoint_without_tape():
    # This is the exact numeric return cone of abstract_nn.MSELoss.forward:
    # mean((pred - target) * (pred - target)).
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "input", label="pred", shape=(2, 2))
    _add(graph, 2, "input", label="target", shape=(2, 2))
    _add(graph, 3, "sub", (1, 2), shape=(2, 2))
    _add(graph, 4, "mul", (3, 3), shape=(2, 2))
    _add(graph, 5, "mean", (4,), shape=())
    graph.roots = [5]

    adjoint = differentiate_process_graph(graph, wrt=(1, 2))
    assert len(adjoint.backward.roots) == 2
    assert set(adjoint.gradient_value_ids) == {1, 2}
    assert adjoint.backward.G.graph["python_backward_callbacks"] is False
    assert any(
        data["op"] == "Call"
        for _node, data in adjoint.backward.G.nodes(data=True)
    )
    assert not {
        "adjoint_reduce_to_shape",
        "adjoint_expand_reduction",
        "adjoint_reduction_size",
    } & {data["op"] for _node, data in adjoint.backward.G.nodes(data=True)}
    assert set(adjoint.backward.G.graph["backward_rule_nodes"].values()) == {
        "sub", "mul", "mean"
    }


def test_linear_forward_loss_backward_is_one_parametric_graph_motion(tmp_path):
    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "input", label="x", shape=(2, 3))
    _add(graph, 2, "input", label="W", shape=(3, 2))
    _add(graph, 3, "input", label="b", shape=(2,))
    _add(graph, 4, "input", label="target", shape=(2, 2))
    _add(graph, 5, "matmul", (1, 2), shape=(2, 2))
    _add(graph, 6, "add", (5, 3), shape=(2, 2))
    _add(graph, 7, "sub", (6, 4), shape=(2, 2))
    _add(graph, 8, "mul", (7, 7), shape=(2, 2))
    _add(graph, 9, "mean", (8,), shape=())
    graph.roots = [9]

    adjoint = differentiate_process_graph(graph, wrt=(1, 2, 3))
    assert all(
        contract.storage in {"resident", "descriptor"}
        and contract.lifetime == "forward_through_backward"
        for contract in adjoint.saved_value_contracts.values()
    )
    assert {
        value_id for value_id, contract
        in adjoint.saved_value_contracts.items()
        if contract.storage == "resident"
    } == {1, 2, 7}
    assert adjoint.binding_graph.graph.nodes[7]["kind"] == "product"
    assert adjoint.binding_graph.graph.nodes[3]["kind"] == "parameter"
    assert adjoint.gradient_contracts[2].binding_name == "W"
    assert adjoint.gradient_contracts[2].accumulation == "sum"
    x = np.asarray([[0.2, -0.3, 0.7], [1.1, 0.4, -0.2]])
    weight = np.asarray([[0.5, -0.1], [0.2, 0.8], [-0.4, 0.3]])
    bias = np.asarray([0.05, -0.15])
    target = np.asarray([[0.1, 0.9], [-0.2, 0.3]])
    prediction = x @ weight + bias
    diff = prediction - target
    grad_prediction = 2.0 * diff / diff.size
    assert set(adjoint.backward.G.graph["backward_rule_nodes"].values()) == {
        "matmul", "add", "sub", "mul", "mean"
    }

    motion = fuse_forward_loss_backward(adjoint)
    assert motion.graph.G.graph["graph_kind"] == "forward_loss_backward_motion"
    assert motion.graph.G.graph["optimizer_included"] is False
    assert motion.loss_value_ids == (9,)
    assert set(motion.gradient_value_ids) == {1, 2, 3}
    assert all(
        data.get("attributes", {}).get("binding_kind") != "saved_forward"
        for _node, data in motion.graph.G.nodes(data=True)
    )
    assert motion.graph.G.nodes[motion.seed_value_ids[9]]["op"] == "const"
    assert {
        data.get("attributes", {}).get("training_motion_phase")
        for _node, data in motion.graph.G.nodes(data=True)
    } == {"forward", "backward"}

    lowering = lower_training_motion_to_repository_ssa(motion)
    assert lowering.shortfalls == ()
    assert set(lowering.outputs) == {"loss_0", "grad_1", "grad_2", "grad_3"}

    from src.compiler.ssa_llvm_backend import (
        compile_artifact,
        emit_ssa_function_to_llvm,
    )
    llvm = emit_ssa_function_to_llvm(
        lowering.module,
        lowering.function_name,
        entry_name=lowering.function_name,
    )
    assert llvm.shortfalls == ()
    buffer_definitions = re.findall(
        r"^\s*(%buffer(?:\.addr)?\.\d+)\s*=", llvm.llvm_ir, re.MULTILINE,
    )
    assert len(buffer_definitions) == len(set(buffer_definitions))
    native = compile_artifact(llvm, directory=tmp_path / "native_motion")
    assert native.library_path is not None and native.library_path.is_file()
    entry = native.entry()

    input_values = {1: x, 2: weight, 3: bias, 4: target}
    buffers = {
        value_id: np.ascontiguousarray(input_values[value_id], dtype=np.float64)
        if value_id in input_values
        else np.zeros(shape or (), dtype=np.float64)
        for value_id, shape in zip(native.buffer_order, native.buffer_shapes)
    }
    pointers = (ctypes.c_void_p * len(native.buffer_order))(*(
        ctypes.c_void_p(buffers[value_id].ctypes.data)
        for value_id in native.buffer_order
    ))
    extents = (ctypes.c_int32 * len(native.extent_order))()

    entry(pointers, extents)
    np.testing.assert_allclose(
        buffers[lowering.outputs["loss_0"]], np.mean(diff * diff),
    )
    np.testing.assert_allclose(
        buffers[lowering.outputs["grad_1"]], grad_prediction @ weight.T,
    )
    np.testing.assert_allclose(
        buffers[lowering.outputs["grad_2"]], x.T @ grad_prediction,
    )
    np.testing.assert_allclose(
        buffers[lowering.outputs["grad_3"]], grad_prediction.sum(axis=0),
    )

    first_loss = float(buffers[lowering.outputs["loss_0"]])
    buffers[2][...] -= 0.1 * buffers[lowering.outputs["grad_2"]]
    buffers[3][...] -= 0.1 * buffers[lowering.outputs["grad_3"]]
    entry(pointers, extents)
    assert float(buffers[lowering.outputs["loss_0"]]) < first_loss


def test_real_abstract_nn_linear_loss_runs_process_graph_adjoint_natively(tmp_path):
    from src.common.tensors.abstract_nn import Linear, MSELoss
    from src.common.tensors.accelerator_backends.ssa_backend import (
        SSATensorOperations,
        SSATensorProgram,
    )
    from src.compiler.ssa_llvm_backend import (
        compile_artifact,
        emit_ssa_function_to_llvm,
        prepare_artifact_execution,
        with_native_sgd_loop,
    )

    program = SSATensorProgram("abstract_nn_linear_mse")
    inputs = SSATensorOperations.input(program, (2, 3))
    weight = SSATensorOperations.input(program, (3, 2))
    bias = SSATensorOperations.input(program, (1, 2))
    targets = SSATensorOperations.input(program, (2, 2))
    layer = Linear(3, 2, like=inputs, init="xavier")
    layer.W = weight
    layer.b = bias
    loss = MSELoss()(layer.forward(inputs), targets)

    forward = abstract_tensor_program_to_process_graph(
        loss,
        bindings={
            "x": inputs, "W": weight, "b": bias, "target": targets,
        },
    )
    assert [forward.G.nodes[node]["op"] for node in nx.topological_sort(
        forward.G
    )] == ["input", "input", "input", "input", "matmul", "add", "sub", "mul", "mean"]
    product = compile_process_graph_backward(
        forward, wrt=(0, 1, 2), packaging="combined",
    )
    assert product.motion is not None
    assert product.binding_graph is product.adjoint.binding_graph
    assert product.graph.G.graph["optimizer_included"] is False

    lowering = lower_training_motion_to_repository_ssa(
        product.motion, function_name="abstract_nn_linear_mse",
    )
    llvm = emit_ssa_function_to_llvm(
        lowering.module, lowering.function_name,
        entry_name=lowering.function_name,
    )
    assert llvm.shortfalls == ()
    loop = with_native_sgd_loop(
        llvm,
        parameter_gradient_pairs=(
            (1, lowering.outputs["grad_1"]),
            (2, lowering.outputs["grad_2"]),
        ),
    )
    native = compile_artifact(llvm, directory=tmp_path / "abstract_nn_native")

    x_value = np.asarray([[0.2, -0.3, 0.7], [1.1, 0.4, -0.2]])
    weight_value = np.asarray([[0.5, -0.1], [0.2, 0.8], [-0.4, 0.3]])
    bias_value = np.asarray([[0.05, -0.15]])
    target_value = np.asarray([[0.1, 0.9], [-0.2, 0.3]])
    inputs_by_id = {
        0: x_value, 1: weight_value, 2: bias_value, 3: target_value,
    }
    buffers = {
        value_id: np.ascontiguousarray(inputs_by_id[value_id], dtype=np.float64)
        if value_id in inputs_by_id else np.zeros(shape or (), dtype=np.float64)
        for value_id, shape in zip(native.buffer_order, native.buffer_shapes)
    }
    pointers = (ctypes.c_void_p * len(native.buffer_order))(*(
        ctypes.c_void_p(buffers[value_id].ctypes.data)
        for value_id in native.buffer_order
    ))
    extents = (ctypes.c_int32 * len(native.extent_order))()
    entry = native.entry()
    entry(pointers, extents)

    prediction = x_value @ weight_value + bias_value
    upstream = 2.0 * (prediction - target_value) / prediction.size
    np.testing.assert_allclose(
        buffers[lowering.outputs["loss_0"]],
        np.mean((prediction - target_value) ** 2),
    )
    np.testing.assert_allclose(
        buffers[lowering.outputs["grad_0"]], upstream @ weight_value.T,
    )
    np.testing.assert_allclose(
        buffers[lowering.outputs["grad_1"]], x_value.T @ upstream,
    )
    np.testing.assert_allclose(
        buffers[lowering.outputs["grad_2"]], upstream.sum(axis=0, keepdims=True),
    )

    first_loss = float(buffers[lowering.outputs["loss_0"]])
    buffers[1][...] -= 0.1 * buffers[lowering.outputs["grad_1"]]
    buffers[2][...] -= 0.1 * buffers[lowering.outputs["grad_2"]]
    entry(pointers, extents)
    assert float(buffers[lowering.outputs["loss_0"]]) < first_loss

    native_loop = compile_artifact(
        loop, directory=tmp_path / "abstract_nn_native_loop",
    )
    execution = prepare_artifact_execution(native_loop, {
        0: x_value,
        1: weight_value.copy(),
        2: bias_value.copy(),
        3: target_value,
        native_loop.training_steps_value_id: 8,
        native_loop.learning_rate_value_id: 0.1,
    })
    execution.run()
    assert float(execution.buffers[lowering.outputs["loss_0"]]) < first_loss
    assert not np.array_equal(execution.buffers[1], weight_value)
    assert not np.array_equal(execution.buffers[2], bias_value)


def test_real_rectconv2d_graph_adjoint_lowers_and_executes_natively(tmp_path):
    from src.common.tensors.abstract_nn import MSELoss, RectConv2d
    from src.common.tensors.accelerator_backends.ssa_backend import (
        SSATensorOperations,
        SSATensorProgram,
    )
    from src.compiler.ssa_llvm_backend import (
        compile_artifact,
        emit_ssa_function_to_llvm,
        prepare_artifact_execution,
    )

    program = SSATensorProgram("rectconv2d_graph_adjoint")
    x, weight, bias, target = [
        SSATensorOperations.input(program, shape)
        for shape in (
            (1, 1, 4, 4), (2, 1, 3, 3), (2,), (1, 2, 4, 4),
        )
    ]
    layer = RectConv2d(1, 2, 3, padding=1, like=x)
    layer.W, layer.b = weight, bias
    loss = MSELoss()(layer.forward(x), target)
    forward = abstract_tensor_program_to_process_graph(loss, bindings={
        "x": x, "weight": weight, "bias": bias, "target": target,
    })
    product = compile_process_graph_backward(
        forward, wrt=(1, 2), packaging="combined",
    )
    lowering = lower_training_motion_to_repository_ssa(
        product.motion, function_name="rectconv2d_graph_adjoint",
    )
    emitted = emit_ssa_function_to_llvm(
        lowering.module, lowering.function_name,
        entry_name=lowering.function_name,
    )
    assert lowering.shortfalls == ()
    assert emitted.shortfalls == ()
    artifact = compile_artifact(emitted, directory=tmp_path)

    x_value = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4) / 15.0
    weight_value = np.asarray([
        [[[.1, -.2, .05], [.3, .4, -.1], [.2, 0., -.3]]],
        [[[-.2, .1, .25], [.05, -.15, .35], [.4, -.05, .2]]],
    ])
    bias_value = np.asarray([.03, -.07])
    target_value = np.stack((
        x_value[:, 0] * .5 + .1,
        1.0 - x_value[:, 0] * .25,
    ), axis=1)
    execution = prepare_artifact_execution(artifact, {
        0: x_value, 1: weight_value, 2: bias_value, 3: target_value,
    }).run()

    padded = np.pad(x_value, ((0, 0), (0, 0), (1, 1), (1, 1)))
    prediction = np.empty_like(target_value)
    for channel in range(2):
        for row in range(4):
            for column in range(4):
                prediction[0, channel, row, column] = (
                    bias_value[channel]
                    + np.sum(
                        padded[0, :, row:row + 3, column:column + 3]
                        * weight_value[channel]
                    )
                )
    output_gradient = 2.0 * (prediction - target_value) / prediction.size
    expected_weight_gradient = np.zeros_like(weight_value)
    for channel in range(2):
        for kh in range(3):
            for kw in range(3):
                expected_weight_gradient[channel, 0, kh, kw] = np.sum(
                    output_gradient[0, channel]
                    * padded[0, 0, kh:kh + 4, kw:kw + 4]
                )
    expected_bias_gradient = output_gradient.sum(axis=(0, 2, 3))

    assert float(execution.buffers[lowering.outputs["loss_0"]]) == pytest.approx(
        float(np.mean((prediction - target_value) ** 2)), abs=1e-15,
    )
    np.testing.assert_allclose(
        execution.buffers[lowering.outputs["grad_1"]],
        expected_weight_gradient,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        execution.buffers[lowering.outputs["grad_2"]],
        expected_bias_gradient,
        rtol=1e-13,
        atol=1e-13,
    )


def test_real_abstract_nn_xor_has_exact_native_adjoint_and_training_loop(
    tmp_path,
):
    from src.common.tensors.abstract_nn import (
        Linear, Model, MSELoss, Sigmoid, Tanh,
    )
    from src.common.tensors.accelerator_backends.ssa_backend import (
        SSATensorOperations,
        SSATensorProgram,
    )
    from src.compiler.ssa_llvm_backend import (
        compile_artifact,
        emit_ssa_function_to_llvm,
        prepare_artifact_execution,
        with_native_sgd_loop,
    )

    program = SSATensorProgram("abstract_nn_xor_mse")
    shapes = ((4, 2), (2, 8), (1, 8), (8, 1), (1, 1), (4, 1))
    x, w1, b1, w2, b2, target = [
        SSATensorOperations.input(program, shape) for shape in shapes
    ]
    first = Linear(2, 8, like=x, init="xavier")
    first.W, first.b = w1, b1
    second = Linear(8, 1, like=x, init="xavier")
    second.W, second.b = w2, b2
    loss = MSELoss()(
        Model([first, second], [Tanh(), Sigmoid()]).forward(x), target,
    )
    forward = abstract_tensor_program_to_process_graph(loss, bindings={
        "x": x, "W1": w1, "b1": b1,
        "W2": w2, "b2": b2, "target": target,
    })
    product = compile_process_graph_backward(
        forward, wrt=(1, 2, 3, 4), packaging="combined",
    )
    assert product.motion is not None
    assert product.adjoint.backward.G.graph["python_backward_callbacks"] is False
    assert product.adjoint.backward.G.graph["backward_rule_registry"].endswith(
        "BACKWARD_RULES"
    )

    lowering = lower_training_motion_to_repository_ssa(
        product.motion, function_name="abstract_nn_xor_mse",
    )
    artifact = emit_ssa_function_to_llvm(
        lowering.module,
        lowering.function_name,
        entry_name=lowering.function_name,
    )
    assert lowering.shortfalls == ()
    assert artifact.shortfalls == ()
    assert artifact.buffer_order == (
        0, 1, 2, 3, 4, 5,
        lowering.outputs["loss_0"],
        lowering.outputs["grad_1"],
        lowering.outputs["grad_2"],
        lowering.outputs["grad_3"],
        lowering.outputs["grad_4"],
    )

    x_value = np.asarray([
        [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0],
    ])
    w1_value = np.asarray([
        [.15, -.2, .35, -.4, .25, .1, -.3, .45],
        [-.3, .4, .2, -.1, .5, -.35, .15, .25],
    ])
    b1_value = np.asarray([[.05, -.1, .08, .02, -.04, .06, .03, -.07]])
    w2_value = np.asarray([
        [.3], [-.25], [.4], [-.35], [.2], [.15], [-.45], [.5],
    ])
    b2_value = np.asarray([[.02]])
    target_value = np.asarray([[0.0], [1.0], [1.0], [0.0]])
    input_values = {
        0: x_value, 1: w1_value, 2: b1_value,
        3: w2_value, 4: b2_value, 5: target_value,
    }

    loop = with_native_sgd_loop(
        artifact,
        parameter_gradient_pairs=tuple(
            (parameter, lowering.outputs[f"grad_{parameter}"])
            for parameter in (1, 2, 3, 4)
        ),
        entry_name="abstract_nn_xor_training_loop",
    )
    native = compile_artifact(
        artifact, directory=tmp_path / "abstract_nn_xor_native",
    )
    execution = prepare_artifact_execution(native, input_values)
    execution.run()
    hidden = np.tanh(x_value @ w1_value + b1_value)
    prediction = 1.0 / (1.0 + np.exp(-(hidden @ w2_value + b2_value)))
    output_adjoint = (
        2.0 * (prediction - target_value) / target_value.size
        * prediction * (1.0 - prediction)
    )
    hidden_adjoint = (
        output_adjoint @ w2_value.T * (1.0 - hidden * hidden)
    )
    references = {
        "loss_0": np.mean((prediction - target_value) ** 2),
        "grad_1": x_value.T @ hidden_adjoint,
        "grad_2": hidden_adjoint.sum(axis=0, keepdims=True),
        "grad_3": hidden.T @ output_adjoint,
        "grad_4": output_adjoint.sum(axis=0, keepdims=True),
    }
    for name, reference in references.items():
        np.testing.assert_allclose(
            execution.buffers[lowering.outputs[name]],
            reference,
            rtol=1e-11,
            atol=1e-12,
        )

    native_loop = compile_artifact(
        loop, directory=tmp_path / "abstract_nn_xor_loop",
    )
    loop_execution = prepare_artifact_execution(native_loop, {
        **{key: value.copy() for key, value in input_values.items()},
        loop.training_steps_value_id: 80,
        loop.learning_rate_value_id: 0.5,
    })
    loop_execution.run()
    trained_hidden = np.tanh(
        x_value @ loop_execution.buffers[1] + loop_execution.buffers[2]
    )
    trained_prediction = 1.0 / (1.0 + np.exp(-(
        trained_hidden @ loop_execution.buffers[3]
        + loop_execution.buffers[4]
    )))
    trained_loss = np.mean((trained_prediction - target_value) ** 2)
    assert trained_loss < references["loss_0"]
    assert all(
        np.isfinite(loop_execution.buffers[value_id]).all()
        for value_id in (1, 2, 3, 4)
    )


def test_control_adjoint_preserves_branch_and_reverses_each_selected_arm():
    from src.compiler.control_source import (
        ConditionalBlock,
        ControlProgram,
        SequenceBlock,
        StatementBlock,
    )
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa
    from src.transmogrifier.ssa_registry import Handler

    forward = ControlProgram(
        root=SequenceBlock((
            StatementBlock(("__scheduled_region_0__",)),
            ConditionalBlock(
                predicate_value_id=50,
                body=StatementBlock((
                    "__scheduled_region_1__",
                    "__scheduled_region_2__",
                )),
                orelse=StatementBlock(("__scheduled_region_3__",)),
            ),
            StatementBlock(("__scheduled_region_4__",)),
        )),
        region_indices=(0, 1, 2, 3, 4),
    )
    result = differentiate_control_program(
        forward,
        forward_to_backward_regions={
            0: 10, 1: 11, 2: 12, 3: 13, 4: 14,
        },
    )
    root = result.backward.root
    assert isinstance(root, SequenceBlock)
    assert root.blocks[0].lines == ("__scheduled_region_14__",)
    branch = root.blocks[1]
    assert isinstance(branch, ConditionalBlock)
    assert branch.predicate_value_id == 50
    assert branch.body.lines == (
        "__scheduled_region_12__",
        "__scheduled_region_11__",
    )
    assert branch.orelse.lines == ("__scheduled_region_13__",)
    assert root.blocks[2].lines == ("__scheduled_region_10__",)

    region_callees = {index: f"backward_region_{index}" for index in range(10, 15)}
    function, shortfalls = lower_control_program_to_ssa(
        result.backward,
        region_callees=region_callees,
        region_signatures={index: ((), ()) for index in region_callees},
    )
    assert shortfalls == ()
    instructions = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
    ]
    assert sum(instruction.op == Handler.CondBr.value for instruction in instructions) == 1
    assert {"if_true", "if_false", "if_merge"} <= set(function.blocks)


def test_process_program_adjoint_partitions_backward_by_forward_region():
    from src.compiler.control_source import ControlProgram, SequenceBlock, StatementBlock

    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "input", label="x", shape=(2,))
    _add(graph, 2, "input", label="weight", shape=(2,))
    _add(graph, 3, "mul", (1, 2), shape=(2,))
    _add(graph, 4, "mean", (3,), shape=())
    graph.roots = [4]
    control = ControlProgram(
        SequenceBlock((
            StatementBlock(("__scheduled_region_0__",)),
            StatementBlock(("__scheduled_region_1__",)),
        )),
        region_indices=(0, 1),
    )

    result = differentiate_process_program(
        graph,
        control,
        region_nodes={0: (1, 2, 3), 1: (4,)},
        wrt=(1, 2),
    )
    assert result.control.forward_to_backward_regions == {0: 2, 1: 3}
    assert result.control.backward.root.blocks[0].lines == (
        "__scheduled_region_3__",
    )
    assert result.control.backward.root.blocks[1].lines == (
        "__scheduled_region_2__",
    )
    for region, node_ids in result.backward_region_nodes.items():
        source_regions = {
            0 if int((result.numeric.backward.G.nodes[node]["attributes"])["source_forward_id"]) in {1, 2, 3}
            else 1
            for node in node_ids
        }
        assert source_regions == ({0} if region == 2 else {1})


def test_program_adjoint_regions_preserve_source_graph_and_publish_boundaries():
    from src.compiler.control_source import ControlProgram, SequenceBlock, StatementBlock

    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "input", label="x", shape=(2,))
    _add(graph, 2, "input", label="weight", shape=(2,))
    _add(graph, 3, "mul", (1, 2), shape=(2,))
    _add(graph, 4, "mean", (3,), shape=())
    graph.roots = [4]
    control = ControlProgram(
        SequenceBlock((
            StatementBlock(("__scheduled_region_0__",)),
            StatementBlock(("__scheduled_region_1__",)),
        )),
        region_indices=(0, 1),
    )
    program = differentiate_process_program(
        graph,
        control,
        region_nodes={0: (1, 2, 3), 1: (4,)},
        wrt=(1, 2),
    )
    source_snapshot = {
        node_id: (
            graph.G.nodes[node_id]["op"],
            tuple(graph.G.nodes[node_id]["parents"]),
        )
        for node_id in graph.G
    }

    regions = isolate_process_program_adjoint_regions(program)

    assert {(region.phase, region.region_id) for region in regions} == {
        ("forward", 0), ("forward", 1), ("backward", 2), ("backward", 3),
    }
    forward_zero = next(
        region for region in regions
        if region.phase == "forward" and region.region_id == 0
    )
    forward_one = next(
        region for region in regions
        if region.phase == "forward" and region.region_id == 1
    )
    assert forward_zero.input_value_ids == (1, 2)
    assert forward_zero.output_value_ids == (3,)
    assert forward_one.input_value_ids == (3,)
    assert forward_one.output_value_ids == (4,)
    assert forward_one.graph.G.nodes[3]["op"] == "input"
    assert forward_one.graph.G.nodes[3]["attributes"]["source_value_id"] == 3
    assert all(
        region.graph.G.graph["fused_program_semantic_authority"] is False
        for region in regions
    )
    assert source_snapshot == {
        node_id: (
            graph.G.nodes[node_id]["op"],
            tuple(graph.G.nodes[node_id]["parents"]),
        )
        for node_id in graph.G
    }


def test_program_adjoint_ledger_includes_predicate_and_loop_history_by_default():
    from src.compiler.control_source import (
        ConditionalBlock,
        ControlProgram,
        LoopBlock,
        SequenceBlock,
        StatementBlock,
    )

    graph = ProcessGraph(materialize_memory=False)
    _add(graph, 1, "input", label="x", shape=(2,))
    _add(graph, 2, "input", label="weight", shape=(2,))
    _add(graph, 3, "mul", (1, 2), shape=(2,))
    _add(graph, 4, "mean", (3,), shape=())
    _add(graph, 5, "const", label="take_branch", shape=())
    _add(graph, 6, "const", label="trip_count", shape=())
    graph.roots = [4]
    control = ControlProgram(
        SequenceBlock((
            ConditionalBlock(
                predicate_value_id=5,
                body=StatementBlock(("__scheduled_region_0__",)),
                orelse=StatementBlock(("__scheduled_region_1__",)),
            ),
            LoopBlock(
                induction="i",
                start="0",
                stop="8",
                step="1",
                body=StatementBlock(("__scheduled_region_1__",)),
                recursion_region_id=9,
            ),
        )),
        region_indices=(0, 1),
    )

    result = differentiate_process_program(
        graph,
        control,
        region_nodes={0: (1, 2, 3, 5), 1: (4, 6)},
        outputs=(4,),
        wrt=(1, 2),
        loop_adjoint_contracts={9: LoopAdjointContract(6)},
    )

    ledger = result.binding_graph.graph
    assert result.binding_graph is not result.numeric.binding_graph
    assert ledger.nodes[5]["kind"] == "predicate"
    assert ledger.nodes[5]["storage"] == "resident"
    assert ledger.nodes[5]["control_consumers"] == (
        ("root.sequence[0]", "branch_selection"),
    )
    assert ledger.nodes[6]["kind"] == "loop_history"
    assert ledger.nodes[6]["backward_input_id"] is None
    assert ledger.nodes[6]["control_consumers"] == (
        ("root.sequence[1]", "reverse_trip_count"),
    )
    assert ledger.graph["semantic_authority"] == "ProcessGraph+ControlProgram"


def test_control_adjoint_rejects_branch_carried_state_without_merge_contract():
    from src.compiler.control_source import ConditionalBlock, ControlProgram, StatementBlock

    control = ControlProgram(ConditionalBlock(
        predicate_value_id=7,
        body=StatementBlock(("__scheduled_region_0__",)),
        carried_aliases=((10, 11, 9, 12),),
    ))
    with pytest.raises(ProcessGraphAutogradError, match="branch-carried"):
        differentiate_control_program(
            control,
            forward_to_backward_regions={0: 1},
        )


def test_conditional_carried_gradient_merge_requires_exact_contract():
    from src.compiler.control_source import ConditionalBlock, ControlProgram, StatementBlock

    forward_aliases = ((10, 11, 9, 12),)
    backward_aliases = ((110, 111, 112, 109),)
    control = ControlProgram(ConditionalBlock(
        predicate_value_id=7,
        body=StatementBlock(("__scheduled_region_0__",)),
        orelse=StatementBlock(("__scheduled_region_1__",)),
        carried_aliases=forward_aliases,
        source_node_id=42,
    ))
    result = differentiate_control_program(
        control,
        forward_to_backward_regions={0: 2, 1: 3},
        conditional_adjoint_contracts={
            42: ConditionalAdjointContract(
                forward_carried_aliases=forward_aliases,
                backward_carried_aliases=backward_aliases,
            ),
        },
    )
    assert result.backward.root.carried_aliases == backward_aliases


def test_counted_loop_adjoint_uses_saved_trip_count_and_descending_cfg():
    from src.compiler.control_source import ControlProgram, LoopBlock, StatementBlock
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa
    from src.transmogrifier.ssa_registry import Handler

    forward = ControlProgram(
        LoopBlock(
            induction="i",
            start="2",
            stop="11",
            step="3",
            body=StatementBlock(("__scheduled_region_0__",)),
            recursion_region_id=7,
        ),
        region_indices=(0,),
    )
    result = differentiate_control_program(
        forward,
        forward_to_backward_regions={0: 1},
        loop_adjoint_contracts={7: LoopAdjointContract(90)},
    )
    loop = result.backward.root
    assert isinstance(loop, LoopBlock)
    assert loop.start == "(2) + ((value_90 - 1) * (3))"
    assert loop.stop == "(2) - 1"
    assert loop.step == "-(3)"
    assert loop.comparison == "gt"
    assert result.backward.uniforms[0].value_id == 90

    function, shortfalls = lower_control_program_to_ssa(
        result.backward,
        region_callees={1: "backward_region_1"},
        region_signatures={1: ((), ())},
    )
    assert shortfalls == ()
    header_ops = [item.op for item in function.blocks["loop_header"].instrs]
    assert Handler.Gt.value in header_ops
    assert Handler.CondBr.value in header_ops


def test_real_control_adjoint_cfg_compiles_and_runs_natively(tmp_path):
    from src.compiler.control_source import (
        ConditionalBlock,
        ControlProgram,
        LoopBlock,
        SequenceBlock,
        StatementBlock,
    )
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa
    from src.compiler.ssa_llvm_backend import (
        compile_artifact,
        emit_ssa_function_to_llvm,
    )
    from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr

    forward = ControlProgram(
        SequenceBlock((
            ConditionalBlock(
                predicate_value_id=50,
                body=StatementBlock(("__scheduled_region_0__",)),
                orelse=StatementBlock(("__scheduled_region_1__",)),
            ),
            LoopBlock(
                induction="i",
                start="0",
                stop="6",
                step="1",
                body=StatementBlock(("__scheduled_region_2__",)),
                recursion_region_id=7,
            ),
        )),
        region_indices=(0, 1, 2),
    )
    adjoint = differentiate_control_program(
        forward,
        forward_to_backward_regions={0: 10, 1: 11, 2: 12},
        loop_adjoint_contracts={7: LoopAdjointContract(90)},
    )
    region_names = {index: f"backward_region_{index}" for index in (10, 11, 12)}
    control_function, shortfalls = lower_control_program_to_ssa(
        adjoint.backward,
        region_callees=region_names,
        region_signatures={index: ((), ()) for index in region_names},
    )
    assert shortfalls == ()
    region_functions = {
        name: Function(
            name, [], {"entry": BasicBlock("entry", [Instr("Ret", [], None)])},
        )
        for name in region_names.values()
    }
    module = IRModule({control_function.name: control_function, **region_functions})
    artifact = emit_ssa_function_to_llvm(
        module, control_function.name, entry_name="native_control_adjoint",
    )
    assert artifact.shortfalls == ()
    native = compile_artifact(artifact, directory=tmp_path / "native_control")
    values = {
        50: ctypes.c_bool(True),
        90: ctypes.c_int32(6),
    }
    buffers = (ctypes.c_void_p * len(native.buffer_order))(*(
        ctypes.cast(ctypes.pointer(values[value_id]), ctypes.c_void_p)
        for value_id in native.buffer_order
    ))
    extents = (ctypes.c_int32 * len(native.extent_order))()
    entry = native.entry()
    entry(buffers, extents)
    values[50].value = False
    values[90].value = 3
    entry(buffers, extents)
