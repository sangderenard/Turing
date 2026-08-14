from __future__ import annotations

import ctypes
import re

import numpy as np
import pytest

from src.compiler.process_graph_autograd import (
    ProcessGraphAutogradError,
    differentiate_process_graph,
    fuse_forward_loss_backward,
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
    # Only the multiplication operands require saved numeric values. Equal
    # static shapes let the registry's unbroadcast helper reduce to identity.
    assert set(adjoint.saved_value_ids) == {1, 2}
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


def _run_backward(adjoint, feeds):
    values = dict(feeds)
    graph = adjoint.backward.G
    for node_id in list(__import__("networkx").topological_sort(graph)):
        if node_id in values:
            continue
        data = graph.nodes[node_id]
        op = data["op"]
        parents = [values[parent] for parent, _role in data["parents"]]
        attrs = data.get("attributes") or {}
        if op == "const":
            values[node_id] = np.asarray(data["constant"])
        elif op == "add":
            values[node_id] = parents[0] + parents[1]
        elif op == "sub":
            values[node_id] = parents[0] - parents[1]
        elif op == "mul":
            values[node_id] = parents[0] * parents[1]
        elif op == "truediv":
            values[node_id] = parents[0] / parents[1]
        elif op == "neg":
            values[node_id] = -parents[0]
        elif op == "matmul":
            values[node_id] = parents[0] @ parents[1]
        elif op == "transpose":
            values[node_id] = np.swapaxes(parents[0], -1, -2)
        elif op == "sum":
            values[node_id] = parents[0].sum(
                axis=attrs.get("dim", attrs.get("axis")),
                keepdims=bool(attrs.get("keepdim", False)),
            )
        elif op == "reshape":
            values[node_id] = parents[0].reshape(tuple(attrs["shape"]))
        else:  # pragma: no cover - test evaluator stays deliberately narrow
            raise AssertionError(op)
    return [values[root] for root in adjoint.backward.roots]


def test_mse_numeric_subgraph_derives_parametric_gradients_without_tape():
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
    pred = np.asarray([[0.2, 0.7], [1.1, -0.4]])
    target = np.asarray([[0.0, 1.0], [0.3, -0.2]])
    diff = pred - target
    saved_forward = {1: pred, 2: target, 3: diff, 4: diff * diff}
    feeds = {
        adjoint.seed_value_ids[5]: np.asarray(1.0),
        **{
            adjoint.saved_value_ids[source]: value
            for source, value in saved_forward.items()
            if source in adjoint.saved_value_ids
        },
    }

    grad_pred, grad_target = _run_backward(adjoint, feeds)
    expected = 2.0 * diff / diff.size
    np.testing.assert_allclose(grad_pred, expected)
    np.testing.assert_allclose(grad_target, -expected)
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
    x = np.asarray([[0.2, -0.3, 0.7], [1.1, 0.4, -0.2]])
    weight = np.asarray([[0.5, -0.1], [0.2, 0.8], [-0.4, 0.3]])
    bias = np.asarray([0.05, -0.15])
    target = np.asarray([[0.1, 0.9], [-0.2, 0.3]])
    prediction = x @ weight + bias
    diff = prediction - target
    saved_forward = {
        1: x, 2: weight, 3: bias, 4: target,
        5: x @ weight, 6: prediction, 7: diff, 8: diff * diff,
    }
    feeds = {
        adjoint.seed_value_ids[9]: np.asarray(1.0),
        **{
            adjoint.saved_value_ids[source]: value
            for source, value in saved_forward.items()
            if source in adjoint.saved_value_ids
        },
    }

    grad_x, grad_weight, grad_bias = _run_backward(adjoint, feeds)
    grad_prediction = 2.0 * diff / diff.size
    np.testing.assert_allclose(grad_x, grad_prediction @ weight.T)
    np.testing.assert_allclose(grad_weight, x.T @ grad_prediction)
    np.testing.assert_allclose(grad_bias, grad_prediction.sum(axis=0))
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
