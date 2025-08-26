import networkx as nx
import numpy as np
from src.common.tensors import AbstractTensor
from src.common.tensors.autograd_process import AutogradProcess


def _is_topological(graph, order):
    pos = {n: i for i, n in enumerate(order)}
    return all(pos[u] < pos[v] for u, v in graph.edges())


def test_autograd_process_training_and_tables():
    autograd = AbstractTensor.autograd
    tape = autograd.tape
    tape._nodes.clear()
    tape.graph.clear()

    x = AbstractTensor.tensor([1.0, 2.0, 3.0])
    y = AbstractTensor.tensor([2.0, 4.0, 6.0])
    w = AbstractTensor.tensor([0.0])
    w.requires_grad = True

    def forward_fn():
        pred = x * w
        err = pred - y
        sq = err * err
        loss_val = sq.sum().item()
        return sq, loss_val

    proc = AutogradProcess(tape)
    proc.training_loop(forward_fn, [w], steps=5, lr=0.01)

    # training loss should decrease
    assert proc.training_log[0]["loss"] > proc.training_log[-1]["loss"]
    # parameter should move toward 2
    assert np.allclose(w.data, np.array([2.0]), atol=0.5)

    # graphs and schedules populated
    assert isinstance(proc.forward_graph, nx.DiGraph)
    assert isinstance(proc.backward_graph, nx.DiGraph)

    assert _is_topological(proc.forward_graph, proc.forward_schedule)
    assert _is_topological(proc.backward_graph, proc.backward_schedule)

    tables = proc.summary_table()
    graph_df = tables["graph"]
    train_df = tables["training"]
    # table should list all nodes with forward/backward order
    assert {"id", "op", "forward_order", "backward_order", "cached", "stage", "param_id", "loss"} <= set(graph_df.columns)
    assert len(train_df) == 5

    tree = proc.process_tree()
    # ensure stages appear in tree and siblings share label
    assert tree.has_node("forward") and tree.has_node("backward")
    forward_children = list(tree.successors("forward"))
    backward_children = list(tree.successors("backward"))
    assert set(forward_children) == set(proc.forward_schedule)
    assert set(backward_children) == set(proc.backward_schedule)


def test_autograd_process_concurrent_levels():
    autograd = AbstractTensor.autograd
    tape = autograd.tape
    tape._nodes.clear()
    tape.graph.clear()

    x = AbstractTensor.tensor([1.0, 2.0])
    y = AbstractTensor.tensor([3.0, 4.0])
    w = AbstractTensor.tensor([1.0])
    w.requires_grad = True

    # two independent branches that can run concurrently
    a = x * w
    b = y * w
    c = a + b
    loss = c.sum()

    proc = AutogradProcess(tape)
    proc.build(loss)

    # schedules should still respect dependencies
    assert _is_topological(proc.forward_graph, proc.forward_schedule)

    # graph should have multiple levels with an intermediate concurrent level
    layer_map = {}
    for nid, data in proc.forward_graph.nodes(data=True):
        layer_map.setdefault(data["layer"], []).append(nid)
    assert len(layer_map) > 2
    layers = list(layer_map.values())
    assert any(len(level) > 1 for level in layers[1:-1])

    tape._nodes.clear()
    tape.graph.clear()
