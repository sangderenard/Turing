import networkx as nx
import numpy as np
from src.common.tensors import AbstractTensor
from src.common.tensors.autograd_process import AutogradProcess


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
    assert proc.forward_schedule == list(nx.topological_sort(proc.forward_graph))
    assert proc.backward_schedule == list(nx.topological_sort(proc.backward_graph))

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

    # schedules should match a topological sort of the forward graph
    assert proc.forward_schedule == list(nx.topological_sort(proc.forward_graph))

    levels = list(nx.topological_generations(proc.forward_graph))
    # graph should have multiple levels with an intermediate concurrent level
    assert len(levels) > 2
    assert any(len(level) > 1 for level in levels[1:-1])

    tape._nodes.clear()
    tape.graph.clear()
