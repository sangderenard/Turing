import numpy as np
from src.common.tensors import AbstractTensor
from src.common.tensors.autograd_process import AutogradProcess


def test_forward_graph_tracks_dependencies():
    autograd = AbstractTensor.autograd
    autograd.tape._nodes.clear()

    a = AbstractTensor.tensor([1.0])
    b = AbstractTensor.tensor([2.0])
    c = AbstractTensor.tensor([3.0])
    d = AbstractTensor.tensor([4.0])

    for t in (a, b, c, d):
        t.requires_grad = True

    x = a * b
    y = c + d
    loss = x + y

    autograd.tape.mark_loss(loss)
    proc = AutogradProcess(autograd.tape)
    proc.build(loss)

    edges = set(proc.forward_graph.edges())
    assert (id(a), id(x)) in edges
    assert (id(b), id(x)) in edges
    assert (id(c), id(y)) in edges
    assert (id(d), id(y)) in edges
    assert (id(x), id(loss)) in edges
    assert (id(y), id(loss)) in edges

    lx = proc.forward_graph.nodes[id(x)]["level"]
    ly = proc.forward_graph.nodes[id(y)]["level"]
    assert lx == ly
