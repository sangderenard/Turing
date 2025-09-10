import pytest
from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autograd import autograd
from src.common.tensors.autoautograd.fluxspring.fs_types import (
    NodeSpec,
    NodeCtrl,
    FluxSpringSpec,
    DECSpec,
)
from src.common.tensors.autoautograd.fluxspring.fs_harness import RingHarness


def test_ring_buffer_propagates_gradients():
    """Ring buffers managed by the harness should participate in autograd.

    Values pushed through :class:`RingHarness` are stored via tensor operations
    (`AT.scatter_row`) rather than Python list mutation. This keeps the autograd
    graph intact, so losses computed from the ring should backprop to the
    original parameter.
    """

    param = AT.tensor(1.0)
    param.requires_grad_(True)

    node = NodeSpec(
        id=0,
        p0=AT.get_tensor([0.0]),
        v0=AT.get_tensor([0.0]),
        mass=AT.tensor(1.0),
        ctrl=NodeCtrl(),
        scripted_axes=[0],
    )
    spec = FluxSpringSpec(
        version="t",
        D=1,
        nodes=[node],
        edges=[],
        faces=[],
        dec=DECSpec(D0=[], D1=[]),
    )
    harness = RingHarness(default_size=3)

    for i in range(6):
        harness.push_node(node.id, param * AT.tensor(float(i + 1)))

    rb = harness.get_node_ring(node.id)
    assert rb is not None
    loss = rb.buf.sum()
    grad = autograd.grad(loss, [param])[0]
    assert grad is not None
    assert float(AT.get_tensor(grad)) != 0.0
