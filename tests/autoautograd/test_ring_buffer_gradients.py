import pytest
from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autograd import autograd
from src.common.tensors.autoautograd.fluxspring.fs_types import NodeSpec, NodeCtrl


def test_ring_buffer_propagates_gradients():
    """Ring buffers on ``NodeSpec`` should participate in autograd.

    Values pushed through :meth:`NodeSpec.push_ring` are stored via tensor
    operations (`AT.scatter_row`) rather than Python list mutation. This keeps
    the autograd graph intact, so losses computed from the ring should backprop
    to the original parameter.
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
        ring_size=3,
    )

    for i in range(6):
        node.push_ring(param * AT.tensor(float(i + 1)))

    loss = node.ring.sum()
    grad = autograd.grad(loss, [param])[0]
    assert grad is not None
    assert float(AT.get_tensor(grad)) != 0.0
