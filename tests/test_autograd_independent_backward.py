import numpy as np
from src.common.tensors import AbstractTensor


def test_independent_backward_per_tensor():
    autograd = AbstractTensor.autograd
    autograd.tape._nodes.clear()

    a = AbstractTensor.tensor([2.0, 3.0])
    b = AbstractTensor.tensor([4.0, 5.0])
    a.requires_grad = True
    b.requires_grad = True

    c = a * b
    c.requires_grad = True
    d = c + a
    e = d / b
    e.requires_grad = True
    f = e ** 2

    g = c - b
    g.requires_grad = True
    h = g + a
    i = g * e

    # Each result should have its own graph node and tape reference
    assert autograd.tape.node(f) is not None
    assert autograd.tape.node(h) is not None
    assert autograd.tape.node(i) is not None
    assert getattr(f, "_grad_tape") is autograd.tape
    assert getattr(h, "_grad_tape") is autograd.tape
    assert getattr(i, "_grad_tape") is autograd.tape

    a_grad_f, b_grad_f = autograd.grad(f, [a, b], retain_graph=True)
    a_grad_h, b_grad_h = autograd.grad(h, [a, b], retain_graph=True)
    a_grad_i, b_grad_i = autograd.grad(i, [a, b])

    a_np = np.array([2.0, 3.0])
    b_np = np.array([4.0, 5.0])
    expected_f_a = 2 * a_np * (1 + 1 / b_np) ** 2
    expected_f_b = -2 * a_np ** 2 * (1 + 1 / b_np) / (b_np ** 2)
    expected_h_a = b_np + 1
    expected_h_b = a_np - 1
    expected_i_a = (2 * a_np - 1) * (b_np + 1)
    expected_i_b = a_np ** 2 - a_np

    assert np.allclose(a_grad_f, expected_f_a)
    assert np.allclose(b_grad_f, expected_f_b)
    assert np.allclose(a_grad_h, expected_h_a)
    assert np.allclose(b_grad_h, expected_h_b)
    assert np.allclose(a_grad_i, expected_i_a)
    assert np.allclose(b_grad_i, expected_i_b)
