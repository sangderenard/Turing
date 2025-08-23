
import pytest
import numpy as np
from src.common.tensors import AbstractTensor

@pytest.mark.parametrize("a, b", [([2.0, 3.0], [4.0, 5.0]), ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]])])
def test_autograd_add_and_mul(a, b):
    t1 = AbstractTensor.tensor(a)
    t2 = AbstractTensor.tensor(b)
    t1.requires_grad = True
    t2.requires_grad = True
    z = t1 * t2 + t2
    t1_grad, t2_grad = AbstractTensor.autograd.grad(z, [t1, t2])
    t1_np = np.array(a)
    t2_np = np.array(b)
    expected_dz_dt1 = t2_np
    expected_dz_dt2 = t1_np + 1
    assert np.allclose(t1_grad, expected_dz_dt1)
    assert np.allclose(t2_grad, expected_dz_dt2)

def test_autograd_pow():
    t = AbstractTensor.tensor([2.0, 3.0])
    exp = AbstractTensor.tensor([3.0, 2.0])
    t.requires_grad = True
    exp.requires_grad = True
    z = t ** exp
    t_grad, exp_grad = AbstractTensor.autograd.grad(z, [t, exp])
    t_np = np.array([2.0, 3.0])
    exp_np = np.array([3.0, 2.0])
    expected_dz_dt = exp_np * t_np ** (exp_np - 1)
    expected_dz_dexp = t_np ** exp_np * np.log(t_np)
    assert np.allclose(t_grad, expected_dz_dt)
    assert np.allclose(exp_grad, expected_dz_dexp)


def test_autograd_records_only_for_grad_inputs():
    autograd = AbstractTensor.autograd
    autograd.tape._nodes.clear()

    a = AbstractTensor.tensor([1.0, 2.0])
    b = AbstractTensor.tensor([3.0, 4.0])

    _ = a + b
    assert not autograd.tape._nodes

    autograd.tape._nodes.clear()
    a.requires_grad = True
    z = a + b

    node = autograd.tape.node(z)
    assert node is not None
    assert {pid for pid, _ in node.parents} == {id(a), id(b)}


def test_autograd_single_tensor_input():
    autograd = AbstractTensor.autograd
    autograd.tape._nodes.clear()

    x = AbstractTensor.tensor([1.0, 2.0, 3.0])
    x.requires_grad = True
    y = x * 2.0

    grad_x = autograd.grad(y, x)[0]
    assert np.allclose(grad_x, np.array([2.0, 2.0, 2.0]))


def test_autograd_complex_sequence():
    autograd = AbstractTensor.autograd
    autograd.tape._nodes.clear()

    x = AbstractTensor.tensor([1.0, 2.0, 3.0])
    y = AbstractTensor.tensor([4.0, 5.0, 6.0])
    z = AbstractTensor.tensor([7.0, 8.0, 9.0])
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True

    w = AbstractTensor.sin(x * y + z)**2 - AbstractTensor.cos(x + y * z)
    grad_w = autograd.grad(w, [x, y, z])

    # PyTorch reference computation for parity validation
    try:
        import torch  # type: ignore
    except Exception:
        pytest.skip("torch not available")
    x_t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y_t = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    z_t = torch.tensor([7.0, 8.0, 9.0], requires_grad=True)
    w_t = torch.sin(x_t * y_t + z_t) ** 2 - torch.cos(x_t + y_t * z_t)
    w_t_sum = w_t.sum()  # To get gradients for all elements
    w_t_sum.backward()
    # Compare forward values
    np.testing.assert_allclose(w, w_t.detach().numpy(), rtol=1e-5, atol=1e-7)
    # Compare gradients
    np.testing.assert_allclose(grad_w[0], x_t.grad.numpy(), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(grad_w[1], y_t.grad.numpy(), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(grad_w[2], z_t.grad.numpy(), rtol=1e-5, atol=1e-7)

if __name__ == "__main__":
    test_autograd_complex_sequence()