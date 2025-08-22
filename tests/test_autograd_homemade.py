
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
    # Assume backward is automatic and accumulates gradients
    z.backward()
    t1_grad = t1.grad
    t2_grad = t2.grad
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
    z.backward()
    t_grad = t.grad
    exp_grad = exp.grad
    t_np = np.array([2.0, 3.0])
    exp_np = np.array([3.0, 2.0])
    expected_dz_dt = exp_np * t_np ** (exp_np - 1)
    expected_dz_dexp = t_np ** exp_np * np.log(t_np)
    assert np.allclose(t_grad, expected_dz_dt)
    assert np.allclose(exp_grad, expected_dz_dexp)
