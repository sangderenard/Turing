import numpy as np
import pytest
from src.common.tensors.numpy_backend import NumPyTensorOperations as T

@pytest.mark.parametrize(
    "op,expected",
    [
        ("sum", np.array([1.0, 1.0, 1.0])),
        ("mean", np.array([1.0 / 3] * 3)),
        ("prod", np.array([6.0, 3.0, 2.0])),
        ("max", np.array([0.0, 0.0, 1.0])),
        ("min", np.array([1.0, 0.0, 0.0])),
    ],
)
def test_reduction_gradients_no_scalar_leak(op, expected):
    x = T.tensor([1.0, 2.0, 3.0])
    x.requires_grad_(True)
    y = getattr(x, op)()
    assert isinstance(y.data, np.ndarray)
    assert y.numel() == 1
    y.backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.data, expected)
