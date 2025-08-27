import pytest

try:
    from src.common.tensors.numpy_backend import NumPyTensorOperations as Tensor
except Exception:  # pragma: no cover - optional dependency
    Tensor = None  # type: ignore

from src.common.tensors.autograd import autograd


@pytest.mark.skipif(Tensor is None, reason="NumPy backend not available")
def test_bw_reshape_handles_negative_one():
    x = Tensor.arange(6).reshape(2, 3).astype("float32")
    x.requires_grad_(True)
    y = x.reshape(3, -1).sum()
    autograd.grad(y, [x])
    assert x._grad.shape == x.shape
    assert x._grad.tolist() == [[1, 1, 1], [1, 1, 1]]
