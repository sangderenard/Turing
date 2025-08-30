import pytest

try:
    from src.common.tensors.numpy_backend import NumPyTensorOperations as Tensor
except Exception:  # pragma: no cover - optional dependency
    Tensor = None  # type: ignore

from src.common.tensors.autograd import autograd
from src.common.tensors.backward import bw_permute


@pytest.mark.skipif(Tensor is None, reason="NumPy backend not available")
def test_bw_permute_handles_axes():
    x = Tensor.arange(6).reshape(2, 3).astype("float32")
    x.requires_grad_(True)
    y = x.permute(1, 0).sum()
    autograd.grad(y, [x])
    assert x._grad.shape == x.shape
    assert x._grad.tolist() == [[1, 1, 1], [1, 1, 1]]

    g = Tensor.ones_like(x.permute(1, 0))
    gx = bw_permute(g, x, [1, 0])
    assert gx.tolist() == [[1, 1, 1], [1, 1, 1]]

