import importlib.util
import numpy as np
import random
import pytest

pytest.skip("RectConv3d.backward not implemented", allow_module_level=True)

from src.common.tensors.abstract_nn.core import RectConv3d

try:
    from src.common.tensors.numpy_backend import NumPyTensorOperations
except Exception:  # pragma: no cover - optional dependency
    NumPyTensorOperations = None

BACKENDS = []
if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_rectconv3d_backward_matches_numerical(backend_name, Backend):
    random.seed(0)
    np.random.seed(0)
    like = Backend.tensor([0.0])
    layer = RectConv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(2, 2, 2),
        like=like,
        bias=True,
    )
    x_data = np.random.randn(1, 1, 3, 3, 3)
    x = Backend.tensor(x_data.tolist())

    y = layer.forward(x)
    grad_out = y * 0 + 1
    dx = layer.backward(grad_out)

    gW_analytic = layer.gW.data.copy()
    gb_analytic = layer.gb.data.copy() if layer.b is not None else None
    dx_analytic = dx.data.copy()

    eps = 1e-5
    # Numerical gradient for weights
    num_gW = np.zeros_like(layer.W.data)
    for idx in np.ndindex(layer.W.data.shape):
        orig = layer.W.data[idx]
        layer.W.data[idx] = orig + eps
        y_pos = layer.forward(x).data.sum()
        layer.W.data[idx] = orig - eps
        y_neg = layer.forward(x).data.sum()
        layer.W.data[idx] = orig
        num_gW[idx] = (y_pos - y_neg) / (2 * eps)
    assert np.allclose(gW_analytic, num_gW, atol=1e-5)

    # Numerical gradient for bias
    if layer.b is not None:
        num_gb = np.zeros_like(layer.b.data)
        for idx in np.ndindex(layer.b.data.shape):
            orig = layer.b.data[idx]
            layer.b.data[idx] = orig + eps
            y_pos = layer.forward(x).data.sum()
            layer.b.data[idx] = orig - eps
            y_neg = layer.forward(x).data.sum()
            layer.b.data[idx] = orig
            num_gb[idx] = (y_pos - y_neg) / (2 * eps)
        assert np.allclose(gb_analytic, num_gb, atol=1e-5)

    # Numerical gradient for input
    num_dx = np.zeros_like(x.data)
    for idx in np.ndindex(x.data.shape):
        orig = x.data[idx]
        x.data[idx] = orig + eps
        y_pos = layer.forward(x).data.sum()
        x.data[idx] = orig - eps
        y_neg = layer.forward(x).data.sum()
        x.data[idx] = orig
        num_dx[idx] = (y_pos - y_neg) / (2 * eps)
    assert np.allclose(dx_analytic, num_dx, atol=1e-5)
