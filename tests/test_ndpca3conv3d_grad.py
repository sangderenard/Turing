import numpy as np
from src.common.tensors.numpy_backend import NumPyTensorOperations as T
from src.common.tensors.abstract_convolution.ndpca3conv import NDPCA3Conv3d
from src.common.tensors.abstraction import AbstractTensor

np.random.seed(0)

def _make_metric(D,H,W):
    g = np.tile(np.eye(3, dtype=np.float32), (D, H, W, 1, 1))
    return T.tensor(g.tolist())

def _finite_diff(layer, x, package, param, idx, eps=1e-5):
    orig = param[idx]
    param[idx] = orig + eps
    y = layer.forward(x, package=package)
    loss_plus = float(np.sum(y.data))
    param[idx] = orig - eps
    y = layer.forward(x, package=package)
    loss_minus = float(np.sum(y.data))
    param[idx] = orig
    return (loss_plus - loss_minus) / (2 * eps)


def _finite_diff_input(layer, x, package, eps=1e-5):
    g = np.zeros_like(x.data)
    it = np.ndindex(*x.data.shape)
    for idx in it:
        orig = x.data[idx]
        x.data[idx] = orig + eps
        y = layer.forward(x, package=package)
        loss_plus = float(np.sum(y.data))
        x.data[idx] = orig - eps
        y = layer.forward(x, package=package)
        loss_minus = float(np.sum(y.data))
        x.data[idx] = orig
        g[idx] = (loss_plus - loss_minus) / (2 * eps)
    return g


def _finite_diff_pointwise(layer, x, package, W, idx, eps=1e-5):
    orig = W[idx]
    W[idx] = orig + eps
    y = layer.forward(x, package=package)
    loss_plus = float(np.sum(y.data))
    W[idx] = orig - eps
    y = layer.forward(x, package=package)
    loss_minus = float(np.sum(y.data))
    W[idx] = orig
    return (loss_plus - loss_minus) / (2 * eps)


def test_ndpca3conv3d_gradients_no_pointwise():
    like = T.tensor([[0.0]])
    conv = NDPCA3Conv3d(1, 1, like=like, grid_shape=(2,2,2), pointwise=False)
    x = T.tensor(np.random.rand(1,1,2,2,2).tolist())
    x.requires_grad_(True)
    metric = _make_metric(2,2,2)
    package = {"metric": {"g": metric, "inv_g": metric}}
    conv.zero_grad()
    y = conv.forward(x, package=package)
    y.sum().backward()

    num_tap = _finite_diff(conv, x, package, conv.taps.data, (0,0))
    assert np.allclose(conv.taps.grad.data[0,0], num_tap, atol=3e-2)

    num_input = _finite_diff_input(conv, x, package)
    assert np.allclose(x.grad.data, num_input, atol=1e-2)


def test_ndpca3conv3d_gradients_with_pointwise():
    like = T.tensor([[0.0]])
    conv = NDPCA3Conv3d(1, 2, like=like, grid_shape=(2,2,2), pointwise=True)
    x = T.tensor(np.random.rand(1,1,2,2,2).tolist())
    x.requires_grad_(True)
    metric = _make_metric(2,2,2)
    package = {"metric": {"g": metric, "inv_g": metric}}
    conv.zero_grad()
    y = conv.forward(x, package=package)
    y.sum().backward()

    num_tap = _finite_diff(conv, x, package, conv.taps.data, (0,1))
    assert np.allclose(conv.taps.grad.data[0,1], num_tap, atol=1e-2)

    num_input = _finite_diff_input(conv, x, package)
    assert np.allclose(x.grad.data, num_input, atol=1e-2)

    W = conv.pointwise.W.data
    num_w = _finite_diff_pointwise(conv, x, package, W, (0,0))
    assert np.allclose(conv.pointwise.W.grad.data[0,0], num_w, atol=1e-2)


def test_ndpca3conv3d_zero_grad_no_pointwise():
    like = T.tensor([[0.0]])
    conv = NDPCA3Conv3d(1, 1, like=like, grid_shape=(2, 2, 2), pointwise=False)
    x = T.tensor(np.random.rand(1,1,2,2,2).tolist())
    x.requires_grad_(True)
    metric = _make_metric(2,2,2)
    package = {"metric": {"g": metric, "inv_g": metric}}
    y = conv.forward(x, package=package)
    y.sum().backward()
    assert conv.taps.grad is not None
    conv.zero_grad()
    assert conv.taps.grad is None


def test_ndpca3conv3d_zero_grad_with_pointwise():
    like = T.tensor([[0.0]])
    conv = NDPCA3Conv3d(1, 2, like=like, grid_shape=(2, 2, 2), pointwise=True)
    x = T.tensor(np.random.rand(1,1,2,2,2).tolist())
    x.requires_grad_(True)
    metric = _make_metric(2,2,2)
    package = {"metric": {"g": metric, "inv_g": metric}}
    y = conv.forward(x, package=package)
    y.sum().backward()
    assert conv.taps.grad is not None
    assert conv.pointwise.W.grad is not None
    conv.zero_grad()
    assert conv.taps.grad is None
    assert conv.pointwise.W.grad is None
