import numpy as np
from src.common.tensors.numpy_backend import NumPyTensorOperations as T
from src.common.tensors.abstract_convolution.ndpca3conv import NDPCA3Conv3d


def _make_metric(D, H, W):
    g = np.tile(np.eye(3, dtype=np.float32), (D, H, W, 1, 1))
    return T.tensor_from_list(g.tolist())


def _numpy_shift(arr, axis, step):
    out = np.zeros_like(arr)
    if step > 0:
        sl_src = [slice(None)] * arr.ndim
        sl_dst = [slice(None)] * arr.ndim
        sl_src[axis] = slice(0, -step)
        sl_dst[axis] = slice(step, None)
        out[tuple(sl_dst)] = arr[tuple(sl_src)]
    elif step < 0:
        step = -step
        sl_src = [slice(None)] * arr.ndim
        sl_dst = [slice(None)] * arr.ndim
        sl_src[axis] = slice(step, None)
        sl_dst[axis] = slice(0, -step)
        out[tuple(sl_dst)] = arr[tuple(sl_src)]
    return out


def _manual_conv(x, offsets, weights):
    y = np.zeros_like(x)
    for off, w in zip(offsets, weights):
        if off == 0:
            y += w * x
        else:
            y += w * (
                _numpy_shift(x, 2, off)
                + _numpy_shift(x, 3, off)
                + _numpy_shift(x, 4, off)
            )
    return y


def test_ndpca3conv3d_asymmetric_stencil():
    like = T.tensor_from_list([[0.0]])
    offsets = (-2, -1, 0, 1, 2)
    conv = NDPCA3Conv3d(
        1,
        1,
        like=like,
        grid_shape=(5, 5, 5),
        pointwise=False,
        stencil_offsets=offsets,
        stencil_length=2,
    )
    weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    for i in range(conv.k):
        conv.taps.data[i] = np.array(weights) / conv.k
    x_np = np.arange(1 * 1 * 5 * 5 * 5, dtype=np.float32).reshape(1, 1, 5, 5, 5)
    x = T.tensor_from_list(x_np.tolist())
    metric = _make_metric(5, 5, 5)
    package = {"metric": {"g": metric, "inv_g": metric}}
    y = conv.forward(x, package=package)
    ref = _manual_conv(x_np, offsets, weights)
    assert np.allclose(y.data, ref, atol=1e-5)


def test_ndpca3conv3d_normalizes_taps():
    like = T.tensor_from_list([[0.0]])
    conv = NDPCA3Conv3d(
        1,
        1,
        like=like,
        grid_shape=(3, 3, 3),
        pointwise=False,
        k=1,
        stencil_offsets=(-1, 0, 1),
        stencil_length=1,
        normalize_taps=True,
    )
    conv.taps.data[0] = np.array([1.0, 1.0, 1.0])
    x_np = np.ones((1, 1, 3, 3, 3), dtype=np.float32)
    x = T.tensor_from_list(x_np.tolist())
    metric = _make_metric(3, 3, 3)
    package = {"metric": {"g": metric, "inv_g": metric}}
    y = conv.forward(x, package=package)
    assert np.allclose(y.data, x_np, atol=1e-5)
