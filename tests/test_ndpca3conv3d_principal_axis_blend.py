import numpy as np
from src.common.tensors.abstract_convolution.ndpca3conv import NDPCA3Conv3d
from src.common.tensors.numpy_backend import NumPyTensorOperations as T


def _make_metric(D, H, W):
    g = np.tile(np.eye(3, dtype=np.float32), (D, H, W, 1, 1))
    return T.tensor_from_list(g.tolist())


def test_principal_axis_blend_weights_sum_to_k():
    like = T.tensor_from_list([[0.0]])
    layer = NDPCA3Conv3d(1, 1, like=like, grid_shape=(2, 2, 2), k=3, pointwise=False)
    metric = _make_metric(2, 2, 2)
    wU, wV, wW = layer._principal_axis_blend(metric)
    assert wU.shape == (2, 2, 2)
    total = (wU + wV + wW).numpy()
    assert np.allclose(total, np.full((2, 2, 2), layer.k), atol=1e-6)
