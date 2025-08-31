import numpy as np
from src.common.tensors.numpy_backend import NumPyTensorOperations as T
from src.common.tensors.abstract_convolution.ndpca3conv import NDPCA3Conv3d

def test_ndpca3conv3d_taps_grad_not_none():
    like = T.tensor([[0.0]])
    conv = NDPCA3Conv3d(1, 1, like=like, grid_shape=(2, 2, 2), pointwise=False)
    x = T.tensor(np.random.rand(1, 1, 2, 2, 2).tolist())
    x.requires_grad_(True)
    g = np.tile(np.eye(3, dtype=np.float32), (2, 2, 2, 1, 1))
    metric = T.tensor(g.tolist())
    package = {"metric": {"g": metric, "inv_g": metric}}
    y = conv.forward(x, package=package)
    y.sum().backward()
    assert conv.taps.grad is not None
    assert conv.taps.grad.data.shape == conv.taps.data.shape
    assert np.all(np.abs(conv.taps.grad.data) > 0)
