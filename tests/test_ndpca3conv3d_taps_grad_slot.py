import numpy as np
from src.common.tensors.numpy_backend import NumPyTensorOperations as T
from src.common.tensors.abstract_convolution.ndpca3conv import NDPCA3Conv3d

def test_ndpca3conv3d_taps_grad_slot():
    like = T.tensor([[0.0]])
    layer = NDPCA3Conv3d(1, 1, like=like, grid_shape=(1, 1, 1), pointwise=False)
    x = T.tensor(np.random.rand(1, 1, 1, 1, 1).tolist())
    x.requires_grad_(True)
    g = np.eye(3, dtype=np.float32)[None, None, None, :, :]
    metric = T.tensor(g.tolist())
    package = {"metric": {"g": metric, "inv_g": metric}}
    y = layer.forward(x, package=package)
    y.sum().backward()
    assert layer.taps.grad is not None
