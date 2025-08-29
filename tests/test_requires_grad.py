from src.common.tensors.numpy_backend import NumPyTensorOperations as T
from src.common.tensors.abstract_nn.core import Linear
from src.common.tensors.abstract_convolution.ndpca3conv import NDPCA3Conv3d

def test_linear_parameters_require_grad():
    like = T.tensor([[0.0]])
    layer = Linear(2, 3, like=like)
    assert layer.W.requires_grad
    assert layer.b is not None and layer.b.requires_grad

def test_ndpca3conv_pointwise_linear_requires_grad():
    like = T.tensor([[0.0]])
    conv = NDPCA3Conv3d(1, 2, like=like, grid_shape=(1, 1, 1), pointwise=True)
    assert conv.taps.requires_grad
    assert conv.pointwise is not None
    assert conv.pointwise.W.requires_grad
