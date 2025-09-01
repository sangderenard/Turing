from src.common.tensors.abstract_nn.transform_layers import Transform2DLayer, Transform3DLayer
from src.common.tensors.abstraction import AbstractTensor as AT


def test_transform2d_layer_expands_to_5d():
    x = AT.randn((2, 4, 5))
    layer = Transform2DLayer()
    y = layer.forward(x)
    assert y.shape == (2, 1, 1, 4, 5)


def test_transform3d_layer_adds_channel():
    x = AT.randn((2, 3, 4, 5))
    layer = Transform3DLayer()
    y = layer.forward(x)
    assert y.shape == (2, 1, 3, 4, 5)
