from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_convolution.ndpca3conv import NDPCA3Conv3d
from src.common.tensors.riemann.grid_block import RiemannGridBlock


def _basic_block(**reg):
    like = AbstractTensor.get_tensor([0.0])
    conv = NDPCA3Conv3d(1, 1, like=like, grid_shape=(2, 2, 2))
    return RiemannGridBlock(conv=conv, package={}, bin_map=None, post_linear=None, regularization=reg)


def test_smooth_bins_penalty_changes():
    like = AbstractTensor.get_tensor([0.0])
    conv = NDPCA3Conv3d(1, 1, like=like, grid_shape=(2, 2, 2))
    bin_map = AbstractTensor.zeros((1, 2, 2, 2))
    block = RiemannGridBlock(
        conv=conv,
        package={},
        bin_map=bin_map,
        post_linear=None,
        regularization={"smooth_bins": 0.5},
    )
    loss0 = float(block.regularization_loss().data)
    block.bin_map.data[0, 0, 0, 0] = 1.0
    loss1 = float(block.regularization_loss().data)
    assert loss1 > loss0


def test_weight_decay_penalty_changes():
    block = _basic_block(weight_decay={"conv": 0.1})
    block.conv.taps.data[:] = 0.0
    loss0 = float(block.regularization_loss().data)
    block.conv.taps.data[0, 0] = 1.0
    loss1 = float(block.regularization_loss().data)
    assert loss1 > loss0
