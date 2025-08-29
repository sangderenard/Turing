import numpy as np
import pytest
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.autograd import autograd
from src.common.tensors.abstract_convolution.ndpca3transform import PCABasisND
from src.common.tensors.riemann.geometry_factory import build_geometry
from src.common.tensors.riemann.grid_block import (
    RiemannGridBlock,
    validate_config,
)


def _example_config():
    return {
        "geometry": {
            "key": "rect_euclidean",
            "grid_shape": (2, 2, 2),
            "boundary_conditions": (True,) * 6,
            "transform_args": {"Lx": 1.0, "Ly": 1.0, "Lz": 1.0},
            "laplace_kwargs": {},
        },
        "casting": {
            "mode": "pre_linear",
            "film": {"enabled": True},
            "coords": {"type": "raw", "dims": 3},
        },
        "conv": {
            "in_channels": 3,
            "out_channels": 4,
            "k": 1,
            "metric_source": "g",
            "boundary_conditions": ("dirichlet",) * 6,
            "pointwise": True,
        },
        "post_linear": {"in_dim": 4, "out_dim": 2},
    }


def _identity_basis():
    AT = AbstractTensor
    mu = AT.zeros(3)
    P = AT.eye(3)
    return PCABasisND(mu=mu, P=P, n=3)


def _phi(U, V, W):
    AT = AbstractTensor
    return AT.stack([U, V, W], dim=-1)


def _pca_demo_config():
    basis = _identity_basis()
    return {
        "geometry": {
            "key": "pca_nd",
            "grid_shape": (2, 2, 2),
            "boundary_conditions": (True,) * 6,
            "transform_args": {"pca_basis": basis, "phi_fn": _phi, "d_visible": 3},
            "laplace_kwargs": {},
        },
        "casting": {"mode": "pre_linear"},
        "conv": {
            "in_channels": 2,
            "out_channels": 3,
            "k": 1,
            "metric_source": "g",
            "boundary_conditions": ("dirichlet",) * 6,
            "pointwise": True,
        },
        "post_linear": {"in_dim": 3, "out_dim": 3},
    }


def test_forward_output_shape_pre_linear():
    cfg = _example_config()
    block = RiemannGridBlock.build_from_config(cfg)
    B = 1
    D, H, W = cfg["geometry"]["grid_shape"]
    x = AbstractTensor.randn((B, cfg["conv"]["in_channels"], D, H, W))
    y = block.forward(x)
    assert y.shape == (B, cfg["post_linear"]["out_dim"], D, H, W)


def test_parameters_registered_pre_linear():
    cfg = _example_config()
    block = RiemannGridBlock.build_from_config(cfg)
    params = block.parameters()
    expected = 8  # pre(2) + film(2) + conv(2) + post(2)
    assert len(params) == expected
    for p in params:
        assert isinstance(p, AbstractTensor)


def test_forward_end_to_end_pre_linear():
    cfg = _example_config()
    block = RiemannGridBlock.build_from_config(cfg)
    D, H, W = cfg["geometry"]["grid_shape"]
    x = AbstractTensor.ones((1, cfg["conv"]["in_channels"], D, H, W))
    y = block.forward(x)
    assert y.shape == (1, cfg["post_linear"]["out_dim"], D, H, W)


def test_forward_output_shape_fixed_mode():
    cfg = _example_config()
    cfg["casting"] = {"mode": "fixed"}
    block = RiemannGridBlock.build_from_config(cfg)
    D, H, W = cfg["geometry"]["grid_shape"]
    x = AbstractTensor.randn((1, cfg["conv"]["in_channels"], D, H, W))
    y = block.forward(x)
    assert y.shape == (1, cfg["post_linear"]["out_dim"], D, H, W)


def test_soft_assign_not_implemented():
    cfg = _example_config()
    cfg["casting"] = {"mode": "soft_assign"}
    block = RiemannGridBlock.build_from_config(cfg)
    D, H, W = cfg["geometry"]["grid_shape"]
    x = AbstractTensor.zeros((1, cfg["conv"]["in_channels"], D, H, W))
    with pytest.raises(NotImplementedError):
        block.forward(x)


def test_casting_row_major_deterministic():
    cfg = _example_config()
    cfg["casting"] = {
        "mode": "pre_linear",
        "film": {"enabled": False},
        "map": "row_major",
    }
    block = RiemannGridBlock.build_from_config(cfg)

    C = cfg["conv"]["in_channels"]
    D, H, W = cfg["geometry"]["grid_shape"]
    size = C * D * H * W

    eye = AbstractTensor.eye(size)
    AbstractTensor.copyto(block.casting.pre_linear.W, eye)
    AbstractTensor.copyto(block.casting.pre_linear.b, block.casting.pre_linear.b * 0)

    x = AbstractTensor.arange(size).reshape(1, C, D, H, W)
    y = block.casting.forward(x)

    assert np.allclose(y.data, x.data)


def test_casting_1to1_mapping():
    cfg = _example_config()
    cfg["casting"] = {
        "mode": "pre_linear",
        "film": {"enabled": False},
        "map": "1to1",
    }
    block = RiemannGridBlock.build_from_config(cfg)

    C = cfg["conv"]["in_channels"]
    D, H, W = cfg["geometry"]["grid_shape"]
    size = C * D * H * W

    eye = AbstractTensor.eye(size)
    AbstractTensor.copyto(block.casting.pre_linear.W, eye)
    AbstractTensor.copyto(block.casting.pre_linear.b, block.casting.pre_linear.b * 0)

    x = AbstractTensor.arange(size).reshape(1, C, D, H, W)
    y = block.casting.forward(x)

    flat = x.reshape(1, size)
    expected = flat.reshape(1, D, H, W, C)
    expected = expected.swapaxes(4, 3)
    expected = expected.swapaxes(3, 2)
    expected = expected.swapaxes(2, 1)

    assert np.allclose(y.data, expected.data)


def test_coordinate_fourier_embedding_dims():
    cfg = _example_config()
    cfg["casting"]["coords"] = {"type": "fourier", "dims": 6}
    cfg["casting"]["inject_coords"] = True
    block = RiemannGridBlock.build_from_config(cfg)
    assert block.casting.coords_as_channels is not None
    assert block.casting.coords_as_channels.shape[1] == 6


def test_validate_config_requires_channels():
    cfg = {
        "geometry": {"key": "rect_euclidean"},
        "conv": {"out_channels": 4},
    }
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_geometry_factory_pca_nd_demo_config():
    cfg = _pca_demo_config()["geometry"]
    _, grid, package = build_geometry(cfg)
    assert grid.U.shape == (2, 2, 2)
    assert isinstance(package, dict)


def test_forward_shape_and_grad_pca_nd():
    cfg = _pca_demo_config()
    block = RiemannGridBlock.build_from_config(cfg)
    B = 2
    D, H, W = cfg["geometry"]["grid_shape"]
    x = AbstractTensor.randn((B, cfg["conv"]["in_channels"], D, H, W), requires_grad=True)
    y = block.forward(x)
    assert y.shape == (B, cfg["conv"]["out_channels"], D, H, W)
    loss = y.sum()
    params = block.parameters()
    grads = autograd.grad(loss, [x] + params)
    assert grads[0].shape == x.shape
    assert all(g is not None for g in grads)


def test_parameter_counts_pca_nd():
    cfg = _pca_demo_config()
    block = RiemannGridBlock.build_from_config(cfg)
    D, H, W = cfg["geometry"]["grid_shape"]
    Cin = cfg["conv"]["in_channels"]
    Cout = cfg["conv"]["out_channels"]
    size = Cin * D * H * W

    pre_params = sum(p.numel() for p in block.casting.pre_linear.parameters())
    conv_params = sum(p.numel() for p in block.conv.parameters())
    post_params = sum(p.numel() for p in block.post_linear.parameters())

    assert pre_params == size * size + size
    expected_conv = cfg["conv"]["k"] * len(block.conv.offsets) + Cin * Cout
    assert conv_params == expected_conv
    expected_post = (
        cfg["post_linear"]["in_dim"] * cfg["post_linear"]["out_dim"]
        + cfg["post_linear"]["out_dim"]
    )
    assert post_params == expected_post
