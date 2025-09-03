from __future__ import annotations

import pytest

pytest.skip("AbstractTensor.unravel_index_ not implemented", allow_module_level=True)

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_convolution.ndpca3transform import PCABasisND
from src.common.tensors.riemann.geometry_factory import build_geometry


def _identity_basis():
    AT = AbstractTensor
    mu = AT.zeros(3)
    P = AT.eye(3)
    return PCABasisND(mu=mu, P=P, n=3)


def _phi(U, V, W):
    AT = AbstractTensor
    return AT.stack([U, V, W], dim=-1)


def _metric(U, V, W, *_, **__):
    AT = AbstractTensor
    g = AT.eye(3).reshape(1, 1, 1, 3, 3).expand(U.shape + (3, 3))
    g_inv = g
    det_g = AT.ones(U.shape)
    return g, g_inv, det_g


def test_rect_euclidean_builds_package():
    cfg = {
        "key": "rect_euclidean",
        "grid_shape": (2, 2, 2),
        "boundary_conditions": (True,) * 6,
        "transform_args": {"Lx": 1.0, "Ly": 1.0, "Lz": 1.0},
        "laplace_kwargs": {},
    }
    transform, grid, package = build_geometry(cfg)
    assert hasattr(transform, "metric_tensor_func")
    assert grid.U.shape == (2, 2, 2)
    assert isinstance(package, dict)


def test_pca_nd_builds_package():
    basis = _identity_basis()
    cfg = {
        "key": "pca_nd",
        "grid_shape": (2, 2, 2),
        "boundary_conditions": (True,) * 6,
        "transform_args": {"pca_basis": basis, "phi_fn": _phi, "d_visible": 3},
        "laplace_kwargs": {},
    }
    _, grid, package = build_geometry(cfg)
    assert grid.U.shape == (2, 2, 2)
    assert isinstance(package, dict)


def test_custom_metric_builds_package():
    cfg = {
        "key": "custom_metric",
        "grid_shape": (2, 2, 2),
        "boundary_conditions": (True,) * 6,
        "transform_args": {
            "Lx": 1.0,
            "Ly": 1.0,
            "Lz": 1.0,
            "metric_tensor_func": _metric,
        },
        "laplace_kwargs": {},
    }
    transform, grid, package = build_geometry(cfg)
    assert transform.metric_tensor_func is _metric
    assert grid.U.shape == (2, 2, 2)
    assert isinstance(package, dict)


def test_unknown_key_raises():
    with pytest.raises(ValueError):
        build_geometry({"key": "does_not_exist"})
