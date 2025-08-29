from __future__ import annotations

"""Factories for common Riemannian geometry packages.

Each builder constructs a :class:`Transform` and matching :class:`GridDomain`
then wraps them in :class:`ManifoldPackage`.  The registry maps configuration
keys to these builder functions so callers can obtain geometry packages from a
simple config dictionary.
"""

from typing import Any, Callable, Dict, Tuple

from ..abstraction import AbstractTensor
from ..abstract_convolution.laplace_nd import GridDomain
from ..abstract_convolution.ndpca3transform import PCANDTransform
from .manifold import ManifoldPackage

# Type alias for builder functions
GeometryBuilder = Callable[[Dict[str, Any]], Tuple[Any, GridDomain, Dict[str, Any]]]


def _build_rect_euclidean(config: Dict[str, Any]):
    grid_shape = config.get("grid_shape", (1, 1, 1))
    boundary_conditions = config.get("boundary_conditions", (True,) * 6)
    transform_args = dict(config.get("transform_args", {}))
    laplace_kwargs = config.get("laplace_kwargs", {})
    Nu, Nv, Nw = grid_shape

    grid = GridDomain.generate_grid_domain(
        "rectangular", N_u=Nu, N_v=Nv, N_w=Nw, **transform_args
    )
    transform = grid.transform
    transform.grid_boundaries = boundary_conditions
    grid.grid_boundaries = boundary_conditions

    manifold = ManifoldPackage(transform, grid, laplace_kwargs=laplace_kwargs)
    package = manifold.laplace_package()
    return transform, grid, package


def _build_pca_nd(config: Dict[str, Any]):
    grid_shape = config.get("grid_shape", (1, 1, 1))
    boundary_conditions = config.get("boundary_conditions", (True,) * 6)
    transform_args = dict(config.get("transform_args", {}))
    laplace_kwargs = config.get("laplace_kwargs", {})
    Nu, Nv, Nw = grid_shape

    transform = PCANDTransform(**transform_args)
    AT = AbstractTensor
    U = AT.linspace(-1.0, 1.0, Nu).reshape(Nu, 1, 1) * AT.ones((1, Nv, Nw))
    V = AT.linspace(-1.0, 1.0, Nv).reshape(1, Nv, 1) * AT.ones((Nu, 1, Nw))
    W = AT.linspace(-1.0, 1.0, Nw).reshape(1, 1, Nw) * AT.ones((Nu, Nv, 1))
    grid = GridDomain(
        U,
        V,
        W,
        grid_boundaries=boundary_conditions,
        transform=transform,
        coordinate_system="rectangular",
    )
    manifold = ManifoldPackage(transform, grid, laplace_kwargs=laplace_kwargs)
    package = manifold.laplace_package()
    return transform, grid, package


def _build_custom_metric(config: Dict[str, Any]):
    grid_shape = config.get("grid_shape", (1, 1, 1))
    boundary_conditions = config.get("boundary_conditions", (True,) * 6)
    transform_args = dict(config.get("transform_args", {}))
    laplace_kwargs = config.get("laplace_kwargs", {})
    Nu, Nv, Nw = grid_shape

    metric_fn = transform_args.pop("metric_tensor_func", None)
    grid = GridDomain.generate_grid_domain(
        "rectangular", N_u=Nu, N_v=Nv, N_w=Nw, **transform_args
    )
    transform = grid.transform
    if metric_fn is not None:
        transform.metric_tensor_func = metric_fn
    transform.grid_boundaries = boundary_conditions
    grid.grid_boundaries = boundary_conditions

    manifold = ManifoldPackage(transform, grid, laplace_kwargs=laplace_kwargs)
    package = manifold.laplace_package()
    return transform, grid, package


GEOMETRY_REGISTRY: Dict[str, GeometryBuilder] = {
    "rect_euclidean": _build_rect_euclidean,
    "pca_nd": _build_pca_nd,
    "custom_metric": _build_custom_metric,
}


def build_geometry(config: Dict[str, Any]):
    key = config.get("key")
    if key not in GEOMETRY_REGISTRY:
        raise ValueError(f"Unknown geometry key '{key}'")
    return GEOMETRY_REGISTRY[key](config)
