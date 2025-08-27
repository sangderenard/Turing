"""
ndpca3_laplace_demo.py
---------------------

A minimal demo/wrapper that builds a 3D Laplace package using BuildLaplace3D and feeds it to NDPCA3Conv3d.
This file demonstrates the intended workflow: build a metric-aware Laplacian package and use it as the 'package' argument for the NDPCA3Conv3d forward pass.
"""

from .laplace_nd import BuildLaplace3D, GridDomain
from .ndpca3conv import NDPCA3Conv3d
from .ndpca3transform import PCABasisND, fit_metric_pca, PCANDTransform
from ..abstraction import AbstractTensor
import numpy as np


def build_pca_transform_and_grid(Nu=8, Nv=8, Nw=8, n=8):
    AT = AbstractTensor
    # Synthesize intrinsic samples (B, n)
    B = 500
    t = AT.arange(0, B, 1)
    t = (t / (B - 1) - 0.5) * 6.283185307179586
    base = AT.stack([
        t.sin(), t.cos(), (2 * t).sin(), (0.5 * t).cos(),
        (0.3 * t).sin(), (1.7 * t).cos(), (0.9 * t).sin(), (1.3 * t).cos()
    ], dim=-1)
    scale = AT.get_tensor([2.0, 1.5, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1])
    u_samples = base * scale
    weights = (-(t**2)).exp()
    M = AT.eye(n)
    diag = AT.get_tensor([1.0, 0.5, 0.25, 2.0, 1.0, 3.0, 0.8, 1.2])
    M = M * diag.reshape(1, -1)
    M = M.swapaxes(-1, -2) * diag.reshape(1, -1)
    basis = fit_metric_pca(u_samples, weights=weights, metric_M=M)
    def phi_fn(U, V, W):
        feats = [U, V, W, (U*V), (V*W), (W*U), (U.sin()), (V.cos())]
        return AT.stack(feats, dim=-1)
    xform = PCANDTransform(basis, phi_fn, d_visible=3)
    # Now build the canonical grid using GridDomain
    grid_domain = GridDomain(
        AT.linspace(-1.0, 1.0, Nu).reshape(Nu, 1, 1) * AT.ones((1, Nv, Nw)),
        AT.linspace(-1.0, 1.0, Nv).reshape(1, Nv, 1) * AT.ones((Nu, 1, Nw)),
        AT.linspace(-1.0, 1.0, Nw).reshape(1, 1, Nw) * AT.ones((Nu, Nv, 1)),
        grid_boundaries=(True,)*6,
        transform=xform,
        coordinate_system="rectangular"
    )
    return xform, grid_domain


pass  # replaced by build_pca_transform_and_grid


def build_laplace_package(grid_domain, xform, boundary_conditions=("dirichlet",)*6):
    builder = BuildLaplace3D(
        grid_domain=grid_domain,
        wave_speed=343,
        precision=getattr(AbstractTensor, "float_dtype_", None) or grid_domain.U.dtype,
        resolution=grid_domain.U.shape[0],
        metric_tensor_func=xform.metric_tensor_func,
        boundary_conditions=boundary_conditions,
        artificial_stability=1e-10,
        device=getattr(grid_domain.U, "device", None),
    )
    _, _, package = builder.build_general_laplace(grid_domain.U, grid_domain.V, grid_domain.W, return_package=True)
    return package


def main():
    # 1. Build transform and canonical grid domain
    xform, grid_domain = build_pca_transform_and_grid(Nu=8, Nv=8, Nw=8)
    package = build_laplace_package(grid_domain, xform)

    # 2. Build a dummy input and NDPCA3Conv3d
    B, C = 2, 4
    x = AbstractTensor.randn((B, C, grid_domain.resolution_u, grid_domain.resolution_v, grid_domain.resolution_w))
    conv = NDPCA3Conv3d(
        in_channels=C,
        out_channels=C,
        like=x,
        grid_shape=(grid_domain.resolution_u, grid_domain.resolution_v, grid_domain.resolution_w),
        boundary_conditions=("dirichlet",)*6,
        k=3,
        eig_from="g",
        pointwise=False,
    )
    y = conv.forward(x, package=package)
    print("Output shape:", y.shape)

if __name__ == "__main__":
    main()
