import numpy as np
import pytest
from src.common.tensors import AbstractTensor
from src.common.tensors.filtered_poisson import filtered_poisson
from src.common.tensors.abstract_convolution.laplace_nd import (
    BuildGraphLaplace,
    BuildLaplace3D,
    GridDomain,
    RectangularTransform,
)


def test_filtered_poisson_residual_small():
    rhs = AbstractTensor.arange(8, dtype=AbstractTensor.float_dtype_).reshape((1, 1, 2, 2, 2))
    try:
        sol = filtered_poisson(rhs, iterations=50)
    except RuntimeError:
        pytest.skip("grid Laplacian builder unavailable")

    transform = RectangularTransform(Lx=1.0, Ly=1.0, Lz=1.0, device="cpu")
    grid_u, grid_v, grid_w = transform.create_grid_mesh(2, 2, 2)
    grid_domain = GridDomain.generate_grid_domain(
        coordinate_system="rectangular", N_u=2, N_v=2, N_w=2, Lx=1.0, Ly=1.0, Lz=1.0, device="cpu"
    )
    builder = BuildLaplace3D(grid_domain=grid_domain, precision=None, resolution=2)
    L_dense, L_sparse, _ = builder.build_general_laplace(
        grid_u=grid_u,
        grid_v=grid_v,
        grid_w=grid_w,
        boundary_conditions=("dirichlet",) * 6,
        device="cpu",
        f=0.0,
    )
    L = L_dense if L_dense is not None else L_sparse.to_dense()
    residual = (L @ sol.reshape(-1)) - rhs.reshape(-1)
    max_err = abs(residual).max().item()
    assert max_err < 1e-2


def test_filtered_poisson_graph_mode_inferred():
    rhs_np = np.array([-1.0, 2.0, -1.0], dtype=float)
    rhs = AbstractTensor.get_tensor(rhs_np)
    adj_np = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    adjacency = AbstractTensor.get_tensor(adj_np, like=rhs)
    builder = BuildGraphLaplace(adjacency)
    L, _, _ = builder.build()
    sol = filtered_poisson(rhs, iterations=200, adjacency=adjacency)
    residual = (L @ sol.reshape(-1)) - rhs
    max_err = abs(residual).max().item()
    assert max_err < 1e-2


def test_filtered_poisson_graph_boundary_normalized():
    rhs_np = np.array([-1.0, 2.0, -1.0], dtype=float)
    rhs = AbstractTensor.get_tensor(rhs_np)
    adj_np = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    adjacency = AbstractTensor.get_tensor(adj_np, like=rhs)
    boundary_mask = AbstractTensor.get_tensor([1.0, 0.0, 0.0], like=rhs)
    boundary_flux = AbstractTensor.get_tensor([1.0, 0.0, 0.0], like=rhs)
    builder = BuildGraphLaplace(
        adjacency,
        normalization="symmetric",
        boundary_mask=boundary_mask,
        boundary_flux=boundary_flux,
    )
    L, _, _ = builder.build()
    sol = filtered_poisson(
        rhs,
        iterations=200,
        adjacency=adjacency,
        boundary_mask=boundary_mask,
        boundary_flux=boundary_flux,
        normalization="symmetric",
    )
    residual = (L @ sol.reshape(-1)) - rhs
    max_err = abs(residual).max().item()
    assert max_err < 1e-2


def test_filtered_poisson_convergence_tol():
    rhs_np = np.array([1.0, 0.0, -1.0], dtype=float)
    rhs = AbstractTensor.get_tensor(rhs_np)
    adj_np = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    adjacency = AbstractTensor.get_tensor(adj_np, like=rhs)
    sol_one = filtered_poisson(rhs, iterations=1, adjacency=adjacency)
    sol_tol = filtered_poisson(rhs, iterations=50, adjacency=adjacency, tol=1e6)
    sol_full = filtered_poisson(rhs, iterations=50, adjacency=adjacency)
    assert AbstractTensor.allclose(sol_tol, sol_one)
    assert not AbstractTensor.allclose(sol_tol, sol_full)


@pytest.mark.parametrize(
    "norm,expected",
    [
        (
            "none",
            [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]],
        ),
        (
            "symmetric",
            [
                [1.0, -1.0 / np.sqrt(2.0), 0.0],
                [-1.0 / np.sqrt(2.0), 1.0, -1.0 / np.sqrt(2.0)],
                [0.0, -1.0 / np.sqrt(2.0), 1.0],
            ],
        ),
        (
            "random_walk",
            [[1.0, -1.0, 0.0], [-0.5, 1.0, -0.5], [0.0, -1.0, 1.0]],
        ),
    ],
)
def test_graph_laplacian_normalization_variants(norm, expected):
    adj_np = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    adjacency = AbstractTensor.get_tensor(adj_np)
    builder = BuildGraphLaplace(adjacency, normalization=norm)
    L, _, _ = builder.build()
    expected_t = AbstractTensor.get_tensor(np.array(expected, dtype=float), like=adjacency)
    assert AbstractTensor.allclose(L, expected_t)
