# Filtered Poisson solver for AbstractTensor grids or graphs.
#
# The solver delegates Laplacian construction to the existing builders in
# ``abstract_convolution.laplace_nd`` so it can operate on either 3‑D grids
# (manifold mode) or generic graphs defined by an adjacency matrix.
from __future__ import annotations

from .abstraction import AbstractTensor
from .abstract_convolution.laplace_nd import (
    BuildGraphLaplace,
    BuildLaplace3D,
    GridDomain,
    RectangularTransform,
)

__all__ = ["filtered_poisson"]


def _build_grid_laplacian(U: int, V: int, W: int, device: str):
    """Helper constructing the grid Laplacian matrix via ``BuildLaplace3D``."""

    transform = RectangularTransform(Lx=1.0, Ly=1.0, Lz=1.0, device=device)
    grid_u, grid_v, grid_w = transform.create_grid_mesh(U, V, W)
    grid_domain = GridDomain.generate_grid_domain(
        coordinate_system="rectangular",
        N_u=U,
        N_v=V,
        N_w=W,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        device=device,
    )
    builder = BuildLaplace3D(grid_domain=grid_domain, precision=None, resolution=max(U, V, W))
    L_dense, L_sparse, _ = builder.build_general_laplace(
        grid_u=grid_u,
        grid_v=grid_v,
        grid_w=grid_w,
        boundary_conditions=("dirichlet",) * 6,
        device=device,
        f=0.0,
    )
    return L_dense if L_dense is not None else L_sparse.to_dense()


def filtered_poisson(
    rhs: AbstractTensor,
    *,
    iterations: int = 50,
    filter_strength: float = 0.0,
    mode: str = "manifold",
    adjacency: AbstractTensor | None = None,
) -> AbstractTensor:
    """Solve ``\nabla^2 u = rhs`` via Jacobi iteration.

    Parameters
    ----------
    rhs:
        Right‑hand side data. ``mode='manifold'`` expects shape ``(1, 1, U, V, W)``
        with each spatial dimension at least 2. ``mode='graph'`` treats the last
        dimension as graph nodes.
    iterations:
        Number of Jacobi iterations to perform.
    filter_strength:
        Optional coefficient for pre‑smoothing the RHS.
    mode:
        ``'manifold'`` or ``'graph'``.
    adjacency:
        Required when ``mode='graph'``.
    """

    rhs = AbstractTensor.get_tensor(rhs)

    if mode == "manifold":
        if rhs.ndim != 5 or rhs.shape[0] != 1 or rhs.shape[1] != 1:
            raise ValueError("manifold mode expects shape (1, 1, U, V, W)")
        U, V, W = rhs.shape[2:]
        try:
            L = _build_grid_laplacian(U, V, W, rhs.device)
        except Exception as e:  # pragma: no cover - builder may be incomplete
            raise RuntimeError("grid Laplacian builder unavailable") from e
        diag = AbstractTensor.diag(L)
        rhs_vec = rhs.reshape(-1)
        omega = 1.0
    elif mode == "graph":
        if adjacency is None:
            raise ValueError("adjacency matrix required for graph mode")
        builder = BuildGraphLaplace(adjacency)
        L, _, pkg = builder.build()
        diag = pkg["degree"]
        rhs_vec = rhs.reshape(-1)
        omega = 0.5
    else:
        raise ValueError("mode must be 'manifold' or 'graph'")

    u_vec = AbstractTensor.zeros_like(rhs_vec)
    if filter_strength > 0.0:
        rhs_vec = rhs_vec + filter_strength * (L @ rhs_vec)

    inv_diag = 1.0 / diag
    for _ in range(int(iterations)):
        lap_u = L @ u_vec
        u_vec = u_vec + (rhs_vec - lap_u) * inv_diag * omega

    return u_vec.reshape(rhs.shape)
