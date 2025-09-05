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


def _build_grid_laplacian(
    U: int,
    V: int,
    W: int,
    device: str,
    *,
    boundary_mask: AbstractTensor | None = None,
    boundary_flux: AbstractTensor | None = None,
    normalize: bool = False,
):
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
    builder = BuildLaplace3D(
        grid_domain=grid_domain, precision=None, resolution=max(U, V, W)
    )
    build_kwargs = {}
    import inspect

    sig = inspect.signature(builder.build_general_laplace)
    if "boundary_mask" in sig.parameters and boundary_mask is not None:
        build_kwargs["boundary_mask"] = boundary_mask
    if "boundary_flux" in sig.parameters and boundary_flux is not None:
        build_kwargs["boundary_flux"] = boundary_flux
    if "normalize" in sig.parameters:
        build_kwargs["normalize"] = normalize

    L_dense, L_sparse, _ = builder.build_general_laplace(
        grid_u=grid_u,
        grid_v=grid_v,
        grid_w=grid_w,
        boundary_conditions=("dirichlet",) * 6,
        device=device,
        f=0.0,
        **build_kwargs,
    )
    return L_dense if L_dense is not None else L_sparse.to_dense()


def filtered_poisson(
    rhs: AbstractTensor,
    *,
    iterations: int = 50,
    filter_strength: float = 0.0,
    mode: str | None = None,
    adjacency: AbstractTensor | None = None,
    boundary_mask: AbstractTensor | None = None,
    boundary_flux: AbstractTensor | None = None,
    normalization: str = "none",
    tol: float | None = None,
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
        ``'manifold'`` or ``'graph'``. If ``None`` the type is inferred from the
        shapes of ``rhs`` and ``adjacency``.
    adjacency:
        Adjacency matrix for graph mode.
    boundary_mask:
        Optional boolean mask marking Neumann boundaries.
    boundary_flux:
        Optional outward flux coefficients for boundary nodes.
    normalization:
        ``"none"``, ``"symmetric"`` or ``"random_walk"``.
    tol:
        Convergence tolerance for early termination of the iteration.
    """

    rhs = AbstractTensor.get_tensor(rhs)
    if adjacency is not None:
        adjacency = AbstractTensor.get_tensor(adjacency, like=rhs)
    if boundary_mask is not None:
        boundary_mask = AbstractTensor.get_tensor(boundary_mask, like=rhs)
    if boundary_flux is not None:
        boundary_flux = AbstractTensor.get_tensor(boundary_flux, like=rhs)

    if mode is None:
        if adjacency is not None:
            mode = "graph"
        elif rhs.ndim == 5 and rhs.shape[0] == 1 and rhs.shape[1] == 1:
            mode = "manifold"
        else:
            raise ValueError("could not infer domain type; specify 'mode'")

    if mode == "manifold":
        if rhs.ndim != 5 or rhs.shape[0] != 1 or rhs.shape[1] != 1:
            raise ValueError("manifold mode expects shape (1, 1, U, V, W)")
        U, V, W = rhs.shape[2:]
        try:
            L = _build_grid_laplacian(
                U,
                V,
                W,
                rhs.device,
                boundary_mask=boundary_mask,
                boundary_flux=boundary_flux,
                normalize=normalization != "none",
            )
        except Exception as e:  # pragma: no cover - builder may be incomplete
            raise RuntimeError("grid Laplacian builder unavailable") from e
        diag = AbstractTensor.diag(L)
        rhs_vec = rhs.reshape(-1)
        omega = 1.0
    elif mode == "graph":
        if adjacency is None:
            raise ValueError("adjacency matrix required for graph mode")
        builder = BuildGraphLaplace(
            adjacency,
            normalization=normalization,
            boundary_mask=boundary_mask,
            boundary_flux=boundary_flux,
        )
        L, _, pkg = builder.build()
        diag = pkg["degree"]
        rhs_vec = rhs.reshape(-1)
        omega = 0.5 if normalization == "none" else 1.0
    else:
        raise ValueError("mode must be 'manifold' or 'graph'")

    u_vec = AbstractTensor.zeros_like(rhs_vec)
    if filter_strength > 0.0:
        rhs_vec = rhs_vec + filter_strength * (L @ rhs_vec)

    inv_diag = 1.0 / diag
    for _ in range(int(iterations)):
        lap_u = L @ u_vec
        new_u = u_vec + (rhs_vec - lap_u) * inv_diag * omega
        if tol is not None:
            delta = AbstractTensor.abs(new_u - u_vec).max()
            if float(delta) < tol:
                u_vec = new_u
                break
        u_vec = new_u

    return u_vec.reshape(rhs.shape)
