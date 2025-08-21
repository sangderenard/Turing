
import math
import types
import numpy as np
import torch
import pytest

from src.common.tensors.abstract_convolution import laplace_nd as laplace

# ---- helpers ----

def scipy_coo_to_torch_sparse(coo, dtype=torch.float64, device="cpu"):
    idx = torch.tensor([coo.row, coo.col], dtype=torch.long, device=device)
    val = torch.tensor(coo.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, size=coo.shape, device=device).coalesce()

def implicit_heat_step(u, L, alpha_dt, tol=1e-8, iters=200):
    """
    Implicit Euler: (I + alpha_dt L) u_{t+dt} = u_t  (SPD solve via CG)
    u: (N,) or (N,C) dense
    L: (N,N) torch.sparse_coo
    """
    N = L.shape[0]
    dev = u.device
    I = torch.sparse_coo_tensor(torch.arange(N, device=dev).repeat(2,1),
                                torch.ones(N, device=dev), L.shape, device=dev).coalesce()
    A = (I + alpha_dt * L).coalesce()

    def cg(b):
        x = torch.zeros_like(b)
        r = b - torch.sparse.mm(A, x)
        p = r.clone()
        rs_old = (r*r).sum()
        for _ in range(iters):
            Ap = torch.sparse.mm(A, p)
            alpha = rs_old / ( (p*Ap).sum() + 1e-18 )
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = (r*r).sum()
            if rs_new.sqrt() < tol:
                break
            p = r + (rs_new/rs_old) * p
            rs_old = rs_new
        return x

    if u.dim()==1:
        return cg(u.unsqueeze(1)).squeeze(1)
    else:
        outs = []
        for c in range(u.size(1)):
            outs.append(cg(u[:,c:c+1]).squeeze(1))
        return torch.stack(outs, dim=1)

def _has_kw(fn, name):
    try:
        import inspect
        sig = inspect.signature(fn)
        return name in sig.parameters
    except Exception:
        return False

# ---- tests ----

@pytest.mark.parametrize("shape", [(8, 8, 8)])
def test_dirichlet_cube_sin_analytic(shape):
    """
    On a unit cube with Dirichlet BCs, f(x,y,z)=sin(pi x) sin(pi y) sin(pi z) satisfies:
    -Δ f = 3π^2 f
    """
    if not hasattr(laplace, "BuildLaplace3D"):
        pytest.skip("BuildLaplace3D not available in this laplace.py")

    N_u, N_v, N_w = shape
    Lx = Ly = Lz = 1.0
    device = "cpu"

    # minimal RectangularTransform + GridDomain hooks if provided, otherwise craft a simple mesh
    
    transform = laplace.RectangularTransform(Lx=Lx, Ly=Ly, Lz=Lz, device=device)
    grid_u, grid_v, grid_w = transform.create_grid_mesh(N_u, N_v, N_w)
    grid_domain = laplace.GridDomain.generate_grid_domain(
        coordinate_system='rectangular', N_u=N_u, N_v=N_v, N_w=N_w,
        Lx=Lx, Ly=Ly, Lz=Lz, device=device
    )


    boundary_conditions = ('dirichlet','dirichlet','dirichlet','dirichlet','dirichlet','dirichlet')
    BL = laplace.BuildLaplace3D(grid_domain=grid_domain, precision=torch.float64, resolution=max(shape))

    # If the builder exposes normalize_offdiag or normalize flag, disable it
    build_fn = BL.build_general_laplace
    kwargs = dict(grid_u=grid_u, grid_v=grid_v, grid_w=grid_w,
                  boundary_conditions=boundary_conditions, device=device, f=0.0)
    for key in ("normalize_offdiag","normalize","row_normalize"):
        if _has_kw(build_fn, key):
            kwargs[key] = False

    L_dense, L_scipy = build_fn(**kwargs)

    # Torch sparse L
    if L_dense is not None:
        L = L_dense.to(torch.float64)
    else:
        L = scipy_coo_to_torch_sparse(L_scipy, dtype=torch.float64, device=device)

    # build f and compute numerical Laplacian
    X, Y, Z = torch.meshgrid(torch.linspace(0, Lx, N_u),
                             torch.linspace(0, Ly, N_v),
                             torch.linspace(0, Lz, N_w), indexing='ij')
    f = torch.sin(math.pi*X) * torch.sin(math.pi*Y) * torch.sin(math.pi*Z)
    f_flat = f.reshape(-1).to(torch.float64)

    if L.is_sparse:
        num = torch.sparse.mm(L, f_flat[:,None]).squeeze(1)
    else:
        num = L @ f_flat

    ana = (3 * (math.pi**2)) * f_flat
    # boundary nodes may be enforced as identity for Dirichlet; ignore them by masking interior
    mask = (X>0) & (X<1) & (Y>0) & (Y<1) & (Z>0) & (Z<1)
    mask = mask.reshape(-1)
    rel_err = ( (num[mask] - ana[mask]).abs() / (ana[mask].abs() + 1e-12) ).max().item()
    assert rel_err < 0.15, f"relative error too high: {rel_err}"

def test_symmetry_and_psd_small_2d():
    # If only 2D builder exists, use it; otherwise craft a small 2D slice via the 2D API/class
    if hasattr(laplace, "BuildLaplace"):
        BL = laplace.BuildLaplace(grid_domain=None, precision=torch.float64, resolution=16)
        N = 12
        u = torch.linspace(0,1,N)
        v = torch.linspace(0,1,N)
        build_fn = BL.build_general_laplace
        kwargs = dict(grid_u=u, grid_v=v, boundary_conditions=('dirichlet','dirichlet','dirichlet','dirichlet'), device="cpu")
        for key in ("normalize_offdiag","normalize","row_normalize"):
            if _has_kw(build_fn, key):
                kwargs[key] = False
        Ldense, Lcoo = build_fn(**kwargs)
        if Ldense is not None:
            L = Ldense
        else:
            L = scipy_coo_to_torch_sparse(Lcoo, dtype=torch.float64, device="cpu")
        # symmetry
        if L.is_sparse:
            Lt = torch.sparse_coo_tensor(L.indices().flip(0), L.values(), L.shape).coalesce()
            diff = (Lt.values() - L.values()).abs().max().item()
            assert diff < 1e-9
        else:
            assert torch.allclose(L, L.T, atol=1e-9)

        # PSD check: y^T L y >= 0
        y = torch.randn(L.shape[0], dtype=torch.float64)
        if L.is_sparse:
            q = torch.dot(y, torch.sparse.mm(L, y[:,None]).squeeze(1))
        else:
            q = torch.dot(y, L @ y)
        assert q.item() >= -1e-8

@pytest.mark.parametrize("shape", [(20, 20, 20)])
def test_sparse_vs_dense_consistency_tiny(shape):
    # Build a very small 3D Laplacian and compare dense vs sparse matmul if both are provided
    if not hasattr(laplace, "BuildLaplace3D"):
        pytest.skip("BuildLaplace3D not available")


    N_u, N_v, N_w = shape
    Lx = Ly = Lz = 1.0
    device = "cpu"

    transform = laplace.RectangularTransform(Lx=Lx, Ly=Ly, Lz=Lz, device=device)
    grid_u, grid_v, grid_w = transform.create_grid_mesh(N_u, N_v, N_w)
    grid_domain = laplace.GridDomain.generate_grid_domain(
        coordinate_system='rectangular', N_u=N_u, N_v=N_v, N_w=N_w,
        Lx=Lx, Ly=Ly, Lz=Lz, device=device
    )

    BL = laplace.BuildLaplace3D(grid_domain=grid_domain, precision=torch.float64, resolution=6)
    N = 6
    u = torch.linspace(0,1,N)
    v = torch.linspace(0,1,N)
    w = torch.linspace(0,1,N)
    kwargs = dict(grid_u=u, grid_v=v, grid_w=w,
                  boundary_conditions=('neumann','neumann','neumann','neumann','neumann','neumann'),
                  device="cpu")
    for key in ("normalize_offdiag","normalize","row_normalize"):
        if _has_kw(BL.build_general_laplace, key):
            kwargs[key] = False
    Ldense, Lcoo = BL.build_general_laplace(**kwargs)
    if Ldense is None:
        pytest.skip("Dense not returned; cannot compare")
    Ls = scipy_coo_to_torch_sparse(Lcoo, dtype=torch.float64, device="cpu")
    x = torch.randn(N**3, dtype=torch.float64)
    y_dense = Ldense @ x
    y_sparse = torch.sparse.mm(Ls, x[:,None]).squeeze(1)
    assert torch.allclose(y_dense, y_sparse, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("shape", [(20, 20, 20)])
def test_heat_diffusion_energy_decreases(shape):


    N_u, N_v, N_w = shape
    Lx = Ly = Lz = 1.0
    device = "cpu"

    transform = laplace.RectangularTransform(Lx=Lx, Ly=Ly, Lz=Lz, device=device)
    grid_u, grid_v, grid_w = transform.create_grid_mesh(N_u, N_v, N_w)
    grid_domain = laplace.GridDomain.generate_grid_domain(
        coordinate_system='rectangular', N_u=N_u, N_v=N_v, N_w=N_w,
        Lx=Lx, Ly=Ly, Lz=Lz, device=device
    )

    # diffusion should monotonically decrease Dirichlet energy u^T L u for small steps
    if not hasattr(laplace, "BuildLaplace3D"):
        pytest.skip("BuildLaplace3D not available")
    BL = laplace.BuildLaplace3D(grid_domain=grid_domain, precision=torch.float64, resolution=8)
    N = 8
    u = torch.linspace(0,1,N)
    v = torch.linspace(0,1,N)
    w = torch.linspace(0,1,N)
    kwargs = dict(grid_u=u, grid_v=v, grid_w=w,
                  boundary_conditions=('neumann','neumann','neumann','neumann','neumann','neumann'),
                  device="cpu")
    for key in ("normalize_offdiag","normalize","row_normalize"):
        if _has_kw(BL.build_general_laplace, key):
            kwargs[key] = False
    Ldense, Lcoo = BL.build_general_laplace(**kwargs)
    L = Ldense if Ldense is not None else scipy_coo_to_torch_sparse(Lcoo, dtype=torch.float64, device="cpu")
    x0 = torch.randn(N**3, dtype=torch.float64)
    if L.is_sparse:
        e0 = torch.dot(x0, torch.sparse.mm(L, x0[:,None]).squeeze(1))
        x1 = implicit_heat_step(x0, L, alpha_dt=0.1)
        e1 = torch.dot(x1, torch.sparse.mm(L, x1[:,None]).squeeze(1))
    else:
        e0 = torch.dot(x0, L @ x0)
        I = torch.eye(L.shape[0], dtype=L.dtype)
        x1 = torch.linalg.solve(I + 0.1*L, x0)
        e1 = torch.dot(x1, L @ x1)
    assert e1 <= e0 + 1e-8
