import numpy as np
from src.common.tensors.numpy_backend import NumPyTensorOperations  # noqa: F401
import pytest
from src.common.tensors.abstract_convolution import laplace_nd as laplace


@pytest.mark.xfail(reason="NumPy backend lacks required ops")
def test_laplace_builds_with_numpy():
    if not hasattr(laplace, "BuildLaplace3D"):
        pytest.skip("BuildLaplace3D not available")
    N = 4
    Lx = Ly = Lz = 1.0
    transform = laplace.RectangularTransform(Lx=Lx, Ly=Ly, Lz=Lz, device="cpu")
    grid_u, grid_v, grid_w = transform.create_grid_mesh(N, N, N)
    grid_domain = laplace.GridDomain.generate_grid_domain(
        coordinate_system="rectangular", N_u=N, N_v=N, N_w=N, Lx=Lx, Ly=Ly, Lz=Lz, device="cpu",
    )
    BL = laplace.BuildLaplace3D(grid_domain=grid_domain, precision=None, resolution=N)
    L_dense, L_scipy = BL.build_general_laplace(
        grid_u=grid_u,
        grid_v=grid_v,
        grid_w=grid_w,
        boundary_conditions=("dirichlet",) * 6,
        device="cpu",
        f=0.0,
    )
    assert L_dense is not None or L_scipy is not None

