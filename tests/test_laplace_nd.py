import numpy as np
from src.common.tensors.numpy_backend import NumPyTensorOperations  # noqa: F401
from src.common.tensors.pure_backend import PurePythonTensorOperations  # noqa: F401
import pytest
from src.common.tensors.abstract_convolution import laplace_nd as laplace


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


def _laplace_power_section(backend_name, backend_cls, N=8):
    from src.common.tensors.abstraction import BACKEND_REGISTRY
    orig = BACKEND_REGISTRY.copy()
    try:
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY[backend_name] = backend_cls
        transform = laplace.RectangularTransform(Lx=1.0, Ly=1.0, Lz=1.0, device="cpu")
        try:
            transform.create_grid_mesh(N, N, N)
        except NotImplementedError:
            pass
        x = backend_cls.linspace(0, 1, N)
        return (x - 0.5) ** 2
    finally:
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY.update(orig)


def test_power_operation_average_time_pure_vs_numpy():
    pure_res = PurePythonTensorOperations.benchmark(
        lambda: _laplace_power_section("pure_python", PurePythonTensorOperations), repeat=3
    )
    numpy_res = NumPyTensorOperations.benchmark(
        lambda: _laplace_power_section("numpy", NumPyTensorOperations), repeat=3
    )
    assert pure_res.mean > 0 and numpy_res.mean > 0
    assert pure_res.mean != numpy_res.mean
