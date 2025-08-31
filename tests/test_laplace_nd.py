import numpy as np
from src.common.tensors.numpy_backend import NumPyTensorOperations  # noqa: F401
from src.common.tensors.pure_backend import PurePythonTensorOperations  # noqa: F401
from src.common.tensors.abstraction import AbstractTensor
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
    L_dense, L_scipy, _ = BL.build_general_laplace(
        grid_u=grid_u,
        grid_v=grid_v,
        grid_w=grid_w,
        boundary_conditions=("dirichlet",) * 6,
        device="cpu",
        f=0.0,
    )
    assert L_dense is not None or L_scipy is not None


def test_edge_index_dtype_long():
    if not hasattr(laplace, "TransformHub"):
        pytest.skip("TransformHub not available")

    edges = [[0, 1], [1, 2]]
    edge_index = AbstractTensor.tensor(edges, dtype=AbstractTensor.long_dtype_)

    data = AbstractTensor.arange(3, dtype=AbstractTensor.long_dtype_)
    _ = data[edge_index]

    assert edge_index.dtype == AbstractTensor.long_dtype_


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
        x.track_time = True
        return (x - 0.5) ** 2
    finally:
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY.update(orig)


def test_power_operation_average_time_pure_vs_numpy():
    pure_prof = PurePythonTensorOperations.benchmark(
        lambda: _laplace_power_section("pure_python", PurePythonTensorOperations), repeat=3
    )
    numpy_prof = NumPyTensorOperations.benchmark(
        lambda: _laplace_power_section("numpy", NumPyTensorOperations), repeat=3
    )
    pure_mean = pure_prof.per_op()["pow"]["mean"]
    numpy_mean = numpy_prof.per_op()["pow"]["mean"]
    assert pure_mean > 0 and numpy_mean > 0
    assert pure_mean != numpy_mean


def test_compute_partials_and_normals_strict(monkeypatch):
    if not hasattr(laplace, "BuildLaplace3D"):
        pytest.skip("BuildLaplace3D not available")
    # Enable strict mode for autograd and ensure it is restored afterwards
    monkeypatch.setenv("AUTOGRAD_STRICT", "1")
    monkeypatch.setattr(AbstractTensor.autograd, "strict", True)

    N = 3
    Lx = Ly = Lz = 1.0
    transform = laplace.RectangularTransform(Lx=Lx, Ly=Ly, Lz=Lz, device="cpu")
    grid_u, grid_v, grid_w = transform.create_grid_mesh(N, N, N)
    grid_domain = laplace.GridDomain.generate_grid_domain(
        coordinate_system="rectangular", N_u=N, N_v=N, N_w=N, Lx=Lx, Ly=Ly, Lz=Lz, device="cpu"
    )
    BL = laplace.BuildLaplace3D(grid_domain=grid_domain, precision=None, resolution=N)

    laplacian_tensor, laplacian_sparse, _ = BL.build_general_laplace(
        grid_u=grid_u,
        grid_v=grid_v,
        grid_w=grid_w,
        boundary_conditions=("dirichlet",) * 6,
        device="cpu",
        f=0.0,
    )
    assert laplacian_tensor is not None or laplacian_sparse is not None
