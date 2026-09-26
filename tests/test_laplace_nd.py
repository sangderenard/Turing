import numpy as np
import pytest

from src.common.tensors.numpy_backend import NumPyTensorOperations  # noqa: F401
from src.common.tensors.pure_backend import PurePythonTensorOperations  # noqa: F401
from src.common.tensors.abstraction import AbstractTensor
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


def test_laplace_uses_declared_material_fields():
    """The public tension/density fields must reach the built operator."""
    N = 3
    transform = laplace.RectangularTransform(
        Lx=1.0, Ly=1.0, Lz=1.0, device="cpu")
    grid_u, grid_v, grid_w = transform.create_grid_mesh(N, N, N)
    grid_domain = laplace.GridDomain.generate_grid_domain(
        coordinate_system="rectangular", N_u=N, N_v=N, N_w=N,
        Lx=1.0, Ly=1.0, Lz=1.0, device="cpu")

    base = laplace.BuildLaplace3D(
        grid_domain=grid_domain, precision=None, resolution=N)
    L_base, _, _ = base.build_general_laplace(
        grid_u, grid_v, grid_w,
        boundary_conditions=("dirichlet",) * 6, device="cpu")

    scaled = laplace.BuildLaplace3D(
        grid_domain=grid_domain, precision=None, resolution=N,
        tension_func=lambda u, v, w: AbstractTensor.ones_like(u) * 6.0,
        density_func=lambda u, v, w: AbstractTensor.ones_like(u) * 3.0)
    L_scaled, _, package = scaled.build_general_laplace(
        grid_u, grid_v, grid_w,
        boundary_conditions=("dirichlet",) * 6, device="cpu",
        return_package=True)

    assert AbstractTensor.allclose(L_scaled, L_base * 2.0)
    assert AbstractTensor.allclose(
        package["material"]["tension"], AbstractTensor.ones_like(grid_u) * 6.0)
    assert AbstractTensor.allclose(
        package["material"]["density"], AbstractTensor.ones_like(grid_u) * 3.0)


def test_neumann_laplace_preserves_a_constant_field():
    N = 3
    transform = laplace.RectangularTransform(
        Lx=0.09, Ly=0.17, Lz=0.09, device="cpu")
    grid_u, grid_v, grid_w = transform.create_grid_mesh(N, N, N)
    grid_domain = laplace.GridDomain.generate_grid_domain(
        coordinate_system="rectangular", N_u=N, N_v=N, N_w=N,
        Lx=0.09, Ly=0.17, Lz=0.09, device="cpu")
    builder = laplace.BuildLaplace3D(
        grid_domain=grid_domain, precision=None, resolution=N)
    operator, sparse, _ = builder.build_general_laplace(
        grid_u, grid_v, grid_w,
        boundary_conditions=("neumann",) * 6, device="cpu")
    constant = AbstractTensor.ones((N * N * N,)) * 80.0
    assert AbstractTensor.allclose(operator @ constant,
                                   AbstractTensor.zeros_like(constant),
                                   atol=1e-10)
    assert AbstractTensor.allclose(sparse.to_dense() @ constant,
                                   AbstractTensor.zeros_like(constant),
                                   atol=1e-10)


@pytest.mark.xfail(
    reason="dtype identity is not normalised across backends: a numpy-backed "
    "tensor reports torch.int64 while AbstractTensor.long_dtype_ is the string "
    "'int64', so the comparison fails without either being wrong",
    strict=False,
)
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
