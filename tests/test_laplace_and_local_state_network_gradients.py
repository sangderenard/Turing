import pytest
import torch
from src.common.tensors.abstract_convolution.laplace_nd import BuildLaplace3D, GridDomain, RectangularTransform
from src.common.tensors.abstract_convolution.local_state_network import LocalStateNetwork, DEFAULT_CONFIGURATION
from src.common.tensors.abstraction import AbstractTensor

def test_laplace_gradient():
    """Test to ensure gradients can be computed for LaplaceND."""
    N_u, N_v, N_w = 20, 20, 20  # Grid resolution
    Lx, Ly, Lz = 1.0, 1.0, 1.0  # Domain size

    transform = RectangularTransform(Lx=Lx, Ly=Ly, Lz=Lz, device='cpu')
    test_tensor = AbstractTensor.randn((N_u, N_v, N_w), device='cpu')
    cls = type(test_tensor)
    dtype = test_tensor.dtype

    grid_u, grid_v, grid_w = transform.create_grid_mesh(N_u, N_v, N_w)
    grid_domain = GridDomain.generate_grid_domain(
        coordinate_system='rectangular',
        N_u=N_u,
        N_v=N_v,
        N_w=N_w,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        device='cpu',
        cls=cls,
        dtype=dtype
    )
    boundary_conditions = ('dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet')
    print(cls.float_dtype)
    build_laplace = BuildLaplace3D(
        grid_domain=grid_domain,
        wave_speed=343,  # Arbitrary value
        precision=cls.float_dtype,
        resolution=N_u,  # Should match N_u, N_v, N_w
        metric_tensor_func=None,  # Use default Euclidean metric
        density_func=None,        # Uniform density
        tension_func=None,        # Uniform tension
        singularity_conditions=None,
        boundary_conditions=boundary_conditions,
        artificial_stability=1e-10
    )

    laplacian_tensor, _ = build_laplace.build_general_laplace(
        grid_u=grid_u,
        grid_v=grid_v,
        grid_w=grid_w,
        boundary_conditions=boundary_conditions,
        grid_boundaries=(True, True, True, True, True, True),
        device='cpu',
        dense=True,
        f=0.0
    )

    laplacian_tensor.backward()

def test_local_state_network_weighted_mode_gradient():
    """Test to ensure gradients of LocalStateNetwork parameters can be computed in weighted mode."""
    N_u, N_v, N_w = 20, 20, 20  # Grid resolution
    Lx, Ly, Lz = 1.0, 1.0, 1.0  # Domain size

    transform = RectangularTransform(Lx=Lx, Ly=Ly, Lz=Lz, device='cpu')
    test_tensor = AbstractTensor.randn((N_u, N_v, N_w), device='cpu')
    cls = type(test_tensor)
    dtype = test_tensor.dtype

    grid_u, grid_v, grid_w = transform.create_grid_mesh(N_u, N_v, N_w)
    grid_domain = GridDomain.generate_grid_domain(
        coordinate_system='rectangular',
        N_u=N_u,
        N_v=N_v,
        N_w=N_w,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        device='cpu',
        cls=cls,
        dtype=dtype
    )
    boundary_conditions = ('dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet')

    build_laplace = BuildLaplace3D(
        grid_domain=grid_domain,
        wave_speed=343,  # Arbitrary value
        precision=cls.float_dtype,
        resolution=N_u,  # Should match N_u, N_v, N_w
        metric_tensor_func=None,  # Use default Euclidean metric
        density_func=None,        # Uniform density
        tension_func=None,        # Uniform tension
        singularity_conditions=None,
        boundary_conditions=boundary_conditions,
        artificial_stability=1e-10
    )

    _, _, package = build_laplace.build_general_laplace(
        grid_u=grid_u,
        grid_v=grid_v,
        grid_w=grid_w,
        boundary_conditions=boundary_conditions,
        grid_boundaries=(True, True, True, True, True, True),
        device='cpu',
        dense=True,
        f=0.0,
        return_package=True
    )
    
    local_state_network = package["local_state_network"]
    

    found_a_param = False
    input_tensor = AbstractTensor.randn((1, N_u, N_v, N_w, 3, 3, 3), requires_grad=True)
    weighted_tensor = local_state_network.forward(input_tensor)[0]
    weighted_tensor.sum().backward()

    # Check gradients for LocalStateNetwork parameters
    for param in local_state_network.parameters():
        if param.requires_grad:
            found_a_param = True
            assert param.grad is not None, "Gradient for LocalStateNetwork parameter in weighted mode is None"
    assert found_a_param, "No LocalStateNetwork parameters found with gradients"

    local_state_network.zero_grad()

def test_local_state_network_modulated_mode_gradient():
    """Test to ensure gradients of LocalStateNetwork parameters can be computed in modulated mode."""
    N_u, N_v, N_w = 20, 20, 20  # Grid resolution
    Lx, Ly, Lz = 1.0, 1.0, 1.0  # Domain size

    transform = RectangularTransform(Lx=Lx, Ly=Ly, Lz=Lz, device='cpu')
    test_tensor = AbstractTensor.randn((N_u, N_v, N_w), device='cpu')
    cls = type(test_tensor)
    dtype = test_tensor.dtype

    grid_u, grid_v, grid_w = transform.create_grid_mesh(N_u, N_v, N_w)
    grid_domain = GridDomain.generate_grid_domain(
        coordinate_system='rectangular',
        N_u=N_u,
        N_v=N_v,
        N_w=N_w,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        device='cpu',
        cls=cls,
        dtype=dtype
    )
    boundary_conditions = ('dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet')

    build_laplace = BuildLaplace3D(
        grid_domain=grid_domain,
        wave_speed=343,  # Arbitrary value
        precision=cls.float_dtype,
        resolution=N_u,  # Should match N_u, N_v, N_w
        metric_tensor_func=None,  # Use default Euclidean metric
        density_func=None,        # Uniform density
        tension_func=None,        # Uniform tension
        singularity_conditions=None,
        boundary_conditions=boundary_conditions,
        artificial_stability=1e-10
    )

    _, _, package = build_laplace.build_general_laplace(
        grid_u=grid_u,
        grid_v=grid_v,
        grid_w=grid_w,
        boundary_conditions=boundary_conditions,
        grid_boundaries=(True, True, True, True, True, True),
        device='cpu',
        dense=True,
        f=0.0,
        return_package=True
    )

    local_state_network = package["local_state_network"]
    local_state_network.zero_grad()  # Clear existing gradients

    found_a_param = False
    input_tensor = AbstractTensor.randn((1, N_u, N_v, N_w, 3, 3, 3), requires_grad=True)
    modulated_tensor = local_state_network.forward(input_tensor)[1]
    modulated_tensor.sum().backward()

    # Check gradients for LocalStateNetwork parameters
    for param in local_state_network.parameters():
        if param.requires_grad:
            found_a_param = True
            assert param.grad is not None, "Gradient for LocalStateNetwork parameter in modulated mode is None"
    assert found_a_param, "No LocalStateNetwork parameters found with gradients"


if __name__ == "__main__":
    test_laplace_gradient()
    test_local_state_network_weighted_mode_gradient()
    test_local_state_network_modulated_mode_gradient()