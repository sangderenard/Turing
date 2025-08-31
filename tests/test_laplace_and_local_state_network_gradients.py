import pytest
from src.common.tensors.abstract_convolution.laplace_nd import BuildLaplace3D, GridDomain, RectangularTransform
from src.common.tensors.abstract_convolution.local_state_network import LocalStateNetwork, DEFAULT_CONFIGURATION
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_nn import Linear


def learnable_metric_tensor_func(u, v, w, dxdu, dydu, dzdu, dxdv, dydv, dzdv, dxdw, dydw, dzdw):
    from src.common.tensors.abstraction import AbstractTensor
    if not hasattr(learnable_metric_tensor_func, "a"):
        learnable_metric_tensor_func.a = AbstractTensor.tensor(1.0, requires_grad=True)
        learnable_metric_tensor_func.b = AbstractTensor.tensor(0.1, requires_grad=True)
    I = AbstractTensor.eye(3).reshape(1, 1, 1, 3, 3)
    uvw = AbstractTensor.stack([u, v, w], dim=-1).reshape(u.shape + (3,))
    outer = uvw.unsqueeze(-1) * uvw.unsqueeze(-2)
    g = learnable_metric_tensor_func.a * I + learnable_metric_tensor_func.b * outer
    g_inv = g
    det = AbstractTensor.ones(u.shape)
    return g, g_inv, det

def test_laplace_gradient():
    """Test to ensure all pipeline parameters (including LSN and metric) receive gradients."""
    from src.common.tensors.abstract_convolution.metric_steered_conv3d import MetricSteeredConv3DWrapper
    grid_shape = (5, 5, 5)
    in_channels = 2
    out_channels = 2
    transform = RectangularTransform(Lx=1.0, Ly=1.0, Lz=1.0, device="cpu")
    wrapper = MetricSteeredConv3DWrapper(
        in_channels=in_channels,
        out_channels=out_channels,
        grid_shape=grid_shape,
        transform=transform,
        boundary_conditions=("dirichlet",) * 6,
        k=3,
        eig_from="g",
        pointwise=True,
        deploy_mode="modulated",
        laplace_kwargs={"metric_tensor_func": learnable_metric_tensor_func, "lambda_reg": 0.5},
    )
    x = AbstractTensor.randn((1, in_channels, *grid_shape), device="cpu")
    y = wrapper.forward(x)
    lsn = wrapper.local_state_network
    reg_loss = getattr(lsn, "_regularization_loss", None)
    if reg_loss is not None:
        loss = y.sum() + reg_loss
    else:
        loss = y.sum()
    loss.backward()
    grad_w = getattr(lsn, "_weighted_padded", None)
    grad_m = getattr(lsn, "_modulated_padded", None)
    if grad_w is not None and grad_m is not None:
        grad_w = getattr(grad_w, "_grad", None) or AbstractTensor.zeros_like(lsn._weighted_padded)
        grad_m = getattr(grad_m, "_grad", None) or AbstractTensor.zeros_like(lsn._modulated_padded)
        lsn.backward(grad_w, grad_m, lambda_reg=0.5)
    # Collect all parameters: wrapper, LSN, and learnable metric
    all_params = list(wrapper.parameters(include_structural=True))
    for name in ["a", "b"]:
        p = getattr(learnable_metric_tensor_func, name, None)
        if p is not None:
            all_params.append(p)
    # Assert all parameters have nonzero gradients
    for param in all_params:
        grad = getattr(param, "grad", None)
        if grad is None:
            grad = getattr(param, "_grad", None)
        label = getattr(param, "_label", str(param))
        assert grad is not None, f"Parameter {label} has no grad"
        assert (grad.abs() > 0).any(), f"Parameter {label} grad is all zero"

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
        deploy_mode="weighted",
        return_package=True,
        lambda_reg=0.5,
    )

    local_state_network = package["local_state_network"]
    package["regularization_loss"].backward()
    zero_w = AbstractTensor.zeros_like(local_state_network._weighted_padded)
    zero_m = AbstractTensor.zeros_like(local_state_network._modulated_padded)
    local_state_network.backward(zero_w, zero_m, lambda_reg=0.5)

    # Log gradient status for all parameters
    for param in package["local_state_network"].parameters(include_all=True):
        print(f"{getattr(param, '_label', 'param')}: grad={'present' if param.grad is not None else 'missing'}")

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
        deploy_mode="modulated",
        return_package=True,
        lambda_reg=0.5,
    )

    local_state_network = package["local_state_network"]
    package["regularization_loss"].backward()
    zero_w = AbstractTensor.zeros_like(local_state_network._weighted_padded)
    zero_m = AbstractTensor.zeros_like(local_state_network._modulated_padded)
    local_state_network.backward(zero_w, zero_m, lambda_reg=0.5)

    # Log gradient status for all parameters
    for param in package["local_state_network"].parameters(include_all=True):
        print(f"{getattr(param, '_label', 'param')}: grad={'present' if param.grad is not None else 'missing'}")

def test_local_state_network_convolutional_modulator_gradient():
    """Test to ensure gradients of LocalStateNetwork parameters can be computed in convolutional modulator mode."""
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

    # Initialize LocalStateNetwork externally ensuring the top level uses RectConv3d
    local_state_network = LocalStateNetwork(
        metric_tensor_func=None,
        grid_shape=(N_u, N_v, N_w),
        switchboard_config=DEFAULT_CONFIGURATION,
        recursion_depth=0,
        max_depth=3,
    )
    

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
        return_package=True,
        deploy_mode="modulated",
        local_state_network=local_state_network,  # Pass the externally initialized LocalStateNetwork
        lambda_reg=0.5,
    )

    local_state_network = package["local_state_network"]
    package["regularization_loss"].backward()
    zero_w = AbstractTensor.zeros_like(local_state_network._weighted_padded)
    zero_m = AbstractTensor.zeros_like(local_state_network._modulated_padded)
    local_state_network.backward(zero_w, zero_m, lambda_reg=0.5)

    # Ensure recursion caps at max_depth
    assert local_state_network.inner_state is not None
    assert local_state_network.inner_state.inner_state is not None
    assert getattr(local_state_network.inner_state.inner_state, "inner_state", None) is None
    assert isinstance(local_state_network.inner_state.inner_state.spatial_layer, Linear)

    # Assert gradients for all parameters, especially convolutional ones
    for param in local_state_network.parameters(include_all=True):
        assert param.grad is not None, f"{getattr(param, '_label', 'param')} missing grad"

def test_local_state_network_with_regularization_loss():
    """Test to ensure gradients of LocalStateNetwork parameters can be computed with regularization loss."""
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
        deploy_mode="modulated",
        return_package=True,
        lambda_reg=.5
    )

    metric_tensor = package["metric"]["g"]
    local_state_network = package["local_state_network"]
    package["regularization_loss"].backward()
    zero_w = AbstractTensor.zeros_like(local_state_network._weighted_padded)
    zero_m = AbstractTensor.zeros_like(local_state_network._modulated_padded)
    local_state_network.backward(zero_w, zero_m, lambda_reg=0.5)

    # Log gradient status for all parameters
    for param in local_state_network.parameters(include_all=True):
        shape = getattr(param, "shape", None)
        print(
            f"{getattr(param, '_label', 'param')} (shape={shape}): grad={'present' if param.grad is not None else 'missing'}"
        )


def test_regularization_loss_backward_registers_grads():
    """Backwarding the returned regularisation loss populates grads."""
    metric_tensor_func = lambda *args, **kwargs: None
    net = LocalStateNetwork(metric_tensor_func, (2, 2, 2), DEFAULT_CONFIGURATION, max_depth=1)
    padded_raw = AbstractTensor.randn((2, 2, 2, 3, 3, 3))
    net.forward(padded_raw, lambda_reg=0.5)
    zero = AbstractTensor.zeros_like(padded_raw)
    net.backward(zero, zero, lambda_reg=0.5)
    for p in net.parameters(include_all=True):
        assert p.grad is not None

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Run specific tests manually.")
    parser.add_argument("--test", type=str, choices=[
        "0",
        "1",
        "2",
        "3",
        "4"
    ], help="Specify the test to run.")

    args = parser.parse_args()

    if args.test == "0":
        test_laplace_gradient()
    elif args.test == "1":
        test_local_state_network_weighted_mode_gradient()
    elif args.test == "2":
        test_local_state_network_modulated_mode_gradient()
    elif args.test == "3":
        test_local_state_network_convolutional_modulator_gradient()
    elif args.test == "4":
        test_local_state_network_with_regularization_loss()
    else:
        print("Running all tests...")
        print("Starting test_laplace_gradient")
        print("We are expecting no gradients for the Laplace operator, because we're using 'raw' deploy mode, which bypasses the local state network.")
        test_laplace_gradient()
        print("Finished test_laplace_gradient")
        print("Starting test_local_state_network_weighted_mode_gradient")
        print("We are expecting gradients for 'g_weighted' in the weighted mode, as it utilizes the local state network's most basic level.")
        test_local_state_network_weighted_mode_gradient()
        print("Finished test_local_state_network_weighted_mode_gradient")
        print("Starting test_local_state_network_modulated_mode_gradient")
        print("We are expecting gradients for the inner linear layer in the modulated mode, as in this mode it is what's utilized")
        test_local_state_network_modulated_mode_gradient()
        print("Finished test_local_state_network_modulated_mode_gradient")
        print("Starting test_local_state_network_convolutional_modulator_gradient")
        print("We are expecting gradients for the convolutional layer in the convolutional modulator mode, as it is the primary component utilized.")
        test_local_state_network_convolutional_modulator_gradient()
        print("Finished test_local_state_network_convolutional_modulator_gradient")
        print("Starting test_local_state_network_with_regularization_loss")
        print("We are expecting gradients for all parameters, including those in the local state network, with the regularization loss affecting the gradient computation.")
        test_local_state_network_with_regularization_loss()
        print("Finished test_local_state_network_with_regularization_loss")