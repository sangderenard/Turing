import numpy as np
import numpy as np
import pytest
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_convolution.local_state_network import LocalStateNetwork, DEFAULT_CONFIGURATION


def dummy_metric(*args):
    return None, None, None


def test_local_state_network_forward_backward_consistency():
    np.random.seed(0)
    net = LocalStateNetwork(
        metric_tensor_func=dummy_metric,
        grid_shape=(1, 1, 1),
        switchboard_config=DEFAULT_CONFIGURATION,
        max_depth=1,
    )
    padded_raw_np = np.random.randn(1, 1, 1, 1, 3, 3, 3).astype(np.float32)
    padded_raw = AbstractTensor.get_tensor(padded_raw_np)

    weighted, modulated, _ = net.forward(padded_raw)

    grad_w_np = np.random.randn(*weighted.shape).astype(np.float32)
    grad_m_np = np.random.randn(*modulated.shape).astype(np.float32)
    grad_w = AbstractTensor.get_tensor(grad_w_np)
    grad_m = AbstractTensor.get_tensor(grad_m_np)

    # Expected gradient (manual computation)
    weight_layer = net.g_weight_layer.reshape((1, 1, 1, 1, 3, 3, 3))
    expected_from_weight = grad_w * weight_layer
    grad_m_view = grad_m.reshape((1, 1, 1, 1, -1))
    flat_grad = grad_m_view.reshape((-1, grad_m_view.shape[-1]))
    WT = net.spatial_layer.W.transpose(0, 1)
    grad_flat_in = flat_grad @ WT
    expected_from_mod = grad_flat_in.reshape((1, 1, 1, 1, 3, 3, 3))
    expected_grad = expected_from_weight + expected_from_mod

    grad_input = net.backward(grad_w, grad_m)

    assert np.allclose(grad_input.data, expected_grad.data, atol=1e-5)

    expected_g_weight = (grad_w * padded_raw).sum(dim=(0, 1, 2, 3))
    assert np.allclose(net.g_weight_layer.grad.data, expected_g_weight.data, atol=1e-5)


def identity_metric(u, v, w):
    eye = AbstractTensor.eye(3).reshape(1, 1, 1, 3, 3)
    det = AbstractTensor.ones((1, 1, 1))
    return eye, eye, det


def test_local_state_network_additional_params():
    net = LocalStateNetwork(
        metric_tensor_func=identity_metric,
        grid_shape=(1, 1, 1),
        switchboard_config=DEFAULT_CONFIGURATION,
        max_depth=1,
    )
    grid = AbstractTensor.zeros((1, 1, 1))
    tension = AbstractTensor.full((1, 1, 1), 2.0)
    density = AbstractTensor.full((1, 1, 1), 0.5)
    out = net(grid, grid, grid, partials=(), additional_params={"tension": tension, "density": density})
    padded = out["padded_raw"]
    assert padded[0, 0, 0, 2, 1, 1].item() == pytest.approx(2.0)
    assert padded[0, 0, 0, 2, 2, 2].item() == pytest.approx(0.5)


def test_zero_grad_preserves_parameters():
    net = LocalStateNetwork(
        metric_tensor_func=dummy_metric,
        grid_shape=(1, 1, 1),
        switchboard_config=DEFAULT_CONFIGURATION,
        max_depth=1,
    )
    initial = net.g_weight_layer.data.copy()
    (net.g_weight_layer * 2).sum().backward()
    assert net.g_weight_layer.grad is not None
    net.zero_grad()
    assert net.g_weight_layer.grad is None
    assert np.allclose(net.g_weight_layer.data, initial)


def test_regularization_produces_weight_gradient():
    """Gradient on g_weight_layer should be non-zero when lambda_reg > 0."""
    net = LocalStateNetwork(
        metric_tensor_func=dummy_metric,
        grid_shape=(1, 1, 1),
        switchboard_config=DEFAULT_CONFIGURATION,
        max_depth=1,
    )
    padded_raw = AbstractTensor.zeros((1, 1, 1, 1, 3, 3, 3))

    zeros = AbstractTensor.zeros_like(padded_raw)

    # Baseline with no regularisation
    net.forward(padded_raw, lambda_reg=0.0)
    net.backward(zeros, zeros, lambda_reg=0.0)
    baseline = net.g_weight_layer.grad.data.copy()

    net.zero_grad()

    # With regularisation enabled
    _, _, reg = net.forward(padded_raw, lambda_reg=0.5)
    net.backward(zeros, zeros, lambda_reg=0.5)
    grad = net.g_weight_layer.grad.data

    assert np.allclose(baseline, 0.0)
    assert not np.allclose(grad, 0.0)


def test_regularization_loss_backward_sets_grad():
    net = LocalStateNetwork(
        metric_tensor_func=dummy_metric,
        grid_shape=(1, 1, 1),
        switchboard_config=DEFAULT_CONFIGURATION,
        max_depth=1,
    )
    weighted = AbstractTensor.zeros((1, 1, 1, 1, 3, 3, 3))
    modulated = AbstractTensor.zeros_like(weighted)
    net.zero_grad()
    reg = net.regularization_loss(weighted, modulated)
    reg.backward()
    assert net.g_weight_layer._grad is not None


def test_wrapper_coordinates_gradients_with_regularization():
    """All parameters should receive gradients when lambda_reg is active."""
    net = LocalStateNetwork(
        metric_tensor_func=dummy_metric,
        grid_shape=(1, 1, 1),
        switchboard_config=DEFAULT_CONFIGURATION,
        max_depth=2,
    )
    padded_raw = AbstractTensor.ones((1, 1, 1, 1, 3, 3, 3))
    net.zero_grad()
    weighted, modulated, _ = net.forward(padded_raw, lambda_reg=0.5)
    grad_w = AbstractTensor.ones_like(weighted)
    grad_m = AbstractTensor.ones_like(modulated)
    net.backward(grad_w, grad_m, lambda_reg=0.5)
    for param in net.parameters(include_all=True, include_structural=True):
        assert getattr(param, "_grad", None) is not None


def test_recursive_backward_propagates_regularization():
    """Inner LocalStateNetwork layers receive regularisation gradients."""
    net = LocalStateNetwork(
        metric_tensor_func=dummy_metric,
        grid_shape=(1, 1, 1),
        switchboard_config=DEFAULT_CONFIGURATION,
        max_depth=2,
    )
    padded_raw = AbstractTensor.zeros((1, 1, 1, 1, 3, 3, 3))
    net.zero_grad()
    weighted, modulated, _ = net.forward(padded_raw, lambda_reg=0.5)
    grad_w = AbstractTensor.ones_like(weighted)
    grad_m = AbstractTensor.zeros_like(modulated)
    net.backward(grad_w, grad_m, lambda_reg=0.5)
    inner = net.inner_state
    assert inner.g_weight_layer.grad is not None
    assert inner.g_weight_layer.grad.abs().sum().item() > 0
    assert inner.g_bias_layer.grad is not None
    assert inner.g_bias_layer.grad.abs().sum().item() > 0
