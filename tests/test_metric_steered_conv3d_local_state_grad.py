import pytest
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_convolution.metric_steered_conv3d import MetricSteeredConv3DWrapper
from src.common.tensors.abstract_convolution.laplace_nd import RectangularTransform


@pytest.mark.parametrize("deploy_mode", ["weighted", "modulated"])
def test_local_state_network_params_receive_grads(deploy_mode):
    N = 2
    transform = RectangularTransform(Lx=1.0, Ly=1.0, Lz=1.0, device="cpu")
    layer = MetricSteeredConv3DWrapper(1, 1, (N, N, N), transform, deploy_mode=deploy_mode)
    x = AbstractTensor.randn((1, 1, N, N, N))
    x.requires_grad_(True)
    out = layer.forward(x)
    lsn = layer.laplace_package['local_state_network']
    loss = out.sum()
    if deploy_mode == "weighted":
        loss = loss + lsn.g_weight_layer.sum() + lsn.g_bias_layer.sum()
    else:  # modulated
        reg = None
        for p in lsn.inner_state.spatial_layer.parameters():
            reg = p.sum() if reg is None else reg + p.sum()
        loss = loss + reg
    loss.backward()
    if deploy_mode == "weighted":
        assert getattr(lsn.g_weight_layer, "_grad", None) is not None
        assert getattr(lsn.g_bias_layer, "_grad", None) is not None
    else:  # modulated
        spatial_params = lsn.inner_state.spatial_layer.parameters()
        assert all(getattr(p, "_grad", None) is not None for p in spatial_params)
        assert getattr(lsn.g_weight_layer, "_grad", None) is None
        assert getattr(lsn.g_bias_layer, "_grad", None) is None


def test_local_state_network_params_receive_grads_raw_mode():
    N = 2
    transform = RectangularTransform(Lx=1.0, Ly=1.0, Lz=1.0, device="cpu")
    layer = MetricSteeredConv3DWrapper(
        1,
        1,
        (N, N, N),
        transform,
        deploy_mode="raw",
        laplace_kwargs={"lambda_reg": 0.5},
    )
    x = AbstractTensor.randn((1, 1, N, N, N))
    layer.forward(x)
    lsn = layer.laplace_package["local_state_network"]
    layer.laplace_package["regularization_loss"].backward()
    zero_w = AbstractTensor.zeros_like(lsn._weighted_padded)
    zero_m = AbstractTensor.zeros_like(lsn._modulated_padded)
    lsn.backward(zero_w, zero_m, lambda_reg=0.5)
    params = lsn.parameters(include_all=True, include_structural=True)
    assert all(getattr(p, "_grad", None) is not None for p in params)
