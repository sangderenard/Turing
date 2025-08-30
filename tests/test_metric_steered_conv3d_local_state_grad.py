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
    loss = out.sum() + lsn.g_weight_layer.sum()
    loss.backward()
    assert any(p.grad is not None for p in lsn.parameters())
