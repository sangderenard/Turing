import pytest
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_convolution.metric_steered_conv3d import MetricSteeredConv3DWrapper
from src.common.tensors.abstract_convolution.laplace_nd import RectangularTransform


def test_local_state_network_params_receive_grads():
    N = 2
    transform = RectangularTransform(Lx=1.0, Ly=1.0, Lz=1.0, device="cpu")
    layer = MetricSteeredConv3DWrapper(1, 1, (N, N, N), transform, laplace_kwargs={"deploy_mode": "weighted"})
    x = AbstractTensor.randn((1, 1, N, N, N))
    x.requires_grad_(True)
    out = layer.forward(x)
    lsn = layer.laplace_package['local_state_network']
    loss = out.sum() + lsn.weight_layer.sum()
    loss.backward()
    for p in lsn.parameters():
        assert p.grad is not None
