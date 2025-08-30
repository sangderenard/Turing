from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_convolution.metric_steered_conv3d import MetricSteeredConv3DWrapper
from src.common.tensors.abstract_convolution.laplace_nd import RectangularTransform
from src.common.tensors.autograd import autograd


def test_raw_mode_excludes_local_state_network_params():
    N = 2
    transform = RectangularTransform(Lx=1.0, Ly=1.0, Lz=1.0, device="cpu")
    layer = MetricSteeredConv3DWrapper(1, 1, (N, N, N), transform, deploy_mode="raw")
    x = AbstractTensor.randn((1, 1, N, N, N))
    out = layer.forward(x)
    lsn = layer.laplace_package["local_state_network"]
    lsn_ids = {id(p) for p in lsn.parameters(include_all=True, include_structural=True)}
    layer_ids = {id(p) for p in layer.parameters()}
    assert lsn_ids.isdisjoint(layer_ids)
    tape_ids = {id(p) for p in autograd.tape.parameter_tensors()}
    assert lsn_ids.isdisjoint(tape_ids)
