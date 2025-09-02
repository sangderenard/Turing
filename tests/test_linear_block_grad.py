from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_nn.linear_block import LinearBlock
from src.common.tensors.autograd import autograd


def test_linear_block_parameters_track_gradients():
    like = AbstractTensor.get_tensor(0)
    block = LinearBlock(4, 2, like)
    params = list(block.parameters())
    for p in params:
        autograd.tape.create_tensor_node(p)
    x = AbstractTensor.randn((3, 4), requires_grad=True)
    y = block.forward(x)
    loss = (y * y).sum()
    loss.backward()
    for layer in block.model.layers:
        assert getattr(layer, "gW", None) is not None
        if layer.b is not None:
            assert getattr(layer, "gb", None) is not None
