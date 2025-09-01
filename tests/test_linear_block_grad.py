from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_nn.linear_block import LinearBlock
from src.common.tensors.autograd import autograd, GradTape


def test_linear_block_parameters_track_gradients():
    autograd.tape = GradTape()
    like = AbstractTensor.get_tensor(0)
    block = LinearBlock(4, 2, like)
    x = AbstractTensor.randn((3, 4), requires_grad=True)
    y = block.forward(x)
    loss = (y * y).sum()
    params = list(block.parameters())
    grads = autograd.grad(loss, params, allow_unused=True)
    assert grads[0] is not None
