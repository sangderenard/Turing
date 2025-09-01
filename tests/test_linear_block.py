import pytest
from src.common.tensors.abstract_nn.linear_block import LinearBlock
from src.common.tensors.abstraction import AbstractTensor as AT


def test_linear_block_debug():
    input_dim = 30
    output_dim = 12
    like = AT.get_tensor()

    model = LinearBlock(input_dim, output_dim, like)

    params = list(model.parameters())
    expected_hidden = int((input_dim + output_dim) / 2)
    # first weight
    assert params[0].shape == (input_dim, expected_hidden)
    # last weight
    assert params[-2].shape == (expected_hidden, output_dim)

    inputs = AT.randn((10, input_dim), requires_grad=True)
    targets = AT.ones((10, output_dim), requires_grad=True) * 0.5

    outputs = model.forward(inputs)
    loss = ((outputs - targets) ** 2).mean()
    loss.backward()
