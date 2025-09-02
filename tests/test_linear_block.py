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

    # Gradients should propagate to all Linear layers
    for layer in model.model.layers:
        assert getattr(layer, "gW", None) is not None
        if layer.b is not None:
            assert getattr(layer, "gb", None) is not None


def test_linear_block_invalid_shape():
    input_dim = 4
    output_dim = 2
    like = AT.get_tensor()

    model = LinearBlock(input_dim, output_dim, like)

    bad_input = AT.randn((3, input_dim + 1))

    with pytest.raises(ValueError):
        model.forward(bad_input)


def test_linear_block_last_axis_features_grad():
    input_dim = 6
    output_dim = 4
    like = AT.get_tensor()
    model = LinearBlock(input_dim, output_dim, like)

    inputs = AT.randn((2, 3, input_dim), requires_grad=True)
    targets = AT.zeros((2, 3, output_dim))

    outputs = model.forward(inputs)
    assert outputs.shape == (2, 3, output_dim)
    loss = outputs.sum()
    loss.backward()

    for layer in model.model.layers:
        assert getattr(layer, "gW", None) is not None
        if layer.b is not None:
            assert getattr(layer, "gb", None) is not None


def test_linear_block_channels_first_grad():
    input_dim = 5
    output_dim = 7
    like = AT.get_tensor()
    model = LinearBlock(input_dim, output_dim, like)

    inputs = AT.randn((4, input_dim, 2, 2), requires_grad=True)
    spatial = 2 * 2
    targets = AT.ones((4, output_dim * spatial))

    outputs = model.forward(inputs)
    assert outputs.shape == (4, output_dim * spatial)
    loss = ((outputs - targets) ** 2).sum()
    loss.backward()

    for layer in model.model.layers:
        assert getattr(layer, "gW", None) is not None
        if layer.b is not None:
            assert getattr(layer, "gb", None) is not None
