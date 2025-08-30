import pytest
from src.common.tensors.abstract_nn.linear_block import LinearBlock
from src.common.tensors.abstraction import AbstractTensor as AT

def test_linear_block_debug():
    input_dim = 30
    hidden_dim = 64
    output_dim = 30
    like = AT.get_tensor()  # Ensure `like` is an instance, not the class itself

    # Initialize the model
    model = LinearBlock(input_dim, hidden_dim, output_dim, like)

    # Debug: Print model parameters
    params = model.parameters()
    print("Model parameters:")
    for i, p in enumerate(params):
        print(f"Param {i}: shape={getattr(p, 'shape', None)}, requires_grad={getattr(p, 'requires_grad', None)}")

    # Create dummy data
    inputs = AT.randn((10, input_dim), requires_grad=True)  # Batch size of 10
    targets = AT.ones((10, output_dim), requires_grad=True) * 0.5

    # Forward pass
    outputs = model.forward(inputs)

    # Debug: Check outputs
    print("Outputs:")
    print(outputs)

    # Compute loss
    loss_fn = lambda pred, target: ((pred - target) ** 2).mean()
    loss = loss_fn(outputs, targets)

    # Debug: Check loss properties before backward
    print("Loss properties before backward:")
    print("Loss value: ",loss)
    print(f"Loss shape: {getattr(loss, 'shape', None)}")
    print(f"Loss requires_grad: {getattr(loss, 'requires_grad', None)}")

    # Backward pass
    loss.backward()

    # Debug: Check gradients after backward
    print("Gradients after backward:")
    params = list(model.parameters())
    grads = [getattr(p, "grad", None) for p in params]
    for i, (p, g) in enumerate(zip(params, grads)):
        label = getattr(p, "_label", None)
        f"Param {i}: label={label}, shape={getattr(p, 'shape', None)}, grad is None={g is None}, grad shape={getattr(g, 'shape', None) if g is not None else None}"

    # Assert that no gradients are None
    assert all(p.grad is not None for p in params), "Some gradients are None"
