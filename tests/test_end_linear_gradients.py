import pytest
from src.common.tensors.autograd import autograd, GradTape
from src.common.tensors.numpy_backend import NumPyTensorOperations as Tensor


def test_end_linear_params_receive_gradients():
    autograd.tape = GradTape()
    W = Tensor.tensor([[0.5], [-0.3], [0.1]])
    W.requires_grad_(True)
    autograd.tape.create_tensor_node(W)
    b = Tensor.tensor([[0.0]])
    b.requires_grad_(True)
    autograd.tape.create_tensor_node(b)
    x = Tensor.tensor([[0.2, -0.1, 0.4]])
    x.requires_grad_(True)
    y = x @ W + b
    target = Tensor.tensor([[0.0]])
    loss = ((y - target) ** 2).mean()
    loss.backward()
    assert getattr(W, "_grad", None) is not None
    assert getattr(b, "_grad", None) is not None
