import numpy as np
from src.common.tensors.numpy_backend import NumPyTensorOperations as T
from src.common.tensors.abstraction import AbstractTensor


def test_grad_attribute():
    x = T.tensor([[1.0, 2.0], [3.0, 4.0]])
    x.requires_grad_(True)
    (x * x).sum().backward()
    assert x.grad is not None


def test_autograd_computes_when_unset():
    x = T.tensor([1.0, 2.0])
    x.requires_grad_(True)
    loss = (x * x).sum()
    x.autograd.tape.mark_loss(loss)
    assert x.grad is None
    AbstractTensor.autograd.grad(loss, [x])
    expected = (x * 2).data
    assert x.grad.data.tolist() == expected.tolist()


def test_zero_grad_resets():
    x = T.tensor([1.0, 2.0])
    x.requires_grad_(True)
    (x * x).sum().backward()
    assert x.grad is not None
    x.zero_grad()
    assert x.grad is None
