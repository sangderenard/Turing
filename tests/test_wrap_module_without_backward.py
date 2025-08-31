import numpy as np
from src.common.tensors import AbstractTensor
from src.common.tensors.abstract_nn import wrap_module
from src.common.tensors.backward import BACKWARD_REGISTRY


class NoBackwardModule:
    def forward(self, x):
        return x * x


def test_wrap_module_skips_backward_registration_and_allows_grad():
    autograd = AbstractTensor.autograd
    autograd.tape._nodes.clear()

    m = NoBackwardModule()
    wrap_module(m)

    name = m.__class__.__name__
    assert f"{name}.forward" not in BACKWARD_REGISTRY._methods
    assert f"{name}.__call__" not in BACKWARD_REGISTRY._methods

    x = AbstractTensor.tensor([2.0], requires_grad=True)
    x.requires_grad_(True)
    y = m.forward(x)
    assert len(autograd.tape._nodes) > 0


class BaseWithBackward:
    def forward(self, x):
        return x * x

    def backward(self, grad):
        return grad


class InheritsBackward(BaseWithBackward):
    pass


def test_wrap_module_skips_inherited_backward_method():
    autograd = AbstractTensor.autograd
    autograd.tape._nodes.clear()

    m = InheritsBackward()
    wrap_module(m)

    name = m.__class__.__name__
    assert f"{name}.forward" not in BACKWARD_REGISTRY._methods
    assert f"{name}.__call__" not in BACKWARD_REGISTRY._methods

    x = AbstractTensor.tensor([3.0], requires_grad=True)
    x.requires_grad_(True)
    y = m.forward(x)
    assert len(autograd.tape._nodes) > 0
