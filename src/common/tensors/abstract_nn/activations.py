from __future__ import annotations
from ..abstraction import AbstractTensor
from .utils import map_unary

class Activation:
    def __call__(self, x: AbstractTensor) -> AbstractTensor:
        return self.forward(x)
    def forward(self, x: AbstractTensor) -> AbstractTensor: ...
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor: ...

class Identity(Activation):
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        return x
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        return grad_out

class ReLU(Activation):
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        mask = x.less(0.0)
        return (1 - mask) * x
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        mask = x.less(0.0)
        return (1 - mask) * grad_out

class Sigmoid(Activation):
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        # Sigmoid: 1 / (1 + exp(-x))
        return 1.0 / (1.0 + (-x).exp())
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        y = self.forward(x)
        return grad_out * y * (1 - y)

class Tanh(Activation):
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        # Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        ex = x.exp()
        enx = (-x).exp()
        return (ex - enx) / (ex + enx)
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        y = self.forward(x)
        return grad_out * (1 - (y * y))
