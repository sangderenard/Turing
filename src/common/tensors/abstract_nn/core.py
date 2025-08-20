from __future__ import annotations
from typing import List
from ..abstraction import AbstractTensor
import random, math
from .utils import from_list_like, zeros_like, transpose2d
from .activations import Identity

def _randn_matrix(rows: int, cols: int, like: AbstractTensor, scale: float = 0.02) -> AbstractTensor:
    data = [[random.gauss(0.0, 1.0) * scale for _ in range(cols)] for _ in range(rows)]
    return from_list_like(data, like=like)

class Linear:
    def __init__(self, in_dim: int, out_dim: int, like: AbstractTensor, bias: bool = True, init: str = "xavier"):
        self.like = like
        scale = math.sqrt(2.0 / float(in_dim + out_dim)) if init == "xavier" else 0.02
        self.W = _randn_matrix(in_dim, out_dim, like=like, scale=scale)
        self.b = from_list_like([[0.0 for _ in range(out_dim)]], like=like) if bias else None
        self.gW = zeros_like(self.W)
        self.gb = zeros_like(self.b) if self.b is not None else None
        self._x = None

    def parameters(self) -> List[AbstractTensor]:
        return [p for p in (self.W, self.b) if p is not None]

    def zero_grad(self):
        self.gW = zeros_like(self.W)
        if self.b is not None:
            self.gb = zeros_like(self.b)

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        self._x = x
        out = x @ self.W
        if self.b is not None:
            b = self.b.broadcast_rows(out.shape[0])
            out = out + b
        return out

    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        xT = self._x.transpose(0, 1)
        self.gW = xT @ grad_out
        if self.b is not None:
            gb_raw = grad_out.sum(dim=0, keepdim=True)
            self.gb = grad_out.ensure_tensor(gb_raw)
        WT = self.W.transpose(0, 1)
        grad_in = grad_out @ WT
        return grad_in

class Model:
    def __init__(self, layers: List[Linear], activation) -> None:
        self.layers = layers
        self.activation = activation if activation is not None else Identity()
        self._pre = []
        self._post = []

    def parameters(self) -> List[AbstractTensor]:
        ps: List[AbstractTensor] = []
        for layer in self.layers:
            ps.extend(layer.parameters())
        return ps

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        self._pre, self._post = [], []
        for layer in self.layers:
            z = layer.forward(x)
            self._pre.append(z)
            x = self.activation(z)
            self._post.append(x)
        return x

    def backward(self, grad_out: AbstractTensor) -> None:
        grads = grad_out
        for i in reversed(range(len(self.layers))):
            z = self._pre[i]
            grads = self.activation.backward(z, grads)
            grads = self.layers[i].backward(grads)

class Sequential(Model):
    pass
