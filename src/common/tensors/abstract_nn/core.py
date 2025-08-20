from __future__ import annotations
from typing import List
from ..abstraction import AbstractTensor
import random, math
from .utils import from_list_like, zeros_like, transpose2d
from .activations import Identity
from ..logger import get_tensors_logger

logger = get_tensors_logger()

def _randn_matrix(rows: int, cols: int, like: AbstractTensor, scale: float = 0.02) -> AbstractTensor:
    data = [[random.gauss(0.0, 1.0) * scale for _ in range(cols)] for _ in range(rows)]
    return from_list_like(data, like=like)

class Linear:
    def __init__(self, in_dim: int, out_dim: int, like: AbstractTensor, bias: bool = True, init: str = "auto_relu"):
        self.like = like
        if init == "he" or init == "auto_relu":
            scale = math.sqrt(2.0 / float(in_dim))
        elif init == "xavier":
            scale = math.sqrt(2.0 / float(in_dim + out_dim))
        else:
            scale = 0.02
        logger.debug(
            f"Linear layer init: in_dim={in_dim}, out_dim={out_dim}, bias={bias}, init={init}, scale={scale}"
        )
        self.W = _randn_matrix(in_dim, out_dim, like=like, scale=scale)
        self.b = from_list_like([[0.0] * out_dim], like=like) if bias else None
        logger.debug(
            f"Linear layer weights shape: {getattr(self.W, 'shape', None)}; bias shape: {getattr(self.b, 'shape', None) if self.b is not None else None}"
        )
        self.gW = zeros_like(self.W)
        self.gb = zeros_like(self.b) if self.b is not None else None
        self._x = None

    def parameters(self) -> List[AbstractTensor]:
        return [p for p in (self.W, self.b) if p is not None]

    def zero_grad(self):
        self.gW = zeros_like(self.W)
        if self.b is not None:
            self.gb = zeros_like(self.b)
        self._x = None

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        logger.debug(f"Linear.forward called with input shape: {getattr(x, 'shape', None)}")
        self._x = x
        out = x @ self.W
        logger.debug(f"Linear matmul output shape: {getattr(out, 'shape', None)}")
        if self.b is not None:
            b = self.b.broadcast_rows(out.shape[0], label="Linear.forward(bias)")
            logger.debug(f"Linear bias broadcasted shape: {getattr(b, 'shape', None)}")
            out = out + b
        return out


    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        if self._x is None:
            raise RuntimeError("Linear.backward called before forward; self._x is None")
        xT = self._x.transpose(0, 1)
        self.gW = xT @ grad_out
        if self.b is not None:
            gb_raw = grad_out.sum(dim=0, keepdim=True)
            self.gb = grad_out.ensure_tensor(gb_raw)
        WT = self.W.transpose(0, 1)
        grad_in = grad_out @ WT
        self._x = None
        return grad_in

class Model:
    def __init__(self, layers: List[Linear], activations) -> None:
        logger.debug(f"Model init with {len(layers)} layers and activations {activations}")
        self.layers = layers
        if isinstance(activations, list):
            assert len(activations) == len(layers), "activations list must match layers"
            self.activations = activations
        else:
            self.activations = [activations] * len(layers)
        self._pre = [None] * len(layers)
        self._post = [None] * len(layers)

    def parameters(self) -> List[AbstractTensor]:
        ps: List[AbstractTensor] = []
        for layer in self.layers:
            ps.extend(layer.parameters())
        return ps

    def grads(self) -> List[AbstractTensor]:
        gs: List[AbstractTensor] = []
        for l in self.layers:
            gs.append(l.gW)
            if l.b is not None:
                gs.append(l.gb)
        return gs

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        logger.debug(f"Model.forward called with input shape: {getattr(x, 'shape', None)}")
        for i, layer in enumerate(self.layers):
            logger.debug(f"Model.forward: passing through layer {i} ({layer})")
            z = layer.forward(x)
            self._pre[i] = z
            act = self.activations[i]
            x = act.forward(z) if act is not None else z
            self._post[i] = x
            logger.debug(f"Model.forward: after activation, shape: {getattr(x, 'shape', None)}")
        return x

    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        g = grad_out
        for i in reversed(range(len(self.layers))):
            act = self.activations[i]
            if act is not None:
                g = act.backward(self._post[i], g)
            g = self.layers[i].backward(g)
        return g

class Sequential(Model):
    pass
