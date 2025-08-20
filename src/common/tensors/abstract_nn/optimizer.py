from __future__ import annotations
from typing import List
from ..abstraction import AbstractTensor
from .utils import zeros_like


class Adam:
    """Simple Adam optimizer.

    The previous implementation keyed internal state by ``id(param)``.  Since
    model parameters are replaced with new tensor objects on every optimisation
    step, Python may recycle object IDs, leading to state being incorrectly
    reused across different parameters.  That manifested as biases suddenly
    growing to the wrong shape (e.g. ``(1, 8)`` instead of ``(1, 1)``) once an
    ID collision occurred.  We instead track state by parameter *index* in the
    list supplied to ``step`` which is stable across updates.
    """

    def __init__(
        self,
        params: List[AbstractTensor],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: List[AbstractTensor] = []
        self.v: List[AbstractTensor] = []
        self.t: int = 0
        self._init_params(params)

    def _init_params(self, params: List[AbstractTensor]):
        # Lazily extend state lists to match the number of parameters.
        while len(self.m) < len(params):
            p = params[len(self.m)]
            self.m.append(zeros_like(p))
            self.v.append(zeros_like(p))

    def step(self, params: List[AbstractTensor], grads: List[AbstractTensor]):
        self._init_params(params)
        self.t += 1
        lr, b1, b2, eps = self.lr, self.beta1, self.beta2, self.eps
        out_params: List[AbstractTensor] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            m = self.m[i]
            v = self.v[i]
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g * g)
            m_hat = m / (1.0 - (b1 ** self.t))
            v_hat = v / (1.0 - (b2 ** self.t))
            p = p - lr * m_hat / ((v_hat ** 0.5) + eps)
            self.m[i], self.v[i] = m, v
            out_params.append(p)
        return out_params
