from __future__ import annotations
from typing import List, Dict
from ..abstraction import AbstractTensor
from .utils import zeros_like
class Adam:
    def __init__(self, params: List[AbstractTensor], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Dict[int, AbstractTensor] = {}
        self.v: Dict[int, AbstractTensor] = {}
        self.t: int = 0
        self._init_params(params)

    def _init_params(self, params: List[AbstractTensor]):
        for p in params:
            pid = id(p)
            if pid not in self.m:
                self.m[pid] = zeros_like(p)
                self.v[pid] = zeros_like(p)

    def step(self, params: List[AbstractTensor], grads: List[AbstractTensor]):
        self._init_params(params)
        self.t += 1
        lr, b1, b2, eps = self.lr, self.beta1, self.beta2, self.eps
        out_params = []
        for p, g in zip(params, grads):
            pid = id(p)
            m = self.m[pid]
            v = self.v[pid]
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g * g)
            m_hat = (1.0 / (1.0 - (b1 ** self.t))) * m
            v_hat = (1.0 / (1.0 - (b2 ** self.t))) * v
            p = p - lr * m_hat / ((v_hat ** 0.5) + eps)
            self.m[pid], self.v[pid] = m, v
            out_params.append(p)
        return out_params
