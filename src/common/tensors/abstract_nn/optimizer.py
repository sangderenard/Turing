from __future__ import annotations
from typing import Dict, List, Tuple
from ..abstraction import AbstractTensor as AT
from .utils import zeros_like


class Adam:
    """Simple Adam optimizer.

    The previous implementation keyed internal state by parameter index.  Some
    calling code, however, generates a fresh parameter list each step and may
    reorder entries.  Index-based bookkeeping would then associate the wrong
    momentum/variance tensors with a parameter, causing shape mismatches.  We
    now map state by the ``id`` of each parameter while retaining a strong
    reference to avoid ID reuse once a tensor is garbage collected.
    """

    def __init__(
        self,
        params: List[AT],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        # Track optimizer state per parameter ID to remain robust even if the
        # caller reorders the parameter list between steps.  We keep a strong
        # reference to each parameter to avoid ``id`` reuse once an object is
        # garbage‑collected.
        self.m: Dict[int, AT] = {}
        self.v: Dict[int, AT] = {}
        self._param_refs: Dict[int, AT] = {}
        self.t: int = 0
        self._init_params(params)

    def _init_params(self, params: List[AT]):
        for p in params:
            key = id(p)
            if key not in self.m or self.m[key].shape != p.shape:
                self.m[key] = zeros_like(p)
                self.v[key] = zeros_like(p)
                self._param_refs[key] = p

    def step(self, params: List[AT], grads: List[AT]):
        self._init_params(params)
        self.t += 1
        lr, b1, b2, eps = self.lr, self.beta1, self.beta2, self.eps
        out_params: List[AT] = []
        for p, g in zip(params, grads):
            key = id(p)
            m = self.m[key]
            v = self.v[key]
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g * g)
            m_hat = m / (1.0 - (b1 ** self.t))
            v_hat = v / (1.0 - (b2 ** self.t))
            p = p - lr * m_hat / ((v_hat ** 0.5) + eps)
            self.m[key], self.v[key] = m, v
            out_params.append(p)
        return out_params


def adam_step(
    p: AT,
    g: AT,
    m: AT,
    v: AT,
    t: AT,
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[AT, AT, AT, AT]:
    """Pure functional, fully-recordable Adam update for a single parameter tensor.

    Inputs are AbstractTensors and the update is expressed entirely via
    AbstractTensor ops so it records on the current tape. Returns
    (p_new, m_new, v_new, t_new).
    """
    # increment step (scalar tensor)
    t_new = t + 1.0
    b1 = float(beta1)
    b2 = float(beta2)
    m_new = b1 * m + (1.0 - b1) * g
    v_new = b2 * v + (1.0 - b2) * (g * g)
    m_hat = m_new / (1.0 - (beta1 ** t_new))
    v_hat = v_new / (1.0 - (beta2 ** t_new))
    denom = (v_hat ** 0.5) + float(eps)
    p_new = p - float(lr) * (m_hat / denom)
    return p_new, m_new, v_new, t_new


__all__ = ["Adam", "adam_step"]
