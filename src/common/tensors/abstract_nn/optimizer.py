from __future__ import annotations
from typing import List, Tuple
from ..abstraction import AbstractTensor as AT
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
        self.m: List[AT] = []
        self.v: List[AT] = []
        self.t: int = 0
        self._init_params(params)

    def _init_params(self, params: List[AT]):
        # Lazily extend state lists to match the number of parameters.
        while len(self.m) < len(params):
            p = params[len(self.m)]
            self.m.append(zeros_like(p))
            self.v.append(zeros_like(p))

    def step(self, params: List[AT], grads: List[AT]):
        self._init_params(params)
        self.t += 1
        lr, b1, b2, eps = self.lr, self.beta1, self.beta2, self.eps
        out_params: List[AT] = []
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
