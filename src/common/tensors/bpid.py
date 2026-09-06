"""Broadcast PID controller for elementwise tensor updates.

This module provides the :class:`BPID` class which maintains a per-parameter
proportional–integral–derivative controller.  The controller operates elementwise
on :class:`AbstractTensor` gradients and returns adjusted gradients that can be
fed into a simple optimizer such as SGD.

The implementation stores integral and previous-error state for each parameter
using its monotonic tensor identity, so list reconstruction and Python address
reuse cannot misassociate controller state.
"""

from __future__ import annotations

from typing import Dict, List

from .abstraction import AbstractTensor as AT, tensor_identity
from .abstract_nn.utils import zeros_like


class BPID:
    """Broadcast PID controller for tensors.

    Parameters are tracked by monotonic tensor identity so state remains
    consistent even when the parameter list is re-created between steps.
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self._integral: Dict[int, AT] = {}
        self._prev_error: Dict[int, AT] = {}

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._integral.clear()
        self._prev_error.clear()

    def _init_params(self, params: List[AT]) -> None:
        for p in params:
            key = tensor_identity(p)
            if key not in self._integral or self._integral[key].shape != p.shape:
                self._integral[key] = zeros_like(p)
                self._prev_error[key] = zeros_like(p)

    def step(self, params: List[AT], grads: List[AT]) -> List[AT]:
        """Process gradients through the PID controller.

        Parameters
        ----------
        params:
            List of parameter tensors corresponding to ``grads``.
        grads:
            Raw gradient tensors (error terms).
        """
        self._init_params(params)
        kp, ki, kd = self.kp, self.ki, self.kd
        out: List[AT] = []
        for p, g in zip(params, grads):
            key = tensor_identity(p)
            integ = self._integral[key] + g
            deriv = g - self._prev_error[key]
            adjusted = kp * g + ki * integ + kd * deriv
            self._integral[key] = integ
            self._prev_error[key] = g
            out.append(adjusted)
        return out


__all__ = ["BPID"]
