# -*- coding: utf-8 -*-
"""Hierarchical timestep control API.

Each simulation component maintains its own adaptive controller. A parent
regulator proposes a timestep based on its metric and instructs its
subordinates to advance by that amount. Subordinates may satisfy the
request by executing multiple micro-steps using their own controller. The
process is strictly top-down—there is **no global negotiation** of the
step size.

The :class:`~src.cells.bath.dt_controller.STController` implements this
policy. Components that wish to expose a uniform interface can conform to
:class:`DtStepper`, which accepts a :class:`~src.common.dt.SuperstepPlan`
and returns a :class:`~src.common.dt.SuperstepResult` after covering the
requested window.
"""

from __future__ import annotations

from typing import Protocol

from .dt import SuperstepPlan, SuperstepResult


class DtStepper(Protocol):
    """Protocol for objects that advance via supersteps.

    Implementors accept a :class:`SuperstepPlan` describing the outer
    timestep request and perform as many internal micro-steps as needed.
    The outcome is reported as a :class:`SuperstepResult`.
    """

    def superstep(self, plan: SuperstepPlan) -> SuperstepResult:  # pragma: no cover - protocol
        """Advance according to ``plan`` and report the result."""


__all__ = ["DtStepper", "SuperstepPlan", "SuperstepResult"]
