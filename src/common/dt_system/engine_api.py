# -*- coding: utf-8 -*-
"""Engine compatibility API for dt-graph runner.

Engines integrate by subclassing DtCompatibleEngine and implementing:
- step(self, dt) -> tuple[bool, Metrics]
- get_metrics(self) -> Metrics  (optional; runner uses step's return path)
- preferred_dt(self) -> float    (optional; controller can supersede)

They also declare their metric targets and may provide a distribution hook to
nonlinearize dt proposals relative to metric ranges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .dt_scaler import Metrics
from .dt_controller import Targets, STController
from .dt_solver import BisectSolverConfig


DistributionFn = Callable[[Metrics, Targets, float], float]
"""Map (metrics, targets, dx) -> dt_penalty (smaller is stricter).

If omitted, the default penalty is computed from CFL and error ratios.
"""


class DtCompatibleEngine:
    """Base compatibility shim engines should extend.

    Engines MUST be able to accept an external state and return the same
    shape/state object after stepping. To enable a staged rollout, the
    legacy ``step(dt)`` remains, and a new ``step_with_state(state, dt)``
    default implementation delegates to ``step`` and returns the input state
    unchanged. Compliant engines should override ``step_with_state``.
    """

    def step(self, dt: float) -> tuple[bool, Metrics]:  # pragma: no cover - interface
        raise NotImplementedError

    # New required capability for compliant engines
    def step_with_state(self, state: object, dt: float) -> tuple[bool, Metrics, object]:  # pragma: no cover - default bridge
        ok, m = self.step(float(dt))
        return ok, m, state

    def preferred_dt(self) -> Optional[float]:  # pragma: no cover - optional
        return None

    def get_metrics(self) -> Optional[Metrics]:  # pragma: no cover - optional
        return None


@dataclass
class EngineRegistration:
    name: str
    engine: DtCompatibleEngine
    targets: Targets
    dx: float
    distribution: Optional[DistributionFn] = None
    # Optional per-engine controller; if None, a copy of the parent ctrl will be used
    ctrl: Optional[STController] = None
    # When True, the GraphBuilder nests this engine in a per-engine RoundNode so it can
    # negotiate its local dt (postscriptive) within the parent slice.
    localize: bool = True
    # Optional opt-in dt solver that replaces the controller for this engine.
    # When provided, GraphBuilder will wrap the engine advance leaf with a RoundNode
    # that uses a bisection solver to consume the parent slice precisely.
    solver_config: Optional[BisectSolverConfig] = None


__all__ = [
    "DtCompatibleEngine",
    "EngineRegistration",
    "DistributionFn",
]
