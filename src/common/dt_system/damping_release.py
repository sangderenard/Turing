"""General damping release: start a core over-damped, relax to its physics.

A pneumatic or hydraulic core (a membrane under gas pressure, a fluid
column, a hydraulic follower) usually begins a run away from equilibrium:
the authored rest geometry is not the solved equilibrium of the loaded
system, so the first steps carry a violent transient that either demands
tiny steps or blows the explicit integrator up.  The classic remedy is to
begin with damping scaled up by a large factor and release it toward the
physical value on a schedule, so the transient is absorbed while the
system settles, after which the real dynamics run untouched.

This module makes that a dt-system wrapper around ANY core, applied
through the core's own damping inputs by a caller-supplied ``apply``:

    state = DampingReleasedState(core, apply=lambda c, f: ..., initial_factor=20.0,
                                 release_time_s=0.05)
    run_superstep(state, ..., advance=release_advance(advance), ...)

``apply(core, factor)`` must set the core's damping to ``factor`` times its
physical value (idempotent: it always scales from the physical base, never
from the last factor).  The release clock lives in the state snapshot, so a
rolled-back step also rolls the schedule back.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from .dt_scaler import coerce_metrics


@dataclass
class DampingReleasedState:
    core: Any
    apply: Callable[[Any, float], None]
    initial_factor: float = 10.0
    release_time_s: float = 0.05
    curve: str = "exponential"      # or "linear"
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_factor < 1.0:
            raise ValueError("initial_factor must be >= 1 (release goes DOWN to the physics)")
        if self.release_time_s <= 0.0:
            raise ValueError("release_time_s must be positive")
        if self.curve not in {"exponential", "linear"}:
            raise ValueError(f"unknown release curve {self.curve!r}")
        self.apply(self.core, self.factor())

    def factor(self) -> float:
        """Damping multiplier at the current release clock (>= 1, -> 1)."""

        excess = self.initial_factor - 1.0
        if self.curve == "linear":
            remaining = max(0.0, 1.0 - self.elapsed_s / self.release_time_s)
            return 1.0 + excess * remaining
        # Exponential: three time constants inside release_time_s brings the
        # excess to under 5 %; the tail continues to shrink monotonically.
        return 1.0 + excess * math.exp(-3.0 * self.elapsed_s / self.release_time_s)

    def released(self, tolerance: float = 1.0e-3) -> bool:
        return self.factor() - 1.0 <= tolerance

    # -- dt-system state contract ---------------------------------------------
    def copy_shallow(self):
        return (self.core.copy_shallow(), self.elapsed_s)

    def restore(self, snapshot) -> None:
        core_snapshot, elapsed = snapshot
        self.core.restore(core_snapshot)
        self.elapsed_s = float(elapsed)
        self.apply(self.core, self.factor())

    def dt_limit_hint(self):
        hint = getattr(self.core, "dt_limit_hint", None)
        return hint() if callable(hint) else None

    def __getattr__(self, name: str):
        return getattr(self.core, name)


def release_advance(advance: Callable[[Any, Any], tuple[Any, Any]]):
    """Wrap ``advance(core, dt)`` so each step runs at the scheduled damping."""

    def advance_released(state: DampingReleasedState, dt):
        factor = state.factor()
        state.apply(state.core, factor)
        ok, metrics = advance(state.core, dt)
        state.elapsed_s += float(dt)
        metrics = coerce_metrics(metrics)
        channels = dict(metrics.error_channels or {})
        channels["damping_factor"] = float(factor)
        metrics.error_channels = channels
        return ok, metrics

    return advance_released


__all__ = ["DampingReleasedState", "release_advance"]
