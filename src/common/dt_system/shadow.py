"""Shadow-trajectory amplification: a predictive stability metric for any core.

A stateful core is advanced twice per step: its real state and a *shadow*
copy carrying a small finite perturbation.  The ratio by which the
perturbation grew during the step is the measured amplification of the step
map along the most unstable direction the perturbation has aligned with (the
perturbation is renormalised after every step, the classic Benettin
construction, so it keeps tracking the dominant growth without ever being
large enough to leave the linear regime).

Why a finite shadow and not a Jacobian: the tire's step map is not usefully
linearisable at rest (normalisations such as ``sqrt(v*v + 1e-30)`` have
enormous local derivatives near zero velocity), so autograd and
finite-difference power iterations disagreed by orders of magnitude while a
finite perturbation propagated through real steps decayed monotonically.  The
shadow costs exactly one extra forward per step, needs no backward graph, and
measures the thing that actually precedes a blow-up.

The wrapper publishes the growth factor as the error channel
``shadow_growth`` and the controller (``Targets.shadow_growth_max``) pins the
next attempt so a step may amplify by at most that factor, assuming the
amplification is exponential in ``dt`` (``growth = exp(rate * dt)``):
``dt_next <= dt * ln(growth_max) / ln(growth)`` whenever ``growth > 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from ..tensors import AbstractTensor
from .dt_scaler import Metrics, coerce_metrics


def _flatten(value: Any) -> np.ndarray:
    data = getattr(value, "data", value)
    return np.asarray(data, dtype=np.float64).reshape(-1)


@dataclass
class ShadowedState:
    """A core plus a perturbed shadow copy of it, one dt-system state.

    ``core`` must provide ``copy_shallow``/``restore`` (the rollback contract)
    and an attribute or callable ``state_vector`` access through
    ``read``/``write``: ``read(core) -> array-like`` returns the evolving
    state and ``write(core, array)`` stores it.  The shadow is a second
    ``copy_shallow`` of the core whose state has been displaced by a
    perturbation of relative size ``delta``.
    """

    core: Any
    read: Callable[[Any], Any]
    write: Callable[[Any, np.ndarray], None]
    make_shadow: Callable[[Any], Any]
    delta: float = 1.0e-6
    shadow: Any = None
    perturbation_norm: float = 0.0
    last_growth: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.shadow is None:
            self.reseed()

    # -- perturbation bookkeeping --------------------------------------------
    def _scale(self) -> float:
        base = _flatten(self.read(self.core))
        return max(float(np.linalg.norm(base)), 1.0) * float(self.delta)

    def reseed(self) -> None:
        """Rebuild the shadow as the core displaced by a fresh random direction."""

        self.shadow = self.make_shadow(self.core)
        base = _flatten(self.read(self.core))
        direction = np.random.default_rng(self.seed).standard_normal(base.shape)
        direction /= max(float(np.linalg.norm(direction)), 1.0e-300)
        self.perturbation_norm = self._scale()
        self.write(self.shadow, (base + self.perturbation_norm * direction).reshape(
            np.asarray(getattr(self.read(self.core), "data", self.read(self.core))).shape))

    def measure_and_renormalise(self) -> float:
        """Growth of the perturbation over the step just taken; then rescale."""

        base = _flatten(self.read(self.core))
        other = _flatten(self.read(self.shadow))
        difference = other - base
        grown = float(np.linalg.norm(difference))
        previous = max(self.perturbation_norm, 1.0e-300)
        growth = grown / previous if math.isfinite(grown) else math.inf
        target = self._scale()
        if not math.isfinite(grown) or grown <= 0.0:
            self.reseed()
        else:
            shape = np.asarray(getattr(self.read(self.core), "data", self.read(self.core))).shape
            self.write(self.shadow, (base + difference * (target / grown)).reshape(shape))
            self.perturbation_norm = target
        self.last_growth = growth
        return growth

    # -- dt-system state contract ---------------------------------------------
    def copy_shallow(self):
        return (self.core.copy_shallow(), self.shadow.copy_shallow(),
                self.perturbation_norm, self.last_growth)

    def restore(self, snapshot) -> None:
        core_snapshot, shadow_snapshot, norm, growth = snapshot
        self.core.restore(core_snapshot)
        self.shadow.restore(shadow_snapshot)
        self.perturbation_norm = float(norm)
        self.last_growth = float(growth)

    def dt_limit_hint(self):
        hint = getattr(self.core, "dt_limit_hint", None)
        return hint() if callable(hint) else None

    def __getattr__(self, name: str):
        # Everything else (displacement_criticality_m, telemetry, ...) is the
        # core's own; the shadow is invisible to callers that only know the
        # core's surface.
        return getattr(self.core, name)


def shadow_advance(advance: Callable[[Any, Any], tuple[Any, Any]]):
    """Wrap a core's ``advance(state, dt)`` to step the shadow alongside it.

    The returned function has the same ``(state, dt) -> (ok, metrics)``
    contract the dt system expects, where ``state`` is a
    :class:`ShadowedState`.  The real step's metrics are returned with the
    extra channel ``shadow_growth``.
    """

    def advance_both(state: ShadowedState, dt):
        ok, metrics = advance(state.core, dt)
        advance(state.shadow, dt)
        growth = state.measure_and_renormalise()
        metrics = coerce_metrics(metrics)
        channels = dict(metrics.error_channels or {})
        channels["shadow_growth"] = float(growth)
        metrics.error_channels = channels
        return ok, metrics

    return advance_both


def shadow_dt_limit(dt: float, growth: float, growth_max: float) -> float | None:
    """``dt * ln(growth_max) / ln(growth)`` for ``growth > 1``; else None.

    Under exponential amplification ``growth = exp(rate * dt)`` this is the
    largest step that would have amplified by only ``growth_max``.
    """

    if not (math.isfinite(growth) and math.isfinite(growth_max)):
        return 0.0 if growth == math.inf else None
    if growth <= 1.0 or growth_max <= 1.0:
        return None
    # AbstractTensor.log(), not math.log(): the compiled path only lowers
    # scalar transcendentals through the tensor-op surface (the same one
    # .sqrt() etc. already use everywhere else in this codebase); a raw
    # math.log() call has no native-materialization path there at all (see
    # docs -- diagnosed against the real managed-tire compile, math.log was
    # the one call left on the old float path when everything else near it
    # was already migrated).
    return float(dt) * float(AbstractTensor.tensor(growth_max).log().item()) / float(
        AbstractTensor.tensor(growth).log().item())


__all__ = ["ShadowedState", "shadow_advance", "shadow_dt_limit"]
