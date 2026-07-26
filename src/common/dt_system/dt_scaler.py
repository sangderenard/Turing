from __future__ import annotations

"""Common metrics and scaling utilities for adaptive dt control."""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import math


@dataclass
class Metrics:
    """Simulation diagnostics collected during a micro-step.

    These fields are intentionally generic so they can be shared across
    simulators. Individual engines may ignore a subset of them. ``sim_frame``
    tracks the outer simulation frame index associated with the metrics, which
    can be useful when aggregating statistics across frames.
    """

    max_vel: float
    max_flux: float
    div_inf: float
    mass_err: float
    osc_flag: bool = False
    stiff_flag: bool = False
    sim_frame: int = 0
    # Wall-clock time of the last step for this engine (milliseconds).
    # Populated in preview mode or when instrumentation is enabled.
    proc_ms: float = 0.0
    # Optional sidechain: absolute dt limit hint proposed by the engine.
    # When provided, the dt controller will clamp the next proposal to this
    # value (min()), centralizing stability control instead of engines
    # self-capping internally.
    dt_limit: float | None = None
    # Named scientific error channels. Controllers compare these against
    # Targets.error_limits without forcing every engine into fluid terminology.
    error_channels: dict[str, float] = field(default_factory=dict)
    hard_failure: bool = False
    advanced_dt: float | None = None


def coerce_metrics(value) -> Metrics:
    """Normalize legacy metric-shaped records into the canonical contract."""

    if isinstance(value, Metrics):
        return value
    if value is None:
        raise TypeError("simulation advance returned no metrics")
    return Metrics(
        max_vel=float(getattr(value, "max_vel", 0.0)),
        max_flux=float(getattr(value, "max_flux", 0.0)),
        div_inf=float(getattr(value, "div_inf", 0.0)),
        mass_err=float(getattr(value, "mass_err", 0.0)),
        osc_flag=bool(getattr(value, "osc_flag", False)),
        stiff_flag=bool(getattr(value, "stiff_flag", False)),
        sim_frame=int(getattr(value, "sim_frame", 0)),
        proc_ms=float(getattr(value, "proc_ms", 0.0)),
        dt_limit=getattr(value, "dt_limit", None),
        error_channels=dict(getattr(value, "error_channels", {}) or {}),
        hard_failure=bool(getattr(value, "hard_failure", False)),
        advanced_dt=getattr(value, "advanced_dt", None),
    )


class ScalerControl:
    """Optional side-channel gain applied after scaling.

    The ``gain`` can be adjusted at runtime to impose additional control on
    the scaled value. When ``enabled`` is False, :meth:`apply` returns the
    input unchanged.
    """

    def __init__(self, gain: float = 1.0, enabled: bool = True) -> None:
        self.gain = gain
        self.enabled = enabled

    def apply(self, value: float) -> float:
        return value * self.gain if self.enabled else value


def scale_metric(
    value: float,
    window: Tuple[float, float],
    *,
    method: str = "linear",
    compression: str = "none",
    control: Optional[ScalerControl] = None,
) -> float:
    """Scale ``value`` into ``[0, 1]`` according to ``window`` and ``method``.

    Parameters
    ----------
    value:
        Raw metric value to scale.
    window:
        ``(lo, hi)`` bounds defining the target range. ``hi`` must be greater
        than ``lo``.
    method:
        ``"harsh"`` performs a step at ``hi``; ``"linear"`` interpolates; and
        ``"curve"`` applies a smooth nonlinear curve (cubic smoothstep).
    compression:
        Optional post-scaling compression: ``"log"`` or ``"sqrt"``.
    control:
        Optional :class:`ScalerControl` to apply after scaling.
    """

    lo, hi = window
    if hi <= lo:
        raise ValueError("window upper bound must exceed lower bound")
    x = (value - lo) / (hi - lo)

    if method == "harsh":
        scaled = 0.0 if x < 1.0 else 1.0
    elif method == "curve":
        x = min(max(x, 0.0), 1.0)
        scaled = x * x * (3.0 - 2.0 * x)  # cubic smoothstep
    else:  # linear
        scaled = min(max(x, 0.0), 1.0)

    if compression == "log":
        scaled = math.log1p(max(scaled, 0.0))
    elif compression == "sqrt":
        scaled = math.sqrt(max(scaled, 0.0))

    if control is not None:
        scaled = control.apply(scaled)
    return float(scaled)


__all__ = [
    "Metrics",
    "coerce_metrics",
    "ScalerControl",
    "scale_metric",
]
