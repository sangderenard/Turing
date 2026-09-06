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


def _scalar(value, default: float = 0.0) -> float:
    """A Python float from a number or a 0-d tensor, never truncated.

    ``float(tensor)`` on an AbstractTensor falls through ``__index__`` and
    TRUNCATES (0.51 -> 0.0), which silently zeroed every sub-metre-per-second
    velocity a tensor-publishing core reported and left the CFL proposal
    unbounded.  ``.item()`` is the exact conversion.
    """

    if value is None:
        return float(default)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(item())
        except (TypeError, ValueError):
            pass
    return float(value)


def coerce_metrics(value) -> Metrics:
    """Normalize legacy metric-shaped records into the canonical contract."""

    if value is None:
        raise TypeError("simulation advance returned no metrics")
    # A core that computes its metrics as 0-d tensors returns a genuine
    # Metrics whose fields are tensors; every comparison and ``float()`` the
    # controller then makes would truncate them.  Normalize BOTH shapes of
    # record to exact Python floats here, once.
    dt_limit = getattr(value, "dt_limit", None)
    advanced_dt = getattr(value, "advanced_dt", None)
    channels = {
        str(name): _scalar(channel)
        for name, channel in (getattr(value, "error_channels", {}) or {}).items()
    }
    normalized = Metrics(
        max_vel=_scalar(getattr(value, "max_vel", 0.0)),
        max_flux=_scalar(getattr(value, "max_flux", 0.0)),
        div_inf=_scalar(getattr(value, "div_inf", 0.0)),
        mass_err=_scalar(getattr(value, "mass_err", 0.0)),
        osc_flag=bool(getattr(value, "osc_flag", False)),
        stiff_flag=bool(getattr(value, "stiff_flag", False)),
        sim_frame=int(getattr(value, "sim_frame", 0)),
        proc_ms=_scalar(getattr(value, "proc_ms", 0.0)),
        dt_limit=None if dt_limit is None else _scalar(dt_limit),
        error_channels=channels,
        hard_failure=bool(getattr(value, "hard_failure", False)),
        advanced_dt=None if advanced_dt is None else _scalar(advanced_dt),
    )
    if isinstance(value, Metrics):
        # Keep the caller's object identity (diagnostics such as
        # ``unresolved_report`` are attached to it later) but with exact
        # scalar fields.
        for name in ("max_vel", "max_flux", "div_inf", "mass_err", "proc_ms",
                     "dt_limit", "error_channels", "advanced_dt"):
            setattr(value, name, getattr(normalized, name))
        return value
    return normalized


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
