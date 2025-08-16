from __future__ import annotations

"""Common metrics and scaling utilities for adaptive dt control."""

from dataclasses import dataclass
from typing import Tuple, Optional
import math


@dataclass
class Metrics:
    """Simulation diagnostics collected during a micro-step.

    These fields are intentionally generic so they can be shared across
    simulators. Individual engines may ignore a subset of them.
    """

    max_vel: float
    max_flux: float
    div_inf: float
    mass_err: float
    osc_flag: bool = False
    stiff_flag: bool = False


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
