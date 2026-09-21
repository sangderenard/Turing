from __future__ import annotations

"""Common metrics and scaling utilities for adaptive dt control."""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import math


# Build-time channel declarations and the explicit shared dt layout.
from .error_channels import (
    channel_name, declare_channel, declared_channels, packed_channel_names,
    unpack_channel_name, empty_channels,
)
from ..tensors import AbstractTensor
from .control_diagnostics import empty_control
from .participants import empty_publication


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
    error_channels: AbstractTensor = field(default_factory=empty_channels)
    error_present: AbstractTensor = field(default_factory=empty_channels)
    # Controller-owned diagnostics have fixed columns, separate from measures.
    control_values: AbstractTensor = field(default_factory=empty_control)
    control_present: AbstractTensor = field(default_factory=empty_control)
    # Participant rows forwarded from the producer's declared storage.
    pub_tau: AbstractTensor = field(default_factory=empty_publication)
    pub_tau_present: AbstractTensor = field(default_factory=empty_publication)
    pub_contract: AbstractTensor = field(default_factory=empty_publication)
    pub_dt_limit: AbstractTensor = field(default_factory=empty_publication)
    pub_dt_limit_present: AbstractTensor = field(default_factory=empty_publication)
    pub_values: AbstractTensor = field(default_factory=empty_publication)
    pub_present: AbstractTensor = field(default_factory=empty_publication)
    pub_limits: AbstractTensor = field(default_factory=empty_publication)
    pub_limits_present: AbstractTensor = field(default_factory=empty_publication)
    hard_failure: bool = False
    advanced_dt: float | None = None
    # Stable diagnostic tokens attached by the controller when it proceeds
    # unresolved.  This is a total record field: ordinary and hard-failure
    # metrics carry the empty report, so native record state never has to
    # encode Python's dynamic-attribute absence as an anonymous input.
    unresolved_report: list[str] = field(default_factory=list)


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
    """Return the canonical tensorized Metrics record without conversion.

    The name remains as a compatibility boundary for existing dt callers.
    Producers own the Metrics ABI and its declared spans; this function does
    not construct a second record or extract tensor scalars.
    """

    return value


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
