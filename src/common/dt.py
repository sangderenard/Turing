# -*- coding: utf-8 -*-
"""dt management structs for adaptive superstepping.

This module defines small dataclasses used to describe a *superstep* request
("plan") and its corresponding outcome ("result"). A superstep advances a
simulation by a precise time window using one or more micro-steps chosen by an
adaptive controller, while enforcing a non-increasing dt policy within the
sequence (unless explicitly disabled). The final micro-step lands exactly on the
requested window via a remainder.

Glossary
--------
- round_max: Target time window for a frame/loop iteration (e.g., 1/60s).
- dt_init: Initial dt proposal from the controller before starting the round.
- dt_next: Controller's proposed dt to start the next round.
- non-increasing policy: Within one round, attempted dt is not allowed to grow
  (it may shrink due to stability or remainder clamping). This preserves
  predictiveness without requiring sequence rebuilds.

Usage
-----
A render loop or orchestrator creates a SuperstepPlan each frame and passes it to
an engine/adapter method that supports superstepping. The engine returns a
SuperstepResult that reports what happened and the `dt_next` for the next frame.

These types are intentionally minimal and engine-agnostic so they can be shared
across cellsim and spatial fluid engines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.cells.bath.dt_controller import Metrics


@dataclass
class SuperstepPlan:
    """Describe a single "superstep" request.

    Attributes
    ----------
    round_max:
        The exact time window to advance during this sequence.
    dt_init:
        The initial dt to attempt on the first micro-step (usually the
        controller's proposed dt from the previous frame).
    allow_increase_mid_round:
        If True, allow dt to grow during the sequence. Defaults to False for
        physically conservative behaviour. Enabling this may be useful for
        game-style time dilation but can reduce predictiveness unless the
        entire sequence is rebuilt.
    eps:
        Numerical tolerance for deciding when the target window has been
        satisfied.
    """
    round_max: float
    dt_init: float
    allow_increase_mid_round: bool = False
    eps: float = 1e-15


@dataclass
class SuperstepResult:
    """Report the outcome of a superstep sequence.

    Attributes
    ----------
    advanced:
        Total time actually advanced. This should match ``round_max`` within
        floating-point tolerance.
    dt_next:
        The controller's suggested dt to use on the next frame.
    steps:
        Number of micro-steps performed to cover the window.
    clamped:
        True if any halving/clamp events occurred due to instability.
    metrics:
        The last-step :class:`~src.cells.bath.dt_controller.Metrics` for UI or
        logging. Optional and engine-dependent.
    """
    advanced: float
    dt_next: float
    steps: int
    clamped: bool
    metrics: Optional[Metrics] = None
