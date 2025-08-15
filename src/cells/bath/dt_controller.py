# -*- coding: utf-8 -*-
"""Simple adaptive timestep controller with PI smoothing."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class Metrics:
    max_vel: float
    max_flux: float
    div_inf: float
    mass_err: float
    osc_flag: bool = False
    stiff_flag: bool = False


@dataclass
class Targets:
    cfl: float
    div_max: float
    mass_max: float


@dataclass
class STController:
    Kp: float = 0.4
    Ki: float = 0.05
    A: float = 1.5
    shrink: float = 0.5
    dt_min: float = 1e-6
    dt_max: float = 1.0
    acc: float = 0.0
    max_vel_ever: float = 1e-30
    clamp_events: int = 0

    def update_dt_max(self, max_vel: float, dx: float) -> None:
        """Update the adaptive ``dt_max`` based on recent velocities.

        The previous implementation tracked the *maximum* velocity seen so
        far which meant that any early spike would permanently clamp the
        timestep.  To let the controller accelerate once the flow calms
        down, we keep a softly decaying envelope of the observed maximum.
        This effectively aims for the minimum of recent maxima across the
        data, allowing ``dt`` to grow when speeds drop.
        """

        # Exponential decay lets old peaks fade so that ``max_vel_ever``
        # approaches the smallest recent maximum velocity.  This provides an
        # upper bound that adapts upward, permitting acceleration when the
        # system slows.
        self.max_vel_ever = max(max_vel, 0.98 * self.max_vel_ever)
        self.dt_max = 0.25 * dx / max(self.max_vel_ever, 1e-30)

    def pi_update(self, dt_prev: float, dt_pen: float, osc: bool) -> float:
        e = math.log(max(dt_pen, self.dt_min)) - math.log(max(dt_prev, self.dt_min))
        self.acc = float(np.clip(self.acc + self.Ki * e, -self.A, self.A))
        log_dt = math.log(max(dt_prev, self.dt_min)) + self.Kp * e + self.acc
        dt_new = float(np.clip(math.exp(log_dt), self.dt_min, self.dt_max))
        if osc:
            dt_new = max(dt_new * self.shrink, self.dt_min)
        return dt_new


def step_with_dt_control(state, dt, dx, targets: Targets, ctrl: STController, advance, retries: int = 0):
    saved = state.copy_shallow()
    ok, metrics = advance(state, dt)
    if (not ok) or (metrics.mass_err > targets.mass_max) or (metrics.div_inf > targets.div_max * 10.0):
        state.restore(saved)
        if retries >= 3:
            ctrl.clamp_events += 1
            return metrics, dt
        dt = max(dt * 0.5, ctrl.dt_min)
        return step_with_dt_control(state, dt, dx, targets, ctrl, advance, retries + 1)

    dt_cfl = targets.cfl * dx / max(metrics.max_vel, 1e-30)
    penalty = max(
        metrics.div_inf / targets.div_max,
        metrics.mass_err / targets.mass_max,
        1.0,
    )
    dt_pen = dt_cfl / penalty
    dt_next = ctrl.pi_update(dt_prev=dt, dt_pen=dt_pen, osc=(metrics.osc_flag or metrics.stiff_flag))
    ctrl.update_dt_max(metrics.max_vel, dx)
    return metrics, dt_next


# Thermodynamics stubs ----------------------------------------------------

def ideal_gas_eos(cell) -> float:
    rho = getattr(cell, "rho", 1.0)
    R = getattr(cell, "R", 1.0)
    T = getattr(cell, "T", 0.0)
    return rho * R * T


def noop_energy_step(cell, dt_half: float) -> None:
    return None
