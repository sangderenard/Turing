# -*- coding: utf-8 -*-
"""Simple adaptive timestep controller with PI smoothing."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from src.common.dt_scaler import Metrics


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

        The previous implementation tracked the maximum velocity seen so far,
        which meant an early spike could permanently clamp the timestep. To
        let the controller accelerate once the flow calms down, track a softly
        decaying envelope of the observed maximum. This effectively aims for
        the minimum of recent maxima, allowing ``dt`` to grow when speeds drop.
        """
        # Exponential decay lets old peaks fade so that ``max_vel_ever``
        # approaches the smallest recent maximum velocity. Use a slightly
        # faster decay so early spikes relax more quickly, and widen the
        # allowable ``dt_max`` so the PI loop isn't over-constrained relative
        # to the CFL target.
        self.max_vel_ever = max(max_vel, 0.95 * self.max_vel_ever)
        # Permit up to roughly one full cell traversal at unit CFL; the PI
        # loop will still pick a fraction via ``targets.cfl``. Keep a small
        # guard to avoid division by zero.
        self.dt_max = 1.0 * dx / max(self.max_vel_ever, 1e-30)

    def pi_update(self, dt_prev: float, dt_pen: float, osc: bool) -> float:
        e = math.log(max(dt_pen, self.dt_min)) - math.log(max(dt_prev, self.dt_min))
        self.acc = float(np.clip(self.acc + self.Ki * e, -self.A, self.A))
        log_dt = math.log(max(dt_prev, self.dt_min)) + self.Kp * e + self.acc
        dt_new = float(np.clip(math.exp(log_dt), self.dt_min, self.dt_max))
        if osc:
            dt_new = max(dt_new * self.shrink, self.dt_min)
        return dt_new


def step_with_dt_control_used(state,
                             dt,
                             dx,
                             targets: Targets,
                             ctrl: STController,
                             advance,
                             retries: int = 0,
                             failures: list[tuple[float, Metrics]] | None = None):
    """Advance with adaptive control, returning (metrics, dt_next, dt_used).

    This engine-agnostic helper drives a single micro-step:
    - Attempts ``advance(state, dt)``; on failure or excessive error, rolls back
      to a shallow copy and retries with ``dt/2`` (bounded retries).
    - On success, computes a new proposal ``dt_next`` from controller PI and CFL
      targets, and reports ``dt_used`` = dt actually advanced.
    - If every retry fails, a report of all attempts is printed and a
      ``RuntimeError`` is raised to abort the caller.

    Parameters
    ----------
    state : Any
        Simulator instance supporting ``copy_shallow``/``restore``.
    dt : float
        Timestep to attempt this micro-step.
    dx : float
        Spatial scale used by CFL.
    targets : Targets
        CFL and error thresholds.
    ctrl : STController
        Adaptive dt controller state.
    advance : Callable[[Any, float], tuple[bool, Metrics]]
        Engine-specific function to advance ``state`` by ``dt`` and compute
        :class:`Metrics`.
    retries : int
        Current retry count (internal recursion).
    failures : list[tuple[float, Metrics]] | None
        Accumulator for failed attempts (internal recursion).

    Returns
    -------
    (Metrics, float, float)
        Tuple of ``(metrics, dt_next, dt_used)``.
    """
    if failures is None:
        failures = []

    saved = state.copy_shallow()
    ok, metrics = advance(state, dt)
    if (not ok) or (metrics.mass_err > targets.mass_max) or (metrics.div_inf > targets.div_max * 10.0):
        state.restore(saved)
        failures.append((float(dt), metrics))
        if retries >= 3:
            ctrl.clamp_events += 1
            lines = [f"timestep controller failed after {len(failures)} attempts:"]
            for i, (dt_f, m) in enumerate(failures, 1):
                lines.append(
                    f"  attempt {i}: dt={dt_f:.6g} mass_err={m.mass_err:.3e} div_inf={m.div_inf:.3e} max_vel={m.max_vel:.3e}"
                )
            print("\n".join(lines))
            raise RuntimeError("adaptive timestep controller failed")
        dt_half = max(dt * 0.5, ctrl.dt_min)
        return step_with_dt_control_used(state, dt_half, dx, targets, ctrl, advance, retries + 1, failures)

    dt_cfl = targets.cfl * dx / max(metrics.max_vel, 1e-30)
    penalty = max(
        metrics.div_inf / targets.div_max,
        metrics.mass_err / targets.mass_max,
        1.0,
    )
    dt_pen = dt_cfl / penalty
    dt_next = ctrl.pi_update(dt_prev=dt, dt_pen=dt_pen, osc=(metrics.osc_flag or metrics.stiff_flag))
    ctrl.update_dt_max(metrics.max_vel, dx)
    return metrics, dt_next, float(dt)


def step_with_dt_control(state, dt, dx, targets: Targets, ctrl: STController, advance, retries: int = 0):
    """Backward-compatible wrapper returning (metrics, dt_next)."""
    metrics, dt_next, _dt_used = step_with_dt_control_used(state, dt, dx, targets, ctrl, advance, retries)
    return metrics, dt_next


def run_superstep(state,
                  round_max: float,
                  dt_init: float,
                  dx: float,
                  targets: Targets,
                  ctrl: STController,
                  advance,
                  *,
                  allow_increase_mid_round: bool = False,
                  max_iters: int = 10000,
                  eps: float = 1e-15):
    """Run one superstep with exact landing and a non-increasing dt policy.

    This routine advances a simulator by precisely ``round_max`` time using one
    or more micro-steps. Within the sequence, the attempted dt is clamped to be
    non-increasing by default (it may be reduced by stability or remainder
    constraints). The final iteration uses the exact remainder to land on the
    target time. This preserves predictiveness without requiring expensive
    sequence rebuilds.

    Returns
    -------
    (advanced, dt_next, metrics)
        ``advanced`` is the total time covered (≈ ``round_max``), ``dt_next`` is
        the proposed dt for the next frame, and ``metrics`` is the last-step
        Metrics.
    """
    total = 0.0
    dt_cap = float(max(dt_init, ctrl.dt_min))
    # Track the controller's raw proposal separately from the in-round cap so
    # that growth suggestions survive to the next meta step.  The original
    # implementation overwrote this with the clamped ``dt_cap`` which meant the
    # outer controller never saw increases when the round started with a tiny
    # seed (e.g. the 1e-6 fallback), effectively locking the simulation to that
    # value.  ``last_dt_next`` now records the unclamped proposal from the final
    # micro-step and is returned to the caller.
    last_dt_next = dt_cap
    last_metrics = None

    iters = 0
    while (round_max - total) > eps and iters < max_iters:
        iters += 1
        remainder = round_max - total
        dt_try = min(dt_cap, remainder)
        metrics, dt_next, dt_used = step_with_dt_control_used(state, dt_try, dx, targets, ctrl, advance)
        last_metrics = metrics

        # If no progress was made (dt_used == 0), prevent infinite loop.
        if dt_used <= 0.0:
            break
        total += dt_used

        # Enforce non-increasing dt within the round unless explicitly allowed.
        if allow_increase_mid_round:
            dt_cap = max(ctrl.dt_min, dt_next)
        else:
            dt_cap = max(ctrl.dt_min, min(dt_cap, dt_next))
        # Preserve the controller's proposal for the next round rather than the
        # capped value used internally this round.
        last_dt_next = dt_next

    return total, last_dt_next, last_metrics


# Struct-based convenience API -------------------------------------------------
from src.common.dt import SuperstepPlan, SuperstepResult

def run_superstep_plan(state,
                       plan: SuperstepPlan,
                       dx: float,
                       targets: Targets,
                       ctrl: STController,
                       advance) -> SuperstepResult:
    """Execute a superstep using :class:`~src.common.dt.SuperstepPlan`.

    This is a thin wrapper around :func:`run_superstep` that speaks in terms of
    plan/result structs for easier orchestration. It enforces the same
    semantics and returns a :class:`~src.common.dt.SuperstepResult`.
    """
    total, dt_next, metrics = run_superstep(
        state,
        plan.round_max,
        plan.dt_init,
        dx,
        targets,
        ctrl,
        advance,
        allow_increase_mid_round=plan.allow_increase_mid_round,
        eps=plan.eps,
    )
    # Minimal guess for clamped: true if next dt is smaller than attempted cap
    clamped = bool(dt_next < max(plan.dt_init, ctrl.dt_min))
    steps = max(1, int(round(total / max(plan.dt_init, 1e-30)))) if total > 0 else 0
    return SuperstepResult(advanced=float(total), dt_next=float(dt_next), steps=steps, clamped=clamped, metrics=metrics)


# Thermodynamics stubs ----------------------------------------------------

def ideal_gas_eos(cell) -> float:
    rho = getattr(cell, "rho", 1.0)
    R = getattr(cell, "R", 1.0)
    T = getattr(cell, "T", 0.0)
    return rho * R * T


def noop_energy_step(cell, dt_half: float) -> None:
    return None
