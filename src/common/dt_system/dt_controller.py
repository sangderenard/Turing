# -*- coding: utf-8 -*-
"""Relocated: dt controller under common/dt_system."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import time

from .dt_scaler import Metrics
from .dt import SuperstepPlan, SuperstepResult
from .debug import dbg, is_enabled, pretty_metrics


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
    dt_min: float | None = None
    dt_max: float | None = None
    acc: float = 0.0
    max_vel_ever: float = 1e-30
    clamp_events: int = 0

    def update_dt_max(self, max_vel: float, dx: float) -> None:
        self.max_vel_ever = max(max_vel, 0.95 * self.max_vel_ever)
        self.dt_max = 1.0 * dx / max(self.max_vel_ever, 1e-30)
        if is_enabled():
            dbg("ctrl").debug(
                f"update_dt_max: max_vel={max_vel:.3e} -> max_vel_ever={self.max_vel_ever:.3e} dt_max={self.dt_max:.3e} dx={dx:.3e}"
            )

    def pi_update(self, dt_prev: float, dt_pen: float, osc: bool,
                  *, dt_min: float | None = None, dt_max: float | None = None) -> float:
        dt_min = self.dt_min if dt_min is None else dt_min
        dt_max = self.dt_max if dt_max is None else dt_max
        floor = dt_min if dt_min is not None else 1e-30
        e = math.log(max(dt_pen, floor)) - math.log(max(dt_prev, floor))
        self.acc = float(np.clip(self.acc + self.Ki * e, -self.A, self.A))
        log_dt = math.log(max(dt_prev, floor)) + self.Kp * e + self.acc
        dt_new = math.exp(log_dt)
        if dt_min is not None:
            dt_new = max(dt_new, dt_min)
        if dt_max is not None:
            dt_new = min(dt_new, dt_max)
        if osc:
            dt_new *= self.shrink
            if dt_min is not None:
                dt_new = max(dt_new, dt_min)
        if is_enabled():
            dbg("ctrl").debug(
                f"pi_update: dt_prev={dt_prev:.6g} dt_pen={dt_pen:.6g} osc={osc} -> dt_new={dt_new:.6g}"
                f" (bounds: dt_min={dt_min} dt_max={dt_max}) acc={self.acc:.3f}"
            )
        return float(dt_new)


def step_with_dt_control_used(state,
                             dt,
                             dx,
                             targets: Targets,
                             ctrl: STController,
                             advance,
                             retries: int = 0,
                             failures: list[tuple[float, Metrics]] | None = None):
    if failures is None:
        failures = []

    saved = state.copy_shallow()
    if is_enabled():
        dbg("ctrl").debug(
            f"advance try: dt={float(dt):.6g} dx={float(dx):.6g} retries={retries}"
        )
    ok, metrics = advance(state, dt)
    if (not ok) or (metrics.mass_err > targets.mass_max) or (metrics.div_inf > targets.div_max * 10.0):
        state.restore(saved)
        failures.append((float(dt), metrics))
        if is_enabled():
            dbg("ctrl").warning(
                f"advance failed: dt={float(dt):.6g} metrics=({pretty_metrics(metrics)})"
            )
        if retries >= 3:
            ctrl.clamp_events += 1
            lines = [f"timestep controller failed after {len(failures)} attempts:"]
            for i, (dt_f, m) in enumerate(failures, 1):
                lines.append(
                    f"  attempt {i}: dt={dt_f:.6g} mass_err={m.mass_err:.3e} div_inf={m.div_inf:.3e} max_vel={m.max_vel:.3e}"
                )
            print("\n".join(lines))
            raise RuntimeError("adaptive timestep controller failed")
        dt_half = dt * 0.5
        if ctrl.dt_min is not None:
            dt_half = max(dt_half, ctrl.dt_min)
        if is_enabled():
            dbg("ctrl").debug(f"retry with dt_half={float(dt_half):.6g}")
        return step_with_dt_control_used(state, dt_half, dx, targets, ctrl, advance, retries + 1, failures)

    dt_cfl = targets.cfl * dx / max(metrics.max_vel, 1e-30)
    penalty = max(
        metrics.div_inf / targets.div_max,
        metrics.mass_err / targets.mass_max,
        1.0,
    )
    dt_pen = dt_cfl / penalty
    dt_next = ctrl.pi_update(
        dt_prev=dt,
        dt_pen=dt_pen,
        osc=(metrics.osc_flag or metrics.stiff_flag),
    )
    # Sidechain limiter: clamp dt_next to any engine-provided absolute limit
    if metrics.dt_limit is not None:
        dt_next = min(dt_next, float(metrics.dt_limit))
    ctrl.update_dt_max(metrics.max_vel, dx)
    if is_enabled():
        dbg("ctrl").debug(
            f"advance ok: used_dt={float(dt):.6g} cfl_dt={dt_cfl:.6g} penalty={penalty:.3f}"
            + (f" dt_limit={metrics.dt_limit:.6g}" if metrics.dt_limit is not None else "")
            + f" -> dt_next={dt_next:.6g} | {pretty_metrics(metrics)}"
        )
    return metrics, dt_next, float(dt)


def step_with_dt_control(state, dt, dx, targets: Targets, ctrl: STController, advance, retries: int = 0):
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
    total = 0.0
    dt_cap = float(dt_init)
    if ctrl.dt_min is not None:
        dt_cap = max(dt_cap, ctrl.dt_min)
    if ctrl.dt_max is not None:
        dt_cap = min(dt_cap, ctrl.dt_max)
    last_dt_next = dt_cap
    last_metrics = None

    iters = 0
    if is_enabled():
        dbg("ctrl").debug(
            f"run_superstep: round_max={round_max:.6g} dt_init={dt_init:.6g} dx={dx:.6g}"
        )
    while (round_max - total) > eps and iters < max_iters:
        iters += 1
        remainder = round_max - total
        dt_try = min(dt_cap, remainder)
        metrics, dt_next, dt_used = step_with_dt_control_used(state, dt_try, dx, targets, ctrl, advance)
        last_metrics = metrics
        if dt_used <= 0.0:
            break
        total += dt_used
        if allow_increase_mid_round:
            dt_cap = dt_next
        else:
            dt_cap = min(dt_cap, dt_next)
        if ctrl.dt_min is not None:
            dt_cap = max(ctrl.dt_min, dt_cap)
        if ctrl.dt_max is not None:
            dt_cap = min(ctrl.dt_max, dt_cap)
        last_dt_next = dt_next
        if is_enabled():
            dbg("ctrl").debug(
                f"  iter={iters} used={dt_used:.6g} total={total:.6g}/{round_max:.6g} next_cap={dt_cap:.6g}"
            )

    return total, last_dt_next, last_metrics


def run_superstep_plan(state,
                       plan: SuperstepPlan,
                       dx: float,
                       targets: Targets,
                       ctrl: STController,
                       advance) -> SuperstepResult:
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
    ref = plan.dt_init
    if ctrl.dt_min is not None:
        ref = max(ref, ctrl.dt_min)
    clamped = bool(dt_next < ref)
    steps = max(1, int(round(total / max(plan.dt_init, 1e-30)))) if total > 0 else 0
    return SuperstepResult(advanced=float(total), dt_next=float(dt_next), steps=steps, clamped=clamped, metrics=metrics)


# ------------------------- Realtime mode (single-step) -----------------------

def step_realtime_once(
    state,
    dt_current: float,
    dx: float,
    targets: Targets,
    ctrl: STController,
    advance,
    *,
    alloc_ms: float,
    allow_exceptions: bool = False,
):
    """Run exactly one advance and set next dt from a time allocation.

    This realtime mode prioritizes liveness: it executes a single step, measures
    wall-clock time, records it into Metrics.proc_ms, and proposes the next dt as
    alloc_ms/1000. In realtime mode we ignore engine-provided dt_limit to preserve
    real-time pacing. No
    retries, no superstep/substep are performed here unless ``allow_exceptions``
    is True, in which case a future extension may try minimal corrective splits
    within the allocation if it demonstrably reduces penalty.
    """
    # Single attempt only; no rollback or halving in realtime mode.
    t0 = time.perf_counter()
    ok, metrics = advance(state, float(dt_current))
    t1 = time.perf_counter()
    elapsed_ms = max((t1 - t0) * 1000.0, 0.0)
    # Attach timing to metrics generically
    try:
        metrics.proc_ms = float(elapsed_ms)
    except Exception:
        pass

    if not ok:
        # On failure, keep dt small (use dt_min if set, else tiny) to avoid explosion next frame
        dt_baseline = ctrl.dt_min if ctrl.dt_min is not None else 1e-6
        if is_enabled():
            dbg("ctrl").warning(
                f"rt advance failed: dt={dt_current:.6g} -> next={dt_baseline:.6g} ({pretty_metrics(metrics)})"
            )
        return metrics, float(dt_baseline), float(dt_current)

    # Base proposal from allocation (thumbnailing simulated time to budget)
    # Ignore engine hard limit (dt_limit) in realtime to maintain pacing.
    dt_next = max(alloc_ms, 0.0) * 1e-3

    # Controller book-keeping still learns dt_max from velocities
    ctrl.update_dt_max(metrics.max_vel, dx)

    if is_enabled():
        dbg("ctrl").debug(
            "rt: "
            f"used_dt={float(dt_current):.6g} alloc={alloc_ms:.3f}ms cost={elapsed_ms:.3f}ms "
            + (f"dt_limit={metrics.dt_limit:.6g} " if metrics.dt_limit is not None else "")
            + f"-> dt_next={dt_next:.6g} | {pretty_metrics(metrics)}"
        )

    return metrics, float(dt_next), float(dt_current)

 
