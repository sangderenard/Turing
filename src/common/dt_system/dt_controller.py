# -*- coding: utf-8 -*-
"""Relocated: dt controller under common/dt_system."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

from ..tensors.abstraction import AbstractTensor

try:  # NumPy is the canonical lightweight backend
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

from .dt_scaler import Metrics, coerce_metrics
from .dt import SuperstepPlan, SuperstepResult
from .debug import dbg, is_enabled, pretty_metrics


def _restore_type(value, ref):
    """Return ``value`` converted to the type of ``ref``."""
    if isinstance(ref, AbstractTensor):
        return value
    val = float(value.item() if isinstance(value, AbstractTensor) else value)
    if np is not None and isinstance(ref, np.ndarray):
        return np.array(val, dtype=ref.dtype)
    if isinstance(ref, list):
        return [val]
    if isinstance(ref, tuple):
        return (val,)
    return val


@dataclass
class Targets:
    cfl: float
    div_max: float
    mass_max: float
    error_limits: dict[str, float] = field(default_factory=dict)


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
        max_vel_t = max_vel if isinstance(max_vel, AbstractTensor) else AbstractTensor.tensor(max_vel)
        self.max_vel_ever = AbstractTensor.maximum(max_vel_t, 0.95 * self.max_vel_ever)
        dx_t = dx if isinstance(dx, AbstractTensor) else AbstractTensor.tensor(dx)
        dt_max_t = dx_t / AbstractTensor.maximum(self.max_vel_ever, 1e-30)
        self.dt_max = _restore_type(dt_max_t, dx)
        if is_enabled():
            dbg("ctrl").debug(
                f"update_dt_max: max_vel={float(max_vel_t.item() if isinstance(max_vel_t, AbstractTensor) else max_vel_t):.3e} "
                f"-> max_vel_ever={float(self.max_vel_ever.item() if isinstance(self.max_vel_ever, AbstractTensor) else self.max_vel_ever):.3e} "
                f"dt_max={float(self.dt_max.item() if isinstance(self.dt_max, AbstractTensor) else self.dt_max):.3e} "
                f"dx={float(dx_t.item() if isinstance(dx_t, AbstractTensor) else dx_t):.3e}"
            )

    def pi_update(self, dt_prev, dt_pen, osc: bool,
                  *, dt_min: float | AbstractTensor | None = None,
                  dt_max: float | AbstractTensor | None = None):
        ref_prev = dt_prev
        dt_prev = dt_prev if isinstance(dt_prev, AbstractTensor) else AbstractTensor.tensor(dt_prev)
        dt_pen = dt_pen if isinstance(dt_pen, AbstractTensor) else AbstractTensor.tensor(dt_pen)
        self.acc = self.acc if isinstance(self.acc, AbstractTensor) else AbstractTensor.tensor(self.acc)
        dt_min = self.dt_min if dt_min is None else dt_min
        dt_max = self.dt_max if dt_max is None else dt_max
        floor = dt_min if dt_min is not None else 1e-30
        floor_t = floor if isinstance(floor, AbstractTensor) else AbstractTensor.tensor(floor)
        e = (AbstractTensor.maximum(dt_pen, floor_t).log() - AbstractTensor.maximum(dt_prev, floor_t).log())
        self.acc = (self.acc + self.Ki * e).clamp(min=-self.A, max=self.A)
        log_dt = AbstractTensor.maximum(dt_prev, floor_t).log() + self.Kp * e + self.acc
        dt_new = log_dt.exp()
        if dt_min is not None:
            dt_min_t = dt_min if isinstance(dt_min, AbstractTensor) else AbstractTensor.tensor(dt_min)
            dt_new = AbstractTensor.maximum(dt_new, dt_min_t)
        if dt_max is not None:
            dt_max_t = dt_max if isinstance(dt_max, AbstractTensor) else AbstractTensor.tensor(dt_max)
            dt_new = AbstractTensor.minimum(dt_new, dt_max_t)
        if osc:
            dt_new = dt_new * self.shrink
            if dt_min is not None:
                dt_new = AbstractTensor.maximum(dt_new, dt_min_t)
        if is_enabled():
            dbg("ctrl").debug(
                f"pi_update: dt_prev={float(dt_prev.item()):.6g} dt_pen={float(dt_pen.item()):.6g} osc={osc} -> dt_new={float(dt_new.item()):.6g}"
                f" (bounds: dt_min={dt_min} dt_max={dt_max}) acc={float(self.acc.item()):.3f}"
            )
        return _restore_type(dt_new, ref_prev)


def step_with_dt_control_used(state,
                             dt,
                             dx,
                             targets: Targets,
                             ctrl: STController,
                             advance,
                             retries: int = 0,
                             max_retries: int = 3,
                             failures: list[tuple[float, Metrics, tuple[str, ...]]] | None = None,
                             ref=None,
                             attempt_log: list[dict] | None = None):
    if failures is None:
        failures = []
    if ref is None:
        ref = dt

    dt_tensor = dt if isinstance(dt, AbstractTensor) else AbstractTensor.tensor(dt)
    dt_for_advance = _restore_type(dt_tensor, ref)

    saved = state.copy_shallow()
    if is_enabled():
        dbg("ctrl").debug(
            f"advance try: dt={float(dt_for_advance):.6g} dx={float(dx.item() if isinstance(dx, AbstractTensor) else dx):.6g} retries={retries}"
        )
    ok, metrics = advance(state, dt_for_advance)
    metrics = coerce_metrics(metrics)
    channel_failure = any(
        float(metrics.error_channels.get(name, 0.0)) > float(limit)
        for name, limit in targets.error_limits.items()
    )
    # Name the term that rejected, not merely that something did. Reporting
    # only mass/div/vel meant a rejection coming from `ok` or a named error
    # channel printed four healthy-looking attempts and no cause.
    reasons: list[str] = []
    if not ok:
        reasons.append("advance reported a physical-bound violation")
    if bool(metrics.hard_failure):
        reasons.append("hard_failure")
    if metrics.mass_err > targets.mass_max:
        reasons.append(
            f"mass_err {float(metrics.mass_err):.3e} > {float(targets.mass_max):.3e}"
        )
    if metrics.div_inf > targets.div_max * 10.0:
        reasons.append(
            f"div_inf {float(metrics.div_inf):.3e} > "
            f"{float(targets.div_max) * 10.0:.3e}"
        )
    reasons.extend(
        f"channel {name} {float(metrics.error_channels.get(name, 0.0)):.3e} "
        f"> {float(limit):.3e}"
        for name, limit in targets.error_limits.items()
        if float(metrics.error_channels.get(name, 0.0)) > float(limit)
    )
    rejected = bool(reasons)
    if attempt_log is not None:
        attempt_log.append({
            "dt": float(dt_for_advance),
            "accepted": not rejected,
            "metrics": metrics,
            "reasons": tuple(reasons),
        })
    if rejected and retries >= max_retries:
        # The search is out of room. Subdivision cannot resolve an
        # irregularity, and a bisection can run out of bracket the same way, so
        # refusing to continue would make every such case fatal. Keep the step
        # that was taken, record exactly what it violated, and carry on: an
        # unresolved step that is traceable is worth more than a halted run.
        ctrl.clamp_events += 1
        metrics.hard_failure = True
        channels = dict(metrics.error_channels or {})
        channels["dt_unresolved"] = float(dt_for_advance)
        channels["dt_unresolved_attempts"] = float(len(failures) + 1)
        metrics.error_channels = channels
        # The trace rides on the metrics; a substep must not narrate. At a
        # pinned audio-rate interior this runs thousands of times a frame, and
        # printing each one buries the very thing it is reporting.
        lines = [
            "timestep controller proceeded unresolved after "
            f"{len(failures) + 1} attempt(s):"
        ]
        for index, (dt_f, m, why) in enumerate(
            (*failures, (float(dt_for_advance), metrics, tuple(reasons))), 1,
        ):
            lines.append(
                f"  attempt {index}: dt={dt_f:.6g} mass_err={m.mass_err:.3e} "
                f"div_inf={m.div_inf:.3e} max_vel={m.max_vel:.3e}"
            )
            lines.extend(f"      rejected by: {reason}" for reason in why)
        if max_retries == 0:
            lines.append(
                "  the substep is pinned, so there was no smaller candidate to "
                "analyse; this is a physical rejection, not an exhausted search."
            )
        elif len({tuple(why) for _dt, _m, why in failures}) <= 1:
            lines.append(
                "  every attempt was rejected for the same reason at every dt, "
                "so subdividing further could not have resolved it."
            )
        channels["dt_unresolved_report"] = 0.0
        metrics.unresolved_report = tuple(lines)
        if is_enabled():
            dbg("ctrl").warning(chr(10).join(lines))
        # Fall through to the ordinary accepted path so the proposal for the
        # next step is computed the same way it always is.
        rejected = False
    if rejected:
        state.restore(saved)
        failures.append((float(dt_for_advance), metrics, tuple(reasons)))
        if is_enabled():
            dbg("ctrl").warning(
                f"advance failed: dt={float(dt.item() if isinstance(dt, AbstractTensor) else dt):.6g} metrics=({pretty_metrics(metrics)})"
            )
        if retries >= max_retries:
            ctrl.clamp_events += 1
            lines = [f"timestep controller failed after {len(failures)} attempts:"]
            for i, (dt_f, m, why) in enumerate(failures, 1):
                lines.append(
                    f"  attempt {i}: dt={dt_f:.6g} mass_err={m.mass_err:.3e} "
                    f"div_inf={m.div_inf:.3e} max_vel={m.max_vel:.3e}"
                )
                lines.extend(f"      rejected by: {reason}" for reason in why)
            if max_retries == 0:
                lines.append(
                    "  the substep is pinned, so there is no smaller candidate "
                    "to analyse: this is a physical rejection, not a dt search "
                    "that ran out of room."
                )
            elif len({tuple(why) for _dt, _m, why in failures}) == 1:
                lines.append(
                    "  every attempt was rejected for the same reason at every "
                    "dt, so halving could not have resolved it."
                )
            print("\n".join(lines))
            raise RuntimeError("adaptive timestep controller failed")
        dt_half = dt_tensor * 0.5
        if (
            ctrl.dt_min is not None
            and float(dt_tensor.item()) >= float(ctrl.dt_min)
        ):
            dt_half = AbstractTensor.maximum(dt_half, ctrl.dt_min)
        if (
            metrics.dt_limit is not None
            and math.isfinite(float(metrics.dt_limit))
            and float(metrics.dt_limit) > 0.0
        ):
            dt_half = AbstractTensor.minimum(dt_half, metrics.dt_limit)
        if is_enabled():
            dbg("ctrl").debug(
                f"retry with dt_half={float(dt_half.item() if isinstance(dt_half, AbstractTensor) else dt_half):.6g}"
            )
        return step_with_dt_control_used(
            state,
            dt_half,
            dx,
            targets,
            ctrl,
            advance,
            retries + 1,
            max_retries,
            failures,
            ref=ref,
            attempt_log=attempt_log,
        )

    dt_cfl = targets.cfl * dx / max(metrics.max_vel, 1e-30)
    penalty = max(
        metrics.div_inf / targets.div_max,
        metrics.mass_err / targets.mass_max,
        *(
            float(metrics.error_channels.get(name, 0.0)) / max(float(limit), 1e-30)
            for name, limit in targets.error_limits.items()
        ),
        1.0,
    )
    dt_pen = dt_cfl / penalty
    dt_next = ctrl.pi_update(
        dt_prev=dt_tensor,
        dt_pen=dt_pen,
        osc=(metrics.osc_flag or metrics.stiff_flag),
    )
    # Sidechain limiter: clamp dt_next to any engine-provided absolute limit
    if metrics.dt_limit is not None:
        dt_next = AbstractTensor.minimum(dt_next, metrics.dt_limit)
    ctrl.update_dt_max(metrics.max_vel, dx)
    if is_enabled():
        dbg("ctrl").debug(
            f"advance ok: used_dt={float(dt.item() if isinstance(dt, AbstractTensor) else dt):.6g} cfl_dt={dt_cfl:.6g} penalty={penalty:.3f}"
            + (f" dt_limit={metrics.dt_limit:.6g}" if metrics.dt_limit is not None else "")
            + f" -> dt_next={float(dt_next.item() if isinstance(dt_next, AbstractTensor) else dt_next):.6g} | {pretty_metrics(metrics)}"
        )
    return metrics, _restore_type(dt_next, ref), _restore_type(dt_tensor, ref)


def step_with_dt_control(state, dt, dx, targets: Targets, ctrl: STController, advance, retries: int = 0):
    metrics, dt_next, _dt_used = step_with_dt_control_used(state, dt, dx, targets, ctrl, advance, retries, ref=dt)
    return metrics, dt_next


def run_superstep(state,
                  round_max: float | AbstractTensor,
                  dt_init: float | AbstractTensor,
                  dx: float,
                  targets: Targets,
                  ctrl: STController,
                  advance,
                  *,
                  substep: str = "steered",
                  substep_dt: float | None = None,
                  allow_increase_mid_round: bool = False,
                  max_iters: int = 10000,
                  eps: float = 1e-15,
                  event_boundaries: tuple[float, ...] = (),
                  attempt_log: list[dict] | None = None):
    if substep not in {"pinned", "steered"}:
        raise ValueError(
            f"unknown substep interior {substep!r}; expected 'pinned' or "
            "'steered'"
        )
    if substep == "pinned":
        if substep_dt is None or float(substep_dt) <= 0.0:
            raise ValueError("a pinned interior requires a positive substep_dt")
        # A pinned substep is a constant, so the controller must not steer it
        # and the CFL ceiling must not raise it. Nothing about the window
        # landing or the rejection test changes.
        dt_init = float(substep_dt)
    ref_dt = dt_init
    round_max_t = round_max if isinstance(round_max, AbstractTensor) else AbstractTensor.tensor(round_max)
    total = AbstractTensor.tensor(0.0)
    dt_cap = dt_init if isinstance(dt_init, AbstractTensor) else AbstractTensor.tensor(dt_init)
    if ctrl.dt_min is not None:
        dt_cap = AbstractTensor.maximum(dt_cap, ctrl.dt_min)
    if ctrl.dt_max is not None:
        dt_cap = AbstractTensor.minimum(dt_cap, ctrl.dt_max)
    last_dt_next = dt_cap
    last_metrics = None
    boundary_values = tuple(sorted({
        float(value)
        for value in event_boundaries
        if eps < float(value) < float(round_max_t.item()) - eps
    }))

    unresolved: list[Metrics] = []
    iters = 0
    if is_enabled():
        dbg("ctrl").debug(
            f"run_superstep: round_max={float(round_max_t.item()):.6g} dt_init={float(dt_cap.item()):.6g} dx={dx:.6g}"
        )
    while (round_max_t - total).item() > eps and iters < max_iters:
        iters += 1
        remainder = round_max_t - total
        dt_try = AbstractTensor.minimum(dt_cap, remainder)
        total_value = float(total.item())
        for boundary in boundary_values:
            if boundary > total_value + eps:
                dt_try = AbstractTensor.minimum(
                    dt_try,
                    AbstractTensor.tensor(boundary - total_value),
                )
                break
        metrics, dt_next, dt_used = step_with_dt_control_used(
            state,
            dt_try,
            dx,
            targets,
            ctrl,
            advance,
            max_retries=0 if substep == "pinned" else 3,
            ref=ref_dt,
            attempt_log=attempt_log,
        )
        last_metrics = metrics
        if float((metrics.error_channels or {}).get("dt_unresolved", 0.0)) > 0.0:
            unresolved.append(metrics)
        if dt_used <= 0.0:
            break
        total += dt_used
        if substep == "pinned":
            # Held at the requested constant; the controller's proposal and its
            # CFL ceiling are both irrelevant here by construction.
            dt_cap = AbstractTensor.tensor(float(substep_dt))
        elif allow_increase_mid_round:
            dt_cap = dt_next
        else:
            dt_cap = AbstractTensor.minimum(dt_cap, dt_next)
        if substep != "pinned":
            if ctrl.dt_min is not None:
                dt_cap = AbstractTensor.maximum(ctrl.dt_min, dt_cap)
            if ctrl.dt_max is not None:
                dt_cap = AbstractTensor.minimum(ctrl.dt_max, dt_cap)
        last_dt_next = dt_next
        if is_enabled():
            dbg("ctrl").debug(
                f"  iter={iters} used={float(dt_used.item() if isinstance(dt_used, AbstractTensor) else dt_used):.6g} total={float(total.item()):.6g}/{round_max:.6g} next_cap={float(dt_cap.item()):.6g}"
            )

    if unresolved:
        first = unresolved[0]
        print(
            f"{len(unresolved)} of {iters} substep(s) advanced unresolved; "
            f"first at dt="
            f"{float(first.error_channels.get('dt_unresolved', 0.0)):.6g}"
        )
        for line in getattr(first, "unresolved_report", ())[1:]:
            print(f"  {line.strip()}")
    remaining = float((round_max_t - total).item())
    if remaining > eps:
        raise RuntimeError(
            "adaptive superstep did not land on its requested window: "
            f"advanced={float(total.item()):.17g} "
            f"round_max={float(round_max_t.item()):.17g} "
            f"remaining={remaining:.17g} iterations={iters}/{max_iters}"
        )

    total_out = _restore_type(total, ref_dt)
    dt_next_out = _restore_type(last_dt_next, ref_dt)
    return total_out, dt_next_out, last_metrics


def run_superstep_plan(state,
                       plan: SuperstepPlan,
                       dx: float,
                       targets: Targets,
                       ctrl: STController,
                       advance) -> SuperstepResult:
    attempt_log: list[dict] = []
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
        event_boundaries=plan.event_boundaries,
        attempt_log=attempt_log,
    )
    total_val = float(total.item() if isinstance(total, AbstractTensor) else total)
    dt_next_val = float(dt_next.item() if isinstance(dt_next, AbstractTensor) else dt_next)
    plan_dt_init_val = float(plan.dt_init.item() if isinstance(plan.dt_init, AbstractTensor) else plan.dt_init)
    ref = plan_dt_init_val
    if ctrl.dt_min is not None:
        ref = max(ref, float(ctrl.dt_min.item() if isinstance(ctrl.dt_min, AbstractTensor) else ctrl.dt_min))
    clamped = bool(dt_next_val < ref)
    accepted = tuple(
        float(item["dt"]) for item in attempt_log if item["accepted"]
    )
    rejected = sum(1 for item in attempt_log if not item["accepted"])
    steps = len(accepted)
    clamped = clamped or rejected > 0
    cumulative = 0.0
    landed = []
    for dt_used in accepted:
        cumulative += dt_used
        if any(abs(cumulative - boundary) <= plan.eps for boundary in plan.event_boundaries):
            landed.append(cumulative)
    return SuperstepResult(
        advanced=total,
        dt_next=dt_next,
        steps=steps,
        clamped=clamped,
        metrics=metrics,
        attempted_dts=tuple(float(item["dt"]) for item in attempt_log),
        accepted_dts=accepted,
        rejected_attempts=rejected,
        landed_boundaries=tuple(landed),
    )


# ------------------------- Realtime mode (single-step) -----------------------

def step_realtime_once(
    state,
    dt_current,
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
    ref_dt = dt_current
    dt_val = float(dt_current.item() if isinstance(dt_current, AbstractTensor) else dt_current)
    t0 = time.perf_counter()
    ok, metrics = advance(state, dt_val)
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
                f"rt advance failed: dt={dt_val:.6g} -> next={float(dt_baseline if not isinstance(dt_baseline, AbstractTensor) else dt_baseline.item()):.6g} ({pretty_metrics(metrics)})"
            )
        return metrics, _restore_type(dt_baseline, ref_dt), _restore_type(dt_val, ref_dt)

    # Base proposal from allocation (thumbnailing simulated time to budget)
    # Ignore engine hard limit (dt_limit) in realtime to maintain pacing.
    dt_next = max(alloc_ms, 0.0) * 1e-3

    # Controller book-keeping still learns dt_max from velocities
    ctrl.update_dt_max(metrics.max_vel, dx)

    if is_enabled():
        dbg("ctrl").debug(
            "rt: "
            f"used_dt={dt_val:.6g} alloc={alloc_ms:.3f}ms cost={elapsed_ms:.3f}ms "
            + (f"dt_limit={metrics.dt_limit:.6g} " if metrics.dt_limit is not None else "")
            + f"-> dt_next={dt_next:.6g} | {pretty_metrics(metrics)}"
        )

    return metrics, _restore_type(dt_next, ref_dt), _restore_type(dt_val, ref_dt)

 
