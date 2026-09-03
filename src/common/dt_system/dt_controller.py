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

# This module's debug-logging calls (``if is_enabled(): dbg(...).debug(...)``)
# were removed entirely, not just guarded. Neither a runtime function-call
# condition nor a folded-constant one gets dead-branch-eliminated by the
# compiler -- it has no pass that drops an ``ast.If`` body once its test
# resolves to a known-false literal -- so logger objects and f-strings
# inside the branch still had no Fortran equivalent and left an
# unregistered, uncompilable region in every function that reached one.
# Deleting them was the actual fix; see tools/HANDOFF_2026-08-17_CRASH.md's
# sibling investigation for the rest of this compile chain's fixes.


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
        return _restore_type(dt_new, ref_prev)


def step_with_dt_control_used(state,
                             dt,
                             dx,
                             targets: Targets,
                             ctrl: STController,
                             advance,
                             retries: int = 0,
                             max_retries: int | None = 3,
                             failures: list[tuple[float, Metrics, tuple[str, ...]]] | None = None,
                             ref=None,
                              attempt_log: list[dict] | None = None,
                              allow_unresolved: bool = False,
                              rollback_threshold_multiplier: float = 1.0):
    if rollback_threshold_multiplier < 1.0:
        raise ValueError("rollback_threshold_multiplier must be >= 1.0")
    if failures is None:
        failures = []
    if ref is None:
        ref = dt

    dt_tensor = dt if isinstance(dt, AbstractTensor) else AbstractTensor.tensor(dt)
    dt_for_advance = _restore_type(dt_tensor, ref)

    saved = state.copy_shallow()
    ok, metrics = advance(state, dt_for_advance)
    metrics = coerce_metrics(metrics)
    rollback_scale = float(rollback_threshold_multiplier)
    # A value between its ordinary limit and rollback_scale * limit is kept.
    # It still contributes its full ratio to the PI penalty below, so the next
    # frame corrects dt without paying for state restoration and recreation.
    soft_reasons: list[str] = []
    if metrics.mass_err > targets.mass_max:
        soft_reasons.append(
            f"mass_err {float(metrics.mass_err):.3e} > "
            f"{float(targets.mass_max):.3e}"
        )
    div_rollback_limit = float(targets.div_max) * 10.0
    if metrics.div_inf > div_rollback_limit:
        soft_reasons.append(
            f"div_inf {float(metrics.div_inf):.3e} > "
            f"{div_rollback_limit:.3e}"
        )
    for name, limit in targets.error_limits.items():
        channel_error = float(metrics.error_channels.get(name, 0.0))
        if channel_error > float(limit):
            # Numeric values remain in metrics.error_channels.  The resident
            # reason sequence carries only a stable rule identity so the same
            # authored controller has a fixed native storage representation.
            soft_reasons.append(name)

    # Physical invalidity and an engine-declared hard failure never enter the
    # soft band. Numeric error channels roll back only at N times the same
    # boundary that used to cause immediate restoration when N == 1.
    reasons: list[str] = []
    if not ok:
        reasons.append("advance reported a physical-bound violation")
    if bool(metrics.hard_failure):
        reasons.append("hard_failure")
    if metrics.mass_err > targets.mass_max * rollback_scale:
        reasons.append(
            f"mass_err {float(metrics.mass_err):.3e} > "
            f"{float(targets.mass_max) * rollback_scale:.3e} rollback limit"
        )
    if metrics.div_inf > div_rollback_limit * rollback_scale:
        reasons.append(
            f"div_inf {float(metrics.div_inf):.3e} > "
            f"{div_rollback_limit * rollback_scale:.3e} rollback limit"
        )
    for name, limit in targets.error_limits.items():
        channel_error = float(metrics.error_channels.get(name, 0.0))
        channel_rollback_limit = float(limit) * rollback_scale
        if channel_error > channel_rollback_limit:
            reasons.append("channel rollback limit")
    rejected = bool(reasons)
    floor_reasons: tuple[str, ...] = ()
    if rejected and ctrl.dt_min is not None:
        dt_floor = float(
            ctrl.dt_min.item()
            if isinstance(ctrl.dt_min, AbstractTensor)
            else ctrl.dt_min
        )
        if float(dt_for_advance) <= dt_floor * (1.0 + 1.0e-12):
            # There is no smaller legal proposal. Retain the state and expose
            # exactly what the floor overruled so the following frame can
            # continue to adjust without a futile restore/recreate cycle.
            floor_reasons = tuple(reasons)
            soft_reasons.append("dt_min retained")
            reasons.clear()
            rejected = False
            ctrl.clamp_events += 1
            channels = dict(metrics.error_channels or {})
            channels["dt_min_retained"] = float(dt_for_advance)
            channels["dt_min_retained_violation_count"] = float(
                len(floor_reasons))
            metrics.error_channels = channels
            metrics.hard_failure = False
    if attempt_log is not None:
        attempt_log.append({
            "dt": float(dt_for_advance),
            "accepted": not rejected,
            "metrics": metrics,
            "reasons": tuple(reasons),
            "soft_reasons": (() if rejected else tuple(soft_reasons)),
            "dt_min_retained_reasons": floor_reasons,
        })
    retries_exhausted = max_retries is not None and retries >= max_retries
    if rejected and retries_exhausted and allow_unresolved:
        # Best-effort callers may explicitly choose to retain a proposal after
        # exhausting refinement. Scientific callers leave this disabled: the
        # default is rollback, never silent commitment of a violating state.
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
            lines.append("      rejected by recorded rule")
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
        # Fall through to the ordinary accepted path so the proposal for the
        # next step is computed the same way it always is.
        rejected = False
    if rejected:
        state.restore(saved)
        failures.append((float(dt_for_advance), metrics, tuple(reasons)))
        if retries_exhausted:
            ctrl.clamp_events += 1
            lines = [f"timestep controller failed after {len(failures)} attempts:"]
            for i, (dt_f, m, why) in enumerate(failures, 1):
                lines.append(
                    f"  attempt {i}: dt={dt_f:.6g} mass_err={m.mass_err:.3e} "
                    f"div_inf={m.div_inf:.3e} max_vel={m.max_vel:.3e}"
                )
                lines.append("      rejected by recorded rule")
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
            metrics.hard_failure = True
            channels = dict(metrics.error_channels or {})
            channels["dt_unresolved"] = float(dt_for_advance)
            channels["dt_unresolved_attempts"] = float(len(failures))
            metrics.error_channels = channels
            # A zero used-dt is the native-safe failure status.  The caller can
            # report a partial window without relying on Python exception
            # semantics, which repository SSA does not yet represent.
            return metrics, _restore_type(dt_tensor * 0.5, ref), _restore_type(
                AbstractTensor.tensor(0.0), ref
            )
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
            allow_unresolved=allow_unresolved,
            rollback_threshold_multiplier=rollback_threshold_multiplier,
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
    return metrics, _restore_type(dt_next, ref), _restore_type(dt_tensor, ref)


def step_with_dt_control(state, dt, dx, targets: Targets, ctrl: STController,
                         advance, retries: int = 0,
                         rollback_threshold_multiplier: float = 1.0):
    metrics, dt_next, _dt_used = step_with_dt_control_used(
        state, dt, dx, targets, ctrl, advance, retries, ref=dt,
        rollback_threshold_multiplier=rollback_threshold_multiplier)
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
                  eps: float = 1e-15,
                  event_boundaries: tuple[float, ...] = (),
                  attempt_log: list[dict] | None = None,
                  allow_unresolved: bool = False,
                  max_retries: int | None = 3,
                  rollback_threshold_multiplier: float = 1.0):
    if rollback_threshold_multiplier < 1.0:
        raise ValueError("rollback_threshold_multiplier must be >= 1.0")
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
    while (round_max_t - total).item() > eps:
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
            max_retries=0 if substep == "pinned" else max_retries,
            ref=ref_dt,
            attempt_log=attempt_log,
            allow_unresolved=allow_unresolved,
            rollback_threshold_multiplier=rollback_threshold_multiplier,
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
        if last_metrics is None:
            last_metrics = Metrics(0.0, 0.0, 0.0, 0.0, hard_failure=True)
        channels = dict(last_metrics.error_channels or {})
        channels["superstep_window_requested_s"] = float(round_max_t.item())
        channels["superstep_window_advanced_s"] = float(total.item())
        channels["superstep_window_remaining_s"] = remaining
        channels["superstep_iteration_count"] = float(iters)
        last_metrics.error_channels = channels

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
        rollback_threshold_multiplier=plan.rollback_threshold_multiplier,
    )
    total_val = float(total.item() if isinstance(total, AbstractTensor) else total)
    dt_next_val = float(dt_next.item() if isinstance(dt_next, AbstractTensor) else dt_next)
    plan_dt_init_val = float(plan.dt_init.item() if isinstance(plan.dt_init, AbstractTensor) else plan.dt_init)
    plan_round_max_val = float(
        plan.round_max.item()
        if isinstance(plan.round_max, AbstractTensor)
        else plan.round_max
    )
    if plan_round_max_val - total_val > plan.eps:
        raise RuntimeError(
            "adaptive timestep controller failed to complete its requested "
            f"window: advanced={total_val:.17g} "
            f"round_max={plan_round_max_val:.17g}"
        )
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
        return metrics, _restore_type(dt_baseline, ref_dt), _restore_type(dt_val, ref_dt)

    # Base proposal from allocation (thumbnailing simulated time to budget)
    # Ignore engine hard limit (dt_limit) in realtime to maintain pacing.
    dt_next = max(alloc_ms, 0.0) * 1e-3

    # Controller book-keeping still learns dt_max from velocities
    ctrl.update_dt_max(metrics.max_vel, dx)

    return metrics, _restore_type(dt_next, ref_dt), _restore_type(dt_val, ref_dt)

 
