# -*- coding: utf-8 -*-
"""Relocated: dt controller under common/dt_system."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable

from ..tensors.abstraction import AbstractTensor

try:  # NumPy is the canonical lightweight backend
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

from .dt_scaler import Metrics, _scalar, coerce_metrics
from .dt import SuperstepPlan, SuperstepResult
from .error_channels import empty_channels, ENERGY_J, POWER_W, SHADOW_GROWTH

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
    error_limits: AbstractTensor = field(default_factory=empty_channels)
    error_limits_present: AbstractTensor = field(default_factory=empty_channels)
    # Energy/power time scale.  When set, a core that publishes the error
    # channels ``energy_j`` (stored energy) and ``power_w`` (magnitude of the
    # rate at which energy is being exchanged) is pinned so one step may
    # exchange at most this fraction of its stored energy:
    # ``dt <= fraction * energy_j / power_w``.  This is a time measured from
    # the system's own response, not from a stiffness formula, and it bounds
    # the NEXT attempt before it is tried.  A step that observed no exchange
    # at all (``power_w == 0``) carries no information about a safe larger
    # step, so dt is held rather than grown.
    energy_exchange_fraction: float | None = None
    # Shadow-trajectory amplification (see ``dt_system.shadow``).  When set,
    # a core advanced through ``shadow_advance`` publishes ``shadow_growth``,
    # the measured factor by which a small perturbation grew during the
    # step, and the next attempt is pinned so a step may amplify by at most
    # this factor: ``dt <= dt * ln(shadow_growth_max) / ln(shadow_growth)``.
    shadow_growth_max: float | None = None


def _shadow_dt_limit(dt_tensor, metrics: Metrics, targets: "Targets"):
    growth_max = getattr(targets, "shadow_growth_max", None)
    if growth_max is None:
        return None
    channels = metrics.error_channels
    if not bool(metrics.error_present[SHADOW_GROWTH].item()):
        return None
    from .shadow import shadow_dt_limit

    return shadow_dt_limit(
        float(dt_tensor.item()), _scalar(channels[SHADOW_GROWTH]), float(growth_max),
    )


def _energy_time_limit(metrics: Metrics, targets: "Targets"):
    """The energy time limit and its presence, as two numerical outputs.

    Absence must cross the helper boundary explicitly. An optional Python
    return loses its presence in native linking and reads back as a present
    zero, which would clamp the next step to zero.
    """

    fraction = getattr(targets, "energy_exchange_fraction", None)
    if fraction is None:
        return 0.0, 0.0
    energy = metrics.error_channels[ENERGY_J:ENERGY_J + 1]
    power = metrics.error_channels[POWER_W:POWER_W + 1]
    present = (metrics.error_present[ENERGY_J:ENERGY_J + 1] * metrics.error_present[POWER_W:POWER_W + 1]
               * energy.isfinite() * power.isfinite() * (energy > 0.0) * (power > 0.0))
    safe_power = AbstractTensor.where(power > 0.0, power, AbstractTensor.ones_like(power))
    limit = AbstractTensor.where(
        present, float(fraction) * energy / safe_power, AbstractTensor.zeros_like(energy))
    return float(limit.item()), float(present.item())


def _no_exchange_observed(metrics: Metrics, targets: "Targets") -> bool:
    fraction = getattr(targets, "energy_exchange_fraction", None)
    channels = metrics.error_channels
    return (
        fraction is not None
        and bool(metrics.error_present[POWER_W].item())
        and _scalar(channels[POWER_W]) <= 0.0
    )


def _participant_bound(spans, dt_proposed, dt_current, fraction):
    """Apply the existing per-participant time law to declared spans."""

    from .participants import exchange_time_bound

    return exchange_time_bound(spans, float(fraction), float(dt_proposed),
                     None if dt_current is None else float(dt_current))


def _apply_energy_sidechain(dt_next, dt_tensor, metrics: Metrics, targets: "Targets"):
    """Pin the next proposal by the energy/power time scale, if published.

    Metrics carries this attempt's publication buffers directly.
    """

    # Per participant first, when the state publishes that way: each binding
    # exchange_time applies as itself, a dilating or subcycling participant pins nobody,
    # and a HOLD prevents growth beyond the current step without shrinking it.
    if int(metrics.pub_exchange_time.shape[0]) > 0:
        fraction = getattr(targets, "energy_exchange_fraction", None)
        if fraction is not None:
            bound = _participant_bound(
                metrics, float(dt_next.item()), float(dt_tensor.item()),
                fraction,
            )
            if bound is not None:
                dt_next = AbstractTensor.minimum(dt_next, bound)

    limit, limit_present = _energy_time_limit(metrics, targets)
    if limit_present:
        dt_next = AbstractTensor.minimum(dt_next, AbstractTensor.tensor(limit))
    if _no_exchange_observed(metrics, targets):
        dt_next = AbstractTensor.minimum(dt_next, dt_tensor)
    growth_limit = _shadow_dt_limit(dt_tensor, metrics, targets)
    if growth_limit is not None:
        dt_next = AbstractTensor.minimum(dt_next, AbstractTensor.tensor(growth_limit))
    return dt_next


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


DistributionFn = Callable[[Metrics, "Targets", float], "AbstractTensor | float"]


def _propose_dt_pen(
    metrics: Metrics,
    targets: "Targets",
    dx,
    distribution,
):
    """Map (metrics, targets, dx) -> dt_pen (smaller is stricter).

    ``distribution``, when supplied, replaces the built-in CFL-plus-error-
    ratio proposal below, so a stateful core with no velocity/length-scale
    concept (a thermal, chemical, or contact-equilibrium system) can propose
    its own timescale from whichever of its own metrics matter, instead of
    being forced through ``max_vel``.
    """
    if distribution is not None:
        return distribution(metrics, targets, dx)
    # Default: CFL from the one velocity metric, softened by the worst ratio
    # across every declared error channel (never just one field alone).
    dt_cfl = targets.cfl * dx / max(metrics.max_vel, 1e-30)
    energy_limit, energy_present = _energy_time_limit(metrics, targets)
    if energy_present:
        dt_cfl = min(float(dt_cfl), energy_limit)
    # The aggregate channel span is always judged against the controller's
    # configured defaults. Participant rows add their own explicitly declared
    # gates; they do not replace the aggregate judgment.
    ratios = AbstractTensor.where(
        metrics.error_present * targets.error_limits_present,
        metrics.error_channels / AbstractTensor.maximum(targets.error_limits, 1e-30),
        AbstractTensor.zeros_like(metrics.error_channels),
    )
    channel_penalty = float(AbstractTensor.maximum(ratios.max(), 1.0).item())
    if int(metrics.pub_values.shape[0]) > 0:
        from .participants import worst_penalty

        channel_penalty = max(channel_penalty, float(worst_penalty(metrics).item()))
    penalty = max(
        metrics.div_inf / targets.div_max,
        metrics.mass_err / targets.mass_max,
        channel_penalty,
        1.0,
    )
    return dt_cfl / penalty


def step_with_dt_control_used(state,
                             dt,
                             dx,
                             targets: Targets,
                             ctrl: STController,
                             advance,
                             retries: int = 0,
                             max_retries: int | None = 256,
                             failures: list[tuple[float, Metrics, tuple[str, ...]]] | None = None,
                             ref=None,
                              attempt_log: list[dict] | None = None,
                              allow_unresolved: bool = True,
                              rollback_threshold_multiplier: float = 1.0,
                              rollback: bool = False,
                              distribution=None):
    if rollback_threshold_multiplier < 1.0:
        raise ValueError("rollback_threshold_multiplier must be >= 1.0")
    if failures is None:
        failures = []
    if ref is None:
        ref = dt

    dt_tensor = dt if isinstance(dt, AbstractTensor) else AbstractTensor.tensor(dt)

    # This used to be tail recursion: the halve-and-retry branch below called
    # this same function again instead of looping.  ``max_retries`` is a
    # caller policy (when to report defeat), not a safety bound -- passing
    # ``max_retries=None`` (a real, already-shipping call, see
    # ``balloon_tire_managed_window``) makes ``retries_exhausted`` permanently
    # False, so the ONLY thing that used to stop the recursion was Python's
    # own ~1000-frame call-stack limit turning it into an uncontrolled
    # ``RecursionError`` instead of the clean, named failure this function
    # already knows how to report.  A plain loop removes the stack risk
    # entirely; ``_ABSOLUTE_STEP_RETRY_CEILING`` below is a fixed circuit
    # breaker that is NOT a parameter, so no caller-supplied value --
    # including ``None`` -- can remove it.  Double-precision halving
    # underflows to a subnormal zero within ~1074 steps regardless of
    # ``dt_min``; the ceiling only needs comfortable margin over that.
    while True:
        dt_for_advance = _restore_type(dt_tensor, ref)

        # ``rollback=False`` is the fast, in-place lane: no shallow copy, no
        # restore, no retry.  Whatever ``advance`` leaves the state in IS
        # the new state, on rejection or not.  This is the "no-save
        # running" configuration for a gametime frame budget where the
        # copy/restore cost itself is what can't be afforded, not the
        # physics.  It always returns on its first pass through this loop.
        saved = state.copy_shallow() if rollback else None
        ok, metrics = advance(state, dt_for_advance)
        metrics = coerce_metrics(metrics)
        # Advance fills the attempt's declared publication buffers.
        spans = metrics
        has_participants = int(metrics.pub_exchange_time.shape[0]) > 0
        rollback_scale = float(rollback_threshold_multiplier)
        # A value between its ordinary limit and rollback_scale * limit is kept.
        # It still contributes its full ratio to the PI penalty below, so the next
        # frame corrects dt without paying for state restoration and recreation.
        soft_reasons: list[str] = []
        if metrics.mass_err > targets.mass_max:
            soft_reasons.append("mass_err")
        div_rollback_limit = float(targets.div_max) * 10.0
        if metrics.div_inf > div_rollback_limit:
            soft_reasons.append("div_inf")
        channel_values = metrics.error_channels
        channel_limits = targets.error_limits
        channel_mask = metrics.error_present * targets.error_limits_present
        soft_channels = channel_mask * (channel_values > channel_limits)
        participant_soft = metrics.pub_present * metrics.pub_limits_present * (
            metrics.pub_values > metrics.pub_limits)
        if bool(soft_channels.any().item()) or (
                has_participants and bool(participant_soft.any().item())):
            soft_reasons.append("channel limit")

        # Physical invalidity and an engine-declared hard failure never enter the
        # soft band. Numeric error channels roll back only at N times the same
        # boundary that used to cause immediate restoration when N == 1.
        reasons: list[str] = []
        if not ok:
            reasons.append("advance reported a physical-bound violation")
        if bool(metrics.hard_failure):
            reasons.append("hard_failure")
        if metrics.mass_err > targets.mass_max * rollback_scale:
            reasons.append("mass_err rollback limit")
        if metrics.div_inf > div_rollback_limit * rollback_scale:
            reasons.append("div_inf rollback limit")
        rollback_channels = channel_mask * (
            channel_values > channel_limits * rollback_scale)
        participant_rollback = metrics.pub_present * metrics.pub_limits_present * (
            metrics.pub_values > metrics.pub_limits * rollback_scale)
        if bool(rollback_channels.any().item()) or (
                has_participants and bool(participant_rollback.any().item())):
            reasons.append("channel rollback limit")
        rejected = bool(reasons)
        if not rollback:
            # No ``saved`` snapshot exists to restore, and retrying would need
            # one (the retry loop below re-attempts from the PRE-advance
            # state).  Report exactly what happened, still steer dt from every
            # declared metric via the same proposal the accepted path uses, and
            # move on: one ``advance`` call, no copy, no retry.  Always returns
            # on the loop's first pass.
            if attempt_log is not None:
                attempt_log.append({
                    "dt": float(dt_for_advance),
                    "accepted": not rejected,
                    "metrics": metrics,
                    "soft_channel_mask": soft_channels.copy(),
                    "rollback_channel_mask": rollback_channels.copy(),
                    "reasons": tuple(reasons),
                    "soft_reasons": soft_reasons,
                    "dt_min_retained_reasons": (),
                })
            if rejected:
                metrics.control_values[0] = float(dt_for_advance)
                metrics.control_present[0] = 1.0

            dt_pen = _propose_dt_pen(metrics, targets, dx, distribution)
            dt_next = ctrl.pi_update(
                dt_prev=dt_tensor,
                dt_pen=dt_pen,
                osc=(metrics.osc_flag or metrics.stiff_flag),
            )
            if metrics.dt_limit is not None:
                dt_next = AbstractTensor.minimum(dt_next, metrics.dt_limit)
            dt_next = _apply_energy_sidechain(dt_next, dt_tensor, metrics, targets)
            ctrl.update_dt_max(metrics.max_vel, dx)
            return metrics, _restore_type(dt_next, ref), _restore_type(dt_tensor, ref)
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
                metrics.control_values[2] = float(dt_for_advance)
                metrics.control_present[2] = 1.0
                metrics.control_values[3] = float(
                    len(floor_reasons) - int(bool(rollback_channels.any().item()))
                    + int(rollback_channels.sum().item()))
                metrics.control_present[3] = 1.0

                metrics.hard_failure = False
        if attempt_log is not None:
            attempt_log.append({
                "dt": float(dt_for_advance),
                "accepted": not rejected,
                "metrics": metrics,
                "soft_channel_mask": soft_channels.copy(),
                "rollback_channel_mask": rollback_channels.copy(),
                "reasons": tuple(reasons),
                "soft_reasons": soft_reasons,
                "dt_min_retained_reasons": floor_reasons,
            })
        # ``retries_exhausted`` has TWO independent, differently-justified
        # causes, never one blanket count:
        #  1. The caller's own declared patience (``max_retries``).  This is
        #     a policy choice and is absent entirely when ``max_retries`` is
        #     None -- a real, already-shipping call (see
        #     ``balloon_tire_managed_window``) that means "no numeric budget,
        #     keep going".
        #  2. ``ctrl.dt_min`` being unset ALSO removes the floor-retention
        #     exit above (that branch never fires with no floor to compare
        #     against), so halving would otherwise continue until literal
        #     float underflow (~1074 steps) -- far too many evaluations of
        #     ``advance`` for a real-time caller to ever pay for, and no
        #     caller-supplied number would change that fact.  What IS always
        #     true, regardless of any parameter, is IEEE-754 itself: halving
        #     eventually reaches a value where halving again is a genuine
        #     no-op (``x * 0.5 == x``), because the result has underflowed
        #     past what float64 can represent as distinct from it.  That
        #     point -- not an earlier, invented approximation of it -- is
        #     the one thing every caller in this configuration is ALREADY
        #     entitled to reach: a caller that passed ``max_retries=None``
        #     asked for exactly this, "keep going until there is truly
        #     nothing smaller to try," and deserves the full ~1074-step
        #     float64 range, not a shortcut that gives up sooner.  It is
        #     still bounded (this is what makes the loop conversion above
        #     safe at all) and still cheap: a few thousand pure-arithmetic
        #     halvings costs nothing next to even one ``advance`` call.
        #  3. There is no sense subdividing past the point where the step can
        #     no longer move the clock.  The test below used to be
        #     ``dt * 0.5 == dt`` -- machine epsilon of ZERO, reached only when
        #     dt denormalises, about 1074 halvings down.  But a step stops being
        #     a smaller step long before that: once ``scale + dt == scale`` for
        #     the scale it started from, adding it changes nothing, and every
        #     candidate below it is arithmetically distinct and physically
        #     identical.  From a 1e-3 window that point is near 2e-19, so the
        #     old rule spent roughly a THOUSAND further halvings, each one
        #     calling every law again, exploring steps that could not advance
        #     time.  The comment above claimed those halvings were pure
        #     arithmetic; they are not, the retry calls ``advance``.
        #
        #     The reported reason has always said "machine epsilon of its
        #     starting scale".  This is that rule, now actually implemented.
        scale = abs(float(_scalar(ref)))
        step = abs(float(dt_tensor.item()))
        numerically_exhausted = (
            ctrl.dt_min is None
            and (
                step * 0.5 == step                       # denormalised: nothing smaller
                or (scale > 0.0 and scale + step == scale)  # cannot move the clock
            )
        )
        retries_exhausted = (
            (max_retries is not None and retries >= max_retries)
            or numerically_exhausted
        )
        if rejected and retries_exhausted and allow_unresolved:
            # Best-effort callers may explicitly choose to retain a proposal after
            # exhausting refinement. Scientific callers leave this disabled: the
            # default is rollback, never silent commitment of a violating state.
            ctrl.clamp_events += 1
            metrics.hard_failure = True
            metrics.control_values[0] = float(dt_for_advance)
            metrics.control_present[0] = 1.0
            metrics.control_values[1] = float(len(failures) + 1)
            metrics.control_present[1] = 1.0

            # The trace rides on the metrics; a substep must not narrate. At a
            # pinned audio-rate interior this runs thousands of times a frame, and
            # printing each one buries the very thing it is reporting.
            lines = [
                "timestep controller proceeded unresolved after recorded attempts"
            ]
            for index, (dt_f, m, why) in enumerate(
                (*failures, (float(dt_for_advance), metrics, tuple(reasons))), 1,
            ):
                lines.append("  attempt rejected")
                lines.append("      rejected by recorded rule")
            if max_retries == 0:
                lines.append(
                    "  the substep is pinned, so there was no smaller candidate to "
                    "analyse; this is a physical rejection, not an exhausted search."
                )
            elif numerically_exhausted:
                lines.append(
                    "  dt has halved to machine epsilon of its starting scale "
                    "with no dt_min floor set; no smaller candidate is "
                    "numerically distinguishable, regardless of max_retries."
                )
            elif len({tuple(why) for _dt, _m, why in failures}) <= 1:
                lines.append(
                    "  every attempt was rejected for the same reason at every dt, "
                    "so subdividing further could not have resolved it."
                )
            metrics.control_values[9] = 0.0
            metrics.control_present[9] = 1.0
            metrics.unresolved_report = list(lines)
            # Fall through to the ordinary accepted path so the proposal for the
            # next step is computed the same way it always is.
            rejected = False
        if rejected:
            state.restore(saved)
            failures.append((float(dt_for_advance), metrics, tuple(reasons)))
            if retries_exhausted:
                ctrl.clamp_events += 1
                lines = ["timestep controller failed after recorded attempts"]
                for i, (dt_f, m, why) in enumerate(failures, 1):
                    lines.append("  attempt rejected")
                    lines.append("      rejected by recorded rule")
                if max_retries == 0:
                    lines.append(
                        "  the substep is pinned, so there is no smaller candidate "
                        "to analyse: this is a physical rejection, not a dt search "
                        "that ran out of room."
                    )
                elif numerically_exhausted:
                    lines.append(
                        "  dt has halved to machine epsilon of its starting scale "
                        "with no dt_min floor set; no smaller candidate is "
                        "numerically distinguishable, regardless of max_retries."
                    )
                elif len({tuple(why) for _dt, _m, why in failures}) == 1:
                    lines.append(
                        "  every attempt was rejected for the same reason at every "
                        "dt, so halving could not have resolved it."
                    )
                print("\n".join(lines))
                metrics.hard_failure = True
                metrics.control_values[0] = float(dt_for_advance)
                metrics.control_present[0] = 1.0
                metrics.control_values[1] = float(len(failures))
                metrics.control_present[1] = 1.0

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
            # Loop rather than recurse: same state, same policy, only the
            # proposed dt and the retry count change between attempts.
            dt_tensor = dt_half
            retries += 1
            continue

        dt_pen = _propose_dt_pen(metrics, targets, dx, distribution)
        dt_next = ctrl.pi_update(
            dt_prev=dt_tensor,
            dt_pen=dt_pen,
            osc=(metrics.osc_flag or metrics.stiff_flag),
        )
        # Sidechain limiter: clamp dt_next to any engine-provided absolute limit
        if metrics.dt_limit is not None:
            dt_next = AbstractTensor.minimum(dt_next, metrics.dt_limit)
        dt_next = _apply_energy_sidechain(dt_next, dt_tensor, metrics, targets)
        ctrl.update_dt_max(metrics.max_vel, dx)
        return metrics, _restore_type(dt_next, ref), _restore_type(dt_tensor, ref)


def step_with_dt_control(state, dt, dx, targets: Targets, ctrl: STController,
                         advance, retries: int = 0,
                         rollback_threshold_multiplier: float = 1.0,
                         rollback: bool = False,
                         distribution=None):
    metrics, dt_next, _dt_used = step_with_dt_control_used(
        state, dt, dx, targets, ctrl, advance, retries, ref=dt,
        rollback_threshold_multiplier=rollback_threshold_multiplier,
        rollback=rollback, distribution=distribution)
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
                  allow_unresolved: bool = True,
                  max_retries: int | None = 256,
                  rollback_threshold_multiplier: float = 1.0,
                  rollback: bool = False,
                  distribution=None,
                  schedule_lattice_steps: int = 0,
                  max_iters: int = 100_000):
    if rollback_threshold_multiplier < 1.0:
        raise ValueError("rollback_threshold_multiplier must be >= 1.0")
    if substep not in {"pinned", "steered"}:
        raise ValueError(
            f"unknown substep interior {substep!r}; expected 'pinned' or "
            "'steered'"
        )
    if schedule_lattice_steps < 0:
        raise ValueError("schedule_lattice_steps must be non-negative")
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
    # Pre-trial pin.  Every later attempt is bounded by the metrics of the
    # step before it, but the FIRST attempt of a round has nothing behind
    # it; a core that knows its own safe step (an authored integration step,
    # a stiffness bound) states it here and the opener never exceeds it.
    #
    # ``dt_limit_hint`` is accepted as EITHER a plain value or a callable.
    # PREFER THE PLAIN VALUE.  A core that has already advanced knows its
    # own limit as a number it just computed, and publishing that number is
    # a field assignment the compiled lane can see; wrapping it in a method
    # buys nothing and costs a call whose body the caller cannot inspect.
    # Reserve the callable form for a core that genuinely cannot know the
    # limit until it is asked -- one that has to poll a sub-engine, or that
    # derives the bound lazily because computing it is not free.  A method
    # that only does ``return self._x`` should be the attribute instead.
    #
    # Accepting both is not politeness: a float attribute silently fails
    # ``callable()`` and the pin is then skipped without a word, so the
    # core's own stability limit is computed every step and discarded.
    hint = getattr(state, "dt_limit_hint", None)
    if substep != "pinned" and hint is not None:
        declared = hint() if callable(hint) else hint
        if declared is not None and math.isfinite(float(declared)) and float(declared) > 0.0:
            dt_cap = AbstractTensor.minimum(dt_cap, AbstractTensor.tensor(float(declared)))
    last_dt_next = dt_cap
    # Never None: a value that is a real Metrics object on one control-flow
    # path and None on another is two different aggregate shapes merged by
    # the SAME variable, and a compiled build cannot unify that (confirmed
    # with tools/repro_keyed_construct.py -- constructing one aggregate
    # object once and only conditionally overwriting its fields compiles
    # cleanly; conditionally rebinding the variable to a freshly-built
    # object does not, even when both objects have identical field names).
    # This is that same "quiet, non-throwing incomplete window" record used
    # below when the loop truly never advances, just built up front instead
    # of only on demand.
    last_metrics = Metrics(
        0.0, 0.0, 0.0, 0.0,
        hard_failure=True,
        unresolved_report=[],
    )
    boundary_values = tuple(sorted({
        float(value)
        for value in event_boundaries
        if eps < float(value) < float(round_max_t.item()) - eps
    }))

    unresolved: list[Metrics] = []
    iters = 0
    iteration_cap_hit = False
    # A pathological parameter set (dt_min unset alongside targets no
    # combination of dt can satisfy, or a round_max/dt ratio in the
    # millions) can make this loop take a very long time to either finish
    # or naturally reach ``dt_used <= 0``.  Never retry unboundedly: after
    # ``max_iters`` rounds, stop and fall through to the SAME quiet,
    # non-throwing "incomplete window" report already used below for a
    # collapsed dt -- this must never raise here, a caller mid-frame
    # cannot afford an exception, only an honest partial result.
    while (round_max_t - total).item() > eps:
        if iters >= max_iters:
            iteration_cap_hit = True
            break
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
        # A collapsed controller proposal cannot advance simulated time.  Do
        # not call physics with zero or a negative value: authored kernels
        # commonly divide by dt, and doing so only corrupts state before the
        # existing incomplete-window path reports no progress.
        dt_try_value = float(dt_try.item())
        if dt_try_value <= 0.0:
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
            rollback=rollback,
            distribution=distribution,
        )
        last_metrics = metrics
        if float(metrics.control_values[0].item()) > 0.0:
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
            if schedule_lattice_steps > 0:
                # Adaptive proposals are decisions, and decisions need a
                # backend-independent boundary. Choose the greatest lattice
                # point no larger than the fully clamped proposal. A proposal
                # below the first point remains unchanged rather than being
                # raised above a safety limit. Remainder/event clamps happen
                # on dt_try, so exact authored landing points remain exact.
                lattice_quantum = round_max_t / float(schedule_lattice_steps)
                lattice_count = (dt_cap / lattice_quantum).floor()
                lattice_value = lattice_count * lattice_quantum
                if lattice_value.item() > 0.0:
                    dt_cap = lattice_value
        last_dt_next = dt_next

    if unresolved:
        first = unresolved[0]
        print(
            f"{len(unresolved)} of {iters} substep(s) advanced unresolved; "
            f"first at dt="
            f"{float(first.control_values[0].item()):.6g}"
        )
        for line in getattr(first, "unresolved_report", ())[1:]:
            print(f"  {line.strip()}")
    remaining = float((round_max_t - total).item())
    if remaining > eps:
        # last_metrics is never None now (see its initialization above), so
        # there is no fallback construction left to bind here.
        last_metrics.control_values[4] = float(round_max_t.item())
        last_metrics.control_present[4] = 1.0
        last_metrics.control_values[5] = float(total.item())
        last_metrics.control_present[5] = 1.0
        last_metrics.control_values[6] = remaining
        last_metrics.control_present[6] = 1.0
        last_metrics.control_values[7] = float(iters)
        last_metrics.control_present[7] = 1.0
        if iteration_cap_hit:
            last_metrics.control_values[8] = float(max_iters)
            last_metrics.control_present[8] = 1.0


    total_out = _restore_type(total, ref_dt)
    dt_next_out = _restore_type(last_dt_next, ref_dt)
    return total_out, dt_next_out, last_metrics


def run_superstep_plan(state,
                       plan: SuperstepPlan,
                       dx: float,
                       targets: Targets,
                       ctrl: STController,
                       advance,
                       distribution=None) -> SuperstepResult:
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
        rollback=plan.rollback,
        distribution=distribution,
        schedule_lattice_steps=plan.schedule_lattice_steps,
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

 
