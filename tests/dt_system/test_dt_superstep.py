
from src.common.tensors import AbstractTensor
import math
from dataclasses import dataclass
import pytest

import pytest

from src.cells.bath.dt_controller import (
    Targets,
    STController,
    run_superstep,
    run_superstep_plan,
    step_with_dt_control_used,
)
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.dt import SuperstepPlan


@dataclass
class FakeState:
    t: float = 0.0

    def copy_shallow(self):
        return FakeState(t=self.t)

    def restore(self, other: "FakeState"):
        self.t = float(other.t)


@dataclass
class CountingState:
    value: float = 0.0
    restore_count: int = 0

    def copy_shallow(self):
        return CountingState(self.value, self.restore_count)

    def restore(self, other: "CountingState"):
        self.value = float(other.value)
        self.restore_count += 1


@pytest.mark.dt
def test_superstep_runs_until_the_requested_window_is_complete():
    state = CountingState()

    def advance(state_local: CountingState, dt: float):
        state_local.value += float(dt)
        return True, Metrics(0.0, 0.0, 0.0, 0.0)

    advanced, _dt_next, metrics = run_superstep(
        state,
        1.0,
        0.25,
        1.0,
        Targets(1.0, 1.0, 1.0),
        STController(),
        advance,
    )

    assert advanced == pytest.approx(1.0)
    assert state.value == pytest.approx(1.0)
    assert metrics.hard_failure is False
    assert not bool(metrics.control_present[6].item())


@pytest.mark.dt
@pytest.mark.parametrize("collapsed_limit", [0.0, -0.1])
def test_superstep_dead_ends_before_advancing_a_collapsed_proposal(collapsed_limit):
    state = CountingState()
    attempted: list[float] = []

    def advance(state_local: CountingState, dt: float):
        attempted.append(float(dt))
        state_local.value += float(dt)
        return True, Metrics(0.0, 0.0, 0.0, 0.0, dt_limit=collapsed_limit)

    advanced, _dt_next, metrics = run_superstep(
        state,
        1.0,
        0.25,
        1.0,
        Targets(1.0, 1.0, 1.0),
        STController(),
        advance,
    )

    assert attempted == [0.25]
    assert advanced == pytest.approx(0.25)
    assert state.value == pytest.approx(0.25)
    assert metrics.control_values[6].item() == pytest.approx(0.75)


@pytest.mark.dt
def test_soft_error_band_retains_state_and_steers_next_dt():
    state = CountingState()
    attempts: list[dict] = []

    def advance(state_local: CountingState, dt: float):
        state_local.value += float(dt)
        return True, Metrics(
            max_vel=1.0,
            max_flux=1.0,
            div_inf=0.0,
            mass_err=0.0,
            error_channels=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5]),
            error_present=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )

    _metrics, dt_next, dt_used = step_with_dt_control_used(
        state,
        0.1,
        1.0,
        Targets(1.0, 1.0, 1.0, error_limits=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                               error_limits_present=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),
        STController(),
        advance,
        attempt_log=attempts,
        rollback_threshold_multiplier=2.0,
    )

    assert dt_used == pytest.approx(0.1)
    assert state.value == pytest.approx(0.1)
    assert state.restore_count == 0
    assert dt_next < 1.0
    assert attempts[0]["accepted"] is True
    assert attempts[0]["reasons"] == ()
    assert attempts[0]["soft_channel_mask"].tolist()[15] == 1.0


@pytest.mark.dt
def test_builtin_error_reasons_are_stable_rule_tokens():
    def run(mass_err: float, div_inf: float, rollback_scale: float):
        attempts: list[dict] = []

        def advance(state_local: CountingState, dt: float):
            state_local.value += float(dt)
            return True, Metrics(1.0, 1.0, div_inf, mass_err)

        step_with_dt_control_used(
            CountingState(),
            0.1,
            1.0,
            Targets(1.0, 1.0, 1.0),
            STController(),
            advance,
            attempt_log=attempts,
            rollback_threshold_multiplier=rollback_scale,
            rollback=False,
        )
        return attempts[0]

    soft = run(1.5, 15.0, 2.0)
    assert soft["soft_reasons"] == ["mass_err", "div_inf"]
    hard = run(2.5, 25.0, 2.0)
    assert hard["reasons"] == (
        "mass_err rollback limit",
        "div_inf rollback limit",
    )


@pytest.mark.dt
def test_error_beyond_soft_band_restores_then_retries():
    state = CountingState()
    attempts: list[dict] = []

    def advance(state_local: CountingState, dt: float):
        state_local.value += float(dt)
        error = 2.5 if float(dt) > 0.05 else 0.5
        return True, Metrics(
            max_vel=1.0,
            max_flux=1.0,
            div_inf=0.0,
            mass_err=0.0,
            error_channels=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, error]),
            error_present=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )

    _metrics, _dt_next, dt_used = step_with_dt_control_used(
        state,
        0.1,
        1.0,
        Targets(1.0, 1.0, 1.0, error_limits=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                               error_limits_present=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),
        STController(),
        advance,
        rollback=True,
        attempt_log=attempts,
        rollback_threshold_multiplier=2.0,
    )

    assert dt_used == pytest.approx(0.05)
    assert state.value == pytest.approx(0.05)
    assert state.restore_count == 1
    assert [item["accepted"] for item in attempts] == [False, True]
    assert "rollback limit" in attempts[0]["reasons"][0]


@pytest.mark.dt
def test_physical_failure_ignores_soft_error_band():
    state = CountingState()

    def advance(state_local: CountingState, dt: float):
        state_local.value += float(dt)
        return float(dt) <= 0.05, Metrics(1.0, 1.0, 0.0, 0.0)

    _metrics, _dt_next, dt_used = step_with_dt_control_used(
        state,
        0.1,
        1.0,
        Targets(1.0, 1.0, 1.0),
        STController(),
        advance,
        rollback=True,
        rollback_threshold_multiplier=100.0,
    )

    assert dt_used == pytest.approx(0.05)
    assert state.value == pytest.approx(0.05)
    assert state.restore_count == 1


@pytest.mark.dt
def test_dt_floor_retains_even_a_hard_proposal_without_restore():
    state = CountingState()
    attempts: list[dict] = []

    def advance(state_local: CountingState, dt: float):
        state_local.value += float(dt)
        return False, Metrics(
            1.0, 1.0, 0.0, 0.0, hard_failure=True,
            error_channels=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]),
            error_present=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )

    metrics, _dt_next, dt_used = step_with_dt_control_used(
        state,
        1.0 / 1024.0,
        1.0,
        Targets(1.0, 1.0, 1.0, error_limits=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                               error_limits_present=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),
        STController(dt_min=1.0 / 1024.0),
        advance,
        rollback=True,
        attempt_log=attempts,
        rollback_threshold_multiplier=2.0,
    )

    assert dt_used == pytest.approx(1.0 / 1024.0)
    assert state.value == pytest.approx(1.0 / 1024.0)
    assert state.restore_count == 0
    assert metrics.hard_failure is False
    assert metrics.control_values[2].item() == pytest.approx(dt_used)
    assert attempts[0]["accepted"] is True
    assert attempts[0]["dt_min_retained_reasons"]


def make_advance(vel_fn, *, fail_over_dt: float | None = None):
    """Create an advance(state, dt) closure for tests.

    - On success, increments state.t by dt and returns Metrics determined by
      vel_fn(state.t).
    - If fail_over_dt is set and dt > fail_over_dt, returns (False, Metrics)
      without mutating state (simulating an instability that forces halving).
    """

    def advance(state: FakeState, dt: float):
        if fail_over_dt is not None and dt > float(fail_over_dt):
            # Return a harmless metrics payload; controller will halve via retry.
            return False, Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0)
        # Use velocity at start of the step for determinism
        v = float(vel_fn(state.t))
        # Progress state time on success
        state.t += float(dt)
        # Mass/div errors are fine; only CFL governs dt
        m = Metrics(max_vel=v, max_flux=v, div_inf=0.0, mass_err=0.0)
        return True, m

    return advance


def assert_non_increasing(seq, *, msg_prefix: str = ""):
    bad = []
    for i in range(1, len(seq)):
        if seq[i] > seq[i - 1] + 1e-15:
            bad.append((i - 1, seq[i - 1], i, seq[i]))
    assert not bad, (
        f"{msg_prefix}dt sequence must be non-increasing within a round; "
        f"violations: {bad}\nseq={seq}"
    )


@pytest.mark.dt
def test_superstep_exact_landing_and_monotone():
    # Constant velocity → constant CFL; PI may adjust dt but must not increase inside round.
    dx = 1.0
    targets = Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6)
    ctrl = STController(dt_min=1e-6)

    state = FakeState()
    attempted: list[float] = []

    def vel_fn(_t):
        return 2.0  # units/sec

    def advance_rec(state_local: FakeState, dt: float):
        attempted.append(float(dt))
        return make_advance(vel_fn)(state_local, dt)

    plan = SuperstepPlan(round_max=1.0, dt_init=0.30)
    res = run_superstep_plan(state, plan, dx, targets, ctrl, advance_rec)

    # Exact landing within tolerance
    assert math.isclose(res.advanced, plan.round_max, rel_tol=0, abs_tol=plan.eps), (
        f"advanced != round_max\nadvanced={res.advanced:.16e}\nround_max={plan.round_max:.16e}"
    )
    # Monotone (non-increasing) attempted dts inside the round
    assert_non_increasing(attempted, msg_prefix="constant vel: ")
    # Should require multiple micro-steps (dt_init above CFL-limited dt)
    assert res.steps >= 2, f"expected at least 2 micro-steps; got {res.steps}\nseq={attempted}"


@pytest.mark.dt
def test_superstep_allows_increase_when_enabled():
    # Velocity decays over time → CFL dt should grow if allowed.
    dx = 1.0
    targets = Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6)
    ctrl = STController(dt_min=1e-6)

    state = FakeState()
    attempted: list[float] = []

    def vel_fn(t):
        # Smoothly decaying velocity
        return 4.0 / (1.0 + t)

    def advance_rec(state_local: FakeState, dt: float):
        attempted.append(float(dt))
        return make_advance(vel_fn)(state_local, dt)

    plan = SuperstepPlan(round_max=0.6, dt_init=0.05, allow_increase_mid_round=True)
    res = run_superstep_plan(state, plan, dx, targets, ctrl, advance_rec)

    # Must make progress and land within eps
    assert math.isclose(res.advanced, plan.round_max, rel_tol=0, abs_tol=plan.eps), (
        f"landing error: advanced={res.advanced:.16e} vs {plan.round_max:.16e}"
    )
    # With increases allowed, sequence should contain at least one growth
    grew = any(attempted[i] > attempted[i - 1] + 1e-12 for i in range(1, len(attempted)))
    assert grew, f"expected a dt increase in sequence when allowed; seq={attempted}"


@pytest.mark.dt
@pytest.mark.fast
def test_superstep_schedule_lattice_stabilizes_low_bit_proposal_noise():
    def run(metric_scale: float):
        attempted: list[float] = []
        state = FakeState()

        def advance(state_local: FakeState, dt: float):
            attempted.append(float(dt))
            state_local.t += float(dt)
            velocity = (2.0 + state_local.t) * metric_scale
            return True, Metrics(velocity, velocity, 0.0, 0.0)

        result = run_superstep_plan(
            state,
            SuperstepPlan(
                round_max=0.6,
                dt_init=0.05,
                allow_increase_mid_round=True,
                schedule_lattice_steps=1 << 20,
            ),
            1.0,
            Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6),
            STController(dt_min=1e-6),
            advance,
        )
        return result, attempted

    incumbent, incumbent_attempts = run(1.0)
    perturbed, perturbed_attempts = run(1.0 + 1e-12)

    assert incumbent_attempts == perturbed_attempts
    assert incumbent.advanced == perturbed.advanced == pytest.approx(0.6)
    quantum = 0.6 / (1 << 20)
    # The caller-authored opener and exact final remainder are not adaptive
    # follow-up decisions. Every interior proposal between them is.
    for dt in incumbent_attempts[1:-1]:
        lattice_index = dt / quantum
        assert lattice_index == pytest.approx(round(lattice_index), abs=1e-9)


@pytest.mark.dt
@pytest.mark.fast
def test_superstep_rejects_a_negative_schedule_lattice():
    with pytest.raises(ValueError, match="schedule_lattice_steps"):
        run_superstep(
            FakeState(),
            0.1,
            0.1,
            1.0,
            Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6),
            STController(),
            make_advance(lambda _time: 1.0),
            schedule_lattice_steps=-1,
        )


@pytest.mark.dt
@pytest.mark.fast
def test_superstep_schedule_lattice_never_raises_a_subquantum_proposal():
    attempted: list[float] = []

    def advance(state_local: FakeState, dt: float):
        attempted.append(float(dt))
        state_local.t += float(dt)
        return True, Metrics(1.0, 1.0, 0.0, 0.0, dt_limit=1e-8)

    run_superstep(
        FakeState(),
        1e-3,
        1e-4,
        1.0,
        Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6),
        STController(),
        advance,
        allow_increase_mid_round=True,
        schedule_lattice_steps=1024,
        max_iters=2,
    )

    assert attempted == pytest.approx([1e-4, 1e-8])


@pytest.mark.dt
def test_halving_on_failure_and_clamped_flag():
    # Force failures for dt > threshold, verifying halving and 'clamped' result.
    dx = 1.0
    targets = Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6)
    ctrl = STController(dt_min=1e-6)

    state = FakeState()
    attempted: list[float] = []

    def vel_fn(_t):
        return 1.0

    def advance_rec(state_local: FakeState, dt: float):
        attempted.append(float(dt))
        return make_advance(vel_fn, fail_over_dt=0.20)(state_local, dt)

    plan = SuperstepPlan(round_max=0.5, dt_init=0.5)
    res = run_superstep_plan(state, plan, dx, targets, ctrl, advance_rec)

    # Expect that the first attempt exceeded the threshold and a halved retry occurred.
    assert attempted[0] > 0.20, f"expected initial attempt > 0.20; seq={attempted}"
    assert any(abs(x - attempted[0] * 0.5) < 1e-12 for x in attempted[1:]), (
        f"expected a halved retry after failure; seq={attempted}"
    )
    assert res.clamped, "result.clamped should be True when halving occurred"


@pytest.mark.dt
@pytest.mark.fast
def test_update_dt_max_decay_envelope():
    ctrl = STController(dt_min=1e-6)
    dx = 1.0

    # Start with a high velocity spike, then drop; dt_max should recover (increase)
    ctrl.update_dt_max(max_vel=100.0, dx=dx)
    dt_after_spike = ctrl.dt_max

    # Apply several lower velocities; as the envelope decays, dt_max should grow
    for _ in range(5):
        ctrl.update_dt_max(max_vel=1.0, dx=dx)
    dt_after_calm = ctrl.dt_max

    assert dt_after_calm > dt_after_spike, (
        f"dt_max did not recover after velocity drop:\n"
        f"after_spike={dt_after_spike:.3e} after_calm={dt_after_calm:.3e}"
    )


@pytest.mark.dt
@pytest.mark.fast
def test_superstep_returns_unclamped_proposal():
    """Controller proposals must survive the round cap for next frame."""
    dx = 1.0
    targets = Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6)
    ctrl = STController(dt_min=1e-6)

    state = FakeState()

    def vel_fn(_t):
        return 0.0

    plan = SuperstepPlan(round_max=1e-6, dt_init=1e-6)
    res = run_superstep_plan(state, plan, dx, targets, ctrl, make_advance(vel_fn))

    assert res.dt_next > plan.dt_init, (
        f"expected dt_next > dt_init when velocity is zero; got {res.dt_next}"
    )


@pytest.mark.dt
def test_controller_reports_and_raises_on_persistent_failure(capsys):
    """Controller should emit a failure report and raise after exhausting retries."""
    dx = 1.0
    targets = Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6)
    ctrl = STController(dt_min=1e-6)

    state = FakeState()

    def vel_fn(_t):
        return 1.0

    # fail_over_dt=0 ensures all attempts fail regardless of dt
    advance = make_advance(vel_fn, fail_over_dt=0.0)
    plan = SuperstepPlan(round_max=0.1, dt_init=0.1)

    with pytest.raises(RuntimeError):
        run_superstep_plan(state, plan, dx, targets, ctrl, advance)

    out = capsys.readouterr().out
    assert "timestep controller failed" in out


@pytest.mark.dt
def test_strict_controller_rolls_back_mutating_persistent_failure():
    """The default policy must never commit an exhausted bad proposal."""

    state = FakeState()
    targets = Targets(cfl=1.0, div_max=1.0, mass_max=1.0)
    ctrl = STController(dt_min=None, dt_max=None)

    def violating_advance(state_local: FakeState, dt: float):
        state_local.t += float(dt)
        return False, Metrics(1.0, 1.0, 0.0, 0.0)

    with pytest.raises(RuntimeError, match="failed to complete"):
        run_superstep_plan(
            state,
            SuperstepPlan(round_max=0.1, dt_init=0.1),
            1.0,
            targets,
            ctrl,
            violating_advance,
        )

    assert state.t == 0.0


@pytest.mark.dt
def test_nested_supersteps_compose():
    """Inner controllers may subdivide outer dt requests."""
    dx_outer = 1.0
    dx_inner = 0.1
    targets = Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6)
    outer_ctrl = STController(dt_min=1e-6)
    inner_ctrl = STController(dt_min=1e-6)

    outer_state = FakeState()
    inner_state = FakeState()
    inner_attempted: list[float] = []

    def inner_advance(state: FakeState, dt: float):
        inner_attempted.append(float(dt))
        state.t += float(dt)
        # High velocity enforces small CFL dt
        m = Metrics(max_vel=10.0, max_flux=10.0, div_inf=0.0, mass_err=0.0)
        return True, m

    def outer_advance(state: FakeState, dt: float):
        plan_inner = SuperstepPlan(round_max=float(dt), dt_init=0.01)
        res_inner = run_superstep_plan(inner_state, plan_inner, dx_inner, targets, inner_ctrl, inner_advance)
        state.t += res_inner.advanced
        m = Metrics(max_vel=1.0, max_flux=1.0, div_inf=0.0, mass_err=0.0)
        return True, m

    plan_outer = SuperstepPlan(round_max=0.2, dt_init=0.2)
    res_outer = run_superstep_plan(outer_state, plan_outer, dx_outer, targets, outer_ctrl, outer_advance)

    assert math.isclose(res_outer.advanced, plan_outer.round_max, rel_tol=0, abs_tol=plan_outer.eps)
    assert len(inner_attempted) > 1, "inner controller should have subdivided the dt"
    assert math.isclose(inner_state.t, plan_outer.round_max, rel_tol=0, abs_tol=plan_outer.eps)
