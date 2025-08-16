import math
from dataclasses import dataclass

import pytest

from src.cells.bath.dt_controller import (
    Targets,
    STController,
    run_superstep_plan,
)
from src.common.dt_scaler import Metrics
from src.common.dt import SuperstepPlan


@dataclass
class FakeState:
    t: float = 0.0

    def copy_shallow(self):
        return FakeState(t=self.t)

    def restore(self, other: "FakeState"):
        self.t = float(other.t)


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
