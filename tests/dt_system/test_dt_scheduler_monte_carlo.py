import math
import pytest
from dataclasses import dataclass

import numpy as np

from src.cells.bath.dt_controller import (
    Targets,
    STController,
    run_superstep_plan,
)
from src.common.dt_system.dt import SuperstepPlan
from src.common.dt_system.dt_scaler import Metrics


@dataclass
class EngineState:
    """Minimal simulation state with copy/restore hooks."""

    t: float = 0.0

    def copy_shallow(self):
        return EngineState(t=self.t)

    def restore(self, other: "EngineState"):
        self.t = float(other.t)


class DynamicEngine:
    """Fake engine with time-varying velocity and optional failure threshold."""

    def __init__(self, params, fail_over_dt=None):
        self.params = params
        self.fail_over_dt = fail_over_dt

    def velocity(self, t):
        a, b, c, d = self.params
        # Always positive, smoothly varying velocity
        return abs(a + b * t + c * math.sin(d * t)) + 1e-9

    def advance(self, state: EngineState, dt: float):
        if self.fail_over_dt is not None and dt > self.fail_over_dt:
            return False, Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0)
        v = self.velocity(state.t)
        state.t += float(dt)
        m = Metrics(max_vel=v, max_flux=v, div_inf=0.0, mass_err=0.0)
        return True, m


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
def test_dt_scheduler_monte_carlo():
    rng = np.random.default_rng(12345)
    for _ in range(50):
        round_max = float(rng.uniform(0.05, 1.0))
        dt_init = float(rng.uniform(round_max * 0.05, round_max * 0.5))
        allow_increase = bool(rng.integers(0, 2))
        dt_min = float(rng.uniform(1e-6, 1e-3))
        dt_init = max(dt_init, dt_min * 2.0)
        dt_max = None
        if rng.random() < 0.5:
            dt_max = float(rng.uniform(dt_init, round_max * 1.5))
        ctrl = STController(dt_min=dt_min, dt_max=dt_max)
        targets = Targets(
            cfl=float(rng.uniform(0.1, 0.9)),
            div_max=float(rng.uniform(1e-6, 1e-3)),
            mass_max=float(rng.uniform(1e-9, 1e-5)),
        )
        dx = float(rng.uniform(0.1, 2.0))
        params = (
            float(rng.uniform(0.0, 3.0)),  # a
            float(rng.uniform(-2.0, 2.0)),  # b
            float(rng.uniform(0.0, 3.0)),  # c
            float(rng.uniform(0.0, 4.0)),  # d
        )
        engine = DynamicEngine(params)
        state = EngineState()
        attempted = []

        def advance_rec(st: EngineState, dt: float):
            ok, m = engine.advance(st, dt)
            if ok:
                attempted.append(float(dt))
            return ok, m

        plan = SuperstepPlan(
            round_max=round_max,
            dt_init=dt_init,
            allow_increase_mid_round=allow_increase,
        )
        res = run_superstep_plan(state, plan, dx, targets, ctrl, advance_rec)

        assert math.isclose(res.advanced, plan.round_max, rel_tol=0.0, abs_tol=plan.eps)
        assert math.isclose(state.t, plan.round_max, rel_tol=0.0, abs_tol=plan.eps)
        if not allow_increase:
            assert_non_increasing(attempted, msg_prefix="mc: ")
        assert res.dt_next > 0.0
        assert res.steps >= 1
        assert res.metrics is not None
        if ctrl.dt_min is not None:
            assert res.dt_next >= ctrl.dt_min - 1e-12
