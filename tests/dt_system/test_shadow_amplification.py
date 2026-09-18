"""Shadow-trajectory amplification as a universal predictive dt metric."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.common.dt_system.dt_controller import STController, Targets, run_superstep
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.shadow import ShadowedState, shadow_advance, shadow_dt_limit


class LinearCore:
    """x' = rate * x, explicit Euler: the step map is exactly (1 + rate*dt)."""

    def __init__(self, rate: float, x0):
        self.rate = float(rate)
        self.x = np.asarray(x0, dtype=np.float64).copy()

    def copy_shallow(self):
        return self.x.copy()

    def restore(self, snapshot):
        self.x[...] = snapshot


def _advance_linear(core: LinearCore, dt):
    core.x = core.x + float(dt) * core.rate * core.x
    return True, Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0)


def _shadowed(rate: float):
    core = LinearCore(rate, [1.0, -2.0, 0.5])
    return ShadowedState(
        core=core,
        read=lambda c: c.x,
        write=lambda c, value: c.x.__setitem__(..., value),
        make_shadow=lambda c: LinearCore(c.rate, c.x),
        delta=1.0e-6,
    )


@pytest.mark.dt
@pytest.mark.fast
def test_shadow_measures_the_exact_step_amplification():
    state = _shadowed(rate=50.0)
    advance = shadow_advance(_advance_linear)
    dt = 4.0e-3                                  # growth = 1 + 50 * 0.004 = 1.2
    _ok, metrics = advance(state, dt)
    assert metrics.error_channels["shadow_growth"] == pytest.approx(1.2, rel=1.0e-6)
    # Renormalised: the perturbation is back at its target size.
    assert state.perturbation_norm == pytest.approx(state._scale())


@pytest.mark.dt
@pytest.mark.fast
def test_shadow_growth_pins_the_next_step():
    state = _shadowed(rate=50.0)
    targets = Targets(cfl=1.0, div_max=1.0, mass_max=1.0, shadow_growth_max=1.05)
    ctrl = STController(dt_min=None, dt_max=None)
    log: list[dict] = []
    total, dt_next, _metrics = run_superstep(
        state, 4.0e-3, 4.0e-3, 1.0, targets, ctrl, shadow_advance(_advance_linear),
        allow_increase_mid_round=True, attempt_log=log, max_iters=50)
    assert float(total) >= 4.0e-3 - 1.0e-15
    first_growth = log[0]["metrics"].error_channels["shadow_growth"]
    assert first_growth == pytest.approx(1.2, rel=1.0e-6)
    # Every later attempt was held under the growth-derived limit.
    limit = shadow_dt_limit(4.0e-3, first_growth, 1.05)
    assert limit == pytest.approx(4.0e-3 * math.log(1.05) / math.log(1.2))
    assert all(row["dt"] <= limit * (1.0 + 1.0e-9) for row in log[1:])


@pytest.mark.dt
@pytest.mark.fast
def test_decaying_system_is_never_pinned_by_the_shadow():
    state = _shadowed(rate=-50.0)
    _ok, metrics = shadow_advance(_advance_linear)(state, 4.0e-3)
    growth = metrics.error_channels["shadow_growth"]
    assert growth == pytest.approx(0.8, rel=1.0e-6)
    assert shadow_dt_limit(4.0e-3, growth, 1.05) is None


@pytest.mark.dt
@pytest.mark.fast
def test_rollback_restores_the_shadow_with_the_core():
    state = _shadowed(rate=50.0)
    before = state.copy_shallow()
    shadow_advance(_advance_linear)(state, 4.0e-3)
    moved_core = state.core.x.copy()
    moved_shadow = state.shadow.x.copy()
    state.restore(before)
    assert not np.allclose(state.core.x, moved_core)
    assert not np.allclose(state.shadow.x, moved_shadow)
    assert np.allclose(state.shadow.x - state.core.x,
                       np.asarray(before[1]) - np.asarray(before[0]))
