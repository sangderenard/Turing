"""Damping release as a dt-system wrapper around any core."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.common.dt_system.damping_release import DampingReleasedState, release_advance
from src.common.dt_system.dt_controller import STController, Targets, run_superstep
from src.common.dt_system.dt_scaler import Metrics


class Oscillator:
    """m x'' = -k x - c x', semi-implicit Euler; ``c`` is the LIVE damping."""

    def __init__(self, x0: float, k: float = 400.0, m: float = 1.0, c_physical: float = 2.0):
        self.x, self.v = float(x0), 0.0
        self.k, self.m, self.c_physical = k, m, c_physical
        self.c = c_physical

    def copy_shallow(self):
        return (self.x, self.v, self.c)

    def restore(self, snapshot):
        self.x, self.v, self.c = snapshot

    def energy(self) -> float:
        return 0.5 * self.k * self.x ** 2 + 0.5 * self.m * self.v ** 2


def _advance(core: Oscillator, dt):
    dt = float(dt)
    core.v += dt * (-core.k * core.x - core.c * core.v) / core.m
    core.x += dt * core.v
    return True, Metrics(max_vel=abs(core.v), max_flux=0.0, div_inf=0.0, mass_err=0.0)


def _apply(core: Oscillator, factor: float) -> None:
    core.c = core.c_physical * factor


def _run(initial_factor: float, seconds: float, dt: float = 1.0e-3):
    core = Oscillator(x0=0.1)
    state = DampingReleasedState(core, apply=_apply, initial_factor=initial_factor,
                                 release_time_s=0.05)
    advance = release_advance(_advance)
    energies, factors = [], []
    t = 0.0
    while t < seconds:
        _ok, metrics = advance(state, dt)
        energies.append(core.energy())
        factors.append(metrics.error_channels["damping_factor"])
        t += dt
    return state, energies, factors


@pytest.mark.dt
@pytest.mark.fast
def test_release_absorbs_the_transient_then_returns_to_physical_damping():
    released_state, released, factors = _run(initial_factor=20.0, seconds=0.2)
    _plain_state, plain, _ = _run(initial_factor=1.0, seconds=0.2)
    # Heavier early damping removes energy faster inside the release window.
    assert released[20] < plain[20]
    # After the window the multiplier is back at the physics.
    assert factors[0] == pytest.approx(20.0)
    assert released_state.released()
    assert released_state.core.c == pytest.approx(released_state.core.c_physical, rel=2.0e-3)


@pytest.mark.dt
@pytest.mark.fast
def test_release_clock_rolls_back_with_the_state():
    core = Oscillator(x0=0.1)
    state = DampingReleasedState(core, apply=_apply, initial_factor=10.0, release_time_s=0.05)
    advance = release_advance(_advance)
    snapshot = state.copy_shallow()
    for _ in range(30):
        advance(state, 1.0e-3)
    assert state.elapsed_s == pytest.approx(0.03)
    state.restore(snapshot)
    assert state.elapsed_s == 0.0
    assert core.c == pytest.approx(core.c_physical * 10.0)


@pytest.mark.dt
@pytest.mark.fast
def test_release_runs_under_the_dt_controller_unchanged():
    core = Oscillator(x0=0.1)
    state = DampingReleasedState(core, apply=_apply, initial_factor=10.0, release_time_s=0.05)
    targets = Targets(cfl=1.0, div_max=1.0, mass_max=1.0)
    ctrl = STController(dt_min=None, dt_max=1.0e-3)
    total, _dt_next, metrics = run_superstep(
        state, 0.02, 1.0e-3, 1.0, targets, ctrl, release_advance(_advance), max_iters=500)
    assert float(total) >= 0.02 - 1.0e-15
    assert 1.0 < metrics.error_channels["damping_factor"] < 10.0
    assert math.isfinite(core.energy())
