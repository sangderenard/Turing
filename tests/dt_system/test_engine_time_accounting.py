from __future__ import annotations

import pytest

from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.engine_api import DtCompatibleEngine
from src.common.dt_system.state_table import StateTable


class ClockedEngine(DtCompatibleEngine):
    def __init__(self, *, self_clocking: bool = False) -> None:
        self.value = 0.0
        self.self_clocking = self_clocking
        self.causal_ceiling_dt = 0.1

    def step(self, dt, state, state_table):
        self.value += float(dt)
        if self.self_clocking:
            self.world_time += float(dt)
        return True, Metrics(0.0, 0.0, 0.0, 0.0), state

    def get_state(self, state=None):
        return state


def _registered(engine):
    table = StateTable()
    engine.register(
        table,
        lambda _: {"pos": (0.0, 0.0), "mass": 0.0},
        [0],
    )
    return table


def test_scientific_causal_ceiling_rejects_without_advancing():
    engine = ClockedEngine()
    table = _registered(engine)

    ok, metrics, _ = engine.step_with_state(
        {}, 0.2, realtime=False, state_table=table
    )

    assert not ok
    assert engine.value == 0.0
    assert engine.world_time == 0.0
    assert engine.observer_time == 0.0
    assert metrics.dt_limit == pytest.approx(0.1)
    assert metrics.advanced_dt == 0.0


def test_realtime_causal_clip_reports_slip_and_actual_advance():
    engine = ClockedEngine()
    table = _registered(engine)

    ok, metrics, _ = engine.step_with_state(
        {}, 0.2, realtime=True, state_table=table
    )

    assert ok
    assert engine.value == pytest.approx(0.1)
    assert engine.world_time == pytest.approx(0.1)
    assert engine.observer_time == pytest.approx(0.1)
    assert metrics.advanced_dt == pytest.approx(0.1)
    assert metrics.error_channels["time_slip"] == pytest.approx(0.1)


def test_engine_owned_clock_is_not_double_advanced():
    engine = ClockedEngine(self_clocking=True)
    table = _registered(engine)

    ok, metrics, _ = engine.step_with_state(
        {}, 0.05, realtime=False, state_table=table
    )

    assert ok
    assert engine.world_time == pytest.approx(0.05)
    assert engine.observer_time == pytest.approx(0.05)
    assert metrics.advanced_dt == pytest.approx(0.05)
