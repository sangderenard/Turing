"""One conformance battery for the two managed-time execution modes.

Tau contracts are exercised separately in ``test_time_contracts``.  They are
participant claims inside scientific reasoning, not additional execution
modes.
"""
from __future__ import annotations

import pytest

from src.common.dt_system.dt_controller import STController, Targets
from src.common.dt_system.dt_graph import GraphBuilder, MetaLoopRunner
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.engine_api import DtCompatibleEngine, EngineRegistration
from src.common.dt_system.realtime import RealtimeConfig, RealtimeState
from src.common.dt_system.state_table import StateTable


class ModeProbe(DtCompatibleEngine):
    def __init__(self, ceiling=0.01):
        self.value = 0.0
        self.world_time = 0.0
        self.observer_time = 0.0
        self.causal_ceiling_dt = float(ceiling)

    def step(self, dt, state=None, state_table=None):
        self.value += float(dt)
        return True, Metrics(0.0, 0.0, 0.0, 0.0), state

    def get_state(self, state=None):
        return state

    def snapshot(self):
        return self.value, self.world_time, self.observer_time

    def restore(self, snapshot):
        self.value, self.world_time, self.observer_time = snapshot


def _registered(engine):
    table = StateTable()
    engine.register(table, lambda _: {"pos": (0.0, 0.0), "mass": 0.0}, [engine])
    return table


@pytest.mark.parametrize(
    "realtime, expected_ok, expected_advance, expected_slip",
    [(False, False, 0.0, 0.0), (True, True, 0.01, 0.01)],
)
def test_causal_ceiling_has_mode_specific_semantics(
        realtime, expected_ok, expected_advance, expected_slip):
    engine = ModeProbe()
    table = _registered(engine)

    ok, metrics, _ = engine.step_with_state(
        {}, 0.02, realtime=realtime, state_table=table)

    assert ok is expected_ok
    assert engine.value == pytest.approx(expected_advance)
    assert engine.world_time == pytest.approx(expected_advance)
    assert metrics.advanced_dt == pytest.approx(expected_advance)
    if expected_slip:
        assert metrics.error_present[9].item()
        assert metrics.error_channels[9].item() == pytest.approx(expected_slip)
    else:
        assert not metrics.error_present[9].item()


def test_admissible_scientific_step_and_rollback_are_exact():
    engine = ModeProbe()
    table = _registered(engine)
    checkpoint = engine.snapshot()

    ok, metrics, _ = engine.step_with_state(
        {}, 0.005, realtime=False, state_table=table)
    assert ok
    assert metrics.advanced_dt == pytest.approx(0.005)
    assert engine.world_time == pytest.approx(0.005)

    engine.restore(checkpoint)
    assert engine.snapshot() == checkpoint


def test_realtime_budget_mode_learns_cost_and_penalty_by_schedule_identity():
    engine = ModeProbe(ceiling=1.0)
    table = _registered(engine)
    targets = Targets(cfl=1.0, div_max=1.0, mass_max=1.0)
    config = RealtimeConfig(budget_ms=2.0, slack=1.0)
    realtime_state = RealtimeState()
    registration = EngineRegistration(
        name="probe", engine=engine, targets=targets, dx=1.0, localize=False)
    node = GraphBuilder(STController(dt_min=1e-9), targets, 1.0).round(
        dt=0.001, engines=[registration], realtime_config=config,
        realtime_state=realtime_state, state_table=table)
    runner = MetaLoopRunner(
        realtime_config=config, realtime_state=realtime_state,
        realtime=True, state_table=table)

    runner.run_round(node, dt=0.001, state_table=table)
    advance = runner._schedule[0]

    assert realtime_state.proc_ms_ma[id(advance)] > 0.0
    assert realtime_state.penalty_ma[id(advance)] >= 1.0
    scheduled = {id(item) for item in runner._schedule}
    assert set(realtime_state.proc_ms_ma) == scheduled
    assert set(realtime_state.penalty_ma) == scheduled
    assert table.get("dt_tape", advance.label, "metrics").proc_ms > 0.0
