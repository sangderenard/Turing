"""The realtime lane hands out milliseconds, not pointer addresses.

`compile_allocations` returns a MAPPING keyed by `id(adv)`.
`MetaLoopRunner.run_round`'s realtime branch zipped the schedule against
that mapping, and zipping a mapping walks its KEYS -- so every engine
received a pointer address as its millisecond allocation.

Measured before the fix: an allocation of 2429530572816 ms, a `step_dt`
of 2.4e9 seconds, and a vehicle that travelled 1.8e21 metres in 120
frames. Nothing raised; the numbers were simply enormous, and enormous
numbers in a physics loop look like a physics bug for a long time before
they look like an indexing bug.

The property worth pinning is not "the fix is present" but "an engine is
stepped with the dt the caller asked for".
"""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.dt_system.dt_controller import STController, Targets
from src.common.dt_system.dt_graph import GraphBuilder, MetaLoopRunner
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.engine_api import DtCompatibleEngine, EngineRegistration
from src.common.dt_system.realtime import RealtimeConfig, RealtimeState
from src.common.dt_system.state_table import StateTable


class Recorder(DtCompatibleEngine):
    """An engine that does nothing but remember what dt it was handed."""

    def __init__(self):
        super().__init__()
        self.seen = []
        self._registration = set()

    def get_state(self, state=None):
        return state if isinstance(state, dict) else {}

    def snapshot(self):
        return None

    def restore(self, snapshot):
        return None

    def step(self, dt, state=None, state_table=None):
        self.seen.append(float(dt))
        return True, Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0,
                             mass_err=0.0), state


def _round(runner_state, dt):
    targets = Targets(cfl=1.0, div_max=1.0, mass_max=1.0)
    builder = GraphBuilder(ctrl=STController(dt_min=1e-7), targets=targets,
                           dx=0.1)
    table = StateTable()
    config = RealtimeConfig(budget_ms=16.7, slack=0.9)
    engines = [Recorder(), Recorder(), Recorder()]
    registrations = [
        EngineRegistration(name=f"probe_{index}", engine=engine,
                           targets=targets, dx=0.1, localize=True)
        for index, engine in enumerate(engines)
    ]
    runner = MetaLoopRunner(realtime_config=config,
                            realtime_state=runner_state, realtime=True,
                            state_table=table)
    node = builder.round(dt=dt, engines=registrations, realtime_config=config,
                         realtime_state=runner_state, state_table=table)
    runner.run_round(node, dt=dt, state_table=table)
    return engines


@pytest.mark.parametrize("dt", [1.0 / 60.0, 1.0e-3])
def test_every_engine_is_stepped_with_a_physical_dt(dt):
    """Seconds, not pointers.

    A pointer address read as milliseconds lands around 1e9 seconds. Any
    bound tighter than "smaller than a minute" catches it, and this one is
    deliberately loose so that legitimate allocation policy can change the
    step without breaking the test.
    """
    for engine in _round(RealtimeState(), dt):
        assert engine.seen, "engine was never stepped"
        for step in engine.seen:
            assert 0.0 < step < 60.0, step


def test_the_step_is_not_derived_from_object_identity():
    """The same graph twice gives the same steps.

    Pointer addresses differ between runs, so a step derived from one is
    not reproducible. A step derived from a declared budget is.
    """
    first = [e.seen for e in _round(RealtimeState(), 1.0 / 60.0)]
    second = [e.seen for e in _round(RealtimeState(), 1.0 / 60.0)]
    assert first == second


def test_without_allocations_the_caller_s_dt_is_honoured():
    """No `realtime_state` means no allocations, and then the round runs
    on the dt it was given rather than on a fallback."""
    dt = 1.0e-3
    for engine in _round(None, dt):
        assert engine.seen == pytest.approx([dt] * len(engine.seen))
