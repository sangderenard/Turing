import time

import pytest

from src.common.dt_system.dt_controller import Targets, STController, step_realtime_once
from src.common.dt_system.dt_scaler import Metrics


class DummyState:
    def copy_shallow(self):
        return self
    def restore(self, saved):
        pass


def make_advance(cost_ms: float, dt_limit: float | None = None):
    def advance(state, dt):
        # Busy-wait to simulate cost; short loop to approximate wall-clock
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) * 1000.0 < cost_ms:
            pass
        return True, Metrics(max_vel=1.0, max_flux=1.0, div_inf=0.0, mass_err=0.0, dt_limit=dt_limit)
    return advance


@pytest.mark.dt
def test_realtime_single_step_uses_allocation_and_ignores_limit():
    state = DummyState()
    targets = Targets(cfl=0.5, div_max=1e3, mass_max=1e3)
    ctrl = STController(dt_min=1e-12)

    # Engine costs ~3 ms per step; allocation is 5 ms; dt_limit = 2e-3
    # Realtime ignores dt_limit for pacing; dt_next should be alloc-based
    advance = make_advance(cost_ms=3.0, dt_limit=2e-3)
    metrics, dt_next, dt_used = step_realtime_once(
        state,
        dt_current=1e-3,
        dx=1.0,
        targets=targets,
        ctrl=ctrl,
        advance=advance,
        alloc_ms=5.0,
    )
    assert abs(dt_next - 0.005) < 1e-6
    assert metrics.proc_ms >= 0.0

    # Without a limit, dt_next should follow allocation as well
    advance2 = make_advance(cost_ms=2.0, dt_limit=None)
    metrics2, dt_next2, _ = step_realtime_once(
        state,
        dt_current=1e-3,
        dx=1.0,
        targets=targets,
        ctrl=ctrl,
        advance=advance2,
        alloc_ms=7.0,
    )
    assert abs(dt_next2 - 0.007) < 1e-6
    assert metrics2.proc_ms >= 0.0
