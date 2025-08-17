import numpy as np
import pytest

from src.common.dt_system.fluid_mechanics.softbody_engine import SoftbodyEngineWrapper
from src.common.dt_system.dt_controller import STController, Targets, step_with_dt_control


class DummySolver:
    def __init__(self):
        self.X = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        self.V = np.array([[0.0, -1.0, 0.0]], dtype=np.float64)

    def step(self, dt):
        self.X += self.V * dt

    def copy_shallow(self):
        return (self.X.copy(), self.V.copy())

    def restore(self, snap):
        self.X[:], self.V[:] = snap

    def max_vertex_speed(self):
        return float(np.max(np.linalg.norm(self.V, axis=1)))

    def _stable_dt(self):
        return 0.05


def penetration(solver):
    y_min = float(np.min(solver.X[:, 1]))
    return max(0.0, -y_min)


@pytest.mark.dt
@pytest.mark.fast
def test_softbody_wrapper_metrics():
    solver = DummySolver()
    eng = SoftbodyEngineWrapper(solver=solver, penetration_fn=penetration)
    ok, m, _ = eng.step(0.1)
    assert ok
    assert m.max_vel == pytest.approx(1.0)
    assert m.div_inf == pytest.approx(0.1)
    assert m.dt_limit == pytest.approx(0.05)


@pytest.mark.dt
@pytest.mark.fast
def test_softbody_wrapper_controller_clamps_dt():
    solver = DummySolver()
    eng = SoftbodyEngineWrapper(solver=solver, penetration_fn=penetration)
    targets = Targets(cfl=1.0, div_max=0.05, mass_max=1.0)
    ctrl = STController(dt_min=1e-6, dt_max=1.0)

    def advance(state, dt):
        ok, m, _ = eng.step(dt)
        return ok, m

    metrics, dt_next = step_with_dt_control(solver, 0.1, 1.0, targets, ctrl, advance)
    assert metrics.div_inf > 0.0
    assert dt_next < 0.1
