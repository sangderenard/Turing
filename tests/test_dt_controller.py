import copy

from src.common.dt_scaler import Metrics
from src.cells.bath.dt_controller import (
    STController,
    Targets,
    step_with_dt_control,
)


class DummyState:
    def __init__(self):
        self.mass = 1.0
        self.vel = 0.1

    def copy_shallow(self):
        return copy.deepcopy(self)

    def restore(self, saved):
        self.__dict__.update(copy.deepcopy(saved.__dict__))

    def step(self, dt):
        # constant velocity, mass stays the same
        pass

    def total_mass(self):
        return self.mass

    def compute_metrics(self, prev_mass):
        return Metrics(
            max_vel=self.vel,
            max_flux=self.vel,
            div_inf=0.0,
            mass_err=abs(self.mass - prev_mass) / prev_mass,
        )


def test_dt_controller_step():
    state = DummyState()
    targets = Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6)
    ctrl = STController(dt_min=1e-6, dt_max=1e-2)
    dx = 0.1

    def advance(state, dt):
        prev_mass = getattr(state, "_last_mass", state.total_mass())
        state.step(dt)
        metrics = state.compute_metrics(prev_mass)
        state._last_mass = state.total_mass()
        return True, metrics

    metrics, dt_next = step_with_dt_control(state, 1e-4, dx, targets, ctrl, advance)
    assert dt_next > 0
    assert isinstance(metrics.max_vel, float)


def test_dt_controller_no_clamps():
    ctrl = STController(dt_min=None, dt_max=None)
    dt_small = ctrl.pi_update(dt_prev=1e-8, dt_pen=1e-9, osc=False)
    assert dt_small < 1e-6
    ctrl2 = STController(dt_min=1e-6, dt_max=None)
    dt_large = ctrl2.pi_update(dt_prev=1.0, dt_pen=10.0, osc=False)
    assert dt_large > 1.0
