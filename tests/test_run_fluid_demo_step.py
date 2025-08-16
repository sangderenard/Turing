from types import SimpleNamespace

from src.cells.softbody.demo import numpy_sim_coordinator as coord
from src.common.dt_scaler import Metrics


class DummyEngine:
    def __init__(self):
        self.params = SimpleNamespace(cfl=0.5, dx=1.0)
        self.dx = 1.0
        self.calls = []

    def total_mass(self):
        return 1.0

    def step(self, dt, substeps=1, *, hooks=None):
        self.calls.append(("step", dt))

    def _substep(self, dt):  # pragma: no cover - should not be used
        self.calls.append(("_substep", dt))

    def compute_metrics(self, prev_mass):
        self.calls.append(("metrics", prev_mass))
        return Metrics(0.0, 0.0, 0.0, 0.0)

    # ``run_superstep`` expects shallow copy/restore hooks for rollback. The
    # demo engine is stateless so these can be no-ops.
    def copy_shallow(self):
        return self

    def restore(self, saved):
        return None


def test_run_fluid_demo_uses_step(monkeypatch):
    engine = DummyEngine()
    monkeypatch.setattr(coord, "make_fluid_engine", lambda kind, dim: engine)

    args = SimpleNamespace(fluid="discrete", sim_dim=2, frames=1, dt=1e-3, debug_render=False)
    coord.run_fluid_demo(args)

    call_names = [c[0] for c in engine.calls]
    assert "step" in call_names
    assert "_substep" not in call_names
    assert call_names.index("metrics") > call_names.index("step")
