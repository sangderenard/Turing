import numpy as np
from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams
from src.common.sim_hooks import SimHooks

def _record_step(fluid):
    records = []
    class Hook(SimHooks):
        def run_pre(self, eng, dt):
            records.append(dt)
    fluid.step(1e-6, hooks=Hook())
    return records[0]

def test_discrete_fluid_nocap_ignores_max_dt():
    pts = np.zeros((2, 3))
    params = FluidParams(max_dt=1e-9, nocap=True)
    fluid = DiscreteFluid(pts, None, None, None, params)
    assert fluid._stable_dt() > params.max_dt
    dt_used = _record_step(fluid)
    assert dt_used > params.max_dt

def test_discrete_fluid_cap_enforced():
    pts = np.zeros((2, 3))
    params = FluidParams(max_dt=1e-9, nocap=False)
    fluid = DiscreteFluid(pts, None, None, None, params)
    assert fluid._stable_dt() <= params.max_dt
    dt_used = _record_step(fluid)
    assert dt_used <= params.max_dt
