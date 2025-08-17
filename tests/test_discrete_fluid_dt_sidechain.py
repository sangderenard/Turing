import math

import numpy as np
import pytest

from src.cells.bath.discrete_fluid import DiscreteFluid
from src.common.dt_system.dt_controller import STController, Targets, step_with_dt_control_used
from src.common.dt_system.fluid_mechanics import BathDiscreteFluidEngine


@pytest.mark.dt
def test_discrete_fluid_dt_sidechain_clamps_next_dt():
    """Controller must clamp dt_next to engine-provided dt_limit (stable_dt).

    We construct a small DiscreteFluid demo, wrap it with the Bath adapter
    (which publishes dt_limit via Metrics), and ask the dt controller to step
    with an overly-large dt. The returned dt_next should be no larger than the
    adapter's dt_limit hint.
    """
    # Build a small, stable dam-break scene
    fluid = DiscreteFluid.demo_dam_break(n_x=6, n_y=8, n_z=6, h=0.06)
    eng = BathDiscreteFluidEngine(sim=fluid)

    # Engine stability hint (used as dt_limit sidechain)
    dt_hint = float(fluid._stable_dt())
    assert math.isfinite(dt_hint) and dt_hint > 0.0

    # Ask controller to use an exaggerated initial dt to force clamping
    dt_init = dt_hint * 25.0
    dx = float(fluid.kernel.h)

    # Targets are permissive; we rely on dt_limit to govern dt_next
    targets = Targets(cfl=0.5, div_max=1e3, mass_max=1e3)
    ctrl = STController(dt_min=1e-12)

    # Advance callback integrates via the engine to capture its Metrics (with dt_limit)
    def advance(state, dt):
        ok, m = eng.step(float(dt))
        return ok, m

    # Use the fluid as the 'state' since it provides copy_shallow/restore
    metrics, dt_next, dt_used = step_with_dt_control_used(
        state=fluid,
        dt=float(dt_init),
        dx=dx,
        targets=targets,
        ctrl=ctrl,
        advance=advance,
    )

    # Sanity on metrics
    assert metrics is not None
    assert metrics.dt_limit is not None and metrics.dt_limit > 0.0

    # Controller must clamp next dt to the engine hint (allow tiny epsilon)
    assert dt_next <= metrics.dt_limit + 1e-15
    # And we actually advanced a positive dt
    assert dt_used > 0.0
