import math
import pytest
from dataclasses import dataclass

import numpy as np

from src.common.dt_system.dt_scaler import Metrics
from src.cells.bath.dt_controller import STController, Targets, step_with_dt_control_used


@dataclass
class _DummyState:
    """Minimal state stub compatible with step_with_dt_control_used.

    No mutable fields are required because our advance() is side-effect free.
    """

    def copy_shallow(self):  # pragma: no cover - trivial
        return self

    def restore(self, saved):  # pragma: no cover - trivial
        return None


def _schedule_piecewise(idx: int) -> float:
    """Velocity step profile: baseline → spike → calm.

    - iters [0, 9]:    v = 1.0
    - iters [10, 13]:  v = 12.0 (spike)
    - iters [14, +):   v = 0.3  (calm)
    """
    if idx < 10:
        return 1.0
    if idx < 14:
        return 12.0
    return 0.3


@pytest.mark.dt
def test_dt_scaler_step_response(capsys):
    """Exercise the dt controller response to a velocity spike and cooldown.

    This focuses on qualitative behavior and reports metrics:
    - reaction time to reach close to the spike-limited CFL target,
    - minimum dt during the spike window (shrink factor),
    - recovery trend after the spike ends (growth factor),
    - envelope-limited dt_max trajectory vs. observed dt.
    """
    dx = 0.01
    targets = Targets(cfl=0.5, div_max=1e12, mass_max=1e12)
    ctrl = STController()
    state = _DummyState()

    # Initial dt seeded from baseline CFL
    v0 = _schedule_piecewise(0)
    dt_target0 = targets.cfl * dx / v0
    dt = dt_target0

    N = 80
    v_series = []
    dt_series = []
    dt_target_series = []
    dt_max_series = []

    def advance(_state, _dt):
        # Metrics for current loop index (bound via closure on v_series length)
        i = len(v_series)
        v = _schedule_piecewise(i)
        m = Metrics(max_vel=v, max_flux=v, div_inf=0.0, mass_err=0.0, sim_frame=i)
        return True, m

    for i in range(N):
        v = _schedule_piecewise(i)
        v_series.append(v)
        dt_target = targets.cfl * dx / max(v, 1e-30)
        dt_target_series.append(dt_target)

        _m, dt_next, dt_used = step_with_dt_control_used(
            state, dt, dx, targets, ctrl, advance
        )
        # Record after controller update (dt_used equals proposed input here)
        dt_series.append(dt_used)
        dt_max_series.append(ctrl.dt_max)
        dt = dt_next

    # Metrics around the spike
    spike_start = 10
    spike_end = 13
    dt_before = dt_series[spike_start - 1]
    dt_min_spike = float(np.min(dt_series[spike_start : spike_end + 3]))
    shrink_factor = dt_min_spike / max(dt_before, 1e-30)

    # Reaction steps: within 20% of the spike-limited target
    spike_target = dt_target_series[spike_start]
    reaction_steps = None
    for k in range(spike_start, min(spike_start + 12, N)):
        if dt_series[k] <= 1.2 * spike_target:
            reaction_steps = k - spike_start
            break

    # Recovery trend: growth factor 20 iterations after spike
    rec_idx = min(spike_end + 20, N - 1)
    growth_factor_20 = dt_series[rec_idx] / max(dt_min_spike, 1e-30)

    # Qualitative assertions (keep generous bounds to avoid flakiness)
    assert shrink_factor < 0.6  # shrinks noticeably on spike
    assert growth_factor_20 > 1.2  # recovers significantly after spike
    if reaction_steps is not None:
        assert reaction_steps <= 6  # reacts within a small number of iterations

    # Human-friendly summary (printed on success as well; captured by pytest)
    calm_target = targets.cfl * dx / 0.3
    env_after_20 = dx / (12.0 * (0.95 ** 20))
    print("=== dt scaler response summary ===")
    print(f"dx={dx}, CFL={targets.cfl}")
    print(f"baseline target dt = {dt_target0:.6g}")
    print(f"spike target dt    = {spike_target:.6g}")
    print(f"calm target dt     = {calm_target:.6g}")
    print(f"min dt in spike    = {dt_min_spike:.6g} (shrink x{shrink_factor:.3f})")
    if reaction_steps is not None:
        print(f"reaction steps     = {reaction_steps} (to within 20% of spike target)")
    print(f"growth@+20 steps   = x{growth_factor_20:.3f} from spike minimum")
    print(f"dt_max@+20 (env)   = {env_after_20:.6g} (envelope-limited)")
    # also print a compact row for quick visual inspection
    row = {
        "dt_before": dt_before,
        "dt_min_spike": dt_min_spike,
        "dt_+20": dt_series[rec_idx],
        "dt_max_+20": dt_max_series[rec_idx],
    }
    print("row:", {k: float(f"{v:.6g}") for k, v in row.items()})
