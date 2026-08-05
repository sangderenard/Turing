from __future__ import annotations

import numpy as np
import pytest

from src.cells.bath.voxel_fluid import VoxelFluidParams, VoxelMACFluid
from src.common.dt_system.fluid_mechanics.voxel_fluid_engine import VoxelFluidEngine


def test_voxel_adapter_uses_actual_mac_state_names_and_reports_divergence():
    simulation = VoxelMACFluid(VoxelFluidParams(6, 4, 1, gravity=(0.0, 0.0, 0.0)))
    engine = VoxelFluidEngine(simulation)

    state = engine.get_state()

    assert set(state) == {"u", "v", "w", "pr", "T", "S"}
    ok, metrics, next_state = engine.step_with_state(state, 0.001)
    assert ok
    assert metrics.div_inf >= 0.0
    assert set(next_state) == set(state)


def test_voxel_metrics_measure_divergence_and_salinity_mass_change():
    simulation = VoxelMACFluid(VoxelFluidParams(
        4, 3, 1, gravity=(0.0, 0.0, 0.0),
    ))
    simulation.u[1, :, :] = 0.25
    simulation.S[1, 1, 0] = 2.0

    metrics = simulation.compute_metrics(prev_mass=1.0)

    assert metrics.max_vel == 0.25
    assert metrics.div_inf == pytest.approx(0.25 / simulation.dx)
    assert metrics.mass_err == 1.0
