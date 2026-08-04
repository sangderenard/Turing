from __future__ import annotations

import numpy as np

from src.cells.bath.voxel_fluid import VoxelFluidParams, VoxelMACFluid
from src.common.dt_system.fluid_mechanics.voxel_fluid_engine import VoxelFluidEngine
from src.common.dt_system.fluid_mechanics.voxel_mac_aot import (
    initial_voxel_mac_arenas,
    make_managed_voxel_reference,
)


def test_voxel_adapter_uses_actual_mac_state_names_and_reports_divergence():
    simulation = VoxelMACFluid(VoxelFluidParams(6, 4, 1, gravity=(0.0, 0.0, 0.0)))
    engine = VoxelFluidEngine(simulation)

    state = engine.get_state()

    assert set(state) == {"u", "v", "w", "pr", "T", "S"}
    ok, metrics, next_state = engine.step_with_state(state, 0.001)
    assert ok
    assert metrics.div_inf >= 0.0
    assert set(next_state) == set(state)


def test_native_mac_topology_maps_are_integer_arenas_with_solid_wall_masks():
    feeds = initial_voxel_mac_arenas(6, 4)

    assert feeds["cell_left"].dtype == np.int64
    assert feeds["pressure_u_left"].shape == ((6 + 1) * 4,)
    assert feeds["pressure_v_down"].shape == (6 * (4 + 1),)
    assert np.count_nonzero(feeds["u_boundary_mask"] == 0.0) == 8
    assert np.count_nonzero(feeds["v_boundary_mask"] == 0.0) == 12


def test_repository_voxel_solver_advances_through_managed_time_runtime():
    reference = make_managed_voxel_reference(8, 6)

    report = reference.advance(0.004)

    assert report.exact_landing
    assert reference.runtime.current_time == 0.004
    assert report.result.metrics.div_inf < 1.0e-8
