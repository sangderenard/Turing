from __future__ import annotations

import ast
import numpy as np
from pathlib import Path
import pytest

from src.common.abstract_tensor_state_machine import (
    is_abstract_tensor_state_machine,
)
from src.common.dt_system.fluid_mechanics.columnar_multifluid_engine import (
    ColumnarMultifluidConfig,
    ColumnarMultifluidEngine,
    ColumnarMultifluidState,
)
from src.common.dt_system.state_table import StateTable
from src.common.tensors import AbstractTensor
from src.compiler.state_machine_ast import plan_marked_state_machines


def _config(**changes):
    values = dict(
        tile_shape=(3, 2),
        slots_per_column=4,
        material_names=("solid", "sand", "water", "player"),
        material_mobility=(0.0, 0.35, 1.0, 0.0),
        player_capture_radius=1.6,
        surface_smoothing=0.0,
    )
    values.update(changes)
    return ColumnarMultifluidConfig(**values)


def _state(config=None):
    config = config or _config()
    state = ColumnarMultifluidState.regular(((0, 0), (1, 0)), config)
    state.fill_columns(
        (
            ((2, 1), (3, 0), (1, 2)),
            ((0, 1), (2, 3), (1, 1)),
        ),
        (
            ((0, 2), (1, 0), (0, 2)),
            ((0, 2), (2, 2), (0, 2)),
        ),
        config,
    )
    return state, config


def test_columnar_engine_is_first_class_state_machine_with_unit_voxel_storage():
    state, config = _state()

    assert is_abstract_tensor_state_machine(ColumnarMultifluidEngine)
    assert state.voxel_occupied.shape == (2, 3, 2, 4)
    assert state.voxel_centroid.shape == (2, 3, 2, 4, 3)
    assert state.voxel_material_fraction.shape == (2, 3, 2, 4, 4)
    assert state.voxel_material_fraction.sum().item() == pytest.approx(
        state.voxel_occupied.to_dtype("float32").sum().item()
    )
    state.validate(config)


def test_columnar_engine_is_ast_visible_as_the_special_state_machine_contract():
    source = Path(
        "src/common/dt_system/fluid_mechanics/columnar_multifluid_engine.py"
    ).read_text(encoding="utf-8")
    plans, shortfalls = plan_marked_state_machines(ast.parse(source))

    assert shortfalls == ()
    plan = next(
        item for item in plans if item.class_name == "ColumnarMultifluidEngine"
    )
    assert plan.state_field == "phase"
    assert plan.case_methods == ((0, "advance_columns"),)


def test_settling_is_one_bulk_managed_transition_and_is_reversible():
    state, config = _state()
    engine = ColumnarMultifluidEngine(state, config)
    snapshot = state.copy_shallow()
    before = np.asarray(state.voxel_centroid.tolist())

    ok, metrics, returned = engine.step(0.02, state, StateTable())

    assert ok and returned is state
    assert metrics.advanced_dt == pytest.approx(0.02)
    assert state.managed_time.item() == pytest.approx(0.02)
    assert not np.array_equal(np.asarray(state.voxel_velocity.tolist()), 0.0)
    state.restore(snapshot)
    assert state.managed_time.item() == 0.0
    assert np.array_equal(np.asarray(state.voxel_centroid.tolist()), before)


def test_player_voxel_opens_one_parallel_local_physics_domain():
    state, config = _state()
    player_index = state.place_player(0, 1, 0, 0, 3, config)
    engine = ColumnarMultifluidEngine(state, config)

    engine.step(0.01, state, StateTable())

    flat_domain = np.asarray(state.voxel_physics_domain.tolist()).reshape(-1)
    flat_occupied = np.asarray(state.voxel_occupied.tolist()).reshape(-1)
    assert flat_domain[player_index] == 1
    assert np.count_nonzero((flat_domain == 1) & flat_occupied) > 1
    # Far occupied material remains column-constrained.
    assert np.count_nonzero((flat_domain == 0) & flat_occupied) > 0


def test_managed_tick_moves_player_sinusoidally_and_loads_spring_sheet():
    state, config = _state()
    player_index = state.place_player(0, 1, 0, 0, 3, config)
    engine = ColumnarMultifluidEngine(state, config)
    snapshot = state.copy_shallow()
    before = np.asarray(state.voxel_centroid.tolist()).reshape(-1, 3)[
        player_index
    ].copy()

    ok, metrics, _ = engine.step(0.025, state, StateTable())

    after = np.asarray(state.voxel_centroid.tolist()).reshape(-1, 3)[player_index]
    displacement = np.asarray(state.column_displacement.tolist())
    assert ok
    assert metrics.advanced_dt == pytest.approx(0.025)
    assert not np.allclose(after, before)
    assert displacement.min() < 0.0
    assert np.count_nonzero(displacement) > 1
    assert np.asarray(state.column_surface_z.tolist()) == pytest.approx(
        np.asarray(state.column_rest_surface_z.tolist()) + displacement
    )

    state.restore(snapshot)
    assert np.array_equal(
        np.asarray(state.column_displacement.tolist()),
        np.zeros(config.tile_shape).reshape((1, *config.tile_shape)).repeat(2, axis=0),
    )
    assert np.asarray(state.voxel_centroid.tolist()).reshape(-1, 3)[
        player_index
    ] == pytest.approx(before)


def test_column_stencil_proposes_only_mobile_downhill_material_flux():
    config = _config(tile_shape=(2, 1), slots_per_column=4)
    state = ColumnarMultifluidState.regular(((0, 0),), config)
    state.fill_columns([[[1], [4]]], [[[0], [2]]], config)
    engine = ColumnarMultifluidEngine(state, config)

    engine.step(0.01, state, StateTable())

    flux = np.asarray(state.transfer_flux.tolist())
    assert flux[..., 0].max() == 0.0
    assert flux[..., 2].max() > 0.0
    assert flux[..., 3].max() == 0.0


def test_youngman_extracts_the_same_bulk_column_and_player_surface_field():
    config = _config(tile_shape=(1, 1), slots_per_column=2)
    state = ColumnarMultifluidState.regular(((0, 0),), config)
    state.fill_columns([[[1]]], [[[0]]], config)
    engine = ColumnarMultifluidEngine(state, config)
    engine.step(0.01, state, StateTable())
    field = engine.surface_field(state, smoothing=0.0)

    samples = AbstractTensor.tensor(((0.5, 0.5, 0.5), (1.4, 0.5, 0.5)))
    values = field(samples).tolist()
    assert values[0] < 0.0
    assert values[1] > 0.0

    extraction = engine.extract_surface(
        (-0.25, 0.5, 1.25),
        (-0.25, 0.5, 1.25),
        (-0.25, 0.5, 1.25),
        state,
        smoothing=0.0,
    )
    assert extraction.triangle_count > 0
    assert extraction.solver_samples is not None
    assert extraction.solver_samples.sample_count > 0


def test_publish_committed_exposes_centroid_as_world_reference():
    state, config = _state()
    engine = ColumnarMultifluidEngine(state, config)
    table = StateTable()
    ok, metrics, _ = engine.step(0.01, state, table)
    assert ok
    engine.publish_committed(state, table, metrics)

    assert table.get("columnar_world", "voxels", "centroid") is state.voxel_centroid
    assert table.get("columnar_world", "managed", "time") is state.managed_time
