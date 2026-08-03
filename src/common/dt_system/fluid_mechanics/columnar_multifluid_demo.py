"""Managed-dt visual demo for the columnar multifluid state machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .columnar_multifluid_engine import (
    ColumnarMultifluidConfig,
    ColumnarMultifluidEngine,
    ColumnarMultifluidState,
)
from ..dt_controller import STController, Targets
from ..state_table import StateTable
from ..time_runtime import ManagedTimeRuntime, TimeWindowRequest


def build_demo_state():
    config = ColumnarMultifluidConfig(
        tile_shape=(10, 7),
        slots_per_column=8,
        material_names=("solid", "sand", "water", "gas", "player"),
        material_mobility=(0.0, 0.4, 1.0, 0.75, 0.0),
        player_capture_radius=2.15,
        transfer_height_threshold=0.8,
        surface_smoothing=0.06,
        collision_damping=0.08,
    )
    state = ColumnarMultifluidState.regular(((0, 0),), config)
    width, height = config.tile_shape
    fill = np.ones((1, width, height), dtype=np.int32)
    material = np.zeros_like(fill)
    for x in range(width):
        for y in range(height):
            distance = ((x - 6.0) ** 2 + (y - 3.5) ** 2) ** 0.5
            if x in {0, width - 1} or y in {0, height - 1}:
                fill[0, x, y] = 4
                material[0, x, y] = 0
            elif distance < 3.2:
                fill[0, x, y] = max(2, 6 - int(distance))
                material[0, x, y] = 1
            elif x < 3 and 2 <= y <= 5:
                fill[0, x, y] = 2
                material[0, x, y] = 2
    state.fill_columns(fill, material, config, density=1.0)
    player_column = (4, 3)
    player_slot = int(fill[0, player_column[0], player_column[1]]) - 1
    state.place_player(
        0, player_column[0], player_column[1], player_slot, 4, config
    )
    return state, config


def run_managed_frame(state, config):
    engine = ColumnarMultifluidEngine(state, config)
    table = StateTable()

    def advance(managed_state, dt):
        ok, metrics, returned = engine.step(dt, managed_state, table)
        if returned is not managed_state:
            raise RuntimeError("columnar engine replaced managed state identity")
        return ok, metrics

    runtime = ManagedTimeRuntime(
        state,
        advance,
        dx=1.0,
        targets=Targets(
            cfl=1.0,
            div_max=1.0,
            mass_max=1.0e-6,
            error_limits={
                "columnar_material_unit_error": 1.0e-6,
                "columnar_nonfinite": 0.0,
            },
        ),
        controller=STController(dt_min=1.0e-8),
        initial_time=0.0,
    )
    report = runtime.advance(TimeWindowRequest(1, 0, 0.0, 0.08, 0.02))
    metrics = report.result.metrics
    if metrics is None:
        raise RuntimeError("managed frame produced no metrics")
    engine.publish_committed(state, table, metrics)
    return engine, table, report


def extract_demo_surface(engine, state, config):
    width, height = config.tile_shape
    maximum = float(state.column_surface_z.max().item()) + 1.0
    # Offset unit-spaced samples put vertices on both sides of unit-box
    # surfaces without evaluating an unnecessary fine all-pairs diagnostic.
    x_axis = np.arange(-0.25, width + 0.751, 1.0)
    y_axis = np.arange(-0.25, height + 0.751, 1.0)
    z_axis = np.arange(config.floor_z - 0.25, maximum + 0.751, 1.0)
    return engine.extract_surface(x_axis, y_axis, z_axis, state)


def render_demo(state, config, extraction, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    surface = np.asarray(state.column_surface_z.tolist())[0].T
    domains = np.asarray(state.voxel_physics_domain.tolist())[0]
    occupied = np.asarray(state.voxel_occupied.tolist())[0]
    domain_map = np.max(np.where(occupied, domains, 0), axis=-1).T
    flux = np.asarray(state.transfer_flux.tolist())
    flux_strength = flux.sum(axis=(1, 2)).reshape(config.tile_shape).T

    figure = plt.figure(figsize=(15.2, 7.8), facecolor="#dcecf1")
    grid = figure.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.65))
    axis_surface = figure.add_subplot(grid[0, 0])
    axis_domain = figure.add_subplot(grid[0, 1])
    axis_flux = figure.add_subplot(grid[1, :2])
    axis_mesh = figure.add_subplot(grid[:, 2], projection="3d")

    image = axis_surface.imshow(
        surface, origin="lower", cmap="Blues", interpolation="nearest"
    )
    axis_surface.set_title("continuous column surface z")
    figure.colorbar(image, ax=axis_surface, fraction=0.046)

    axis_domain.imshow(
        domain_map, origin="lower", cmap="winter", interpolation="nearest"
    )
    axis_domain.set_title("player-unified physics domain")

    flux_image = axis_flux.imshow(
        flux_strength, origin="lower", cmap="magma", interpolation="nearest"
    )
    axis_flux.set_title("four-neighbor material transfer proposal")
    figure.colorbar(flux_image, ax=axis_flux, fraction=0.025)

    triangles = np.asarray(extraction.triangles)
    if len(triangles):
        mesh = Poly3DCollection(
            triangles,
            facecolor=(0.35, 0.72, 0.82, 0.78),
            edgecolor=(0.12, 0.36, 0.43, 0.12),
            linewidth=0.25,
        )
        axis_mesh.add_collection3d(mesh)
    player_indices = np.asarray(state.player_voxel_index.tolist(), dtype=np.int64)
    centroids = np.asarray(state.voxel_centroid.tolist()).reshape(-1, 3)
    if len(player_indices):
        player = centroids[player_indices]
        axis_mesh.scatter(
            player[:, 0], player[:, 1], player[:, 2],
            s=110, c="#f7ffff", edgecolors="#087f9b", linewidths=2,
            label="player voxel",
        )
        axis_mesh.legend(loc="upper left")
    axis_mesh.set_xlim(-0.25, config.tile_shape[0] + 0.25)
    axis_mesh.set_ylim(-0.25, config.tile_shape[1] + 0.25)
    axis_mesh.set_zlim(config.floor_z - 0.25, max(2.0, float(surface.max()) + 1.0))
    axis_mesh.set_box_aspect((config.tile_shape[0], config.tile_shape[1], 6))
    axis_mesh.view_init(elev=28, azim=-128)
    axis_mesh.set_title(
        f"YoungMan bulk crossing · {extraction.triangle_count:,} triangles"
    )
    axis_mesh.set_xlabel("grid x")
    axis_mesh.set_ylabel("grid y")
    axis_mesh.set_zlabel("continuous z")

    for axis in (axis_surface, axis_domain, axis_flux):
        axis.set_xlabel("grid x")
        axis.set_ylabel("grid y")
    figure.suptitle(
        "Columnar multifluid game state — one occupied slot is one physics voxel",
        color="#173f4d",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def run_demo(output_directory: Path):
    state, config = build_demo_state()
    engine, table, report = run_managed_frame(state, config)
    extraction = extract_demo_surface(engine, state, config)
    output_directory.mkdir(parents=True, exist_ok=True)
    image_path = output_directory / "columnar_multifluid.png"
    state_path = output_directory / "world_state.json"
    render_demo(state, config, extraction, image_path)
    payload = {
        "schema": "columnar-multifluid-game-state-v1",
        "managed_time": float(state.managed_time.item()),
        "accepted_dts": tuple(float(value) for value in report.result.accepted_dts),
        "tile_shape": config.tile_shape,
        "slots_per_column": config.slots_per_column,
        "material_names": config.material_names,
        "tile_coord": state.tile_coord.tolist(),
        "column_centroid": state.column_centroid.tolist(),
        "voxel_occupied": state.voxel_occupied.tolist(),
        "voxel_centroid": state.voxel_centroid.tolist(),
        "voxel_velocity": state.voxel_velocity.tolist(),
        "voxel_material_fraction": state.voxel_material_fraction.tolist(),
        "voxel_physics_domain": state.voxel_physics_domain.tolist(),
        "column_surface_z": state.column_surface_z.tolist(),
        "transfer_flux": state.transfer_flux.tolist(),
        "youngman_triangles": extraction.triangles.tolist(),
    }
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    return image_path, state_path, state, extraction, table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("build/columnar_multifluid_demo"),
    )
    arguments = parser.parse_args()
    image_path, state_path, state, extraction, _table = run_demo(
        arguments.output_directory
    )
    print(
        f"accepted t={float(state.managed_time.item()):.3f}; "
        f"voxels={int(state.voxel_occupied.to_dtype('int32').sum().item())}; "
        f"triangles={extraction.triangle_count}"
    )
    print(image_path.resolve())
    print(state_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_demo_state",
    "extract_demo_surface",
    "render_demo",
    "run_demo",
    "run_managed_frame",
]
