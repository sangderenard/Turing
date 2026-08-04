"""Compiler-facing two-dimensional specialization of the Bath MAC solver.

The object-oriented :class:`src.cells.bath.voxel_fluid.VoxelMACFluid` remains
the scientific Python reference.  This module expresses the same arena layout
and projection cycle as one ordinary AbstractTensor-compatible callable so the
repository compiler can lower it to SSA and then Fortran.  Arrays are never
materialized as source literals; the native shell owns and feeds every arena.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..dt_controller import STController, Targets
from ..time_runtime import ManagedTimeRuntime, TimeWindowRequest
from .voxel_fluid_engine import VoxelFluidEngine
from ....cells.bath.voxel_fluid import VoxelFluidParams, VoxelMACFluid


def build_voxel_mac_aot_source(nx: int, ny: int, pressure_iterations: int = 24) -> str:
    """Return the statically shaped MAC step consumed by the normal compiler."""

    if nx < 4 or ny < 4:
        raise ValueError("native MAC display requires at least a 4 by 4 grid")
    if pressure_iterations < 1:
        raise ValueError("pressure_iterations must be positive")
    jacobi = "\n".join(
        f"""    pressure_left = next_pressure[cell_left]
    pressure_right = next_pressure[cell_right]
    pressure_down = next_pressure[cell_down]
    pressure_up = next_pressure[cell_up]
    next_pressure = 0.25 * (pressure_left + pressure_right + pressure_down + pressure_up - pressure_rhs * dx * dx)
    next_pressure = next_pressure - next_pressure.sum() / {float(nx * ny)!r}"""
        for _ in range(int(pressure_iterations))
    )
    return f'''\
def voxel_mac_rgb_step(u, v, pressure, dye, managed_time, dt, dx, rho, buoyancy, viscosity, diffusion, cell_left, cell_right, cell_down, cell_up, divergence_u_left, divergence_u_right, divergence_v_down, divergence_v_up, pressure_u_left, pressure_u_right, pressure_v_down, pressure_v_up, dye_v_down, dye_v_up, u_boundary_mask, v_boundary_mask):
    # Salinity-like buoyancy, matching VoxelMACFluid._add_body_forces.
    dye_at_v = 0.5 * (dye[dye_v_down] + dye[dye_v_up])
    forced_v = (v + dt * buoyancy * dye_at_v) * v_boundary_mask

    # A small implicit-viscosity surrogate keeps the compiler step bounded;
    # pressure projection below is the actual incompressibility constraint.
    velocity_decay = 1.0 / (1.0 + viscosity * dt / (dx * dx))
    projected_u = u * velocity_decay * u_boundary_mask
    projected_v = forced_v * velocity_decay
    divergence = (
        projected_u[divergence_u_right] - projected_u[divergence_u_left]
        + projected_v[divergence_v_up] - projected_v[divergence_v_down]
    ) / dx
    pressure_rhs = (rho / dt) * (
        divergence - divergence.sum() / {float(nx * ny)!r}
    )
    next_pressure = pressure
{jacobi}

    grad_x = (next_pressure[pressure_u_right] - next_pressure[pressure_u_left]) / dx
    grad_y = (next_pressure[pressure_v_up] - next_pressure[pressure_v_down]) / dx
    next_u = (projected_u - (dt / rho) * grad_x) * u_boundary_mask
    next_v = (projected_v - (dt / rho) * grad_y) * v_boundary_mask

    # Cell-centred scalar transport driven by the projected MAC faces.
    cell_u = 0.5 * (next_u[divergence_u_right] + next_u[divergence_u_left])
    cell_v = 0.5 * (next_v[divergence_v_up] + next_v[divergence_v_down])
    dye_left = dye[cell_left]
    dye_right = dye[cell_right]
    dye_down = dye[cell_down]
    dye_up = dye[cell_up]
    dye_laplacian = (dye_left + dye_right + dye_down + dye_up - 4.0 * dye) / (dx * dx)
    next_dye = (
        dye
        - dt * (
            cell_u * (dye_right - dye_left) / (2.0 * dx)
            + cell_v * (dye_up - dye_down) / (2.0 * dx)
        )
        + diffusion * dt * dye_laplacian
    ).maximum(0.0).minimum(1.0)

    pressure_magnitude = next_pressure.abs()
    pressure_normalized = (
        (pressure_magnitude - pressure_magnitude.min())
        / (pressure_magnitude.max() - pressure_magnitude.min() + 1.0e-9)
    ).maximum(0.0).minimum(1.0)
    darkness = 1.0 - 0.72 * pressure_normalized
    red_grid = ((8.0 + 48.0 * next_dye) * darkness).reshape(({nx}, {ny}))
    green_grid = ((25.0 + 195.0 * next_dye) * darkness).reshape(({nx}, {ny}))
    blue_grid = ((62.0 + 188.0 * next_dye) * darkness).reshape(({nx}, {ny}))

    # VoxelMACFluid uses (x,y,z); the display ABI consumes row-major (y,x).
    red = red_grid.permute((1, 0)).reshape((-1,))
    green = green_grid.permute((1, 0)).reshape((-1,))
    blue = blue_grid.permute((1, 0)).reshape((-1,))
    next_time = managed_time + dt
    return next_u, next_v, next_pressure, next_dye, next_time, red, green, blue
'''


def initial_voxel_mac_arenas(nx: int, ny: int, *, dx: float = 0.025) -> dict[str, Any]:
    """Create a fluid-only plume state in caller-owned NumPy arenas."""

    x = (np.arange(nx, dtype=np.float64) + 0.5) / float(nx)
    y = (np.arange(ny, dtype=np.float64) + 0.5) / float(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    dye = np.exp(-((xx - 0.5) ** 2 / 0.018 + (yy - 0.16) ** 2 / 0.008))
    cells = np.arange(nx * ny, dtype=np.int64).reshape(nx, ny)
    u_faces = np.arange((nx + 1) * ny, dtype=np.int64).reshape(nx + 1, ny)
    v_faces = np.arange(nx * (ny + 1), dtype=np.int64).reshape(nx, ny + 1)

    cell_left = np.concatenate((cells[:1], cells[:-1]), axis=0)
    cell_right = np.concatenate((cells[1:], cells[-1:]), axis=0)
    cell_down = np.concatenate((cells[:, :1], cells[:, :-1]), axis=1)
    cell_up = np.concatenate((cells[:, 1:], cells[:, -1:]), axis=1)

    pressure_u_left = np.empty_like(u_faces)
    pressure_u_right = np.empty_like(u_faces)
    pressure_u_left[0] = pressure_u_right[0] = cells[0]
    pressure_u_left[-1] = pressure_u_right[-1] = cells[-1]
    pressure_u_left[1:-1] = cells[:-1]
    pressure_u_right[1:-1] = cells[1:]
    pressure_v_down = np.empty_like(v_faces)
    pressure_v_up = np.empty_like(v_faces)
    pressure_v_down[:, 0] = pressure_v_up[:, 0] = cells[:, 0]
    pressure_v_down[:, -1] = pressure_v_up[:, -1] = cells[:, -1]
    pressure_v_down[:, 1:-1] = cells[:, :-1]
    pressure_v_up[:, 1:-1] = cells[:, 1:]

    dye_v_down = pressure_v_down.copy()
    dye_v_up = pressure_v_up.copy()
    u_mask = np.ones_like(u_faces, dtype=np.float64)
    v_mask = np.ones_like(v_faces, dtype=np.float64)
    u_mask[[0, -1], :] = 0.0
    v_mask[:, [0, -1]] = 0.0
    return {
        "u": np.zeros((nx + 1) * ny, dtype=np.float64),
        "v": np.zeros(nx * (ny + 1), dtype=np.float64),
        "pressure": np.zeros(nx * ny, dtype=np.float64),
        "dye": np.ascontiguousarray(dye.reshape(-1)),
        "managed_time": np.asarray([0.0], dtype=np.float64),
        "dt": np.asarray([0.004], dtype=np.float64),
        "dx": np.asarray([float(dx)], dtype=np.float64),
        "rho": np.asarray([1000.0], dtype=np.float64),
        "buoyancy": np.asarray([5.5], dtype=np.float64),
        "viscosity": np.asarray([2.0e-4], dtype=np.float64),
        "diffusion": np.asarray([2.5e-4], dtype=np.float64),
        "cell_left": np.ascontiguousarray(cell_left.reshape(-1)),
        "cell_right": np.ascontiguousarray(cell_right.reshape(-1)),
        "cell_down": np.ascontiguousarray(cell_down.reshape(-1)),
        "cell_up": np.ascontiguousarray(cell_up.reshape(-1)),
        "divergence_u_left": np.ascontiguousarray(u_faces[:-1].reshape(-1)),
        "divergence_u_right": np.ascontiguousarray(u_faces[1:].reshape(-1)),
        "divergence_v_down": np.ascontiguousarray(v_faces[:, :-1].reshape(-1)),
        "divergence_v_up": np.ascontiguousarray(v_faces[:, 1:].reshape(-1)),
        "pressure_u_left": np.ascontiguousarray(pressure_u_left.reshape(-1)),
        "pressure_u_right": np.ascontiguousarray(pressure_u_right.reshape(-1)),
        "pressure_v_down": np.ascontiguousarray(pressure_v_down.reshape(-1)),
        "pressure_v_up": np.ascontiguousarray(pressure_v_up.reshape(-1)),
        "dye_v_down": np.ascontiguousarray(dye_v_down.reshape(-1)),
        "dye_v_up": np.ascontiguousarray(dye_v_up.reshape(-1)),
        "u_boundary_mask": np.ascontiguousarray(u_mask.reshape(-1)),
        "v_boundary_mask": np.ascontiguousarray(v_mask.reshape(-1)),
    }


@dataclass
class ManagedVoxelReference:
    """The repository MAC solver admitted through the managed-time boundary."""

    simulation: VoxelMACFluid
    engine: VoxelFluidEngine
    runtime: ManagedTimeRuntime
    request_id: int = 0

    def advance(self, duration: float, *, dt_initial: float = 0.004):
        self.request_id += 1
        start = self.runtime.current_time
        return self.runtime.advance(TimeWindowRequest(
            request_id=self.request_id,
            generation=self.runtime.generation,
            t_start=start,
            t_end=start + float(duration),
            dt_initial=float(dt_initial),
        ))


def make_managed_voxel_reference(nx: int, ny: int, *, dx: float = 0.025) -> ManagedVoxelReference:
    """Construct the actual NumPy solver under :class:`ManagedTimeRuntime`."""

    params = VoxelFluidParams(
        nx=int(nx), ny=int(ny), nz=1, dx=float(dx),
        gravity=(0.0, -9.81, 0.0), nu=2.0e-4,
    )
    simulation = VoxelMACFluid(params)
    seed = initial_voxel_mac_arenas(nx, ny, dx=dx)["dye"].reshape(nx, ny, 1)
    simulation.S[...] = seed
    engine = VoxelFluidEngine(simulation)

    def advance(state: VoxelMACFluid, dt: float):
        ok, metrics, _ = engine.step(float(dt))
        return ok, metrics

    runtime = ManagedTimeRuntime(
        simulation,
        advance,
        dx=float(dx),
        targets=Targets(cfl=0.5, div_max=2.0e-4, mass_max=1.0),
        controller=STController(dt_min=1.0e-6, dt_max=0.004),
    )
    return ManagedVoxelReference(simulation, engine, runtime)


__all__ = [
    "ManagedVoxelReference",
    "build_voxel_mac_aot_source",
    "initial_voxel_mac_arenas",
    "make_managed_voxel_reference",
]
