"""Compile the repository MAC fluid and dt system into a native RGB window.

There is no alternate solver or authored Fortran here.  The Python source
below is ingested by ProcessGraph, calls ``VoxelMACFluid._substep`` through
its real class method, and advances it through ``run_superstep``.  The normal
registered Fortran target and common C shell own all later compilation,
arena feedback, standalone linkage, and Win32 presentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from ..cells.bath.voxel_fluid import VoxelFluidParams, VoxelMACFluid
from ..common.dt_system.dt_controller import STController, Targets, run_superstep
from .fortran_c_shell import (
    FortranCShellExecutable,
    compile_ast_fortran_c_shell,
)


NATIVE_VOXEL_SOURCE = """
def fluid_advance(state, dt):
    previous_mass = np.sum(state.S)
    voxel_substep(state, dt)
    metrics = state.compute_metrics(previous_mass)
    return True, metrics


def native_voxel_frame(
    state, targets, controller, frame_duration, dt_initial
):
    advanced, dt_next, metrics = run_superstep(
        state,
        frame_duration,
        dt_initial,
        state.dx,
        targets,
        controller,
        fluid_advance,
    )

    u_out = state.u + 0.0
    v_out = state.v + 0.0
    w_out = state.w + 0.0
    pressure_out = state.pr
    temperature_out = state.T + 0.0
    salinity_out = state.S
    dt_out = dt_next + 0.0

    pressure_magnitude = np.abs(state.pr)
    pressure_floor = state.p.rho0 * 9.81 * state.dx
    pressure_scale = np.maximum(np.max(pressure_magnitude), pressure_floor)
    darkness = 1.0 - 0.58 * np.minimum(
        pressure_magnitude / pressure_scale, 1.0
    )
    dye = np.minimum(np.maximum(state.S, 0.0), 1.0)
    red = ((10.0 + 55.0 * dye) * darkness).reshape((-1,))
    green = ((28.0 + 190.0 * dye) * darkness).reshape((-1,))
    blue = ((75.0 + 180.0 * dye) * darkness).reshape((-1,))
    return (
        u_out,
        v_out,
        w_out,
        pressure_out,
        temperature_out,
        salinity_out,
        dt_out,
        red,
        green,
        blue,
    )
"""


OUTPUT_NAMES = (
    "u_out",
    "v_out",
    "w_out",
    "pressure_out",
    "temperature_out",
    "salinity_out",
    "dt_out",
    "red",
    "green",
    "blue",
)


def _initial_state(width: int, height: int) -> VoxelMACFluid:
    params = VoxelFluidParams(
        nx=int(height),
        ny=int(width),
        nz=1,
        dx=0.025,
        # Display rows are the solver's first axis. Positive x is therefore
        # visually downward, so this aligns gravity with the native window.
        gravity=(9.81, 0.0, 0.0),
    )
    state = VoxelMACFluid(params)
    row = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
    column = (np.arange(width, dtype=np.float64) + 0.5) / float(width)
    yy, xx = np.meshgrid(row, column, indexing="ij")
    plume = np.exp(
        -((xx - 0.5) ** 2 / 0.022 + (yy - 0.82) ** 2 / 0.006)
    )
    state.S[:, :, 0] = np.minimum(0.82 * plume, 1.0)
    state.T[:, :, 0] += 8.0 * plume
    return state


def compile_native_voxel_fluid(
    output_directory: str | Path,
    *,
    width: int = 192,
    height: int = 128,
    frame_duration: float = 0.025,
) -> FortranCShellExecutable:
    """Compile the real Bath fluid frame through AST/AOT and the C shell."""

    if width < 4 or height < 4:
        raise ValueError("native voxel display dimensions must be at least 4")
    if frame_duration <= 0.0:
        raise ValueError("frame_duration must be positive")
    output = Path(output_directory).resolve()
    started = time.perf_counter()
    return compile_ast_fortran_c_shell(
        NATIVE_VOXEL_SOURCE,
        "native_voxel_frame",
        {
            "state": _initial_state(width, height),
            "targets": Targets(cfl=0.5, div_max=1.0e-2, mass_max=1.0e-3),
            "controller": STController(dt_min=1.0e-6, dt_max=0.025),
            "frame_duration": np.asarray(
                [float(frame_duration)], dtype=np.float64
            ),
            "dt_initial": np.asarray([0.01], dtype=np.float64),
        },
        output,
        python_bindings={
            "np": np,
            "voxel_substep": VoxelMACFluid._substep,
            "run_superstep": run_superstep,
        },
        output_names=OUTPUT_NAMES,
        state_feedback={
            "state.u": "u_out",
            "state.v": "v_out",
            "state.w": "w_out",
            "state.pr": "pressure_out",
            "state.T": "temperature_out",
            "state.S": "salinity_out",
            "dt_initial": "dt_out",
        },
        display={
            "width": int(width),
            "height": int(height),
            "title": "Turing AST / dt-system voxel fluid",
            "frame_delay_ms": 1,
        },
        name="native_voxel_fluid",
        standalone=True,
        checkpoint=output / "aot-checkpoint",
        mutable_parameters=("dt_initial",),
        progress=lambda message: print(
            f"{time.perf_counter() - started:9.3f}s {message}", flush=True
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("build/native-voxel-fluid")
    )
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--frame-duration", type=float, default=0.025)
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="zero runs continuously until the native window closes",
    )
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)
    artifact = compile_native_voxel_fluid(
        args.output,
        width=args.width,
        height=args.height,
        frame_duration=args.frame_duration,
    )
    print(artifact.executable_path, flush=True)
    if not args.compile_only:
        artifact.run(
            frames=args.frames,
            capture_output=bool(args.frames),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NATIVE_VOXEL_SOURCE",
    "compile_native_voxel_fluid",
    "main",
]
