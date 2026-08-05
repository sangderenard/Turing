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
    voxel_substep(state, dt)
    metrics = state.compute_metrics(0.0)
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
        max_iters=64,
    )

    u_out = state.u + 0.0
    v_out = state.v + 0.0
    w_out = state.w + 0.0
    pressure_out = state.pr
    temperature_out = state.T + 0.0
    salinity_out = state.S
    dt_out = dt_next + 0.0

    pressure_magnitude = np.abs(state.pr)
    pressure_scale = np.maximum(np.max(pressure_magnitude), 1.0e-12)
    darkness = 1.0 - 0.72 * pressure_magnitude / pressure_scale
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
        gravity=(0.0, -9.81, 0.0),
        nu=0.0,
        thermal_diffusivity=0.0,
        solute_diffusivity=0.0,
        pressure_tol=2.0e-3,
        pressure_maxiter=1,
        visc_maxiter=0,
    )
    state = VoxelMACFluid(params)
    row = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
    column = (np.arange(width, dtype=np.float64) + 0.5) / float(width)
    yy, xx = np.meshgrid(row, column, indexing="ij")
    plume = np.exp(
        -((xx - 0.5) ** 2 / 0.022 + (yy - 0.82) ** 2 / 0.006)
    )
    ripple = 0.16 * (1.0 + np.sin(10.0 * np.pi * xx))
    state.S[:, :, 0] = np.minimum(plume * (0.82 + ripple), 1.0)
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
            "targets": Targets(cfl=1.5, div_max=5.0, mass_max=5.0),
            "controller": STController(dt_min=1.0e-6, dt_max=0.08),
            "frame_duration": np.asarray(
                [float(frame_duration)], dtype=np.float64
            ),
            "dt_initial": np.asarray([0.05], dtype=np.float64),
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
