"""Compile Python tensor programs into the common native Fortran display shell.

This is deliberately a thin compiler entrypoint: Python enters the AST/AOT
compiler, the registered Fortran target consumes its projected numeric IR, and
the existing C shell owns display, arenas, feedback, and profiling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from ..common.tensors.accelerator_backends.aot_compile import (
    compile_ast_aot,
    project_public_numerical_program,
)
from .fortran_c_shell import FortranCShellExecutable, compile_fortran_module_c_shell
from .machine_targets import get_target
from .shell_io import (
    ShellIOBinding,
    ShellIOCapability,
    ShellIOManifest,
    ShellIORequest,
    attach_shell_io,
)
from .site_bundle import SourceContract, discover_source_contract
from .ssa_fortran_backend import FortranEmissionError, FortranModule


IMAGE_PROBE_SOURCE = """
def fortran_image_probe(column_x, column_y):
    red = column_x * 255.0
    green = column_y * 255.0
    blue = (1.0 - column_x) * (1.0 - column_y) * 255.0
    return red, green, blue
"""


@dataclass(frozen=True)
class NativeFortranDisplay:
    executable: FortranCShellExecutable
    module: FortranModule

    @property
    def executable_path(self) -> Path:
        return self.executable.executable_path

    def run(self, *, frames: int = 0, capture_output: bool = False):
        return self.executable.run(frames=frames, capture_output=capture_output)

    __call__ = run


def _array(value: Any, count: int) -> np.ndarray:
    if isinstance(value, Mapping):
        if set(value) == {"literal"}:
            value = value["literal"]
        elif set(value) == {"values"}:
            value = value["values"]
        else:
            raise ValueError("feed mappings require exactly 'literal' or 'values'")
    array = np.asarray(0.0 if value is None else value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(count, array.item(), dtype=np.float64)
    if array.size != count:
        raise ValueError(f"feed has {array.size} elements; display requires {count}")
    return np.ascontiguousarray(array.reshape(count))


def compile_fortran_display(
    source: str,
    entrypoint: str,
    feeds: Mapping[str, Any],
    output_directory: str | Path,
    *,
    width: int,
    height: int,
    title: str,
    state_feedback: Mapping[str, str] | None = None,
    constant_map: Mapping[str, Any] | None = None,
    mutable_parameters: Sequence[str] = (),
    remove_loops: bool = True,
    unroll_limit: int = 4096,
    bake_mode: str = "one_shot",
    schedule_preference: str = "asap",
    python_bindings: Mapping[str, Any] | None = None,
    output_names: Sequence[str] | None = None,
    render_fps: float = 60.0,
) -> NativeFortranDisplay:
    """Compile one RGB-producing Python entrypoint through the shared stack."""

    if not float(render_fps) > 0.0:
        raise ValueError("render_fps must be positive")

    aot = compile_ast_aot(
        source,
        entrypoint,
        dict(feeds),
        backend="c",
        remove_loops=remove_loops,
        unroll_limit=unroll_limit,
        precompile_only=True,
        bake_mode=bake_mode,
        schedule_preference=schedule_preference,
        constant_map=dict(constant_map or {}),
        mutable_parameters=tuple(mutable_parameters),
        python_bindings=dict(python_bindings or {}),
    )
    program = project_public_numerical_program(aot)
    if output_names is not None:
        names = tuple(map(str, output_names))
        if len(names) != len(program.outputs):
            raise ValueError(
                f"received {len(names)} output names for "
                f"{len(program.outputs)} captured outputs"
            )
        program = replace(
            program,
            outputs={
                name: value_id
                for name, value_id in zip(names, program.outputs.values())
            },
        )
    emitted = get_target("fortran").emit(program, name=entrypoint)
    if not emitted.complete or emitted.module is None:
        raise FortranEmissionError(
            "Fortran target could not emit display program: "
            + "; ".join(emitted.shortfalls)
        )
    entry = emitted.module.api.entry_point(entrypoint)
    outputs = {
        str(parameter.source_name or parameter.name)
        for parameter in entry.parameters
        if parameter.role == "output"
    }
    missing = {"red", "green", "blue"} - outputs
    if missing:
        raise ValueError("display program lacks RGB outputs: " + ", ".join(sorted(missing)))
    manifest = ShellIOManifest(
        requests=(ShellIORequest.create(
            ShellIOCapability.DISPLAY,
            attributes={
                "pixel_format": "rgb_f64_planar",
                "width": int(width),
                "height": int(height),
                "title": str(title),
                "frame_delay_ms": max(0, int(round(1000.0 / float(render_fps)))),
            },
        ),),
        bindings=tuple(
            ShellIOBinding(f"display.{channel}", entrypoint, channel)
            for channel in ("red", "green", "blue")
        ),
    )
    module = replace(
        emitted.module,
        api=attach_shell_io(emitted.module.api, manifest),
    )
    executable = compile_fortran_module_c_shell(
        module,
        dict(feeds),
        output_directory,
        entrypoint=entrypoint,
        state_feedback=dict(state_feedback or {}),
        name=entrypoint,
    )
    return NativeFortranDisplay(executable, module)


def compile_image_probe(
    output_directory: str | Path,
    *,
    width: int = 320,
    height: int = 180,
) -> NativeFortranDisplay:
    x = np.tile(np.linspace(0.0, 1.0, width), height)
    y = np.repeat(np.linspace(0.0, 1.0, height), width)
    return compile_fortran_display(
        IMAGE_PROBE_SOURCE,
        "fortran_image_probe",
        {"column_x": x, "column_y": y},
        output_directory,
        width=width,
        height=height,
        title="Turing Fortran RGB shell probe",
    )


def _contract_feeds(contract: SourceContract) -> dict[str, np.ndarray]:
    count = int(contract.width) * int(contract.height)
    return {
        name: _array(value, count)
        for name, value in contract.feeds.items()
    }


def compile_columnar_multifluid_display(
    output_directory: str | Path,
) -> NativeFortranDisplay:
    from ..common.dt_system.fluid_mechanics.columnar_multifluid_web_demo import (
        FORTRAN_SOURCE,
    )


def compile_voxel_mac_display(
    output_directory: str | Path,
    *,
    width: int = 96,
    height: int = 64,
    pressure_iterations: int = 24,
) -> NativeFortranDisplay:
    """Compile the fluid-only Bath MAC specialization through normal AOT/SSA."""

    from ..common.dt_system.fluid_mechanics.voxel_mac_aot import (
        build_voxel_mac_aot_source,
        initial_voxel_mac_arenas,
    )
    source = build_voxel_mac_aot_source(width, height, pressure_iterations)
    feeds = initial_voxel_mac_arenas(width, height)
    return compile_fortran_display(
        source,
        "voxel_mac_rgb_step",
        feeds,
        output_directory,
        width=width,
        height=height,
        title="Turing managed-time MAC fluid",
        state_feedback={
            "u": "next_u",
            "v": "next_v",
            "pressure": "next_pressure",
            "dye": "next_dye",
            "managed_time": "next_time",
        },
        constant_map={},
        mutable_parameters=tuple(feeds),
        remove_loops=True,
        unroll_limit=4096,
        bake_mode="whole_program",
        schedule_preference="asap",
        output_names=(
            "next_u", "next_v", "next_pressure", "next_dye", "next_time",
            "red", "green", "blue",
        ),
        render_fps=60.0,
    )

    contract = discover_source_contract(FORTRAN_SOURCE)
    return compile_fortran_display(
        FORTRAN_SOURCE,
        contract.entrypoint,
        _contract_feeds(contract),
        output_directory,
        width=contract.width,
        height=contract.height,
        title=contract.title,
        state_feedback=contract.state_feedback,
        # The published browser page deliberately bakes its terrain arrays.
        # This native diagnostic instead keeps every plane in a caller-owned
        # arena so it exercises the Fortran span/index ABI without embedding
        # a second, static-data compilation policy.
        constant_map={},
        mutable_parameters=contract.mutable_parameters,
        remove_loops=contract.remove_loops,
        unroll_limit=contract.unroll_limit,
        bake_mode=contract.bake_mode,
        schedule_preference=contract.schedule_preference,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "columnar", "voxel"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--width", type=int, default=320, help="probe width")
    parser.add_argument("--height", type=int, default=180, help="probe height")
    parser.add_argument("--pressure-iterations", type=int, default=24)
    arguments = parser.parse_args(argv)
    output = arguments.output or Path(f"build/native-fortran-{arguments.mode}")
    if arguments.mode == "probe":
        artifact = compile_image_probe(output, width=arguments.width, height=arguments.height)
    elif arguments.mode == "columnar":
        artifact = compile_columnar_multifluid_display(output)
    else:
        width = arguments.width if arguments.width != 320 else 96
        height = arguments.height if arguments.height != 180 else 64
        artifact = compile_voxel_mac_display(
            output,
            width=width,
            height=height,
            pressure_iterations=arguments.pressure_iterations,
        )
    print(artifact.executable_path)
    if not arguments.compile_only:
        artifact.run(frames=arguments.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IMAGE_PROBE_SOURCE",
    "NativeFortranDisplay",
    "compile_columnar_multifluid_display",
    "compile_fortran_display",
    "compile_image_probe",
    "compile_voxel_mac_display",
    "main",
]
