"""Python-authored perspective projection compiled for AbstractUI Canvas2D.

The browser owns triangle ordering and painting.  This module owns the numeric
camera transform: authored Python is captured as a numerical region, lowered
through the common fused IR, and emitted as a WebAssembly array kernel.
"""

from __future__ import annotations

import base64
import contextlib
from dataclasses import dataclass
from functools import lru_cache
import io
import warnings
from typing import Any

import numpy as np

from ..common.tensors.accelerator_backends.aot_compile import (
    compile_ast_aot,
    project_public_numerical_program,
)
from .fused_program_wasm_backend import emit_wasm_module


SOFTWARE_MESH_PROJECT_SOURCE = """\
def project_vertex(vertex_x, vertex_y, vertex_z,
                   camera_x, camera_y, camera_z,
                   forward_x, forward_y, forward_z,
                   right_x, right_y, right_z,
                   up_x, up_y, up_z, width, height):
    relative_x = vertex_x - camera_x
    relative_y = vertex_y - camera_y
    relative_z = vertex_z - camera_z
    view_x = relative_x * right_x + relative_y * right_y + relative_z * right_z
    view_y = relative_x * up_x + relative_y * up_y + relative_z * up_z
    view_z = relative_x * forward_x + relative_y * forward_y + relative_z * forward_z
    screen_x = width * 0.5 + view_x * height * 0.7142857142857143 / view_z
    screen_y = height * 0.5 - view_y * height * 0.7142857142857143 / view_z
    return screen_x, screen_y, view_z
"""

SOFTWARE_MESH_INPUTS = (
    "vertex_x", "camera_x", "vertex_y", "camera_y", "vertex_z", "camera_z",
    "right_x", "right_y", "right_z", "up_x", "up_y", "up_z",
    "forward_x", "forward_y", "forward_z", "width", "height",
)
SOFTWARE_MESH_OUTPUTS = ("screen_x", "screen_y", "view_z")
_SOURCE_PARAMETERS = (
    "vertex_x", "vertex_y", "vertex_z",
    "camera_x", "camera_y", "camera_z",
    "forward_x", "forward_y", "forward_z",
    "right_x", "right_y", "right_z",
    "up_x", "up_y", "up_z", "width", "height",
)


@dataclass(frozen=True)
class SoftwareMeshWasm:
    """One browser-callable projection artifact and its retained provenance."""

    binary: bytes
    entrypoint: str
    parameters: tuple[dict[str, str], ...]
    operation_count: int
    reserved_bytes: int

    def to_model(self) -> dict[str, Any]:
        return {
            "schema": "abstract-ui-software-mesh-wasm-v0",
            "source_language": "python",
            "source": SOFTWARE_MESH_PROJECT_SOURCE,
            "entrypoint": self.entrypoint,
            "parameters": [dict(parameter) for parameter in self.parameters],
            "inputs": list(SOFTWARE_MESH_INPUTS),
            "outputs": list(SOFTWARE_MESH_OUTPUTS),
            "value_type": "float64",
            "binary_base64": base64.b64encode(self.binary).decode("ascii"),
            "binary_bytes": len(self.binary),
            "operation_count": self.operation_count,
            "reserved_bytes": self.reserved_bytes,
            "lowering": [
                "python-ast", "captured-numerical-region", "fused-program",
                "webassembly",
            ],
        }


@lru_cache(maxsize=1)
def compile_software_mesh_wasm() -> SoftwareMeshWasm:
    """Compile the retained Python projection source to a complete WASM ABI."""

    samples = {name: np.ones(2, dtype=np.float64) for name in _SOURCE_PARAMETERS}
    # Keep the discovery trace away from the projection singularity.
    samples.update(vertex_z=np.full(2, 3.0), camera_z=np.zeros(2))
    captured_output = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(captured_output):
        warnings.simplefilter("ignore")
        compilation = compile_ast_aot(
            SOFTWARE_MESH_PROJECT_SOURCE,
            "project_vertex",
            samples,
            backend="c",
            precompile_only=True,
            mutable_parameters=_SOURCE_PARAMETERS,
        )
    program = project_public_numerical_program(compilation)
    module = emit_wasm_module(
        program,
        name="abstract_ui_software_mesh",
        function_name="project_mesh",
        dtype="float64",
    )
    if not module.complete or module.binary is None:
        raise RuntimeError(module.shortfall_report())
    entry = module.api.entry_points[0]
    parameters = tuple(
        {"name": parameter.name, "role": parameter.role, "dtype": parameter.dtype}
        for parameter in entry.parameters
    )
    return SoftwareMeshWasm(
        binary=module.binary,
        entrypoint=entry.symbol,
        parameters=parameters,
        operation_count=len(program.steps),
        reserved_bytes=int(module.api.metadata.get("reserved_bytes", 0)),
    )


__all__ = [
    "SOFTWARE_MESH_INPUTS",
    "SOFTWARE_MESH_OUTPUTS",
    "SOFTWARE_MESH_PROJECT_SOURCE",
    "SoftwareMeshWasm",
    "compile_software_mesh_wasm",
]
