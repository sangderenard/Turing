"""Identity-preserving parametric scene meshes for AbstractUI.

The topology and editing vocabulary in this module are backend-neutral.  The
small numerical kernel is authored in Python and compiled through the common
fused-program pipeline to WebAssembly; browser code only realizes the result.
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


SCENE_MESH_SOURCE = """\
def instantiate_box_vertex(center_x, center_z, half_x, half_z, height,
                           unit_x, unit_y, unit_z):
    vertex_x = center_x + half_x * unit_x
    vertex_y = height * unit_y
    vertex_z = center_z + half_z * unit_z
    return vertex_x, vertex_y, vertex_z
"""

SCENE_MESH_INPUTS = (
    "center_x", "half_x", "unit_x", "height", "unit_y",
    "center_z", "half_z", "unit_z",
)
SCENE_MESH_OUTPUTS = ("vertex_x", "vertex_y", "vertex_z")
_SOURCE_PARAMETERS = (
    "center_x", "center_z", "half_x", "half_z", "height",
    "unit_x", "unit_y", "unit_z",
)

# A box's eight parametric corners and six ordered triangle faces.  Keeping
# these in the model makes vertex order a published contract, not browser lore.
BOX_CORNERS = (
    (-1, 0, -1), (1, 0, -1), (1, 1, -1), (-1, 1, -1),
    (-1, 0, 1), (1, 0, 1), (1, 1, 1), (-1, 1, 1),
)
BOX_FACES = (
    ((4, 5, 6, 4, 6, 7), (0, 0, 1)),
    ((1, 0, 3, 1, 3, 2), (0, 0, -1)),
    ((5, 1, 2, 5, 2, 6), (1, 0, 0)),
    ((0, 4, 7, 0, 7, 3), (-1, 0, 0)),
    ((7, 6, 2, 7, 2, 3), (0, 1, 0)),
    ((0, 1, 5, 0, 5, 4), (0, -1, 0)),
)


@dataclass(frozen=True, slots=True)
class ParametricFormInstruction:
    """One portable edit offered by an object's Form menu."""

    identity: str
    label: str
    parameter: str
    operation: str
    operand: float | None = None

    def to_data(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "identity": self.identity,
            "label": self.label,
            "parameter": self.parameter,
            "operation": self.operation,
        }
        if self.operand is not None:
            result["operand"] = self.operand
        return result


DEFAULT_FORM_INSTRUCTIONS = (
    ParametricFormInstruction("form:height-up", "height +", "height", "scale", 1.2),
    ParametricFormInstruction("form:height-down", "height −", "height", "scale", 1 / 1.2),
    ParametricFormInstruction("form:widen", "widen", "half_extent.x", "scale", 1.2),
    ParametricFormInstruction("form:narrow", "narrow", "half_extent.x", "scale", 1 / 1.2),
    ParametricFormInstruction("form:deepen", "deepen", "half_extent.z", "scale", 1.2),
    ParametricFormInstruction("form:shallow", "shallow", "half_extent.z", "scale", 1 / 1.2),
    ParametricFormInstruction("form:reset", "reset form", "*", "restore"),
)


@dataclass(frozen=True)
class SceneMeshWasm:
    binary: bytes
    entrypoint: str
    parameters: tuple[dict[str, str], ...]
    operation_count: int
    reserved_bytes: int

    def to_model(self) -> dict[str, Any]:
        return {
            "schema": "abstract-ui-scene-mesh-v0",
            "source_language": "python",
            "source": SCENE_MESH_SOURCE,
            "entrypoint": self.entrypoint,
            "parameters": [dict(parameter) for parameter in self.parameters],
            "inputs": list(SCENE_MESH_INPUTS),
            "outputs": list(SCENE_MESH_OUTPUTS),
            "value_type": "float64",
            "binary_base64": base64.b64encode(self.binary).decode("ascii"),
            "binary_bytes": len(self.binary),
            "operation_count": self.operation_count,
            "reserved_bytes": self.reserved_bytes,
            "lowering": [
                "python-ast", "captured-numerical-region", "fused-program",
                "webassembly",
            ],
            "topology": {
                "primitive": "triangles",
                "corners": [list(corner) for corner in BOX_CORNERS],
                "faces": [
                    {"corners": list(indices), "normal": list(normal)}
                    for indices, normal in BOX_FACES
                ],
                "vertices_per_instance": 36,
            },
            "identity_spans": {
                "source": "document_geometry.boxes[].identity",
                "ordering": "document-order",
                "unit": "vertex",
                "span_size": "composed-primitive-count-times-36",
                "primitive_vertices": 36,
            },
            "boundary_contract": {
                "source": "dom-border",
                "meaning": "wall",
                "identity": "document-object-identity",
                "height_parameter": "height",
                "thickness_parameter": "boundary.thickness",
                "opening_order": "document-order",
                "opening_kinds": ["door", "window", "portal"],
                "composition": "identity-preserving-boundary-union",
                "floor": "mandatory-slab",
                "interior": "hollow",
                "ceiling_rule": "height-at-absolute-maximum",
                "absolute_maximum_height": 4.0,
                "opening_operation": "boundary-minus-ordered-openings",
                "bevel_parameter": "radius",
                "bevel_realization": "unimplemented",
            },
            "revision": 0,
            "context_menu": {
                "selection": "viewer-camera.crosshair",
                "target": "identity-span",
                "items": [{
                    "identity": "context:form",
                    "label": "Form",
                    "kind": "submenu",
                    "instructions": [
                        instruction.to_data() for instruction in DEFAULT_FORM_INSTRUCTIONS
                    ],
                }],
            },
            "round_trip": {
                "mesh_sink": "viewer.geometry",
                "document_sink": "dom[data-node-id=identity]",
                "event": "apply-form",
            },
        }


@lru_cache(maxsize=1)
def compile_scene_mesh_wasm() -> SceneMeshWasm:
    """Compile the parametric box constructor and retain its public ABI."""

    samples = {name: np.ones(2, dtype=np.float64) for name in _SOURCE_PARAMETERS}
    captured_output = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(captured_output):
        warnings.simplefilter("ignore")
        compilation = compile_ast_aot(
            SCENE_MESH_SOURCE,
            "instantiate_box_vertex",
            samples,
            backend="c",
            precompile_only=True,
            mutable_parameters=_SOURCE_PARAMETERS,
        )
    program = project_public_numerical_program(compilation)
    module = emit_wasm_module(
        program,
        name="abstract_ui_scene_mesh",
        function_name="instantiate_scene_mesh",
        dtype="float64",
    )
    if not module.complete or module.binary is None:
        raise RuntimeError(module.shortfall_report())
    entry = module.api.entry_points[0]
    return SceneMeshWasm(
        binary=module.binary,
        entrypoint=entry.symbol,
        parameters=tuple({
            "name": parameter.name,
            "role": parameter.role,
            "dtype": parameter.dtype,
        } for parameter in entry.parameters),
        operation_count=len(program.steps),
        reserved_bytes=int(module.api.metadata.get("reserved_bytes", 0)),
    )


__all__ = [
    "BOX_CORNERS", "BOX_FACES", "DEFAULT_FORM_INSTRUCTIONS",
    "ParametricFormInstruction", "SCENE_MESH_INPUTS", "SCENE_MESH_OUTPUTS",
    "SCENE_MESH_SOURCE", "SceneMeshWasm", "compile_scene_mesh_wasm",
]
