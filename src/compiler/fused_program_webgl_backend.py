"""Emit a WebGL 2 fragment program from a numeric ``FusedProgram``.

WebGL is not desktop OpenGL compute.  It has no compute shaders or SSBOs, so
the existing GLSL backend's storage/dispatch wrapper cannot be relabelled and
sent to a browser.  What *is* shared is the finite scalar expression table.
This adapter reuses that table and gives it a WebGL-native raster contract:

* one input tensor per ``sampler2D``;
* one element per fragment, addressed by ``gl_FragCoord``;
* up to four scalar results in the red components of floating-point color
  targets (WebGL 2's guaranteed minimum draw-buffer count);
* a host-provided full-screen draw and same-sized input textures.

Integer textures, structural tensor operations, control flow, and programs
needing more than the guaranteed four draw buffers are named shortfalls rather
than silently pretending to be WebGL compute.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from ..common.tensors.accelerator_backends.glsl_backend import (
    GLSL_FLOAT_SCALAR_OPS,
    emit_scalar_expression,
)
from ..common.tensors.fused_ir import (
    FusedProgram,
    ordered_feed_ids,
    uniform_tensor_constant,
)
from .compiled_program_api import CompiledProgramAPI, EntryPoint


FULLSCREEN_VERTEX_SHADER = """#version 300 es
precision highp float;

const vec2 TURING_TRIANGLE[3] = vec2[3](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

void main() {
    gl_Position = vec4(TURING_TRIANGLE[gl_VertexID], 0.0, 1.0);
}
"""


@dataclass(frozen=True)
class WebGLShortfall:
    step_id: int
    operation: str
    reason: str

    def format(self) -> str:
        location = "program" if self.step_id < 0 else f"step {self.step_id}"
        return f"{location} ({self.operation}): {self.reason}"


@dataclass(frozen=True)
class WebGLFragmentModule:
    name: str
    source: str
    shortfalls: tuple[WebGLShortfall, ...]
    api: CompiledProgramAPI
    vertex_source: str = FULLSCREEN_VERTEX_SHADER

    @property
    def complete(self) -> bool:
        return not self.shortfalls


def supported_operations() -> frozenset[str]:
    """The float scalar vocabulary inherited from the desktop GLSL table."""

    return GLSL_FLOAT_SCALAR_OPS | frozenset({"tensor_from_list"})


def _literal(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("WebGL scalar literals must be finite")
    text = format(number, ".17g")
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text


def _api(
    name: str,
    feed_ids: tuple[int, ...],
    outputs: Mapping[str, int],
    *,
    output_layout: str,
    input_sampling: str,
) -> CompiledProgramAPI:
    return CompiledProgramAPI(
        module=name,
        language="webgl2-glsl-es",
        entry="main",
        entry_points=(
            EntryPoint(
                name="main",
                symbol="main",
                kind="numerical",
                note=(
                    "draw one full-screen triangle into a same-sized floating-"
                    "point framebuffer; bind each feed texture by value id"
                ),
            ),
        ),
        metadata={
            "execution_model": "fragment-raster",
            "glsl_version": "300 es",
            "input_storage": "sampler2D red channel",
            "input_sampling": input_sampling,
            "output_storage": (
                "one RGBA presentation color"
                if output_layout == "rgba"
                else "one vec4 draw buffer per output; scalar is red channel"
            ),
            "output_layout": output_layout,
            "required_draw_buffers": 1 if output_layout == "rgba" else len(outputs),
            "required_extension": (
                None if output_layout == "rgba" else "EXT_color_buffer_float"
            ),
            "feed_bindings": [
                {"value_id": value_id, "uniform": f"turing_feed_{value_id}"}
                for value_id in feed_ids
            ],
            "outputs": [
                {
                    "name": output_name,
                    "value_id": value_id,
                    "location": 0 if output_layout == "rgba" else index,
                    "channel": "rgba"[index] if output_layout == "rgba" else "r",
                }
                for index, (output_name, value_id) in enumerate(outputs.items())
            ],
            "vertex_shader": FULLSCREEN_VERTEX_SHADER,
        },
    )


def emit_webgl_fragment_module(
    program: FusedProgram,
    *,
    name: str = "program",
    output_layout: str = "draw-buffers",
    input_sampling: str = "texel",
) -> WebGLFragmentModule:
    """Lower one straight-line, elementwise program to GLSL ES 3.00."""

    if output_layout not in {"draw-buffers", "rgba"}:
        raise ValueError("output_layout must be 'draw-buffers' or 'rgba'")
    if input_sampling not in {"texel", "normalized"}:
        raise ValueError("input_sampling must be 'texel' or 'normalized'")

    feed_ids = tuple(ordered_feed_ids(program))
    shortfalls: list[WebGLShortfall] = []
    names = {value_id: f"v_{value_id}" for value_id in feed_ids}
    if input_sampling == "normalized":
        body = [
            "    vec2 turing_uv = gl_FragCoord.xy / max(turing_resolution, vec2(1.0));",
            "    turing_uv.y = 1.0 - turing_uv.y;",
            *(
                f"    float {names[value_id]} = texture("
                f"turing_feed_{value_id}, turing_uv).r;"
                for value_id in feed_ids
            ),
        ]
    else:
        body = [
            "    ivec2 turing_coordinate = ivec2(gl_FragCoord.xy);",
            *(
                f"    float {names[value_id]} = texelFetch("
                f"turing_feed_{value_id}, turing_coordinate, 0).r;"
                for value_id in feed_ids
            ),
        ]

    for step in program.steps:
        result = f"v_{step.result_id}"
        names[step.result_id] = result
        if step.op_name == "tensor_from_list":
            value = uniform_tensor_constant(
                step.attrs.get("values", step.attrs.get("data"))
            )
            if value is None:
                shortfalls.append(WebGLShortfall(
                    step.step_id,
                    step.op_name,
                    "non-uniform constants need a texture binding",
                ))
                body.append(f"    float {result} = 0.0; // unsupported constant")
                continue
            try:
                literal = _literal(value)
            except ValueError as error:
                shortfalls.append(WebGLShortfall(
                    step.step_id, step.op_name, str(error),
                ))
                literal = "0.0"
            body.append(f"    float {result} = {literal};")
            continue

        if step.op_name not in GLSL_FLOAT_SCALAR_OPS:
            shortfalls.append(WebGLShortfall(
                step.step_id,
                step.op_name,
                "not in the WebGL float-scalar expression surface",
            ))
            body.append(f"    float {result} = 0.0; // unsupported operation")
            continue

        unknown_attrs = set(step.attrs) - {"right_scalar", "left_scalar", "reverse"}
        if unknown_attrs:
            shortfalls.append(WebGLShortfall(
                step.step_id,
                step.op_name,
                "unsupported attributes: " + ", ".join(sorted(unknown_attrs)),
            ))

        operands = [names.get(value_id) for value_id in step.input_ids]
        if any(value is None for value in operands):
            shortfalls.append(WebGLShortfall(
                step.step_id,
                step.op_name,
                "an operand has no fragment-local producer",
            ))
            body.append(f"    float {result} = 0.0; // missing operand")
            continue

        left = operands[0] if operands else None
        right = operands[1] if len(operands) > 1 else None
        if right is None and "right_scalar" in step.attrs:
            try:
                right = _literal(step.attrs["right_scalar"])
            except ValueError as error:
                shortfalls.append(WebGLShortfall(
                    step.step_id, step.op_name, str(error),
                ))
        elif right is None and "left_scalar" in step.attrs:
            try:
                right = left
                left = _literal(step.attrs["left_scalar"])
            except ValueError as error:
                shortfalls.append(WebGLShortfall(
                    step.step_id, step.op_name, str(error),
                ))

        try:
            if left is None:
                raise ValueError("operation has no left operand")
            expression = emit_scalar_expression(
                step.op_name,
                left,
                right,
                reverse=bool(step.attrs.get("reverse", False)),
                out_type="float",
            )
        except (KeyError, ValueError) as error:
            shortfalls.append(WebGLShortfall(
                step.step_id, step.op_name, str(error),
            ))
            expression = "0.0"
        body.append(f"    float {result} = {expression};")

    if not program.outputs or len(program.outputs) > 4:
        shortfalls.append(WebGLShortfall(
            -1,
            "outputs",
            "WebGL 2 guarantees between one and four fragment draw buffers",
        ))
    output_expressions: list[str] = []
    for output_id in program.outputs.values():
        output = names.get(output_id, "0.0")
        if output_id not in names:
            shortfalls.append(WebGLShortfall(
                -1, "output", f"value {output_id} has no producer",
            ))
        output_expressions.append(output)

    if output_layout == "rgba":
        packed = list(output_expressions[:4])
        while len(packed) < 3:
            packed.append("0.0")
        if len(packed) < 4:
            packed.append("1.0")
        output_declarations = ["layout(location = 0) out vec4 turing_output_0;"]
        output_assignments = [
            "    turing_output_0 = vec4(" + ", ".join(packed) + ");"
        ]
    else:
        output_declarations = [
            f"layout(location = {index}) out vec4 turing_output_{index};"
            for index in range(len(program.outputs))
        ]
        output_assignments = [
            f"    turing_output_{index} = vec4({output}, 0.0, 0.0, 1.0);"
            for index, output in enumerate(output_expressions)
        ]

    source = "\n".join((
        "#version 300 es",
        "precision highp float;",
        "precision highp int;",
        "",
        *(("uniform vec2 turing_resolution;",) if input_sampling == "normalized" else ()),
        *(f"uniform sampler2D turing_feed_{value_id};" for value_id in feed_ids),
        *output_declarations,
        "",
        "void main() {",
        *body,
        *output_assignments,
        "}",
        "",
    ))
    return WebGLFragmentModule(
        name=name,
        source=source,
        shortfalls=tuple(shortfalls),
        api=_api(
            name,
            feed_ids,
            program.outputs,
            output_layout=output_layout,
            input_sampling=input_sampling,
        ),
    )


__all__ = [
    "FULLSCREEN_VERTEX_SHADER",
    "WebGLFragmentModule",
    "WebGLShortfall",
    "emit_webgl_fragment_module",
    "supported_operations",
]
