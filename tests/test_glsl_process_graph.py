from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.demo_mandelbrot_fusion import (
    mandelbrot_jpeg_planes,
    normalized_plane,
    parametric_mandelbrot_escape,
)
from src.common.tensors.accelerator_backends.glsl_backend import (
    GLChunk,
    GLContextUnavailable,
    require_gl_context,
)
from src.common.tensors.accelerator_backends.mandelbrot_encoder_program import (
    build_mandelbrot_encoder_process_graph,
)
from src.common.tensors.fused_ir import Meta
from src.common.tensors.numpy_backend import NumPyTensorOperations as NT
from src.compiler.glsl_process_graph import compile_process_graph_glsl


TENSOR_INPUTS = (
    "unit_x",
    "unit_y",
    "center_x",
    "center_y",
    "span",
    "family_mix",
    "julia_x",
    "julia_y",
    "palette_phase",
    "color_drive",
)
SCALAR_INPUTS = TENSOR_INPUTS[2:]
OUTPUTS = (
    "counts",
    "luminance",
    "blue_difference",
    "red_difference",
)


def _compile(iterations: int):
    graph = build_mandelbrot_encoder_process_graph(
        entrypoint="mandelbrot_display_master"
    )
    return compile_process_graph_glsl(
        graph,
        specializations={"iterations": iterations},
        input_meta={
            name: Meta(dtype="float32", device="glsl")
            for name in TENSOR_INPUTS
        },
        scalar_tensor_inputs=SCALAR_INPUTS,
        output_names=OUTPUTS,
    )


def test_process_graph_loop_becomes_one_structured_glsl_loop():
    short = _compile(3)
    long = _compile(3000)

    assert short.loop_count == long.loop_count == 1
    assert short.primitive_count == long.primitive_count == 92
    assert short.source.count("for (int loop0_i") == 1
    assert long.source.count("for (int loop0_i") == 1
    assert "loop0_i < 3" in short.source
    assert "loop0_i < 3000" in long.source
    assert short.input_names == (
        "unit_x",
        "unit_y",
        "__scalar_controls__",
    )
    assert short.scalar_input_order == SCALAR_INPUTS


def test_structured_shader_matches_original_abstracttensor_program():
    try:
        require_gl_context()
    except GLContextUnavailable as error:
        pytest.skip(f"no OpenGL 4.3+ compute context: {error}")

    iterations = 8
    width, height = 32, 16
    compiled = _compile(iterations)
    unit_x, unit_y = normalized_plane(width, height)
    scalar_values = {
        "center_x": -0.72,
        "center_y": 0.1,
        "span": 2.4,
        "family_mix": 0.0,
        "julia_x": -0.72,
        "julia_y": 0.24,
        "palette_phase": 0.0,
        "color_drive": 0.52,
    }
    feeds = {
        "unit_x": GLChunk.from_numpy(unit_x).to_gpu(),
        "unit_y": GLChunk.from_numpy(unit_y).to_gpu(),
        compiled.scalar_buffer_name: GLChunk.from_numpy(
            np.asarray(
                [
                    scalar_values[name]
                    for name in compiled.scalar_input_order
                ],
                dtype=np.float32,
            )
        ).to_gpu(),
    }
    outputs = compiled.execute(feeds)
    try:
        tensor_args = [
            NT.tensor(np.asarray([scalar_values[name]], np.float32))
            for name in SCALAR_INPUTS[:6]
        ]
        counts = parametric_mandelbrot_escape(
            NT.tensor(unit_x),
            NT.tensor(unit_y),
            *tensor_args,
            iterations,
        )
        planes = mandelbrot_jpeg_planes(
            counts,
            iterations,
            NT.tensor(
                np.asarray([scalar_values["palette_phase"]], np.float32)
            ),
            NT.tensor(
                np.asarray([scalar_values["color_drive"]], np.float32)
            ),
        )
        expected = {
            "counts": np.asarray(counts.tolist(), dtype=np.float32),
            **{
                name: np.asarray(value.tolist(), dtype=np.float32)
                for name, value in zip(OUTPUTS[1:], planes)
            },
        }
        for name in OUTPUTS:
            actual = outputs[name].numpy()
            tolerance = 0.0 if name == "counts" else 2e-5
            assert np.allclose(actual, expected[name], atol=tolerance, rtol=0)
    finally:
        for chunk in (*feeds.values(), *outputs.values()):
            chunk.release()
