from __future__ import annotations

import ast
import inspect
import numpy as np
import struct
import textwrap

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.accelerator_backends.mandelbrot_encoder_program import (
    build_mandelbrot_recording_process_graph,
    mandelbrot_frame_program,
    mandelbrot_recording_program,
)


def _tensor_inputs(frame_count: int = 2):
    width = height = 16
    unit_x = [
        (column - (width - 1) / 2) / height
        for _row in range(height)
        for column in range(width)
    ]
    unit_y = [
        (row - (height - 1) / 2) / height
        for row in range(height)
        for _column in range(width)
    ]
    return width, height, (
        AT.tensor(unit_x),
        AT.tensor(unit_y),
        AT.tensor([-0.75 + 0.01 * index for index in range(frame_count)]),
        AT.tensor([0.01 * index for index in range(frame_count)]),
        AT.tensor([3.0 - 0.1 * index for index in range(frame_count)]),
        AT.tensor([0.05 * index for index in range(frame_count)]),
        AT.tensor([-0.72 for _ in range(frame_count)]),
        AT.tensor([0.24 for _ in range(frame_count)]),
        AT.tensor([0.1 * index for index in range(frame_count)]),
        AT.tensor([0.52 + 0.02 * index for index in range(frame_count)]),
        AT.tensor([
            0.1 if sample & 1 else -0.1
            for sample in range(frame_count * 100)
        ]),
    )


def test_program_is_one_abstract_tensor_pipeline():
    source = textwrap.dedent(inspect.getsource(mandelbrot_recording_program))
    tree = ast.parse(source)
    function = tree.body[0]

    assert isinstance(function, ast.FunctionDef)
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(function))

    called_names = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert called_names == {
        "encode_jfif_resident",
        "mandelbrot_frame_program",
        "range",
        "tuple",
    }
    assert "mjpeg_frames" not in called_attributes
    assert {
        "MJPEGAVIWriter",
        "encode_jfif",
        "encode_ycbcr_jfif",
        "prepare_jpeg_encoding_resources",
        "append_frame",
        "append_audio_tensor",
    }.isdisjoint(called_names | called_attributes)
    assert "numpy" not in source
    assert "np." not in source

    frame_source = textwrap.dedent(
        inspect.getsource(mandelbrot_frame_program)
    )
    frame_tree = ast.parse(frame_source)
    frame_calls = {
        node.func.id
        for node in ast.walk(frame_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    frame_attributes = {
        node.func.attr
        for node in ast.walk(frame_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert frame_calls == {"range"}
    assert "stack" in frame_attributes
    assert "numpy" not in frame_source
    assert "np." not in frame_source


def test_program_emits_batched_jpeg_packets_without_a_file_boundary():
    with AT.use_backend("numpy"):
        width, height, inputs = _tensor_inputs()
        packets, frames, counts = mandelbrot_recording_program(
            *inputs[:-1],
            width=width,
            height=height,
            iterations=8,
        )

    assert len(packets) == 2
    encoded = []
    for octets, byte_count in packets:
        count = int(np.asarray(byte_count.tolist()).reshape(-1)[0])
        payload = bytes(
            np.asarray(octets.tolist(), dtype=np.uint8).reshape(-1)[:count]
        )
        encoded.append(payload)
    assert all(
        packet.startswith(b"\xff\xd8") and packet.endswith(b"\xff\xd9")
        for packet in encoded
    )
    assert frames.shape == (2, height, width, 3)
    assert counts.shape == (2, height, width)


def test_process_graph_has_only_the_single_pipeline_entry():
    graph = build_mandelbrot_recording_process_graph()
    entry = graph.function_table.entry("mandelbrot_recording_program")

    assert entry.graph is not None
    assert graph.G.graph["program_entrypoint"] == "mandelbrot_recording_program"
    assert graph.G.graph["source_scope"] == (
        "abstract_tensor_mandelbrot_audio_to_avi"
    )
