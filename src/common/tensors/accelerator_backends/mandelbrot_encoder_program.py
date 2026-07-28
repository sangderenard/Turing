"""One explicit AbstractTensor Mandelbrot-to-JFIF ProcessGraph entry point."""

from __future__ import annotations

import ast
import contextlib
import io
import inspect
import textwrap
from pathlib import Path

from ..abstraction import AbstractTensor


def mandelbrot_display_master(
    unit_x: AbstractTensor,
    unit_y: AbstractTensor,
    center_x: AbstractTensor,
    center_y: AbstractTensor,
    span: AbstractTensor,
    family_mix: AbstractTensor,
    julia_x: AbstractTensor,
    julia_y: AbstractTensor,
    palette_phase: AbstractTensor,
    color_drive: AbstractTensor,
    *,
    iterations: int,
):
    """Return the resident solve and display/encoder color planes."""

    from .demo_mandelbrot_fusion import (
        mandelbrot_jpeg_planes,
        parametric_mandelbrot_escape,
    )

    counts = parametric_mandelbrot_escape(
        unit_x,
        unit_y,
        center_x,
        center_y,
        span,
        family_mix,
        julia_x,
        julia_y,
        iterations,
        clamp=1e18,
    )
    luminance, blue_difference, red_difference = mandelbrot_jpeg_planes(
        counts,
        iterations,
        palette_phase,
        color_drive,
    )
    return counts, luminance, blue_difference, red_difference


def mandelbrot_recording_program(
    unit_x: AbstractTensor,
    unit_y: AbstractTensor,
    center_x: AbstractTensor,
    center_y: AbstractTensor,
    span: AbstractTensor,
    family_mix: AbstractTensor,
    julia_x: AbstractTensor,
    julia_y: AbstractTensor,
    palette_phase: AbstractTensor,
    color_drive: AbstractTensor,
    *,
    width: int,
    height: int,
    iterations: int,
    avi_writer,
    audio_samples=None,
    resources=None,
):
    """Solve, encode one JPEG frame, then interleave it into an AVI stream."""

    from .demo_mandelbrot_fusion import (
        mandelbrot_jpeg_planes,
        parametric_mandelbrot_escape,
    )
    from ..autograd import autograd
    from ..compression.block_transform import (
        block_view_2d,
        dct_2d_blocks,
    )
    from ..compression.coefficient_events import (
        collect_component_block_coefficient_events,
        slice_block_coefficient_events,
    )
    from ..compression.jpeg.frame import (
        _EntropyTensorAccumulator,
        _color_header,
        prepare_jpeg_encoding_resources,
    )
    from ..compression.jpeg.scan import (
        encode_baseline_color_component_scan,
    )

    counts = parametric_mandelbrot_escape(
        unit_x,
        unit_y,
        center_x,
        center_y,
        span,
        family_mix,
        julia_x,
        julia_y,
        iterations,
    )
    luminance, blue_difference, red_difference = mandelbrot_jpeg_planes(
        counts,
        iterations,
        palette_phase,
        color_drive,
    )
    planes = (
        luminance.reshape(height, width),
        blue_difference.reshape(height, width),
        red_difference.reshape(height, width),
    )
    resources = resources or prepare_jpeg_encoding_resources(planes[0])

    with autograd.no_grad():
        # Keep every numerical compression stage in this submitted function:
        # component batching/blocking -> DCT -> quantization -> rounding ->
        # zigzag ordering.  Each operation remains ordinary AbstractTensor
        # composition and therefore stays visible to ProcessGraph lowering.
        component_samples = AbstractTensor.stack(planes, dim=0)
        blocks = block_view_2d(
            component_samples - 128.0,
            block_height=8,
            block_width=8,
        )
        transformed = dct_2d_blocks(
            blocks,
            basis=resources.dct_basis,
        )
        scaled = transformed / resources.ycbcr_quantization
        quantized = scaled.sign() * ((scaled.abs() + 0.500001) // 1)
        flattened = quantized.reshape(
            *(quantized.shape[:-2] + (64,))
        )
        coefficients = flattened[..., resources.zigzag]

        # Coefficient events are the explicit boundary between the numerical
        # tensor transform and the canonical JPEG Huffman tables.
        events = collect_component_block_coefficient_events(
            coefficients,
            max_magnitude_bits=11,
            previous_dc=(0, 0, 0),
        )
        block_count = coefficients[0].reshape(-1, 64).shape[0]
        y_events = slice_block_coefficient_events(
            events,
            0,
            block_count,
        )
        chroma_events = slice_block_coefficient_events(
            events,
            block_count,
            block_count * 3,
        )
        huffman_scan = encode_baseline_color_component_scan(
            y_events,
            chroma_events,
            luma_dc_table=resources.luma_dc_table,
            luma_ac_table=resources.luma_ac_table,
            chroma_dc_table=resources.chroma_dc_table,
            chroma_ac_table=resources.chroma_ac_table,
        )
        entropy = _EntropyTensorAccumulator()
        entropy_bytes = entropy.append(huffman_scan, final=True)
        trailing_bytes = entropy.finish()

    jpeg_frame = b"".join(
        (
            _color_header(height, width),
            entropy_bytes,
            trailing_bytes,
            b"\xFF\xD9",
        )
    )

    # AVI/OpenDML framing and A/V ordering remain explicit in the same
    # ProcessGraph entrypoint: one video packet, followed by its PCM packet.
    avi_writer.append_frame(jpeg_frame)
    if audio_samples is not None:
        avi_writer.append_audio_tensor(audio_samples)

    return counts, planes, coefficients, jpeg_frame


# Compatibility name for callers that used the former entrypoint spelling.
mandelbrot_jpeg_master = mandelbrot_recording_program


def mandelbrot_recording_function_ast() -> ast.Module:
    """Load only the saved, explicit AbstractTensor program into Python AST."""

    source = textwrap.dedent(inspect.getsource(mandelbrot_recording_program))
    return ast.parse(
        source,
        filename=str(Path(__file__).resolve()),
    )


def mandelbrot_display_function_ast() -> ast.Module:
    """Collect the complete solve/display/encode entrypoint program AST."""

    from .demo_mandelbrot_fusion import (
        mandelbrot_jpeg_planes,
        parametric_mandelbrot_escape,
    )

    source = "\n\n".join(
        textwrap.dedent(inspect.getsource(function))
        for function in (
            parametric_mandelbrot_escape,
            mandelbrot_jpeg_planes,
            mandelbrot_display_master,
            mandelbrot_recording_program,
        )
    )
    return ast.parse(source, filename=str(Path(__file__).resolve()))


def build_mandelbrot_encoder_process_graph(
    *,
    profile: str = "tensor_control",
    entrypoint: str = "mandelbrot_recording_program",
):
    """Build the complete structural solve/JPEG/AVI ProcessGraph."""

    from ....transmogrifier.graph.graph_express2 import ProcessGraph

    graph = ProcessGraph(materialize_memory=False)
    # The legacy structural importer still prints its node walk
    # unconditionally.  That diagnostic is not part of this demo's compiler
    # contract, so keep it out of the live render/profile stream.
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(mandelbrot_display_function_ast())
    graph.G.graph.update(
        program_name="mandelbrot_display",
        program_entrypoint=entrypoint,
        semantic_profile=profile,
    )
    return graph


def build_mandelbrot_recording_process_graph():
    """Submit the complete saved recording function to ``build_from_ast``."""

    from ....transmogrifier.graph.graph_express2 import ProcessGraph
    from ..topological_reducer import reduce_abstract_tensor_topology

    program_ast = mandelbrot_recording_function_ast()
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(program_ast)
    graph.G.graph.update(
        program_name="mandelbrot_recording_program",
        program_entrypoint="mandelbrot_recording_program",
        source_kind="python_ast",
        source_scope="solve_through_avi",
    )
    reduced_graph = reduce_abstract_tensor_topology(graph)
    return reduced_graph


__all__ = [
    "build_mandelbrot_encoder_process_graph",
    "build_mandelbrot_recording_process_graph",
    "mandelbrot_display_function_ast",
    "mandelbrot_display_master",
    "mandelbrot_jpeg_master",
    "mandelbrot_recording_function_ast",
    "mandelbrot_recording_program",
]
