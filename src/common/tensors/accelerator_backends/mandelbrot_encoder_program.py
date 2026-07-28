"""Whole-source Mandelbrot-to-JFIF ProcessGraph entry point.

The numerical function composes the original Mandelbrot, palette, and JPEG
functions.  ``build_mandelbrot_encoder_process_graph`` imports this file, the
original demo file, and every compression source file as one semantic program.
It does not execute or record an AbstractTensor tape.
"""

from __future__ import annotations

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


def mandelbrot_jpeg_master(
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
    resources,
):
    """Compose the original solve and complete 4:4:4 JFIF encoder."""

    # Avoid eagerly importing the demo/UI module from this source-compiler
    # entrypoint. The AST resolves these original definitions from the supplied
    # source bundle; ordinary execution imports them when the master is called.
    from .demo_mandelbrot_fusion import (
        mandelbrot_jpeg_planes,
        parametric_mandelbrot_escape,
    )
    from ..compression.jpeg.frame import encode_ycbcr_jfif

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
    encoded = encode_ycbcr_jfif(
        planes,
        mcu_rows_per_batch=(height + 7) // 8,
        resources=resources,
    )
    return counts, encoded


def mandelbrot_encoder_source_files() -> tuple[Path, ...]:
    """Return the original source bundle constituting the master program."""

    tensor_root = Path(__file__).resolve().parents[1]
    compression = tensor_root / "compression"
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("demo_mandelbrot_fusion.py"),
    ]
    sources.extend(sorted(compression.rglob("*.py")))
    return tuple(sources)


def build_mandelbrot_encoder_process_graph(
    *,
    profile: str = "tensor_control",
    entrypoint: str = "mandelbrot_jpeg_master",
):
    """Ingest the source bundle without executing the program.

    ``tensor_control`` is the numerical compiler projection, ``program`` keeps
    every node in the transitive entrypoint program, and ``complete`` retains
    the entire source bundle for syntax-coverage and archaeology audits.
    """

    from ....transmogrifier.graph.graph_express2 import ProcessGraph

    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        mandelbrot_encoder_source_files(),
        entrypoint=entrypoint,
        profile=profile,
    )
    return graph


def build_mandelbrot_recording_process_graph(*, profile: str = "program"):
    """Ingest the actual recording loop as one start-to-finish ProcessGraph.

    This is not a second rendering loop. It selects the existing ``animate_glsl``
    implementation as the root; semantic AST ingestion follows its statically
    declared compiled entrypoint into the original solve and follows the live
    encoder/writer calls through JPEG bytes, audio, AVI indexes, final header
    patching, and close.
    """

    graph = build_mandelbrot_encoder_process_graph(
        profile=profile,
        entrypoint="animate_glsl",
    )
    graph.G.graph.update(
        program_name="mandelbrot_recording",
        program_entrypoint="animate_glsl",
    )
    return graph


__all__ = [
    "build_mandelbrot_encoder_process_graph",
    "build_mandelbrot_recording_process_graph",
    "mandelbrot_display_master",
    "mandelbrot_encoder_source_files",
    "mandelbrot_jpeg_master",
]
