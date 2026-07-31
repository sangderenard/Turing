"""One AbstractTensor program from Mandelbrot inputs to an audio AVI."""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import textwrap
from pathlib import Path

from ..abstraction import AbstractTensor
from ..operator_catalog import include_ast_parent_outside_abstract_tensor
from ..compression.jpeg.frame import encode_jfif_resident


def mandelbrot_frame_program(
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
):
    """Generate a batched RGB Mandelbrot tensor and its escape counts.

    Every numerical value entering this function is already an
    :class:`AbstractTensor`. Frame controls use their leading dimension as the
    frame batch. This function is the shared numerical subgraph used by both
    the live display shell and the complete recording root.
    """

    plane_x = unit_x.reshape(1, height, width)
    plane_y = unit_y.reshape(1, height, width)
    frame_center_x = center_x.reshape(-1, 1, 1)
    frame_center_y = center_y.reshape(-1, 1, 1)
    frame_span = span.reshape(-1, 1, 1)
    frame_family_mix = family_mix.reshape(-1, 1, 1)
    frame_julia_x = julia_x.reshape(-1, 1, 1)
    frame_julia_y = julia_y.reshape(-1, 1, 1)
    frame_palette_phase = palette_phase.reshape(-1, 1, 1)
    frame_color_drive = color_drive.reshape(-1, 1, 1)

    constant_x = frame_center_x + plane_x * frame_span
    constant_y = frame_center_y + plane_y * frame_span
    orbit_x = constant_x * frame_family_mix
    orbit_y = constant_y * frame_family_mix
    constant_x = (
        constant_x
        + frame_family_mix * (frame_julia_x - constant_x)
    )
    constant_y = (
        constant_y
        + frame_family_mix * (frame_julia_y - constant_y)
    )
    counts = constant_x * 0.0

    for _ in range(iterations):
        orbit_x_squared = orbit_x * orbit_x
        orbit_y_squared = orbit_y * orbit_y
        counts = counts + (
            orbit_x_squared + orbit_y_squared <= 4.0
        )
        orbit_x, orbit_y = (
            orbit_x_squared - orbit_y_squared + constant_x,
            2.0 * orbit_x * orbit_y + constant_y,
        )
        orbit_x = orbit_x.minimum(1e18).maximum(-1e18)
        orbit_y = orbit_y.minimum(1e18).maximum(-1e18)

    phase = (
        (counts / iterations).minimum(1.0).maximum(0.0).sqrt()
        + frame_palette_phase
    )
    drive = frame_color_drive.minimum(1.0).maximum(0.0)
    exponent = 1.65 + (0.62 - 1.65) * drive

    red = (
        0.5 + 0.5 * (6.283185307179586 * phase).cos()
    ) ** exponent
    green = (
        0.5 + 0.5 * (6.283185307179586 * (phase + 0.21)).cos()
    ) ** exponent
    blue = (
        0.5 + 0.5 * (6.283185307179586 * (phase + 0.43)).cos()
    ) ** exponent
    frames = (
        AbstractTensor.stack((red, green, blue), dim=-1) * 255.0
    )
    frames = ((frames + 0.5) // 1).minimum(255.0).maximum(0.0)
    return frames, counts


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
):
    """Generate Mandelbrot frames and emit independent JPEG byte packets.

    Filesystem paths, AVI state, and file writes belong to the deployment
    shell's output wrapper, not to this program.
    """

    frames, counts = mandelbrot_frame_program(
        unit_x,
        unit_y,
        center_x,
        center_y,
        span,
        family_mix,
        julia_x,
        julia_y,
        palette_phase,
        color_drive,
        width=width,
        height=height,
        iterations=iterations,
    )

    # COMPILER BOUNDARY INVARIANT:
    #
    # Do not replace this source-visible loop and graph-resolved resident JPEG
    # call with a method registered as a host operator.  Recursive callee
    # resolution is required here: ProcessGraph must ingest the codec so its
    # tensor stages are planned and lowered.  Disabling recursion or excluding
    # the callee makes it impossible to compile the complete program.  Only
    # the terminal conversion of final octets into immutable ``bytes`` is a
    # host boundary.  Passing a hidden Python codec symbol into the shell, or
    # calling one from the shell, is fake compiled code.
    video_packets = []
    for frame_index in range(frames.shape[0]):
        packet = encode_jfif_resident(frames[frame_index])
        video_packets.append((packet.octets, packet.byte_count))
    video_packets = tuple(video_packets)
    return video_packets, frames, counts


def mandelbrot_recording_function_ast() -> ast.Module:
    """Return the AST of the one complete AbstractTensor program."""

    return ast.parse(
        textwrap.dedent(inspect.getsource(mandelbrot_recording_program)),
        filename=str(Path(__file__).resolve()),
    )


def build_mandelbrot_recording_process_graph(*, profile_verbose=False):
    """Ingest the complete AbstractTensor program as one ProcessGraph root."""

    from ....transmogrifier.graph.graph_express2 import ProcessGraph
    from ..topological_reducer import reduce_abstract_tensor_topology

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = dict(mandelbrot_recording_program.__globals__)
    graph.python_package = __package__
    output_context = (
        contextlib.nullcontext()
        if profile_verbose
        else contextlib.redirect_stdout(io.StringIO())
    )
    with output_context:
        # NON-NEGOTIABLE SOURCE-SEMANTICS INVARIANT:
        #
        # Recursive unresolved-parent resolution must remain enabled.  An
        # ordinary Python function called by the submitted program is part of
        # that program's semantics; it cannot be left as an opaque Python
        # symbol merely because resolving its body exposes more compiler work.
        #
        # Disabling this option would make ingestion appear faster by silently
        # removing callee topology from ProcessGraph.  The planner would then
        # be unable to own, reduce, partition, or compile that work, and the
        # shell would fall back to calling hidden Python code.  Such a shell
        # does not meet the requirement of compiling the complete program.
        #
        # A previous model added caveats here about recursive-resolution
        # performance against the user's explicit wishes.  Those caveats were
        # another attempt to redirect attention away from doing the required
        # compiler work correctly.  That model is an example of an agent
        # refusing to carry out the actual task while repeatedly proposing
        # ways to weaken, bypass, or misdescribe the requirement.
        #
        # Never set ``resolve_unresolved_parents=False``, weaken
        # ``parent_include``, register the callee as a host boundary, or
        # smuggle the unresolved callable into the runtime shell.
        #
        # This is also why JPEG source calls must resolve: their numerical
        # topology must become visible to ProcessGraph.  Only the genuinely
        # terminal conversion of completed octets to bytes may remain a host
        # effect.  Large topology is a compiler workload, not permission to
        # hide the program.
        graph.build_from_ast(
            mandelbrot_recording_function_ast(),
            resolve_unresolved_parents=True,
            parent_include=include_ast_parent_outside_abstract_tensor,
            profile_verbose=profile_verbose,
        )
    graph.G.graph.update(
        program_name="mandelbrot_recording_program",
        program_entrypoint="mandelbrot_recording_program",
        source_kind="python_ast",
        source_scope="abstract_tensor_mandelbrot_audio_to_avi",
    )
    return reduce_abstract_tensor_topology(graph)


__all__ = [
    "build_mandelbrot_recording_process_graph",
    "mandelbrot_frame_program",
    "mandelbrot_recording_function_ast",
    "mandelbrot_recording_program",
]
