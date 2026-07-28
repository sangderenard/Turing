"""Mandelbrot/JPEG demo awaiting a real ProcessGraph-to-GLSL compiler.

The former structural-AST reinterpretation shortcut is deliberately removed.
The mathematical, compression, and rendering helpers remain here, but GLSL
compilation must not resume until it consumes the ProcessGraph's scheduled
operation and control nodes directly.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np


# ---------------------------------------------------------------------------
# the program
# ---------------------------------------------------------------------------

# Escaped orbits diverge without bound. In float32 they reach inf within a few
# more iterations, then inf-inf produces NaN, and the two backends round that
# boundary differently -- the first version of this demo agreed with numpy on
# only 99.9577% of pixels, max |diff| = 18, purely from overflow timing.
#
# Pinning |zx|,|zy| to ORBIT_CLAMP fixes it exactly. Points still inside the set
# satisfy |z| <= 2 by definition, so clamping at 1e18 cannot affect them; escaped
# points freeze at a magnitude whose square (1e36) is still finite in float32 and
# still far above the escape radius, so their count stays frozen either way.
# minimum/maximum are in both backends' vocabularies already.
ORBIT_CLAMP = 1e18


def mandelbrot_escape(cx, cy, iterations: int, clamp: float = ORBIT_CLAMP):
    """Ordinary backend-agnostic AbstractTensor Mandelbrot computation."""

    zx = cx * 0.0
    zy = cx * 0.0
    count = cx * 0.0
    for _ in range(iterations):
        zx2, zy2 = zx * zx, zy * zy
        count = count + (zx2 + zy2 <= 4.0)
        zx, zy = zx2 - zy2 + cx, 2.0 * zx * zy + cy
        zx = zx.minimum(clamp).maximum(-clamp)
        zy = zy.minimum(clamp).maximum(-clamp)
    return count


def parametric_mandelbrot_escape(
    unit_x,
    unit_y,
    center_x,
    center_y,
    span,
    family_mix,
    julia_x,
    julia_y,
    iterations: int,
    clamp: float = ORBIT_CLAMP,
):
    """Continuous Mandelbrot-to-Julia quadratic-family solve.

    Complex values remain paired real/imaginary AbstractTensors. This is
    algebraically the complex recurrence while retaining the scalar canonical
    operator vocabulary understood by every lowering target.
    """
    cx = center_x + unit_x * span
    cy = center_y + unit_y * span
    zx = cx * family_mix
    zy = cy * family_mix
    constant_x = cx + family_mix * (julia_x - cx)
    constant_y = cy + family_mix * (julia_y - cy)
    count = cx * 0.0
    for _ in range(iterations):
        zx2, zy2 = zx * zx, zy * zy
        count = count + (zx2 + zy2 <= 4.0)
        zx, zy = zx2 - zy2 + constant_x, 2.0 * zx * zy + constant_y
        zx = zx.minimum(clamp).maximum(-clamp)
        zy = zy.minimum(clamp).maximum(-clamp)
    return count


def capture_mandelbrot(cx: np.ndarray, cy: np.ndarray, iterations: int):
    """Reject the retired execution-tape compiler shortcut."""

    raise RuntimeError(
        "Mandelbrot GradTape capture is disabled; compile the complete "
        "program through AST -> ProcessGraph instead"
    )


def capture_parametric_mandelbrot(iterations: int):
    """Reject the retired execution-tape compiler shortcut."""

    raise RuntimeError(
        "Parametric Mandelbrot GradTape capture is disabled; compile the "
        "complete program through AST -> ProcessGraph instead"
    )


def mandelbrot_jpeg_planes(
    counts,
    iterations: int,
    palette_phase,
    color_drive,
):
    """Compose the display palette and JPEG 4:4:4 planes elementwise.

    This is ordinary AbstractTensor math. A ProcessGraph backend optimizer may
    keep it beside the shared Mandelbrot producer and expose count/Y/Cb/Cr as
    four outputs of one dispatch.
    """

    phase = (
        (counts / max(iterations, 1)).minimum(1.0).maximum(0.0).sqrt()
        + palette_phase
    )
    drive = color_drive.minimum(1.0).maximum(0.0)
    exponent = 1.65 + (0.62 - 1.65) * drive

    def channel(offset: float):
        wave = (
            0.5
            + 0.5
            * (6.283185307179586 * (phase + offset)).cos()
        ) ** exponent
        return ((wave * 255.0 + 0.5) // 1).minimum(255.0).maximum(0.0)

    red = channel(0.0)
    green = channel(0.21)
    blue = channel(0.43)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    blue_difference = (
        -0.168736 * red - 0.331264 * green + 0.5 * blue + 128.0
    )
    red_difference = (
        0.5 * red - 0.418688 * green - 0.081312 * blue + 128.0
    )
    return (
        luminance.minimum(255.0).maximum(0.0),
        blue_difference.minimum(255.0).maximum(0.0),
        red_difference.minimum(255.0).maximum(0.0),
    )


def capture_parametric_mandelbrot_encoder(iterations: int):
    """Reject the partial tape capture previously presented as compilation."""

    raise RuntimeError(
        "Partial Mandelbrot/JPEG GradTape capture is disabled; the complete "
        "encoder must enter through AST -> ProcessGraph before lowering"
    )


def compile_parametric_mandelbrot_glsl(iterations: int):
    """Refuse the deleted AST-reinterpretation compiler shortcut."""

    raise RuntimeError(
        "ProcessGraph-to-GLSL compilation is unavailable: the structural AST "
        "reinterpretation shortcut was removed because it did not compile "
        "the ProcessGraph's scheduled operation and control nodes."
    )


def run_abstract_numpy(cx: np.ndarray, cy: np.ndarray, iterations: int):
    """Run the exact same AbstractTensor function on the NumPy backend."""

    from ..numpy_backend import NumPyTensorOperations

    result = mandelbrot_escape(
        NumPyTensorOperations.tensor(cx),
        NumPyTensorOperations.tensor(cy),
        iterations,
    )
    return np.asarray(result.tolist(), dtype=cx.dtype)


def complex_plane(width: int, height: int, center: complex, span: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Row-major cx/cy grids, flattened, float32."""
    aspect = width / height
    xs = np.linspace(center.real - span * aspect / 2,
                     center.real + span * aspect / 2, width, dtype=np.float32)
    ys = np.linspace(center.imag - span / 2,
                     center.imag + span / 2, height, dtype=np.float32)
    cx, cy = np.meshgrid(xs, ys)
    return (np.ascontiguousarray(cx.ravel()),
            np.ascontiguousarray(cy.ravel()))


def normalized_plane(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Dimensionless pixel coordinates consumed by the parametric solve."""
    aspect = width / height
    xs = np.linspace(-0.5 * aspect, 0.5 * aspect, width, dtype=np.float32)
    ys = np.linspace(-0.5, 0.5, height, dtype=np.float32)
    unit_x, unit_y = np.meshgrid(xs, ys)
    return (
        np.ascontiguousarray(unit_x.ravel()),
        np.ascontiguousarray(unit_y.ravel()),
    )


def animated_camera(
    center: complex, span: float, phase: float
) -> tuple[complex, float]:
    """A visibly varied but continuous tour around the requested view.

    The exponential span modulation covers roughly an order of magnitude,
    while incommensurate lateral frequencies avoid a short repeating orbit.
    """
    phase = float(phase)
    log_zoom = (
        1.25 * np.sin(0.71 * phase)
        + 0.45 * np.sin(1.93 * phase)
    )
    animated_span = float(span) * float(np.exp(log_zoom))
    dx = float(span) * (
        0.58 * np.sin(0.83 * phase)
        + 0.22 * np.sin(2.17 * phase)
    )
    dy = float(span) * (
        0.48 * (np.sin(0.97 * phase + 0.61) - np.sin(0.61))
        + 0.19 * np.sin(1.67 * phase)
    )
    return center + complex(dx, dy), animated_span


def dream_parameters(
    center: complex,
    span: float,
    travel: float,
    *,
    bass: float,
    low_mid: float,
    high_mid: float,
    reaction: float,
    zoom_rate: float,
) -> tuple[complex, float, float, complex]:
    """Map restrained spectral controls into a detailed complex-family view."""
    reaction = max(0.0, float(reaction))
    # Keep this particular tour in the Mandelbrot-detailed portion of the
    # continuous family. Larger family excursions need a different camera
    # chart; applying them to a deep Mandelbrot view erases its structure.
    family_mix = 0.04 + 0.18 * (
        0.5 + 0.5 * np.sin(
        0.24 * travel + reaction * 0.38 * (low_mid - 0.5)
        )
    )

    # c = mu/2 - mu^2/4 parameterizes the Mandelbrot main cardioid. Keeping
    # |mu| < 1 produces connected Julia sets instead of mostly empty dust.
    mu_radius = np.clip(
        0.58 + reaction * 0.08 * (low_mid - 0.5), 0.46, 0.72
    )
    mu_angle = 0.31 * travel + reaction * 0.42 * (high_mid - 0.5)
    mu = mu_radius * np.exp(1j * mu_angle)
    julia_constant = 0.5 * mu - 0.25 * mu * mu

    mandelbrot_center, mandelbrot_span = animated_camera(
        center, span, travel
    )
    # Preserve the detailed target c-plane exactly under the family transform:
    # (1-mix)*pixel + mix*julia == mandelbrot_pixel.
    family_scale = max(1.0 - family_mix, 1e-6)
    animated_center = (
        mandelbrot_center - family_mix * julia_constant
    ) / family_scale
    animated_span = float(np.exp(
        np.log(max(mandelbrot_span / family_scale, 1e-15))
        - zoom_rate * travel
        + reaction * 0.08 * (0.5 - bass)
    ))
    return animated_center, animated_span, float(family_mix), julia_constant


def detail_state_features(
    center: complex,
    span: float,
    travels: np.ndarray,
    *,
    bass: np.ndarray,
    low_mid: np.ndarray,
    high_mid: np.ndarray,
    reaction: float,
    zoom_rate: float,
) -> tuple[np.ndarray, list[tuple[complex, float, float, complex]]]:
    """Describe candidate camera states for the learned detail controller."""
    from .mandelbrot_detail_network import dream_features

    travels = np.asarray(travels, dtype=np.float64)
    bass = np.broadcast_to(np.asarray(bass, dtype=np.float64), travels.shape)
    low_mid = np.broadcast_to(
        np.asarray(low_mid, dtype=np.float64), travels.shape
    )
    high_mid = np.broadcast_to(
        np.asarray(high_mid, dtype=np.float64), travels.shape
    )
    states = [
        dream_parameters(
            center,
            span,
            float(travel),
            bass=float(b),
            low_mid=float(lm),
            high_mid=float(hm),
            reaction=reaction,
            zoom_rate=zoom_rate,
        )
        for travel, b, lm, hm in zip(travels, bass, low_mid, high_mid)
    ]
    return (
        dream_features(
            travels,
            bass,
            low_mid,
            high_mid,
            np.asarray([state[1] for state in states]),
            np.asarray([state[2] for state in states]),
        ),
        states,
    )


def build_detail_controller(
    center: complex,
    span: float,
    *,
    iterations: int,
    samples: int,
    epochs: int,
    reaction: float,
    zoom_rate: float,
):
    """Train AbstractNN on batched low-resolution AbstractTensor solves."""
    from ..autograd import autograd
    from ..numpy_backend import NumPyTensorOperations as NT
    from .mandelbrot_detail_network import (
        detail_scores,
        train_detail_controller,
    )

    travels = np.linspace(0.0, 36.0, samples, endpoint=False)
    bass = 0.5 + 0.5 * np.sin(0.83 * travels + 0.2)
    low_mid = 0.5 + 0.5 * np.sin(0.57 * travels + 0.8)
    high_mid = 0.5 + 0.5 * np.sin(0.73 * travels + 1.7)
    features, states = detail_state_features(
        center,
        span,
        travels,
        bass=bass,
        low_mid=low_mid,
        high_mid=high_mid,
        reaction=reaction,
        zoom_rate=zoom_rate,
    )

    train_width, train_height = 48, 30
    unit_x, unit_y = normalized_plane(train_width, train_height)
    unit_x = np.broadcast_to(unit_x, (samples, unit_x.size)).copy()
    unit_y = np.broadcast_to(unit_y, (samples, unit_y.size)).copy()
    as_column = lambda values: np.asarray(values, dtype=np.float32)[:, None]
    with autograd.no_grad():
        field = parametric_mandelbrot_escape(
            NT.tensor(unit_x),
            NT.tensor(unit_y),
            NT.tensor(as_column([state[0].real for state in states])),
            NT.tensor(as_column([state[0].imag for state in states])),
            NT.tensor(as_column([state[1] for state in states])),
            NT.tensor(as_column([state[2] for state in states])),
            NT.tensor(as_column([state[3].real for state in states])),
            NT.tensor(as_column([state[3].imag for state in states])),
            min(iterations, 40),
        )
    fields = np.asarray(field.tolist(), dtype=np.float32).reshape(
        samples, train_height, train_width
    )
    scores = detail_scores(fields, min(iterations, 40))
    return train_detail_controller(features, scores, epochs=epochs), scores


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

def _replacement_feeds(captured, cx, cy):
    """Bind replacement arrays by matching the two captured root identities."""

    feed_ids = list(captured.feeds)
    if len(feed_ids) != 2:
        raise ValueError(f"expected cx/cy capture roots, found {len(feed_ids)}")
    return {feed_ids[0]: cx, feed_ids[1]: cy}


def run_glsl(captured, cx, cy):
    from .glsl_backend import execute_program

    return execute_program(
        captured.program, _replacement_feeds(captured, cx, cy)
    ).numpy()


def run_glsl_frame_batch(
    program,
    roles,
    unit_x,
    unit_y,
    *,
    centers,
    spans,
    family_mixes,
    julia_constants,
):
    """Solve an outer batch of animation frames in one fused GLSL dispatch.

    The captured program is unchanged. AbstractTensor broadcasting supplies
    the extra axis: coordinates are ``(1, pixels)`` and per-frame controls are
    ``(frames, 1)``, producing a resident ``(frames, pixels)`` result.
    """

    from .glsl_backend import GLChunk, execute_program

    centers = np.asarray(centers, dtype=np.complex64).reshape(-1)
    spans = np.asarray(spans, dtype=np.float32).reshape(-1)
    family_mixes = np.asarray(family_mixes, dtype=np.float32).reshape(-1)
    julia_constants = np.asarray(
        julia_constants, dtype=np.complex64
    ).reshape(-1)
    frame_count = centers.size
    if frame_count < 1:
        raise ValueError("frame batch must contain at least one frame")
    if any(
        values.size != frame_count
        for values in (spans, family_mixes, julia_constants)
    ):
        raise ValueError("all frame-batch controls must have equal length")
    unit_x = np.asarray(unit_x, dtype=np.float32).reshape(1, -1)
    unit_y = np.asarray(unit_y, dtype=np.float32).reshape(1, -1)
    if unit_x.shape != unit_y.shape:
        raise ValueError("unit_x and unit_y must contain equal pixel counts")

    column = lambda values: np.asarray(values, dtype=np.float32).reshape(-1, 1)
    feeds = {
        roles["unit_x"]: GLChunk.from_numpy(unit_x).to_gpu(),
        roles["unit_y"]: GLChunk.from_numpy(unit_y).to_gpu(),
        roles["center_x"]: GLChunk.from_numpy(column(centers.real)).to_gpu(),
        roles["center_y"]: GLChunk.from_numpy(column(centers.imag)).to_gpu(),
        roles["span"]: GLChunk.from_numpy(column(spans)).to_gpu(),
        roles["family_mix"]: GLChunk.from_numpy(
            column(family_mixes)
        ).to_gpu(),
        roles["julia_x"]: GLChunk.from_numpy(
            column(julia_constants.real)
        ).to_gpu(),
        roles["julia_y"]: GLChunk.from_numpy(
            column(julia_constants.imag)
        ).to_gpu(),
    }
    try:
        return execute_program(program, feeds)
    finally:
        for chunk in feeds.values():
            chunk.release()


def run_c(captured, cx, cy):
    from .c_backend import CTensor
    from .c_primitive_program import execute_fused_program

    feeds = {
        feed_id: CTensor.from_list(array.tolist(), array.shape)
        for feed_id, array in _replacement_feeds(captured, cx, cy).items()
    }
    return np.asarray(
        execute_fused_program(captured.program, feeds).tolist(), dtype=np.float32
    )


def c_workspace_bytes(program, elements: int) -> int:
    """The C interpreter allocates one full slot array per instruction result."""
    return (len(program.feeds) + len(program.steps)) * elements * 8


def animate_glsl(
    *,
    width: int,
    height: int,
    iterations: int,
    center: complex,
    span: float,
    speed: float = 1.0,
    zoom_rate: float = 0.0,
    audio_path: str | Path | None = None,
    audio_gain: float = 1.0,
    reaction: float = 0.20,
    detail_network: bool = True,
    detail_samples: int = 60,
    detail_epochs: int = 20,
    play_audio: bool = True,
    max_frames: int | None = None,
    record_avi: str | Path | None = None,
    record_fps: float = 30.0,
    record_pcm_dtype: str = "s16le",
    record_segment_bytes: int = 1 << 30,
    profile: bool = False,
) -> None:
    """Run the parameterized solve continuously with resident GPU buffers."""
    import pygame
    from OpenGL import GL
    from OpenGL.GL.shaders import compileProgram, compileShader

    from .gl_context import require_gl_context
    from .glsl_backend import (
        GLChunk,
        dispatch_stats,
        shader_cache_stats,
    )

    compile_started = time.perf_counter()
    program, _ = compile_parametric_mandelbrot_glsl(iterations)
    compile_ms = (time.perf_counter() - compile_started) * 1e3
    unit_x, unit_y = normalized_plane(width, height)

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.set_mode(
        (width, height),
        pygame.OPENGL | pygame.DOUBLEBUF,
        vsync=0,
    )
    pygame.display.set_caption("Parametric AbstractTensor Mandelbrot — GLSL")
    info = require_gl_context()
    print(
        f"gpu     : {info['renderer']} (context: {info['source']})\n"
        f"program : {program.source_node_count} selected ProcessGraph nodes -> "
        f"{program.primitive_count} GLSL primitives + "
        f"{program.loop_count} structured loop; one dispatch\n"
        f"compile : {compile_ms:.1f} ms AST/ProcessGraph/source",
        flush=True,
    )

    controller = None
    if detail_network:
        controller, measured_scores = build_detail_controller(
            center,
            span,
            iterations=iterations,
            samples=detail_samples,
            epochs=detail_epochs,
            reaction=reaction,
            zoom_rate=zoom_rate,
        )
        print(
            "detail  : AbstractNN/Adam "
            f"{controller.samples} states x {controller.epochs} epochs | "
            f"loss {controller.initial_loss:.4g}->{controller.final_loss:.4g} | "
            f"holdout r={controller.validation_correlation:.3f} | "
            f"measured {measured_scores.min():.2f}..{measured_scores.max():.2f}",
            flush=True,
        )

    audio = None
    audio_playback_ready = False
    if audio_path is not None:
        pluck = Path(__file__).resolve().parents[5] / "spectral-analyzer"
        if str(pluck) not in sys.path:
            sys.path.insert(0, str(pluck))
        from audio_reactive_controls import AudioReactiveControlStream

        audio = AudioReactiveControlStream(audio_path, gain=audio_gain)
        if play_audio and record_avi is None:
            try:
                pygame.mixer.init(
                    frequency=audio.sample_rate,
                    size=-16,
                    channels=2,
                    buffer=2048,
                )
                pygame.mixer.music.load(str(audio.path))
                audio_playback_ready = True
            except pygame.error as error:
                print(
                    f"audio   : playback unavailable ({error}); "
                    "analysis continues",
                    flush=True,
                )
        elif play_audio and record_avi is not None:
            print(
                "audio   : live playback disabled during offline recording; "
                "PCM remains synchronized to recorded frame time",
                flush=True,
            )
        print(
            f"audio   : {audio.path} | {audio.sample_rate} Hz | "
            "fftfree bass/low-mid/high-mid/treble controls",
            flush=True,
        )

    recorder = None
    audio_scheduler = None
    recorded_audio_position = 0
    if record_avi is not None:
        from ..abstraction import AbstractTensor
        from ..compression.containers.avi import MJPEGAVIWriter
        from ..compression.pcm import PCMFormat, RationalAudioScheduler

        pcm_format = (
            PCMFormat(
                sample_rate=audio.sample_rate,
                channels=1,
                sample_format=record_pcm_dtype,
            )
            if audio is not None
            else None
        )
        recorder = MJPEGAVIWriter(
            record_avi,
            width=width,
            height=height,
            fps=record_fps,
            pcm_format=pcm_format,
            opendml=True,
            segment_bytes=record_segment_bytes,
        )
        if pcm_format is not None:
            audio_scheduler = RationalAudioScheduler(
                sample_rate=pcm_format.sample_rate,
                fps=record_fps,
            )
        print(
            f"record  : {record_avi} | {record_fps:g} fps | "
            "4:4:4 MJPEG/OpenDML"
            + (
                f" + {record_pcm_dtype} mono PCM"
                if pcm_format is not None
                else ""
            ),
            flush=True,
        )

    feeds = {
        "unit_x": GLChunk.from_numpy(unit_x).to_gpu(),
        "unit_y": GLChunk.from_numpy(unit_y).to_gpu(),
        program.scalar_buffer_name: GLChunk.from_numpy(
            np.asarray(
                [
                    center.real,
                    center.imag,
                    span,
                    0.0,
                    -0.72,
                    0.24,
                    0.0,
                    0.52,
                ],
                np.float32,
            )
        ).to_gpu(),
    }
    fused_outputs = {
        name: GLChunk((height * width,), dtype=dtype).to_gpu()
        for name, dtype in zip(program.output_names, program.output_dtypes)
    }
    jpeg_resources = None
    if recorder is not None:
        from ..compression.jpeg.frame import (
            prepare_jpeg_encoding_resources,
        )
        from .glsl_backend import fuse_elementwise, reshape_chunk
        from .glsl_tensor_backend import GLSLTensorOperations

        exemplar = GLSLTensorOperations()
        exemplar.data = reshape_chunk(
            fused_outputs["luminance"], (height, width)
        )
        with AbstractTensor.use_backend("glsl"), fuse_elementwise():
            jpeg_resources = prepare_jpeg_encoding_resources(exemplar)
    display_program = compileProgram(
        compileShader(
            """#version 430 core
            const vec2 corners[3] = vec2[3](
                vec2(-1,-1), vec2(3,-1), vec2(-1,3));
            void main(){ gl_Position=vec4(corners[gl_VertexID],0,1); }""",
            GL.GL_VERTEX_SHADER,
        ),
        compileShader(
            """#version 430 core
            layout(std430, binding=0) readonly buffer YPlane { float y_plane[]; };
            layout(std430, binding=1) readonly buffer CbPlane { float cb_plane[]; };
            layout(std430, binding=2) readonly buffer CrPlane { float cr_plane[]; };
            uniform uint image_width;
            uniform uint image_height;
            out vec4 color;
            void main(){
                uint x=uint(gl_FragCoord.x);
                uint y=uint(gl_FragCoord.y);
                if(x>=image_width || y>=image_height){ color=vec4(0); return; }
                uint index=y*image_width+x;
                float yy=y_plane[index];
                float cb=cb_plane[index]-128.0;
                float cr=cr_plane[index]-128.0;
                vec3 rgb=vec3(
                    yy+1.402*cr,
                    yy-0.344136*cb-0.714136*cr,
                    yy+1.772*cb
                )/255.0;
                color=vec4(clamp(rgb,0.0,1.0),1.0);
            }""",
            GL.GL_FRAGMENT_SHADER,
        ),
    )
    vao = int(GL.glGenVertexArrays(1))
    width_location = GL.glGetUniformLocation(display_program, "image_width")
    height_location = GL.glGetUniformLocation(display_program, "image_height")

    if audio_playback_ready:
        pygame.mixer.music.play(loops=-1)
    started = time.perf_counter()
    previous_time = started
    travel = 0.0
    report_started = started
    report_frames = 0
    frame = 0
    dispatch_baseline = dispatch_stats()
    cache_baseline = shader_cache_stats()
    frame_dispatches = 0
    predicted_detail = 1.0
    profile_rows: list[dict[str, float]] = []
    running = True
    try:
        while running and (max_frames is None or frame < max_frames):
            frame_started = time.perf_counter()
            dispatch_before_frame = dispatch_stats()["calls"]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            now = time.perf_counter()
            if recorder is not None:
                elapsed = frame / record_fps
                delta = 1.0 / record_fps
            else:
                elapsed = now - started
                delta = max(0.0, now - previous_time)
            previous_time = now
            if audio is not None:
                controls = audio.sample(elapsed)
                loudness = controls.loudness
                bass = controls.bass
                low_mid = controls.low_mid
                high_mid = controls.high_mid
                treble = controls.treble
            else:
                loudness = 0.38 + 0.22 * np.sin(elapsed * 1.31)
                bass = 0.5 + 0.5 * np.sin(elapsed * 0.83)
                low_mid = 0.5 + 0.5 * np.sin(elapsed * 0.57 + 0.8)
                high_mid = 0.5 + 0.5 * np.sin(elapsed * 0.73 + 1.7)
                treble = 0.5 + 0.5 * np.sin(elapsed * 1.17 + 0.3)
            # The camera's path speed is the integral of loudness. The learned
            # controller only changes how long we dwell: detailed states slow
            # the tour, while predicted bland states pass quickly.
            detail_speed = 1.0
            if controller is not None:
                candidates = travel + np.asarray([0.0, 0.45, 0.9])
                candidate_features, _ = detail_state_features(
                    center,
                    span,
                    candidates,
                    bass=np.full(3, bass),
                    low_mid=np.full(3, low_mid),
                    high_mid=np.full(3, high_mid),
                    reaction=reaction,
                    zoom_rate=zoom_rate,
                )
                predicted = controller.predict(candidate_features)
                predicted_detail = float(predicted[0])
                best_ahead = float(np.argmax(predicted)) * 0.45
                detail_speed = (
                    0.45 + 1.8 * (1.0 - predicted_detail)
                    + 0.35 * best_ahead
                )
            travel += delta * speed * detail_speed * (
                0.28 + reaction * 1.35 * loudness
            )
            (
                animated_center,
                animated_span,
                family_mix,
                julia_constant,
            ) = dream_parameters(
                center,
                span,
                travel,
                bass=bass,
                low_mid=low_mid,
                high_mid=high_mid,
                reaction=reaction,
                zoom_rate=zoom_rate,
            )
            julia_x, julia_y = julia_constant.real, julia_constant.imag
            palette_phase = float(
                0.028 * travel + reaction * 0.09 * (treble - 0.5)
            )
            color_drive = float(
                0.52 + reaction * 0.24 * (high_mid - 0.5)
            )
            controls_finished = time.perf_counter()
            scalar_values = {
                "center_x": animated_center.real,
                "center_y": animated_center.imag,
                "span": animated_span,
                "family_mix": family_mix,
                "julia_x": julia_x,
                "julia_y": julia_y,
                "palette_phase": palette_phase,
                "color_drive": color_drive,
            }
            feeds[program.scalar_buffer_name].update_numpy(
                np.asarray(
                    [
                        scalar_values[name]
                        for name in program.scalar_input_order
                    ],
                    np.float32,
                )
            ).to_gpu()
            uploads_finished = time.perf_counter()
            query = None
            if profile:
                generated = GL.glGenQueries(1)
                query = int(np.asarray(generated).reshape(-1)[0])
                GL.glBeginQuery(GL.GL_TIME_ELAPSED, query)
            submit_started = time.perf_counter()
            program.execute(feeds, outs=fused_outputs)
            submit_finished = time.perf_counter()
            gpu_ms = 0.0
            if query is not None:
                import ctypes

                GL.glEndQuery(GL.GL_TIME_ELAPSED)
                elapsed_ns = ctypes.c_uint64()
                GL.glGetQueryObjectui64v(
                    query,
                    GL.GL_QUERY_RESULT,
                    ctypes.byref(elapsed_ns),
                )
                gpu_ms = elapsed_ns.value / 1e6
                GL.glDeleteQueries(1, (query,))
            compute_finished = time.perf_counter()

            present_started = time.perf_counter()
            surface = pygame.display.get_surface()
            draw_width, draw_height = surface.get_size()
            GL.glViewport(0, 0, draw_width, draw_height)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glClearColor(0.008, 0.012, 0.028, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glUseProgram(display_program)
            GL.glUniform1ui(width_location, width)
            GL.glUniform1ui(height_location, height)
            GL.glBindBufferBase(
                GL.GL_SHADER_STORAGE_BUFFER,
                0,
                fused_outputs["luminance"].buffer_id,
            )
            GL.glBindBufferBase(
                GL.GL_SHADER_STORAGE_BUFFER,
                1,
                fused_outputs["blue_difference"].buffer_id,
            )
            GL.glBindBufferBase(
                GL.GL_SHADER_STORAGE_BUFFER,
                2,
                fused_outputs["red_difference"].buffer_id,
            )
            GL.glBindVertexArray(vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
            GL.glBindVertexArray(0)
            pygame.display.flip()
            present_finished = time.perf_counter()
            encode_started = present_finished
            if recorder is not None:
                recorder.append_frame(
                    tensor_ycbcr_jpeg_bytes(
                        (
                            fused_outputs["luminance"],
                            fused_outputs["blue_difference"],
                            fused_outputs["red_difference"],
                        ),
                        width,
                        height,
                        resources=jpeg_resources,
                    )
                )
                if audio_scheduler is not None:
                    count = audio_scheduler.samples_for_next_frame()
                    indices = (
                        np.arange(count, dtype=np.int64)
                        + recorded_audio_position
                    ) % len(audio.samples)
                    with AbstractTensor.use_backend("glsl"):
                        recorder.append_audio_tensor(
                            AbstractTensor.tensor(audio.samples[indices])
                        )
                    recorded_audio_position += count
            encode_finished = time.perf_counter()
            frame += 1
            frame_dispatches = (
                dispatch_stats()["calls"] - dispatch_before_frame
            )
            report_frames += 1
            if profile:
                profile_rows.append(
                    {
                        "control": (
                            controls_finished - frame_started
                        ) * 1e3,
                        "uploads": (
                            uploads_finished - controls_finished
                        ) * 1e3,
                        "submit": (
                            submit_finished - submit_started
                        ) * 1e3,
                        "compute_wait": (
                            compute_finished - submit_started
                        ) * 1e3,
                        "gpu": gpu_ms,
                        "present": (
                            present_finished - present_started
                        ) * 1e3,
                        "encode": (
                            encode_finished - encode_started
                        ) * 1e3,
                        "total": (
                            encode_finished - frame_started
                        ) * 1e3,
                    }
                )
            if now - report_started >= 0.5:
                fps = report_frames / (now - report_started)
                pygame.display.set_caption(
                    "Parametric AbstractTensor Mandelbrot — GLSL | "
                    f"{fps:.1f} solve+render fps | span {animated_span:.5g} | "
                    f"family {family_mix:.2f} | detail {predicted_detail:.2f} | "
                    f"loud {loudness:.2f} | GL launches {frame_dispatches}"
                )
                report_started, report_frames = now, 0
        elapsed = time.perf_counter() - started
        print(
            f"animated: {frame} solve+render frames in {elapsed:.3f}s "
            f"({frame / max(elapsed, 1e-9):.1f} fps)",
            flush=True,
        )
        dispatch_final = dispatch_stats()
        cache_final = shader_cache_stats()
        dispatch_count = dispatch_final["calls"] - dispatch_baseline["calls"]
        print(
            "dispatch: "
            f"{dispatch_count} physical GLSL launches "
            f"({dispatch_count / max(frame, 1):.1f}/frame) | "
            f"shader cache "
            f"{cache_final['hits'] - cache_baseline['hits']} hits / "
            f"{cache_final['misses'] - cache_baseline['misses']} misses",
            flush=True,
        )
        if profile_rows:
            warmup = min(5, max(0, len(profile_rows) - 1))
            steady = profile_rows[warmup:] or profile_rows
            print(
                f"profile : steady frames ({warmup} warmup frames excluded)"
            )
            for name in (
                "control",
                "uploads",
                "submit",
                "compute_wait",
                "gpu",
                "present",
                "encode",
                "total",
            ):
                values = np.asarray(
                    [row[name] for row in steady], dtype=np.float64
                )
                print(
                    f"  {name:12s} mean {values.mean():8.3f} ms | "
                    f"p95 {np.quantile(values, 0.95):8.3f} ms",
                    flush=True,
                )
    finally:
        for chunk in feeds.values():
            chunk.release()
        for chunk in fused_outputs.values():
            chunk.release()
        if audio is not None:
            audio.close()
        if audio_playback_ready:
            pygame.mixer.music.stop()
        if recorder is not None:
            recorder.close()
        if jpeg_resources is not None:
            jpeg_resources.release()
        GL.glDeleteVertexArrays(1, (vao,))
        GL.glDeleteProgram(display_program)
        pygame.quit()


# ---------------------------------------------------------------------------
# picture
# ---------------------------------------------------------------------------

def save_image(counts: np.ndarray, width: int, height: int, path: Path,
               cmap: str = "blue_fire", vignette_tile: int = 0) -> Path:
    """Colour with the repository's own colormap rather than a private one.

    ``vignette_tile`` is off (0) by default and for a reason worth recording:
    ``render_cache.add_vignette`` is not a border vignette, it **upscales**, turning
    every input pixel into a ``tile x tile`` bubble. It is built for small conv
    feature maps, where that reads as pixel art. Applied blind to a 1600x1200
    render at its default ``tile=8`` it silently produces a 12800x9600, 11 MB
    image -- which is exactly what the first version of this demo did.
    """
    from PIL import Image
    from ..abstract_convolution.render_cache import add_vignette, apply_colormap

    frame = counts.reshape(height, width)
    # sqrt spreads the low counts, where nearly all the visible structure lives
    rgb = apply_colormap(np.sqrt(frame), cmap=cmap)
    if vignette_tile:
        rgb = add_vignette(rgb, tile=vignette_tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path


def tensor_jpeg_bytes(
    counts,
    width: int,
    height: int,
    iterations: int,
    *,
    palette_phase: float = 0.0,
    color_drive: float = 0.52,
) -> bytes:
    """Encode the displayed GLSL palette through AbstractTensor JPEG."""
    from ..abstraction import AbstractTensor as AT
    from .glsl_backend import (
        GLChunk,
        dispatch_batch,
        fuse_elementwise,
        reshape_chunk,
    )
    from .glsl_tensor_backend import GLSLTensorOperations

    if width % 8 or height % 8:
        raise ValueError("JPEG dimensions must be divisible by eight")
    with AT.use_backend("glsl"):
        if isinstance(counts, GLChunk):
            field = GLSLTensorOperations()
            field.data = reshape_chunk(counts, (height, width))
        else:
            field = AT.tensor(counts.reshape(height, width))
        with dispatch_batch(), fuse_elementwise():
            phase = (
                (field / max(iterations, 1)).clamp(0.0, 1.0).sqrt()
                + float(palette_phase)
            )
            exponent = (
                1.65
                + (0.62 - 1.65) * min(1.0, max(0.0, float(color_drive)))
            )
            rgb = AT.stack(
                (
                    (
                        0.5
                        + 0.5
                        * (6.283185307179586 * phase).cos()
                    ) ** exponent,
                    (
                        0.5
                        + 0.5
                        * (
                            6.283185307179586
                            * (phase + 0.21)
                        ).cos()
                    ) ** exponent,
                    (
                        0.5
                        + 0.5
                        * (
                            6.283185307179586
                            * (phase + 0.43)
                        ).cos()
                    ) ** exponent,
                ),
                dim=-1,
            ) * 255.0
            samples = ((rgb + 0.5) // 1).clamp(0.0, 255.0)
            # Fewer, wider MCU batches materially improve accelerator occupancy:
            # each batch reuses the same AbstractTensor entropy pipeline, while
            # tiny batches repeat hundreds of launches over undersized tensors.
            return samples.jpg(
                mcu_rows_per_batch=min(32, max(1, (height + 7) // 8))
            )


def tensor_ycbcr_jpeg_bytes(
    planes,
    width: int,
    height: int,
    *,
    resources=None,
) -> bytes:
    """Encode resident Y/Cb/Cr outputs without rebuilding RGB."""

    from ..abstraction import AbstractTensor as AT
    from ..compression.jpeg.frame import encode_ycbcr_jfif
    from .glsl_backend import (
        GLChunk,
        dispatch_batch,
        fuse_elementwise,
        reshape_chunk,
    )
    from .glsl_tensor_backend import GLSLTensorOperations

    if width % 8 or height % 8:
        raise ValueError("JPEG dimensions must be divisible by eight")
    if len(planes) != 3:
        raise ValueError("YCbCr encoding needs exactly three planes")
    with AT.use_backend("glsl"):
        wrapped = []
        for plane in planes:
            if isinstance(plane, GLChunk):
                tensor = GLSLTensorOperations()
                tensor.data = reshape_chunk(plane, (height, width))
            else:
                tensor = AT.tensor(plane.reshape(height, width))
            wrapped.append(tensor)
        # The ProcessGraph front end hands us resident Y/Cb/Cr planes. Keep
        # every eligible encoder expression deferred until a true structural
        # or reduction boundary, just as the older RGB entry point already
        # does. Without this scope the graph-optimized front end accidentally
        # fell back to one GLSL launch per primitive throughout JPEG.
        with dispatch_batch(), fuse_elementwise():
            return encode_ycbcr_jfif(
                tuple(wrapped),
                mcu_rows_per_batch=min(32, max(1, (height + 7) // 8)),
                resources=resources,
            )


def save_tensor_jpeg(
    counts: np.ndarray,
    width: int,
    height: int,
    iterations: int,
    path: Path,
) -> Path:
    """Write one fused GLSL solve as a 4:4:4 AbstractTensor JPEG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        tensor_jpeg_bytes(counts, width, height, iterations)
    )
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--iterations", type=int, default=64)
    ap.add_argument("--center", type=complex, default=complex(-0.743643887, 0.131825904))
    ap.add_argument("--span", type=float, default=0.004)
    ap.add_argument("--cmap", default="blue_fire")
    ap.add_argument("--vignette-tile", type=int, default=0,
                    help="per-pixel bubble vignette; UPSCALES by this factor "
                         "(render_cache.add_vignette default is 8). 0 = off.")
    ap.add_argument("--out", type=Path, default=Path("mandelbrot_fused.png"))
    ap.add_argument("--c-probe", type=int, default=48,
                    help="edge length of the small grid cross-checked on the C backend")
    ap.add_argument("--skip-c", action="store_true")
    ap.add_argument(
        "--only-glsl",
        action="store_true",
        help="render with GLSL only; skip the NumPy/f64 oracles and C probe",
    )
    ap.add_argument(
        "--animate",
        action="store_true",
        help="continuously execute the scalar-parameterized GLSL solve",
    )
    ap.add_argument(
        "--animation-speed",
        type=float,
        default=1.0,
        help="camera parameter cycles per wall-clock second multiplier",
    )
    ap.add_argument(
        "--animation-frames",
        type=int,
        default=0,
        help="stop after this many frames; 0 runs until ESC",
    )
    ap.add_argument(
        "--zoom-rate",
        type=float,
        default=0.0,
        help="positive continuously tightens the view; negative loosens it",
    )
    ap.add_argument(
        "--audio",
        type=Path,
        help="loop an audio file and drive the complex dream path with fftfree",
    )
    ap.add_argument(
        "--audio-gain",
        type=float,
        default=1.0,
        help="gain before adaptive spectral control normalization",
    )
    ap.add_argument(
        "--reaction",
        type=float,
        default=0.20,
        help="audio modulation depth; 0 is the restrained autonomous path",
    )
    ap.add_argument(
        "--no-detail-network",
        action="store_true",
        help="disable the tiny AbstractNN controller that skips bland states",
    )
    ap.add_argument(
        "--detail-samples",
        type=int,
        default=60,
        help="low-resolution dream states used to train the detail controller",
    )
    ap.add_argument(
        "--detail-epochs",
        type=int,
        default=20,
        help="AbstractNN training epochs for the live detail controller",
    )
    ap.add_argument(
        "--silent-audio",
        action="store_true",
        help="analyze --audio without playing it",
    )
    ap.add_argument(
        "--record-avi",
        type=Path,
        help="record the animated GLSL solve as 4:4:4 MJPEG/OpenDML AVI",
    )
    ap.add_argument("--record-fps", type=float, default=30.0)
    ap.add_argument(
        "--record-pcm-dtype",
        choices=("s16le", "f32le"),
        default="s16le",
    )
    ap.add_argument(
        "--record-segment-bytes",
        type=int,
        default=1 << 30,
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="synchronize GPU timer queries and print per-stage timings",
    )
    args = ap.parse_args(argv)

    if args.animate:
        animate_glsl(
            width=args.width,
            height=args.height,
            iterations=args.iterations,
            center=args.center,
            span=args.span,
            speed=args.animation_speed,
            zoom_rate=args.zoom_rate,
            audio_path=args.audio,
            audio_gain=args.audio_gain,
            reaction=args.reaction,
            detail_network=not args.no_detail_network,
            detail_samples=args.detail_samples,
            detail_epochs=args.detail_epochs,
            play_audio=not args.silent_audio,
            max_frames=args.animation_frames or None,
            record_avi=args.record_avi,
            record_fps=args.record_fps,
            record_pcm_dtype=args.record_pcm_dtype,
            record_segment_bytes=args.record_segment_bytes,
            profile=args.profile,
        )
        return 0

    elements = args.width * args.height
    print(f"image   : {args.width}x{args.height} = {elements:,} pixels")

    cx, cy = complex_plane(args.width, args.height, args.center, args.span)
    compiled, _ = compile_parametric_mandelbrot_glsl(args.iterations)
    print(
        f"program : {compiled.source_node_count} ProcessGraph nodes -> "
        f"{compiled.primitive_count} primitives + "
        f"{compiled.loop_count} structured loop"
    )

    # -- GPU ---------------------------------------------------------------
    from .gl_context import require_gl_context
    from .glsl_backend import GLChunk

    info = require_gl_context()
    print(f"gpu     : {info['renderer']} (context: {info['source']})")

    unit_x, unit_y = normalized_plane(args.width, args.height)
    static_scalars = {
        "center_x": args.center.real,
        "center_y": args.center.imag,
        "span": args.span,
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
                    static_scalars[name]
                    for name in compiled.scalar_input_order
                ],
                dtype=np.float32,
            )
        ).to_gpu(),
    }
    t0 = time.perf_counter()
    gpu_outputs = compiled.execute(feeds)
    gpu = gpu_outputs["counts"].numpy().copy()
    gpu_ms = (time.perf_counter() - t0) * 1e3
    print(
        f"glsl    : {gpu_ms:8.1f} ms  "
        f"({args.iterations} loop iterations x {elements:,} px, one dispatch)"
    )
    for chunk in (*feeds.values(), *gpu_outputs.values()):
        chunk.release()

    if not args.only_glsl:
        # -- oracle --------------------------------------------------------
        t0 = time.perf_counter()
        ref = run_abstract_numpy(cx, cy, args.iterations)
        np_ms = (time.perf_counter() - t0) * 1e3
        print(f"numpy   : {np_ms:8.1f} ms  (same AbstractTensor function)")

        max_err = float(np.max(np.abs(gpu - ref)))
        agree = float(np.mean(gpu == ref)) * 100.0
        print(f"agree   : {agree:.4f}% exact vs numpy-f32, max |diff| = {max_err:g}")

        # Escape-time is chaotic: a 1-ULP boundary difference can change the
        # escape iteration. Compare both float32 paths with float64 so precision
        # sensitivity is not mistaken for a lowering defect.
        ref64 = run_abstract_numpy(
            cx.astype(np.float64), cy.astype(np.float64), args.iterations
        )
        gpu_vs64 = float(np.mean(gpu == ref64)) * 100.0
        np_vs64 = float(np.mean(ref == ref64)) * 100.0
        print(f"vs f64  : glsl-f32 {gpu_vs64:.4f}%, numpy-f32 {np_vs64:.4f}% "
              f"-- both f32 paths differ from f64 by a comparable margin")
        if max_err > 0:
            disagree = gpu != ref
            c2 = ref.reshape(args.height, args.width)
            edge = np.zeros_like(c2, dtype=bool)
            edge[1:-1, 1:-1] = (
                (c2[1:-1, 1:-1] != c2[:-2, 1:-1])
                | (c2[1:-1, 1:-1] != c2[2:, 1:-1])
                | (c2[1:-1, 1:-1] != c2[1:-1, :-2])
                | (c2[1:-1, 1:-1] != c2[1:-1, 2:])
            )
            on_edge = float(np.mean(edge.ravel()[disagree])) * 100.0
            print(f"        : {disagree.sum():,} disagreeing px, {on_edge:.1f}% of them sit "
                  f"on an escape-count boundary (chaotic sensitivity, not a lowering bug)")
    else:
        print("verify  : skipped (--only-glsl)")

    # -- C backend, on a small grid ----------------------------------------
    if not args.skip_c and not args.only_glsl:
        print(
            "c probe : skipped; the C backend does not yet lower structured "
            "ProcessGraph loops (no tape fallback)"
        )

    if args.out.suffix.lower() in {".jpg", ".jpeg"}:
        out = save_tensor_jpeg(
            gpu,
            args.width,
            args.height,
            args.iterations,
            args.out,
        )
    else:
        out = save_image(
            gpu,
            args.width,
            args.height,
            args.out,
            args.cmap,
            vignette_tile=args.vignette_tile,
        )
    print(f"wrote   : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
