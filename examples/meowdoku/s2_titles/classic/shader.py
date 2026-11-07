"""The Meowdoku board as one authored per-pixel law, compiled to a shader.

The board is drawn by ``meowdoku_pixels``: an elementwise AbstractTensor
function of the pixel's normalized position and of the board state sampled
at that position.  It is ordinary authored Python, so it runs eagerly on the
numpy backend (the Python version, which is the reference) and it is
captured by the compiler's AOT front end and lowered by the existing machine
targets:

* ``webgl`` -- a GLSL ES 3.00 fragment shader (``fused_program_webgl_backend``),
  rgba presentation output, feeds sampled at the normalized fragment position
  so a 5x5 board texture stretches over the canvas with NEAREST filtering;
* ``glsl`` -- the desktop compute shader (``machine_targets`` ``glsl``).

Nothing here is hand-written GLSL.  The only vocabulary used is the one both
targets list (``machine_targets.capabilities()``): arithmetic, floor, abs,
sin, min/max, comparisons.  Masks are comparisons used as numbers, exactly as
``dom_shader_demo`` writes them, so the eager numpy evaluation and the shader
agree: a bool sum is clamped with ``minimum(1.0)`` before it is used.

Feeds (all float):

``u``, ``v``       the pixel's normalized position, one value per pixel
``cat``, ``x_mark``, ``region``   the board, one value per cell (n x n)
``hearts``, ``n``  scalars (1 x 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

FEED_NAMES = ("u", "v", "cat", "x_mark", "region", "hearts", "n")
OUTPUT_NAMES = ("red", "green", "blue")

SHADER_SOURCE = '''
def meowdoku_pixels(u, v, cat, x_mark, region, hearts, n):
    gx = u * n
    gy = v * n
    fx = gx - gx.floor()
    fy = gy - gy.floor()
    share = region / n
    base_r = 0.62 + 0.30 * (share * 6.2831853 + 0.0).sin()
    base_g = 0.62 + 0.30 * (share * 6.2831853 + 2.0943951).sin()
    base_b = 0.62 + 0.30 * (share * 6.2831853 + 4.1887902).sin()
    on_line = (fx < 0.035) + (fy < 0.035) + (fx > 0.965) + (fy > 0.965)
    grid = on_line.minimum(1.0)
    dx = fx - 0.5
    dy = fy - 0.5
    body = ((dx * dx + dy * dy) < 0.075) * cat
    ear_left = (((dx + 0.18).abs() + (dy + 0.24).abs()) < 0.13) * cat
    ear_right = (((dx - 0.18).abs() + (dy + 0.24).abs()) < 0.13) * cat
    cat_ink = (body + ear_left + ear_right).minimum(1.0)
    inside = (fx > 0.22) * (fx < 0.78) * (fy > 0.22) * (fy < 0.78)
    cross = (((fx - fy).abs() < 0.07) + ((fx + fy - 1.0).abs() < 0.07)).minimum(1.0) * inside * x_mark
    alive = hearts > 0.0
    tone = 0.35 + 0.65 * alive
    red = base_r * tone
    green = base_g * tone
    blue = base_b * tone
    red = red * (1.0 - grid) + 0.12 * grid
    green = green * (1.0 - grid) + 0.12 * grid
    blue = blue * (1.0 - grid) + 0.14 * grid
    red = red * (1.0 - cross) + 0.30 * cross
    green = green * (1.0 - cross) + 0.30 * cross
    blue = blue * (1.0 - cross) + 0.32 * cross
    red = red * (1.0 - cat_ink) + 0.96 * cat_ink
    green = green * (1.0 - cat_ink) + 0.55 * cat_ink
    blue = blue * (1.0 - cat_ink) + 0.16 * cat_ink
    return red, green, blue
'''

_NAMESPACE: dict[str, Any] = {}
exec(SHADER_SOURCE, _NAMESPACE)
meowdoku_pixels = _NAMESPACE["meowdoku_pixels"]


def board_feeds(columns: Mapping[str, np.ndarray], n: int, hearts: float,
                width: int, height: int) -> dict[str, np.ndarray]:
    """The shader's feeds for one board state: ``u``/``v`` per pixel
    (row 0 is the top of the canvas), the board columns as ``n x n`` arrays
    (row 0 is board row 0), ``hearts`` and ``n`` as ``1 x 1``."""

    px = (np.arange(width, dtype=np.float64) + 0.5) / float(width)
    py = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
    u = np.broadcast_to(px[None, :], (height, width)).copy()
    v = np.broadcast_to(py[:, None], (height, width)).copy()
    return {
        "u": u, "v": v,
        "cat": np.asarray(columns["cat"], dtype=np.float64).reshape(n, n),
        "x_mark": np.asarray(columns["x_mark"], dtype=np.float64).reshape(n, n),
        "region": np.asarray(columns["region"], dtype=np.float64).reshape(n, n),
        "hearts": np.full((1, 1), float(hearts)),
        "n": np.full((1, 1), float(n)),
    }


def session_feeds(session, width: int, height: int) -> dict[str, np.ndarray]:
    """``board_feeds`` from a live ``MeowdokuSession`` (the Python game)."""

    def value(name):
        held = getattr(session, name)
        return held() if callable(held) else held

    return board_feeds(
        {"cat": value("cat"), "x_mark": value("x_mark"),
         "region": session.puzzle.region},
        session.n, value("hearts"), width, height,
    )


def _nearest_upsampled(feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every feed as a per-pixel array, sampled the way the fragment shader
    samples it: NEAREST at the pixel's normalized position."""

    u, v = feeds["u"], feeds["v"]
    height, width = u.shape
    sampled = {"u": u, "v": v}
    for name in ("cat", "x_mark", "region", "hearts", "n"):
        table = np.asarray(feeds[name], dtype=np.float64)
        rows, cols = table.shape
        row_index = np.minimum((v * rows).astype(int), rows - 1)
        col_index = np.minimum((u * cols).astype(int), cols - 1)
        sampled[name] = table[row_index, col_index]
    return sampled


def render_reference(feeds: Mapping[str, np.ndarray]) -> np.ndarray:
    """The Python version: the authored law on the numpy backend, as an
    ``(height, width, 3)`` float image in [0, 1]."""

    from src.common.tensors import AbstractTensor

    sampled = _nearest_upsampled(feeds)
    with AbstractTensor.use_backend("numpy"):
        tensors = [AbstractTensor.tensor(sampled[name]) for name in FEED_NAMES]
        channels = meowdoku_pixels(*tensors)
        image = np.stack([np.asarray(channel.numpy(), dtype=np.float64)
                          for channel in channels], axis=-1)
    return np.clip(image, 0.0, 1.0)


@dataclass(frozen=True)
class MeowdokuShader:
    """The captured program and its two shader artifacts."""

    program: Any
    feed_ids: dict[str, int]
    webgl: Any      # fused_program_webgl_backend.WebGLFragmentModule
    glsl: Any       # machine_targets.TargetArtifact (compute)

    @property
    def complete(self) -> bool:
        return bool(self.webgl.complete and self.glsl.complete)


def compile_shader(width: int = 8, height: int = 8, n: int = 2) -> MeowdokuShader:
    """Capture ``meowdoku_pixels`` through the AOT front end and emit it as
    the WebGL fragment shader and the GLSL compute shader.

    The capture feeds are sample arrays of one common shape; they fix the
    dtype records, not the sizes a page binds later."""

    from src.compiler import machine_targets
    from src.compiler.fused_program_wasm_backend import feed_names, program_feed_order
    from src.compiler.fused_program_webgl_backend import emit_webgl_fragment_module
    from tests.wasm_captured_programs import captured_program

    columns = {
        "cat": np.zeros(n * n), "x_mark": np.zeros(n * n),
        "region": np.arange(n * n, dtype=np.float64) % n,
    }
    columns["cat"][0] = 1.0
    columns["x_mark"][-1] = 1.0
    sample = _nearest_upsampled(board_feeds(columns, n, 3.0, width, height))
    program = captured_program(SHADER_SOURCE, **{name: sample[name] for name in FEED_NAMES})
    # The capture records which source binding each feed came from
    # (``capture_feed_origins``); ``feed_names`` reads it.  Never zip the
    # parameter list against ``program_feed_order``: that order is first
    # use, and ``n`` is used before ``v`` here.
    order = program_feed_order(program)
    feed_ids = dict(zip(feed_names(program, order), order))
    if set(feed_ids) != set(FEED_NAMES):
        raise RuntimeError(f"captured feeds {sorted(feed_ids)} are not {FEED_NAMES}")
    webgl = emit_webgl_fragment_module(
        program, name="meowdoku_board", output_layout="rgba",
        input_sampling="normalized",
    )
    glsl = machine_targets.emit(program, "glsl", name="meowdoku_board")
    return MeowdokuShader(program, feed_ids, webgl, glsl)


__all__ = [
    "FEED_NAMES", "OUTPUT_NAMES", "SHADER_SOURCE", "MeowdokuShader",
    "board_feeds", "compile_shader", "meowdoku_pixels", "render_reference",
    "session_feeds",
]
