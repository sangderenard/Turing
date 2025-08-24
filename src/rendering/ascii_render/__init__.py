"""Basic ASCII renderer operating on a numpy array canvas.

This module is intentionally minimal and serves as a placeholder for a future,
more capable rendering system.  It can draw primitive shapes (points, lines,
circles and triangles) onto an integer or floating point canvas and export a
very small ASCII art approximation using a luminance ramp.
"""

from __future__ import annotations

from typing import Tuple
from io import StringIO
from contextlib import redirect_stdout
import os
import time

import numpy as np

# Import relevant ascii_diff components
from src.rendering.ascii_diff import (
    draw_diff,
    PixelFrameBuffer,
    default_subunit_batch_to_chars,
    DEFAULT_DRAW_ASCII_RAMP,
)

__all__ = ["AsciiRenderer"]


class AsciiRenderer:
    """Draw simple shapes onto a numpy canvas and render ASCII art.

    Parameters
    ----------
    width, height:
        Canvas size in pixels.
    depth:
        Number of value channels per pixel.  Defaults to ``1``.
    float_mode:
        If ``True`` the canvas uses ``float`` values, otherwise ``uint8``.
    """

    ramp = np.asarray(list(" .:-=+*#%@"))

    def __init__(self, width: int, height: int, depth: int = 1, *, float_mode: bool = False,
                 char_cell_pixel_height: int = 1, char_cell_pixel_width: int = 1,
                 enable_fg_color: bool = False, enable_bg_color: bool = False) -> None:
        dtype = float if float_mode else np.uint8
        self.canvas = np.zeros((height, width, depth), dtype=dtype)
        # Maintain a persistent frame buffer for diffing
        self._fb = PixelFrameBuffer((height, width))
        # Profiling support toggled via the TURING_PROFILE env var
        self.profile = bool(int(os.getenv("TURING_PROFILE", "0")))
        self.profile_stats: dict[str, float] = {"to_ascii_diff_ms": 0.0}
        # Record per-call durations when profiling
        self.to_ascii_diff_durations: list[float] = []
        # Char cell and color settings
        self.char_cell_pixel_height = char_cell_pixel_height
        self.char_cell_pixel_width = char_cell_pixel_width
        self.enable_fg_color = enable_fg_color
        self.enable_bg_color = enable_bg_color

    # -- canvas helpers -------------------------------------------------
    def clear(self, value: float | int = 0) -> None:
        """Fill the canvas with ``value``."""

        self.canvas[...] = value

    def _plot(self, x: int, y: int, value: float | int) -> None:
        if 0 <= x < self.canvas.shape[1] and 0 <= y < self.canvas.shape[0]:
            self.canvas[y, x, ...] = value

    # -- primitive shapes ------------------------------------------------
    def point(self, x: int, y: int, value: float | int = 1) -> None:
        self._plot(x, y, value)

    def line(self, x0: int, y0: int, x1: int, y1: int, value: float | int = 1) -> None:
        """Draw a line from ``(x0, y0)`` to ``(x1, y1)`` using Bresenham's algorithm."""

        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self._plot(x0, y0, value)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def circle(self, cx: int, cy: int, radius: int, value: float | int = 1) -> None:
        """Draw the circumference of a circle."""

        x, y, err = radius, 0, 0
        while x >= y:
            pts = [
                (cx + x, cy + y),
                (cx + y, cy + x),
                (cx - y, cy + x),
                (cx - x, cy + y),
                (cx - x, cy - y),
                (cx - y, cy - x),
                (cx + y, cy - x),
                (cx + x, cy - y),
            ]
            for px, py in pts:
                self._plot(px, py, value)
            y += 1
            if err <= 0:
                err += 2 * y + 1
            if err > 0:
                x -= 1
                err -= 2 * x + 1

    def triangle(
        self,
        p0: Tuple[int, int],
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        value: float | int = 1,
    ) -> None:
        """Draw a triangle by connecting three vertices."""

        self.line(*p0, *p1, value)
        self.line(*p1, *p2, value)
        self.line(*p2, *p0, value)

    # -- image operations ------------------------------------------------
    def paint(self, image: np.ndarray, x: int = 0, y: int = 0) -> None:
        """Blit ``image`` onto the canvas with the top-left corner at ``(x, y)``.

        The image is broadcast or reduced to match the canvas depth.  Values
        outside the canvas bounds are clipped.  This is a placeholder for more
        advanced composition routines.
        """

        h, w = image.shape[:2]
        x_end = min(self.canvas.shape[1], x + w)
        y_end = min(self.canvas.shape[0], y + h)
        if x_end <= x or y_end <= y:
            return

        target = self.canvas[y:y_end, x:x_end, ...]
        src = image[: y_end - y, : x_end - x, ...]

        if src.ndim == 2:
            src = src[..., None]

        if src.shape[2] == target.shape[2]:
            target[...] = src
        elif target.shape[2] == 1:
            target[..., 0] = src.mean(axis=2)
        elif src.shape[2] == 1:
            target[...] = np.repeat(src, target.shape[2], axis=2)
        else:
            depth = min(src.shape[2], target.shape[2])
            target[..., :depth] = src[..., :depth]

    # -- ASCII conversion ------------------------------------------------
    def to_ascii(self) -> str:
        """Return an ASCII art representation of the canvas.

        Luminance is computed as the mean over depth and mapped to a tiny ramp of
        characters.  This is intentionally simplistic.
        """

        luminance = self.canvas.mean(axis=2)
        norm = luminance - luminance.min()
        maxv = norm.max()
        if maxv > 0:
            norm /= maxv
        idx = (norm * (len(self.ramp) - 1)).astype(int)
        rows = ["".join(self.ramp[row]) for row in idx]
        return "\n".join(rows)

    # -- ascii_diff integration ------------------------------------------------
    def to_ascii_diff(self, ramp: str | None = None,
                      char_cell_pixel_height: int = None,
                      char_cell_pixel_width: int = None,
                      enable_fg_color: bool = None,
                      enable_bg_color: bool = None) -> str:
        """
        Return an ASCII diff of the current canvas using a persistent frame buffer.

        Only regions that changed since the last call are emitted.
        """
        start = time.perf_counter() if self.profile else None

        tensor = self.canvas
        if tensor.shape[2] == 1:
            rgb = np.repeat(tensor, 3, axis=2)
        elif tensor.shape[2] >= 3:
            rgb = tensor[..., :3]
        else:
            rgb = np.zeros((*tensor.shape[:2], 3), dtype=tensor.dtype)
            rgb[..., : tensor.shape[2]] = tensor
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        self._fb.update_render(rgb)
        updates = self._fb.get_diff_and_promote()
        if not updates:
            return ""

        # Convert updates to the subunit format expected by draw_diff
        changed_subunits = [
            (y, x, np.array([[[r, g, b]]], dtype=np.uint8))
            for y, x, (r, g, b) in updates
        ]

        ascii_ramp = ramp or DEFAULT_DRAW_ASCII_RAMP
        buffer = StringIO()
        # Use instance settings unless overridden
        c_h = char_cell_pixel_height if char_cell_pixel_height is not None else self.char_cell_pixel_height
        c_w = char_cell_pixel_width if char_cell_pixel_width is not None else self.char_cell_pixel_width
        fg = enable_fg_color if enable_fg_color is not None else self.enable_fg_color
        bg = enable_bg_color if enable_bg_color is not None else self.enable_bg_color
        #with redirect_stdout(buffer):
        print("draw diff starting")
        draw_diff(
                changed_subunits,
                char_cell_pixel_height=c_h,
                char_cell_pixel_width=c_w,
                subunit_to_char_kernel=default_subunit_batch_to_chars,
                active_ascii_ramp=ascii_ramp,
                enable_fg_color=fg,
                enable_bg_color=bg,
            )
        print("draw diff over")
        ascii_out = buffer.getvalue()
        if self.profile and start is not None:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.profile_stats["to_ascii_diff_ms"] += elapsed
            self.to_ascii_diff_durations.append(elapsed)
        return ascii_out
