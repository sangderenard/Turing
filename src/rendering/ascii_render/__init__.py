"""Basic ASCII renderer operating on a numpy array canvas.

This module is intentionally minimal and serves as a placeholder for a future,
more capable rendering system.  It can draw primitive shapes (points, lines,
circles and triangles) onto an integer or floating point canvas and export a
very small ASCII art approximation using a luminance ramp.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

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

    def __init__(self, width: int, height: int, depth: int = 1, *, float_mode: bool = False) -> None:
        dtype = float if float_mode else np.uint8
        self.canvas = np.zeros((height, width, depth), dtype=dtype)

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

        The image's channel depth must match the canvas depth.  Values outside
        the canvas bounds are clipped.  This is a placeholder for more advanced
        composition routines.
        """

        h, w = image.shape[:2]
        x_end = min(self.canvas.shape[1], x + w)
        y_end = min(self.canvas.shape[0], y + h)
        if x_end <= x or y_end <= y:
            return
        self.canvas[y:y_end, x:x_end, ...] = image[: y_end - y, : x_end - x, ...]

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
