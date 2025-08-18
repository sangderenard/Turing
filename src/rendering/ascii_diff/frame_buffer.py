"""Triple-buffered ASCII frame diff manager."""
from __future__ import annotations

import numpy as np
import threading


class PixelFrameBuffer:
    """Manage pixel data frames for flicker-free terminal rendering.
    Each 'pixel' corresponds to a character cell in the terminal.
    """

    def __init__(self, shape: tuple[int, int], diff_threshold: int = 0):
        """Allocate buffers for ``shape`` (rows, cols) and initialize contents.
        Buffers will store RGB data, so actual shape is (rows, cols, 3).
        `diff_threshold`: Sum of absolute differences in RGB channels needed to mark a pixel as changed.
                          0 means any difference. Max is 3 * 255 = 765.
        """
        self.lock = threading.Lock()
        self.buffer_shape = (shape[0], shape[1], 3) # rows, cols, RGB
        self.default_pixel = np.array([0, 0, 0], dtype=np.uint8) # Black
        self.buffer_render = np.full(self.buffer_shape, self.default_pixel, dtype=np.uint8)
        self.buffer_next = np.full(self.buffer_shape, self.default_pixel, dtype=np.uint8)
        self.buffer_display = np.full(self.buffer_shape, self.default_pixel, dtype=np.uint8)
        self.diff_threshold = max(0, diff_threshold) # Ensure threshold is not negative
        self._force_full_diff_next_call = True

    def __repr__(self) -> str:
        return (
            f"PixelFrameBuffer(shape={self.buffer_shape[:2]}, "
            f"diff_threshold={self.diff_threshold})"
        )

    def _resize(self, shape: tuple[int, int]) -> None:
        """Resize internal buffers to ``shape`` without acquiring the lock."""
        self.buffer_shape = (shape[0], shape[1], 3)
        self.buffer_render = np.full(
            self.buffer_shape, self.default_pixel, dtype=np.uint8
        )
        self.buffer_next = np.full(
            self.buffer_shape, self.default_pixel, dtype=np.uint8
        )
        self.buffer_display = np.full(
            self.buffer_shape, self.default_pixel, dtype=np.uint8
        )
        self._force_full_diff_next_call = True

    def resize(self, shape: tuple[int, int]) -> None:
        """Thread-safe wrapper around :meth:`_resize`."""
        with self.lock:
            self._resize(shape)

    def update_render(self, new_data: np.ndarray) -> None:
        """Update the render buffer with ``new_data``.

        Resizes internal buffers if ``new_data`` shape differs from
        the current configuration.
        """
        if new_data.shape != self.buffer_shape:
            self.resize((new_data.shape[0], new_data.shape[1]))
        with self.lock:
            np.copyto(self.buffer_render, new_data)

    def force_full_redraw_next_frame(self) -> None:
        """Signals that the next call to get_diff_and_promote should return all pixels."""
        self._force_full_diff_next_call = True

    def get_diff_and_promote(self) -> list[tuple[int, int, tuple[int, int, int]]]:
        """Return changed cells since last promotion.

        Copies the render buffer into the next buffer, computes the
        difference against the display buffer, updates the display
        buffer, and returns a list of ``(y, x, (r,g,b))`` tuples for the
        cells that changed.
        """
        with self.lock:
            np.copyto(self.buffer_next, self.buffer_render)
        
        if self._force_full_diff_next_call:
            # Force all pixels to be considered changed
            rows, cols, _ = self.buffer_shape
            ys, xs = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
            coords = np.stack([ys.ravel(), xs.ravel()], axis=1)
            self._force_full_diff_next_call = False # Reset the flag
        else:
            if self.diff_threshold == 0:
                # Original behavior: any difference in any channel
                diff_mask = np.any(self.buffer_next != self.buffer_display, axis=2)
            else:
                # Calculate sum of absolute differences for RGB channels
                abs_diff = np.abs(self.buffer_next.astype(np.int16) - self.buffer_display.astype(np.int16))
                sum_abs_diff = np.sum(abs_diff, axis=2)
                diff_mask = sum_abs_diff > self.diff_threshold
            coords = np.argwhere(diff_mask)

        updates: list[tuple[int, int, tuple[int, int, int]]] = []
        for y, x in coords:
            pixel_val = self.buffer_next[y, x]
            updates.append((int(y), int(x), (int(pixel_val[0]), int(pixel_val[1]), int(pixel_val[2]))))
        np.copyto(self.buffer_display, self.buffer_next)
        return updates
