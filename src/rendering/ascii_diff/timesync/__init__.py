#!/usr/bin/env python3
"""Time synchronization utilities."""
from __future__ import annotations

# --- END HEADER ---

from .core import (
    sync_offset,
    get_offset,
    set_offset,
    adjust_datetime,
    now,
    utcnow,
)
from .ascii_digits import (
    compose_ascii_digits,
    render_ascii_to_array,
)
from .clock_renderer import ClockRenderer
from .console import (
    init_colorama_for_windows,
    full_clear_and_reset_cursor,
    reset_cursor_to_top,
)
from ..frame_buffer import PixelFrameBuffer # Changed from AsciiFrameBuffer
from ..render_thread import render_loop
from ..draw import draw_diff, flexible_subunit_kernel
try:
    from ..subunit_window import SubunitWindow
except Exception:  # pragma: no cover - optional
    SubunitWindow = None  # type: ignore
from . import _internet  # exported for tests
from .render_backend import RenderingBackend

__all__ = [
    "sync_offset",
    "get_offset",
    "set_offset",
    "adjust_datetime",
    "now",
    "utcnow",
    "compose_ascii_digits",
    "render_ascii_to_array",
    "ClockRenderer",
    "init_colorama_for_windows",
    "full_clear_and_reset_cursor",
    "reset_cursor_to_top",
    "PixelFrameBuffer", # Changed from AsciiFrameBuffer
    "render_loop",
    "draw_diff",
    "flexible_subunit_kernel",
    "SubunitWindow",
    "_internet",
    "RenderingBackend",
]
