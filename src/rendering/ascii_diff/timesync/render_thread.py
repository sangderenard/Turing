#!/usr/bin/env python3
"""Background thread for producing ASCII frames."""
from __future__ import annotations

import numpy as np
import threading
import time
from typing import Callable
from .frame_buffer import PixelFrameBuffer # Changed from AsciiFrameBuffer
# --- END HEADER ---


def render_loop(
    framebuffer: PixelFrameBuffer, # Changed from AsciiFrameBuffer
    render_fn: Callable[[], np.ndarray],
    fps: float,
    stop_event: threading.Event,
) -> None:
    """Continuously render frames until ``stop_event`` is set."""
    delay = 1.0 / max(fps, 0.1)
    while not stop_event.is_set():
        pixel_frame = render_fn() # render_fn now returns a (H, W, 3) pixel array
        framebuffer.update_render(pixel_frame)
        time.sleep(delay)
