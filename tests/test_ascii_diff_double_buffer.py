import numpy as np
import pytest

from src.common.double_buffer import DoubleBuffer

from src.rendering.ascii_render import AsciiRenderer
from src.rendering.ascii_diff import PixelFrameBuffer, draw_diff, DEFAULT_DRAW_ASCII_RAMP


import time

def test_ascii_diff_animation():
    width, height = 32, 16
    r = AsciiRenderer(width, height)
    prev_canvas = None
    frames = 20
    for frame in range(frames):
        r.clear()
        # Animate a moving circle and a bouncing line
        cx = int((width // 2) + (width // 3) * np.sin(frame * 2 * np.pi / frames))
        cy = int((height // 2) + (height // 3) * np.cos(frame * 2 * np.pi / frames))
        r.circle(cx, cy, 4, value=1)
        r.line(frame % width, 0, width - 1 - (frame % width), height - 1, value=1)
        # Use the new ascii_diff output method
        ascii_out = r.to_ascii_diff(prev_buffer=prev_canvas)
        print(f"\x1b[2J\x1b[HFrame {frame+1}/{frames}\n" + ascii_out)
        prev_canvas = r.canvas.copy()
        time.sleep(0.1)
