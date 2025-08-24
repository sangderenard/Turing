import numpy as np
import os
import pytest
from src.rendering.ascii_render import AsciiRenderer
from src.rendering.ascii_diff import ThreadedAsciiDiffPrinter

@pytest.mark.skipif(os.name != "nt", reason="fast console requires Windows")
def test_ascii_diff_animation():
    width, height = 32, 16
    r = AsciiRenderer(width, height)
    printer = ThreadedAsciiDiffPrinter()
    frames = 10
    for frame in range(frames):
        r.clear()
        cx = int((width // 2) + (width // 3) * np.sin(frame * 2 * np.pi / frames))
        cy = int((height // 2) + (height // 3) * np.cos(frame * 2 * np.pi / frames))
        r.circle(cx, cy, 4, value=1)
        r.line(frame % width, 0, width - 1 - (frame % width), height - 1, value=1)
        ascii_out = r.to_ascii_diff()
        if ascii_out:
            printer.enqueue(ascii_out)
    printer.wait_until_empty()
    printer.stop()
