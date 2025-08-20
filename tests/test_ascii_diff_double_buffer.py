import numpy as np
import time

from src.rendering.ascii_render import AsciiRenderer

def test_ascii_diff_animation():
    width, height = 32, 16
    r = AsciiRenderer(width, height)
    frames = 20
    print("\x1b[2J", end="")  # Clear screen once
    for frame in range(frames):
        r.clear()
        # Animate a moving circle and a bouncing line
        cx = int((width // 2) + (width // 3) * np.sin(frame * 2 * np.pi / frames))
        cy = int((height // 2) + (height // 3) * np.cos(frame * 2 * np.pi / frames))
        r.circle(cx, cy, 4, value=1)
        r.line(frame % width, 0, width - 1 - (frame % width), height - 1, value=1)
        # Emit only changed regions
        ascii_out = r.to_ascii_diff()
        print(ascii_out, end="")
        time.sleep(0.05)
