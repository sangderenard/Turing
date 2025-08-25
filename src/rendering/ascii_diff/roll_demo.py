"""Demo: roll a tensor image along both axes and display via RenderChooser."""

from __future__ import annotations

import time
import numpy as np
from src.rendering.render_chooser import RenderChooser


def roll_image(img: np.ndarray, t: int, period_y: int, period_x: int) -> np.ndarray:
    """Return ``img`` rolled along Y and X using sinusoidal shifts.

    ``period_y`` and ``period_x`` control the oscillation period for the
    respective axes.  The shifts are quarter amplitude of the image size.
    """

    shift_y = int(np.sin(2 * np.pi * t / period_y) * img.shape[0] / 4)
    shift_x = int(np.sin(2 * np.pi * t / period_x) * img.shape[1] / 4)
    return np.roll(np.roll(img, shift_y, axis=0), shift_x, axis=1)


def main() -> None:
    width, height = 128, 128
    base = np.indices((height, width)).sum(axis=0) % 2 * 255

    chooser = RenderChooser(width, height, mode="ascii")
    period_y = height
    period_x = width * 2
    try:
        for t in range(max(period_y, period_x)):
            frame = roll_image(base, t, period_y, period_x)
            chooser.render({"image": frame})
            time.sleep(0.05)
    finally:
        chooser.close()


if __name__ == "__main__":
    main()
