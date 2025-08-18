"""Minimal demo for ASCII diff rendering."""
from __future__ import annotations

import time
from PIL import Image
import numpy as np
import sys

from .frame_buffer import PixelFrameBuffer
from .draw import (
    draw_diff,
    get_changed_subunits,
    default_subunit_batch_to_chars,
)
from .console import full_clear_and_reset_cursor, reset_cursor_to_top


def load_pixels(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def main(image_path: str) -> None:
    frame_a = load_pixels(image_path)
    frame_b = np.flip(frame_a, axis=1)  # simple transformation

    char_h, char_w = 8, 4

    old_frame = np.zeros_like(frame_a)
    full_clear_and_reset_cursor()

    for frame in [frame_a, frame_b]:
        changed = get_changed_subunits(old_frame, frame, char_h, char_w)
        draw_diff(
            changed,
            char_cell_pixel_height=char_h,
            char_cell_pixel_width=char_w,
            subunit_to_char_kernel=default_subunit_batch_to_chars,
        )
        sys.stdout.flush()
        time.sleep(1)
        old_frame = frame
        reset_cursor_to_top()


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "timesync/analogback.png"
    main(img_path)

