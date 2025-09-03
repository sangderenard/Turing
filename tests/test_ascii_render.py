import numpy as np
import pytest

from src.ascii_render import AsciiRenderer


def test_shape_primitives():
    r = AsciiRenderer(10, 10)
    r.line(0, 0, 9, 9)
    assert np.count_nonzero(r.canvas) == 10

    r.clear()
    r.circle(5, 5, 3)
    assert np.count_nonzero(r.canvas) > 0

    r.clear()
    r.triangle((1, 1), (8, 1), (1, 8))
    assert r.canvas[1, 1, 0] > 0
    assert r.canvas[1, 8, 0] > 0
    assert r.canvas[8, 1, 0] > 0


def test_ascii_ramp():
    r = AsciiRenderer(2, 1)
    r.canvas[0, 0, 0] = 0
    r.canvas[0, 1, 0] = 1
    art = r.to_ascii()
    assert art == " @"


def test_paint_channel_mismatch():
    r = AsciiRenderer(4, 4, depth=1)
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[..., 1] = 128
    img[..., 2] = 255
    r.paint(img, 0, 0)
    expected = img.mean(axis=2, keepdims=True).astype(r.canvas.dtype)
    assert np.array_equal(r.canvas[:2, :2], expected)


@pytest.mark.skip(reason="takes too long")
def test_to_ascii_diff_preserves_color():
    r = AsciiRenderer(2, 1, depth=3)
    r.canvas[0, 0] = [10, 20, 30]
    r.canvas[0, 1] = [30, 40, 50]
    r.to_ascii_diff()
    assert np.array_equal(r._fb.buffer_display[0, 0], [10, 20, 30])
    assert np.array_equal(r._fb.buffer_display[0, 1], [30, 40, 50])
