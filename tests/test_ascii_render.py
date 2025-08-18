import numpy as np

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
