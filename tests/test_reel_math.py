import math

from src.visualizations.reel_math import tape_radius, tangent_points


def test_tape_radius_linear():
    assert tape_radius(30, 80, 0, 100) == 30
    assert tape_radius(30, 80, 100, 100) == 80
    assert tape_radius(30, 80, 50, 100) == 55


def test_tangent_points_distance():
    cx, cy, r = 0.0, 0.0, 1.0
    px, py = 3.0, 4.0
    t1, t2 = tangent_points(cx, cy, r, px, py)
    for tx, ty in (t1, t2):
        dist = math.hypot(tx - cx, ty - cy)
        assert math.isclose(dist, r, rel_tol=1e-6)
