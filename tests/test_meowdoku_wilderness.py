"""The wilderness as a walker meets it: leaving a map, walking the sphere, and the local patch admitted as the
standard studio scene by the incumbent mesh environment."""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import wilderness as W  # noqa: E402
from meowdoku.studio_mesh_environment import StudioMeshEnvironment  # noqa: E402
from src.compiler.abstract_ui_wilderness import cosine, position  # noqa: E402

MAP = "world:repo:" + "c" * 40 + "/dir:."


def test_leaving_a_map_starts_at_its_own_place_with_a_grace_period():
    walker = W.leave_map(MAP, now=100.0)
    assert walker.point == position(MAP)
    assert math.isclose(W._dot(walker.east, walker.north), 0.0, abs_tol=1e-12)
    assert all(math.isclose(W._dot(walker.point, v), 0.0, abs_tol=1e-12) for v in (walker.east, walker.north))
    assert walker.can_return(100.0 + W.GRACE_S) and not walker.can_return(100.0 + W.GRACE_S + 0.1)


def test_walking_is_flat_locally_and_the_ground_is_the_same_every_visit():
    walker = W.leave_map(MAP, now=0.0)
    ahead = W.walk(walker, 30.0, 40.0)
    # 50 m of ground is 50 m of arc on the sphere
    assert math.isclose(W.arc(walker.point, ahead.point) * walker.radius_m, 50.0, rel_tol=1e-6)
    back = W.walk(ahead, -30.0, -40.0)
    assert W.arc(walker.point, back.point) * walker.radius_m < 1e-3
    assert W.ground(W.place(walker, 30.0, 40.0)) == W.ground(ahead.point)
    assert all(math.isclose(W._dot(ahead.point, v), 0.0, abs_tol=1e-9) for v in (ahead.east, ahead.north))


def test_the_patch_is_admitted_by_the_studio_mesh_environment():
    walker = W.leave_map(MAP, now=0.0)
    packet = W.patch(walker, grid=12)
    environment = StudioMeshEnvironment(packet)        # the real admission: one conforming open sheet
    assert len(environment.parts) == 1 and not environment.parts[0].closed
    assert len(environment.parts[0].triangles) == 2 * 12 * 12
    heights = [v[1] for v in packet["parts"][0]["vertices"]]
    low, high = packet["room"]["low"][1], packet["room"]["high"][1]
    assert low < min(heights) and high > max(heights)
    assert abs(heights[len(heights) // 2]) < 1e-6        # the walker stands on their own ground
    assert packet["wilderness"]["solid_m"] == W.SOLID_M and packet["wilderness"]["view_m"] == W.VIEW_M


def test_maps_are_met_only_by_coincidence_of_place():
    walker = W.leave_map(MAP, now=0.0)
    beside = W.walk(walker, 300.0, 0.0).point
    far = position("world:repo:" + "d" * 40 + "/dir:.")
    met = W.patch(walker, grid=4, maps=(("beside", beside), ("far", far), (MAP, walker.point)))["wilderness"]["encounters"]
    assert [m["identity"] for m in met] == ["beside"]
    assert math.isclose(met[0]["distance_m"], 300.0, rel_tol=1e-6) and met[0]["direction"][0] > 0
    with pytest.raises(ValueError):
        W.patch(walker, solid_m=10.0, view_m=5.0)
