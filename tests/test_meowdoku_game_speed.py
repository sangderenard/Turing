"""Game clock speed and world zoom (``examples/meowdoku/world_view.py``).

Speed scales how much law time a wall second lands; it is not the law.  So: the server's step count over a wall
interval scales with the speed, the dt the controller chose does not, and the same landed windows give a bit-identical
law at any speed.  Zoom is render-only for the law, but the view window the server sends from widens with it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.environ.setdefault("MEOWDOKU_LAW", "numpy")

import src.common.tensors.numpy_backend  # noqa: E402,F401
from meowdoku import entities as ent  # noqa: E402
from meowdoku import realtime_socket as rs  # noqa: E402
from meowdoku import scroller as sc  # noqa: E402
from meowdoku import sidescroller as ss  # noqa: E402
from meowdoku import world_view as wv  # noqa: E402


class Wall:
    """A fake ``time.perf_counter`` the test advances by hand (dyadic steps stay exact in floating point)."""

    def __init__(self, monkeypatch, start=1024.0):
        self.now = start
        monkeypatch.setattr(ent.time, "perf_counter", lambda: self.now)


def _side(wall, speed, seed=3):
    game = ss.SideScroller("t", seed=seed)
    game.set_speed(speed)
    game.control({"move_axis": 1.0})
    game.tick()                                  # round zero: instantiates the state
    return game


def _run(game, wall, gap, ticks):
    for _ in range(ticks):
        wall.now += gap
        game.control({"move_axis": 1.0})
        game.tick()


def test_the_default_speed_is_low_and_games_without_the_control_keep_wall_time(monkeypatch):
    wall = Wall(monkeypatch)
    game = ss.SideScroller("t", seed=1)
    assert game.speed == wv.SPEED_DEFAULT == 0.4
    assert game.state()["speed"]["default"] == 0.4 and (game.state()["speed"]["min"], game.state()["speed"]["max"]) == (0.05, 2.0)
    other = ent.EntityGame.__new__(ent.EntityGame)
    assert other.SPEED_CONTROL is False


def test_the_law_time_landed_over_a_wall_interval_scales_with_the_speed_and_dt_does_not(monkeypatch):
    wall = Wall(monkeypatch)
    landed, dts, calls = {}, {}, {}
    for speed in (0.125, 0.25, 0.5, 1.0, 2.0):
        game = _side(wall, speed)
        c0 = float(game.session.column("clock")[0])
        n0 = game.session.calls()
        _run(game, wall, 0.5, 4)                            # two wall seconds, in half-second ticks
        landed[speed] = float(game.session.column("clock")[0]) - c0
        dts[speed] = game.session.dt
        calls[speed] = game.session.calls() - n0
    for speed, seconds in landed.items():
        assert seconds == pytest.approx(2.0 * speed, rel=1e-9), landed
    # the step count (law calls) follows the speed; the step itself does not: every speed chose the same dt
    assert calls[2.0] > calls[1.0] > calls[0.5] > calls[0.25] > calls[0.125], calls
    assert calls[2.0] / calls[0.125] == pytest.approx(16.0, rel=0.35), calls
    assert len(set(round(v, 12) for v in dts.values())) == 1, dts


def test_the_law_is_bit_identical_at_any_speed_for_the_same_landed_windows(monkeypatch):
    wall = Wall(monkeypatch)
    results = []
    for speed, gap in ((0.25, 0.5), (0.5, 0.25), (1.0, 0.125), (2.0, 0.0625)):       # every tick lands 1/8 s of play
        game = _side(wall, speed)
        _run(game, wall, gap, 24)
        columns = {name: np.array(game.session.column(name)) for name in game.session.columns}
        results.append((game.session.dt, columns))
    base_dt, base = results[0]
    for dt, columns in results[1:]:
        assert dt == base_dt
        for name, values in base.items():
            assert np.array_equal(values, columns[name], equal_nan=True), name


def test_a_speed_change_keeps_the_virtual_wall_continuous_and_law_wall_tracks_the_law_clock(monkeypatch):
    wall = Wall(monkeypatch)
    game = _side(wall, 0.5)
    offsets = []
    for speed in (0.5, 0.5, 0.125, 0.125, 2.0, 2.0, 0.25):
        for _ in range(8):
            wall.now += 0.125
            game.control({"move_axis": 1.0, "speed": speed})
            game.tick()
            offsets.append(game.law_wall - float(game.session.column("clock")[0]))
    # (law_wall - clock) is the virtual wall instant of law time zero: constant across speed changes
    assert max(offsets) - min(offsets) < 1e-9, (min(offsets), max(offsets))
    # and the virtual wall is continuous at a change
    before = game.virtual(wall.now)
    game.set_speed(1.0, wall.now)
    assert game.virtual(wall.now) == pytest.approx(before, abs=1e-12) and game.speed_gen == 5


def test_the_snapshot_carries_the_speed_in_force_and_the_exchange_in_the_virtual_wall(monkeypatch):
    wall = Wall(monkeypatch)
    game = _side(wall, 0.25)
    wall.now += 0.5
    game.control({})
    game.tick()
    encoder = rs.SnapshotEncoder()
    binary, _meta = encoder.encode(game, game.state(), echo=5.0, received=game.virtual(wall.now))
    assert binary[0] == rs.PROTOCOL == 14
    state = game.state()
    assert state["speed"]["value"] == 0.25 and state["speed"]["gen"] == game.speed_gen


def test_the_sync_hint_is_wall_time_so_it_stretches_as_the_clock_slows(monkeypatch):
    wall = Wall(monkeypatch)
    fast, slow = _side(wall, 1.0), _side(wall, 0.125)
    for game in (fast, slow):
        col = game.session.column
        col("alive")[ss.WALKER_LANES[0]] = 1.0
        col("x")[ss.WALKER_LANES[0]] = col("x")[0] + 3.0
        col("y")[ss.WALKER_LANES[0]] = col("y")[0]
        col("vx")[ss.WALKER_LANES[0]] = -1.0
    assert slow.sync_hint() >= fast.sync_hint()
    assert slow.sync_hint() <= slow.SYNC_MAX


# ------------------------------------------------------------------------------------------------ the zoom
def test_the_zoom_floor_comes_from_the_world_the_game_supplies():
    assert wv.zoom_floor(9.0, 16.0, None) == 1.0                          # nothing beyond the field: zoom in only
    reach = {"left": 2.0, "right": 13.0, "up": 20.0, "down": 0.0}
    floor = wv.zoom_floor(9.0, 10.0, reach)
    assert floor == pytest.approx(9.0 / 24.0)
    assert wv.clamp_zoom(0.01, 9.0, 10.0, reach) == pytest.approx(floor) and wv.clamp_zoom(99, 9.0, 10.0, reach) == wv.ZOOM_MAX
    assert wv.clamp_zoom(float("nan"), 9.0, 10.0, reach) == 1.0


def test_the_view_window_widens_as_the_zoom_goes_out_and_slides_inside_the_world():
    reach = {"left": 2.0, "right": 13.0, "up": 20.0, "down": 0.0}
    one = wv.view_window(9.0, 10.0, 1.0, reach)
    assert (one["x0"], one["x1"], one["y0"], one["y1"]) == (0.0, 9.0, 0.0, 10.0)
    widths = [wv.view_window(9.0, 10.0, z, reach) for z in (1.0, 0.8, 0.6, 0.4, 0.375)]
    assert all(b["x1"] - b["x0"] > a["x1"] - a["x0"] for a, b in zip(widths, widths[1:]))
    for w in widths:
        assert w["x0"] >= -2.0 - 1e-9 and w["x1"] <= 22.0 + 1e-9 and w["y1"] <= 10.0 + 1e-9 and w["y0"] >= -20.0 - 1e-9
    zin = wv.view_window(9.0, 10.0, 2.0, reach)
    assert zin["x1"] - zin["x0"] == pytest.approx(4.5) and zin["x0"] == pytest.approx(2.25)


def test_zooming_out_widens_what_the_side_scroller_publishes_and_the_game_clamps_it(monkeypatch):
    wall = Wall(monkeypatch)
    game = _side(wall, 1.0)
    game.control({"zoom": 0.5})
    z = game.state()["zoom"]
    assert z["value"] == 0.5 and z["min"] == pytest.approx(9.0 / (9.0 + 1.6 + 5.0), rel=0.05) or z["min"] < 0.6
    assert z["window"]["x1"] - z["window"]["x0"] == pytest.approx(game.width / 0.5)
    game.control({"zoom": 0.001})
    assert game.state()["zoom"]["value"] == pytest.approx(game.state()["zoom"]["min"])


def test_bullet_hell_sends_more_bodies_when_zoomed_out_and_the_pilots_observation_is_unchanged(monkeypatch):
    wall = Wall(monkeypatch)
    game = sc.BulletScroller("t")
    game.control({"speed": 1.0})
    game.tick()
    # put bodies on a ring of the patch, outside the field at zoom 1
    s = game.session
    for _ in range(40):
        wall.now += 0.125
        game.control({})
        game.tick()
    count = {}
    for zoom in (1.0, 0.8, game.state()["zoom"]["min"]):
        game.control({"zoom": zoom})
        count[zoom] = len(game.entities())
    values = list(count.values())
    assert values[0] <= values[1] <= values[2], count
    game.control({"zoom": 1.0})
    window = game.view_window()
    assert (window["x0"], window["x1"]) == (0.0, game.width)
    # the zoom is not an input of the law or of the pilot's observation: the cull radius the pilot reads is a constant
    assert sc.PATCH_R == 15.0
    state = game.state()
    assert "zoom" not in game.CONTROLS and state["zoom"]["min"] < 1.0
    # and the zoom never changed a law column
    before = {name: np.array(game.session.column(name)) for name in game.session.columns}
    game.control({"zoom": 0.6})
    assert all(np.array_equal(before[name], game.session.column(name)) for name in before)
