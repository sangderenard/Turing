"""Space invaders: the classic block, free fire, and the attack waves.

The law is run through the NumPy product (the translatable lane); the native lane is checked against it by
examples/meowdoku/invaders_parity.py, and the wave schedule by examples/meowdoku/invaders_probe.py.  The tests here
are the short, fast reads of the same claims.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.environ.setdefault("MEOWDOKU_LAW", "numpy")

from meowdoku import invaders as inv  # noqa: E402
from meowdoku import invaders_probe as probe  # noqa: E402


def test_level_curve_is_declared_data_and_only_gets_harder():
    rows = [dict(zip(inv.LEVEL_COLUMNS, row)) for row in inv.LEVELS]
    assert rows[0]["gap"] == 0.0 and rows[0]["rows"] == 5          # level 1: the classic block alone
    assert all(r["gap"] > 0.0 for r in rows[1:])                    # then waves mix in
    gaps = [r["gap"] for r in rows[1:]]
    assert gaps == sorted(gaps, reverse=True)                       # launched more often
    assert [r["rows"] for r in rows] == sorted((r["rows"] for r in rows), reverse=True)
    assert rows[-1]["groups"] == inv.WAVE_GROUPS and rows[-1]["rows"] == 1   # later levels are mostly waves
    assert max(r["speed"] for r in rows) <= inv.MAX_BULLET_SPEED


def test_level_one_is_classic_with_no_waves():
    r = probe.observe(0, 25.0)
    assert not r["launched"] and r["peak"]["ships"] == 0 and r["peak"]["bullets"] == 0
    assert r["fleet_moved"] > 0.5 and r["alive_fleet"] == inv.FLEET_ROWS * inv.FLEET_COLS


def test_a_formation_enters_fires_telegraphed_volleys_and_leaves_on_schedule():
    r = probe.observe(2, 22.0)                                      # level 3: fans at most, one formation
    assert r["launched"] and r["left"]
    step = probe.STEP
    for k, t0 in r["launched"].items():
        if k not in r["left"]:
            continue
        lag = ((k - inv.WAVE0) % inv.WAVE_SIZE) * inv.WAVE_LAG
        assert abs(r["left"][k] - (t0 + lag + inv.WAVE_ENTER + r["hold"][k] + inv.WAVE_EXIT)) <= 3 * step
        times = sorted({round(b, 3) for b in r["births"][k] if t0 <= b <= r["left"][k]})
        assert times, "a ship that holds fires its pattern"
        for t in times:
            on_clock = t0 + lag + inv.WAVE_ENTER + inv.VOLLEY_FIRST
            assert abs(((t - on_clock) / inv.VOLLEY_EVERY) - round((t - on_clock) / inv.VOLLEY_EVERY)) * inv.VOLLEY_EVERY <= 2.5 * step
            telegraphed = [c for c in r["tele"][k] if t - inv.TELEGRAPH - 2 * step <= c < t]
            assert len(telegraphed) >= 0.6 * inv.TELEGRAPH / step


@pytest.mark.parametrize("level", [4, 7])
def test_bullets_stay_within_the_simplicity_limits(level):
    r = probe.observe(level, 24.0)
    assert r["peak"]["bullets"] <= inv.WBULLETS
    assert r["peak"]["speed"] <= inv.MAX_BULLET_SPEED + 1e-6
    assert max(r["shots_at"].values()) <= inv.MAX_SHOTS_PER_EVENT
    assert r["peak"]["groups"] <= dict(zip(inv.LEVEL_COLUMNS, inv.LEVELS[level]))["groups"]


def test_freefire_still_sprays_and_the_classic_shot_still_fires():
    for freefire, lanes in ((False, (1, 2)), (True, inv.SPRAY_LANES)):
        game = inv.SpaceInvaders("t", freefire=freefire)
        col = game.session.column
        game.control({"inputs": [{"seq": 1, "dur": 1.0 / 30.0, "fire": 1.0, "steer_x": -1.0, "move_axis": 0.0}]})
        game.tick()
        for i in range(2, 12):
            game.control({"inputs": [{"seq": i, "dur": 1.0 / 30.0, "fire": 1.0, "steer_x": -1.0, "move_axis": 0.0}]})
            game.tick()
        shots = np.flatnonzero(np.array(col("alive"))[lanes[0]:lanes[1]] > 0.0)
        assert len(shots) >= 1
        if freefire:
            assert np.array(col("alive"))[1] == 0.0                  # lane 1 rests while the spray fires


def test_the_page_draws_waves_through_the_same_entity_records():
    game = inv.SpaceInvaders("t")
    game.control({"inputs": [{"seq": 1, "dur": 1.0 / 30.0}]})
    game.tick()
    game.session.column("wave")[...] = 5.0
    game.session.column("w_clock")[...] = 0.0
    for i in range(2, 200):
        game.control({"inputs": [{"seq": i, "dur": 1.0 / 30.0}]})
        game.tick()
        game.session.column("lives")[...] = inv.LIVES
    meshes = {e["mesh"] for e in game.entities()}
    assert {"star"} <= meshes and meshes & {"invader", "crab", "alien"}
    for e in game.entities():
        assert set(e) >= {"id", "x", "y", "vx", "vy", "mesh", "color", "size"}
    assert all(e["color"] in game.palette() for e in game.entities())
