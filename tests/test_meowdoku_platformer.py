"""The climber (``examples/meowdoku/platformer.py``): a climb of stacked FLOORS, each reachable from the one below.

* every floor is a row of bricks with one hole (as Ice Climber): every run of bricks is at least ``RUN_MIN`` long, so
  no step of the climb is an isolated brick;
* floors are spaced under what the bounce climbs and over what the frog and a brick need to stand between them;
* REACHABILITY is proved with the actual law: from every place a frog can stand on a floor (at rest, the worst
  case) the bounce -- the shared ``entity_motion.vertical_axis`` at ``JUMP`` -- with the sideways law
  ``entity_motion.damped_axis`` at its air speed carries the feet down onto the next floor's top;
* the same holds for the floors the law RECYCLES as the world scrolls (the real step law, run on the NumPy lane).
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
from meowdoku import entity_motion as em  # noqa: E402
from meowdoku import platformer as p  # noqa: E402

DT = 1.0 / 120.0


def floors_of(x, y, alive):
    """[(top, [present brick x...]), ...] from the floor lanes, lowest (largest y) first."""
    out = []
    for k in range(p.SLOTS):
        lanes = range(p.PLATFORMS[0] + k * p.CELLS, p.PLATFORMS[0] + (k + 1) * p.CELLS)
        out.append((float(y[lanes[0]]) - p.BRICK, [float(x[i]) for i in lanes if alive[i] > 0.0]))
    return sorted(out, key=lambda f: -f[0])


def runs(xs):
    xs, out, run = sorted(xs), [], 1
    for a, b in zip(xs, xs[1:]):
        if abs(b - a - 1.0) < 1e-6:
            run += 1
        else:
            out.append(run)
            run = 1
    out.append(run)
    return out


def bounce_lands(x0, axis, top, bricks, next_top, next_bricks, field_w):
    """From rest on a floor at ``x0`` the bounce launches (-JUMP) with the stick at ``axis``: do the feet come down onto
    the next floor's top over one of its bricks?  The laws are the game's own, stepped at 120 Hz."""
    y, vy, vx, x, t = top - p.PLAYER_RY, -p.JUMP, 0.0, x0, 0.0
    while t < 3.0:
        x, vx = em.damped_axis(x, vx, float(axis), -1.0, p.MARGIN, field_w - p.MARGIN, p.SPEED, p.TAU, p.REACH, DT)
        feet = y + p.PLAYER_RY
        y1, vy1, *_ = em.vertical_axis(y, vy, 1.0, 1.0, 0.0, 0.0, 0.0, DT, p.GRAVITY, 0.0, 1.0e9, p.FALL_MAX, 0.0, 1.0, 0.0, 0.0)
        if vy1 > 0.0 and feet <= next_top <= y1 + p.PLAYER_RY and any(abs(x - bx) < p.BRICK + p.FOOT for bx in next_bricks):
            return True
        y, vy, t = y1, vy1, t + DT
    return False


def assert_climb(floors, field_w):
    for (top, bricks), (next_top, next_bricks) in zip(floors, floors[1:]):
        spacing = top - next_top
        assert p.GAP_MIN - 1e-6 <= spacing <= p.GAP_MAX_LIMIT + 1e-6, spacing
        assert spacing < p.BOUNCE_APEX
        assert all(r >= p.RUN_MIN for r in runs(bricks)), runs(bricks)
        # every place a frog can stand (every brick centre and its edges) reaches the next floor with SOME stick
        for bx in bricks:
            for dx in (-0.4, 0.0, 0.4):
                if not any(abs(bx + dx - b) < p.BRICK + p.FOOT - 1e-9 for b in bricks):
                    continue
                assert any(bounce_lands(bx + dx, a, top, bricks, next_top, next_bricks, field_w) for a in (-1, 0, 1)), (bx + dx, top, next_top)


def test_the_bounce_climbs_less_than_it_can_and_the_frog_fits_between_floors():
    assert p.GAP_MAX_LIMIT < p.BOUNCE_APEX
    assert p.GAP_MIN >= 2.0 * p.BRICK + 2.0 * p.PLAYER_RY + 0.2 - 1e-9 - 0.2          # a brick's thickness and the frog's height
    assert abs(p.JUMP ** 2 / (2.0 * p.GRAVITY) - p.BOUNCE_APEX) < 1e-9


@pytest.mark.parametrize("width", [9.0, 9.5, 10.0])
def test_every_floor_of_a_new_climb_is_reachable_from_the_one_below(width):
    for seed in range(12):
        cols = p.initial_columns(width, seed)
        floors = floors_of(cols["x"], cols["y"], cols["alive"])
        assert len(floors) == p.SLOTS and len(floors[0][1]) == int(np.floor(width))              # the ground floor is whole
        assert_climb(floors, width)


def test_the_floors_the_law_recycles_stay_reachable_as_the_world_scrolls():
    game = p.Platformer(seed=5, aspect=0.6)
    game.session.control({})
    game.session.advance(1 / 60)
    state = game.session.state
    seen = 0
    for step in range(2400):
        state.y[0] = p.SCROLL_Y - 0.03                      # held at the scroll line: the world scrolls steadily
        state.vy[0] = 0.0
        game.session.control({"move_axis": 0.0})
        game.session.advance(1 / 60)
        if step % 150 == 149:
            width = float(game.session.column("field_w")[0])
            floors = floors_of(game.session.column("x"), game.session.column("y"), game.session.column("alive"))
            assert len(floors) == p.SLOTS
            assert_climb(floors, width)
            seen += 1
    assert seen == 16
    assert float(game.session.column("climbed")[0]) > 3.0 * p.HEIGHT                    # many floors went round


def test_the_player_is_the_chosen_character_and_the_choice_is_a_profile_setting():
    from meowdoku import characters, profiles
    assert "avatar" in profiles.SETTING_KEYS                           # persisted with the player (the profile), not the page
    for module_name, make in (("platformer", lambda: p.Platformer(seed=1, aspect=0.6)),):
        game = make()
        game.session.control({})
        game.session.advance(1 / 60)
        assert game.hero_mesh == "hero_frog" and game.state()["characters"]["chosen"] == "frog"
        game.set_hero("fox")
        state = game.state()
        player = next(e for e in state["entities"] if e["id"].startswith("0:"))
        assert player["mesh"] == "hero_fox" and state["characters"]["chosen"] == "fox"
        assert [t["character"] for t in state["pieces"]] == list(characters.CHARACTERS)
        assert [t["lit"] for t in state["pieces"]].count(True) == 1
        game.set_hero("no such character")                              # refused: the choice stays
        assert game.hero == "fox"
