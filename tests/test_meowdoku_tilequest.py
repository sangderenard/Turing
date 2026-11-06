"""Tile Quest and the walkers substrate: the map, the flow fields, the kinds' choice networks and the law.

The law is run through the NumPy product (the translatable lane); the native
lane is checked against it by examples/meowdoku/tilequest_probe.py and the
parity script, not here.
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

from meowdoku import tilequest as tq  # noqa: E402
from meowdoku import walkers  # noqa: E402


@pytest.mark.parametrize("seed", range(8))
def test_every_goal_is_reachable_from_the_start(seed):
    m = tq.make_map(seed)
    passable = m.passable()
    sx, sy = int(m.start[0]), int(m.start[1])
    assert passable[sy, sx]
    for field in m.fields[:4]:
        assert field.dist[sy, sx] < walkers.FlowField.UNREACHABLE
    for cx, cy in m.homes + m.works + m.lairs:
        assert passable[cy, cx]
    for tx, ty in m.towers:
        assert passable[int(ty), int(tx)]


def test_descent_direction_reaches_the_goal_from_every_walkable_cell():
    m = tq.make_map(3)
    field = m.fields[tq.F_TOWN]
    reachable = np.argwhere(m.passable() & (field.dist < walkers.FlowField.UNREACHABLE))
    for cy, cx in reachable[::7]:
        x, y = cx, cy
        for _ in range(80):
            if field.dist[y, x] == 0.0:
                break
            dx, dy = field.dx[y, x], field.dy[y, x]
            assert (dx, dy) != (0.0, 0.0), f"stuck at {(x, y)}"
            x, y = x + int(round(dx)), y + int(round(dy))
        assert field.dist[y, x] == 0.0


def test_two_equidistant_goals_do_not_cancel():
    ground = np.ones((3, 5), dtype=bool)
    field = walkers.FlowField.build(ground, [(0, 1), (4, 1)])
    assert (field.dx[1, 2], field.dy[1, 2]) != (0.0, 0.0)


def test_kinds_choose_their_motivations():
    v, wolf, tower = tq.KINDS[tq.VILLAGER], tq.KINDS[tq.WOLF], tq.KINDS[tq.TOWER]
    assert walkers.explain(v, health=1.0, goal_far=0.5)[0] == "follow"                  # a villager commutes by default
    assert walkers.explain(v, health=1.0, goal_far=0.5, foe_prox=0.7)[0] == "flee"
    assert walkers.explain(v, health=1.0, goal_far=0.5, busy=1.0)[0] == "rest"
    assert walkers.explain(v, health=1.0, goal_far=0.5, need=0.95)[0] == "rest"
    assert walkers.explain(wolf, goal_far=1.0, health=1.0)[0] == "follow"
    assert walkers.explain(wolf, goal_far=1.0, health=1.0, foe_prox=0.6)[0] == "seek"
    assert walkers.explain(wolf, goal_far=1.0, health=1.0, foe_prox=0.9, foe_close=0.9)[0] == "attack"
    assert walkers.explain(wolf, goal_far=1.0, foe_prox=0.6, health=0.2)[0] == "flee"
    assert walkers.explain(wolf, health=0.2, refuge_near=0.9)[0] == "rest"
    assert walkers.explain(wolf, health=1.0, refuge_near=0.9, goal_far=1.0)[0] == "follow"   # a healthy wolf at its lair still goes out
    assert walkers.explain(tower)[0] == "rest"
    assert walkers.explain(tower, health=1.0, foe_prox=0.1)[0] == "attack"


def test_the_law_runs_and_the_villagers_commute():
    game = tq.TileQuest(seed=3)
    s = game.session
    s.control({})
    s.advance(1.0 / 60.0)
    s.state.x[0], s.state.y[0] = 4.5, 4.5
    for _ in range(900):                       # 30 s
        s.advance(1.0 / 30.0)
    assert s.column("trips")[0] >= 5
    assert s.column("kills")[0] >= 1           # the towers (the player never swung)
    assert np.isfinite(s.column("x")).all() and np.isfinite(s.column("y")).all()
    ent = slice(0, tq.ENT)
    open_ground = game.map.passable()
    for lane in np.flatnonzero(s.column("alive")[ent] > 0.0):
        if int(s.column("kind")[lane]) not in (tq.ARROW,):
            cx, cy = int(s.column("x")[lane]), int(s.column("y")[lane])
            assert open_ground[cy, cx], f"lane {lane} stands in a wall at {(cx, cy)}"
    st = game.state()
    assert st["predict"]["adopt_axes"] == ["x", "y"] and st["you"]["alive"]
