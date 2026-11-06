"""World Walk and the sphere half of the walkers substrate: the node graph, the flow fields, the kinds' choice
networks and the law.

The law is run through the NumPy product (the translatable lane); the native lane is checked against it by
examples/meowdoku/worldwalk_probe.py and the parity script, not here.
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

from meowdoku import walkers  # noqa: E402
from meowdoku import worldwalk as ww  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return ww.world()


@pytest.fixture(scope="module")
def played():
    game = ww.WorldWalk(seed=0)
    s = game.session
    s.control({})
    s.advance(1.0 / 60.0)
    s.state.spd[0] = 0.0
    for _ in range(1200):                      # 40 s
        s.advance(1.0 / 30.0)
    return game


def test_the_fields_reach_every_town_from_every_town(world):
    for field in world.fields[:3]:
        for node in world.site_nodes[:1] + [world.site_nodes[i] for i in world.towns]:
            assert field.dist[node] < walkers.SphereField.UNREACHABLE


def test_descent_reaches_the_goal_from_every_walkable_node(world):
    field = world.fields[ww.F_TOWN]
    reachable = np.flatnonzero(world.graph.passable & (field.dist < walkers.SphereField.UNREACHABLE))
    assert len(reachable) > 1000
    for start in reachable[::17]:
        at = int(start)
        for _ in range(400):
            if field.dist[at] == 0.0:
                break
            nxt = int(field.nxt[at])
            assert nxt != at, f"stuck at node {at}"
            assert field.dist[nxt] < field.dist[at]
            assert world.graph.edge(at, nxt) < np.inf
            at = nxt
        assert field.dist[at] == 0.0


def test_no_edge_a_walker_follows_cuts_a_corner_of_water(world):
    g = world.graph
    for a in range(0, len(g.nodes), 7):
        for b in g.adjacent[a]:
            if g.edge(a, b) == np.inf:
                continue
            for t in (0.25, 0.5, 0.75):
                p = g.nodes[a] * (1.0 - t) + g.nodes[b] * t
                assert g.passable[int(np.argmax(g.nodes @ p))]


def test_roads_are_cheap_to_walk(world):
    a, b = next(iter(world.road_edges))
    plain = walkers.SphereField.build(world.graph, [b])
    cheap = walkers.SphereField.build(world.graph, [b], discount=world.road_edges, factor=ww.ROAD_DISCOUNT)
    assert cheap.dist[a] < plain.dist[a]


def test_clearance_is_zero_on_water_and_grows_inland(world):
    clear = world.fields[ww.F_AVOID]
    assert (clear.dist[~world.graph.passable] == 0.0).all()
    assert clear.dist[world.graph.passable].max() > 3.0


def test_kinds_choose_their_motivations():
    settler, raider, beast, tower = (ww.KINDS[k] for k in (ww.SETTLER, ww.RAIDER, ww.BEAST, ww.TOWER))
    assert walkers.explain(settler, health=1.0, goal_far=0.5)[0] == "follow"            # a settler commutes by default
    assert walkers.explain(settler, health=1.0, goal_far=0.5, foe_prox=0.7)[0] == "flee"
    assert walkers.explain(settler, health=1.0, goal_far=0.5, busy=1.0)[0] == "rest"
    builder = ww.KINDS[ww.BUILDER]
    assert walkers.explain(builder, health=1.0, goal_far=0.5)[0] == "follow"
    assert walkers.explain(builder, health=1.0, goal_far=0.5, foe_prox=0.4)[0] == "follow"   # bolder than a settler
    assert walkers.explain(raider, goal_far=1.0, health=1.0)[0] == "follow"             # a raider goes for the towns
    assert walkers.explain(raider, goal_far=1.0, health=1.0, foe_prox=0.6)[0] == "seek"
    assert walkers.explain(raider, goal_far=1.0, health=1.0, foe_prox=0.9, foe_close=0.9)[0] == "attack"
    assert walkers.explain(raider, goal_far=1.0, foe_prox=0.6, health=0.2)[0] == "flee"
    assert walkers.explain(beast, health=1.0)[0] == "wander"
    assert walkers.explain(beast, health=1.0, foe_prox=0.7)[0] == "seek"
    assert walkers.explain(tower)[0] == "rest"
    assert walkers.explain(tower, health=1.0, foe_prox=0.1)[0] == "attack"


def test_the_law_runs_and_the_world_lives(played):
    s = played.session
    ent = slice(0, ww.ENT)
    for name in ("x", "y", "ux", "uy", "uz", "hx", "hy", "hz"):
        assert np.isfinite(s.column(name)).all(), name
    assert s.column("trips")[0] >= 3                       # settlers commuted
    assert s.column("built")[0] >= 1                       # a builder founded a light
    assert s.column("alive")[ww.L0:ww.ENT].sum() >= 1
    u = np.stack([s.column("ux"), s.column("uy"), s.column("uz")], axis=1)[ent]
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-6)          # bodies stay on the sphere


def test_walkers_stay_on_land_and_know_their_node(played, world):
    s = played.session
    node = s.column("node")[:ww.ENT].astype(int) - ww.MAP0
    u = np.stack([s.column("ux"), s.column("uy"), s.column("uz")], axis=1)[:ww.ENT]
    walkers_live = (s.column("alive")[:ww.ENT] > 0.0) & (s.column("kind")[:ww.ENT] >= ww.SETTLER) & (s.column("kind")[:ww.ENT] <= ww.BEAST)
    nearest = np.argmax(u @ world.nodes.T, axis=1)
    assert (world.tile[nearest][walkers_live] != ww.WATER).all()           # none stands in water
    assert (nearest == node)[walkers_live].all()                           # the climb tracks the true nearest node


def test_lights_are_founded_on_roads_apart_from_each_other(played, world):
    s = played.session
    lit = np.flatnonzero(s.column("alive")[ww.L0:ww.ENT] > 0.0) + ww.L0
    nodes = s.column("node")[lit].astype(int) - ww.MAP0
    assert (world.tile[nodes] == ww.ROAD).all()
    u = np.stack([s.column("ux"), s.column("uy"), s.column("uz")], axis=1)[lit]
    gaps = ww.RADIUS * np.arccos(np.clip(u @ u.T, -1.0, 1.0))
    gaps[np.diag_indices(len(lit))] = 99.0
    assert gaps.min() >= ww.BUILD_GAP * 0.9


def test_the_page_contract(played):
    st = played.state()
    assert st["scene"]["terrain"] is True and st["you"]["alive"]
    assert st["predict"]["adopt_axes"] == ["yaw", "spd", "bank"]
    assert st["predict"]["patch"]["radius"] == ww.RADIUS and "travel" in st["predict"]
    assert st["roads"] and st["globe"]["radius"] == ww.RADIUS
    assert all("mesh" in e and "x" in e and "y" in e for e in st["entities"])
    assert st["entities"][0]["id"].startswith("0:")                         # the wizard, at the patch's centre
