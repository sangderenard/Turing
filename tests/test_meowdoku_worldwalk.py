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
            assert world.edge[at][nxt] < np.inf                       # (the walk-time cost: water is slow, not barred)
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
    wet = (world.mat[nearest][walkers_live] == ww.WATER).sum()
    assert wet <= 0.1 * walkers_live.sum()                                  # (they may swim, but paths avoid water: a stray at most)
    assert (nearest == node)[walkers_live].all()                           # the climb tracks the true nearest node


def test_lights_are_founded_on_roads_apart_from_each_other(played, world):
    s = played.session
    lit = np.flatnonzero(s.column("alive")[ww.L0:ww.ENT] > 0.0) + ww.L0
    nodes = s.column("node")[lit].astype(int) - ww.MAP0
    assert (world.mat[nodes] == ww.ROAD).all()
    u = np.stack([s.column("ux"), s.column("uy"), s.column("uz")], axis=1)[lit]
    gaps = ww.RADIUS * np.arccos(np.clip(u @ u.T, -1.0, 1.0))
    gaps[np.diag_indices(len(lit))] = 99.0
    assert gaps.min() >= ww.BUILD_GAP * 0.9


def test_the_page_contract(played):
    st = played.state()
    assert st["scene"]["terrain"] is True and st["you"]["alive"]
    assert st["predict"]["adopt_axes"] == ["yaw", "spd"]
    assert st["predict"]["patch"]["radius"] == ww.RADIUS and "travel" in st["predict"]
    assert st["roads"] and st["globe"]["radius"] == ww.RADIUS
    assert all("mesh" in e and "x" in e and "y" in e for e in st["entities"])
    assert st["entities"][0]["id"].startswith("0:")                         # the wizard, at the patch's centre


# ====================================================================================================
# The wizard walks (no ship, no throttle): surface materials, click paths, free walking, built tiles.
# ====================================================================================================
from meowdoku import entity_motion as em  # noqa: E402

STEP = 1.0 / 30.0


def fresh(seed=0):
    game = ww.WorldWalk(seed=seed)
    game.session.control({})
    game.session.advance(1.0 / 60.0)             # round zero instantiates the state
    return game


def put(game, node, lane=0, toward=None, still=True):
    """A lane at a node, heading toward a node (or any tangent) and standing still."""
    w, st = ww.world(), game.session.state
    p = w.nodes[node]
    g = w.nodes[toward] if toward is not None else np.array([st.hx[lane], st.hy[lane], st.hz[lane]]) + 0.3
    h = g - (g @ p) * p
    h /= np.linalg.norm(h) + 1e-12
    for name, v in zip(("ux", "uy", "uz"), p):
        getattr(st, name)[lane] = v
    for name, v in zip(("hx", "hy", "hz"), h):
        getattr(st, name)[lane] = v
    st.node[lane] = ww.MAP0 + node
    if still:
        st.spd[lane] = 0.0


def run(game, seconds, controls=None):
    s = game.session
    s.control({"move_axis": 0.0, "move_axis_y": 0.0, "fire": 0.0, **(controls or {})})      # (the page sends every control each tick)
    for _ in range(int(round(seconds / STEP))):
        s.advance(STEP)


def arc(game, a, b=None):
    s, w = game.session, ww.world()
    pa = np.array([s.column("ux")[a], s.column("uy")[a], s.column("uz")[a]])
    pb = w.nodes[b] if b is not None else None
    return ww.RADIUS * float(np.arccos(np.clip(pa @ pb, -1.0, 1.0)))


def where_is(game, lane=0):
    s = game.session
    return np.array([s.column("ux")[lane], s.column("uy")[lane], s.column("uz")[lane]])


FORWARD = {"move_axis_y": -1.0}            # up is forward


def test_the_surface_has_a_material_a_node_and_the_speeds_are_the_table(world):
    mats = {int(m) for m in np.unique(world.mat)}
    assert {ww.GRASS, ww.ROAD, ww.SAND, ww.ROCK, ww.WATER} <= mats
    assert (world.mat[sorted(world.road_nodes)] == ww.ROAD).all()           # ROAD is its own material, from roads.plan
    assert (world.mat[~world.dry] == ww.WATER).all()
    assert np.array_equal(world.slope, np.round(world.slope * 100) / 100)    # (hundredths, k/100: the page's bytes/100 are the same numbers)
    top = {name: float(em.walk_speed(ww.WALK_BASE, np.asarray([float(i)]), np.asarray([0.0]))[0][0])
           for name, (i, _f) in em.WALK_MATERIALS.items()}
    assert top["road"] > top["grass"] > top["sand"] > top["rock"] > top["water"] > 0.0 and top["wall"] == 0.0
    assert top["water"] <= 0.06 * top["grass"]                              # a swim: all but blocked, not blocked
    assert em.walk_speed(1.0, np.asarray([0.0]), np.asarray([0.3]))[0][0] < em.walk_speed(1.0, np.asarray([0.0]), np.asarray([0.0]))[0][0]


@pytest.mark.parametrize("material", [ww.GRASS, ww.ROAD, ww.SAND, ww.ROCK, ww.WATER])
def test_the_wizard_walks_at_the_surface_speed(world, material):
    """Held forward on a node of each material, the wizard's pace settles at walk_speed(base, material, slope)."""
    pool = np.flatnonzero((world.mat == material) & (world.slope <= 0.1))
    node = int(pool[len(pool) // 2])
    game = fresh()
    s = game.session
    for _ in range(48):                                      # (re-seated each step: the same node throughout)
        put(game, node, toward=int(world.graph.adjacent[node][0]), still=False)
        s.control(FORWARD)
        s.advance(1.0 / 60.0)
    want = float(em.walk_speed(ww.WALK_BASE, np.asarray([material]), np.asarray([world.slope[node]]))[0][0])
    assert s.column("spd")[0] == pytest.approx(want, rel=0.03)


def test_every_kind_walks_by_the_same_law(world):
    """A raider in the water crawls (its pace is the law's, a twentieth of grass), the same raider on grass strides."""
    game = fresh()
    s = game.session
    lane = ww.R0
    shore = next(n for n in np.flatnonzero(world.mat == ww.WATER)
                 if all(world.mat[b] == ww.WATER for b in world.graph.adjacent[n]))
    grass = int(np.flatnonzero((world.mat == ww.GRASS) & (world.slope <= 0.05))[5])

    def pace(node):
        top = 0.0
        for i in range(60):
            put(game, node, lane, still=False)
            s.column("alive")[lane] = 1.0
            s.column("mot")[lane] = 0.0
            s.column("think")[lane] = 9.0                            # (keep its choice: wander)
            s.advance(1.0 / 30.0)
            if i >= 40:                                                  # (settled: the pace eases to the law's)
                top = max(top, float(s.column("gspd")[lane]))
        return top

    swim, walk = pace(shore), pace(grass)
    base = walkers.PACE[walkers.WANDER] * ww.KINDS[ww.RAIDER].speed
    law = lambda node: float(em.walk_speed(base, np.asarray([world.mat[node]]), np.asarray([world.slope[node]]))[0][0])
    assert swim == pytest.approx(law(shore), rel=0.05) and walk == pytest.approx(law(grass), rel=0.15)
    assert swim < 0.07 * walk


def test_free_walking_stops_when_released(world):
    node = int(np.flatnonzero((world.mat == ww.GRASS) & (world.slope <= 0.05))[10])
    game = fresh()
    put(game, node, toward=int(world.graph.adjacent[node][0]))
    start = where_is(game)
    run(game, 1.5, FORWARD)
    walked = ww.RADIUS * float(np.arccos(np.clip(start @ where_is(game), -1, 1)))
    assert walked > 1.5 and game.session.column("spd")[0] > 1.0
    run(game, 1.2, {})                                              # let go: it stops (no held throttle, no always-forward)
    assert abs(game.session.column("spd")[0]) < 0.02
    rest = where_is(game)
    run(game, 1.0, {})
    assert ww.RADIUS * float(np.arccos(np.clip(rest @ where_is(game), -1, 1))) < 0.02
    # and it never moves by itself from the first tick
    game = fresh()
    start = where_is(game)
    run(game, 2.0, {})
    assert ww.RADIUS * float(np.arccos(np.clip(start @ where_is(game), -1, 1))) < 0.02


def test_turning_and_backing_up(world):
    node = int(np.flatnonzero((world.mat == ww.GRASS) & (world.slope <= 0.05))[10])
    game = fresh()
    put(game, node, toward=int(world.graph.adjacent[node][0]))
    h0 = np.array([game.session.column(n)[0] for n in ("hx", "hy", "hz")])
    run(game, 0.5, {"move_axis": 1.0})                              # right: turns without walking
    h1 = np.array([game.session.column(n)[0] for n in ("hx", "hy", "hz")])
    assert float(h0 @ h1) < 0.9 and abs(game.session.column("spd")[0]) < 0.02
    start = where_is(game)
    run(game, 1.0, {"move_axis_y": 1.0})                            # down backs up, at half the pace
    assert game.session.column("spd")[0] < -0.5
    assert float((where_is(game) - start) @ h1) < 0.0


def _towns(world):
    return [world.site_nodes[i] for i in world.towns]


def _nearest_town_pair(world, ground):
    towns = _towns(world)
    best = None
    for i, a in enumerate(towns):
        for b in towns[i + 1:]:
            path = ground.path(a, b)
            if path and (best is None or len(path) < len(best)):
                best = path
    return best


def test_the_click_path_prefers_roads_and_the_wizard_walks_it(world):
    game = fresh()
    path = _nearest_town_pair(world, game.ground)
    assert path and (world.mat[path] == ww.ROAD).mean() > 0.5       # roads: the cheapest way is along them
    assert not (world.mat[path] == ww.WATER).any()
    put(game, path[0], toward=path[1])
    got = game.goto(node=path[-1])
    assert got == path
    s = game.session
    assert s.column("route")[0] == 1.0 and s.column("rp_len")[0] == len(path) - 1
    assert (np.abs(np.stack([s.column("rp_x"), s.column("rp_y"), s.column("rp_z")], axis=1)[ww.MAP0:ww.MAP0 + len(path) - 1]
                   - world.nodes[path[1:]]) < 1e-12).all()
    seconds = sum(game.ground.cost(a, b) for a, b in zip(path[:-1], path[1:])) / ww.WALK_BASE
    run(game, seconds * 1.6 + 3.0)
    assert s.column("route")[0] == 0.0                                # arrived: the path is done
    assert arc(game, 0, path[-1]) < 1.0
    run(game, 1.5)
    assert abs(s.column("spd")[0]) < 0.02                             # and it stands there


def test_free_input_cancels_the_click_path(world):
    game = fresh()
    path = _nearest_town_pair(world, game.ground)
    put(game, path[0], toward=path[1])
    game.goto(node=path[-1])
    run(game, 1.0, {})
    assert game.session.column("route")[0] == 1.0 and game.session.column("spd")[0] > 1.0
    run(game, 0.2, {"move_axis": 1.0})
    assert game.session.column("route")[0] == 0.0
    run(game, 1.0, {})
    assert abs(game.session.column("spd")[0]) < 0.02              # cancelled means standing, not walking on


def test_water_is_swum_slowly_or_walked_around(world):
    ground = world.ground()
    rng = np.random.default_rng(4)
    main = np.flatnonzero((world.comp == world.main) & (world.mat != ww.ROAD))
    wet = 0
    for _ in range(40):                                             # across the one land: water only if it saves a lot
        a, b = (int(x) for x in rng.choice(main, 2, replace=False))
        path = ground.path(a, b)
        assert path is not None
        wet += int((world.mat[path] == ww.WATER).any())
    assert wet <= 3
    island = int(np.flatnonzero((world.comp >= 0) & (world.comp != world.main))[0])
    start = _towns(world)[0]
    path = ground.path(start, island)
    assert path is not None and (world.mat[path] == ww.WATER).any()   # the only way is a swim
    wet_edges = [(a, b) for a, b in zip(path[:-1], path[1:]) if ww.WATER in (world.mat[a], world.mat[b])]
    dry_edges = [(a, b) for a, b in zip(path[:-1], path[1:]) if ww.WATER not in (world.mat[a], world.mat[b])]
    assert wet_edges and dry_edges
    mean = lambda edges: sum(ground.cost(a, b) for a, b in edges) / len(edges)
    assert mean(wet_edges) > 5.0 * mean(dry_edges)                  # a wet edge costs many dry ones: it swims only because it must
    game = fresh()
    put(game, start, toward=path[1])
    game.goto(node=island)
    run(game, 3.0)
    assert game.session.column("route")[0] == 1.0                    # (no ship's leaps: it is on its way, on foot)


def _buildable_beside_road(world, ground):
    return [n for n in range(len(world.nodes)) if ground.buildable(n) is None]


def test_a_tile_is_placed_only_beside_a_road_and_oriented_along_it(world):
    game = fresh()
    ground = game.ground
    ok = _buildable_beside_road(world, ground)
    assert len(ok) > 50
    far = next(n for n in range(len(world.nodes)) if world.mat[n] == ww.GRASS and n not in world.road_nodes
               and not any(b in world.road_nodes for b in world.graph.adjacent[n]))
    with pytest.raises(ValueError, match="not beside a road"):
        game.place("house", node=far)                                # not road-adjacent: refused
    road = next(iter(world.road_nodes))
    with pytest.raises(ValueError, match="beside a road, not on it"):
        game.place("house", node=road)
    water = int(np.flatnonzero(world.mat == ww.WATER)[0])
    with pytest.raises(ValueError, match="water"):
        game.place("market", node=water)
    node = ok[0]
    lane = game.place("house", node=node)                           # road-adjacent: accepted
    c = game.session.column
    assert c("alive")[lane] == 1.0 and c("kind")[lane] == ww.HOUSE and int(c("node")[lane]) == ww.MAP0 + node
    with pytest.raises(ValueError, match="something stands there"):
        game.place("house", node=node)
    # oriented along the road: its heading is the road's tangent there, within the road's own turn
    h = np.array([c("hx")[lane], c("hy")[lane], c("hz")[lane]])
    beside = [b for b in world.graph.adjacent[node] if b in world.road_nodes]
    along = [np.abs(h @ (world.nodes[b2] - world.nodes[b1]) / np.linalg.norm(world.nodes[b2] - world.nodes[b1]))
             for b in beside for b1 in sorted(world.road_links[b])[:1] for b2 in sorted(world.road_links[b])[-1:] if b1 != b2]
    assert not along or max(along) > 0.5
    assert abs(np.linalg.norm(h) - 1.0) < 1e-9 and abs(h @ world.nodes[node]) < 1e-9
    # a tap snaps to the road: the nearest buildable node, within reach
    p = world.nodes[ok[40]]
    snapped = game.place("market", point=p)
    assert int(c("node")[snapped]) - ww.MAP0 in ok
    with pytest.raises(ValueError, match="cannot build"):
        game.place("wall", point=world.nodes[far])


def test_a_wall_bars_its_node_and_walkers_reroute(world):
    game = fresh()
    ground = game.ground
    rng = np.random.default_rng(2)
    main = np.flatnonzero(world.comp == world.main)
    pick = None
    for _ in range(400):
        a, b = (int(x) for x in rng.choice(main, 2, replace=False))
        path = ground.path(a, b)
        if path and len(path) > 8:
            inner = [n for n in path[2:-2] if ground.buildable(n) is None]
            if inner:
                pick = (a, b, path, inner[0])
                break
    assert pick, "no pair with a buildable node on its path"
    a, b, path, x = pick
    before = sum(ground.cost(p, q) for p, q in zip(path[:-1], path[1:]))
    put(game, a, toward=path[1])
    game.goto(node=b)
    assert game.route["nodes"] == path
    lane = game.place("wall", node=x)                                # a wall on the path: it is laid again around it
    assert game.ground.mat[x] == ww.WALL and game.session.column("mat")[ww.MAP0 + x] == ww.WALL
    assert x not in game.route["nodes"]
    around = game.route["nodes"]
    assert around[0] == a and around[-1] == b
    assert sum(game.ground.cost(p, q) for p, q in zip(around[:-1], around[1:])) >= before - 1e-9
    for field in game.ground.fields[:4]:                              # no flow field leads into the wall
        assert not (field.nxt == x)[np.arange(len(field.nxt)) != x].any()
    # the wizard cannot walk into it
    nb = next(n for n in world.graph.adjacent[x] if game.ground.mat[n] != ww.WALL and game.ground.mat[n] != ww.WATER)
    put(game, nb, toward=x)
    seen = set()
    s = game.session
    s.control(FORWARD)
    for _ in range(90):
        s.advance(STEP)
        seen.add(int(s.column("node")[0]) - ww.MAP0)
    assert x not in seen
    assert game.session.column("alive")[lane] == 1.0
    # a path to the wall itself is refused
    with pytest.raises(ValueError, match="wall bars"):
        game.goto(node=x)


def test_towers_shoot_and_houses_are_homes(world):
    game = fresh()
    s = game.session
    ok = _buildable_beside_road(world, game.ground)
    # a tower raised beside a road shoots the raider that comes near it
    node = ok[3]
    lane = game.place("tower", node=node)
    assert ww.T0 + ww.N_SITES <= lane < ww.A0 and s.column("alive")[lane] == 1.0
    raider = ww.R0
    near = next(n for n in world.graph.adjacent[node] if game.ground.mat[n] not in (ww.WATER, ww.WALL))
    put(game, near, raider)
    s.column("alive")[raider] = 1.0
    hp0 = s.column("hp")[raider]
    put(game, ok[-1], 0)                                              # the wizard far away
    worst = hp0
    for _ in range(300):
        s.advance(STEP)
        worst = min(worst, s.column("hp")[raider] if s.column("alive")[raider] > 0 else 0.0)
        if s.column("alive")[raider] <= 0.0:
            break
    assert worst < hp0                                                # the tower's arrow hurt (or killed) it
    # a house is a settler's home, and mends the folk about it
    house_node = ok[10]
    h_lane = game.place("house", node=house_node)
    homes = s.column("home_node")[ww.V0:ww.B0]
    assert (homes == ww.MAP0 + house_node).sum() == 1
    s.column("alive")[ww.R0:ww.T0] = 0.0                              # a peaceful hour
    s.column("timer")[ww.R0:ww.T0] = 1.0e9
    put(game, house_node, 0)                                          # (at its door: within HOUSE_REACH)
    s.column("hp")[0] = 2.0
    run(game, 3.0, {})
    sheltered = s.column("hp")[0]
    far = fresh()
    far.session.column("alive")[ww.R0:ww.T0] = 0.0
    far.session.column("hp")[0] = 2.0
    run(far, 3.0, {})
    assert sheltered > far.session.column("hp")[0] + 0.5
    assert s.column("alive")[h_lane] == 1.0


def test_the_walker_declares_itself_and_its_controls_to_the_page(played):
    st = played.state()
    assert st["avatar"] == "walker"                                    # the page draws no rocket lights, strobe, headlights or shield
    assert [c["id"] for c in st["controls_ui"]] == ["tool", "here", "stop"]
    assert {"tool"} <= {c["id"] for c in st["controls_ui"] if c["kind"] == "select"}
    walk = st["predict"]["walk"]
    assert walk["materials"] == {name: f for name, (_i, f) in em.WALK_MATERIALS.items()}
    assert set(st["predict"]["travel"]) >= {"ux", "wp", "route"} and "bind" not in st["predict"]
    assert st["walk"]["nodes"] == len(ww.world().nodes) and len(st["walk"]["mat"]) == st["walk"]["nodes"]
    assert st["walking"] == {"id": 0, "wp": 0, "active": False}
    assert not any(k in st["controls"] for k in ("move_axis_r",))
    # the controls bar's select drives the tool
    played.do({"type": "worldwalk_tool", "tool": "house"})
    assert played.state()["tool"] == "house"
    played.do({"type": "worldwalk_tool", "tool": "walk"})
    with pytest.raises(ValueError):
        played.do({"type": "worldwalk_tool", "tool": "castle"})
