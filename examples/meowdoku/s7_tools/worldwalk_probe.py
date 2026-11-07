"""Headless behaviour probe for World Walk (no server, no page).

Each scenario is a fresh game advanced through the dt system exactly as the
server does (``session.advance``), the wizard standing still (throttle zero)
unless the scenario moves it:

    walk      speeds per surface material (the wizard held forward on each); a click path across land
              prefers the roads; a target across water is swum slowly (or walked around); free
              walking STOPS on release and any stick cancels a click path
    place     a tile not beside a road is refused, one beside a road is accepted and oriented along
              it; a wall on a walker's path reroutes it and bars its node
    commute   nobody plays; every settler should reach a town of the other
              group and come back (arrivals read from the phase flips), and
              the walkers should keep to their own land (water is a slow swim, so paths avoid it)
    build     the builders should found lights on the roads, one at a time,
              none on top of another
    defend    the wizard idle at a town; the towers alone should kill raiders
              or beasts that come, and the raiders should leave their lairs
              for the towns
    hunt      the wizard stands next to a lair; the beasts' and raiders'
              choice networks should bring them to it (distance falls, a blow
              lands)

    python examples/meowdoku/worldwalk_probe.py

The lane is MEOWDOKU_LAW as everywhere (eigen needs TURING_EIGEN_INCLUDE).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.s2_titles.worldwalk import worldwalk as ww  # noqa: E402

STEP = 1.0 / 30.0


def fresh(seed: int = 0) -> ww.WorldWalk:
    game = ww.WorldWalk(seed=seed)
    game.session.control({})
    game.session.advance(1.0 / 60.0)             # round zero instantiates the state (the wizard stands: no held throttle)
    return game


def put_wizard(game: ww.WorldWalk, node: int) -> None:
    """The wizard at a node (position, heading kept tangent, node)."""
    st = game.session.state
    p = ww.world().nodes[node]
    h = np.array([st.hx[0], st.hy[0], st.hz[0]])
    h = h - (h @ p) * p
    h /= np.linalg.norm(h) + 1e-12
    st.ux[0], st.uy[0], st.uz[0] = p
    st.hx[0], st.hy[0], st.hz[0] = h
    st.node[0] = ww.MAP0 + node


def arc_to_wizard(s) -> np.ndarray:
    u = np.stack([s.column("ux"), s.column("uy"), s.column("uz")], axis=1)[:ww.ENT]
    return ww.RADIUS * np.arctan2(np.linalg.norm(np.cross(u, u[0]), axis=1), u @ u[0])


def commute(seed: int, seconds: float = 150.0) -> dict:
    game = fresh(seed)
    s = game.session
    w = ww.world()
    s.state.alive[ww.R0:ww.T0] = 0.0             # a peaceful world: the raiders and beasts never come
    s.state.timer[ww.R0:ww.T0] = 1.0e9
    commuters = slice(ww.V0, ww.R0)
    n = ww.R0 - ww.V0
    phase = s.column("phase")[commuters].copy()
    out_trips, back_trips = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    wet = 0
    t = 0.0
    while t < seconds:
        s.advance(STEP)
        t += STEP
        now = s.column("phase")[commuters]
        out_trips += (phase < 0.5) & (now > 0.5)
        back_trips += (phase > 0.5) & (now < 0.5)
        phase = now.copy()
        wet += int(((s.column("mat")[s.column("node")[:ww.ENT].astype(int)] == ww.WATER)[:ww.T0] & (s.column("alive")[:ww.T0] > 0.0))[1:].sum())
    return {"out": out_trips.tolist(), "back": back_trips.tolist(),
            "all_went_out": bool((out_trips > 0).all()), "all_came_back": bool((back_trips > 0).all()),
            "trips": float(s.column("trips")[0]), "walker_steps_on_water": wet}


def build(seed: int, seconds: float = 120.0) -> dict:
    game = fresh(seed)
    s = game.session
    w = ww.world()
    t = 0.0
    while t < seconds:
        s.advance(STEP)
        t += STEP
    lit = np.flatnonzero(s.column("alive")[ww.L0:ww.ENT] > 0.0) + ww.L0
    u = np.stack([s.column("ux"), s.column("uy"), s.column("uz")], axis=1)[lit]
    gaps = []
    for i in range(len(lit)):
        for j in range(i):
            gaps.append(ww.RADIUS * float(np.arccos(np.clip(u[i] @ u[j], -1.0, 1.0))))
    nodes = s.column("node")[lit].astype(int) - ww.MAP0
    return {"lights": int(len(lit)), "founded": float(s.column("built")[0]),
            "all_on_roads": bool(all(w.mat[k] == ww.ROAD for k in nodes)) if len(lit) else None,
            "closest_pair": min(gaps) if gaps else None}


def defend(seed: int, seconds: float = 180.0) -> dict:
    game = fresh(seed)
    s = game.session
    w = ww.world()
    put_wizard(game, w.site_nodes[w.towns_a[0]])
    t = 0.0
    kills = []
    left_lair = 0.0
    near_town = 99.0
    sites = w.nodes[[w.site_nodes[i] for i in w.towns]]
    while t < seconds:
        s.advance(STEP)
        t += STEP
        kills.append(float(s.column("kills")[0]))
        if int(t / STEP) % 30 == 0:
            r = np.stack([s.column("ux"), s.column("uy"), s.column("uz")], axis=1)[ww.R0:ww.M0]
            alive = s.column("alive")[ww.R0:ww.M0] > 0.0
            if alive.any():
                near_town = min(near_town, float(ww.RADIUS * np.arccos(np.clip(r[alive] @ sites.T, -1, 1)).min()))
            home = s.column("home_node")[ww.R0:ww.M0].astype(int) - ww.MAP0
            moved = ww.RADIUS * np.arccos(np.clip((r * w.nodes[home]).sum(axis=1), -1, 1))
            left_lair = max(left_lair, float(moved[alive].max()) if alive.any() else 0.0)
    return {"kills": kills[-1], "first_kill_s": next((i * STEP for i, k in enumerate(kills) if k > 0), None),
            "raiders_farthest_from_lair": left_lair, "raiders_closest_to_a_town": near_town,
            "wizard_hp": float(s.column("lives")[0])}


def hunt(seed: int, seconds: float = 40.0) -> dict:
    game = fresh(seed)
    s = game.session
    w = ww.world()
    lair = w.lairs[0]
    # a few steps from the lair, on the main land
    near = np.argsort(-(w.nodes @ w.nodes[lair]))[3]
    put_wizard(game, int(near))
    hp0 = float(s.column("lives")[0])
    nearest, first_blow = [], None
    t = 0.0
    monsters = slice(ww.R0, ww.T0)
    while t < seconds:
        s.advance(STEP)
        t += STEP
        alive = s.column("alive")[monsters] > 0.0
        d = arc_to_wizard(s)[monsters]
        nearest.append(float(d[alive].min()) if alive.any() else 99.0)
        if first_blow is None and float(s.column("lives")[0]) < hp0:
            first_blow = t
    return {"nearest_at_start": nearest[0], "nearest_min": min(nearest), "first_blow_s": first_blow,
            "wizard_hp": float(s.column("lives")[0])}


def place_wizard_still(game: ww.WorldWalk, node: int, toward: int | None = None) -> None:
    """The wizard at a node standing still, facing a neighbour."""
    st, w = game.session.state, ww.world()
    p = w.nodes[node]
    g = w.nodes[toward if toward is not None else w.graph.adjacent[node][0]]
    h = g - (g @ p) * p
    h /= np.linalg.norm(h) + 1e-12
    st.ux[0], st.uy[0], st.uz[0] = p
    st.hx[0], st.hy[0], st.hz[0] = h
    st.node[0] = ww.MAP0 + node
    st.spd[0] = 0.0


def hold(game: ww.WorldWalk, seconds: float, **controls) -> None:
    s = game.session
    s.control({"move_axis": 0.0, "move_axis_y": 0.0, "fire": 0.0, **controls})
    for _ in range(int(round(seconds / STEP))):
        s.advance(STEP)


def walk(seed: int) -> dict:
    """Speeds per material, a click path over land, a swim, free walking and its cancel."""
    from meowdoku.s1_engine import entity_motion as em
    w = ww.world()
    out = {"speed_by_material": {}}
    for name, (index, _f) in em.WALK_MATERIALS.items():
        if name == "wall":
            continue
        pool = np.flatnonzero((w.mat == index) & (w.slope <= 0.1))
        node = int(pool[len(pool) // 2])
        game = fresh(seed)
        for _ in range(48):
            put_wizard(game, node)
            game.session.control({"move_axis_y": -1.0})
            game.session.advance(1.0 / 60.0)
        want = float(em.walk_speed(ww.WALK_BASE, np.asarray([float(index)]), np.asarray([w.slope[node]]))[0][0])
        out["speed_by_material"][name] = (round(float(game.session.column("spd")[0]), 3), round(want, 3))
    game = fresh(seed)
    towns = [w.site_nodes[i] for i in w.towns]
    path = min((p for a in towns for b in towns if a < b for p in [game.ground.path(a, b)] if p), key=len)
    place_wizard_still(game, path[0], path[1])
    game.goto(node=path[-1])
    started, t = 0.0, 0.0
    while game.session.column("route")[0] > 0.5 and t < 400.0:
        game.session.advance(STEP)
        t += STEP
    out["click_path"] = {"nodes": len(path), "road_share": round(float((w.mat[path] == ww.ROAD).mean()), 2),
                         "water_nodes": int((w.mat[path] == ww.WATER).sum()), "seconds": round(t, 1),
                         "arrived_within": round(float(np.arccos(np.clip(game.session_point(0) @ w.nodes[path[-1]], -1, 1)) * ww.RADIUS), 2)}
    isl = int(np.flatnonzero((w.comp >= 0) & (w.comp != w.main))[0])
    swim = game.ground.path(towns[0], isl)
    out["to_an_island"] = {"nodes": len(swim), "water_nodes": int((w.mat[swim] == ww.WATER).sum())}
    place_wizard_still(game, towns[0])
    hold(game, 1.5, move_axis_y=-1.0)
    moving = float(game.session.column("spd")[0])
    hold(game, 1.5)
    out["free_walk"] = {"spd_held": round(moving, 2), "spd_released": round(float(game.session.column("spd")[0]), 4)}
    place_wizard_still(game, path[0], path[1])
    game.goto(node=path[-1])
    hold(game, 1.0)
    walking = float(game.session.column("route")[0])
    hold(game, 0.3, move_axis=1.0)
    out["input_cancels"] = {"route_before": walking, "route_after": float(game.session.column("route")[0])}
    return out


def place(seed: int) -> dict:
    """A tile away from the roads is refused; one beside a road is accepted (along it); a wall reroutes."""
    w = ww.world()
    game = fresh(seed)
    ground = game.ground
    far = next(n for n in range(len(w.nodes)) if w.mat[n] == ww.GRASS and n not in w.road_nodes
               and not any(b in w.road_nodes for b in w.graph.adjacent[n]))
    refused = None
    try:
        game.place("house", node=far)
    except ValueError as error:
        refused = str(error)
    ok = next(n for n in range(len(w.nodes)) if ground.buildable(n) is None)
    lane = game.place("house", node=ok)
    rng = np.random.default_rng(2)
    main = np.flatnonzero(w.comp == w.main)
    rerouted = None
    for _ in range(400):
        a, b = (int(x) for x in rng.choice(main, 2, replace=False))
        path = ground.path(a, b)
        inner = [n for n in (path or [])[2:-2] if ground.buildable(n) is None]
        if inner:
            place_wizard_still(game, a, path[1])
            game.goto(node=b)
            before = len(game.route["nodes"])
            game.place("wall", node=inner[0])
            after = game.route["nodes"]
            rerouted = {"wall_node": inner[0], "path_before": before, "path_after": len(after),
                        "wall_on_new_path": inner[0] in after, "bars_node": bool(game.session.column("mat")[ww.MAP0 + inner[0]] == ww.WALL)}
            break
    return {"far_refused": refused, "near_accepted_lane": lane, "rerouted": rerouted}


def main(argv: list[str]) -> int:
    seed = int(argv[1]) if len(argv) > 1 else 0
    checks = {
        "walk": lambda r: (all(abs(got - want) <= 0.03 * want + 1e-3 for got, want in r["speed_by_material"].values())
                           and r["click_path"]["road_share"] > 0.5 and r["click_path"]["water_nodes"] == 0
                           and r["click_path"]["arrived_within"] < 1.0 and r["to_an_island"]["water_nodes"] > 0
                           and r["free_walk"]["spd_held"] > 1.0 and abs(r["free_walk"]["spd_released"]) < 0.02
                           and r["input_cancels"]["route_before"] == 1.0 and r["input_cancels"]["route_after"] == 0.0),
        "place": lambda r: (r["far_refused"] is not None and "not beside a road" in r["far_refused"]
                            and r["rerouted"] is not None and not r["rerouted"]["wall_on_new_path"] and r["rerouted"]["bars_node"]),
        "commute": lambda r: r["all_went_out"] and r["all_came_back"] and r["walker_steps_on_water"] <= 0.01 * (150.0 / STEP) * (ww.T0 - 1),   # (they may swim, but the paths avoid it)
        "build": lambda r: r["lights"] >= 4 and (r["closest_pair"] or 0.0) >= ww.BUILD_GAP * 0.9 and r["all_on_roads"],
        "defend": lambda r: r["kills"] > 0 and r["raiders_farthest_from_lair"] > 5.0,
        "hunt": lambda r: r["nearest_min"] < 1.0 and r["first_blow_s"] is not None,
    }
    failed = []
    for name, run in (("walk", walk), ("place", place), ("commute", commute), ("build", build), ("defend", defend), ("hunt", hunt)):
        started = time.perf_counter()
        result = run(seed)
        print(f"{name:8s} {result}  [{time.perf_counter() - started:.1f}s wall]", flush=True)
        ok = checks[name](result)
        print(f"{name:8s} {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
