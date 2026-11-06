"""Delve: the room-graph platformer -- its law, the gravity-aware traversal, the world generator, room transitions.

The law is run through the NumPy product (the translatable lane); the native lane is checked against it by
``examples/meowdoku/delve_parity.py`` (MEOWDOKU_LAW=llvm), and the generator at scale by
``examples/meowdoku/delve_probe.py survey``.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.environ.setdefault("MEOWDOKU_LAW", "numpy")

from meowdoku import delve, delve_parity, delve_plan, delve_probe, delve_world  # noqa: E402
from meowdoku.delve_rules import (  # noqa: E402
    EMPTY, LADDER, MAP0, MAXH, MAXW, ONEWAY, SCREEN_H, SCREEN_W, SOLID, SPIKE, camera_axis)

SEEDS = range(8)


def room_grid(width=24, height=10, **cells):
    g = np.zeros((height, width), dtype=int)
    g[0, :] = g[height - 1, :] = SOLID
    g[:, 0] = g[:, width - 1] = SOLID
    return g


# ------------------------------------------------------------------ the traversal is the law's own physics
@pytest.mark.parametrize("seed", [3, 5, 9])
def test_the_mirror_is_the_law(seed):
    world = delve_world.generate(seed)
    result = delve_parity.mirror_vs_law(world, np.random.default_rng(seed), scenarios=12, steps=100)
    assert result["steps_compared"] > 500
    assert result["worst"] < 1e-9


def test_single_jump_clears_the_gaps_the_law_allows_and_not_the_next():
    # the reach is DERIVED from the shared vertical law (exact ballistic, apex hang, JUMP_APEX): a gap of four
    # tiles is crossed by one jump, five is not (re-derived when gravity moved to G_EARTH / UNIT_M)
    for gap, crosses in ((2, True), (4, True), (5, False)):
        grid = room_grid(width=30)
        grid[8, 8:8 + gap] = SPIKE
        plan = delve_plan.RoomPlan.build(grid)
        assert (("s", 2 * (8 + gap) + 2, 9) in plan.reach([("s", 3, 9)], False)) is crosses
    step = room_grid()
    step[7, 5:11] = ONEWAY                                   # feet row 7: two above the floor
    high = room_grid()
    high[6, 5:11] = ONEWAY                                   # three above
    assert any(p[2] == 7 for p in delve_plan.RoomPlan.build(step).reach([("s", 3, 9)], False))
    plan = delve_plan.RoomPlan.build(high)
    assert not any(p[2] == 6 for p in plan.reach([("s", 3, 9)], False))
    assert any(p[2] == 6 for p in plan.reach([("s", 3, 9)], True))             # the double jump climbs it


def test_a_ladder_connects_what_a_jump_cannot():
    grid = room_grid(16, 14)
    grid[5, 8:13] = ONEWAY                                   # feet row 5: eight above the floor (row 13)
    grid[5, 10] = LADDER
    grid[6:13, 10] = LADDER
    plan = delve_plan.RoomPlan.build(grid)
    assert any(p[2] == 5 and p[0] == "s" for p in plan.reach([("s", 3, 13)], False))
    grid[6:13, 10] = EMPTY
    grid[5, 10] = ONEWAY
    assert not any(p[2] == 5 for p in delve_plan.RoomPlan.build(grid).reach([("s", 3, 13)], True))


# ------------------------------------------------------------------ the generator
def _worlds():
    return [delve_world.generate(seed) for seed in SEEDS]


@pytest.fixture(scope="module")
def worlds():
    return _worlds()


def test_generation_is_deterministic():
    a, b, c = delve_world.generate(4), delve_world.generate(4), delve_world.generate(5)
    assert [r.name for r in a.rooms] == [r.name for r in b.rooms]
    assert all((x.grid == y.grid).all() for x, y in zip(a.rooms, b.rooms))
    assert a.links == b.links and a.main_path == b.main_path
    assert a.links != c.links or any(x.grid.shape != y.grid.shape or (x.grid != y.grid).any() for x, y in zip(a.rooms, c.rooms))


def test_rooms_tile_the_map_and_the_doors_line_up(worlds):
    for world in worlds:
        owner = -np.ones((world.gh, world.gw), dtype=int)
        for r in world.rooms:
            assert (owner[r.gy:r.gy + r.ch, r.gx:r.gx + r.cw] < 0).all()
            owner[r.gy:r.gy + r.ch, r.gx:r.gx + r.cw] = r.id
            assert r.grid.shape == (r.h, r.w) and r.h <= MAXH and r.w <= MAXW
            assert (r.grid[0, :][np.array([c for c in range(r.w) if not any(
                d.side == "U" and d.a <= c < d.a + 2 for d in r.doors)])] == SOLID).all()
        assert (owner >= 0).all()
        for (a, da), (b, db) in world.links.items():
            A, B = world.rooms[a], world.rooms[b]
            assert world.links[(b, db)] == (a, da)
            d0, d1 = A.doors[da], B.doors[db]
            if d0.side in "LR":
                assert {d0.side, d1.side} == {"L", "R"}
                assert A.oy + d0.floor == B.oy + d1.floor                   # the corridors are the same world row
                left, right = (A, B) if d0.side == "R" else (B, A)
                assert left.ox + left.w == right.ox                         # and the rooms touch there
            else:
                assert {d0.side, d1.side} == {"U", "D"}
                assert A.ox + d0.a == B.ox + d1.a                           # a shaft is the same world column
                top, bottom = (A, B) if d0.side == "D" else (B, A)
                assert top.oy + top.h == bottom.oy


def test_the_graph_is_connected_with_a_main_path_loops_and_side_branches(worlds):
    for world in worlds:
        seen, stack = {world.start_room}, [world.start_room]
        while stack:
            r = stack.pop()
            for d in range(len(world.rooms[r].doors)):
                o = world.links[(r, d)][0]
                if o not in seen:
                    seen.add(o)
                    stack.append(o)
        assert len(seen) == len(world.rooms)
        assert world.main_path[0] == world.start_room and world.main_path[-1] == world.goal_room
        for a, b in zip(world.main_path, world.main_path[1:]):
            assert any(world.links[(a, d)][0] == b for d in range(len(world.rooms[a].doors)))
    assert any(w.report["loops"] > 0 for w in worlds)
    assert any(len(w.main_path) < len(w.rooms) for w in worlds)               # side branches exist


def test_every_door_is_reachable_from_every_entry(worlds):
    for world in worlds:
        assert world.report["ok"], world.report["why"]
        for room in world.rooms:
            for d in range(len(room.doors)):
                reach = room.plan.reach([room.entry_place(d)], True)
                for e in range(len(room.doors)):
                    place = room.exit_place(e)
                    if place is not None:
                        assert place in reach, (room.id, d, e)


def test_the_gate_needs_the_ability_and_the_orb_comes_first(worlds):
    gated = [w for w in worlds if w.gate_room is not None]
    assert gated, "no gated world in the sample"
    for world in gated:
        assert world.report["orb_reachable_without_ability"]
        assert world.report["reached_without_ability"] < len(world.rooms) == world.report["reached_with_ability"]
        room = world.rooms[world.gate_room]
        gate = next(d for d, door in enumerate(room.doors) if door.gate)
        others = [d for d in range(len(room.doors)) if d != gate]
        assert any(room.exit_place(gate) not in room.plan.reach([room.entry_place(d)], False) for d in others)
        assert all(room.exit_place(gate) in room.plan.reach([room.entry_place(d)], True) for d in others)
        orb_room = world.rooms[world.orb_room]
        assert orb_room.orb is not None and orb_room.grid[orb_room.orb[1], orb_room.orb[0]] == 6


def test_one_way_drops_have_a_way_back(worlds):
    many = [delve_world.generate(seed) for seed in range(8, 24)]
    drops = [w for w in worlds + many if w.report["one_way_joins"]]
    assert drops, "no one-way drop in 24 worlds"
    for world in drops:
        assert world.report["ok"]                                              # the analysis demands a return from every arrival


def test_a_large_world_is_valid():
    world = delve_world.generate(2, "large")
    assert len(world.rooms) >= 12 and world.report["ok"]
    assert {r.kind for r in world.rooms} >= {"chamber", "hall"}


# ------------------------------------------------------------------ the law and the game
def _quiet(seed=3):
    return delve_probe.quiet_game(seed)


def test_the_player_jumps_stands_and_collects():
    game = _quiet()
    s = game.session
    room = game.world.rooms[game.room]
    y0 = float(s.column("y")[0])
    delve_probe.run(game, 0.0, 1.0, 0.0, 1 / 60)
    for _ in range(12):
        delve_probe.run(game, 0.0, 1.0, 0.0, 1 / 60)
    assert float(s.column("y")[0]) < y0 - 0.5                                   # airborne
    delve_probe.settle(game, 1.0)
    assert delve_probe.supported(game)
    assert abs(float(s.column("y")[0]) - y0) < 0.05
    assert room.grid.shape == (10, 16)


def test_a_gem_is_picked_up_and_stays_gone_after_leaving_and_coming_back():
    game = _quiet(7)
    world = game.world
    room = next(r for r in world.rooms if (r.grid == 5).any())
    gy, gx = [int(v[0]) for v in np.nonzero(room.grid == 5)]
    delve_probe.teleport(game, room.id, gx + 0.5, gy + 0.5)
    before = float(game.session.column("score")[0])
    delve_probe.run(game, 0.0, 0.0, 0.0, 1 / 60)
    assert float(game.session.column("score")[0]) == before + 100.0
    tiles = np.asarray(game.session.column("tile")[MAP0:]).reshape(MAXH, MAXW)
    assert tiles[gy, gx] == EMPTY
    other = next(r for r in world.rooms if r.id != room.id)
    delve_probe.teleport(game, other.id, 2.0, other.h - 1.5)
    game.saved[room.id] = tiles[:room.h, :room.w].copy()                       # (a transition keeps the room as left)
    game.room = room.id
    game.load_room(game.writable(), room.id, (gx + 0.5, gy + 0.5, False))
    assert np.asarray(game.session.column("tile")[MAP0:]).reshape(MAXH, MAXW)[gy, gx] == EMPTY


def test_leaving_a_room_changes_the_columns_never_the_law():
    game = _quiet(3)
    piece, product = game.session.piece, game.session.piece.product
    legs = delve_probe.world_route(game.world)
    first = game.room
    room = game.world.rooms[first]
    assert delve_probe.go_to(game, room.exit_place(legs[0]["exit"]), False)
    assert delve_probe.leave(game, legs[0]["exit"])
    assert game.room != first and game.serial == 1
    assert game.session.piece is piece and game.session.piece.product is product            # no recompile
    other = game.world.rooms[game.room]
    tiles = np.asarray(game.session.column("tile")[MAP0:]).reshape(MAXH, MAXW)[:other.h, :other.w]
    assert (tiles[tiles != 5] == other.grid[other.grid != 5]).all() or (tiles == other.grid).all()
    assert float(game.session.column("room_w")[0]) == other.w and float(game.session.column("room_h")[0]) == other.h
    arrival = other.arrival_at(game.world.link(first, legs[0]["exit"])[1])
    assert abs(float(game.session.column("x")[0]) - arrival[0]) < 1.5


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_a_scripted_route_walks_through_rooms_and_transitions_to_the_goal(seed):
    result = delve_probe.route(seed)
    assert result["ok"], result.get("why")
    assert result["transitions"] >= 2 and result["ability"]


def test_planned_jumps_land_where_the_plan_says_in_the_real_game():
    result = delve_probe.edges(3, 24)
    assert result["landed_as_planned"] >= result["sampled"] - 1, result["misses"]


# ------------------------------------------------------------------ the page and the server
def test_the_camera_formula_is_the_same_in_python_and_in_the_page():
    cases = [(5.0, 16.0, 16.0), (3.0, 10.0, 32.0), (30.0, 10.0, 32.0), (16.0, 10.0, 32.0), (9.0, 10.0, 30.0), (0.0, 9.0, 48.0),
             (47.9, 18.0, 48.0)]
    py = [camera_axis(*c) for c in cases]
    assert py[0] == 0.0 and py[1] == 0.0 and py[2] == 22.0 and py[3] == 11.0
    source = (ROOT / "examples/meowdoku/web/app.js").read_text(encoding="utf-8")
    line = re.search(r"const axis = \(pos, win, room\) => [^;]+;", source).group(0)
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = f"{line}\nconsole.log(JSON.stringify({json.dumps([list(c) for c in cases])}.map(([p, w, r]) => axis(p, w, r))));"
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == py


def _table():
    from meowdoku import server
    t = object.__new__(server.Table)
    t.web_dir = None
    t.profiles = t.player = t.player_token = t.last_award = t.recorded = t.game = None
    t.bank_start, t.free_start, t.boards, t.levels = {}, {}, {}, []
    t.lock = threading.RLock()
    return t


def test_the_server_serves_delve_through_the_shared_controls_ui():
    from meowdoku.realtime_socket import SnapshotEncoder
    table = _table()
    state = table.new({"game": "delve", "aspect": 1.7})
    assert "delve" in table.catalog()["games"]
    assert state["board"] == "free" and state["controls"] == ["move_axis", "move_axis_y", "steer_x", "fire"]
    ids = [c["id"] for c in state["controls_ui"]]
    assert ids == ["map", "new", "size", "gated"] and all(c["kind"] in ("button", "select") for c in state["controls_ui"])
    encoder = SnapshotEncoder()
    for _ in range(3):
        state = table.action({"type": "tick", "move_axis": 1.0, "fire": 0.0, "move_axis_y": 0.0, "steer_x": -1.0})
        binary, meta = encoder.encode(table.game, state)
        assert binary
    json.dumps(state)
    assert state["roomview"]["w"] >= SCREEN_W and state["predict"]["adopt_axes"] == ["x"]
    assert table.action({"type": "delve_map"})["world"]["show"] is True
    assert table.action({"type": "delve_map"})["world"]["show"] is False
    before = table.game.world_seed
    table.action({"type": "delve_new"})
    assert table.game.world_seed == before + 1
    assert table.action({"type": "delve_size", "size": "large"})["world"]["grid"] == list(delve_world.SIZES["large"])
    assert table.action({"type": "layout", "aspect": 0.6})["width"] == 9.0


def test_the_map_overlay_draws_visited_rooms_and_joins_on_their_doors():
    game = delve_probe.quiet_game(3)
    legs = delve_probe.world_route(game.world)
    room = game.world.rooms[game.room]
    delve_probe.go_to(game, room.exit_place(legs[0]["exit"]), False)
    delve_probe.leave(game, legs[0]["exit"])
    over = game.state()["world"]
    seen = [r["id"] for r in over["rooms"] if r["seen"]]
    assert len(seen) == 2 and over["current"] == game.room
    for j in over["joins"]:
        a, b = game.world.rooms[j["a"]], game.world.rooms[j["b"]]
        on_a = a.ox <= j["x"] <= a.ox + a.w and a.oy <= j["y"] <= a.oy + a.h
        on_b = b.ox <= j["x"] <= b.ox + b.w and b.oy <= j["y"] <= b.oy + b.h
        assert on_a and on_b                                                  # the door is on the border both rooms share
