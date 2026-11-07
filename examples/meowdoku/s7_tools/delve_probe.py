"""Headless probes for Delve (no server, no page).

    python examples/meowdoku/delve_probe.py survey [first_seed] [count] [small|large]
        generate many worlds; report how many pass on the FIRST layout, why the rest were
        re-rolled (the failure cases), and the final worlds' reachability numbers
    python examples/meowdoku/delve_probe.py edges [seed] [count]
        replay planned edges through the REAL game (the dt system, 1/60 s ticks): does the
        body land where the plan said it would?
    python examples/meowdoku/delve_probe.py route [seed]
        walk a scripted route from the start room to the goal through the real law: the
        orb first when the way is gated, through every room transition on the way

The lane is MEOWDOKU_LAW as everywhere (eigen needs TURING_EIGEN_INCLUDE; ``llvm`` needs nothing).
"""

from __future__ import annotations

import collections
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.s2_titles.delve import delve
from meowdoku.s2_titles.delve import delve_world  # noqa: E402
from meowdoku.s2_titles.delve.delve_plan import PLAN_DT, Variant  # noqa: E402
from meowdoku.s2_titles.delve.delve_rules import (  # noqa: E402
    ENT, LADDER, MAP0, MAXH, MAXW, ONEWAY, PLAYER_RY, SOLID, SPEED)

TICK = 1.0 / 60.0


# ------------------------------------------------------------------ the survey
def survey(first: int = 0, count: int = 40, size: str = "small", gated: bool = True) -> dict:
    """First-layout success and the reasons for the re-rolls, over ``count`` seeds."""
    first_ok, reasons, final, seconds = 0, collections.Counter(), [], 0.0
    for seed in range(first, first + count):
        t0 = time.perf_counter()
        rng = np.random.default_rng([seed, 1])
        try:
            delve_world._generate_once(rng, seed, size, gated)
            first_ok += 1
        except delve_world.GenerationFailed as error:
            reasons[re.sub(r"\d+", "N", str(error)).split(" (")[0][:96]] += 1
        world = delve_world.generate(seed, size, gated)
        seconds += time.perf_counter() - t0
        final.append(world)
    rooms = sum(len(w.rooms) for w in final)
    return {"seeds": count, "size": size, "first_layout_ok": first_ok, "reroll_reasons": dict(reasons.most_common()),
            "worlds_valid": len(final), "rooms": rooms,
            "gated_worlds": sum(1 for w in final if w.gate_room is not None),
            "orb_reachable_without_ability": sum(1 for w in final if w.report["orb_reachable_without_ability"]),
            "all_rooms_reachable_with_ability": sum(1 for w in final if w.report["reached_with_ability"] == len(w.rooms)),
            "rooms_behind_gate": sum(len(w.rooms) - w.report["reached_without_ability"] for w in final if w.gate_room is not None),
            "one_way_drops": sum(w.report["one_way_joins"] for w in final), "loops": sum(w.report["loops"] for w in final),
            "mean_attempts": float(np.mean([w.report["attempt"] for w in final])),
            "seconds_per_world": seconds / count}


# ------------------------------------------------------------------ driving the real game
def quiet_game(seed: int, size: str = "small", gated: bool = True, enemies: bool = False) -> delve.Delve:
    game = delve.Delve(seed=seed, size=size, gated=gated, aspect=1.7)
    game.quiet = not enemies
    game.session.control({})
    game.session.advance(TICK)
    # round zero has loaded the start room WITH enemies; load it again quiet
    if game.quiet:
        game.load_room(game.writable(), game.room, game.world.rooms[game.room].spawn_arrival())
    return game


def col(game: delve.Delve, name: str, lane: int = 0) -> float:
    return float(game.session.column(name)[lane])


def teleport(game: delve.Delve, room_id: int, x: float, y: float, vx: float = 0.0, climbing: bool = False) -> None:
    """Put the player at ``(x, y)`` in ``room_id`` (a proof tool: how a planned edge starts)."""
    if game.room != room_id:
        game.room = room_id
        game.visited.add(room_id)
        game.load_room(game.writable(), room_id, (x, y, climbing))
    cols = game.writable()
    cols["x"][0], cols["y"][0], cols["vx"][0], cols["vy"][0] = x, y, vx, 0.0
    cols["climb"][0], cols["buf"][0], cols["coy"][0], cols["pf"][0], cols["exit"][...] = float(climbing), 0.0, 0.0, 0.0, 0.0
    game.session.state.air[0] = 1.0


def tile_at(game: delve.Delve, x: float, y: float) -> int:
    room = game.world.rooms[game.room]
    grid = np.asarray(game.session.column("tile")[MAP0:]).reshape(MAXH, MAXW)
    return int(grid[int(np.clip(np.floor(y), 0, room.h - 1)), int(np.clip(np.floor(x), 0, room.w - 1))])


def supported(game: delve.Delve) -> bool:
    """Is the body resting on a solid or a plank (the law's own ground sense)?"""
    x, y = col(game, "x"), col(game, "y")
    feet = y + PLAYER_RY
    row = int(np.floor(feet + 0.05))
    below = [tile_at(game, x + dx, feet + 0.05) for dx in (-0.27, 0.27)]
    return -0.05 <= col(game, "vy") < 0.5 and (SOLID in below or (ONEWAY in below and feet <= row + 0.02))


def place_of(game: delve.Delve):
    """The plan place the player is at now (the valid one nearest), or None while airborne."""
    x, y = col(game, "x"), col(game, "y")
    if supported(game):
        plan, row = game.world.rooms[game.room].plan, int(round(y + PLAYER_RY))
        for qx in sorted({int(round(x * 2)), int(np.floor(x * 2)), int(np.ceil(x * 2))}, key=lambda q: abs(q / 2.0 - x)):
            if plan.standing_ok(qx, row):
                return ("s", qx, row)
    if tile_at(game, x, y) == LADDER and ("l", int(np.floor(x)), int(np.floor(y))) in game.world.rooms[game.room].plan.edges:
        return ("l", int(np.floor(x)), int(np.floor(y)))
    return None


def run(game: delve.Delve, axis: float, fire: float, axy: float, seconds: float) -> None:
    game.session.control({"move_axis": axis, "fire": fire, "move_axis_y": axy, "steer_x": -1.0})
    game.session.advance(seconds)
    game.check_exit()


def settle(game: delve.Delve, seconds: float = 0.4) -> None:
    for _ in range(int(seconds / TICK)):
        run(game, 0.0, 0.0, 0.0, TICK)


def play_edge(game: delve.Delve, plan, src, edge, room_id: int | None = None) -> None:
    """Execute one planned edge in the real game, from the place it was planned from."""
    room_id = game.room if room_id is None else room_id
    if edge.variant is None:                                   # a walk or a ladder move
        dst = edge.dst
        for _ in range(120):
            x, y = col(game, "x"), col(game, "y")
            if dst[0] == "s":
                tx = dst[1] / 2.0
                if abs(x - tx) < 0.2 + 0.06 * abs(col(game, "vx")):
                    break
                run(game, 1.0 if tx > x else -1.0, 0.0, 0.0, TICK)
            else:
                ty = dst[2] + 0.5
                if abs(y - ty) < 0.12 or game.room != room_id:
                    break
                # a climber keeps to the middle of its ladder (the head clears the ceiling's opening)
                run(game, float(np.clip((dst[1] + 0.5 - x) * 4.0, -1.0, 1.0)), 0.0, 1.0 if ty > y else -1.0, TICK)
        settle(game, 0.15)
        return
    v: Variant = edge.variant
    sx = src[1] / 2.0 if src[0] == "s" else src[1] + 0.5
    sy = src[2] - PLAYER_RY - 1.0e-3 if src[0] == "s" else src[2] + 0.5
    teleport(game, room_id, sx, sy, vx=v.axis * SPEED if (v.run and src[0] == "s") else 0.0, climbing=src[0] == "l")
    airborne, left_ladder = False, False                       # (the plan arms a ladder grab once the body has left every ladder cell)
    for k in range(int(2.2 / TICK)):
        tt = k * TICK
        in_ladder = tile_at(game, col(game, "x"), col(game, "y")) == LADDER
        grab = airborne and left_ladder and in_ladder              # reaching for a ladder it has come to
        run(game, float(v.axis), 1.0 if v.pressed(tt) else 0.0, -1.0 if grab else 0.0, TICK)
        if game.room != room_id:
            return
        left_ladder = left_ladder or not in_ladder
        airborne = airborne or not supported(game)
        if airborne and (supported(game) or (left_ladder and col(game, "climb") > 0.5)):
            break


# ------------------------------------------------------------------ edges: does the real game land where the plan says
def edges(seed: int = 3, count: int = 60, tick: float = 1.0 / 60.0) -> dict:
    """Replay a sample of planned arcs through the real game (``tick``-second rounds of the dt system; the plan
    itself is made at 1/60 s) and compare where the body ends up with the place the plan said."""
    global TICK
    TICK, saved = tick, TICK
    try:
        return _edges(seed, count, tick)
    finally:
        TICK = saved


def _edges(seed: int, count: int, tick: float) -> dict:
    game = quiet_game(seed)
    rng = np.random.default_rng(seed)
    pool = []
    for room in game.world.rooms:
        for src, outs in room.plan.edges.items():
            for dst, e in outs.items():
                if e.variant is not None and dst[0] == "s":
                    pool.append((room.id, src, dst, e))
    chosen = [pool[int(i)] for i in rng.permutation(len(pool))[:count]]
    ok, misses = 0, []
    for room_id, src, dst, e in chosen:
        room = game.world.rooms[room_id]
        game.session.state.ability[...] = 1.0
        play_edge(game, room.plan, src, e, room_id)
        got = place_of(game)
        good = game.room == room_id and got is not None and got[0] == "s" and got[2] == dst[2] and abs(got[1] - dst[1]) <= 1
        ok += bool(good)
        if not good:
            misses.append({"room": room_id, "from": src, "planned": dst, "got": got, "variant": e.variant, "double": e.needs_double})
    return {"tick": round(tick, 5), "sampled": len(chosen), "landed_as_planned": ok, "misses": misses[:8]}


# ------------------------------------------------------------------ the route
def world_route(world: delve_world.World, goal: int | None = None, start: tuple | None = None) -> list[dict] | None:
    """Room by room from the start to ``goal``: for each room the arrival door, whether to take the orb on the
    way, and the door to leave by (a breadth-first search over (room, arrival, ability))."""
    goal = world.goal_room if goal is None else goal
    rooms = world.rooms
    # ``start``: (room, arrival door or -1, ability) -- from where the player is now, not the world's start
    start = start or (world.start_room, -1, world.gate_room is None)       # no gate: nothing to find first
    prev, queue = {start: None}, [start]
    for node in queue:
        room_id, d, ability = node
        room = rooms[room_id]
        entry = room.spawn_place() if d < 0 else room.entry_place(d)
        reach = room.plan.reach([entry], ability)
        if room_id == goal and ability:
            break
        if not ability and room.orb is not None and ("s", room.orb[0] * 2 + 1, room.orb[1] + 1) in reach:
            nxt = (room_id, d, True)
            if nxt not in prev:
                prev[nxt] = (node, {"orb": True})
                queue.append(nxt)
        for e in range(len(room.doors)):
            p = room.exit_place(e)
            if p is None or p not in reach:
                continue
            other = world.links.get((room_id, e))
            if other is None:
                continue
            nxt = (other[0], other[1], ability)
            if nxt not in prev:
                prev[nxt] = (node, {"exit": e})
                queue.append(nxt)
    end = next((n for n in prev if n[0] == goal and n[2]), None)
    if end is None:
        return None
    steps = []
    while prev[end] is not None:
        end, how = prev[end][0], prev[end][1]
        steps.append(how)
    steps.reverse()
    legs, current = [], {"orb": False}
    for how in steps:
        if "orb" in how:
            current["orb"] = True
        else:
            current["exit"] = how["exit"]
            legs.append(current)
            current = {"orb": False}
    return legs


def go_to(game: delve.Delve, target, ability: bool, limit: int = 60) -> bool:
    """Closed loop inside one room: from wherever the player stands, take the plan's route to ``target``."""
    room = game.world.rooms[game.room]
    room_id = game.room
    for _ in range(limit):
        here = place_of(game)
        if here is None:
            settle(game, 0.3)
            here = place_of(game)
            if here is None:
                return False
        if here == target or (target[0] == "s" and here[0] == "s" and here[2] == target[2] and abs(here[1] - target[1]) <= 1):
            return True
        path = room.plan.path(here, target, ability)
        if not path:
            return False
        edge = path[0]
        play_edge(game, room.plan, here, edge)
        if game.room != room_id:
            return False
    return False


def leave(game: delve.Delve, door: int, seconds: float = 4.0) -> bool:
    """From the door's exit place, go out through it (walk out, climb out, or step into the drop)."""
    room_id = game.room
    d = game.world.rooms[room_id].doors[door]
    axis, axy = {"L": (-1.0, 0.0), "R": (1.0, 0.0), "U": (0.0, -1.0)}.get(d.side, (0.0, 1.0))
    if d.side == "D" and d.oneway:
        axis, axy = 1.0, 0.0                                     # the exit place is left of the hole
    for _ in range(int(seconds / TICK)):
        if d.side in ("U", "D") and not d.oneway:                # up or down a ladder: keep to its middle
            axis = float(np.clip((d.a + 0.5 - col(game, "x")) * 4.0, -1.0, 1.0))
        run(game, axis, 0.0, axy, TICK)
        if game.room != room_id:
            return True
    return False


def route(seed: int = 3, size: str = "small", gated: bool = True, enemies: bool = False) -> dict:
    """Walk the scripted route from the start room to the goal room through the real law."""
    game = quiet_game(seed, size, gated, enemies)
    world = game.world
    legs = world_route(world)
    if legs is None:
        return {"seed": seed, "ok": False, "why": "no route in the world's own graph"}
    log, started = [], time.perf_counter()
    game.session.state.ability[...] = 0.0 if world.gate_room is not None else 1.0
    for leg in legs:
        room = world.rooms[game.room]
        record = {"room": room.name, "id": room.id, "took_orb": False}
        if game.room == world.start_room and not game.hist:
            teleport(game, room.id, *room.spawn_arrival()[:2])
        if leg["orb"]:
            orb = ("s", room.orb[0] * 2 + 1, room.orb[1] + 1)
            go_to(game, orb, col(game, "ability") > 0.5)
            settle(game, 0.2)
            record["took_orb"] = col(game, "ability") > 0.5
        e = leg["exit"]
        reached = go_to(game, room.exit_place(e), col(game, "ability") > 0.5)
        record["reached_door"] = reached
        record["left"] = leave(game, e) if reached else False
        record["door"] = room.doors[e].side
        log.append(record)
        if not record["left"]:
            return {"seed": seed, "ok": False, "why": f"stuck in {room.name} ({room.id}) going to door {room.doors[e].side}",
                    "log": log, "at": (col(game, "x"), col(game, "y"))}
    return {"seed": seed, "ok": game.room == world.goal_room, "rooms_walked": len(log), "transitions": game.serial,
            "orb_taken": any(r["took_orb"] for r in log), "ability": col(game, "ability") > 0.5,
            "hearts": col(game, "lives"), "gems": int(col(game, "gems")), "gated": world.gate_room is not None,
            "game_seconds": col(game, "clock"), "wall_seconds": time.perf_counter() - started, "lane": game.session.piece.lane,
            "rooms": [r["room"] for r in log]}


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "route"
    args = sys.argv[2:]
    if what == "survey":
        print(survey(int(args[0]) if args else 0, int(args[1]) if len(args) > 1 else 40, args[2] if len(args) > 2 else "small"))
    elif what == "edges":
        for tick in (1.0 / 60.0, 1.0 / 30.0, 1.0 / 120.0):
            print(edges(int(args[0]) if args else 3, int(args[1]) if len(args) > 1 else 60, tick))
    else:
        print(route(int(args[0]) if args else 3, enemies="--enemies" in args))
