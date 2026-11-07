"""The law against its mirror, and one lane against another, on real states.

``mirror_vs_law``: the player's body physics in ``delve_plan.Mirror`` (what the
generator proved doors reachable with) against ``delve.delve_step`` (what the
game runs), stepped side by side from the same states with the same controls
over real generated rooms.  They must agree to rounding.

``lane_vs_numpy``: the same columns stepped through the compiled lane
(MEOWDOKU_LAW) and through the NumPy product, including the step that crosses
a room transition (a different room's columns written into the same compiled
law).  ``native_vs_numpy_on_route``: the compiled law handed the very columns
the real game passed the NumPy product while it played the scripted route.
Run as a script (MEOWDOKU_LAW=numpy, the recording lane; the compiled lane is
built explicitly, one compile):

    MEOWDOKU_LAW=numpy python examples/meowdoku/delve_parity.py [seeds...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.s2_titles.delve import delve
from meowdoku.s2_titles.delve import delve_plan
from meowdoku.s2_titles.delve import delve_world  # noqa: E402
from meowdoku.s2_titles.delve.delve_rules import LANES, MAP0, PLAYER_RY  # noqa: E402

DT = 1.0 / 30.0


def law_columns(room: delve_world.Room, place_xy, climbing: bool, ability: bool) -> dict[str, np.ndarray]:
    cols = delve.blank_columns()
    cols["ability"][:] = 1.0 if ability else 0.0
    delve.place_room(cols, room.grid, [], [], (place_xy[0], place_xy[1], climbing))
    return cols


def step_law(piece: delve.DelvePiece, cols: dict, dt: float, controls: dict) -> None:
    for name, value in controls.items():
        cols[name][...] = value
    out = piece(*(cols[n] if n != "dt" else np.full(LANES, dt) for n in piece.argument_names))
    for name, value in zip(delve.STATE, out):
        cols[name] = np.array(value, dtype=np.float64)


def mirror_vs_law(world: delve_world.World, rng: np.random.Generator, scenarios: int = 40, steps: int = 90, debug: bool = False) -> dict:
    """Random control scripts from random reachable places in the world's rooms; the largest disagreement."""
    piece = delve.DelvePiece()
    piece.compile(delve.blank_columns())
    worst, compared = 0.0, 0
    for _ in range(scenarios):
        room = world.rooms[int(rng.integers(len(world.rooms)))]
        places = [p for p in room.plan.places]
        p = places[int(rng.integers(len(places)))]
        if p[0] == "s":
            x, y, climbing = p[1] / 2.0, p[2] - PLAYER_RY - 1.0e-3, False
        else:
            x, y, climbing = p[1] + 0.5, p[2] + 0.5, True
        ability = bool(rng.integers(2))
        cols = law_columns(room, (x, y), climbing, ability)
        mirror = delve_plan.Mirror(room.grid)
        s = {"x": np.array([x]), "y": np.array([y]), "vx": np.zeros(1), "vy": np.zeros(1), "buf": np.zeros(1),
             "coy": np.zeros(1), "air": np.ones(1), "pf": np.zeros(1), "climb": np.array([1.0 if climbing else 0.0])}
        axis, fire, axy = 0.0, 0.0, 0.0
        for k in range(steps):
            if k % 6 == 0:
                axis = float(rng.choice([-1.0, 0.0, 1.0]))
                axy = float(rng.choice([-1.0, 0.0, 0.0, 1.0]))
            if k % 5 == 0:
                fire = float(rng.random() < 0.4)
            step_law(piece, cols, DT, {"move_axis": axis, "fire": fire, "move_axis_y": axy})
            land, spiked, dbl = mirror.step(s, np.array([axis]), np.array([fire]), np.array([axy]), np.array([ability]), DT)
            if spiked[0] or (cols["ability"][0] > 0.5) != ability or cols["exit"][0] > 0 or cols["x"][0] < 0.2 or cols["x"][0] > room.w - 0.2:
                break                      # hazards, doors and the room's edge are not the mirror's business
            error = max(abs(cols["x"][0] - s["x"][0]), abs(cols["y"][0] - s["y"][0]), abs(cols["vy"][0] - s["vy"][0]))
            worst, compared = max(worst, float(error)), compared + 1
            if error > 1e-6 and debug:
                print('MISMATCH room', room.id, 'place', p, 'step', k, 'ability', ability, 'ctl', axis, fire, axy, 'law', cols['x'][0], cols['y'][0], cols['vy'][0], 'mirror', s['x'][0], s['y'][0], s['vy'][0])
                break
    return {"scenarios": scenarios, "steps_compared": compared, "worst": worst}


def lane_vs_numpy(world: delve_world.World, rng: np.random.Generator, rounds: int = 120) -> dict:
    """The compiled lane against the NumPy product over a play that crosses a room transition.

    Both are given the same columns and the same controls at the same dt; after every step every state column
    must match.  Midway the player is carried through a real room transition (``Delve.transition``), which
    rewrites tile / bounds / enemy columns in place, and the two lanes continue on the new room."""
    import importlib
    import os
    lane = os.environ.get("MEOWDOKU_LAW", "eigen")
    numpy_piece = delve.DelvePiece()
    native_piece = delve.DelvePiece()
    cols = delve.blank_columns()
    room = world.rooms[world.start_room]
    delve.place_room(cols, room.grid, room.walkers, room.flyers, room.spawn_arrival())
    numpy_piece._compile_numpy(cols)
    native_piece._compile_native(cols, "develop")             # the compiled lane, explicitly (whatever MEOWDOKU_LAW says)
    ref = {k: v.copy() for k, v in cols.items()}
    nat = {k: v.copy() for k, v in cols.items()}
    worst, steps, transitions = 0.0, 0, 0
    axis = 1.0
    here = room.id
    for k in range(rounds):
        if k % 9 == 0:
            axis = float(rng.choice([-1.0, 1.0, 1.0]))
        ctl = {"move_axis": axis, "fire": float(rng.random() < 0.3), "move_axis_y": float(rng.choice([-1.0, 0.0, 1.0]))}
        for c in (ref, nat):
            for name, value in ctl.items():
                c[name][...] = value
        args_ref = [ref[n] if n != "dt" else np.full(LANES, DT) for n in numpy_piece.argument_names]
        args_nat = [nat[n] if n != "dt" else np.full(LANES, DT) for n in native_piece.argument_names]
        out_ref, out_nat = numpy_piece(*args_ref), native_piece(*args_nat)
        for name, a, b in zip(delve.STATE, out_ref, out_nat):
            ref[name], nat[name] = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
            worst = max(worst, float(np.max(np.abs(ref[name] - nat[name]))))
        steps += 1
        if k in (rounds // 2, rounds // 4 * 3):
            # a room transition: the same rewrite into both lanes' columns
            there = world.rooms[(here + 1) % len(world.rooms)]
            for c in (ref, nat):
                delve.place_room(c, there.grid, there.walkers, there.flyers, there.spawn_arrival())
            here = there.id
            transitions += 1
    return {"lane": native_piece.lane, "steps": steps, "transitions": transitions, "worst": worst}


def native_vs_numpy_on_route(seed: int = 3, keep_near_transition: int = 25, every: int = 15) -> dict:
    """The compiled lane against the NumPy product on REAL states: play the scripted route (``delve_probe.route``)
    through the real game on the NumPy lane, recording every law call made within ``keep_near_transition`` calls of a
    room transition and one in ``every`` of the rest -- the columns the dt system passed in and what the product
    returned -- then give each recorded call to the compiled law and compare all of its outputs.  A transition rewrites
    tiles, bounds and enemies in the very columns the compiled law is handed, so the calls either side of one are the
    ones that matter."""
    import os
    from meowdoku.s7_tools import delve_probe
    if os.environ.get("MEOWDOKU_LAW") != "numpy":
        raise RuntimeError("record on the NumPy lane: MEOWDOKU_LAW=numpy (the compiled lane is built here explicitly)")
    records: list = []
    calls = {"n": 0, "last_transition": -10**9, "serial": 0}
    original = delve.DelvePiece.__call__

    def recording(self, *columns):
        out = original(self, *columns)
        calls["n"] += 1
        game = recording.game
        if game is not None and game.serial != calls["serial"]:
            calls["serial"], calls["last_transition"] = game.serial, calls["n"]
        records.append((calls["n"], [np.array(c, dtype=np.float64) for c in columns], [np.array(o) for o in out])
                       if calls["n"] - calls["last_transition"] < keep_near_transition or calls["n"] % every == 0 else None)
        return out

    recording.game = None
    delve.DelvePiece.__call__ = recording
    try:
        game = delve_probe.quiet_game(seed)
        recording.game = game
        result = delve_probe.route(seed)
    finally:
        delve.DelvePiece.__call__ = original
    kept = [r for r in records if r is not None]
    native = delve.DelvePiece()
    base = delve.blank_columns()
    native._compile_native(base, "develop")
    worst, near = 0.0, 0
    for n, columns, expected in kept:
        got = native(*columns)
        worst = max(worst, max(float(np.max(np.abs(np.asarray(g) - np.asarray(e)))) for g, e in zip(got, expected)))
        near += n - calls["last_transition"] < keep_near_transition
    return {"seed": seed, "route_ok": result.get("ok"), "law_calls": calls["n"], "compared": len(kept),
            "transitions": result.get("transitions"), "lane": native.lane, "worst": worst}


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [3]
    for seed in seeds:
        world = delve_world.generate(seed)
        rng = np.random.default_rng(seed)
        print(f"seed {seed} mirror vs law:", mirror_vs_law(world, rng))
        print(f"seed {seed} lane vs numpy:", lane_vs_numpy(world, rng))
        if os.environ.get("MEOWDOKU_LAW") == "numpy":
            print(f"seed {seed} compiled law on the route's real states (NumPy product recorded):", native_vs_numpy_on_route(seed))
