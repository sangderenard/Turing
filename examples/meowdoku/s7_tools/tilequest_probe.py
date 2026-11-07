"""Headless behaviour probe for Tile Quest (no server, no page).

Three scenarios on the same seeded map, each a fresh game advanced through the
dt system exactly as the server does (``session.advance``):

    hunt     the player stands still in the open east of the lair rooms; the
             monsters' choice networks should bring them to it (distance
             falls, a blow lands)
    commute  nobody plays; every villager should reach its workplace and
             its home (arrivals read from the phase flips, per lane)
    defend   the player stays inside the village and never swings; the towers
             alone should kill monsters

    python examples/meowdoku/tilequest_probe.py [seed]

The lane is MEOWDOKU_LAW as everywhere (eigen needs TURING_EIGEN_INCLUDE).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.s2_titles.tilequest import tilequest as tq  # noqa: E402

STEP = 1.0 / 30.0


def fresh(seed: int) -> tq.TileQuest:
    game = tq.TileQuest(seed=seed)
    game.session.control({})
    game.session.advance(1.0 / 60.0)             # round zero instantiates the state
    return game


def put_player(game: tq.TileQuest, x: float, y: float) -> None:
    assert game.map.passable()[int(y), int(x)], f"({x}, {y}) is not open ground on seed {game.seed}"
    game.session.state.x[0], game.session.state.y[0] = x, y


def hunt(seed: int, seconds: float = 30.0) -> dict:
    game = fresh(seed)
    s = game.session
    put_player(game, 17.5, 7.5)                  # the east bank, just outside the rooms' doors
    s.control({})
    monsters = slice(tq.S0, tq.T0)
    nearest, first_blow = [], None
    hp0 = float(s.column("lives")[0])
    t = 0.0
    while t < seconds:
        s.advance(STEP)
        t += STEP
        alive = s.column("alive")[monsters] > 0.0
        d = np.hypot(s.column("x")[monsters] - s.column("x")[0], s.column("y")[monsters] - s.column("y")[0])
        nearest.append(float(d[alive].min()) if alive.any() else 99.0)
        if first_blow is None and float(s.column("lives")[0]) < hp0:
            first_blow = t
    return {"nearest_at_start": nearest[0], "nearest_min": min(nearest),
            "first_blow_s": first_blow, "player_hp": float(s.column("lives")[0])}


def commute(seed: int, seconds: float = 120.0) -> dict:
    game = fresh(seed)
    s = game.session
    put_player(game, 9.5, 8.5)
    villagers = slice(tq.V0, tq.S0)
    phase = s.column("phase")[villagers].copy()
    to_work = np.zeros(tq.N_VILLAGERS, dtype=int)
    to_home = np.zeros(tq.N_VILLAGERS, dtype=int)
    t = 0.0
    while t < seconds:
        s.advance(STEP)
        t += STEP
        now = s.column("phase")[villagers]
        to_work += (phase < 0.5) & (now > 0.5)          # flipped 0 -> 1: arrived at a workplace
        to_home += (phase > 0.5) & (now < 0.5)          # flipped 1 -> 0: arrived home
        phase = now.copy()
    return {"arrivals_at_work": to_work.tolist(), "arrivals_home": to_home.tolist(),
            "all_reached_work": bool((to_work > 0).all()), "all_reached_home": bool((to_home > 0).all()),
            "trips": float(s.column("trips")[0]), "villagers_lost": float(s.column("losses")[0])}


def defend(seed: int, seconds: float = 90.0) -> dict:
    game = fresh(seed)
    s = game.session
    put_player(game, 4.5, 4.5)                    # the village's far corner, no swing ever
    t = 0.0
    kills_over_time = []
    while t < seconds:
        s.advance(STEP)
        t += STEP
        kills_over_time.append(float(s.column("kills")[0]))
        put_player(game, 4.5, 4.5) if float(s.column("lives")[0]) > 0 else None
    arrows_flown = float(s.column("born")[tq.A0:tq.ENT].max())
    return {"kills_by_towers": kills_over_time[-1], "first_kill_s": next((i * STEP for i, k in enumerate(kills_over_time) if k > 0), None),
            "last_arrow_born": arrows_flown}


def main(argv: list[str]) -> int:
    seed = int(argv[1]) if len(argv) > 1 else 3
    failed = []
    for name, run in (("hunt", hunt), ("commute", commute), ("defend", defend)):
        started = time.perf_counter()
        result = run(seed)
        print(f"{name:8s} {result}  [{time.perf_counter() - started:.1f}s wall]", flush=True)
        ok = {"hunt": lambda r: r["nearest_min"] < 1.0 and r["nearest_min"] < r["nearest_at_start"] and r["first_blow_s"] is not None,
              "commute": lambda r: r["all_reached_work"] and r["all_reached_home"],
              "defend": lambda r: r["kills_by_towers"] > 0}[name](result)
        print(f"{name:8s} {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
