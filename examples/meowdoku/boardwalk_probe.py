"""Headless behaviour probe for Boardwalk (no server, no page).

A fresh seeded game advanced through the dt system exactly as the server does
(``session.advance``), the player confirming every turn (``fire``) with the
network's own choice unless a scenario says otherwise.  Every resolved turn is
read back from the columns and printed, and the rules are checked on them:

    allowance   what each choice spent (stroll half, rest none, run more),
                against the roll, the draft and tiredness
    forks       the shortcuts taken, by kind (a brigand strolls into the bog
                road, a racer follows the route, a peddler avoids it)
    circumstances  bog tiles cost double energy, a foe on a tile ends the
                move there, treasure pays its pile, a market trades for a
                rester, duels move coins
    player      the pad: ``move_axis`` +1 forces the fork onto the shortcut,
                ``move_axis_y`` -1 runs (farther than the same roll's walk)

    python examples/meowdoku/boardwalk_probe.py [seed] [turns]

The lane is MEOWDOKU_LAW as everywhere (eigen needs TURING_EIGEN_INCLUDE).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import boardwalk as bw  # noqa: E402
from meowdoku import walkers  # noqa: E402

NAMES = ["pilgrim", "peddler", "peddler", "brigand", "brigand", "brigand", "knight", "rival", "rival"]
STEP = 0.1
WORDS = {i: m for i, m in enumerate(walkers.MOTIVATIONS)}
TURN_WORD = {"wander": "stroll", "follow": "route", "seek": "hunt", "attack": "fight", "flee": "run", "rest": "rest"}


def fresh(seed: int) -> bw.Boardwalk:
    game = bw.Boardwalk(seed=seed)
    game.session.control({})
    game.session.advance(1.0 / 60.0)             # round zero instantiates the state
    return game


def play(game: bw.Boardwalk, turns: int, controls: dict | None = None, log: bool = True) -> list[dict]:
    """Land turns until ``turns`` have resolved; one record per turn."""
    s = game.session
    s.control({"fire": 1.0, **(controls or {})})
    records, seen = [], int(s.column("turns")[0])
    guard = 0
    while len(records) < turns and s.column("over")[0] < 0.5 and guard < 100000:
        before = {k: s.column(k)[:bw.ENT].copy() for k in ("node", "energy", "coin", "hp", "kit", "laps")}
        actor = int(s.column("active")[0])
        s.advance(STEP)
        guard += 1
        if int(s.column("turns")[0]) == seen:
            continue
        seen = int(s.column("turns")[0])
        after = {k: s.column(k)[:bw.ENT].copy() for k in before}
        rec = {"turn": seen, "round": int(s.column("rnd")[0]), "lane": actor, "who": NAMES[actor],
               "roll": int(s.column("roll")[actor]), "choice": TURN_WORD[WORDS[int(s.column("mot")[actor])]],
               "allow": int(s.column("allow")[actor]), "moved": int(s.column("moved")[actor]),
               "from": int(before["node"][actor]), "to": int(after["node"][actor]),
               "kind_to": int(game.board.kinds[int(after["node"][actor])]),
               "energy": (float(before["energy"][actor]), float(after["energy"][actor])),
               "coin": (float(before["coin"][actor]), float(after["coin"][actor])),
               "coins_all": float((after["coin"] - before["coin"]).sum()),
               "hp": (float(before["hp"][actor]), float(after["hp"][actor])),
               "laps": float(after["laps"][actor]), "duel": float(s.column("duels")[0])}
        records.append(rec)
        if log:
            print(f"  t{rec['turn']:3d} r{rec['round']:2d} {rec['who']:8s}#{actor} roll {rec['roll']} {rec['choice']:6s} "
                  f"allow {rec['allow']} moved {rec['moved']} tile {rec['from']:2d}->{rec['to']:2d} "
                  f"(kind {rec['kind_to']}) en {rec['energy'][0]:.0f}->{rec['energy'][1]:.0f} "
                  f"coin {rec['coin'][0]:.0f}->{rec['coin'][1]:.0f} hp {rec['hp'][0]:.0f}->{rec['hp'][1]:.0f}", flush=True)
    return records


def rules(seed: int, turns: int) -> dict:
    game = fresh(seed)
    records = play(game, turns)
    s = game.session
    kinds = game.board.kinds
    out = {"turns": len(records), "rounds": int(s.column("rnd")[0]), "duels": float(s.column("duels")[0]),
           "trades": float(s.column("trades")[0])}
    # allowance: never beyond STEPS; a rest spends none; moved never beyond allowance
    out["allowance_ok"] = all(r["moved"] <= r["allow"] <= bw.STEPS and (r["choice"] != "rest" or r["allow"] == 0) for r in records)
    out["choices"] = {c: sum(1 for r in records if r["choice"] == c) for c in TURN_WORD.values()}
    out["avg_moved_by_choice"] = {c: round(float(np.mean([r["moved"] for r in records if r["choice"] == c])), 2)
                                  for c in TURN_WORD.values() if any(r["choice"] == c for r in records)}
    # bog: stepping onto a bog tile costs double energy against the walk's plain steps
    out["forks_a"] = {NAMES[k]: int(s.column("forka")[k]) for k in range(bw.ENT)}
    out["forks_b"] = {NAMES[k]: int(s.column("forkb")[k]) for k in range(bw.ENT)}
    out["laps"] = [int(v) for v in s.column("laps")[:bw.ENT]]
    out["stopped_early"] = sum(1 for r in records if 0 < r["moved"] < r["allow"] and r["kind_to"] in bw.STOPPING)
    out["treasure_landings"] = sum(1 for r in records if r["kind_to"] == bw.TREASURE and r["coin"][1] > r["coin"][0])
    out["market_rests"] = sum(1 for r in records if r["kind_to"] == bw.MARKET and r["choice"] == "rest" and r["moved"] == 0)
    return out


def put_player(game: bw.Boardwalk, tile: int) -> None:
    state = game.session.state
    state.node[0] = float(tile)
    state.lastnode[0] = float(tile)
    state.x[0] = game.board.pos[tile][0] + 0.5 + float(game.session.column("jx")[0])
    state.y[0] = game.board.pos[tile][1] + 0.5 + float(game.session.column("jy")[0])


def pad(seed: int) -> dict:
    """The player's pad against the network's own turn, same seed and dice, from
    tile 2: one step short of the fork whose branches are tile 4 (the ring) and
    tile 34 (the shortcut)."""
    result = {}
    for label, controls in (("network", {}), ("fork-B", {"move_axis": 1.0}), ("fork-A", {"move_axis": -1.0}),
                            ("run", {"move_axis_y": -1.0}), ("rest", {"move_axis_y": 1.0})):
        game = fresh(seed)
        put_player(game, 2)
        records = [r for r in play(game, 9, controls, log=False) if r["lane"] == 0]
        result[label] = {"choice": records[0]["choice"], "roll": records[0]["roll"], "allow": records[0]["allow"],
                         "moved": records[0]["moved"], "to": records[0]["to"],
                         "forks_b": int(game.session.column("forkb")[0])}
    return result


def main(argv: list[str]) -> int:
    seed = int(argv[1]) if len(argv) > 1 else 3
    turns = int(argv[2]) if len(argv) > 2 else 120
    started = time.perf_counter()
    result = rules(seed, turns)
    print(f"rules    {result}  [{time.perf_counter() - started:.1f}s wall]", flush=True)
    started = time.perf_counter()
    pads = pad(seed)
    print(f"pad      {pads}  [{time.perf_counter() - started:.1f}s wall]", flush=True)
    ok = (result["allowance_ok"] and result["turns"] >= turns and sum(result["forks_b"].values()) > 0
          and sum(result["forks_a"].values()) > 0 and pads["rest"]["moved"] == 0 and pads["run"]["allow"] > pads["network"]["allow"]
          and pads["fork-A"]["to"] != pads["fork-B"]["to"] and pads["fork-B"]["forks_b"] == 1 and pads["fork-A"]["forks_b"] == 0
          and result["duels"] + result["trades"] > 0)
    print(f"boardwalk {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
