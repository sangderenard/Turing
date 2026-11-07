"""Parity of the invaders law: the NumPy product against the native (LLVM) compile.

    python examples/meowdoku/invaders_parity.py [--level 6] [--seconds 25] [--lane eigen]

Plays the real game on the NumPy product (the dodger at the stick, the level
set so attack waves are in flight), and at every decision step runs BOTH
products on the state exactly as the dt system would hand it to the law, over
several step sizes, comparing every output column (max absolute difference).
It reports how many of the compared states had waves in flight, wave bullets
alive, a telegraphing ship and a volley leaving, so that "parity on real
states, including active waves" is read from the run itself.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), str(ROOT.parent / "engine_toy")]

import numpy as np  # noqa: E402

from meowdoku import entities
from meowdoku.s2_titles.invaders import invaders as inv  # noqa: E402


def piece_on(lane: str, columns: dict) -> inv.InvadersPiece:
    entities.LAW_LANE = lane
    piece = inv.InvadersPiece()
    piece.compile(columns)
    return piece


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--lane", default="eigen")
    args = parser.parse_args()

    columns = inv.initial_columns()
    columns["dt"] = np.full(inv.LANES, 1.0 / 60.0)
    numpy_piece = piece_on("numpy", columns)
    started = time.perf_counter()
    native_piece = piece_on(args.lane, columns)
    print(f"native piece lane={native_piece.lane} (built/loaded in {time.perf_counter() - started:.0f}s)", flush=True)
    if native_piece.lane == "numpy":
        print("FAIL: the native lane did not build (fell back to numpy)")
        return 1

    entities.LAW_LANE = "numpy"
    game = inv.SpaceInvaders("parity", seed=3.0)
    col = game.session.column
    seq = [0]

    def advance(controls):
        seq[0] += 1
        game.control({"inputs": [{"seq": seq[0], "dur": 1.0 / 30.0, "steer_x": -1.0, **controls}]})
        game.tick()

    advance({"move_axis": 0.0, "fire": 0.0})
    col("wave")[...] = float(args.level)
    col("w_clock")[...] = 0.0
    from meowdoku.s3_train import autopilot as ap
    worst = {name: 0.0 for name in native_piece.output_names}
    seen = {"states": 0, "waves in flight": 0, "wave bullets alive": 0, "telegraphing": 0, "volley leaving": 0, "wave launching": 0}
    for _ in range(int(args.seconds * 30)):
        c = ap.dodger_controls(game)
        advance({"move_axis": c[0], "fire": c[2]})
        col("lives")[...] = inv.LIVES
        for dt in (1.0 / 240.0, 1.0 / 60.0, 1.0 / 30.0):
            args_in = [np.full(inv.LANES, dt) if name == "dt" else np.array(col(name), dtype=np.float64)
                       for name in numpy_piece.argument_names]
            a = numpy_piece(*[x.copy() for x in args_in])
            b = native_piece(*[x.copy() for x in args_in])
            for name, u, v in zip(native_piece.output_names, a, b):
                worst[name] = max(worst[name], float(np.max(np.abs(np.asarray(u) - np.asarray(v)))))
            alive = np.array(col("alive"))
            seen["states"] += 1
            seen["waves in flight"] += int(alive[inv.WAVE0:inv.WBUL0].sum() > 0)
            seen["wave bullets alive"] += int(alive[inv.WBUL0:].sum() > 0)
            seen["telegraphing"] += int(np.array(col("tele")).sum() > 0)
            seen["volley leaving"] += int(a[STATE_INDEX["alive"]][inv.WBUL0:].sum() > alive[inv.WBUL0:].sum())
            seen["wave launching"] += int(a[STATE_INDEX["alive"]][inv.WAVE0:inv.WBUL0].sum() > alive[inv.WAVE0:inv.WBUL0].sum())
    print("coverage:", seen)
    for name, d in worst.items():
        print(f"  {name:12s} max |numpy - native| = {d:.3e}")
    limit = max(worst.values())
    print("PARITY", "OK" if limit <= 1e-9 else f"DIFFERS (max {limit:.3e})")
    return 0 if limit <= 1e-9 else 1


STATE_INDEX = {name: i for i, name in enumerate(inv.STATE)}

if __name__ == "__main__":
    raise SystemExit(main())
