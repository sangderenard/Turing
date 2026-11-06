"""Fit the Meowdoku enemy brains (``examples/meowdoku/scroller.py: enemy_brain``)
to the hand-written decisions they replaced.

The hand-written rule is kept HERE as the reference (``hand``), written from
the same inputs the brain receives.  Samples are (a) lane states recorded
from the real law running eagerly -- every ``enemy_brain`` call is spied --
and (b) a sweep that covers the cases play rarely reaches (UFOs beside lit
pins, planters at a town).  The sparse weights ``BRAIN_W`` (masked by
``BRAIN_MASK``) are tuned by coordinate descent on turn / speed error and on
fire / hover / land disagreement, then written between the BRAIN markers in
scroller.py.

    python tools/meowdoku_brain_fit.py [--ticks N] [--write]
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "examples", Path(r"C:\dev\Powershell\engine_toy")):
    if str(p) not in sys.path:
        sys.path.append(str(p))

import src.common.tensors.numpy_backend  # noqa: E402,F401
from src.common.tensors import AbstractTensor  # noqa: E402
from meowdoku import scroller as sc  # noqa: E402

NAMES = ("kind", "bp", "bm", "bg", "bt", "seen_abs", "dist", "face_dot", "glow_gap", "glow_has", "town_gap", "town_has")


def hand(s):
    """The decisions the law used to make inline (turn, speed at tempo 1, fire, hover, land)."""
    kind = s["kind"]
    spinner, planter = kind == sc.SPINNER, kind == sc.PLANTER
    facing = s["face_dot"] < -0.3
    head_on = (s["seen_abs"] < 0.9) & ((s["dist"] > sc.TURN_IN) | facing)
    want = np.where(head_on, s["bp"], s["bm"])
    abducting = spinner & (s["glow_has"] > 0) & (s["glow_gap"] < sc.UFO_SENSE)
    want = np.where(abducting, s["bg"], want)
    hovering = abducting & (s["glow_gap"] < sc.ABDUCT_R)
    planting = planter & (s["town_has"] > 0)
    want = np.where(planting, s["bt"], want)
    landing = planting & (s["town_gap"] < sc.LAND_R)
    passed = (s["dist"] < sc.PASS_R) & (np.abs(s["bp"]) > 1.6)
    agility = np.where(spinner, sc.SPINNER_TURN, sc.DRIFTER_TURN)
    turn = np.where(passed & ~abducting & ~planting, 0.0, np.clip(sc.HUNT_GAIN * want, -1.0, 1.0) * agility)
    tempo_term = 0.8 + 0.4        # tempo >= 1
    spd = np.where(spinner, sc.SPINNER_SPD, sc.DRIFTER_SPD) * tempo_term * np.where(facing, 1.0, sc.BEHIND_BOOST)
    spd = np.where(hovering, sc.HOVER_SPD, np.where(abducting, sc.SPINNER_SPD, np.where(planting, sc.PLANTER_SPD, spd)))
    in_sights = (np.abs(s["bp"]) < sc.SIGHT) & (s["dist"] < sc.FIRE_RANGE)
    fire = (kind != sc.PLANTER) & np.where(spinner, s["dist"] < sc.RING_RANGE, in_sights)
    return turn, spd, fire, hovering, landing


def net(s):
    """The brain on numpy samples (tempo 1): turn, speed, fire, hover, land."""
    args = [AbstractTensor.get_tensor(np.asarray(s[n], dtype=np.float64)) for n in NAMES]
    args[0] = args[0]
    # glow_has / town_has are booleans in the law
    args[9] = args[9] > 0.5
    args[11] = args[11] > 0.5
    turn, cruise, role, fire, hover, land = sc.enemy_brain(*args)
    g = lambda t: np.asarray(t.numpy(), dtype=np.float64)
    return g(turn), g(cruise) * (0.8 + 0.4) + g(role), g(fire) > 0.5, g(hover) > 0.5, g(land) > 0.5


def record(ticks):
    """Inputs of every enemy_brain call while the real law runs eagerly."""
    rows = []
    original = sc.enemy_brain

    def spy(*a):
        vals = {n: np.asarray(v.numpy()) for n, v in zip(NAMES, a)}
        live = vals["kind"] > 0
        rows.append({n: v[sc.ENEMIES[0]:sc.ENEMIES[1]].astype(np.float64) for n, v in vals.items()})
        return original(*a)

    sc.enemy_brain = spy
    try:
        g = sc.BulletScroller("fit")
        g.session.piece.product = None
        g.control({}); g.tick()
        g.set_autopilot(True)
        for _ in range(ticks):
            g.control({}); g.tick()
    finally:
        sc.enemy_brain = original
    return {n: np.concatenate([r[n] for r in rows]) for n in NAMES}


def sweep(n, seed=7):
    rng = np.random.default_rng(seed)
    near = lambda: np.where(rng.random(n) < 0.5, rng.uniform(0, 3, n), rng.uniform(0, 40, n))
    return {
        "kind": rng.choice([sc.DRIFTER, sc.SPINNER, sc.PLANTER, sc.ACE, sc.BOSS_KIND], n).astype(np.float64),
        "bp": rng.uniform(-math.pi, math.pi, n), "bm": rng.uniform(-math.pi, math.pi, n),
        "bg": rng.uniform(-math.pi, math.pi, n), "bt": rng.uniform(-math.pi, math.pi, n),
        "seen_abs": rng.uniform(0, math.pi, n),
        "dist": np.where(rng.random(n) < 0.6, rng.uniform(0, 12, n), rng.uniform(0, 40, n)),
        "face_dot": rng.uniform(-1, 1, n),
        "glow_gap": near(), "glow_has": (rng.random(n) < 0.5).astype(np.float64),
        "town_gap": near(), "town_has": (rng.random(n) < 0.6).astype(np.float64),
    }


def pick(s, mask):
    return {n: v[mask] for n, v in s.items()}


def score(s):
    """Per-kind quality of the brain against the hand rule."""
    ht, hs, hf, hh, hl = hand(s)
    nt, ns, nf, nh, nl = net(s)
    return {
        "n": int(len(ht)),
        "turn_mae": float(np.mean(np.abs(nt - ht))), "turn_bad": float(np.mean(np.abs(nt - ht) > 0.05)),
        "spd_mae": float(np.mean(np.abs(ns - hs))), "spd_bad": float(np.mean(np.abs(ns - hs) > 0.05)),
        "fire_agree": float(np.mean(nf == hf)), "hover_agree": float(np.mean(nh == hh)), "land_agree": float(np.mean(nl == hl)),
    }


def loss(s):
    ht, hs, hf, hh, hl = hand(s)
    nt, ns, nf, nh, nl = net(s)
    return (np.mean((nt - ht) ** 2) + np.mean((ns - hs) ** 2) / 10.0
            + np.mean(nf != hf) + np.mean(nh != hh) + np.mean(nl != hl))


KIND_ROWS = (sc.DRIFTER, sc.SPINNER, sc.PLANTER)


def fit(samples, rounds=1):
    grid = np.concatenate([np.linspace(0.7, 1.4, 15)])
    for _ in range(rounds):
        for row in range(4):
            sel = (samples["kind"] == KIND_ROWS[row]) if row < 3 else ~np.isin(samples["kind"], KIND_ROWS)
            part = pick(samples, sel)
            if not len(part["kind"]):
                continue
            for j, name in enumerate(sc.BRAIN_PARAMS):
                if not sc.BRAIN_MASK[row][j] or name in ("agility", "cruise"):   # exact physical gains stay
                    continue
                base = sc.BRAIN_W[row][j]
                best = (loss(part), base)
                for m in grid:
                    sc.BRAIN_W[row][j] = base * m
                    value = loss(part)
                    if value < best[0] - 1e-12:
                        best = (value, base * m)
                sc.BRAIN_W[row][j] = best[1]


def report(tag, recorded, swept):
    print(f"--- {tag}")
    for label, data in (("recorded", recorded), ("sweep", swept)):
        for row, name in enumerate(("drifter", "spinner", "planter", "ace/boss")):
            sel = (data["kind"] == KIND_ROWS[row]) if row < 3 else ~np.isin(data["kind"], KIND_ROWS) & (data["kind"] > 0)
            if sel.sum() == 0:
                continue
            q = score(pick(data, sel))
            if row == 3:
                print(f"{label:9s} {name:9s} n={q['n']:6d} fire_agree={q['fire_agree']:.4f}")
            else:
                print(f"{label:9s} {name:9s} n={q['n']:6d} turn MAE={q['turn_mae']:.4f} (>0.05: {q['turn_bad']:.4f}) "
                      f"speed MAE={q['spd_mae']:.4f} (>0.05: {q['spd_bad']:.4f}) fire={q['fire_agree']:.4f} "
                      f"hover={q['hover_agree']:.4f} land={q['land_agree']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--sweep", type=int, default=20000)
    args = ap.parse_args()
    recorded = record(args.ticks)
    live = recorded["kind"] > 0
    recorded = {n: v[live] for n, v in recorded.items()}
    print(f"recorded {len(recorded['kind'])} lane-states; kinds:",
          {int(k): int((recorded['kind'] == k).sum()) for k in np.unique(recorded['kind'])})
    swept = sweep(args.sweep)
    held = sweep(args.sweep, seed=99)           # held-out sweep, never fitted
    report("analytic initialisation", recorded, held)
    pool = {n: np.concatenate([recorded[n], swept[n]]) for n in NAMES}
    fit(pool)
    report("after fit (recorded + sweep fitted; sweep rows below are held-out)", recorded, held)
    block = "BRAIN_W = [\n" + "".join("    [" + ", ".join(f"{w:.6g}" for w in row) + "],\n" for row in sc.BRAIN_W) + "]\n"
    print(block)
    if args.write:
        path = ROOT / "examples" / "meowdoku" / "scroller.py"
        text = path.read_text(encoding="utf-8")
        new = re.sub(r"(# --- BRAIN begin[^\n]*\n).*?(# --- BRAIN end)", lambda m: m.group(1) + block + m.group(2), text, flags=re.S)
        path.write_text(new, encoding="utf-8")
        print("written")


if __name__ == "__main__":
    main()
