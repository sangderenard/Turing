"""Headless probe of the invaders law's attack waves, on the real laws.

    python examples/meowdoku/invaders_probe.py

Checks, against the law's own declared timeline and limits (invaders.py):

* level 1 is the classic block alone (no wave ever launches, the block marches);
* later levels launch formations on schedule, in the level's declared numbers
  (formations at once, ships per formation, rows of the block that remain);
* every wave ship enters, fires its pattern's volleys at the declared times
  (each one telegraphed first) and leaves, on its wave clock;
* bullet counts and speeds stay inside the simplicity limits (the pool, the
  speed ceiling, the shots per ship per volley);
* the later levels are mostly waves.

The player is held at rest and never loses (lives are topped up each step) so
that the schedule can be read; the law is otherwise the one the game runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

import numpy as np  # noqa: E402

from meowdoku.s2_titles.invaders import invaders as inv  # noqa: E402

STEP = 1.0 / 30.0


class Run:
    """One game driven at a fixed decision step with a held control."""

    def __init__(self, level: int, seed: float = 3.0, fire: float = 0.0):
        self.game = inv.SpaceInvaders("probe", seed=seed)
        self.col = self.game.session.column
        self.seq = 0
        self.fire = fire
        self.step({})
        self.col("wave")[...] = float(level)
        self.col("w_clock")[...] = 0.0          # launch at once: the probe reads the schedule, not the first delay

    def step(self, controls: dict) -> None:
        self.seq += 1
        self.game.control({"inputs": [{"seq": self.seq, "dur": STEP, "move_axis": 0.0, "steer_x": -1.0,
                                       "fire": self.fire, **controls}]})
        self.game.tick()
        if "lives" in self.game.session.columns:
            self.col("lives")[...] = inv.LIVES
            self.col("energy")[0] = 1.0


def observe(level: int, seconds: float, fire: float = 0.0, seed: float = 3.0) -> dict:
    run = Run(level, seed, fire)
    col = run.col
    lanes_w = range(*inv.WAVE_LANES)
    lanes_b = range(*inv.WBULLET_LANES)
    launched: dict[int, float] = {}
    pat_first: dict[int, float] = {}
    left: dict[int, float] = {}
    births: dict[int, list[float]] = {k: [] for k in lanes_w}
    seen_born = {k: -1.0 for k in lanes_b}
    tele_at: dict[int, list[float]] = {k: [] for k in lanes_w}
    peak = {"bullets": 0, "ships": 0, "speed": 0.0, "groups": 0}
    shots_at: dict[tuple[int, float], int] = {}
    fleet_x0 = None
    fleet_moved = 0.0
    wave_time = 0.0
    prev_alive = {k: False for k in lanes_w}
    max_ship_speed = 0.0
    while wave_time < seconds:
        run.step({})
        clock = float(col("clock")[0])
        wave_time = clock
        alive = col("alive")
        x, y, vx, vy, born, owner = col("x"), col("y"), col("vx"), col("vy"), col("born"), col("owner")
        for k in lanes_w:
            a = alive[k] > 0.0
            if a and not prev_alive[k]:
                if k not in launched:
                    launched[k] = clock
                    pat_first[k] = float(col("w_pat")[k])
            if (not a) and prev_alive[k] and col("boom")[k] <= 0.0 and k in launched:
                left.setdefault(k, clock)
            prev_alive[k] = a
            if a and col("tele")[k] > 0.5:
                tele_at[k].append(clock)
            if a:
                max_ship_speed = max(max_ship_speed, float(np.hypot(vx[k], vy[k])))
        for k in lanes_b:
            if alive[k] > 0.0:
                if born[k] != seen_born[k]:
                    seen_born[k] = float(born[k])
                    o = int(owner[k])
                    births[o].append(float(born[k]))
                    shots_at[(o, round(float(born[k]), 3))] = shots_at.get((o, round(float(born[k]), 3)), 0) + 1
                peak["speed"] = max(peak["speed"], float(np.hypot(vx[k], vy[k])))
        peak["bullets"] = max(peak["bullets"], int((alive[inv.WBUL0:] > 0.0).sum()))
        peak["ships"] = max(peak["ships"], int((alive[inv.WAVE0:inv.WBUL0] > 0.0).sum()))
        groups_now = len({int(col("w_group")[k]) for k in lanes_w if alive[k] > 0.0})
        peak["groups"] = max(peak["groups"], groups_now)
        fleet = np.flatnonzero(alive[inv.FLEET0:inv.SHIELD0] > 0.0) + inv.FLEET0
        if len(fleet):
            here = (float(x[fleet].mean()), float(y[fleet].mean()))
            if fleet_x0 is None:
                fleet_x0 = here
            fleet_moved = max(fleet_moved, abs(here[0] - fleet_x0[0]) + abs(here[1] - fleet_x0[1]))
    return {"launched": launched, "left": left, "births": births, "tele": tele_at, "peak": peak,
            "shots_at": shots_at, "fleet_moved": fleet_moved, "max_ship_speed": max_ship_speed,
            "pattern": pat_first, "hold": {k: float(col("w_hold")[k]) for k in lanes_w},
            "rows": int((col("alive")[inv.FLEET0:inv.SHIELD0] > 0.0).sum() // inv.FLEET_COLS),
            "alive_fleet": int((col("alive")[inv.FLEET0:inv.SHIELD0] > 0.0).sum())}


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    return ok


def main() -> int:
    good = True
    print("level 1 (classic only):")
    r = observe(0, 60.0)
    good &= check("no wave ship ever launched", not r["launched"] and r["peak"]["ships"] == 0)
    good &= check("no wave bullets", r["peak"]["bullets"] == 0)
    good &= check("the block marches", r["fleet_moved"] > 0.5, f"block centre moved {r['fleet_moved']:.2f}")
    good &= check("the block is all five rows", r["alive_fleet"] == 55)

    print("later levels (wave schedule against the declared timeline):")
    summary = []
    for level in range(1, len(inv.LEVELS) + 1):
        cfg = dict(zip(inv.LEVEL_COLUMNS, inv.LEVELS[min(level, len(inv.LEVELS) - 1)]))
        r = observe(level, 40.0)
        launched, left, births = r["launched"], r["left"], r["births"]
        ok_level = True
        ok_level &= check(f"L{level + 1}: waves launch", len(launched) > 0, f"{len(launched)} ship launches")
        ok_level &= check(f"L{level + 1}: formations/ships within the level",
                          r["peak"]["groups"] <= cfg["groups"] and r["peak"]["ships"] <= cfg["groups"] * cfg["ships"],
                          f"peak groups {r['peak']['groups']}/{int(cfg['groups'])}, ships {r['peak']['ships']}/{int(cfg['groups'] * cfg['ships'])}")
        # every ship that left did so on its wave clock; every ship that launched early enough left
        bad_exit = []
        for k, t0 in launched.items():
            hold = r["hold"][k]
            expect = t0 + inv.WAVE_ENTER + hold + inv.WAVE_EXIT + ((k - inv.WAVE0) % inv.WAVE_SIZE) * inv.WAVE_LAG
            if k in left and abs(left[k] - expect) > 3.0 * STEP:
                bad_exit.append((k, round(left[k] - expect, 3)))
        ok_level &= check(f"L{level + 1}: ships exit on schedule", not bad_exit and len(left) > 0, f"{len(left)} exits, off by {bad_exit[:3]}")
        # volleys: times = launch + lag + enter + first + k * every ; telegraphed first
        bad_fire, bad_tele, volleys_seen = [], [], 0
        for k, t0 in launched.items():
            lag = ((k - inv.WAVE0) % inv.WAVE_SIZE) * inv.WAVE_LAG
            hold = r["hold"][k]
            n_vol = int((hold - 0.6 - inv.VOLLEY_FIRST) / inv.VOLLEY_EVERY + 0.5)
            window_end = t0 + lag + inv.WAVE_ENTER + hold + inv.WAVE_EXIT + STEP
            times = sorted({round(b, 3) for b in births[k] if t0 <= b <= window_end})
            expected = [t0 + lag + inv.WAVE_ENTER + inv.VOLLEY_FIRST + i * inv.VOLLEY_EVERY for i in range(n_vol)]
            if k in left or t0 + lag + inv.WAVE_ENTER + hold < 40.0 - 1.0:
                firing = r["pattern"][k]
                # ring/spiral fire from leaders only, alternating from half the ship each volley: a ship may skip volleys
                if firing in (0.0, 1.0):
                    if len(times) != n_vol:
                        bad_fire.append((k, len(times), n_vol))
                for t in times:
                    volleys_seen += 1
                    if min(abs(t - e) for e in expected) > 2.5 * STEP:
                        bad_fire.append((k, t))
                    window = [c for c in r["tele"][k] if t - inv.TELEGRAPH - 2 * STEP <= c < t]
                    if len(window) < 0.6 * inv.TELEGRAPH / STEP:
                        bad_tele.append((k, t, len(window)))
        ok_level &= check(f"L{level + 1}: volleys at the declared times", not bad_fire and volleys_seen > 0,
                          f"{volleys_seen} volleys; off: {bad_fire[:3]}")
        ok_level &= check(f"L{level + 1}: every volley telegraphed first", not bad_tele, f"missing: {bad_tele[:3]}")
        most = max(r["shots_at"].values()) if r["shots_at"] else 0
        ok_level &= check(f"L{level + 1}: simplicity limits", r["peak"]["bullets"] <= inv.WBULLETS and r["peak"]["speed"] <= inv.MAX_BULLET_SPEED + 1e-6
                          and most <= inv.MAX_SHOTS_PER_EVENT,
                          f"peak bullets {r['peak']['bullets']}/{inv.WBULLETS}, speed {r['peak']['speed']:.2f}/{inv.MAX_BULLET_SPEED}, "
                          f"most shots in one volley {most}/{inv.MAX_SHOTS_PER_EVENT}")
        patterns = sorted({int(p) for k, p in r["pattern"].items()})
        summary.append((level + 1, cfg, r["peak"], patterns, r["max_ship_speed"]))
        good &= ok_level
    print("summary (level: peak ships, peak bullets, patterns drawn, fastest ship):")
    for level, cfg, peak, patterns, fastest in summary:
        print(f"  L{level}: rows {int(cfg['rows'])} ships {peak['ships']} bullets {peak['bullets']} patterns {patterns} ship speed {fastest:.1f}")
    late = [s for s in summary if s[0] >= len(inv.LEVELS)]
    good &= check("the last levels are mostly waves (one classic row, three formations)",
                  bool(late) and late[0][1]["rows"] == 1 and late[0][2]["ships"] >= 12)
    print("PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
