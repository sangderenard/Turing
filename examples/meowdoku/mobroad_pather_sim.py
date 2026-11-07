"""Mobroad's pather, scored and trained OFFLINE on a host-side crowd (NumPy; no law is run, so a candidate costs milliseconds, not the eager law's 0.35 s/step).

The pather is ``fountain.mob_pather_accel`` (a few dozen weights, tanh, a walking acceleration capped at ``MOB_WALK``).  This file does not replace it: ``forward``
is the same arithmetic over a batch of weight sets (a test holds it equal to the law's own function), and the crowd around it is the minimum that makes arrival
a CROWD question:
    the ROAD     the law's own ``sdf_field`` (``fountain.chamber_distance``) sampled once on a grid: the same pillars, jaws, funnel barriers and pegs;
    the WALK     the law's own integrator for a free grain: implicit ground friction (``fl_mu`` Stokes drag, ``kdrag``) plus the species' linear drag, the pather's
                 acceleration as the only force (gravity is orthogonal to the scene), at a coarser step (``DT_SIM``);
    the CROWD    grains do not overlap each other or a wall (position-based: overlaps pushed apart, the velocity is what the step really moved);
    the GOAL     a grain within ``MOB_ATTACK_R`` of the player has arrived (and is removed).
Score = the fraction of the crowd that has arrived at the player by the horizon (plus a small progress term to give the search a slope).

Train:  ``python -m meowdoku.mobroad_pather_sim --train`` (from ``examples``) writes ``mobroad_pather.json`` (a separable-covariance evolution strategy over the
weights starting from the hand-set ones).  The weights are RUNTIME UNIFORMS of the law: nothing recompiles.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from meowdoku.s2_titles.fountain import fountain as F

DT_SIM = 1.0 / 60.0
RADIUS = F.SPECIES[0][1]
KDRAG = 4.5 * F.MOB_MARCH["fl_mu"] / (F.SPECIES[0][2] * RADIUS * RADIUS)        # the law's implicit fluid drag (species 0, density 1)
DRAG = F.SPECIES[0][4]                                                           # ... and the species' own linear drag
VMAX = 0.5
GRID = 0.004
OUT = pathlib.Path(__file__).with_name("mobroad_pather.json")


class Road:
    """The road's SDF (the law's ``sdf_field``) on a grid: ``(d, nx, ny)`` by bilinear lookup."""

    def __init__(self, h: float = GRID):
        w, top = F.MOB_ROAD_HALF, F.MOB_ROAD_TOP
        self.h, self.x0 = h, -w - 4 * h
        self.xs = np.arange(self.x0, w + 5 * h, h)
        self.ys = np.arange(-4 * h, top + 5 * h, h)
        u = F.mobroad_uniforms(256)
        gx, gy = np.meshgrid(self.xs, self.ys)
        d, nx, ny = F.chamber_distance(u, gx.ravel(), gy.ravel())
        shape = gx.shape
        self.d, self.nx, self.ny = d.reshape(shape), nx.reshape(shape), ny.reshape(shape)

    def at(self, x, y):
        fx = np.clip((x - self.x0) / self.h, 0.0, len(self.xs) - 1.001)
        fy = np.clip((y - self.ys[0]) / self.h, 0.0, len(self.ys) - 1.001)
        i, j = fx.astype(np.int64), fy.astype(np.int64)
        tx, ty = fx - i, fy - j

        def bil(a):
            return (a[j, i] * (1 - tx) + a[j, i + 1] * tx) * (1 - ty) + (a[j + 1, i] * (1 - tx) + a[j + 1, i + 1] * tx) * ty

        nx, ny = bil(self.nx), bil(self.ny)
        inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1e-12)
        return bil(self.d), nx * inv, ny * inv


_ROAD: Road | None = None


def road() -> Road:
    global _ROAD
    if _ROAD is None:
        _ROAD = Road()
    return _ROAD


def forward(w, x, y, vx, vy, d, nx, ny, tx, ty):
    """``fountain.mob_pather_accel`` over a batch: ``w`` (B, MOB_PW), the rest (B, N) (``tx, ty`` (B, 1))."""
    nin, nh = F.MOB_PATHER_IN, F.MOB_PATHER_H
    dx, dy = tx - x, ty - y
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    ux, uy = dx / dist, dy / dist
    feats = np.stack([ux * nx + uy * ny, uy * nx - ux * ny, (vx * nx + vy * ny) / F.MOB_VSCALE, (vy * nx - vx * ny) / F.MOB_VSCALE,
                      np.clip(d, 0.0, 2 * F.MOB_DSCALE) / F.MOB_DSCALE, np.clip(dist / F.MOB_ROAD_TOP, 0.0, 1.0)], axis=-1)           # (B, N, 6)
    w1 = w[:, : nh * nin].reshape(-1, nh, nin)
    b1 = w[:, nh * nin: nh * nin + nh]
    base = nh * nin + nh
    w2 = w[:, base: base + 2 * nh].reshape(-1, 2, nh)
    b2 = w[:, base + 2 * nh: base + 2 * nh + 2]
    hid = np.tanh(np.einsum("bni,bhi->bnh", feats, w1) + b1[:, None, :])
    o = np.einsum("bnh,bkh->bnk", hid, w2) + b2[:, None, :]
    o_n, o_t = o[..., 0], o[..., 1]
    scale = F.MOB_WALK / np.clip(np.sqrt(o_n * o_n + o_t * o_t + 1e-12), 1.0, 1e9)
    a_n, a_t = o_n * scale, o_t * scale
    return a_n * nx - a_t * ny, a_n * ny + a_t * nx


def simulate(w, targets, n: int = 96, horizon: float = 18.0, seed: int = 0, spawn: float = 2.0, start=None, rd: Road | None = None):
    """Run ``len(w)`` weight sets, each on every target in ``targets`` (a list of player x), one crowd of ``n`` grains per (set, target) entering across the
    road's top over ``spawn`` seconds (the same entries for every set).  ``start``: ``(x, y)`` per grain instead (a lone grain test).  Returns
    ``{"arrived": (S, T), "progress": (S, T), "t_arrive": (S, T), "track": x, y of the first grain over time}``."""
    rd = rd or road()
    w = np.asarray(w, dtype=np.float64).reshape(-1, F.MOB_PW)
    s_n, t_n = len(w), len(targets)
    bsz = s_n * t_n
    wb = np.repeat(w, t_n, axis=0)
    tx = np.tile(np.asarray(targets, dtype=np.float64), s_n)[:, None]
    ty = np.full((bsz, 1), F.MOB_PLAYER_Y)
    rng = np.random.default_rng(seed)
    if start is None:
        sx = rng.uniform(-F.MOB_BAR_HALF, F.MOB_BAR_HALF, n)
        sy = np.full(n, F.MOB_NOZZLE_Y)
        t_spawn = np.sort(rng.uniform(0.0, spawn, n))
    else:
        sx, sy = np.atleast_1d(start[0]).astype(np.float64), np.atleast_1d(start[1]).astype(np.float64)
        n, t_spawn = len(sx), np.zeros(len(sx))
    x = np.tile(sx, (bsz, 1))
    y = np.tile(sy, (bsz, 1))
    vx, vy = np.zeros((bsz, n)), np.full((bsz, n), -0.06 if start is None else 0.0)
    live = np.zeros((bsz, n), dtype=bool)
    done = np.zeros((bsz, n), dtype=bool)
    t_arr = np.full((bsz, n), np.inf)
    d0 = np.sqrt((x - tx) ** 2 + (y - ty) ** 2)
    eye = np.eye(n, dtype=bool)[None]
    track = []
    steps = int(round(horizon / DT_SIM))
    ar2 = F.MOB_ATTACK_R ** 2
    for step in range(steps):
        t = step * DT_SIM
        live |= (t_spawn[None, :] <= t) & ~done
        d, nx, ny = rd.at(x, y)
        ax, ay = forward(wb, x, y, vx, vy, d, nx, ny, tx, ty)
        vxn = (vx + (ax - DRAG * vx) * DT_SIM) / (1.0 + KDRAG * DT_SIM)
        vyn = (vy + (ay - DRAG * vy) * DT_SIM) / (1.0 + KDRAG * DT_SIM)
        xo, yo = x.copy(), y.copy()
        x = np.where(live, x + vxn * DT_SIM, x)
        y = np.where(live, y + vyn * DT_SIM, y)
        for _ in range(2):
            if n > 1:
                ddx, ddy = x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :]
                dd = np.sqrt(ddx * ddx + ddy * ddy + 1e-12)
                pair = live[:, :, None] & live[:, None, :] & ~eye
                ov = np.where(pair, np.maximum(0.0, 2 * RADIUS - dd), 0.0) * 0.5 / dd
                x = x + (ov * ddx).sum(axis=2)
                y = y + (ov * ddy).sum(axis=2)
            d2, mx, my = rd.at(x, y)
            pen = np.where(live, np.maximum(0.0, RADIUS - d2), 0.0)
            x, y = x + pen * mx, y + pen * my
        vx, vy = (x - xo) / DT_SIM, (y - yo) / DT_SIM
        sp = np.sqrt(vx * vx + vy * vy + 1e-12)
        cap = np.minimum(1.0, VMAX / sp)
        vx, vy = vx * cap, vy * cap
        hit = live & (((x - tx) ** 2 + (y - ty) ** 2) < ar2)
        t_arr = np.where(hit, t, t_arr)
        done |= hit
        live &= ~hit
        if start is not None:
            track.append((x[:, 0].copy(), y[:, 0].copy()))
        if step % 30 == 0 and bool(done.all()):
            break
    dist = np.sqrt((x - tx) ** 2 + (y - ty) ** 2)
    prog = np.where(done, 1.0, np.clip(1.0 - dist / d0, 0.0, 1.0))
    shape = (s_n, t_n)
    return {"arrived": done.mean(axis=1).reshape(shape), "progress": prog.mean(axis=1).reshape(shape), "t_arrive": t_arr.reshape(s_n, t_n, n), "track": track}


def score(w, targets, n=96, horizon=18.0, seed=0):
    w = np.asarray(w).reshape(-1, F.MOB_PW)
    out = []
    for i in range(0, len(w), 6):                                           # (chunks: the pair tensors are batch x n x n)
        r = simulate(w[i:i + 6], targets, n, horizon, seed)
        out.append(r["arrived"].mean(axis=1) + 0.05 * r["progress"].mean(axis=1))
    return np.concatenate(out)


HELD_OUT = (-0.5, -0.25, 0.0, 0.25, 0.5)


def arrival_rate(w, n=96, horizon=18.0, seeds=(101, 202), targets=HELD_OUT) -> float:
    """The fraction of the crowd that reaches the player, mean over the held-out player places and seeds."""
    return float(np.mean([simulate([w], targets, n, horizon, s)["arrived"].mean() for s in seeds]))


def train(generations: int = 30, pop: int = 24, k_targets: int = 3, n: int = 96, horizon: float = 18.0, seed: int = 7, sigma0: float = 0.35):
    hand = np.array(F.mob_pather_hand())
    rng = np.random.default_rng(seed)
    mean, sig = hand.copy(), np.full(len(hand), sigma0)
    best, best_f = hand.copy(), -1.0
    mu = pop // 4
    for g in range(generations):
        started = time.time()
        targets = list(rng.uniform(-0.55, 0.55, k_targets))
        z = rng.standard_normal((pop, len(hand)))
        cand = np.vstack([mean[None], mean + sig * z])                     # (the mean is scored too)
        f = score(cand, targets, n, horizon, seed=int(rng.integers(1 << 30)))
        order = np.argsort(-f)
        elite = cand[order[:mu]]
        if f[order[0]] > best_f or g == 0:
            pass
        mean = elite.mean(axis=0)
        sig = np.clip(0.7 * sig + 0.3 * elite.std(axis=0) * 1.2, 0.02, 0.6)
        print(f"gen {g:2d}  best {f[order[0]]:.3f}  mean-of-pop {f.mean():.3f}  mean-weights {f[0]:.3f}  sigma {sig.mean():.3f}  ({time.time() - started:.0f}s)", flush=True)
        if f[order[0]] > best_f:
            best_f, best = float(f[order[0]]), cand[order[0]].copy()
        np.save(OUT.with_suffix(".ckpt.npy"), np.stack([mean, best]))        # (a crash loses one generation, not the run)
    return hand, mean, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.train:
        hand, mean, best = train(a.generations, a.pop, n=a.n)
        # the final choice: the better of the search's mean and best on a fresh, larger evaluation
        cands = {"hand": hand, "mean": mean, "best": best}
        rates = {k: arrival_rate(v, a.n) for k, v in cands.items()}
        print({k: round(v, 4) for k, v in rates.items()}, flush=True)
        pick = max(("mean", "best"), key=lambda k: rates[k])
        OUT.write_text(json.dumps({"hand": [float(v) for v in hand], "trained": [float(v) for v in cands[pick]],
                                   "report": {"surrogate": "mobroad_pather_sim", "held_out_arrival": {"hand": rates["hand"], "trained": rates[pick]}, "n": a.n,
                                              "generations": a.generations, "pop": a.pop, "picked": pick}}, indent=1), encoding="utf-8")
        print("wrote", OUT, flush=True)
    if a.eval:
        for kind in ("hand", "trained"):
            print(kind, round(arrival_rate(F.mob_pather_weights(kind), a.n), 4), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    main()
