"""World Walk's law, native against NumPy: the same script of play in two lanes, the state compared.

The lane is ``MEOWDOKU_LAW`` (read when ``entities`` is imported), so the two runs are two processes:

    set MEOWDOKU_LAW=numpy && python examples/meowdoku/worldwalk_parity.py run numpy.npz
    set MEOWDOKU_LAW=llvm  && python examples/meowdoku/worldwalk_parity.py run native.npz   (eigen: TURING_EIGEN_INCLUDE)
    python examples/meowdoku/worldwalk_parity.py compare numpy.npz native.npz

The script is the whole of the new law at once: a click path laid across the roads, free walking and a turn, then
a house, a tower and a wall raised beside a road (the wall reroutes the path) while the settlers, raiders and
towers go about their business.  ``compare`` prints, per column, the largest difference over every lane and step
sampled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import worldwalk as ww  # noqa: E402

STEP = 1.0 / 30.0
COLUMNS = ("alive", "ux", "uy", "uz", "hx", "hy", "hz", "spd", "yaw", "gspd", "hp", "node", "x", "y", "route", "wp",
           "map_u", "map_v", "map_gx", "map_gy", "lives", "mot", "timer", "cool", "mat")


def script(seconds: float = 24.0):
    """Yield (time, game) after each landed step of the scripted play."""
    w = ww.world()
    game = ww.WorldWalk(seed=0)
    s = game.session
    s.control({})
    s.advance(1.0 / 60.0)
    ground = game.ground
    towns = [w.site_nodes[i] for i in w.towns]
    path = min((p for a in towns for b in towns if a < b for p in [ground.path(a, b)] if p), key=len)
    st = s.state
    p = w.nodes[path[0]]
    g = w.nodes[path[1]]
    h = g - (g @ p) * p
    h /= np.linalg.norm(h)
    for name, v in zip(("ux", "uy", "uz"), p):
        getattr(st, name)[0] = v
    for name, v in zip(("hx", "hy", "hz"), h):
        getattr(st, name)[0] = v
    st.node[0] = ww.MAP0 + path[0]
    game.goto(node=path[-1])
    buildable = [n for n in range(len(w.nodes)) if ground.buildable(n) is None]
    inner = next((n for n in path[2:-2] if ground.buildable(n) is None), buildable[7])
    held = {"move_axis": 0.0, "move_axis_y": 0.0, "fire": 0.0}
    t, built = 0.0, set()
    while t < seconds:
        if t >= 8.0 and "free" not in built:                      # free walking cancels the path; then a turn
            built.add("free")
            held = {"move_axis": 0.0, "move_axis_y": -1.0, "fire": 0.0}
        if t >= 11.0 and "turn" not in built:
            built.add("turn")
            held = {"move_axis": 1.0, "move_axis_y": -0.5, "fire": 1.0}
        if t >= 13.0 and "release" not in built:
            built.add("release")
            held = {"move_axis": 0.0, "move_axis_y": 0.0, "fire": 0.0}
        if t >= 14.0 and "builds" not in built:
            built.add("builds")
            game.place("house", node=buildable[3])
            game.place("tower", node=buildable[11])
            game.goto(node=path[-1])                              # a path again (the wizard walked off it): then a wall on it
            game.place("wall", node=inner if game.ground.buildable(inner) is None else buildable[19])
        s.control(held)
        s.advance(STEP)
        t += STEP
        yield t, game


def run(out: str) -> None:
    rows = {name: [] for name in COLUMNS}
    times = []
    for t, game in script():
        if int(round(t / STEP)) % 6 == 0:
            times.append(t)
            for name in COLUMNS:
                rows[name].append(np.array(game.session.column(name)[:ww.ENT + 4], dtype=np.float64))
    np.savez(out, t=np.asarray(times), **{k: np.stack(v) for k, v in rows.items()})
    from meowdoku.entities import LAW_LANE
    print(f"[parity] lane {LAW_LANE}: {len(times)} samples of {len(COLUMNS)} columns -> {out}", flush=True)


def compare(a: str, b: str) -> int:
    x, y = np.load(a), np.load(b)
    worst = 0.0
    for name in COLUMNS:
        d = np.abs(x[name] - y[name])
        scale = max(1.0, float(np.abs(x[name]).max()))
        step = int(np.argmax(d.max(axis=1)))
        print(f"{name:8s} max abs diff {d.max():.3e}  (column scale {scale:.3g}; worst at t={float(x['t'][step]):.1f}s)", flush=True)
        worst = max(worst, float(d.max()) / scale)
    print(f"[parity] worst relative difference {worst:.3e}", flush=True)
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        raise SystemExit(compare(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit(__doc__)
