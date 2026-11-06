"""Teach and compare the declared games' autopilots (``pilot_decl.py``).

    python examples/meowdoku/pilot_train.py train   --game delve [--arch token|graph|both]
    python examples/meowdoku/pilot_train.py compare --game delve [--seeds 6]
    python examples/meowdoku/pilot_train.py all     [--games delve platformer ...]

A TEACHER (``pilot_teachers.py``) flies headless games of the real law at the declaration's decision
step; every state is recorded as the declaration's ``joint`` observation with the teacher's outputs as
its label (``collect``).  The first round is the teacher flying (with brief random departures, so the
data covers states off its path); the next rounds are DAgger: the student flies some of the time, the
teacher still labels every state.  Each network (token, graph) is then fit on its slice of the rows with
``autopilot.PolicyTrainer`` (the repo's autograd + Adam) and saved as ``<game>.<arch>.npz`` -- never over
an existing file unless ``--overwrite``.  ``compare`` flies teacher, token net, graph net and a do-nothing
baseline over held-out seeds and prints progress, imitation error and decision cost.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[1] / "examples")]

from meowdoku import autopilot as ap  # noqa: E402
from meowdoku import pilot_decl as pd  # noqa: E402

TRAIN_SEEDS = range(100, 100 + 400)
HELD_SEEDS = range(1000, 1006)


def fly(name: str, seed: int, controller, seconds: float | None = None, record: bool = True, cost: dict | None = None) -> dict:
    """One headless episode on the real law.  ``controller(game, joint_row_or_None, label) -> outputs``; the
    teacher labels every state.  Returns the progress numbers (+ the recorded rows)."""
    decl = pd.DECLS[name]
    game = decl.make(seed)
    game.control({})
    game.tick()
    teacher = decl.make_teacher(game)
    seconds = decl.seconds if seconds is None else seconds
    xs, ys = [], []
    t_obs = t_net = 0.0
    steps = int(round(seconds / decl.step))
    for i in range(steps):
        if game.result() is not None:
            break
        label = teacher.act(game)
        t0 = time.perf_counter()
        out = controller(game, label)
        if cost is not None:
            t_net += time.perf_counter() - t0
        if record:
            xs.append(decl.observe(game, "joint"))
            ys.append(label)
        game.control({"inputs": [{"seq": i + 1, "dur": decl.step, **decl.controls_from(game, out)}]})
        game.tick()
    result = decl.progress(game)
    result["steps"] = i + 1
    result["dead"] = float(game.result() is not None)
    if cost is not None:
        cost["decisions"] = cost.get("decisions", 0) + i + 1
        cost["seconds"] = cost.get("seconds", 0.0) + t_net
    if record:
        result["x"], result["y"] = np.array(xs), np.array(ys)
    return result


def noisy(controller, seed: int, decl):
    """The controller with brief random departures (a few decisions of random sticks) so the rows cover off-path states."""
    rng = np.random.default_rng(seed)
    hold = {"left": 0, "out": None}

    def act(game, label):
        if hold["left"] > 0:
            hold["left"] -= 1
            return hold["out"]
        if rng.random() < 0.05:
            hold["left"] = int(rng.integers(2, 10))
            hold["out"] = np.array([rng.uniform(-1, 1) if c.kind == "axis" else rng.choice([-0.9, 0.9]) for c in decl.controls])
            return hold["out"]
        return controller(game, label)
    return act


def student(name: str, params: dict, beta: float = 0.0, seed: int = 0):
    """The network as a controller; with ``beta`` the teacher flies a share of each second (DAgger)."""
    decl = pd.DECLS[name]
    arch = ap.arch_of(params)
    run = ap.numpy_product(1, *ap.widths_of(params), arch=ap.arch_key(params), like=params)
    rng = np.random.default_rng(seed)
    state = {"left": 0, "teacher": False}

    def act(game, label):
        if beta > 0.0:
            if state["left"] <= 0:
                state["left"], state["teacher"] = int(round(1.0 / decl.step)), rng.random() < beta
            state["left"] -= 1
            if state["teacher"]:
                return label
        return run(decl.observe(game, arch), params)[0]
    return act


def collect(name: str, seeds, controller_of, seconds: float | None = None) -> tuple[np.ndarray, np.ndarray, list]:
    xs, ys, rows = [], [], []
    for seed in seeds:
        r = fly(name, seed, controller_of(seed), seconds)
        xs.append(r.pop("x")); ys.append(r.pop("y")); rows.append(r)
    return np.concatenate(xs), np.concatenate(ys), rows


def fit(params: dict, x: np.ndarray, y: np.ndarray, epochs: int, seed: int = 0, trainer=None, lr: float = 2e-3):
    trainer = trainer or ap.PolicyTrainer(params, lr=lr)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        order, total = rng.permutation(len(x)), 0.0
        for at in range(0, len(x), 256):
            rows = order[at:at + 256]
            total += trainer.step(x[rows], y[rows]) * len(rows)
        losses.append(total / len(x))
    return trainer, losses


def mse(params: dict, x: np.ndarray, y: np.ndarray) -> float:
    from meowdoku.idle_trainer import forward_rows        # the traced NumPy product: gradient-free
    return float(np.mean((forward_rows(params, x) - y) ** 2))


def teacher_rows_path(name: str) -> Path:
    return ap.BUILD / f"teacher_{name}.npz"


def train(name: str, archs=("token", "graph"), episodes: int = 12, rounds: int = 2, epochs: int = 40,
          overwrite: bool = False, seed: int = 0, log=print, seconds: float | None = None) -> dict:
    decl = pd.DECLS[name]
    started = time.perf_counter()
    # round 0: the teacher flying (a clean pass and noisy ones)
    seeds = list(TRAIN_SEEDS)[:episodes]
    cx, cy, _ = collect(name, seeds[:max(2, episodes // 3)], lambda s: (lambda g, label: label), seconds)
    nx, ny, _ = collect(name, seeds[max(2, episodes // 3):], lambda s: noisy(lambda g, label: label, s, decl), seconds)
    x, y = np.concatenate([cx, nx]), np.concatenate([cy, ny])
    log(f"[{name}] teacher rows: {len(x)} ({time.perf_counter() - started:.0f}s)")
    out = {}
    held_x, held_y, _ = collect(name, list(HELD_SEEDS)[:3], lambda s: noisy(lambda g, label: label, s, decl), seconds)
    for arch in archs:
        path = ap.decl_weights_path(name, arch)
        if path.exists() and not overwrite:
            log(f"[{name}] {path.name} exists: not overwritten (training a candidate in memory only)")
        params = decl.init_params(arch, seed)
        trainer, losses = fit(params, decl.slice(arch, x), y, epochs, seed)
        ax, ay = x, y
        for r in range(rounds):                             # DAgger: the student flies, the teacher labels
            params = trainer.params()
            sx, sy, _ = collect(name, [s + 17 * (r + 1) for s in seeds[:max(3, episodes // 2)]],
                                lambda s: student(name, params, beta=0.3, seed=s), seconds)
            ax, ay = np.concatenate([ax, sx]), np.concatenate([ay, sy])
            trainer, losses = fit(params, decl.slice(arch, ax), ay, max(10, epochs // 2), seed + r + 1, trainer)
        params = trainer.params()
        out[arch] = {"rows": int(len(ax)), "loss_first": losses[0], "loss_last": losses[-1],
                     "held_mse": mse(params, decl.slice(arch, held_x), held_y), "params": int(sum(params[n].size for n in ap.param_names(params)))}
        if not path.exists() or overwrite:
            ap.save_params(params, path)
            out[arch]["saved"] = str(path)
        else:
            out[arch]["candidate_kept_in_memory"] = True
            out[arch]["params_obj"] = params
        log(f"[{name}/{arch}] {json.dumps({k: v for k, v in out[arch].items() if k != 'params_obj'})} ({time.perf_counter() - started:.0f}s)")
    ap.BUILD.mkdir(parents=True, exist_ok=True)
    tpath = teacher_rows_path(name)
    if not tpath.exists():
        np.savez(tpath, x=ax, y=ay)
    return out


def compare(name: str, seeds=None, seconds: float | None = None, log=print) -> dict:
    decl = pd.DECLS[name]
    seeds = list(seeds or HELD_SEEDS)
    held_x, held_y, _ = collect(name, seeds[:3], lambda s: (lambda g, label: label))
    contenders = {"teacher": lambda s: (lambda g, label: label), "idle": lambda s: (lambda g, label: np.where(
        [c.kind == "button" for c in decl.controls], -0.9, 0.0))}
    paramsets = {}
    for arch in pd.ARCHS:
        params = ap.load_params(ap.decl_weights_path(name, arch))
        if params is not None:
            paramsets[arch] = params
            contenders[arch] = (lambda p: lambda s: student(name, p, 0.0, s))(params)
    table = {}
    for label, make in contenders.items():
        cost, runs = {}, []
        for s in seeds:
            runs.append(fly(name, s, make(s), seconds, record=False, cost=cost))
        keys = [k for k in runs[0] if k not in ("steps",)]
        row = {k: float(np.mean([r[k] for r in runs])) for k in keys}
        row["decision_us"] = 1e6 * cost["seconds"] / max(1, cost["decisions"])
        if label in paramsets:
            row["imitation_mse"] = mse(paramsets[label], decl.slice(label, held_x), held_y)
            row["params"] = int(sum(paramsets[label][n].size for n in ap.param_names(paramsets[label])))
        table[label] = row
        log(f"[{name}] {label:8s} {json.dumps({k: round(v, 3) for k, v in row.items()})}")
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("train", "compare", "all"):
        c = sub.add_parser(cmd)
        c.add_argument("--game", choices=list(pd.DECLS))
        c.add_argument("--games", nargs="*", default=list(pd.DECLS))
        c.add_argument("--arch", choices=("token", "graph", "both"), default="both")
        c.add_argument("--episodes", type=int, default=12)
        c.add_argument("--rounds", type=int, default=2)
        c.add_argument("--epochs", type=int, default=40)
        c.add_argument("--seeds", type=int, default=6)
        c.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    games = [args.game] if args.game else args.games
    archs = pd.ARCHS if args.arch == "both" else (args.arch,)
    for g in games:
        if args.cmd in ("train", "all"):
            train(g, archs, args.episodes, args.rounds, args.epochs, args.overwrite)
        if args.cmd in ("compare", "all"):
            compare(g, list(HELD_SEEDS)[:args.seeds] if args.seeds <= len(HELD_SEEDS) else list(range(1000, 1000 + args.seeds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
