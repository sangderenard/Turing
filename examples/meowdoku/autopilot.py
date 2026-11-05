"""An autopilot for the entity games: a network that works in entity position space.

The observation is fixed-width and read straight from the session columns (no
rendering): the ship's own state, the ``K`` hostile lanes nearest in
time-to-contact (relative position, relative velocity, a kind flag; zero
padded) and a few game-wide scalars.  The policy is a small AbstractTensor MLP
(``policy_forward``: tanh, tanh, tanh) giving ``move_axis``, ``move_axis_y``
and a fire signal (fire when it is positive); it is deterministic.

Training runs against the real game laws: episodes are ``EntityGame`` ticks
(the NumPy-product LawPieces), input-timed at ``STEP`` = 1/30 s per decision.
The scripted dodger labels states for a short imitation fit (the repo's
autograd + ``Adam``), then evolution strategies (antithetic, rank-shaped)
fine-tune the weights over parallel processes.

Lanes of the same forward: ``eager`` (AbstractTensor), ``numpy`` (the traced
tape printed by ``tape_to_source``, as ``LawPiece`` does) and ``llvm``
(``compile_native_graph_forward`` under a work contract, the
``native_forward.py`` route).

    python examples/meowdoku/autopilot.py train --game space_invaders
    python examples/meowdoku/autopilot.py train --game bullet_hell
    python examples/meowdoku/autopilot.py bench
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

BUILD = ROOT / "build" / "meowdoku" / "autopilot"
STEP = 1.0 / 30.0                  # one decision per STEP seconds of play
K = 14                             # hostile lanes in the observation
HIDDEN = (64, 64)
PARAMS = ("W1", "b1", "W2", "b2", "W3", "b3")
OUTPUTS = 3                        # move_axis, move_axis_y, fire
FEATURES = 5 + 5 * K + 4

GAMES = ("space_invaders", "bullet_hell")


def make_game(name: str):
    from meowdoku.invaders import SpaceInvaders
    from meowdoku.scroller import BulletScroller
    return {"space_invaders": SpaceInvaders, "bullet_hell": BulletScroller}[name]("autopilot")


# ------------------------------------------------------------------ features
# One function per game, reading the session columns.  Layout (FEATURES wide):
#   0..4     ship x (-1..1 across the field), y (-1..1 down), vx, vy, shield charge
#   5..74    K hostile lanes by time-to-contact: dx, dy, dvx, dvy (relative to the
#            ship, scaled) and a kind flag (0 = padding)
#   75..78   field width, remaining fleet/enemies, lives, weapon ready

DX, DY, DV = 8.0, 10.0, 10.0


def _observe(game, hostile_kind, remaining, ready, height: float) -> np.ndarray:
    col = game.session.column
    k = game.PLAYER_LANE
    alive, x, y, vx, vy = col("alive"), col("x"), col("y"), col("vx"), col("vy")
    field_w = float(col("field_w")[0])
    out = np.zeros(FEATURES)
    out[0] = 2.0 * x[k] / field_w - 1.0
    out[1] = 2.0 * y[k] / height - 1.0
    out[2], out[3] = vx[k] / DV, vy[k] / DV
    out[4] = col("energy")[k]
    lanes = np.concatenate([np.arange(a, b) for a, b in game.HOSTILE_LANES])
    lanes = lanes[alive[lanes] > 0.0]
    if len(lanes):
        dx, dy = x[lanes] - x[k], y[lanes] - y[k]
        dvx, dvy = vx[lanes] - vx[k], vy[lanes] - vy[k]
        dist = np.hypot(dx, dy)
        closing = -(dx * dvx + dy * dvy) / np.maximum(dist, 1e-6)
        speed = np.maximum(closing, 0.25 * np.hypot(dvx, dvy) + 1e-3)
        order = np.argsort(np.maximum(dist - 0.5, 0.0) / speed, kind="stable")[:K]
        n = len(order)
        rows = np.stack([dx[order] / DX, dy[order] / DY, dvx[order] / DV, dvy[order] / DV,
                         hostile_kind(col, lanes[order])], axis=1)
        out[5:5 + 5 * n] = rows.reshape(-1)
    out[5 + 5 * K:] = (field_w / 10.0, remaining(col), float(col("lives")[0]) / 3.0, ready(col))
    return out


def invaders_features(game) -> np.ndarray:
    from meowdoku import invaders as inv
    flag = {inv.BOMB: 1.0, inv.UFO: 0.75, inv.INVADER: 0.5}
    return _observe(
        game,
        lambda col, lanes: np.array([flag.get(s, 0.5) for s in col("slot")[lanes]]),
        lambda col: float(col("alive")[inv.FLEET0:inv.SHIELD0].sum()) / inv.INVADERS,
        lambda col: 1.0 if col("alive")[1] <= 0.0 else 0.0,
        inv.HEIGHT)


def scroller_features(game) -> np.ndarray:
    from meowdoku import scroller as sc

    def kind(col, lanes):
        slot, k = col("slot")[lanes], col("kind")[lanes]
        return np.where(slot == sc.BULLET, 1.0, np.where(k == sc.SPINNER, 0.75, 0.5))

    return _observe(
        game, kind,
        lambda col: float(col("alive")[sc.ENEMIES[0]:sc.ENEMIES[1]].sum()) / (sc.ENEMIES[1] - sc.ENEMIES[0]),
        lambda col: 1.0 if col("timer")[0] <= 0.0 else 0.0,
        sc.HEIGHT)


FEATURE_FUNCTIONS = {"space_invaders": invaders_features, "bullet_hell": scroller_features}


def controls_from(game, out: np.ndarray) -> dict[str, float]:
    """The policy's three outputs as the game's control columns."""
    values = {"move_axis": float(out[0]), "move_axis_y": float(out[1]) if "move_axis_y" in game.CONTROLS else 0.0,
              "steer_x": -1.0, "steer_y": -1.0, "fire": 1.0 if out[2] > 0.0 else 0.0}
    return {name: values[name] for name in game.CONTROLS}


# ------------------------------------------------------------------- the net

def policy_forward(x, W1, b1, W2, b2, W3, b3):
    """x (N, FEATURES) -> (N, 3) in [-1, 1].  AbstractTensor arithmetic only,
    so the same text is traced for the NumPy product and the LLVM forward."""
    h = (x @ W1 + b1).tanh()
    h = (h @ W2 + b2).tanh()
    return (h @ W3 + b3).tanh()


def init_params(seed: int = 0, features: int = FEATURES, hidden=HIDDEN) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sizes = (features, *hidden, OUTPUTS)
    params = {}
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), start=1):
        params[f"W{i}"] = rng.normal(0.0, math.sqrt(1.0 / a), (a, b))
        params[f"b{i}"] = np.zeros((1, b))
    return params


def flatten(params: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([params[n].reshape(-1) for n in PARAMS])


def unflatten(vector: np.ndarray, like: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out, at = {}, 0
    for n in PARAMS:
        size = like[n].size
        out[n] = np.asarray(vector[at:at + size], dtype=np.float64).reshape(like[n].shape)
        at += size
    return out


def save_params(params: dict[str, np.ndarray], path: Path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez(partial, **params, **{k: np.asarray(v) for k, v in meta.items()})
    partial.replace(path)


def load_params(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    data = np.load(path)
    return {n: np.asarray(data[n], dtype=np.float64) for n in PARAMS}


def weights_path(game: str) -> Path:
    return BUILD / f"{game}.npz"


def eager_forward(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """The AbstractTensor forward itself (numpy backend, no tape)."""
    from src.common.tensors import AbstractTensor as AT
    with AT.use_backend("numpy"), AT.autograd.no_grad():
        out = policy_forward(AT.tensor(np.asarray(x, dtype=np.float64).reshape(-1, FEATURES)),
                             *(AT.tensor(params[n]) for n in PARAMS))
        return np.asarray(out.numpy(), dtype=np.float64)


# NumPy lane: the forward traced through AbstractTensor and printed as a NumPy
# function by tape_to_source (the product LawPiece uses for the laws).

_PRODUCTS: dict = {}


def numpy_product(batch: int = 1):
    if batch in _PRODUCTS:
        return _PRODUCTS[batch]
    from src.common.tensors.autograd import autograd
    from src.common.tensors.numpy_backend import NumPyTensorOperations
    from src.common.tensors.accelerator_backends import tape_to_source
    from src.common.tensors.accelerator_backends.tape_to_source import emit_tape_source
    digest = hashlib.sha256((inspect.getsource(policy_forward) + inspect.getsource(tape_to_source)
                             + f"{FEATURES}{HIDDEN}{batch}").encode()).hexdigest()[:16]
    entry = "autopilot_policy"
    path = BUILD / f"{entry}_b{batch}_{digest}.py"
    if not path.exists():
        like = init_params()
        sample = {"x": NumPyTensorOperations.tensor(np.zeros((batch, FEATURES))),
                  **{n: NumPyTensorOperations.tensor(like[n]) for n in PARAMS}}
        with autograd.forward_capture() as tape:
            produced = policy_forward(*sample.values())
        source = emit_tape_source(tape, sample, {"action": produced}, backend="numpy", function_name=entry)
        header = (f'"""{entry}: compiled NumPy product of meowdoku.autopilot.policy_forward (batch {batch}).\n\n'
                  f"Generated by AbstractTensor tape tracing + tape_to_source (numpy dialect), digest {digest}.\n"
                  f'Do not edit: change policy_forward and it recompiles.\n"""\n')
        BUILD.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(f".{os.getpid()}.tmp")
        partial.write_text(header + source, encoding="utf-8")
        partial.replace(path)
    spec = importlib.util.spec_from_file_location(f"meowdoku_{entry}_{batch}_{digest}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, entry)

    def run(x, params):
        out = function(np.asarray(x, dtype=np.float64).reshape(batch, FEATURES), *(params[n] for n in PARAMS))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return np.asarray(out, dtype=np.float64).reshape(batch, OUTPUTS)

    run.path = path
    _PRODUCTS[batch] = run
    return run


# LLVM lane: the route of native_forward.py -- trace through the SSA tensor
# backend and compile with compile_native_graph_forward under a work contract.

import itertools  # noqa: E402
_BUILDS = itertools.count()      # one directory per compile: Windows will not overwrite a loaded DLL


class NativePolicy:
    def __init__(self, params: dict[str, np.ndarray], batch: int = 1, contract: str = "develop",
                 directory: Path | None = None):
        from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
        from src.compiler.llvm_training_runtime import compile_native_graph_forward
        from src.compiler.ssa_llvm_backend import prepare_artifact_execution
        from src.compiler.work_contract import PRESETS, scoped_work_contract
        self.batch, self.contract = int(batch), contract
        name = f"autopilot_policy_b{self.batch}_{contract}"
        directory = Path(directory or BUILD / "native")
        started = time.perf_counter()
        with scoped_work_contract(PRESETS[contract]):
            program = SSATensorProgram(name)
            shapes = {"x": (self.batch, FEATURES), **{n: tuple(params[n].shape) for n in PARAMS}}
            bindings = {key: SSATensorOperations.input(program, shape) for key, shape in shapes.items()}
            action = policy_forward(bindings["x"], *(bindings[n] for n in PARAMS))
            self.forward = compile_native_graph_forward(
                action, bindings=bindings, source="meowdoku.autopilot.policy_forward",
                name=name, directory=directory / name / f"p{os.getpid()}_{next(_BUILDS)}")
        self.compile_seconds = time.perf_counter() - started
        ids = dict(self.forward.input_value_ids)
        self._execution = prepare_artifact_execution(self.forward.artifact, {
            **{ids[n]: np.ascontiguousarray(params[n]) for n in PARAMS}, ids["x"]: np.zeros((self.batch, FEATURES))})
        self._x = np.asarray(self._execution.buffers[ids["x"]])
        self._out = np.asarray(self._execution.buffers[self.forward.output_value_ids[0]])
        self._weights = {n: np.asarray(self._execution.buffers[ids[n]]) for n in PARAMS}

    def set_weights(self, params: dict[str, np.ndarray]) -> None:
        for n in PARAMS:
            np.copyto(self._weights[n].reshape(-1), np.asarray(params[n], dtype=np.float64).reshape(-1))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._x.reshape(-1)[:] = np.asarray(x, dtype=np.float64).reshape(-1)
        self._execution.run()
        return self._out.reshape(self.batch, OUTPUTS).copy()


class PolicyTrainer:
    """The imitation fit: the forward on the repo's autograd tape, ``Adam``."""

    def __init__(self, params: dict[str, np.ndarray], lr: float = 2e-3):
        from src.common.tensors import AbstractTensor as AT
        from src.common.tensors.abstract_nn import Adam
        with AT.use_backend("numpy"):
            self.tensors = {n: AT.tensor(params[n]) for n in PARAMS}
            for t in self.tensors.values():
                t.requires_grad_()
            self.optimizer = Adam([self.tensors[n] for n in PARAMS], lr=lr)

    def step(self, x: np.ndarray, target: np.ndarray) -> float:
        from src.common.tensors import AbstractTensor as AT
        with AT.use_backend("numpy"):
            params = [self.tensors[n] for n in PARAMS]
            out = policy_forward(AT.tensor(np.asarray(x, dtype=np.float64)), *params)
            diff = out - AT.tensor(np.asarray(target, dtype=np.float64))
            loss = (diff * diff).mean()
            grads = AT.autograd.grad(loss, params)
            with AT.autograd.no_grad():
                for p, q in zip(params, self.optimizer.step(params, grads)):
                    AT.copyto(p, q)
            return float(np.asarray(loss.numpy()))

    def params(self) -> dict[str, np.ndarray]:
        return {n: np.asarray(self.tensors[n].numpy(), dtype=np.float64).copy() for n in PARAMS}


# --------------------------------------------------------------- the pilot

_PILOT_WEIGHTS: dict = {}


def preload() -> list[str]:
    """Load every trained weight file that exists (the server calls this at start)."""
    loaded = []
    for name in GAMES:
        params = load_params(weights_path(name))
        if params is not None:
            _PILOT_WEIGHTS[name] = params
            loaded.append(name)
    if loaded:
        numpy_product(1)
    return loaded


class Pilot:
    """The compiled policy deciding one game's controls (NumPy lane; the LLVM
    lane when ``MEOWDOKU_PILOT=llvm`` and it compiles)."""

    def __init__(self, name: str, params: dict[str, np.ndarray]):
        self.name, self.params = name, params
        self.features = FEATURE_FUNCTIONS[name]
        self.lane = "numpy"
        self.run = lambda x: numpy_product(1)(x, self.params)
        if os.environ.get("MEOWDOKU_PILOT", "numpy") == "llvm":
            try:
                native = NativePolicy(params, batch=1)
                self.run, self.lane = native, "llvm"
            except Exception as error:      # keep NumPy; say why
                print(f"autopilot: llvm lane refused ({type(error).__name__}: {str(error)[:200]}); numpy", flush=True)

    def controls(self, game) -> dict[str, float]:
        return controls_from(game, self.run(self.features(game))[0])


def pilot_for(game) -> Pilot:
    name = game.game
    if name not in _PILOT_WEIGHTS:
        params = load_params(weights_path(name))
        if params is None:
            raise ValueError(f"no trained autopilot for {name}: {weights_path(name)} is missing")
        _PILOT_WEIGHTS[name] = params
    return Pilot(name, _PILOT_WEIGHTS[name])


def available(name: str) -> bool:
    return name in _PILOT_WEIGHTS or weights_path(name).exists()


# ---------------------------------------------------------- scripted baselines

def dodger_controls(game) -> tuple[float, float, float]:
    """The scripted dodger: (move_axis, move_axis_y, fire)."""
    return (_invaders_dodger if game.game == "space_invaders" else _scroller_dodger)(game)


def _invaders_dodger(game):
    from meowdoku import invaders as inv
    col = game.session.column
    alive, x, y, vy = col("alive"), col("x"), col("y"), col("vy")
    px, py = float(x[0]), float(y[0])
    bombs = np.array([k for k in (*inv.BOMB_LANES, inv.UFO_LANE) if alive[k] > 0.0 and k != inv.UFO_LANE], dtype=int)
    fleet = np.flatnonzero(alive[inv.FLEET0:inv.SHIELD0] > 0.0) + inv.FLEET0
    field_w = float(col("field_w")[0])
    cands = np.arange(0.7, field_w - 0.69, 0.35)
    cost = np.abs(cands - px) * 0.05
    if len(fleet):
        cost += 0.12 * np.min(np.abs(cands[:, None] - x[fleet][None, :]), axis=1)
    if len(bombs):
        t_hit = np.maximum((py - y[bombs]) / inv.BOMB_SPEED, 0.0)                       # when each bomb reaches the ship's row
        ahead = (y[bombs] < py + 0.3) & (t_hit < 2.0)
        reach = np.minimum(np.abs(cands[:, None] - px), inv.PLAYER_SPEED * t_hit[None, :] + 0.0)
        where = px + np.sign(cands[:, None] - px) * reach                                 # ship x when the bomb arrives
        near = np.abs(where - x[bombs][None, :]) < 0.75
        cost += 50.0 * (near & ahead[None, :]).sum(axis=1)
        cost += 1.0 * ((np.abs(cands[:, None] - x[bombs][None, :]) < 1.0) & ahead[None, :]).sum(axis=1)
    target = float(cands[int(np.argmin(cost))])
    gap = target - px
    return float(np.clip(gap * 3.0, -1.0, 1.0)), 0.0, 1.0


_ACTIONS = np.array([(ax, ay) for ax in (-1.0, 0.0, 1.0) for ay in (-1.0, 0.0, 1.0)])


def _scroller_dodger(game):
    from meowdoku import scroller as sc
    col = game.session.column
    alive, x, y, vx, vy, slot = col("alive"), col("x"), col("y"), col("vx"), col("vy"), col("slot")
    field_w = float(col("field_w")[0])
    hostile = np.concatenate([np.arange(a, b) for a, b in game.HOSTILE_LANES])
    hostile = hostile[alive[hostile] > 0.0]
    steps = 12
    t = np.arange(1, steps + 1) * 0.05
    # candidate ship paths: velocity relaxes toward axis * SPEED, walls clip
    pos = np.zeros((9, steps, 2))
    v = np.tile([vx[0], vy[0]], (9, 1))
    p = np.tile([x[0], y[0]], (9, 1))
    for i in range(steps):
        v = v + (_ACTIONS * sc.SPEED - v) * (0.05 / (sc.TAU + 0.05))
        p = p + v * 0.05
        p = np.clip(p, [sc.MARGIN, sc.MARGIN], [field_w - sc.MARGIN, sc.HEIGHT - sc.MARGIN])
        pos[:, i] = p
    cost = np.zeros(9)
    if len(hostile):
        bx = x[hostile][None, :] + vx[hostile][None, :] * t[:, None]                    # (steps, M)
        by = y[hostile][None, :] + vy[hostile][None, :] * t[:, None]
        ddx = np.abs(pos[:, :, 0, None] - bx[None])                                      # (9, steps, M)
        ddy = np.abs(pos[:, :, 1, None] - by[None])
        radius = np.where(slot[hostile] == sc.ENEMY, 0.75, 0.42)[None, None, :]
        hit = (ddx < radius) & (ddy < radius + 0.15)
        cost += 100.0 * hit.sum(axis=(1, 2))
        close = np.hypot(ddx, ddy)
        cost += 1.5 * np.maximum(1.4 - close, 0.0).sum(axis=(1, 2)) / steps
    enemies = np.flatnonzero(alive[sc.ENEMIES[0]:sc.ENEMIES[1]] > 0.0) + sc.ENEMIES[0]
    final = pos[:, -1]
    if len(enemies):
        cost += 0.04 * np.min(np.abs(final[:, 0, None] - x[enemies][None, :]), axis=1)
    cost += 0.02 * np.abs(final[:, 1] - (sc.HEIGHT - 3.0))
    cost += 0.01 * np.abs(final[:, 0] - field_w / 2.0)
    best = _ACTIONS[int(np.argmin(cost))]
    return float(best[0]), float(best[1]), 1.0


def make_random(seed: int):
    """A random policy: uniform axes and a coin for fire, each held about 0.2 s."""
    rng = np.random.default_rng(seed)
    state = {"left": 0, "out": np.zeros(3)}

    def act(game, features):
        if state["left"] <= 0:
            state["out"] = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.choice([-1.0, 1.0])])
            state["left"] = 6
        state["left"] -= 1
        return state["out"]
    return act


def make_dodger():
    def act(game, features):
        a, b, f = dodger_controls(game)
        return np.array([a, b, 1.0])
    return act


# ------------------------------------------------------------------ episodes

def run_episode(name: str, act, seed: int, seconds: float = 40.0, record=None, warm: bool = True) -> dict:
    """One episode on the real game laws, input-timed at STEP.  ``act(game,
    features) -> (3,)``.  A seeded random idle opening varies the world's
    clock (the laws' dice) and the ship's start.  Returns score, survival
    seconds (world clock while lives remain), lives left and wave."""
    game = make_game(name)
    features = FEATURE_FUNCTIONS[name]
    rng = np.random.default_rng(seed)
    seq = [0]

    def advance(out_controls: dict) -> None:
        seq[0] += 1
        game.control({"inputs": [{"seq": seq[0], "dur": STEP, **out_controls}]})
        game.tick()

    idle = {"move_axis": 0.0, "move_axis_y": 0.0, "steer_x": -1.0, "steer_y": -1.0, "fire": 0.0}
    advance({k: v for k, v in idle.items() if k in game.CONTROLS})
    if warm:
        drift = {**idle, "move_axis": float(rng.uniform(-1, 1))}
        for i in range(int(rng.integers(0, 60))):
            advance({k: v for k, v in drift.items() if k in game.CONTROLS})
    col = game.session.column
    start = float(col("clock")[0])
    steps = int(round(seconds / STEP))
    for _ in range(steps):
        if col("lives")[0] <= 0.0:
            break
        f = features(game)
        if record is not None:
            record.append(f)
        out = act(game, f)
        advance(controls_from(game, out))
    wave = float(col("wave")[0]) if "wave" in game.session.columns else 0.0
    return {"score": float(col("score")[0]), "survival": float(col("clock")[0]) - start,
            "lives": float(col("lives")[0]), "wave": wave, "seed": seed}


def policy_act(name: str, params: dict[str, np.ndarray]):
    run = numpy_product(1)

    def act(game, f):
        return run(f, params)[0]
    return act


def _worker_eval(task):
    """task = (game, theta, like_shapes, seeds, seconds) -> list of episode dicts."""
    name, theta, shapes, seeds, seconds = task
    like = {n: np.zeros(shape) for n, shape in shapes.items()}
    params = unflatten(theta, like)
    act = policy_act(name, params)
    return [run_episode(name, act, int(s), seconds) for s in seeds]


def _worker_baseline(task):
    kind, name, seed, seconds = task
    act = make_random(seed) if kind == "random" else make_dodger()
    return run_episode(name, act, int(seed), seconds)


def summarize(episodes: list[dict]) -> dict:
    s = np.array([e["score"] for e in episodes])
    t = np.array([e["survival"] for e in episodes])
    return {"n": len(episodes), "score_mean": float(s.mean()), "score_std": float(s.std()),
            "survival_mean": float(t.mean()), "survival_std": float(t.std()),
            "lives_mean": float(np.mean([e["lives"] for e in episodes]))}


def fitness_of(episodes: list[dict]) -> float:
    """Score plus ten points per second survived, averaged over episodes."""
    return float(np.mean([e["score"] + 10.0 * e["survival"] for e in episodes]))


# ------------------------------------------------------------------ training

def collect_imitation(name: str, episodes: int, seconds: float, seed: int = 500):
    """States visited by the dodger (with random perturbation so the data
    covers off-path states), labelled by the dodger's clean decision."""
    xs, ys = [], []
    for e in range(episodes):
        rng = np.random.default_rng(seed + e)
        hold = {"left": 0, "out": None}

        def act(game, f, rng=rng, hold=hold):
            a, b, fire = dodger_controls(game)
            xs.append(f)
            ys.append([a * 0.95, b * 0.95, 0.9 if fire > 0 else -0.9])
            if hold["left"] > 0:
                hold["left"] -= 1
                return hold["out"]
            if rng.random() < 0.04:                                  # drift off the dodger's path for a moment
                hold["left"] = int(rng.integers(3, 12))
                hold["out"] = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 1.0])
            return np.array([a, b, 1.0])
        run_episode(name, act, seed + e, seconds)
    return np.array(xs), np.array(ys)


def imitate(name: str, epochs: int = 60, episodes: int = 10, seconds: float = 30.0, seed: int = 0):
    started = time.perf_counter()
    x, y = collect_imitation(name, episodes, seconds)
    collected = time.perf_counter() - started
    trainer = PolicyTrainer(init_params(seed), lr=2e-3)
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        total = 0.0
        for i in range(0, len(x), 256):
            idx = order[i:i + 256]
            total += trainer.step(x[idx], y[idx]) * len(idx)
        losses.append(total / len(x))
    return trainer.params(), {"samples": int(len(x)), "collect_s": collected,
                              "fit_s": time.perf_counter() - started - collected,
                              "loss_first": losses[0], "loss_last": losses[-1]}


def train(name: str, workers: int, generations: int, pairs: int, sigma: float, lr: float,
          seconds: float, epochs: int, imit_episodes: int, seed: int = 0) -> dict:
    import multiprocessing as mp
    BUILD.mkdir(parents=True, exist_ok=True)
    numpy_product(1)                                    # built once here; workers only load
    t0 = time.perf_counter()
    print(f"[{name}] imitation of the scripted dodger ...", flush=True)
    params, imit = imitate(name, epochs=epochs, episodes=imit_episodes, seconds=seconds, seed=seed)
    print(f"[{name}] imitation: {imit}", flush=True)
    shapes = {n: params[n].shape for n in PARAMS}
    theta = flatten(params)
    held_seeds = list(range(1000, 1008))
    rng = np.random.default_rng(seed + 1)
    ctx = mp.get_context("spawn")
    curve = []
    # Adam on the ES gradient estimate
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    best = (-1e18, theta.copy())
    with ctx.Pool(workers) as pool:
        for gen in range(generations + 1):
            g0 = time.perf_counter()
            noise = rng.standard_normal((pairs, theta.size))
            train_seeds = [int(s) for s in rng.integers(10_000, 1_000_000, 2)]
            tasks = [(name, theta, shapes, [s], seconds) for s in held_seeds]
            cand = []
            for i in range(pairs):
                for sign in (+1.0, -1.0):
                    cand.append((name, theta + sign * sigma * noise[i], shapes, train_seeds, seconds))
            results = pool.map(_worker_eval, tasks + (cand if gen < generations else []), chunksize=1)
            held = [r[0] for r in results[:len(held_seeds)]]
            row = {"generation": gen, **{f"held_{k}": v_ for k, v_ in summarize(held).items()}}
            if gen < generations:
                fits = np.array([fitness_of(r) for r in results[len(held_seeds):]]).reshape(pairs, 2)
                row["pop_fitness_mean"] = float(fits.mean())
                row["pop_fitness_max"] = float(fits.max())
                ranks = np.argsort(np.argsort(fits.reshape(-1))).reshape(pairs, 2) / (2 * pairs - 1) - 0.5
                weight = ranks[:, 0] - ranks[:, 1]
                grad = (weight[:, None] * noise).sum(axis=0) / (2 * pairs * sigma)
                m = 0.9 * m + 0.1 * grad
                v = 0.999 * v + 0.001 * grad * grad
                step = (m / (1 - 0.9 ** (gen + 1))) / (np.sqrt(v / (1 - 0.999 ** (gen + 1))) + 1e-8)
                held_fit = fitness_of(held)
                if held_fit > best[0]:
                    best = (held_fit, theta.copy())
                theta = theta + lr * step
            else:
                held_fit = fitness_of(held)
                if held_fit > best[0]:
                    best = (held_fit, theta.copy())
            row["held_fitness"] = fitness_of(held)
            row["gen_s"] = time.perf_counter() - g0
            curve.append(row)
            print(f"[{name}] gen {gen:3d} held score {row['held_score_mean']:8.1f} survival {row['held_survival_mean']:5.1f}s "
                  f"fitness {row['held_fitness']:8.1f} pop {row.get('pop_fitness_mean', float('nan')):8.1f} ({row['gen_s']:.0f}s)", flush=True)
            save_params(unflatten(best[1], params), weights_path(name), game=name, features=FEATURES, k=K)
    wall = time.perf_counter() - t0
    result = {"game": name, "wall_s": wall, "imitation": imit, "curve": curve, "best_held_fitness": best[0],
              "sigma": sigma, "lr": lr, "pairs": pairs, "seconds_per_episode": seconds}
    (BUILD / f"{name}_curve.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def compare(name: str, workers: int, seeds: list[int], seconds: float) -> dict:
    """Random, scripted dodger, imitation-only start and the trained weights on the same seeds."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    out = {}
    with ctx.Pool(workers) as pool:
        for kind in ("random", "dodger"):
            out[kind] = summarize(pool.map(_worker_baseline, [(kind, name, s, seconds) for s in seeds], chunksize=1))
        params = load_params(weights_path(name))
        if params is not None:
            shapes = {n: params[n].shape for n in PARAMS}
            res = pool.map(_worker_eval, [(name, flatten(params), shapes, [s], seconds) for s in seeds], chunksize=1)
            out["trained"] = summarize([r[0] for r in res])
    return out


def cmd_train(args) -> None:
    result = train(args.game, args.workers, args.generations, args.pairs, args.sigma, args.lr,
                   args.seconds, args.epochs, args.imit_episodes)
    print(f"[{args.game}] trained in {result['wall_s']:.0f}s", flush=True)
    final = compare(args.game, args.workers, list(range(2000, 2000 + args.final_seeds)), args.final_seconds)
    print(f"[{args.game}] final comparison ({args.final_seeds} unseen seeds, {args.final_seconds:.0f}s cap):", flush=True)
    for kind, s in final.items():
        print(f"  {kind:8s} score {s['score_mean']:8.1f} +- {s['score_std']:6.1f}  survival {s['survival_mean']:5.1f}s +- {s['survival_std']:4.1f}"
              f"  lives left {s['lives_mean']:.2f}", flush=True)
    (BUILD / f"{args.game}_final.json").write_text(json.dumps(final, indent=1), encoding="utf-8")


def cmd_compare(args) -> None:
    final = compare(args.game, args.workers, list(range(2000, 2000 + args.final_seeds)), args.final_seconds)
    print(json.dumps(final, indent=1))


def _median_us(fn, repeats: int) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return float(np.median(samples)) * 1e6


def cmd_bench(args) -> None:
    """Parity of the NumPy and LLVM lanes against the eager forward on recorded
    feature batches, and time per call at batch 1 and 64."""
    for name in GAMES:
        params = load_params(weights_path(name))
        if params is None:
            print(f"[{name}] no weights; train first", flush=True)
            continue
        record: list = []
        run_episode(name, policy_act(name, params), 3000, args.seconds, record=record)
        feats = np.array(record)
        print(f"[{name}] recorded {len(feats)} feature rows", flush=True)
        for batch in (1, 64):
            x = feats[:batch] if len(feats) >= batch else np.resize(feats, (batch, FEATURES))
            xs = np.array([feats[(i * 7) % len(feats)] for i in range(batch)])
            ref = eager_forward(params, xs)
            line = {"eager_us": _median_us(lambda: eager_forward(params, xs), 30)}
            product = numpy_product(batch)
            line["numpy_diff"] = float(np.abs(product(xs, params) - ref).max())
            line["numpy_us"] = _median_us(lambda: product(xs, params), 300)
            # parity over every recorded row (chunks of this batch)
            worst = 0.0
            for i in range(0, len(feats) - batch + 1, batch):
                chunk = feats[i:i + batch]
                worst = max(worst, float(np.abs(product(chunk, params) - eager_forward(params, chunk)).max()))
            line["numpy_diff_all"] = worst
            for contract in args.contracts:
                try:
                    native = NativePolicy(params, batch=batch, contract=contract)
                    got = native(xs)
                    line[f"llvm_{contract}_diff"] = float(np.abs(got - ref).max())
                    line[f"llvm_{contract}_us"] = _median_us(lambda: native(xs), 300)
                    line[f"llvm_{contract}_compile_s"] = native.compile_seconds
                    worst = 0.0
                    for i in range(0, len(feats) - batch + 1, batch):
                        chunk = feats[i:i + batch]
                        worst = max(worst, float(np.abs(native(chunk) - eager_forward(params, chunk)).max()))
                    line[f"llvm_{contract}_diff_all"] = worst
                except Exception as error:
                    line[f"llvm_{contract}_refusal"] = f"{type(error).__name__}: {str(error)[:600]}"
            print(f"[{name}] batch {batch}: " + json.dumps(line, indent=1), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--game", choices=GAMES, required=True)
    t.add_argument("--workers", type=int, default=11)
    t.add_argument("--generations", type=int, default=24)
    t.add_argument("--pairs", type=int, default=20)
    t.add_argument("--sigma", type=float, default=0.05)
    t.add_argument("--lr", type=float, default=0.02)
    t.add_argument("--seconds", type=float, default=40.0)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--imit-episodes", type=int, default=10)
    t.add_argument("--final-seeds", type=int, default=16)
    t.add_argument("--final-seconds", type=float, default=60.0)
    c = sub.add_parser("compare")
    c.add_argument("--game", choices=GAMES, required=True)
    c.add_argument("--workers", type=int, default=11)
    c.add_argument("--final-seeds", type=int, default=16)
    c.add_argument("--final-seconds", type=float, default=60.0)
    b = sub.add_parser("bench")
    b.add_argument("--seconds", type=float, default=10.0)
    b.add_argument("--contracts", nargs="*", default=["develop"])
    args = parser.parse_args(argv)
    {"train": cmd_train, "compare": cmd_compare, "bench": cmd_bench}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
