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
#: A game with height above the board (bullet hell): the ship row and every
#: hostile row carry height and its rate too, and the policy steers height.
FEATURES_DEPTH = 7 + 7 * K + 4
#: bullet hell's five outputs: roll, throttle, fire, pitch, rudder (two sticks)
GAME_WIDTHS = {"space_invaders": (FEATURES, OUTPUTS), "bullet_hell": (FEATURES_DEPTH, 5)}


def widths_of(params: dict) -> tuple[int, int]:
    """(features, outputs) of a policy: its first and last layers say."""
    return int(params["W1"].shape[0]), int(params["W3"].shape[1])

GAMES = ("space_invaders", "bullet_hell")


def make_game(name: str):
    from meowdoku.invaders import SpaceInvaders
    from meowdoku.scroller import BulletScroller
    game = {"space_invaders": SpaceInvaders, "bullet_hell": BulletScroller}[name]("autopilot")
    game.learns = False          # training / evaluation episodes are not play to learn from
    return game


# ------------------------------------------------------------------ features
# One function per game, reading the session columns.  Layout (FEATURES wide):
#   0..4     ship x (-1..1 across the field), y (-1..1 down), vx, vy, shield charge
#   5..74    K hostile lanes by time-to-contact: dx, dy, dvx, dvy (relative to the
#            ship, scaled) and a kind flag (0 = padding)
#   75..78   field width, remaining fleet/enemies, lives, weapon ready

DX, DY, DV = 8.0, 10.0, 10.0


def _observe(game, hostile_kind, remaining, ready, height: float, depth: float | None = None) -> np.ndarray:
    """The observation; with ``depth`` (a field with height) the ship row is
    x, y, z, vx, vy, vz, shield and each hostile row dx, dy, dz, dvx, dvy, dvz,
    kind, ordered by three-dimensional time to contact."""
    col = game.session.column
    k = game.PLAYER_LANE
    alive, x, y, vx, vy = col("alive"), col("x"), col("y"), col("vx"), col("vy")
    field_w = float(col("field_w")[0])
    has_z = depth is not None
    z = col("z") if has_z else np.zeros_like(x)
    vz = col("vz") if has_z else np.zeros_like(x)
    head, row = (7, 7) if has_z else (5, 5)
    out = np.zeros(head + row * K + 4)
    if has_z:
        out[0:7] = (2.0 * x[k] / field_w - 1.0, 2.0 * y[k] / height - 1.0, 2.0 * z[k] / depth - 1.0,
                    vx[k] / DV, vy[k] / DV, vz[k] / DV, col("energy")[k])
    else:
        out[0:5] = (2.0 * x[k] / field_w - 1.0, 2.0 * y[k] / height - 1.0, vx[k] / DV, vy[k] / DV, col("energy")[k])
    lanes = np.concatenate([np.arange(a, b) for a, b in game.HOSTILE_LANES])
    lanes = lanes[alive[lanes] > 0.0]
    if len(lanes):
        dx, dy, dz = x[lanes] - x[k], y[lanes] - y[k], z[lanes] - z[k]
        dvx, dvy, dvz = vx[lanes] - vx[k], vy[lanes] - vy[k], vz[lanes] - vz[k]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        closing = -(dx * dvx + dy * dvy + dz * dvz) / np.maximum(dist, 1e-6)
        speed = np.maximum(closing, 0.25 * np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz) + 1e-3)
        order = np.argsort(np.maximum(dist - 0.5, 0.0) / speed, kind="stable")[:K]
        n = len(order)
        kinds = hostile_kind(col, lanes[order])
        if has_z:
            rows = np.stack([dx[order] / DX, dy[order] / DY, dz[order] / DX, dvx[order] / DV, dvy[order] / DV,
                             dvz[order] / DV, kinds], axis=1)
        else:
            rows = np.stack([dx[order] / DX, dy[order] / DY, dvx[order] / DV, dvy[order] / DV, kinds], axis=1)
        out[head:head + row * n] = rows.reshape(-1)
    lives_scale = float(getattr(game, "HEALTH_MAX", None) or 3.0)
    out[head + row * K:] = (field_w / 10.0, remaining(col), float(col("lives")[0]) / lives_scale, ready(col))
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
        sc.HEIGHT, getattr(sc, "DEPTH", None))


FEATURE_FUNCTIONS = {"space_invaders": invaders_features, "bullet_hell": scroller_features}


def controls_from(game, out: np.ndarray) -> dict[str, float]:
    """The policy's three outputs as the game's control columns."""
    values = {"move_axis": float(out[0]), "move_axis_y": float(out[1]) if "move_axis_y" in game.CONTROLS else 0.0,
              # A policy with a fourth output steers height; an older one holds it.
              "move_axis_z": float(out[3]) if len(out) > 3 else 0.0,
              "steer_x": -1.0, "steer_y": -1.0, "fire": 1.0 if out[2] > 0.0 else 0.0}
    if len(out) > 4:
        values["move_axis_r"] = float(out[4])          # a fifth output: the rudder
    # Controls the policy does not drive (a game's alternate fire, its bomb
    # style, ...) rest at the game's own initial value for them.
    rest = getattr(game, "_control_rest", None)
    if rest is None:
        columns = game.initial_columns()
        rest = game._control_rest = {name: float(columns[name][0]) for name in game.CONTROLS if name in columns}
    return {name: values[name] if name in values else rest.get(name, 0.0) for name in game.CONTROLS}


# ------------------------------------------------------------------- the net

def policy_forward(x, W1, b1, W2, b2, W3, b3):
    """x (N, FEATURES) -> (N, 3) in [-1, 1].  AbstractTensor arithmetic only,
    so the same text is traced for the NumPy product and the LLVM forward."""
    h = (x @ W1 + b1).tanh()
    h = (h @ W2 + b2).tanh()
    return (h @ W3 + b3).tanh()


def init_params(seed: int = 0, features: int = FEATURES, hidden=HIDDEN, outputs: int = OUTPUTS) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sizes = (features, *hidden, outputs)
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
        out = policy_forward(AT.tensor(np.asarray(x, dtype=np.float64).reshape(-1, widths_of(params)[0])),
                             *(AT.tensor(params[n]) for n in PARAMS))
        return np.asarray(out.numpy(), dtype=np.float64)


# NumPy lane: the forward traced through AbstractTensor and printed as a NumPy
# function by tape_to_source (the product LawPiece uses for the laws).

_PRODUCTS: dict = {}


def numpy_product(batch: int = 1, features: int = FEATURES, outputs: int = OUTPUTS):
    key = (batch, features, outputs)
    if key in _PRODUCTS:
        return _PRODUCTS[key]
    from src.common.tensors.autograd import autograd
    from src.common.tensors.numpy_backend import NumPyTensorOperations
    from src.common.tensors.accelerator_backends import tape_to_source
    from src.common.tensors.accelerator_backends.tape_to_source import emit_tape_source
    digest = hashlib.sha256((inspect.getsource(policy_forward) + inspect.getsource(tape_to_source)
                             + f"{features}{HIDDEN}{outputs}{batch}").encode()).hexdigest()[:16]
    entry = "autopilot_policy"
    path = BUILD / f"{entry}_b{batch}_{digest}.py"
    if not path.exists():
        like = init_params(features=features, outputs=outputs)
        sample = {"x": NumPyTensorOperations.tensor(np.zeros((batch, features))),
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
        out = function(np.asarray(x, dtype=np.float64).reshape(batch, features), *(params[n] for n in PARAMS))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return np.asarray(out, dtype=np.float64).reshape(batch, outputs)

    run.path = path
    _PRODUCTS[key] = run
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
        self.features, self.outputs = widths_of(params)
        name = f"autopilot_policy_b{self.batch}_{contract}"
        directory = Path(directory or BUILD / "native")
        started = time.perf_counter()
        with scoped_work_contract(PRESETS[contract]):
            program = SSATensorProgram(name)
            shapes = {"x": (self.batch, self.features), **{n: tuple(params[n].shape) for n in PARAMS}}
            bindings = {key: SSATensorOperations.input(program, shape) for key, shape in shapes.items()}
            action = policy_forward(bindings["x"], *(bindings[n] for n in PARAMS))
            self.forward = compile_native_graph_forward(
                action, bindings=bindings, source="meowdoku.autopilot.policy_forward",
                name=name, directory=directory / name / f"p{os.getpid()}_{next(_BUILDS)}")
        self.compile_seconds = time.perf_counter() - started
        ids = dict(self.forward.input_value_ids)
        self._execution = prepare_artifact_execution(self.forward.artifact, {
            **{ids[n]: np.ascontiguousarray(params[n]) for n in PARAMS}, ids["x"]: np.zeros((self.batch, self.features))})
        self._x = np.asarray(self._execution.buffers[ids["x"]])
        self._out = np.asarray(self._execution.buffers[self.forward.output_value_ids[0]])
        self._weights = {n: np.asarray(self._execution.buffers[ids[n]]) for n in PARAMS}

    def set_weights(self, params: dict[str, np.ndarray]) -> None:
        for n in PARAMS:
            np.copyto(self._weights[n].reshape(-1), np.asarray(params[n], dtype=np.float64).reshape(-1))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._x.reshape(-1)[:] = np.asarray(x, dtype=np.float64).reshape(-1)
        self._execution.run()
        return self._out.reshape(self.batch, self.outputs).copy()


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


def fitting_params(name: str) -> dict[str, np.ndarray] | None:
    """The game's saved weights, only if they fit its observation and outputs
    (a law that grew an axis needs retraining; old weights are not forced in)."""
    params = load_params(weights_path(name))
    if params is None:
        return None
    features, outputs = widths_of(params)
    want_f, want_o = GAME_WIDTHS[name]
    if features == want_f and outputs < want_o:
        # The game grew controls (new outputs): the learned layers carry over
        # and each new output starts at zero, to be learned from play.
        grow = want_o - outputs
        params = {**params, "W3": np.concatenate([params["W3"], np.zeros((params["W3"].shape[0], grow))], axis=1),
                  "b3": np.concatenate([params["b3"], np.zeros((1, grow))], axis=1)}
    if widths_of(params) != GAME_WIDTHS[name]:
        return None
    return params


def preload() -> list[str]:
    """Load every trained weight file that fits (the server calls this at start)."""
    loaded = []
    for name in GAMES:
        params = fitting_params(name)
        if params is not None:
            _PILOT_WEIGHTS[name] = params
            numpy_product(1, *GAME_WIDTHS[name])
            loaded.append(name)
    return loaded


class Pilot:
    """The compiled policy deciding one game's controls (NumPy lane; the LLVM
    lane when ``MEOWDOKU_PILOT=llvm`` and it compiles)."""

    def __init__(self, name: str, params: dict[str, np.ndarray]):
        self.name, self.params = name, params
        self.features = FEATURE_FUNCTIONS[name]
        self.lane = "numpy"
        product = numpy_product(1, *widths_of(params))
        self.run = lambda x: product(x, self.params)
        if os.environ.get("MEOWDOKU_PILOT", "numpy") == "llvm":
            try:
                native = NativePolicy(params, batch=1)
                self.run, self.lane = native, "llvm"
            except Exception as error:      # keep NumPy; say why
                print(f"autopilot: llvm lane refused ({type(error).__name__}: {str(error)[:200]}); numpy", flush=True)

    def controls(self, game) -> dict[str, float]:
        return controls_from(game, self.run(self.features(game))[0])


class _MirroredView:
    """The board as a piloted enemy sees it: the patch of the sphere about
    ITSELF, its heading up the screen (sphere_motion.project, the projection
    the law draws the player's patch with), itself as "the ship", the player
    as an enemy craft and the player's shots and missiles as bullets, its
    health as its vitals -- so the policy trained to survive and score as the
    player flies it unchanged."""

    def __init__(self, game, lane: int):
        from meowdoku import scroller as sc
        self.game = game.game
        self.PLAYER_LANE = lane
        self.HOSTILE_LANES = ((0, 1), sc.SHOTS, sc.MISSILES)
        self.HEALTH_MAX = 1.0
        self._sc, self._real, self._lane = sc, game.session, lane
        self.session = self
        self._patch = self._project()

    def _project(self) -> dict[str, np.ndarray]:
        from meowdoku import sphere_motion as sm
        from src.common.tensors import AbstractTensor
        sc, real, k = self._sc, self._real, self._lane
        T = lambda a: AbstractTensor.get_tensor(np.asarray(a, dtype=np.float64))  # noqa: E731
        col = {name: real.column(name) for name in ("ux", "uy", "uz", "hx", "hy", "hz", "spd", "field_w")}
        u = (T(col["ux"]), T(col["uy"]), T(col["uz"]))
        h = (T(col["hx"]), T(col["hy"]), T(col["hz"]))
        uk = tuple(T(col[n][k]) for n in ("ux", "uy", "uz"))
        hk = tuple(T(col[n][k]) for n in ("hx", "hy", "hz"))
        X, Y = sm.project(u, uk, hk, sc.RADIUS)
        VX, VY = sm.patch_velocity(h, T(col["spd"]), X, Y, hk, T(col["spd"][k]), uk, T(0.0))
        cx, cy = 0.5 * float(col["field_w"][0]), sc.PLAYER_Y
        out = {"x": cx + np.asarray(X.numpy()), "y": cy - np.asarray(Y.numpy()),
               "vx": np.asarray(VX.numpy()), "vy": -np.asarray(VY.numpy())}
        out["x"][k], out["y"][k], out["vx"][k], out["vy"][k] = cx, cy, 0.0, 0.0
        return out

    def column(self, name: str) -> np.ndarray:
        sc, k = self._sc, self._lane
        if name in self._patch:
            return np.array(self._patch[name], dtype=np.float64, copy=True)
        col = np.array(self._real.column(name), dtype=np.float64, copy=True)
        if name == "energy":
            col[k] = 1.0
        elif name == "lives":
            hp, top = float(self._real.column("hp")[k]), sc.BOSS_HP if k == sc.BOSS_LANE else sc.ACE_HP
            col[...] = max(0.0, min(1.0, hp / top))
        elif name == "timer":
            col[0] = self._real.column("timer")[k]
        elif name == "slot":
            # The player registers as a bad guy (an enemy craft to line up
            # on); its shots and missiles as incoming bullets to dodge.
            col[(col == sc.SHOT) | (col == sc.MISSILE_SLOT)] = sc.BULLET
            col[0] = sc.ENEMY
        return col


def pilot_controls(game, lane: int) -> tuple[float, float, float, float]:
    """(turn, throttle axis, height axis, fire) for a piloted enemy: the
    game's policy on the craft's own patch, its answers as the player's
    controls mean them (x axis turns, y axis throttles)."""
    name = game.game
    params = _PILOT_WEIGHTS.get(name) or fitting_params(name)
    if params is None:
        return 0.0, 0.0, 0.0, 1.0
    _PILOT_WEIGHTS.setdefault(name, params)
    view = _MirroredView(game, lane)
    out = numpy_product(1, *widths_of(params))(FEATURE_FUNCTIONS[name](view), params)[0]
    turn = float(out[0]) + (float(out[4]) if len(out) > 4 else 0.0)     # roll and rudder both turn a pilot
    return (max(-1.0, min(1.0, turn)), float(out[1]), float(out[3]) if len(out) > 3 else 0.0, 1.0 if out[2] > 0.0 else 0.0)


def pilot_for(game) -> Pilot:
    name = game.game
    if name not in _PILOT_WEIGHTS:
        params = fitting_params(name)
        if params is None:
            raise ValueError(f"no trained autopilot for {name} fitting its current observation "
                             f"{GAME_WIDTHS[name]}: train it ({weights_path(name)})")
        _PILOT_WEIGHTS[name] = params
    return Pilot(name, _PILOT_WEIGHTS[name])


def available(name: str) -> bool:
    return name in _PILOT_WEIGHTS or fitting_params(name) is not None


# ---------------------------------------------------------- scripted baselines

def dodger_controls(game) -> tuple[float, ...]:
    """The scripted dodger: (move_axis, move_axis_y, fire[, move_axis_z])."""
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


_ACTIONS_3D = np.array([(ax, ay, az) for ax in (-1.0, 0.0, 1.0) for ay in (-1.0, 0.0, 1.0)
                        for az in (-1.0, 0.0, 1.0)])


def _scroller_dodger(game):
    """Bullet hell teacher: 27 candidate moves (x, y, height) rolled forward
    0.6 s against straight-line hostiles; a hit needs depth overlap too.
    Prefers to sit under / level with the nearest enemy so its shots land."""
    from meowdoku import scroller as sc
    col = game.session.column
    alive, x, y, vx, vy, slot = col("alive"), col("x"), col("y"), col("vx"), col("vy"), col("slot")
    z, vz, rz = col("z"), col("vz"), col("rz")
    field_w = float(col("field_w")[0])
    hostile = np.concatenate([np.arange(a, b) for a, b in game.HOSTILE_LANES])
    hostile = hostile[alive[hostile] > 0.0]
    steps, n = 12, len(_ACTIONS_3D)
    t = np.arange(1, steps + 1) * 0.05
    pos = np.zeros((n, steps, 3))
    v = np.tile([vx[0], vy[0], vz[0]], (n, 1))
    p = np.tile([x[0], y[0], z[0]], (n, 1))
    lo = [sc.MARGIN, sc.MARGIN, sc.Z_MARGIN]
    hi = [field_w - sc.MARGIN, sc.HEIGHT - sc.MARGIN, sc.DEPTH - sc.Z_MARGIN]
    for i in range(steps):
        v = v + (_ACTIONS_3D * sc.SPEED - v) * (0.05 / (sc.TAU + 0.05))
        p = np.clip(p + v * 0.05, lo, hi)
        pos[:, i] = p
    cost = np.zeros(n)
    if len(hostile):
        bx = x[hostile][None, :] + vx[hostile][None, :] * t[:, None]
        by = y[hostile][None, :] + vy[hostile][None, :] * t[:, None]
        bz = z[hostile][None, :] + vz[hostile][None, :] * t[:, None]
        ddx = np.abs(pos[:, :, 0, None] - bx[None])
        ddy = np.abs(pos[:, :, 1, None] - by[None])
        ddz = np.abs(pos[:, :, 2, None] - bz[None])
        radius = np.where(slot[hostile] == sc.ENEMY, 0.75, 0.42)[None, None, :]
        depth_r = (rz[0] + rz[hostile])[None, None, :] + 0.1
        hit = (ddx < radius) & (ddy < radius + 0.15) & (ddz < depth_r)
        cost += 100.0 * hit.sum(axis=(1, 2))
        close = np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
        cost += 1.5 * np.maximum(1.4 - close, 0.0).sum(axis=(1, 2)) / steps
    enemies = np.flatnonzero(alive[sc.ENEMIES[0]:sc.ENEMIES[1]] > 0.0) + sc.ENEMIES[0]
    final = pos[:, -1]
    if len(enemies):
        nearest = enemies[np.argmin(np.abs(x[enemies] - x[0]) + 0.3 * np.abs(y[enemies] - y[0]))]
        cost += 0.04 * np.abs(final[:, 0] - x[nearest])
        cost += 0.05 * np.abs(final[:, 2] - z[nearest])
    cost += 0.02 * np.abs(final[:, 1] - (sc.HEIGHT - 3.0))
    cost += 0.01 * np.abs(final[:, 0] - field_w / 2.0)
    best = _ACTIONS_3D[int(np.argmin(cost))]
    return float(best[0]), float(best[1]), 1.0, float(best[2])


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
        decision = dodger_controls(game)
        return np.array([decision[0], decision[1], 1.0, *decision[3:]])
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

    idle = {"move_axis": 0.0, "move_axis_y": 0.0, "move_axis_z": 0.0, "steer_x": -1.0, "steer_y": -1.0, "fire": 0.0}
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
    run = numpy_product(1, *widths_of(params))

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

        outputs = GAME_WIDTHS[name][1]

        def act(game, f, rng=rng, hold=hold):
            decision = dodger_controls(game)
            a, b, fire = decision[:3]
            c = decision[3] if len(decision) > 3 else 0.0
            xs.append(f)
            label = [a * 0.95, b * 0.95, 0.9 if fire > 0 else -0.9] + ([c * 0.95] if outputs > 3 else [])
            ys.append(label)
            if hold["left"] > 0:
                hold["left"] -= 1
                return hold["out"]
            if rng.random() < 0.04:                                  # drift off the dodger's path for a moment
                hold["left"] = int(rng.integers(3, 12))
                hold["out"] = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 1.0]
                                       + ([rng.uniform(-1, 1)] if outputs > 3 else []))
            return np.array([a, b, 1.0] + ([c] if outputs > 3 else []))
        run_episode(name, act, seed + e, seconds)
    return np.array(xs), np.array(ys)


def imitate(name: str, epochs: int = 60, episodes: int = 10, seconds: float = 30.0, seed: int = 0):
    started = time.perf_counter()
    x, y = collect_imitation(name, episodes, seconds)
    collected = time.perf_counter() - started
    features, outputs = GAME_WIDTHS[name]
    trainer = PolicyTrainer(init_params(seed, features=features, outputs=outputs), lr=2e-3)
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
            save_params(unflatten(best[1], params), weights_path(name), game=name, features=widths_of(params)[0], k=K)
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
            x = feats[:batch] if len(feats) >= batch else np.resize(feats, (batch, feats.shape[1]))
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


# ------------------------------------------------------- learning from play
# Every decision the server lands in an autopilot game is traced as
# (observation, controls used) -- the observation the policy sees and the
# three outputs it would have had to produce.  When a game ends:
#   * a HUMAN episode is appended to the kept records (human_<game>.npz) and
#     the policy post-trains on it (with a replay of earlier human episodes,
#     so one game does not overwrite the rest), then the weights are saved;
#   * a DEMO episode trains the policy only while demo learning is on for that
#     game, and only if it scored at least the demo's running average
#     (self-imitation of its better runs: reinforcement, not self-copying).
# Training runs on a background thread; the live pilots read the same weights
# dict, which is updated in place under LEARN_LOCK.

import threading as _threading

LEARN_LOCK = _threading.Lock()
DEMO_LEARNING: dict[str, bool] = {name: False for name in GAMES}
LEARN_STATUS: dict[str, dict] = {name: {} for name in GAMES}
_DEMO_SCORES: dict[str, list[float]] = {name: [] for name in GAMES}
POST_EPOCHS, REPLAY_ROWS, MIN_ROWS = 4, 4096, 32
#: SPANS OF ACTION, not whole tapes: the last SPAN_ROWS decisions wait in a
#: ring; a rise in score keeps them (the lead-up to a success), damage drops
#: them; kept rows go to the learner every FLUSH_ROWS, mid-game.  Human
#: records keep at most RECORD_ROWS (the newest).
SPAN_ROWS, FLUSH_ROWS, RECORD_ROWS = 48, 256, 20000
#: The page's learning panel: the last HISTORY runs (score, kept or not) and
#: training passes (loss), per game.
HISTORY = 30
LEARN_HISTORY: dict[str, list[dict]] = {name: [] for name in GAMES}
TRAINING: dict[str, bool] = {name: False for name in GAMES}


def _history(name: str, **entry) -> None:
    rows = LEARN_HISTORY.setdefault(name, [])
    rows.append({"at": time.strftime("%H:%M:%S"), **entry})
    del rows[:-HISTORY]


# Learning state survives a server restart: the demo-learning switches, the
# run history, the demo's running scores and the last training status.
LEARNING_STATE = BUILD / "learning_state.json"


def _save_learning() -> None:
    try:
        BUILD.mkdir(parents=True, exist_ok=True)
        partial = LEARNING_STATE.with_suffix(".partial.json")
        partial.write_text(json.dumps({"demo_learning": DEMO_LEARNING, "history": LEARN_HISTORY,
                                       "demo_scores": {k: v[-20:] for k, v in _DEMO_SCORES.items()},
                                       "status": LEARN_STATUS}), encoding="utf-8")
        partial.replace(LEARNING_STATE)
    except OSError as error:
        print(f"[autopilot] learning state not saved: {error}", flush=True)


def load_learning() -> None:
    """Restore the learning state (the server calls this at start)."""
    try:
        data = json.loads(LEARNING_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for name in GAMES:
        DEMO_LEARNING[name] = bool(data.get("demo_learning", {}).get(name, False))
        LEARN_HISTORY[name] = list(data.get("history", {}).get(name, []))[-HISTORY:]
        _DEMO_SCORES[name] = [float(s) for s in data.get("demo_scores", {}).get(name, [])]
        LEARN_STATUS[name] = dict(data.get("status", {}).get(name, {}))


def human_records_path(game: str) -> Path:
    """Kept human episodes, versioned by the observation width they were taken
    with (a law that grows an axis starts a new record set)."""
    features, outputs = GAME_WIDTHS[game]
    return BUILD / (f"human_{game}_f{features}.npz" if outputs == 3 else f"human_{game}_f{features}_o{outputs}.npz")


def target_from(game, controls: dict) -> np.ndarray:
    """The controls a decision used, as the policy's three outputs.  A held
    pointer is its axis equivalent under the motion law: the law commands
    clamp((target - x) / reach, +-speed), i.e. axis = clamp(gap / speed, +-1)."""
    motion = game.motion() or {}
    out = np.zeros(GAME_WIDTHS.get(game.game, (FEATURES, OUTPUTS))[1])
    k = game.PLAYER_LANE
    for i, (axis_name, steer_name, pos) in enumerate((("move_axis", "steer_x", "x"), ("move_axis_y", "steer_y", "y"))):
        steer = float(controls.get(steer_name, -1.0))
        if steer >= 0.0 and motion.get(pos) and k is not None:
            gap = (steer - float(game.session.column(pos)[k])) / float(motion.get("reach", 1.0))
            out[i] = max(-1.0, min(1.0, gap / float(motion.get("speed", 1.0))))
        else:
            out[i] = max(-1.0, min(1.0, float(controls.get(axis_name, 0.0))))
    out[2] = 1.0 if float(controls.get("fire", 0.0)) > 0.0 else -1.0
    if len(out) > 3:
        out[3] = max(-1.0, min(1.0, float(controls.get("move_axis_z", 0.0))))
    if len(out) > 4:
        out[4] = max(-1.0, min(1.0, float(controls.get("move_axis_r", 0.0))))
    return out


class PlayTrace:
    """The spans of an episode worth learning from (see SPAN_ROWS): a ring of
    recent decisions, and the kept rows -- lead-ups to a rise in score."""

    def __init__(self):
        import collections
        self.ring = collections.deque(maxlen=SPAN_ROWS)
        self.x: list[np.ndarray] = []       # kept rows
        self.y: list[np.ndarray] = []
        self.demo = None                    # True / False once the first decision lands
        self.last = None                    # (score, lives, shield) at the last decision
        self.spans = 0

    def record(self, game, controls: dict, demo: bool) -> bool:
        """One decision; True when enough kept rows wait for the learner."""
        if game.game not in FEATURE_FUNCTIONS or game.session.state is None:
            return False
        if self.demo is None:
            self.demo = demo
        elif self.demo != demo:            # who drives changed mid-game: keep one kind per episode
            return False
        col = game.session.column
        k = game.PLAYER_LANE or 0
        now = (float(col("score")[0]), float(col("lives")[0]),
               float(col("energy")[k]) if "energy" in game.session.columns else 0.0)
        if self.last is not None:
            if now[1] < self.last[1] or now[2] < self.last[2] - 1e-9:
                self.ring.clear()           # hurt: the lead-up is not one to imitate
            elif now[0] > self.last[0] and self.ring:
                for x, y in self.ring:      # scored: keep the lead-up
                    self.x.append(x); self.y.append(y)
                self.ring.clear()
                self.spans += 1
        self.last = now
        self.ring.append((np.asarray(FEATURE_FUNCTIONS[game.game](game), dtype=np.float64).reshape(-1),
                          target_from(game, controls)))
        return len(self.x) >= FLUSH_ROWS

    def take(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The kept rows (and forget them)."""
        if not self.x:
            return None
        x, y = np.stack(self.x), np.stack(self.y)
        self.x, self.y = [], []
        return x, y


def _fit(params: dict, x: np.ndarray, y: np.ndarray, epochs: int) -> tuple[dict, float]:
    """The fit itself, run in a worker PROCESS (``_TRAINER``): a training pass
    on a thread held the game server's interpreter lock and stalled every
    game's ticks for its duration."""
    trainer = PolicyTrainer(params, lr=1e-3)
    rng, loss = np.random.default_rng(), float("nan")
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for at in range(0, len(x), 256):
            rows = order[at:at + 256]
            loss = trainer.step(x[rows], y[rows])
    return trainer.params(), loss


_TRAINER = None


def _trainer_pool():
    global _TRAINER
    if _TRAINER is None:
        import concurrent.futures
        import multiprocessing
        _TRAINER = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    return _TRAINER


def _post_train(name: str, x: np.ndarray, y: np.ndarray, label: str, kept_tail: int = 0) -> None:
    """Fit the live policy on ``x -> y`` (plus replayed human rows), update
    it in place and save it.  Runs on its own thread, which waits on the
    fit in the trainer process."""
    TRAINING[name] = True
    try:
        with LEARN_LOCK:
            live = _PILOT_WEIGHTS.get(name) or fitting_params(name)
            if live is None:
                return
            params = {n: v.copy() for n, v in live.items()}
        records = human_records_path(name)
        if records.exists():
            data = np.load(records)
            rx, ry = data["x"], data["y"]
            if kept_tail:                       # this episode is already the records' tail
                rx, ry = rx[:-kept_tail], ry[:-kept_tail]
            if len(rx) > REPLAY_ROWS:
                pick = np.random.default_rng().choice(len(rx), REPLAY_ROWS, replace=False)
                rx, ry = rx[pick], ry[pick]
            x, y = np.concatenate([x, rx]), np.concatenate([y, ry])
        fitted, loss = _trainer_pool().submit(_fit, params, x, y, POST_EPOCHS).result()
        with LEARN_LOCK:
            target = _PILOT_WEIGHTS.setdefault(name, fitted)
            for n in PARAMS:
                target[n] = fitted[n]          # the live pilots read this dict
            save_params(target, weights_path(name))
        LEARN_STATUS[name] = {"last": label, "rows": int(len(x)), "loss": round(loss, 4), "at": time.strftime("%H:%M:%S")}
        _history(name, kind="trained", loss=round(loss, 5), rows=int(len(x)), label=label)
        _save_learning()
        print(f"[autopilot] {name}: post-trained on {label} ({len(x)} rows, loss {loss:.4f})", flush=True)
    except Exception as error:  # noqa: BLE001 - learning must never take the game down
        LEARN_STATUS[name] = {"last": f"failed: {type(error).__name__}: {str(error)[:80]}"}
        print(f"[autopilot] {name}: post-training failed: {error}", flush=True)
    finally:
        TRAINING[name] = False


def finish_episode(game, trace: PlayTrace, final: bool = True) -> None:
    """Hand a trace's kept spans to the learner: at the game's end, or
    mid-game once FLUSH_ROWS have gathered (``final`` False)."""
    name = game.game
    if name not in FEATURE_FUNCTIONS or trace.demo is None or len(trace.x) < MIN_ROWS:
        return
    x, y = trace.take()
    score = float(game.session.column("score")[0]) if game.session.state is not None else 0.0
    if not trace.demo:
        path = human_records_path(name)
        with LEARN_LOCK:
            if path.exists():
                old = np.load(path)
                kx, ky = np.concatenate([old["x"], x]), np.concatenate([old["y"], y])
                episodes = np.concatenate([old["episode_score"], [score]])
            else:
                kx, ky, episodes = x, y, np.asarray([score])
            kx, ky = kx[-RECORD_ROWS:], ky[-RECORD_ROWS:]          # the newest rows only
            BUILD.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial.npz")
            np.savez(partial, x=kx, y=ky, episode_score=episodes)
            partial.replace(path)
        _HUMAN_GAMES[name] = int(len(episodes))
        label = f"human {'game' if final else 'spans'} {len(episodes)} (score {score:.0f}, {len(x)} span rows)"
        kept_tail = len(x)
    else:
        scores = _DEMO_SCORES[name]
        baseline = float(np.mean(scores[-10:])) if scores else -1.0
        if final:
            scores.append(score)             # only finished runs make the running average
        # mid-game spans are successes by construction (lead-ups to scoring);
        # a finished run is kept when it beats the running average
        kept = bool(DEMO_LEARNING.get(name)) and (not final or score >= baseline)
        if DEMO_LEARNING.get(name):
            _history(name, kind="run" if final else "spans", score=score, average=max(baseline, 0.0), kept=kept)
            _save_learning()
        if not kept:
            LEARN_STATUS[name] = {**LEARN_STATUS.get(name, {}),
                                  "demo": f"demo run {score:.0f} vs average {max(baseline, 0):.0f}"
                                          + ("" if DEMO_LEARNING.get(name) else " (learning off)")}
            return
        label = (f"demo run (score {score:.0f} >= average {max(baseline, 0):.0f})" if final
                 else f"demo spans ({len(x)} rows of lead-ups to scoring)")
        kept_tail = 0
    _threading.Thread(target=_post_train, args=(name, x, y, label, kept_tail), name=f"post-train-{name}", daemon=True).start()


_HUMAN_GAMES: dict[str, int] = {}


def learning_summary(name: str) -> dict:
    if name not in _HUMAN_GAMES:                  # read the record file once, then count in memory
        records = human_records_path(name)
        try:
            _HUMAN_GAMES[name] = int(len(np.load(records)["episode_score"])) if records.exists() else 0
        except Exception:  # noqa: BLE001
            _HUMAN_GAMES[name] = 0
    return {"demo_learning": bool(DEMO_LEARNING.get(name)), "human_games": _HUMAN_GAMES[name],
            "training": bool(TRAINING.get(name)), "history": list(LEARN_HISTORY.get(name, ())),
            **LEARN_STATUS.get(name, {})}
