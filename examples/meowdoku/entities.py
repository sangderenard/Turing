"""Entity games: free-board state machines driven by the dt system on wall time.

Meowdoku's session lands one move as one round.  An entity game is the same
dt system used for continuous play: the state is a span of entity LANES (one
lane per slot an entity can occupy), every column a contiguous float64 span
the dt system owns, and one server tick lands the wall time since the last
tick as one round.  The law publishes ``dt_limit`` (how far its fastest
entity may move before a collision could be stepped over), so the dt
controller subdivides the round into as many law calls as tileless motion
needs.  The board has no cells, only positions, and the dt system is what
keeps them honest.

A game declares:

- a piece (``llvm_dt_system.PIECE_API``) whose law reads the columns and
  writes ``<name>_next``;
- ``initial_columns()``: every column at round zero;
- ``CONTROLS``: the control columns a shell writes before a round (here the
  page's held keys and pointer, as the tick payload carries them);
- ``entities()``: the visible entities as ``{id, x, y, vx, vy, mesh, color,
  size, roll, spin}`` in board units (``x`` across, ``y`` down the board),
  for the page to draw and extrapolate between ticks.

PREDICTION AND RECONCILIATION (the classic client-side prediction loop).
The server is the authority over every lane.  The player's own ship moves by
``entity_motion.damped_axis``, which the page also runs, compiled to
WebAssembly from the same text.  The page records its input as INTERVALS --
what was held and for how long, a new interval whenever the controls change
-- and each tick sends the intervals it closed, numbered ``seq``.  The
server lands each interval as its own round with exactly those controls (in
a world of one player, the player's input timing is the clock: the session
advances by the page's measured durations, each capped at ``max_frame``).
The reply carries ``ack`` (the last ``seq`` landed) and ``you`` (the ship's
lane after it); the page resets its ship to ``you`` and replays every
interval after ``ack`` plus the one still open, so its ship answers input at
once and lands where the server puts it -- the only difference left is the
step size each side integrates with.  A tick without intervals (an
unpredicting page) lands wall time as before.  Every other lane is drawn
from the server's state, carried forward by its velocity.  A game declares
``PLAYER_LANE`` and ``MOTION`` (the motion law's constants and the walls
per axis); ``state()`` publishes them as ``predict``.

Collisions over many lanes: ``block_hits`` compares a slice of lanes
against another (bullets against the player, shots against enemies)
instead of every lane against every lane.

Rewind and reset are not game code: the session snapshots the state with
``copy_shallow`` every ``rewind_every`` seconds of play and ``undo`` restores
one from about two seconds back (the pair ``run_superstep`` uses for in-round
rollback); ``reset`` restores round zero.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import math
import numpy as np

from meowdoku.games import Game
from src.common.tensors import AbstractTensor
from meowdoku.session import ROOT  # noqa: F401  (puts the repo and examples on sys.path)
from src.common.dt_system.dt_controller import STController, Targets  # noqa: E402
from llvm_dt_system import dt_system  # noqa: E402


# ---------------------------------------------------------------- law helpers
# Span idioms shared by entity laws; every operand is an AbstractTensor span.

def as_float(mask):
    return mask.to_dtype("float64")


def noise(value):
    """Deterministic per-lane noise in [0, 1): the law's dice."""
    raw = (value * 12.9898 + 78.233).sin() * 43758.5453
    return raw - raw.floor()


def first_of(mask, lane):
    """The lowest lane where ``mask`` holds (all false: none)."""
    return mask & (lane == AbstractTensor.where(mask, lane, 1.0e9).min())


def block_hits(x, y, rx, ry, live, harms, rows: tuple[int, int], cols: tuple[int, int]):
    """Hit counts of the lanes in ``rows`` by the lanes in ``cols``, as a
    full-length span (zero outside ``rows``).

    ``harms`` is a (len(rows), len(cols)) boolean span of which pairs may
    hurt (or None: all).  Overlap is box against box with half extents
    ``rx``, ``ry``.  The block is the only part of the lane-by-lane matrix
    computed: hundreds of bullets against one ship cost hundreds of tests,
    not hundreds squared."""
    r0, r1 = rows
    c0, c1 = cols
    n = int(x.shape[0])
    a, b = r1 - r0, c1 - c0
    touch = (((x[r0:r1].reshape((a, 1)) - x[c0:c1].reshape((1, b))).abs()
              < (rx[r0:r1].reshape((a, 1)) + rx[c0:c1].reshape((1, b))))
             & ((y[r0:r1].reshape((a, 1)) - y[c0:c1].reshape((1, b))).abs()
                < (ry[r0:r1].reshape((a, 1)) + ry[c0:c1].reshape((1, b))))
             & live[r0:r1].reshape((a, 1)) & live[c0:c1].reshape((1, b)))
    if harms is not None:
        touch = touch & harms
    counts = as_float(touch).sum(dim=1)
    parts = [x[:r0] * 0.0] if r0 > 0 else []
    parts.append(counts)
    if r1 < n:
        parts.append(x[r1:] * 0.0)
    return AbstractTensor.cat(parts, dim=0) if len(parts) > 1 else counts


class EntitySession:
    """One persistent dt-system state over entity lanes, advanced on demand."""

    def __init__(self, piece, columns: dict[str, np.ndarray], controls: tuple[str, ...], *,
                 rewind_every: float = 0.5, rewind_depth: int = 40):
        self.piece = piece
        self.pieces = [piece]
        self.columns = columns
        self.controls = tuple(controls)
        self.targets = Targets(cfl=0.5, div_max=1e9, mass_max=1e9, energy_exchange_fraction=0.2)
        self.controller = STController(dt_min=1e-6)
        self.state = None
        self.dt = 1.0 / 60.0
        self.rewind_every = float(rewind_every)
        self.rewind_depth = int(rewind_depth)
        self.snapshots: list = []
        self.since_snapshot = 0.0
        self.telemetry = None
        self.last_round = (0.0, 0)          # (seconds landed, law calls) of the last advance

    def control(self, values: dict[str, float]) -> None:
        """Write the control columns (broadcast to every lane) for the next round."""
        for name in self.controls:
            if name in values:
                target = self.columns[name] if self.state is None else getattr(self.state, name)
                target[...] = float(values[name])

    def advance(self, seconds: float) -> None:
        """Land ``seconds`` of play as one round; the controller subdivides it."""
        if seconds <= 0.0:
            return
        before = self.calls()
        self.state, self.controller, results = dt_system(
            self.pieces, self.columns, rounds=1, round_dt=float(seconds), dx=1.0,
            targets=self.targets, controller=self.controller, state=self.state,
            # Play is one continuous run cut into ticks: the next tick's first
            # attempt is the controller's last proposal, not the whole window
            # (dt_system's first round has no ceiling of its own).
            dt_initial=None if self.state is None else min(self.dt, float(seconds)))
        advanced, self.dt, telemetry = results[0]
        if abs(float(advanced) - seconds) > 1e-9 * max(1.0, seconds):
            raise RuntimeError(f"a tick must land as one window: advanced={advanced} of {seconds}")
        self.telemetry = telemetry
        self.last_round = (float(seconds), self.calls() - before)
        self.since_snapshot += seconds
        if not self.snapshots or self.since_snapshot >= self.rewind_every:
            self.snapshots.append(self.state.copy_shallow())
            del self.snapshots[:-self.rewind_depth]
            self.since_snapshot = 0.0

    def rewind(self, steps: int = 4) -> None:
        """Back about ``steps * rewind_every`` seconds: the dt system's own restore."""
        if self.state is None or not self.snapshots:
            return
        keep = max(1, len(self.snapshots) - int(steps))
        snapshot = self.snapshots[keep - 1]
        del self.snapshots[keep:]
        self.state.restore(snapshot)
        self.since_snapshot = 0.0

    def reset(self, fresh: dict[str, np.ndarray]) -> None:
        if self.state is None:
            self.columns.update(fresh)
            return
        names, _schedule = self.state.bound_pieces
        lanes = int(next(iter(fresh.values())).shape[0])
        self.state.restore(tuple(fresh[name] if name in fresh else np.zeros(lanes)
                                 for name in (*names, "dt")))
        self.snapshots = []
        self.since_snapshot = 0.0

    def column(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self.state, name) if self.state is not None else self.columns[name])

    def calls(self) -> int:
        """How many times the dt system has called the law, from its own ledger."""
        if self.state is None:
            return 0
        return int(self.state.wall_cost_ledger.report()["participants"][self.piece.entry]["calls"])


# ---------------------------------------------------------------- law pieces

#: Where compiled law products are kept (ignored by git): one file per law and
#: source digest, so an unchanged law loads its product and a changed one
#: recompiles.
LAW_PRODUCTS = Path(__file__).resolve().parents[2] / "build" / "meowdoku" / "laws"
#: The lane a law runs in.  ``numpy``: the law traced through AbstractTensor
#: (the translatable source) and printed as a standalone NumPy function by
#: ``tape_to_source`` (the product).  ``eager``: the AbstractTensor law itself.
#: The ladder above NumPy (WebAssembly in the page, LLVM over the socket,
#: Eigen) slots in here as further lanes.
LAW_LANE = os.environ.get("MEOWDOKU_LAW", "numpy")


class LawPiece:
    """An entity law as a dt-system piece (``llvm_dt_system.PIECE_API``).

    A game declares ``entry``, ``law`` (the AbstractTensor step, the source),
    ``argument_names``, ``output_names`` and ``batch``.  ``compile(columns)``
    makes the product for ``LAW_LANE``; ``__call__`` runs the product when
    there is one and the source otherwise."""

    entry: str = ""
    law = None
    argument_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()

    def __init__(self, batch: int):
        self.batch = int(batch)
        self._bound = None
        self.product = None
        self.lane = "eager"
        self.product_path: Path | None = None

    def instantiate(self, columns, outputs=None):
        self._bound = tuple(columns[name] for name in self.argument_names)
        return ()

    def _sources(self) -> str:
        """Everything the product is a function of: the law's module, the
        engine helpers and motion law it calls, and the printer."""
        import inspect
        from src.common.tensors.accelerator_backends import tape_to_source
        from meowdoku import entity_motion
        modules = (inspect.getmodule(type(self).law), sys.modules[__name__], entity_motion, tape_to_source)
        return "\0".join(inspect.getsource(module) for module in modules)

    def compile(self, columns: dict[str, np.ndarray]) -> None:
        if LAW_LANE != "numpy":
            return
        import hashlib
        import importlib.util
        digest = hashlib.sha256(self._sources().encode("utf-8")).hexdigest()[:16]
        path = LAW_PRODUCTS / f"{self.entry}_{digest}.py"
        if not path.exists():
            from src.common.tensors.autograd import autograd
            from src.common.tensors.numpy_backend import NumPyTensorOperations
            from src.common.tensors.accelerator_backends.tape_to_source import emit_tape_source
            sample = {name: NumPyTensorOperations.tensor(np.asarray(
                columns[name] if name in columns else np.full(self.batch, 1.0 / 60.0), dtype=np.float64))
                for name in self.argument_names}
            with autograd.forward_capture() as tape:
                produced = type(self).law(*sample.values())
            source = emit_tape_source(tape, sample, dict(zip(self.output_names, produced)),
                                      backend="numpy", function_name=self.entry)
            header = (f'"""{self.entry}: compiled NumPy product of {type(self).law.__module__}.{type(self).law.__name__}.\n\n'
                      f"Generated by AbstractTensor tape tracing + tape_to_source (numpy dialect) from source digest\n"
                      f'{digest}.  Do not edit: change the law and it recompiles.\n"""\n')
            LAW_PRODUCTS.mkdir(parents=True, exist_ok=True)
            path.write_text(header + source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(f"meowdoku_law_{self.entry}_{digest}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.product, self.lane, self.product_path = getattr(module, self.entry), "numpy", path

    def __call__(self, *columns):
        if len(columns) != len(self.argument_names):
            raise TypeError(f"{self.entry}: takes {len(self.argument_names)} columns, got {len(columns)}")
        if self.product is not None:
            produced = self.product(*(np.asarray(column, dtype=np.float64) for column in columns))
            return tuple(np.broadcast_to(np.asarray(value, dtype=np.float64), (self.batch,)).copy()
                         for value in produced)
        tensors = [AbstractTensor.get_tensor(np.asarray(column, dtype=np.float64)) for column in columns]
        return tuple(np.broadcast_to(np.asarray(value.numpy(), dtype=np.float64), (self.batch,)).copy()
                     for value in type(self).law(*tensors))


class EntityGame(Game):
    """A free-board game: entity lanes on a plain floor, advanced by ticks.

    The page sends ``tick`` actions carrying its held controls; each tick
    lands the wall time since the previous one (capped at ``max_frame``, so a
    hidden tab never jumps the game).  ``undo`` rewinds; ``layout`` fits the
    field to the screen's shape."""

    input = "realtime"
    width = 15.0
    height = 13.0
    max_frame = 0.25          # longer gaps (a hidden tab) are dropped, shorter ones landed whole
    CONTROLS: tuple[str, ...] = ()
    #: The player's lane and its motion law's constants: ``speed``, ``tau``,
    #: ``reach`` and, per predicted axis (``x``, and ``y`` for 2D games),
    #: the walls ``[lo, hi]``.  None: nothing is predicted.
    PLAYER_LANE: int | None = None
    MOTION: dict[str, Any] | None = None

    def __init__(self, board_id: str = ""):
        super().__init__(board_id)
        piece = self.make_piece()
        columns = self.initial_columns()
        if hasattr(piece, "compile"):
            piece.compile(columns)
        self.session = EntitySession(piece, columns, self.CONTROLS)
        self.last_tick: float | None = None
        self.ticks = 0
        self.held: dict[str, float] = {}
        self.ack = None           # the last input interval landed
        self.epoch = 0            # bumped when the world jumps (reset, rewind, relayout)
        self.intervals: list[dict] = []

    # declared by the game ----------------------------------------------------
    def make_piece(self):
        raise NotImplementedError

    def initial_columns(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def entities(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def palette(self) -> dict[str, list[float]]:
        return {}

    def describe(self) -> str:
        return ""

    #: Lane ranges ``(start, stop)`` holding things that can hurt the player;
    #: the sync hint is read from them.
    HOSTILE_LANES: tuple[tuple[int, int], ...] = ()
    #: The scene contract: which layers of the scene this game has.  The page
    #: draws a layer only if the game declares it AND the viewer has not
    #: switched it off; a layer the game leaves out is never drawn.
    #:   floor  -- the slab the entities move over
    #:   glass  -- the glass plate under the board
    #:   arcade -- the ring of arcade lights
    SCENE: dict[str, bool] = {"floor": True, "glass": True, "arcade": True}
    #: Check-in delay bounds (seconds) the sync hint moves between.
    SYNC_MIN, SYNC_MAX = 0.03, 0.25

    def sync_hint(self) -> float:
        """The longest a faithful page may wait between check-ins now.

        The shortest time-to-contact between the player's ship and any live
        hostile lane (distance over closing speed; a hostile moving away
        counts at its own speed), times 0.35, clipped to [SYNC_MIN,
        SYNC_MAX]: with nothing close a page may relax to a quarter second,
        with a bullet closing it is held near 30 ms so new threats arrive
        in time.  No hostiles declared, or no ship: SYNC_MAX."""
        s = self.session
        if s.state is None or self.PLAYER_LANE is None or not self.HOSTILE_LANES:
            return self.SYNC_MAX
        col = s.column
        k = self.PLAYER_LANE
        if col("alive")[k] <= 0.0:
            return self.SYNC_MAX
        lanes = np.concatenate([np.arange(a, b) for a, b in self.HOSTILE_LANES])
        live = lanes[col("alive")[lanes] > 0.0]
        if not len(live):
            return self.SYNC_MAX
        dx, dy = col("x")[live] - col("x")[k], col("y")[live] - col("y")[k]
        dvx, dvy = col("vx")[live] - col("vx")[k], col("vy")[live] - col("vy")[k]
        distance = np.maximum(np.hypot(dx, dy) - 0.5, 0.0)
        closing = -(dx * dvx + dy * dvy) / np.maximum(np.hypot(dx, dy), 1e-6)
        speed = np.maximum(closing, 0.25 * np.hypot(dvx, dvy) + 1e-3)
        contact = float(np.min(distance / speed))
        return float(np.clip(0.35 * contact, self.SYNC_MIN, self.SYNC_MAX))

    def motion(self) -> dict[str, Any] | None:
        """``MOTION`` as published; a game whose walls move overrides it."""
        return self.MOTION

    # verbs -------------------------------------------------------------------
    #: How far (units) the page's own ship may sit from the server's replay of
    #: the same inputs and still be adopted.  Beyond it the server keeps its
    #: own state: that is the keyframe the page eases to, so a disagreement
    #: is bounded, never interminable.
    ADOPT_BOUND = 1.0

    def control(self, payload: dict) -> None:
        self.held = {name: float(payload[name]) for name in self.CONTROLS if name in payload}
        self.intervals = [dict(item) for item in payload.get("inputs") or ()]
        self.reported = dict(payload["ship"]) if isinstance(payload.get("ship"), dict) else None

    def adopt_reported(self) -> None:
        """CLIENT UPDATES SERVER.  The page steps its own ship with the same
        motion law over the same input intervals and reports where that left
        it (``ship``: seq, x, y, vx, vy at the end of interval ``seq``).  Once
        the server has landed that interval, it adopts the page's ship for
        the player's lane on every axis the motion law declares -- the
        server's replay differs only by drift -- unless the two are more than
        ADOPT_BOUND apart (then the server's own state stands as the keyframe).
        Written in place into the dt system's state columns, as ``control``
        writes the control columns."""
        ship, self.reported = getattr(self, "reported", None), None
        motion, k = self.motion(), self.PLAYER_LANE
        if not ship or not motion or k is None or self.session.state is None or self.ack is None:
            return
        if int(ship.get("seq", -1)) != self.ack or self.session.column("alive")[k] <= 0.0:
            return
        axes = [(pos, vel) for axis, pos, vel in (("x", "x", "vx"), ("y", "y", "vy")) if motion.get(axis)]
        try:
            values = {name: float(ship[name]) for pos, vel in axes for name in (pos, vel)}
        except (KeyError, TypeError, ValueError):
            return
        if not all(np.isfinite(list(values.values()))):
            return
        error = math.hypot(*(values[pos] - float(self.session.column(pos)[k]) for pos, _ in axes))
        self.adopted = error <= self.ADOPT_BOUND
        if self.adopted:
            for name, value in values.items():
                getattr(self.session.state, name)[k] = value

    def tick(self) -> None:
        if self.intervals and self.session.state is not None:
            # Input-timed: each interval is a round with its own controls.
            for interval in self.intervals:
                if self.ack is not None and int(interval["seq"]) <= self.ack:
                    continue          # resent and already landed
                self.held = {name: float(interval[name]) for name in self.CONTROLS if name in interval}
                self.session.control(self.held)
                self.session.advance(min(self.max_frame, max(0.0, float(interval.get("dur", 0.0)))))
                self.ack = int(interval["seq"])
            self.adopt_reported()
            self.intervals = []
            self.last_tick = time.perf_counter()
            self.ticks += 1
            return
        now = time.perf_counter()
        seconds = 0.0 if self.last_tick is None else min(self.max_frame, now - self.last_tick)
        self.last_tick = now
        self.session.control(self.held)
        if self.session.state is None:
            seconds = max(seconds, 1.0 / 60.0)      # round zero instantiates the state
        self.session.advance(seconds)
        self.ticks += 1

    def undo(self) -> None:
        self.session.rewind()
        self.last_tick = None
        self.epoch += 1

    def reset(self) -> None:
        self.session.reset(self.initial_columns())
        self.last_tick = None
        self.epoch += 1

    def state(self) -> dict[str, Any]:
        st = self.base_state(int(round(self.height)), int(round(self.width)))
        st["board"] = "free"
        st["scene"] = dict(self.SCENE)
        st["width"], st["height"] = self.width, self.height
        st["palette"] = {"floor": [0.10, 0.11, 0.16], **self.palette()}
        st["cells"] = ["floor"]
        st["marks"], st["hints"] = [], []
        st["entities"] = self.entities() if self.session.state is not None else []
        seconds, calls = self.session.last_round
        st["clock"] = {"tick_s": seconds, "law_calls": calls, "dt": float(self.session.dt)}
        st["status"] = self.describe()
        st["moves"] = self.ticks
        st["over"] = self.session.state is not None and self.result() is not None
        st["world_clock"] = float(self.session.column("clock")[0]) if self.session.state is not None else 0.0
        st["score"] = float(self.session.column("score")[0]) if self.session.state is not None else 0.0
        st["sync_hint"] = self.sync_hint()
        if self.PLAYER_LANE is not None and self.session.state is not None:
            k, col = self.PLAYER_LANE, self.session.column
            st["predict"] = {"lane": k, **(self.motion() or {})}
            st["ack"] = self.ack
            st["you"] = {"x": float(col("x")[k]), "y": float(col("y")[k]), "vx": float(col("vx")[k]),
                         "vy": float(col("vy")[k]), "alive": bool(col("alive")[k] > 0.0),
                         "shield": float(col("energy")[k]) if "energy" in self.session.columns else 0.0}
        return st
