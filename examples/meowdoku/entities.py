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

import hashlib
import importlib
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


def block_hits(x, y, rx, ry, live, harms, rows: tuple[int, int], cols: tuple[int, int], z=None, rz=None):
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
    if z is not None:
        # A field with height (``z``, depth half extents ``rz``): boxes must
        # overlap in depth as well.
        touch = touch & ((z[r0:r1].reshape((a, 1)) - z[c0:c1].reshape((1, b))).abs()
                         < (rz[r0:r1].reshape((a, 1)) + rz[c0:c1].reshape((1, b))))
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
                 rewind_every: float = 0.5, rewind_depth: int = 40,
                 numpy_compiled: bool | None = None):
        self.piece = piece
        # The dt system's own functions as the NumPy product of their
        # AbstractTensor source (llvm_dt_system.numpy_compiled_program): the
        # controller's per-step numerical work leaves AbstractTensor dispatch;
        # the law piece stays linked and called from the controller loop.
        self.numpy_compiled = (DT_LANE == "numpy") if numpy_compiled is None else bool(numpy_compiled)
        self.pieces = [piece]
        self.columns = columns
        self.controls = tuple(controls)
        # cfl 1: the laws' dt_limit already carries its own 0.9 safety factor
        # Entity games publish no energy exchange: no exchange-time scheduling
        # (configured, it made every participant HOLD, so a step that shrank
        # once could never regrow).
        self.targets = Targets(cfl=1.0, div_max=1e9, mass_max=1e9, energy_exchange_fraction=None)
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
            # Real-time play keeps no backups: the dt system's no-save,
            # in-place, no-retry lane, declared rather than inherited.
            rollback=False,
            numpy_compiled=self.numpy_compiled,
            # Play is one continuous run cut into ticks: the next tick's first
            # attempt is the controller's last proposal, not the whole window
            # (dt_system's first round has no ceiling of its own).
            dt_initial=None if self.state is None else min(self.dt, float(seconds)))
        advanced, proposed, telemetry = results[0]
        # A window shorter than the step only clips that one step: the dt
        # system's proposal after it is the window's length, not a judgement
        # of the physics.  Keeping it would shrink every later first attempt
        # for good (a 3 ms input interval left the controller at 3 ms and a
        # 250 ms tick at 2000+ law calls).  So the clipped proposal is kept
        # only when the window was not the limit.
        if float(seconds) >= self.dt or float(proposed) < float(seconds):
            self.dt = float(proposed)
        if abs(float(advanced) - seconds) > 1e-9 * max(1.0, seconds):
            raise RuntimeError(f"a tick must land as one window: advanced={advanced} of {seconds}")
        self.telemetry = telemetry
        self.last_round = (float(seconds), self.calls() - before)
        self.since_snapshot += seconds
        if self.rewind_depth and (not self.snapshots or self.since_snapshot >= self.rewind_every):
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

#: How the dt system's own functions run: ``numpy`` (the product of their
#: AbstractTensor source, ``llvm_dt_system.numpy_compiled_program``) or
#: ``eager`` (as written, AbstractTensor dispatch).
DT_LANE = os.environ.get("MEOWDOKU_DT", "numpy")

#: Where compiled law products are kept (ignored by git): one file per law and
#: source digest, so an unchanged law loads its product and a changed one
#: recompiles.
LAW_PRODUCTS = Path(__file__).resolve().parents[2] / "build" / "meowdoku" / "laws"
#: The lane a law runs in.  ``numpy``: the law traced through AbstractTensor
#: (the translatable source) and printed as a standalone NumPy function by
#: ``tape_to_source`` (the product).  ``eager``: the AbstractTensor law itself.
#: ``llvm`` / ``eigen``: the same law traced through the SSA tensor backend
#: and compiled natively by ``compile_native_graph_forward`` under that work
#: contract (``eigen`` links Eigen's kernels; it measured fastest for the
#: scroller: 4.1 ms a step against NumPy's 8.0, bit-identical).
LAW_LANE = os.environ.get("MEOWDOKU_LAW", "eigen")
#: Where natively compiled laws are kept: one directory per law, source
#: digest, argument list and contract.
LAW_NATIVE = LAW_PRODUCTS.parent / "native_law"


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
    #: PARAMETER inputs: arrays that are not shaped like lanes (a network's
    #: weights), name -> shape.  They follow ``argument_names`` in the law's
    #: arguments, are SSA inputs of those shapes in the native compile,
    #: arrays of the NumPy product, and named inputs of the page's wasm
    #: (``graph_wasm``).  The game sets their values with ``set_parameters``
    #: -- per landing, never a recompile; the digest carries their names and
    #: shapes only.
    parameter_shapes: dict[str, tuple[int, ...]] = {}
    #: Modules besides the law's own whose text the product is a function of
    #: (a shared substrate the law calls, e.g. ``meowdoku.walkers``): hashed
    #: into the source digest so editing one recompiles the law.
    source_modules: tuple[str, ...] = ()

    def __init__(self, batch: int):
        self.batch = int(batch)
        self._bound = None
        self._parameters = {n: np.zeros(shape) for n, shape in self.parameter_shapes.items()}
        self._parameter_buffers: dict[str, np.ndarray] = {}
        self.product = None
        self.lane = "eager"
        self.product_path: Path | None = None

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.parameter_shapes)

    def set_parameters(self, values: dict[str, Any]) -> None:
        """Set the parameter inputs (a name -> array dict; names the law does
        not declare are ignored, declared ones must be present and of their
        shape).  The values are copied into the native execution's input
        buffers in place -- or into the arrays the NumPy product is called
        with -- so new values never recompile anything."""
        for name, shape in self.parameter_shapes.items():
            if name not in values:
                raise KeyError(f"{self.entry}: parameter {name!r} not given")
            array = np.asarray(values[name], dtype=np.float64)
            if tuple(array.shape) != tuple(shape):
                raise ValueError(f"{self.entry}: parameter {name!r} has shape {array.shape}, declared {tuple(shape)}")
            np.copyto(self._parameters[name], array)
            buffer = self._parameter_buffers.get(name)
            if buffer is not None:
                np.copyto(buffer.reshape(-1), array.reshape(-1))

    def instantiate(self, columns, outputs=None):
        self._bound = tuple(columns[name] for name in self.argument_names)
        return ()

    def __init_subclass__(cls, **kwargs):
        """Capture the sources the product is a function of NOW, when the law
        is defined: then disk and the code in memory agree.  Reading them at
        compile time instead let a process that imported an older law hash
        the newer file on disk and file its OLD law's product under the NEW
        digest (a training run's workers did, one second after an edit)."""
        super().__init_subclass__(**kwargs)
        cls._captured_sources = cls._read_sources()

    _captured_sources: str = ""

    def _sources(self) -> str:
        """Everything the product is a function of, as captured when the law
        was defined (see ``__init_subclass__``)."""
        return type(self)._captured_sources or type(self)._read_sources()

    @classmethod
    def _read_sources(cls) -> str:
        """The law's module, the engine helpers and motion law it calls, and
        the printer."""
        import inspect
        if cls.law is None:
            return ""
        from src.common.tensors.accelerator_backends import tape_to_source
        from meowdoku import entity_motion
        law_module = sys.modules.get(getattr(cls.law, "__module__", "")) or inspect.getmodule(cls.law)
        modules = (law_module, sys.modules[__name__], entity_motion, tape_to_source,
                   *(importlib.import_module(name) for name in cls.source_modules))
        return "\0".join(inspect.getsource(module) for module in modules)

    def _signature(self) -> str:
        """Argument names, and the parameters' names and SHAPES (never their
        values), as the short key a product is filed under."""
        text = ",".join(self.argument_names)
        if self.parameter_shapes:
            text += "|" + ",".join(f"{n}{tuple(shape)}" for n, shape in self.parameter_shapes.items())
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

    def compile(self, columns: dict[str, np.ndarray]) -> None:
        if LAW_LANE in {"llvm", "develop", "eigen", "fast"}:
            try:
                self._compile_native(columns, "develop" if LAW_LANE == "llvm" else LAW_LANE)
                return
            except Exception as exc:              # the NumPy product is the fallback, said aloud
                print(f"[law] {self.entry}: native lane {LAW_LANE} refused ({type(exc).__name__}: "
                      f"{str(exc)[:300]}); running the NumPy product", flush=True)
        elif LAW_LANE != "numpy":
            return
        self._compile_numpy(columns)

    def _compile_native(self, columns: dict[str, np.ndarray], contract: str) -> None:
        """Trace the law through the SSA tensor backend and compile it
        natively (``compile_native_graph_forward``), cached by source digest,
        argument list and contract.  Each call writes the columns into one
        resident execution's buffers and runs the entry."""
        import hashlib
        import pickle
        digest = hashlib.sha256(self._sources().encode("utf-8")).hexdigest()[:16]
        signature = self._signature()
        shapes = {name: (tuple(np.shape(columns[name])) if name in columns else ()) for name in self.argument_names}
        shapes.update({n: tuple(shape) for n, shape in self.parameter_shapes.items()})
        names = (*self.argument_names, *self.parameter_names)
        name = f"{self.entry}_{contract}"
        home = LAW_NATIVE / f"{self.entry}_{digest}_{signature}_{contract}"
        cached = home / "forward.pkl"
        forward = pickle.loads(cached.read_bytes()) if cached.exists() else None
        if forward is None:
            from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
            from src.compiler.llvm_training_runtime import compile_native_graph_forward
            from src.compiler.work_contract import PRESETS, scoped_work_contract
            started = time.perf_counter()
            with scoped_work_contract(PRESETS[contract]):
                program = SSATensorProgram(name)
                bindings = {n: SSATensorOperations.input(program, shapes[n]) for n in names}
                with program.activate():
                    produced = type(self).law(*(bindings[n] for n in names))
                forward = compile_native_graph_forward(
                    produced, bindings=bindings, source=self._sources(), name=name, directory=home)
            home.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(pickle.dumps(forward))
            print(f"[law] {self.entry}: compiled natively ({contract}) in {time.perf_counter() - started:.1f}s", flush=True)
        from src.compiler.ssa_llvm_backend import prepare_artifact_execution
        ids = forward.input_value_ids
        execution = prepare_artifact_execution(forward.artifact, {
            ids[n]: (self._parameters[n] if n in self._parameters else np.zeros(shapes[n]) if shapes[n] else np.asarray(0.0))
            for n in names})
        inputs = [np.asarray(execution.buffers[ids[n]]).reshape(-1) for n in self.argument_names]
        self._parameter_buffers = {n: np.asarray(execution.buffers[ids[n]]) for n in self.parameter_names}
        outputs = [np.asarray(execution.buffers[i]).reshape(-1) for i in forward.output_value_ids]
        batch = self.batch

        def product(*cols):
            for buffer, column in zip(inputs, cols):
                buffer[...] = np.asarray(column, dtype=np.float64).reshape(-1) if buffer.size > 1 else float(np.asarray(column).reshape(-1)[0])
            execution.run()
            return tuple(out if out.size == batch else np.broadcast_to(out, (batch,)) for out in outputs)

        self.product, self.lane, self.product_path = product, contract, home
        self._native = (forward, execution)

    def _compile_numpy(self, columns: dict[str, np.ndarray]) -> None:
        import hashlib
        import importlib.util
        import inspect
        digest = hashlib.sha256(self._sources().encode("utf-8")).hexdigest()[:16]
        # The product is also keyed by the law's argument list, and a product
        # whose parameters differ from it is never used: a process running an
        # OLDER law (a long-lived server, a training pool) that hashed the
        # newer file on disk would otherwise file its old law's product under
        # the new digest -- the live server did, with three columns missing.
        signature = self._signature()
        names = (*self.argument_names, *self.parameter_names)
        path = LAW_PRODUCTS / f"{self.entry}_{digest}_{signature}.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location(f"meowdoku_law_check_{self.entry}_{digest}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if tuple(inspect.signature(getattr(module, self.entry)).parameters) != names:
                path.unlink()                        # not this law's product: make it again
        if not path.exists():
            from src.common.tensors.autograd import autograd
            from src.common.tensors.numpy_backend import NumPyTensorOperations
            from src.common.tensors.accelerator_backends.tape_to_source import emit_tape_source
            sample = {name: NumPyTensorOperations.tensor(np.asarray(
                columns[name] if name in columns else self._parameters[name] if name in self._parameters
                else np.full(self.batch, 1.0 / 60.0), dtype=np.float64))
                for name in names}
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
        function = getattr(module, self.entry)
        # the parameters ride behind the columns, as the law's last arguments
        self.product = (lambda *cols: function(*cols, *(self._parameters[n] for n in self.parameter_names))
                        ) if self.parameter_shapes else function
        self.lane, self.product_path = "numpy", path

    def __call__(self, *columns):
        if len(columns) != len(self.argument_names):
            raise TypeError(f"{self.entry}: takes {len(self.argument_names)} columns, got {len(columns)}")
        if self.product is not None:
            produced = self.product(*(np.asarray(column, dtype=np.float64) for column in columns))
            return tuple(np.broadcast_to(np.asarray(value, dtype=np.float64), (self.batch,)).copy()
                         for value in produced)
        tensors = [AbstractTensor.get_tensor(np.asarray(column, dtype=np.float64))
                   for column in (*columns, *(self._parameters[n] for n in self.parameter_names))]
        return tuple(np.broadcast_to(np.asarray(value.numpy(), dtype=np.float64), (self.batch,)).copy()
                     for value in type(self).law(*tensors))


def _declared(name):
    from meowdoku.autopilot import declared
    return declared(name)


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
    #: A platformer-style game offers the character tile set (``characters.py``): the page shows the tiles left of the
    #: board, a tap chooses (a profile setting, ``avatar``), and the game draws its player as ``hero_mesh``.
    CHARACTERS_ON = False
    hero = "frog"

    def set_hero(self, key) -> None:
        from meowdoku import characters
        if key in characters.CHARACTERS:
            self.hero = str(key)

    @property
    def hero_mesh(self) -> str:
        from meowdoku import characters
        return characters.mesh_of(self.hero)

    def __init__(self, board_id: str = ""):
        super().__init__(board_id)
        piece = self.make_piece()
        columns = self.initial_columns()
        if hasattr(piece, "compile"):
            piece.compile(columns)
        # No rewind snapshots: real-time games are never rolled back (the
        # page's own history is the record; undo is for the board games).
        self.session = EntitySession(piece, columns, self.CONTROLS, rewind_depth=0)
        self.sync_parameters()
        self.last_tick: float | None = None
        self.ticks = 0
        self.held: dict[str, float] = {}
        self.ack = None           # the last input interval landed
        self.epoch = 0            # bumped when the world jumps (reset, rewind, relayout)
        self.intervals: list[dict] = []
        self.autopilot = False    # when set, the compiled policy (autopilot.py) supplies the controls
        self.pilot = None
        self._pilot_hold = 0.0    # seconds left on the policy's current decision

    #: Seconds one autopilot decision is held.  The policy's outputs are
    #: control values held over an interval -- streamed like a player's
    #: inputs -- not a per-frame command, so it decides at 10 Hz (a third of
    #: the policy work; the page predicts the ship between decisions).
    PILOT_STEP = 1.0 / 10.0

    def set_autopilot(self, on: bool) -> None:
        """Hand the controls to the trained policy (server side); the page keeps drawing."""
        if on:
            from meowdoku.autopilot import pilot_for
            self.pilot = pilot_for(self)         # refuses when there are no trained weights
            decl = _declared(self.game)
            if decl is not None:
                self.PILOT_STEP = decl.step      # a declared game holds each decision for its own step
        self.autopilot = bool(on)
        self._pilot_hold = 0.0

    def _tracing(self) -> bool:
        """Whether a decision made now would be recorded for learning."""
        from meowdoku.autopilot import FEATURE_FUNCTIONS
        return not (self.game not in FEATURE_FUNCTIONS or self.session.state is None or getattr(self, "_ended", False)
                    or not getattr(self, "learns", True))

    def _trace(self, controls: dict, demo: bool, features=None) -> None:
        """Record one decision for learning (games with an autopilot only);
        ``features`` the observation it was made on, when the caller has it."""
        from meowdoku.autopilot import PlayTrace
        if not self._tracing():
            return
        if getattr(self, "trace", None) is None:
            self.trace = PlayTrace()
        if self.trace.record(self, controls, demo, features):
            # enough spans of action gathered: learn from them now, mid-game
            from meowdoku.autopilot import finish_episode
            finish_episode(self, self.trace, final=False)

    def _end_check(self) -> None:
        """The first landing after the game is over hands its trace to the learner."""
        if getattr(self, "_ended", False) or self.session.state is None or self.result() is None:
            return
        self._ended = True
        trace, self.trace = getattr(self, "trace", None), None
        if trace is not None:
            from meowdoku.autopilot import finish_episode
            finish_episode(self, trace)

    def drive_settings(self) -> None:
        """Settings that follow who is driving (a game overrides; default none)."""

    def law_pilot_on(self) -> bool:
        """Whether the autopilot's network runs INSIDE the law (a game whose
        law carries the pilot overrides; default: the Python pilot flies)."""
        return False

    def law_pilot_controls(self) -> dict[str, float]:
        """The controls the law's own pilot decided on its last step."""
        raise NotImplementedError

    def parameters(self) -> dict[str, np.ndarray] | None:
        """The law's PARAMETER inputs now (``LawPiece.parameter_shapes``): a
        game with a network in its law returns its current weights; None: none."""
        return None

    def sync_parameters(self) -> None:
        """Hand the law its parameter inputs (copied in place; no recompile)."""
        piece = self.session.piece
        if getattr(piece, "parameter_shapes", None):
            values = self.parameters()
            if values is not None:
                piece.set_parameters(values)

    def _land(self, seconds: float, held: dict) -> None:
        """Land ``seconds`` with ``held`` controls, or with the policy's own
        when the autopilot is on (one decision per PILOT_STEP)."""
        if self.session.state is not None:
            self.drive_settings()
        if self.autopilot and self.pilot is not None and self.session.state is not None and self.law_pilot_on():
            self._land_in_law(seconds)
            return
        if not self.autopilot or self.pilot is None or self.session.state is None:
            self.held = held
            if seconds > 0.0:
                self._trace(held, demo=False)
            self.session.control(held)
            self.session.advance(seconds)
            self._end_check()
            return
        left = seconds
        while left > 1e-9:
            if self._pilot_hold <= 1e-9:
                self.held = self.pilot.controls(self)
                self._pilot_hold = self.PILOT_STEP
                self._trace(self.held, demo=True)
            step = min(left, self._pilot_hold)
            self.session.control(self.held)
            self.session.advance(step)
            left -= step
            self._pilot_hold -= step
        self._end_check()

    def _land_in_law(self, seconds: float) -> None:
        """Land ``seconds`` with the network flying INSIDE the law.  The law
        owns the pilot's decision clock (a decision every PILOT_INTERVAL of law
        time, in effect PILOT_PREDELAY later; nothing for Python to do).  The
        demo trace records each decision as the law takes it: the observation
        is read first, the one law step that takes the decision lands on its
        own, and the decision (read back from the law's own columns) is the
        row's target -- the in-law decision, on the observation it was made on."""
        from meowdoku.autopilot import FEATURE_FUNCTIONS
        self.session.control(self.law_pilot_rest())      # what the network does not drive
        left = seconds
        while left > 1e-9:
            if not self._tracing():
                self.session.advance(left)
                left = 0.0
            elif self.law_pilot_due():
                features = FEATURE_FUNCTIONS[self.game](self)
                step = min(left, self.session.dt)
                self.session.advance(step)
                left -= step
                self._trace(self.law_pilot_decision(), demo=True, features=features)
            else:
                step = min(left, max(self.law_pilot_hold(), 1e-6))
                self.session.advance(step)
                left -= step
        self.held = self.law_pilot_controls()
        self._end_check()

    def law_pilot_rest(self) -> dict[str, float]:
        """Every control at rest (the network's own overwrite the law's)."""
        raise NotImplementedError

    def law_pilot_due(self) -> bool:
        return False

    def law_pilot_hold(self) -> float:
        return 0.0

    def law_pilot_decision(self) -> dict[str, float]:
        raise NotImplementedError

    def law_pilot_intervals(self) -> list[dict]:
        """The law's pilot decisions the page integrates (held + pending); none by default."""
        return []

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
    #: Layers the game would like ON unless the viewer turned them off for it
    #: (the page's switches default to off otherwise).
    SCENE_PREFER: dict[str, bool] | None = None
    #: A game whose ``lives`` column is health points declares their maximum
    #: (the page draws a health bar); None: lives are lives.
    HEALTH_MAX: float | None = None
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
    #: Axes the game's own law moves the player along besides the motion law
    #: (a side scroller shifts the whole world, the player too, in x).  The
    #: page cannot know that shift, so its report is never adopted there.
    SCROLL_AXES: tuple[str, ...] = ()

    def adopt_axes(self) -> list[str]:
        motion = self.motion() or {}
        # ("yaw", "spd": a spherical game's turn and throttle, columns of the
        # same names with velocities "vyaw", "vspd")
        return [axis for axis in ("x", "y", "z", "yaw", "spd", "bank") if motion.get(axis) and axis not in self.SCROLL_AXES]

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
        outcome = self._adopt_outcome()
        self.count_authority(outcome)

    def count_authority(self, outcome: str) -> None:
        """Who authored the player's ship on this landed step: ``client``
        (the page's report adopted), ``kept`` (reported but too far apart:
        the server's state stood), ``unreported`` (no usable report), or
        ``server`` (no input intervals: the server stepped on its own)."""
        counts = getattr(self, "authority", None)
        if counts is None:
            counts = self.authority = {"client": 0, "kept": 0, "unreported": 0, "server": 0}
        counts[outcome] += 1

    def _adopt_outcome(self) -> str:
        ship, self.reported = getattr(self, "reported", None), None
        motion, k = self.motion(), self.PLAYER_LANE
        if not ship or not motion or k is None or self.session.state is None or self.ack is None:
            return "unreported"
        if int(ship.get("seq", -1)) != self.ack or self.session.column("alive")[k] <= 0.0:
            return "unreported"
        axes = [(axis, "v" + axis) for axis in self.adopt_axes()]
        if not axes:
            return "unreported"
        try:
            values = {name: float(ship[name]) for pos, vel in axes for name in (pos, vel)}
        except (KeyError, TypeError, ValueError):
            return "unreported"
        if not all(np.isfinite(list(values.values()))):
            return "unreported"
        error = math.hypot(*(values[pos] - float(self.session.column(pos)[k]) for pos, _ in axes))
        self.adopted = error <= self.ADOPT_BOUND
        if self.adopted:
            for name, value in values.items():
                getattr(self.session.state, name)[k] = value
            # State the motion declares stepped with the ship (a spherical
            # game's land: ``travel``) is adopted with it, when reported.
            for name in motion.get("travel") or ():
                if name in ship and np.isfinite(float(ship[name])):
                    getattr(self.session.state, name)[k] = float(ship[name])
            return "client"
        return "kept"

    def tick(self) -> None:
        if self.intervals and self.session.state is not None:
            # Input-timed: each interval is a round with its own controls.
            for interval in self.intervals:
                if self.ack is not None and int(interval["seq"]) <= self.ack:
                    continue          # resent and already landed
                held = {name: float(interval[name]) for name in self.CONTROLS if name in interval}
                self._land(min(self.max_frame, max(0.0, float(interval.get("dur", 0.0)))), held)
                self.ack = int(interval["seq"])
            self.adopt_reported()
            self.intervals = []
            self.last_tick = time.perf_counter()
            self.ticks += 1
            return
        now = time.perf_counter()
        seconds = 0.0 if self.last_tick is None else min(self.max_frame, now - self.last_tick)
        self.last_tick = now
        if self.session.state is None:
            seconds = max(seconds, 1.0 / 60.0)      # round zero instantiates the state
        self._land(seconds, self.held)
        self.count_authority("server")
        self.ticks += 1

    def undo(self) -> None:
        self.session.rewind()
        self.last_tick = None
        self.epoch += 1

    def reset(self) -> None:
        self.session.reset(self.initial_columns())
        self.last_tick = None
        self.epoch += 1
        self.trace, self._ended = None, False          # a new episode to learn from
        self.authority = None                           # authorship counts start over

    def state(self) -> dict[str, Any]:
        st = self.base_state(int(round(self.height)), int(round(self.width)))
        st["board"] = "free"
        st["scene"] = dict(self.SCENE)
        glow = getattr(self, "emissive", None)
        if glow:
            st["emissive"] = dict(glow)            # palette entries that glow (the page's materials)
        if self.SCENE_PREFER:
            st["scene_prefer"] = dict(self.SCENE_PREFER)
        st["width"], st["height"] = self.width, self.height
        st["palette"] = {"floor": [0.10, 0.11, 0.16], **self.palette()}
        st["cells"] = ["floor"]
        st["marks"], st["hints"] = [], []
        st["entities"] = self.entities() if self.session.state is not None else []
        if self.CHARACTERS_ON:
            from meowdoku import characters
            st["pieces"] = characters.tile_pieces(self.hero)
            st["characters"] = {"chosen": self.hero, "keys": list(characters.CHARACTERS)}
        seconds, calls = self.session.last_round
        st["clock"] = {"tick_s": seconds, "law_calls": calls, "dt": float(self.session.dt)}
        st["status"] = self.describe()
        st["moves"] = self.ticks
        st["over"] = self.session.state is not None and self.result() is not None
        st["world_clock"] = float(self.session.column("clock")[0]) if self.session.state is not None else 0.0
        st["score"] = float(self.session.column("score")[0]) if self.session.state is not None else 0.0
        st["sync_hint"] = self.sync_hint()
        st["autopilot"] = self.autopilot
        st["controls"] = list(self.CONTROLS)           # the control names this game's law takes
        if getattr(self, "MODES", None):
            st["modes"] = {"labels": dict(self.MODES), "on": self.modes()}
        from meowdoku.autopilot import FEATURE_FUNCTIONS, learning_summary
        if self.game in FEATURE_FUNCTIONS:
            st["learning"] = learning_summary(self.game)
            from meowdoku.autopilot import available, declared
            st["autopilot_ready"] = bool(available(self.game))         # trained weights fit: the page may hand it the controls
            if declared(self.game) is not None:
                from meowdoku.pilot_decl import arch_for
                st["autopilot_net"] = arch_for(self.game)              # token or graph
        if self.PLAYER_LANE is not None and self.session.state is not None:
            k, col = self.PLAYER_LANE, self.session.column
            st["predict"] = {"lane": k, "adopt": self.ADOPT_BOUND, "adopt_axes": self.adopt_axes(), **(self.motion() or {})}
            st["ack"] = self.ack
            st["authority"] = {**(getattr(self, "authority", None) or {}), "axes": self.adopt_axes(),
                               "scroll_axes": list(self.SCROLL_AXES)}
            st["you"] = {"x": float(col("x")[k]), "y": float(col("y")[k]), "vx": float(col("vx")[k]),
                         "vy": float(col("vy")[k]), "alive": bool(col("alive")[k] > 0.0),
                         "shield": float(col("energy")[k]) if "energy" in self.session.columns else 0.0,
                         # height above the board (games with a third axis; else 0)
                         "z": float(col("z")[k]) if "z" in self.session.columns else 0.0,
                         # health as a fraction (a game declaring HEALTH_MAX; else 0)
                         "health": (float(col("lives")[k]) / self.HEALTH_MAX) if self.HEALTH_MAX else 0.0,
                         "vz": float(col("vz")[k]) if "vz" in self.session.columns else 0.0}
            for axis in ("yaw", "spd", "bank"):         # a spherical game's turn, throttle and bank
                if axis in self.session.columns:
                    st["you"][axis], st["you"]["v" + axis] = float(col(axis)[k]), float(col("v" + axis)[k])
            # the demo pilot's controls (the page predicts the ship with them,
            # on the same laws, while the autopilot flies)
            held = getattr(self, "held", {}) or {}
            st["you"]["pilot"] = ([float(held.get(n, 0.0)) for n in ("move_axis", "move_axis_y", "move_axis_z", "move_axis_r", "fire")]
                                  if self.autopilot else [0.0] * 5)
            # the pilot's decisions, held and pending (law-time intervals: the page integrates them)
            st["you"]["pilot_intervals"] = self.law_pilot_intervals() if self.autopilot else []
            for name in ("map_u", "map_v", "map_gx", "map_gy", "beam", "ux", "uy", "uz", "hx", "hy", "hz",
                         "coy", "buf", "pf"):          # (coy/buf/pf: a platformer's coyote time, buffered press, last press)
                # the land under it, the laser's reach, its place on a sphere
                if name in self.session.columns:
                    st["you"][name] = float(col(name)[k])
        return st
