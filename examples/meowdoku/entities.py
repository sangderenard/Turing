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

Rewind and reset are not game code: the session snapshots the state with
``copy_shallow`` every ``rewind_every`` seconds of play and ``undo`` restores
one from about two seconds back (the pair ``run_superstep`` uses for in-round
rollback); ``reset`` restores round zero.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from meowdoku.games import Game
from meowdoku.session import ROOT  # noqa: F401  (puts the repo and examples on sys.path)
from src.common.dt_system.dt_controller import STController, Targets  # noqa: E402
from llvm_dt_system import dt_system  # noqa: E402


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

    def __init__(self, board_id: str = ""):
        super().__init__(board_id)
        self.session = EntitySession(self.make_piece(), self.initial_columns(), self.CONTROLS)
        self.last_tick: float | None = None
        self.ticks = 0
        self.held: dict[str, float] = {}

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

    # verbs -------------------------------------------------------------------
    def control(self, payload: dict) -> None:
        self.held = {name: float(payload[name]) for name in self.CONTROLS if name in payload}

    def tick(self) -> None:
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

    def reset(self) -> None:
        self.session.reset(self.initial_columns())
        self.last_tick = None

    def state(self) -> dict[str, Any]:
        st = self.base_state(int(round(self.height)), int(round(self.width)))
        st["board"] = "free"
        st["width"], st["height"] = self.width, self.height
        st["palette"] = {"floor": [0.10, 0.11, 0.16], **self.palette()}
        st["cells"] = ["floor"]
        st["marks"], st["hints"] = [], []
        st["entities"] = self.entities() if self.session.state is not None else []
        seconds, calls = self.session.last_round
        st["clock"] = {"tick_s": seconds, "law_calls": calls, "dt": float(self.session.dt)}
        st["status"] = self.describe()
        st["moves"] = self.ticks
        return st
