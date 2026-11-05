"""A Meowdoku game driven by the dt system: one move is one round.

This is ``llvm_dt_system.dt_system`` used exactly as its docstring says a
persistent state is used: the first round instantiates the state, every
later round hands that same state back, and the caller writes its columns
in place between rounds.  The dt system owns the state's one contiguous
span, its save and restore, and the controller kept alive across rounds.

The Python game does one thing with input: it hands a ``MeowdokuControls``
record to the round (``hand``).  The record's fields are state columns the
law reads (``program.CONTROLS`` names the semantic action each is filled
from), so a compiled shell writes the same slots from its devices.

Undo and reset are therefore not game code: a move is preceded by
``copy_shallow`` and undone by ``restore``, the same pair ``run_superstep``
uses for in-round rollback -- and a restored state restores its controls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from src.common.dt_system.dt_controller import STController, Targets  # noqa: E402
from llvm_dt_system import TELEMETRY_FIELDS, dt_system  # noqa: E402
from meowdoku.program import (  # noqa: E402
    CONTROLS, MeowdokuControls, MeowdokuPiece, initial_columns, place_controls,
    toggle_controls,
)
from meowdoku.puzzle import generate  # noqa: E402

#: One move per round.  ``dt_limit`` is published as 1.0 by the law, so the
#: controller has no reason to subdivide, and the window is the move.
MOVE_DT = 1.0
TAU = TELEMETRY_FIELDS.index("tau")
RESIDUAL = TELEMETRY_FIELDS.index("residual")


class MeowdokuSession:
    def __init__(self, n: int = 6, seed: int = 0, puzzle=None):
        # A prebaked puzzle (region map + the leaf the game scores against)
        # may be handed in; otherwise one is generated for (n, seed).
        self.puzzle = puzzle if puzzle is not None else generate(n, seed)
        self.n = int(self.puzzle.n)
        self.columns = initial_columns(self.puzzle)
        self.pieces = [MeowdokuPiece(self.puzzle.cells)]
        self.targets = Targets(cfl=0.5, div_max=1e9, mass_max=1e9,
                               energy_exchange_fraction=0.2)
        self.controller = STController(dt_min=MOVE_DT * 1e-6)
        self.state = None
        self.history: list = []
        self.telemetry = None

    # ------------------------------------------------------------ rounds
    def _round(self) -> None:
        self.state, self.controller, results = dt_system(
            self.pieces, self.columns, rounds=1, round_dt=MOVE_DT, dx=1.0,
            targets=self.targets, controller=self.controller, state=self.state)
        advanced, _dt_next, telemetry = results[0]
        if telemetry[TAU] != 1.0 or advanced != MOVE_DT:
            raise RuntimeError(
                f"a move must land as exactly one window: advanced={advanced}, "
                f"tau={telemetry[TAU]}")
        self.telemetry = telemetry

    def hand(self, controls: MeowdokuControls) -> None:
        """Hand one round's controller state to the dt system and run it."""
        values = controls.columns(self.puzzle.cells)
        if self.state is None:
            # Round zero: the state is made from the initial columns, so the
            # first round's controls ride in on them.
            for name in CONTROLS:
                self.columns[name][...] = values[name]
            self._round()
            self.history = [self.state.copy_shallow()]
            return
        self.history.append(self.state.copy_shallow())
        for name in CONTROLS:
            getattr(self.state, name)[...] = values[name]
        self._round()

    def place(self, r: int, c: int) -> None:
        self.hand(place_controls(r, c))

    def toggle_x(self, r: int, c: int) -> None:
        self.hand(toggle_controls(r, c))

    def undo(self) -> None:
        """Back one move: the dt system's own restore."""
        if not self.history:
            return
        snapshot = self.history.pop()
        if self.state is None:
            return
        self.state.restore(snapshot)
        if not self.history:
            # The first move's snapshot is the board after that move; past it
            # is round zero, which the initial columns still describe.
            self.state.restore(self._round_zero())

    def reset(self) -> None:
        if self.state is None:
            return
        self.state.restore(self._round_zero())
        self.history = []

    def _round_zero(self):
        fresh = initial_columns(self.puzzle)
        names, _schedule = self.state.bound_pieces
        return tuple(fresh[name] if name in fresh else np.zeros(self.puzzle.cells)
                     for name in (*names, "dt"))

    # ------------------------------------------------------------ reading
    def _column(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self.state, name) if self.state is not None
                          else self.columns[name])

    @property
    def controls(self) -> MeowdokuControls:
        """The controls the state currently holds (restored with it)."""
        return MeowdokuControls(*(float(self._column(name)[0]) for name in CONTROLS))

    @property
    def cat(self) -> np.ndarray:
        return self._column("cat").reshape(self.n, self.n)

    @property
    def x_mark(self) -> np.ndarray:
        return self._column("x_mark").reshape(self.n, self.n)

    @property
    def hearts(self) -> int:
        return int(self._column("hearts")[0])

    @property
    def residual(self) -> int:
        return int(round(float((self.puzzle.solution * (1.0 - self._column("cat"))).sum())))

    @property
    def won(self) -> bool:
        return self.residual == 0 and self.hearts > 0

    @property
    def lost(self) -> bool:
        return self.hearts <= 0

    def calls(self) -> int:
        """How many times the dt system called the law, from its own ledger."""
        if self.state is None:
            return 0
        return int(self.state.wall_cost_ledger.report()["participants"]["meowdoku_step"]["calls"])
