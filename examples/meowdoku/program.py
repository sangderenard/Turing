"""Meowdoku as one dt-system piece: the game is the evolving state.

The board is a span of ``n*n`` cells, one lane per cell, every column a
contiguous float64 span the dt system owns (``state_spans``).  One player
move is one round.  The step law is masked span arithmetic with no branch
-- the same form as the x86 read head's ``transition`` -- so the same text
runs eagerly here and is what is handed to the compiler later.

Columns (state, one cell per lane):
    row, col, region   static geometry
    solution           hidden cat mask (the "mines" the hearts are judged by)
    cat, x_mark        the player's board
    hearts             lives, broadcast to every lane
    select_row, select_col, primary_action, secondary_action
                       the CONTROLS: the cell the player points at and the
                       two edge-triggered actions, broadcast to every lane.
                       A shell writes them before a round from the semantic
                       actions ``CONTROLS`` names; the law derives the move.

Published per round:
    cat_next, x_mark_next, hearts_next   the state the dt system lands
    residual   cats still to place (0 = solved); the dt system's own word
               for a law's convergence measure
    dt_limit   1.0: one move per round, never subdivided
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.tensors import AbstractTensor

HEARTS = 3.0

#: The control record: state column -> the semantic action a shell fills it
#: from (the viewport control-binding vocabulary; ``pointer-cell-*`` is the
#: pointer resolved to a board cell by the page's board geometry).  This is
#: what ``dt_system_contract(controls=CONTROLS)`` declares on the PieceState
#: fields and what every host contract publishes per slot.
CONTROLS = {
    "select_row": "pointer-cell-row",
    "select_col": "pointer-cell-col",
    "primary_action": "primary-action",
    "secondary_action": "secondary-action",
}
NO_CELL = -1.0


@dataclass(frozen=True)
class MeowdokuControls:
    """One round's controller state, as the Python game hands it over.

    ``select_row``/``select_col`` name the cell under the pointer (``-1``:
    none); ``primary_action`` places a cat, ``secondary_action`` toggles an
    X.  Both actions are edge-triggered: a shell sets one for the round in
    which it fired and clears it after."""

    select_row: float = NO_CELL
    select_col: float = NO_CELL
    primary_action: float = 0.0
    secondary_action: float = 0.0

    def columns(self, cells: int) -> dict[str, np.ndarray]:
        """The record as state columns: every field broadcast to the lanes."""
        return {name: np.full(cells, float(getattr(self, name))) for name in CONTROLS}


def meowdoku_step(row, col, region, solution, cat, x_mark, hearts,
                  select_row, select_col, primary_action, secondary_action, dt):
    """Advance the board by one move.  Every operand is an (n*n,) span."""
    cells = int(row.shape[0])
    r1 = row.reshape((cells, 1))
    r2 = row.reshape((1, cells))
    c1 = col.reshape((cells, 1))
    c2 = col.reshape((1, cells))
    g1 = region.reshape((cells, 1))
    g2 = region.reshape((1, cells))
    # Pairwise constraint incidence, derived from the geometry columns alone:
    # same row, same column, same region, or touching (diagonals included).
    touching = ((r1 - r2).abs() <= 1.0) & ((c1 - c2).abs() <= 1.0)
    conflict = (r1 == r2) | (c1 == c2) | (g1 == g2) | touching
    # Cats already constraining each cell.  A cell's own cat counts itself,
    # which is harmless: a cell holding a cat is never a placement candidate.
    load = (conflict.to_dtype("float") * cat.reshape((1, cells))).sum(dim=1)

    remaining = (solution * (1.0 - cat)).sum()
    active = (hearts > 0.0) & (remaining > 0.0)
    # The move, derived from the controls: the pointed cell and which action
    # fired.  Primary wins when both fire in one round.
    selected = (row == select_row) & (col == select_col)
    primary = primary_action > 0.0
    secondary = (secondary_action > 0.0) & ~primary
    place = active & selected & primary & (cat == 0.0)
    toggle = active & selected & secondary & (cat == 0.0)
    # A legal placement obeys the visible rules AND sits on the hidden
    # solution.  Anything else placed costs a heart and lands nothing.
    correct = place & (load == 0.0) & (solution == 1.0)
    wrong = place & ~correct

    cat_next = AbstractTensor.where(correct, 1.0, cat)
    x_mark_next = AbstractTensor.where(
        toggle, 1.0 - x_mark, AbstractTensor.where(correct, 0.0, x_mark))
    hearts_next = hearts - wrong.to_dtype("float").sum()
    residual = (solution * (1.0 - cat_next)).sum() + cat_next * 0.0
    dt_limit = cat_next * 0.0 + 1.0
    return cat_next, x_mark_next, hearts_next, residual, dt_limit


class MeowdokuPiece:
    """The law as a dt-system piece (``llvm_dt_system.PIECE_API``)."""

    entry = "meowdoku_step"
    argument_names = ("row", "col", "region", "solution", "cat", "x_mark",
                      "hearts", *CONTROLS, "dt")
    output_names = ("cat_next", "x_mark_next", "hearts_next", "residual", "dt_limit")

    def __init__(self, batch: int):
        self.batch = int(batch)
        self._bound = None

    def instantiate(self, columns, outputs=None):
        self._bound = tuple(columns[name] for name in self.argument_names)
        return ()

    def __call__(self, *columns):
        if len(columns) != len(self.argument_names):
            raise TypeError(
                f"{self.entry}: takes {len(self.argument_names)} columns, got {len(columns)}")
        tensors = [AbstractTensor.get_tensor(np.asarray(column, dtype=np.float64))
                   for column in columns]
        produced = meowdoku_step(*tensors)
        return tuple(
            np.broadcast_to(np.asarray(value.numpy(), dtype=np.float64), (self.batch,)).copy()
            for value in produced
        )


def initial_columns(puzzle) -> dict[str, np.ndarray]:
    """Every state column at round zero for ``puzzle``: no controls yet."""
    cells = puzzle.cells
    columns = puzzle.columns()
    columns["cat"] = np.zeros(cells)
    columns["x_mark"] = np.zeros(cells)
    columns["hearts"] = np.full(cells, HEARTS)
    columns.update(MeowdokuControls().columns(cells))
    return columns


def place_controls(r: int, c: int) -> MeowdokuControls:
    return MeowdokuControls(float(r), float(c), 1.0, 0.0)


def toggle_controls(r: int, c: int) -> MeowdokuControls:
    return MeowdokuControls(float(r), float(c), 0.0, 1.0)
