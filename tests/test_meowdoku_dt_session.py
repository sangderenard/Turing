"""Meowdoku played through the dt system: one move, one round, one call."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku.session import RESIDUAL, MeowdokuSession  # noqa: E402


def test_each_move_is_exactly_one_round_and_one_law_call():
    game = MeowdokuSession(n=5, seed=1)
    cols = game.puzzle.solution_columns()
    game.place(0, cols[0])
    assert game.calls() == 1
    assert game.cat[0, cols[0]] == 1.0
    assert game.telemetry[RESIDUAL] == 4.0
    game.toggle_x(3, 3)
    assert game.calls() == 2
    assert game.x_mark[3, 3] == 1.0
    game.toggle_x(3, 3)
    assert game.calls() == 3 and game.x_mark[3, 3] == 0.0


def test_undo_and_reset_are_the_dt_systems_restore():
    game = MeowdokuSession(n=5, seed=1)
    cols = game.puzzle.solution_columns()
    game.place(0, cols[0])
    game.toggle_x(2, 2)
    wrong = next((r, c) for r in range(5) for c in range(5) if cols[r] != c and r > 1)
    game.place(*wrong)
    assert game.hearts == 2 and game.x_mark[2, 2] == 1.0

    game.undo()
    assert game.hearts == 3 and game.x_mark[2, 2] == 1.0 and game.cat[0, cols[0]] == 1.0
    game.undo()
    assert game.x_mark[2, 2] == 0.0 and game.cat[0, cols[0]] == 1.0
    game.undo()
    assert game.cat.sum() == 0.0 and game.hearts == 3
    game.undo()                                   # nothing left: stays at round zero
    assert game.cat.sum() == 0.0

    for r in range(5):
        game.place(r, cols[r])
    assert game.won
    game.reset()
    assert not game.won and game.cat.sum() == 0.0 and game.hearts == 3
    # The restored state is live: play continues on the same span.
    game.place(0, cols[0])
    assert game.cat[0, cols[0]] == 1.0


def test_full_game_to_win_through_the_dt_system():
    game = MeowdokuSession(n=8, seed=7)
    cols = game.puzzle.solution_columns()
    for r in range(8):
        game.place(r, cols[r])
    assert game.won and game.calls() == 8
    np.testing.assert_array_equal(game.cat.reshape(-1), game.puzzle.solution)
