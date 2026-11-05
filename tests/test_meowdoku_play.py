"""The window runner's shell: pointer and buttons become control records."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku.play import Game  # noqa: E402


def test_clicks_place_cats_and_toggle_marks_through_the_session():
    game = Game(n=5, seed=1, size=500)
    cols = game.session.puzzle.solution_columns()
    cell = 100
    assert game.pointer_cell(cols[0] * cell + 50, 50) == (0, cols[0])
    assert game.pointer_cell(-1, 50) is None and game.pointer_cell(50, 500) is None

    assert game.click(cols[0] * cell + 50, 50, 1)          # left: place a correct cat
    assert game.session.cat[0, cols[0]] == 1.0 and game.session.hearts == 3
    assert game.click(2 * cell + 50, 2 * cell + 50, 3)     # right: toggle an X
    assert game.session.x_mark[2, 2] == 1.0
    assert not game.click(2 * cell + 50, 2 * cell + 50, 2) # middle button: no action
    wrong = next((r, c) for r in range(5) for c in range(5) if cols[r] != c and r > 1)
    assert game.click(wrong[1] * cell + 50, wrong[0] * cell + 50, 1)
    assert game.session.hearts == 2 and "heart" in game.message

    game.key("u")
    assert game.session.hearts == 3
    image = game.board()
    assert image.shape == (500, 500, 3) and image.dtype.name == "uint8"
    assert "hearts 3" in game.status()
    game.key("n")
    assert game.session.cat.sum() == 0 and game.seed == 2
