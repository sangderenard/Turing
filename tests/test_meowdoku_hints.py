"""Meowdoku hints (sound deductions, no dt-system effect) and palettes."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku.games import PALETTES, MeowdokuGame  # noqa: E402
from meowdoku.puzzle import generate, next_deduction  # noqa: E402


def _board(n=5, seed=1):
    p = generate(n, seed)
    return {"id": "t", "n": n, "region": p.region.astype(int).tolist(), "solution": p.solution.astype(int).tolist(),
            "hues": [k / n for k in range(n)]}


def test_hint_is_sound_and_leaves_the_dt_state_alone():
    board = _board()
    game = MeowdokuGame(board)
    game.cell(0, int(np.argmax(np.asarray(board["solution"]).reshape(5, 5)[0])), "cat")
    before = (game.session.cat.copy(), game.session.x_mark.copy(), game.session.hearts, game.session.calls())
    game.hint()
    st = game.state()
    assert st["hints_used"] == 1 and sum(st["hints"]) >= 1 and "1 hint" in st["status"]
    solution = np.asarray(board["solution"])
    cells = [i for i, v in enumerate(st["hints"]) if v]
    if st["hint"].startswith("hint: cat"):
        assert all(solution[i] == 1 for i in cells)
    else:
        assert all(solution[i] == 0 for i in cells)
    after = (game.session.cat, game.session.x_mark, game.session.hearts, game.session.calls())
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_every_hint_along_a_solve_is_sound():
    board = _board(6, 3)
    region = np.asarray(board["region"]).reshape(6, 6)
    solution = np.asarray(board["solution"]).reshape(6, 6)
    cats = np.zeros((6, 6), dtype=int)
    placed = 0
    for _ in range(40):
        found = next_deduction(region, cats, solution)
        if not found["cells"]:
            break
        flat = solution.reshape(-1)
        if found["kind"] == "cat":
            assert all(flat[i] == 1 for i in found["cells"])
            cats.reshape(-1)[found["cells"][0]] = 1
            placed += 1
        else:
            assert all(flat[i] == 0 for i in found["cells"])
            break
    assert placed >= 1


def test_palettes_give_distinct_region_colours():
    board = _board()
    game = MeowdokuGame(board)
    for name in PALETTES:
        game.set_palette(name)
        st = game.state()
        colours = {tuple(st["palette"][f"region{k}"]) for k in range(5)}
        assert len(colours) == 5 and st["palette_name"] == name
