"""Meowdoku cat roster: one deterministic identity per solution cat, lit once that cat is home."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku.games import MeowdokuGame  # noqa: E402
from meowdoku.puzzle import generate  # noqa: E402


def _board(n=5, seed=1, ident="t"):
    p = generate(n, seed)
    return {"id": ident, "n": n, "region": p.region.astype(int).tolist(), "solution": p.solution.astype(int).tolist(),
            "hues": [k / n for k in range(n)]}


def test_identities_are_ordered_stable_and_seeded_by_board_and_cell():
    board = _board()
    cells = [int(i) for i in np.flatnonzero(np.asarray(board["solution"]))]
    a, b = MeowdokuGame(board).roster(), MeowdokuGame(board).roster()
    assert a == b and [r["cell"] for r in a] == cells == sorted(cells)
    assert [r["id"] for r in a] == [f"cat-{c}" for c in cells] and [r["index"] for r in a] == list(range(len(cells)))
    assert len({r["coat"] for r in a}) == len(a)
    other = MeowdokuGame(_board(ident="u")).roster()
    assert [r["coat"] for r in other] != [r["coat"] for r in a]
    assert not any(r["lit"] for r in a)


def test_a_cat_lights_when_that_cat_is_placed_or_revealed():
    board = _board()
    game = MeowdokuGame(board)
    first = game.roster()[0]
    game.cell(*divmod(first["cell"], 5), "cat")
    st = game.state()
    lit = {r["id"]: r["lit"] for r in st["roster"]}
    assert lit[first["id"]] and sum(lit.values()) == 1
    pieces = {p["id"]: p for p in st["pieces"] if p["id"].startswith("roster-")}
    assert pieces[f"roster-{first['id']}"]["color"] == "catlit" and pieces[f"roster-{first['id']}"]["cell"] == "roster:0"
    assert sum(p["color"] == "catdim" for p in pieces.values()) == len(pieces) - 1
    assert "catlit" in st["emissive"]
    game.power("cat")                                   # a revealed cat lights its own roster entry
    assert sum(r["lit"] for r in game.state()["roster"]) == 2
