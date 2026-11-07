"""Wordle OVERTIME: guesses past six are accepted, the board is a six-row window, the rack learns from all guesses, give-up reveals."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku.games import Wordle  # noqa: E402


def _game(extra=0):
    g = Wordle(3)
    pool = sorted(w for w in g.words if len(w) == 5 and w != g.answer)
    wrong = pool[:10]
    return g, wrong


def _play(g, words):
    for w in words:
        g.text(w); g.submit()


def test_overtime_guesses_accepted_and_window_is_last_six_rows():
    g, wrong = _game()
    _play(g, wrong[:5])
    st = g.state()
    assert st["overtime"] == 0 and len(g.guesses) == 5 and not st["lost"]
    _play(g, wrong[5:8])                       # guesses 6, 7, 8: the sixth wrong guess does NOT end the game
    st = g.state()
    assert len(g.guesses) == 8 and not st["lost"] and not st["won"] and "was" not in st["status"]
    assert st["overtime"] == st["window_offset"] == 3
    ids = {p["id"] for p in st["pieces"] if p["id"][:1] == "g" and p["id"][1:2].isdigit()}
    assert {i.split("-")[0] for i in ids} == {"g3", "g4", "g5", "g6", "g7"}     # the last five guesses + the entry row
    # evidence colours are intact on the window's rows
    row0 = st["cells"][:5]
    assert row0 == Wordle.score(wrong[3], g.answer)
    assert st["cells"][5 * 5:] == ["cell"] * 5           # the fresh entry row


def test_rack_evidence_covers_all_guesses():
    g, wrong = _game()
    first = wrong[0]
    _play(g, wrong[:8])
    st = g.state()
    scrolled = set(first) - set("".join(wrong[3:8]))
    assert scrolled, "need a letter only the scrolled-off guess used"
    rack = {p["letter"]: st["rack"]["cells"][i] for i, p in enumerate(st["rack"]["pieces"]) if "letter" in p}
    # (rack cells are padded per row; index by piece cell instead)
    by_cell = {p["letter"]: st["rack"]["cells"][p["cell"]] for p in st["rack"]["pieces"]}
    for ch in scrolled:
        assert by_cell[ch.upper()] != "unknown"


def test_give_up_reveals_the_answer_and_ends_the_game():
    g, wrong = _game()
    _play(g, wrong[:7])
    assert g.state()["give_up"] is not None
    g.give_up()
    st = g.state()
    assert st["lost"] and g.answer.upper() in st["status"] and st["give_up"] is None
    n = len(g.guesses)
    g.text(g.answer); g.submit()
    assert len(g.guesses) == n                       # over: no more guesses
    assert g.result() is None


def test_overtime_win_scores_zero_and_marks_overtime():
    g, wrong = _game()
    _play(g, wrong[:7])
    g.text(g.answer); g.submit()
    r = g.result()
    assert g.won and r["score"] == 0 and r["overtime"] == 2 and r["detail"] == f"{g.answer.upper()} in 6+2"
    assert not g.state()["lost"] and len(g.state()["pieces"]) > 0
    # a regulation win still scores 7 - guesses
    h, wrong = _game()
    _play(h, wrong[:2]); h.text(h.answer); h.submit()
    assert h.result()["score"] == 4 and "overtime" not in h.result()


def test_undo_across_the_window_boundary():
    g, wrong = _game()
    _play(g, wrong[:6])
    assert g.state()["window_offset"] == 1
    g.undo()
    assert g.state()["window_offset"] == 0 and len(g.guesses) == 5
    g.give_up(); assert g.state()["lost"]
    g.undo(); assert not g.gave_up and not g.state()["lost"]
    g.reset(); assert g.guesses == [] and not g.gave_up


def test_entry_bar_state():
    g, wrong = _game()
    g.text("ab")
    bar = g.state()["bar"]
    assert bar["cols"] == 7 and bar["cells"][0] == bar["cells"][-1] == "entry"
    assert bar["cells"][1:3] == ["typed", "typed"] and bar["cells"][3:6] == ["unknown"] * 3
    assert [p["mesh"] for p in bar["pieces"] if p["id"].startswith("bar-") and p["cell"] in (1, 2)] == ["A", "B"]


def test_overtime_win_breaks_the_streak(tmp_path):
    from meowdoku.profiles import Profiles
    p = Profiles(tmp_path / "p.json")
    p.ensure("amy")
    assert p.award("amy", "wordle", {"score": 3})["streak"] == 1
    assert p.award("amy", "wordle", {"score": 0, "overtime": 1})["streak"] == 0
