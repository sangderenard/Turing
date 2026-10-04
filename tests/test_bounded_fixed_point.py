"""Shared loop guard: receipts on the book, bound and recurrence refusals."""

from __future__ import annotations

import pytest

from src.compiler.bounded_fixed_point import BoundedFixedPoint, bound_for
from src.compiler.concordance_declarations import FIXED_POINT_ROUND
from src.compiler.identity_concordance import ConcordanceRefusal, IdentityBook


def _rows(book, name="loop", scope=("s",)):
    page = book.pages[FIXED_POINT_ROUND.name]
    found = []
    for index in range(1, 50):
        fact = page.latest((name, scope, index))
        if fact is not None:
            found.append(fact)
    return found


def test_converges_within_bound_posts_one_row_per_round():
    book, lines = IdentityBook(), []
    guard = BoundedFixedPoint("loop", 5, scope=("s",), progress=lines.append,
                              book=book)
    plan = [(True, (1, 2)), (True, (3,)), (False, ())]
    results = [guard.round(c, changed_ids=ids, state=n)
               for n, (c, ids) in enumerate(plan)]
    assert results == [True, True, False]
    rows = _rows(book)
    assert [r[0] for r in rows] == [True, True, False]
    assert rows[0][2] == (1, 2) and rows[1][2] == (3,)
    assert lines[0] == "loop round 1/5 changed=2 ids=[1, 2]"
    assert len(lines) == 3


def test_receipts_chain_through_edges():
    book = IdentityBook()
    guard = BoundedFixedPoint("loop", 5, scope=("s",), book=book)
    guard.round(True, state=1)
    guard.round(False, state=2)
    second = book.latest_ref(FIXED_POINT_ROUND, ("loop", ("s",), 2))
    first = book.latest_ref(FIXED_POINT_ROUND, ("loop", ("s",), 1))
    assert [src.key for src, _ in book.edges_into(second)] == [first.key]


def test_bound_exhaustion_refuses_with_last_ids():
    book = IdentityBook()
    guard = BoundedFixedPoint("loop", 2, scope=("s",), book=book)
    guard.round(True, changed_ids=(7,), state="a")
    with pytest.raises(ConcordanceRefusal,
                       match=r"did not converge after 2 rounds.*\[9\]"):
        guard.round(True, changed_ids=(9,), state="b")
    assert len(_rows(book)) == 2          # the receipt precedes the refusal


def test_bound_reached_without_change_is_clean():
    guard = BoundedFixedPoint("loop", 1, scope=("s",), book=IdentityBook())
    assert guard.round(False, state=0) is False


def test_recurrence_refuse():
    book = IdentityBook()
    guard = BoundedFixedPoint("loop", 9, scope=("s",), book=book, state="a")
    guard.round(True, state="b")
    with pytest.raises(ConcordanceRefusal, match="period 1"):
        guard.round(True, state="b")
    assert _rows(book)[-1][3] == 1


def test_recurrence_to_initial_state_refuses():
    guard = BoundedFixedPoint("loop", 9, scope=("s",), book=IdentityBook(),
                              state="a")
    with pytest.raises(ConcordanceRefusal, match="period 1"):
        guard.round(True, state="a")


def test_recurrence_record_keeps_going():
    book = IdentityBook()
    guard = BoundedFixedPoint("loop", 9, scope=("s",), book=book,
                              recurrence="record", state="a")
    guard.round(True, state="b")
    assert guard.round(True, state="b") is True
    assert guard.round(False, state="b") is False
    assert [r[3] for r in _rows(book)] == [0, 1, 0]


def test_bound_for_formulas():
    assert bound_for("frame", functions=2, call_operands=3, records=4) == 10
    assert bound_for("planner", nodes=5, callsites=2) == 8
    assert bound_for("abi", functions=2, call_operands=3) == 6
    assert bound_for("other", functions=1, nodes=1) == 3
    assert bound_for("frame") == 1
