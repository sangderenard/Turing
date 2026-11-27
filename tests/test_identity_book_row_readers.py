"""A reader waiting on an absent row is woken by whichever writer posts it.

Ticket 53cc36f3: the call-result class readers of
``glsl_deployment_strategy._publish_call_result_class`` were woken only by
that function's own post, and were keyed by ``id(book)``.  The wait now
belongs to the book (``IdentityBook.await_row``) and ``post`` wakes it.
"""
import copy
import pickle

from src.compiler import glsl_deployment_strategy
from src.compiler.concordance_declarations import INGESTION, INGEST_SOURCE
from src.compiler.identity_concordance import (
    Derived, Mode, Novel, RowField, RowFieldKind as KeyKind,
    begin_identity_book, declare_page, end_identity_book,
)


ROWS = declare_page(
    "test_row_readers_rows", (RowField("name", KeyKind.NAME),), int,
)


def _seed(book, name, value):
    return book.post(ROWS, (name,), value, stage=INGESTION,
                     provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)


def test_any_post_of_the_row_wakes_its_reader_once():
    book, token = begin_identity_book()
    try:
        source = _seed(book, "source", 1)
        seen = []
        book.await_row(ROWS, ("late",),
                       lambda: seen.append(book.page(ROWS).latest(("late",))))
        _seed(book, "other", 2)
        assert seen == []
        # Not the writer the reader expected: a plain DERIVED post.
        book.post(ROWS, ("late",), 7, stage=INGESTION,
                  provenance=Derived((source,)), mode=Mode.CONCORD)
        assert seen == [7]
        # Woken once: a same-fact repost does not run it again.
        book.post(ROWS, ("late",), 7, stage=INGESTION,
                  provenance=Derived((source,)), mode=Mode.CONCORD)
        assert seen == [7]
    finally:
        end_identity_book(token)


def test_a_reader_may_post_and_wait_again_from_inside_the_wake():
    book, token = begin_identity_book()
    try:
        source = _seed(book, "source", 1)
        order = []

        def first():
            order.append("first")
            book.await_row(ROWS, ("b",), lambda: order.append("second"))
            book.post(ROWS, ("b",), 2, stage=INGESTION,
                      provenance=Derived((source,)), mode=Mode.CONCORD)

        book.await_row(ROWS, ("a",), first)
        book.post(ROWS, ("a",), 1, stage=INGESTION,
                  provenance=Derived((source,)), mode=Mode.CONCORD)
        assert order == ["first", "second"]
    finally:
        end_identity_book(token)


def test_readers_belong_to_their_book_and_are_not_copied():
    book, token = begin_identity_book()
    try:
        source = _seed(book, "source", 1)
        seen = []
        book.await_row(ROWS, ("late",), lambda: seen.append("woken"))
        for other in (pickle.loads(pickle.dumps(book)), copy.deepcopy(book)):
            other.post(ROWS, ("late",), 3, stage=INGESTION,
                       provenance=Derived((source,)), mode=Mode.CONCORD)
        assert seen == []
    finally:
        end_identity_book(token)
    other, token = begin_identity_book()
    try:
        _seed(other, "late", 4)
        assert seen == []
    finally:
        end_identity_book(token)


def test_call_result_readers_have_no_process_global_table():
    assert not hasattr(glsl_deployment_strategy, "_CALL_RESULT_CLASS_READERS")
