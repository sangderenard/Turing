"""A reader waiting on an absent row is woken by whichever writer posts it.

Ticket 53cc36f3: the call-result class readers of
``glsl_deployment_strategy._publish_call_result_class`` were woken only by
that function's own post, and were keyed by ``id(book)``.  The wait now
belongs to the book (``IdentityBook.await_row``) and ``post`` wakes it.
``await_row`` looks and registers in one transaction with ``post``'s commit
and hands the reader the cell, so no post falls between a look and a wait.
"""
import copy
import inspect
import pickle
import threading

import pytest

from src.compiler import glsl_deployment_strategy
from src.compiler.concordance_declarations import INGESTION, INGEST_SOURCE
from src.compiler.identity_concordance import (
    Derived, Mode, Novel, RowField, RowFieldKind as KeyKind,
    begin_identity_book, concurrent_identity_publication, declare_page,
    end_identity_book,
)


ROWS = declare_page(
    "test_row_readers_rows", (RowField("name", KeyKind.NAME),), int,
)


def _seed(book, name, value):
    return book.post(ROWS, (name,), value, stage=INGESTION,
                     provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)


def _derive(book, name, value, source):
    return book.post(ROWS, (name,), value, stage=INGESTION,
                     provenance=Derived((source,)), mode=Mode.CONCORD)


def _fact(book, ref):
    return book.page(ROWS).cell(ref.row, ref.column)


def test_any_post_of_the_row_wakes_its_reader_once_with_its_cell():
    book, token = begin_identity_book()
    try:
        source = _seed(book, "source", 1)
        seen = []
        book.await_row(ROWS, ("late",), lambda ref: seen.append(ref))
        _seed(book, "other", 2)
        assert seen == []
        # Not the writer the reader expected: a plain DERIVED post.
        posted = _derive(book, "late", 7, source)
        assert seen == [posted]
        assert _fact(book, seen[0]) == 7
        # Woken once: a same-fact repost does not run it again.
        _derive(book, "late", 7, source)
        assert seen == [posted]
    finally:
        end_identity_book(token)


def test_a_present_row_is_handed_to_the_reader_at_once():
    book, token = begin_identity_book()
    try:
        present = _seed(book, "here", 5)
        seen = []
        book.await_row(ROWS, ("here",), lambda ref: seen.append(ref))
        assert seen == [present]
        assert not book.__dict__.get("_row_readers")
    finally:
        end_identity_book(token)


def test_a_reader_may_post_and_wait_again_from_inside_the_wake():
    book, token = begin_identity_book()
    try:
        source = _seed(book, "source", 1)
        order = []

        def first(ref):
            order.append("first")
            book.await_row(ROWS, ("b",), lambda ref: order.append("second"))
            _derive(book, "b", 2, source)

        book.await_row(ROWS, ("a",), first)
        _derive(book, "a", 1, source)
        assert order == ["first", "second"]
    finally:
        end_identity_book(token)


def test_a_failing_reader_does_not_cancel_the_others_and_the_row_stays():
    book, token = begin_identity_book()
    try:
        source = _seed(book, "source", 1)
        ran = []

        def failing(ref):
            ran.append("middle")
            raise RuntimeError("middle reader failed")

        book.await_row(ROWS, ("late",), lambda ref: ran.append("first"))
        book.await_row(ROWS, ("late",), failing)
        book.await_row(ROWS, ("late",), lambda ref: ran.append("last"))
        with pytest.raises(RuntimeError, match="middle reader failed"):
            _derive(book, "late", 9, source)
        assert ran == ["first", "middle", "last"]
        # The row is committed: a reader failed, not the write.
        committed = book.latest_ref(ROWS, ("late",))
        assert committed is not None and _fact(book, committed) == 9
        assert not book.__dict__.get("_row_readers")
    finally:
        end_identity_book(token)


def test_later_reader_failures_are_noted_on_the_first():
    book, token = begin_identity_book()
    try:
        source = _seed(book, "source", 1)

        def fail(message):
            def reader(ref):
                raise ValueError(message)
            return reader

        book.await_row(ROWS, ("late",), fail("first failure"))
        book.await_row(ROWS, ("late",), fail("second failure"))
        with pytest.raises(ValueError, match="first failure") as raised:
            _derive(book, "late", 3, source)
        assert any("second failure" in note
                   for note in getattr(raised.value, "__notes__", ()))
    finally:
        end_identity_book(token)


def test_no_post_falls_between_the_look_and_the_wait(monkeypatch):
    """The interleaving of review bf742186: a post of the row arrives after
    the reader looked and found nothing, before it registered.  Under
    concurrent publication the look and the registration hold the lock
    ``post`` commits under, so that post waits, then wakes the reader."""
    book, token = begin_identity_book()
    try:
        source = _seed(book, "source", 1)
        with concurrent_identity_publication(book):
            concurrent = type(book)
            looked = concurrent.latest_ref
            seen, written = [], []
            writer = threading.Thread(
                target=lambda: written.append(_derive(book, "race", 11, source)),
            )
            interleaved = []

            def look_then_let_the_writer_in(self, page, row):
                found = looked(self, page, row)
                if page is ROWS and row == ("race",) and not interleaved:
                    interleaved.append(found)
                    writer.start()
                    # The writer must not commit while this reader sits
                    # between its look and its registration.
                    writer.join(timeout=0.5)
                    interleaved.append(writer.is_alive())
                return found

            monkeypatch.setattr(concurrent, "latest_ref",
                                look_then_let_the_writer_in)
            book.await_row(ROWS, ("race",), lambda ref: seen.append(ref))
            writer.join(timeout=10)
            assert not writer.is_alive()
            assert interleaved == [None, True], (
                "a post committed between await_row's look and its wait"
            )
            assert len(written) == 1
            assert seen == written
            assert _fact(book, seen[0]) == 11
    finally:
        end_identity_book(token)


def test_readers_belong_to_their_book_and_are_not_copied():
    book, token = begin_identity_book()
    try:
        source = _seed(book, "source", 1)
        seen = []
        book.await_row(ROWS, ("late",), lambda ref: seen.append("woken"))
        for other in (pickle.loads(pickle.dumps(book)), copy.deepcopy(book)):
            _derive(other, "late", 3, source)
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


def test_publish_call_result_class_acts_on_the_handed_cell_only():
    """No second lookup gap: the publisher never reads the returned row
    itself; it acts on the Ref ``await_row`` hands to its reader."""
    publish = inspect.getsource(
        glsl_deployment_strategy._publish_call_result_class)
    assert "await_row" in publish
    assert "latest_ref" not in publish
