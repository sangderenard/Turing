"""A reader waiting on an absent row is woken by whichever writer posts it.

Ticket 53cc36f3: the call-result class readers of
``glsl_deployment_strategy._publish_call_result_class`` were woken only by
that function's own post, and were keyed by ``id(book)``.  The wait now
belongs to the book (``IdentityBook.await_row``) and ``post`` wakes it.
``await_row`` looks and registers in one transaction with ``post``'s commit
and hands the reader the cell, so no post falls between a look and a wait.
"""
import contextvars
import copy
import pickle
from types import SimpleNamespace
import threading

import networkx as nx
import pytest

from src.common.tensors import topological_reducer
from src.compiler import glsl_deployment_strategy
from src.compiler.concordance_declarations import (
    INGESTION, INGEST_SOURCE, REDUCTION, SOURCE_SPAN, SOURCE_VALUE_CLASS,
    SpanFact,
)
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


CALLER_SCOPE = ("test_row_readers", "caller")
CALLEE_SCOPE = ("test_row_readers", "callee")
CALL_ID = 3
CALL_VALUE = 30


def _span(book, name):
    return book.post(
        SOURCE_SPAN, ("test_row_readers", name, ()),
        SpanFact("Name", 1, 0, 1, 0, name),
        stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
        mode=Mode.CONCORD,
    )


def _caller(book, monkeypatch):
    """A caller graph whose call node ``CALL_ID`` carries ``CALL_VALUE``;
    its identity cell is a span the test authors."""
    graph = nx.DiGraph()
    graph.graph["authored_function_identity"] = CALLER_SCOPE
    graph.add_node(CALL_ID, value_id=CALL_VALUE)
    identity = _span(book, "call-node")
    monkeypatch.setattr(topological_reducer, "node_identity_cell",
                        lambda caller, node_id, **_: identity)
    return SimpleNamespace(G=graph), identity


def test_class_call_result_follows_the_handed_ref_not_the_latest(monkeypatch):
    """The reader acts on the Ref ``await_row`` hands it.  The callee's row
    holds an older and a newer cell of different class and width; handed
    the older Ref, the caller's class and its provenance edge follow it."""
    book, token = begin_identity_book()
    try:
        caller, identity = _caller(book, monkeypatch)
        span = _span(book, "returned")
        row = (CALLEE_SCOPE, 5)
        older = book.post(SOURCE_VALUE_CLASS, row, ("builtins.int", 1, "old"),
                          stage=REDUCTION, provenance=Derived((span,)),
                          mode=Mode.CONCORD)
        newer = book.post(SOURCE_VALUE_CLASS, row,
                          ("builtins.float", 2, "new"), stage=REDUCTION,
                          provenance=Derived((_span(book, "revised"),)),
                          mode=Mode.REVISE)
        assert newer != older
        assert book.latest_ref(SOURCE_VALUE_CLASS, row) == newer

        glsl_deployment_strategy._class_call_result(
            book, caller, CALL_ID, older)

        page = book.page(SOURCE_VALUE_CLASS)
        classed = book.latest_ref(SOURCE_VALUE_CLASS,
                                  (CALLER_SCOPE, CALL_VALUE))
        assert classed is not None
        assert tuple(page.cell(classed.row, classed.column)[:2]) == (
            "builtins.int", 1)
        sources = {source for source, _stage in book.edges_into(classed)}
        assert older in sources and identity in sources
        assert newer not in sources
    finally:
        end_identity_book(token)


def test_the_call_result_reader_and_a_reducer_writer_race_safely(
        monkeypatch):
    """Review 176f3a29, comments 816 and 820: the call-result reader runs
    outside the publication lock, so it may class the caller's call result
    while one of the reducer's writers classes the same row.  Both agree on
    the class and width, not on the free ``source`` string.  The reader
    looks at the row, and the reducer writer is let in right after that
    look: it must wait until the reader's post is committed, then concord
    with it.  Before the look and the post were one transaction, both found
    the row absent and the second post refused the first."""
    book, token = begin_identity_book()
    try:
        caller, _identity = _caller(book, monkeypatch)
        returned = topological_reducer._concord_source_value_class(
            CALLEE_SCOPE, 5, "builtins.int", source="callee writer",
            provenance=Derived((_span(book, "returned"),)))
        assert returned == ("builtins.int", 1)
        handed = book.latest_ref(SOURCE_VALUE_CLASS, (CALLEE_SCOPE, 5))
        reducer_span = _span(book, "reducer")
        target = (CALLER_SCOPE, CALL_VALUE)

        with concurrent_identity_publication(book):
            page_type = type(book.page(SOURCE_VALUE_CLASS))
            looked = page_type.latest
            results, errors = [], []
            # Review 7c0ad27d, comment 825: "the writer is still alive after
            # a short join" also holds when the scheduler has not run it
            # yet.  The writer thread reports instead which came first: an
            # acquisition of the publication lock that has to wait (the
            # reader's transaction holds it), or its own end.
            waited, settled = threading.Event(), threading.Event()
            writer_threads = []

            class _ReportingLock:
                def __init__(self, lock):
                    self._lock = lock

                def acquire(self, blocking=True, timeout=-1):
                    if (writer_threads
                            and threading.get_ident() == writer_threads[0]
                            and blocking):
                        if self._lock.acquire(blocking=False):
                            return True
                        waited.set()
                        settled.set()
                    return self._lock.acquire(blocking, timeout)

                def release(self):
                    self._lock.release()

                def __enter__(self):
                    self.acquire()
                    return self

                def __exit__(self, *exc):
                    self.release()

            book._publication_lock = _ReportingLock(book._publication_lock)

            def reducer_writer():
                writer_threads.append(threading.get_ident())
                try:
                    results.append(
                        topological_reducer._concord_source_value_class(
                            CALLER_SCOPE, CALL_VALUE, "builtins.int",
                            source="reducer writer",
                            provenance=Derived((reducer_span,))))
                except Exception as error:  # reported by the assertion
                    errors.append(error)
                finally:
                    settled.set()

            context = contextvars.copy_context()
            writer = threading.Thread(target=context.run,
                                      args=(reducer_writer,))
            reader_thread = threading.get_ident()
            looks, interleaved = [], []

            def look_then_let_the_writer_in(self, row, *args, **kwargs):
                found = looked(self, row, *args, **kwargs)
                if (tuple(row) == target
                        and threading.get_ident() == reader_thread
                        and not interleaved):
                    looks.append(found)
                    # The first look is the reader's own early return in
                    # ``_class_call_result``; the second is the one the
                    # post is decided on.
                    if len(looks) == 2:
                        writer.start()
                        assert settled.wait(timeout=10), (
                            "the reducer writer neither waited for the "
                            "publication lock nor finished")
                        interleaved.append(waited.is_set())
                return found

            monkeypatch.setattr(page_type, "latest",
                                look_then_let_the_writer_in)
            glsl_deployment_strategy._class_call_result(
                book, caller, CALL_ID, handed)
            writer.join(timeout=10)
            assert not writer.is_alive()
        assert looks == [None, None]
        assert errors == [], f"the reducer writer was refused: {errors!r}"
        assert results == [("builtins.int", 1)]
        assert interleaved == [True], (
            "the reducer writer ran between the look and the post without "
            "waiting for the reader's transaction")
        page = book.page(SOURCE_VALUE_CLASS)
        history = page.history(target)
        assert len(history) == 1, history
        _column, fact = history[0]
        assert tuple(fact) == ("builtins.int", 1, "resolved call result")
        classed = book.latest_ref(SOURCE_VALUE_CLASS, target)
        sources = {source for source, _stage in book.edges_into(classed)}
        assert handed in sources and reducer_span in sources
    finally:
        end_identity_book(token)
