"""Live concordance follows exact causes, including unchanged-claim edges."""
import pickle

from src.compiler.concordance_declarations import INGESTION, INGEST_SOURCE
from src.compiler.identity_concordance import (
    Derived, Mode, Novel, RowField, RowFieldKind as KeyKind, begin_identity_book,
    declare_page, end_identity_book,
)


SOURCE = declare_page("test_live_source_dependencies", (RowField("name", KeyKind.NAME),), int)
VIEW = declare_page("test_live_source_dependency_view", (RowField("name", KeyKind.NAME),), tuple)


def _causes(read, claim, historical=None):
    sources = read.sources(historical) if historical is not None else read.sources(SOURCE, (claim,))
    return tuple((source.row, read(source.page, source.row)) for source in sources)


def _seed(book, name, value):
    return book.post(SOURCE, (name,), value, stage=INGESTION,
                     provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)


def test_new_cause_invalidates_live_view_even_when_claim_generation_is_unchanged():
    book, token = begin_identity_book()
    try:
        first, second = _seed(book, "first", 1), _seed(book, "second", 2)
        claim = book.post(SOURCE, ("claim",), 7, stage=INGESTION,
                          provenance=Derived((first,)), mode=Mode.CONCORD)
        view = book.derive_current(VIEW, ("view",), _causes, ("claim",), stage=INGESTION)
        assert book.page(VIEW).latest(view.row) == ((("first",), 1),)
        same_claim = book.post(SOURCE, ("claim",), 7, stage=INGESTION,
                               provenance=Derived((second,)), mode=Mode.CONCORD)
        assert same_claim == claim
        assert book.stale_derivations()
        current = book.current_ref(VIEW, view.row)
        assert current != view
        assert book.page(VIEW).latest(view.row) == ((("first",), 1), (("second",), 2))
        assert not book.stale_derivations()
        restored = pickle.loads(pickle.dumps(book))
        assert restored.current_ref(VIEW, view.row) == current
    finally:
        end_identity_book(token)


def test_historical_source_traversal_does_not_jump_to_the_revised_claim():
    book, token = begin_identity_book()
    try:
        first, second = _seed(book, "first", 1), _seed(book, "second", 2)
        old = book.post(SOURCE, ("claim",), 7, stage=INGESTION,
                        provenance=Derived((first,)), mode=Mode.CONCORD)
        book.post(SOURCE, ("claim",), 8, stage=INGESTION,
                  provenance=Derived((second,)), mode=Mode.REVISE)
        current = book.derive_current(VIEW, ("history",), _causes, ("claim", old), stage=INGESTION)
        assert book.page(VIEW).latest(current.row) == ((("first",), 1),)
    finally:
        end_identity_book(token)
