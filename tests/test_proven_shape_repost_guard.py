"""record_proven_shape does not restate a fact it already posted."""

from src.compiler.concordance_declarations import (
    CANONICAL_VALUE,
    INGEST_SOURCE,
    INGESTION,
)
from src.compiler.identity_concordance import (
    _shape_key,
    Mode,
    Novel,
    begin_identity_book,
    current_identity_book,
    end_identity_book,
    invalidate_proven_shape,
    record_proven_shape,
)


def _cells(book, count):
    refs = []
    for value_id in range(1, count + 1):
        book.post(
            CANONICAL_VALUE, (("callers", 0), value_id),
            ("Input", "input", str(value_id)),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        refs.append(book.latest_ref(CANONICAL_VALUE, (("callers", 0), value_id)))
    return refs


def test_repeat_of_a_recorded_conflict_is_not_posted_nor_flipped():
    _book, token = begin_identity_book()
    try:
        book = current_identity_book()
        page = book.page("proven_shape")
        row = (_shape_key("tick"), 20)
        c1, c2, c3, c4 = _cells(book, 4)
        record_proven_shape("tick", 20, (8, 1, 1), "float64", 1, sources=(c1,))
        record_proven_shape("tick", 20, (8,), "float64", 0, sources=(c2, c3))
        assert page.latest(row)[0] == "conflicting"
        length = len(page.history(row))

        record_proven_shape("tick", 20, (8,), "float64", 0, sources=(c2, c3))
        assert len(page.history(row)) == length
        assert page.latest(row)[0] == "conflicting"

        record_proven_shape("tick", 20, (8,), "float64", 0, sources=(c2, c3, c4))
        assert len(page.history(row)) == length + 1
        assert page.latest(row) == ("conflicting", (8,), "float64", 0)
    finally:
        end_identity_book(token)


def test_invalidated_proof_at_a_passed_level_is_posted_once():
    _book, token = begin_identity_book()
    try:
        book = current_identity_book()
        page = book.page("proven_shape")
        row = (_shape_key("recurrence"), 68)
        c1, c2 = _cells(book, 2)
        record_proven_shape("recurrence", 68, (4,), "float64", 11, sources=(c1,))
        invalidate_proven_shape(
            "recurrence", 68, 7, "loop-continuation-rewired",
            source_cells=(c2,),
        )
        record_proven_shape("recurrence", 68, (4,), "float64", 13, sources=(c1,))
        assert page.latest(row) == ("proven", (4,), "float64", 13)
        length = len(page.history(row))
        record_proven_shape("recurrence", 68, (4,), "float64", 13, sources=(c1,))
        assert len(page.history(row)) == length
    finally:
        end_identity_book(token)
