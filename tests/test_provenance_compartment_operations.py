"""Coarse book operations: queries, provenance searches and complete formatting.

The installer runs these with the canonical module already selected native.
They also establish the incumbent Python behavior without compiling anything.
"""
import lzma
import hashlib

import pytest

from src.compiler.identity_concordance import (
    Derived, IdentityBook, Mode, Novel, Registry, RowField, RowFieldKind,
    render_identity_book, write_identity_log,
)


def _chain(count=100):
    registry = Registry()
    page = registry.declare_page("compartment_value", (
        RowField("identity", RowFieldKind.VALUE_ID),), int)
    stage = registry.declare_stage("compartment")
    origin = registry.declare_transform("compartment_root", 0)
    book = IdentityBook(registry=registry)
    refs = []
    for identity in range(count):
        refs.append(book.post(page, (identity,), identity, stage=stage,
                              provenance=Derived((refs[-1],)) if refs else Novel(origin, ()),
                              mode=Mode.CONCORD))
    return book, page, stage, refs


def test_provenance_searches_and_history_preserve_exact_cells():
    book, page, stage, refs = _chain()
    cursor, count = refs[-1], 1
    while cursor != refs[0]:
        ((previous, recorded_stage),) = book.edges_into(cursor)
        assert recorded_stage == stage
        assert (cursor, stage) in book.edges_out_of(previous)
        cursor = previous
        count += 1
    assert count == 100
    assert book.latest_by_page((99,))[page.name] == 99
    assert book.page(page).history((99,)) == ((0, 99),)
    assert book.latest_ref(page, (99,)) == refs[-1]
    with pytest.raises(ValueError):
        book.post(page, (99,), -1, stage=stage, provenance=Derived((refs[-2],)), mode=Mode.CONCORD)


def test_whole_book_format_and_compressed_write_are_complete_operations(tmp_path):
    book, page, stage, refs = _chain()
    text = render_identity_book(book)
    # Independently confirmed on untouched main: exact complete formatting,
    # not just agreement between two implementations on this branch.
    assert hashlib.sha256(text.encode()).hexdigest() == (
        "912c1bf93a1c0e1ff52b5dfffd731e6114f3d2210f9e36e30044baa582288620")
    assert "compartment_value" in text
    assert "compartment_root" in text
    assert "99" in text
    path = write_identity_log(book, tmp_path / "complete")
    assert path is not None
    with lzma.open(path, "rt") as stream:
        assert stream.read() == text + "\n"
