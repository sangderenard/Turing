"""A refused post reports the local neighborhood of the cells it stood on.

Pure data-structure tests on hand-posted ``IdentityBook`` rows: no lowering.
"""
from __future__ import annotations

import pytest

from src.compiler.identity_concordance import (
    Derived, IdentityBook, Mode, Novel, Registry, RowField, RowFieldKind,
    annotate_failure_with_neighborhood,
)

FIELDS = (
    RowField("scope", RowFieldKind.SCOPE),
    RowField("id", RowFieldKind.VALUE_ID),
)


def _book():
    registry = Registry()
    pages = {
        name: registry.declare_page(name, FIELDS, object)
        for name in ("read", "node", "binding")
    }
    stage = registry.declare_stage("probe")
    root = registry.declare_transform("probe_root", 0)
    return IdentityBook(registry=registry), pages, stage, root


def _disagreement():
    """Two reads of one consumer slot: 'fact' is recorded, then a second
    read proposes 'FRAME_BINDING' from different source cells."""
    book, pages, stage, root = _book()
    consumer = book.post(
        pages["node"], ("s", 1), "call", stage=stage,
        provenance=Novel(root, ()), mode=Mode.CONCORD,
    )
    first_binding = book.post(
        pages["binding"], ("s", 10), "fact", stage=stage,
        provenance=Novel(root, ()), mode=Mode.CONCORD,
    )
    second_binding = book.post(
        pages["binding"], ("s", 11), "FRAME_BINDING", stage=stage,
        provenance=Novel(root, ()), mode=Mode.CONCORD,
    )
    slot = book.post(
        pages["read"], ("s", 2), "fact", stage=stage,
        provenance=Derived((consumer, first_binding)), mode=Mode.CONCORD,
    )
    with pytest.raises(ValueError) as caught:
        book.post(
            pages["read"], ("s", 2), "FRAME_BINDING", stage=stage,
            provenance=Derived((consumer, second_binding)),
            mode=Mode.CONCORD,
        )
    return book, caught.value, slot, consumer, first_binding, second_binding


def test_refusal_keeps_its_type_and_names_its_claim():
    book, error, slot, *_ = _disagreement()
    assert type(error) is ValueError
    assert "disagreement" in str(error)
    claim, = error.concordance_claims
    assert claim.page == "read" and claim.mode == "CONCORD"
    assert dict((role, ref) for role, ref in claim.seeds)["incumbent"] == slot


def test_neighborhood_joins_both_competing_derivations():
    book, error, slot, consumer, first, second = _disagreement()
    lines = annotate_failure_with_neighborhood(error, book)
    text = "\n".join(lines)
    assert "failed post read" in text and "proposed='FRAME_BINDING'" in text
    # The incumbent's own derivation, the proposal's sources, and the cell
    # both stand on (the consumer) are all in the one neighborhood.
    assert "fact='fact'" in text and "fact='FRAME_BINDING'" in text
    assert "fact='call'" in text
    assert "--probe-->" in text
    assert error.__notes__ == [text]
    # Reported once.
    annotate_failure_with_neighborhood(error, book)
    assert len(error.__notes__) == 1


def test_reach_is_bounded_and_reports_what_it_left_out():
    book, pages, stage, root = _book()
    hub = book.post(
        pages["node"], ("s", 1), "hub", stage=stage,
        provenance=Novel(root, ()), mode=Mode.CONCORD,
    )
    for index in range(20):
        book.post(
            pages["read"], ("s", 100 + index), f"d{index}", stage=stage,
            provenance=Derived((hub,)), mode=Mode.CONCORD,
        )
    hood = book.neighborhood((hub,), radius=2, max_nodes=40, fanout=4)
    assert len(hood.nodes) == 5
    seed, = (node for node in hood.nodes if node.seed)
    assert seed.omitted == 16
    kept = {node.fact for node in hood.nodes if not node.seed}
    assert kept == {"'d19'", "'d18'", "'d17'", "'d16'"}  # newest first
    assert "(+16 neighbor(s) not shown)" in "\n".join(hood.lines())
    capped = book.neighborhood((hub,), radius=2, max_nodes=3, fanout=4)
    assert len(capped.nodes) == 3 and capped.truncated
