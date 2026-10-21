"""A result storage lease for an accepted callee cites the callee's value cell
through the callee's owner book (an IMPORTED_CELL receipt), and a missing row
refuses by name instead of dropping out of the cause set."""

import hashlib

import pytest

from src.compiler.frame_identity_book import (
    _frame_cells,
    _frame_required_value_cell,
    _frame_value_cell,
)
from src.compiler.identity_concordance import (
    ConcordanceRefusal,
    begin_identity_book,
    end_identity_book,
)
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

from tests.compiler_provenance_inputs import publish_authored_ssa

IMPORTED = "concordance_imported_cell"


def _callee():
    value = SSAValue(1, "float64")
    return Function(
        "accepted", [], {"entry": BasicBlock("entry", [Instr("Ret", [value], None)])},
    )


def _accepted_in_owner_book():
    digest = hashlib.sha256(b"lease-fixture").hexdigest()
    callee = _callee()
    owner, token = begin_identity_book()
    try:
        publish_authored_ssa({"accepted": callee})
        callee.metadata["accepted_ssa_component"] = {
            "schema": "turing.accepted-ssa-component.v1",
            "artifact_sha256": digest,
            "identity_book": owner,
        }
    finally:
        end_identity_book(token)
    return callee, owner, digest


def test_accepted_callee_value_cell_is_cited_by_imported_receipt():
    callee, owner, digest = _accepted_in_owner_book()
    ambient, token = begin_identity_book()
    try:
        # The row exists only in the owner book; the ambient book has none.
        assert _frame_value_cell(callee, 1, book=ambient) is None
        causes = _frame_cells(_frame_required_value_cell(callee, 1))
        assert len(causes) == 1 and causes[0].page.name == IMPORTED
        assert causes[0].row[0] == digest
        assert ambient.page(IMPORTED).rows()
    finally:
        end_identity_book(token)


def test_missing_owner_row_refuses_by_name():
    callee, owner, digest = _accepted_in_owner_book()
    ambient, token = begin_identity_book()
    try:
        with pytest.raises(ConcordanceRefusal) as refusal:
            _frame_required_value_cell(callee, 99)
        message = str(refusal.value)
        assert "'accepted'" in message and "99" in message
        assert "accepted component" in message
    finally:
        end_identity_book(token)


def test_ambient_callee_is_cited_directly():
    callee = _callee()
    ambient, token = begin_identity_book()
    try:
        publish_authored_ssa({"accepted": callee})
        cell = _frame_required_value_cell(callee, 1)
        assert cell is not None and cell.page.name != IMPORTED
        assert not ambient.page(IMPORTED).rows()
    finally:
        end_identity_book(token)
