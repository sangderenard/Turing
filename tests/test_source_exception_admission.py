"""An authored catch must not disappear from a complete native artifact."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.glsl_deployment_strategy import CompilationSubdivisionRequired
from src.compiler.concordance_declarations import (
    CONTROL_BLOCK, ControlBlockKind,
    NAME_BINDING, SOURCE_EXCEPTION_ADMISSION, SOURCE_EXCEPTION_SITE_STATE,
    SOURCE_EXCEPTION_TRANSFER, SOURCE_EXCEPTION_HANDLER_CLASS,
    SOURCE_EXCEPTION_HANDLER_ENTRY,
    SOURCE_EXCEPTION_HANDLER_PHI,
)
from src.compiler.identity_concordance import (
    CorrelationTable, Unresolved, begin_identity_book, end_identity_book,
)


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)


def _lower(source: str, name: str, **kwargs):
    return lower_ast_source_to_ssa(
        source, "root", name=name, extraction_contract=CONTRACT,
        progress=lambda _message: None, progress_bars=False,
        **kwargs,
    )


def test_authored_catch_without_exceptional_control_refuses():
    # Before admission, the C artifact for NaN returned INT64_MIN instead of
    # running the ValueError handler and returning 2.
    source = """\
def root(x: float) -> int:
    try:
        y = int(x)
    except ValueError:
        y = 2
    return y
"""
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired) as error:
            _lower(source, "exception_admission_catch", identity_book=book)
        page = book.page(SOURCE_EXCEPTION_ADMISSION)
        row, = tuple(page.rows())
        source_cell, = row
        obligation = page.latest(row)
        assert isinstance(obligation, Unresolved)
        assert obligation.read == (source_cell,)
        receipt = book.latest_ref(SOURCE_EXCEPTION_ADMISSION, row)
        assert {cell for cell, _stage in book.edges_into(receipt)} == {
            source_cell
        }
        findings = CorrelationTable._source_exception_admission_findings(
            SimpleNamespace(metadata={"identity_book": book})
        )
        assert len(findings) == 1
        assert "no executable exceptional control" in findings[0].detail
        try_blocks = tuple(
            row for row in book.page(CONTROL_BLOCK).rows()
            if row[1] is ControlBlockKind.NATIVE_TRY
        )
        assert len(try_blocks) == 1
        try_fact = book.page(CONTROL_BLOCK).latest(try_blocks[0])
        assert len(try_fact.carried) == 1
        source_module = SimpleNamespace(metadata={"identity_book": book})
        for check in (
            CorrelationTable._source_exception_site_state_findings,
            CorrelationTable._source_exception_transfer_findings,
            CorrelationTable._source_exception_handler_class_findings,
            CorrelationTable._source_exception_handler_entry_findings,
            CorrelationTable._source_exception_handler_phi_findings,
            CorrelationTable._source_exception_continuation_phi_findings,
        ):
            assert check(source_module) == []
    finally:
        end_identity_book(token)
    boundary, = error.value.to_failure_mapping()["subdivision_boundaries"]
    assert boundary["kind"] == "source-exception-control-owner"
    assert "Ref('ingestion_value'" in boundary["source_try_cell"]


def test_authored_finally_without_exceptional_control_refuses():
    source = """\
def root(x: float) -> int:
    y = 0
    try:
        y = int(x)
    finally:
        y = y + 1
    return y
"""
    with pytest.raises(CompilationSubdivisionRequired) as error:
        _lower(source, "exception_admission_finally")
    boundary, = error.value.to_failure_mapping()["subdivision_boundaries"]
    assert boundary["kind"] == "source-exception-control-owner"


def test_every_retained_try_has_an_admission_obligation():
    source = """\
def root(x: float) -> int:
    try:
        y = int(x)
    except ValueError:
        y = 2
    try:
        z = int(x)
    finally:
        y = y + 1
    return y + z
"""
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired) as error:
            _lower(source, "exception_admission_all_tries", identity_book=book)
        rows = tuple(book.page(SOURCE_EXCEPTION_ADMISSION).rows())
        assert len(rows) == 2
        assert all(
            isinstance(book.page(SOURCE_EXCEPTION_ADMISSION).latest(row),
                       Unresolved)
            for row in rows
        )
    finally:
        end_identity_book(token)
    boundaries = error.value.to_failure_mapping()["subdivision_boundaries"]
    assert {item["source_try_cell"] for item in boundaries} == {
        repr(row[0]) for row in rows
    }


def test_exception_site_cites_binding_after_preceding_write():
    source = """\
def root(x: float) -> int:
    y = 3
    try:
        y = y + 1
        z = int(x)
    except ValueError:
        return y
    return z
"""
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_site_binding", identity_book=book)
        sites = tuple(book.page(SOURCE_EXCEPTION_SITE_STATE).rows())
        assert len(sites) == 1
        site = book.page(SOURCE_EXCEPTION_SITE_STATE).latest(sites[0])
        assert site.required_names == ("y",)
        binding, = (cell for name, cell in site.bindings if name == "y")
        versions = tuple(
            row for row in book.page(NAME_BINDING).rows()
            if row[0] == sites[0][0] and row[1] == "y"
        )
        assert len(versions) >= 2
        assert binding.row == max(versions, key=lambda row: row[2])
        transfer = book.page(SOURCE_EXCEPTION_TRANSFER).latest(sites[0])
        assert transfer.entry_bindings == site.bindings
        assert transfer.catch_try == site.scopes[-1][0]
        entries = tuple(book.page(SOURCE_EXCEPTION_HANDLER_ENTRY).rows())
        assert len(entries) == 1
        entry = book.page(SOURCE_EXCEPTION_HANDLER_ENTRY).latest(entries[0])
        assert entry.required_names == ("y",)
        assert entry.incoming_bindings == ((
            "y", ((entry.candidate_transfers[0], binding),),
        ),)
    finally:
        end_identity_book(token)


def test_handler_raise_routes_through_inner_finalizer_to_outer_try():
    source = """\
def root(x: int) -> int:
    try:
        try:
            raise ValueError("first")
        except ValueError:
            raise TypeError("second")
        finally:
            x = x + 1
    except TypeError:
        return 7
    return x
"""
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_nested_transfer", identity_book=book)
        transfers = tuple(
            book.page(SOURCE_EXCEPTION_TRANSFER).latest(row)
            for row in book.page(SOURCE_EXCEPTION_TRANSFER).rows()
        )
        from_handler = tuple(
            transfer for transfer in transfers
            if transfer.source_kind == "raise"
            and len(transfer.scopes) == 2
            and transfer.scopes[-1][1] == "handler"
        )
        assert len(from_handler) == 1
        transfer, = from_handler
        assert transfer.catch_try == transfer.scopes[0][0]
        assert transfer.unwind_finally == (transfer.scopes[-1][0],)
        classes = tuple(
            book.page(SOURCE_EXCEPTION_HANDLER_CLASS).latest(row)
            for row in book.page(SOURCE_EXCEPTION_HANDLER_CLASS).rows()
        )
        assert len(classes) == 2
        assert all(
            len(fact.declarations) == 1
            and fact.declarations[0][1] is not None
            for fact in classes
        )
    finally:
        end_identity_book(token)


def test_handler_phi_joins_distinct_exception_site_bindings():
    source = """\
def root(x: float) -> int:
    y = 0
    try:
        y = 1
        a = int(x)
        y = 2
        b = int(x)
    except ValueError:
        return y
    return y + a + b
"""
    book, token = begin_identity_book()
    try:
        with pytest.raises(CompilationSubdivisionRequired):
            _lower(source, "exception_handler_phi", identity_book=book)
        rows = tuple(book.page(SOURCE_EXCEPTION_HANDLER_PHI).rows())
        assert len(rows) == 1
        phi = book.page(SOURCE_EXCEPTION_HANDLER_PHI).latest(rows[0])
        assert phi.name == "y"
        assert len(phi.arms) == 2
        assert phi.arms[0][1] != phi.arms[1][1]
    finally:
        end_identity_book(token)


def test_cast_without_authored_try_still_lowers():
    source = """\
def root(x: float) -> int:
    return int(x)
"""
    module, _outputs, exports = _lower(source, "exception_admission_plain")
    assert exports[0] in module.functions
