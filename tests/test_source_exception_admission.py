"""An authored catch must not disappear from a complete native artifact."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.glsl_deployment_strategy import CompilationSubdivisionRequired
from src.compiler.concordance_declarations import SOURCE_EXCEPTION_ADMISSION
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


def test_cast_without_authored_try_still_lowers():
    source = """\
def root(x: float) -> int:
    return int(x)
"""
    module, _outputs, exports = _lower(source, "exception_admission_plain")
    assert exports[0] in module.functions
