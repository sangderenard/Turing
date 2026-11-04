"""A cross-frame shape withdrawal cites the changed source scope."""

from pathlib import Path

import pytest

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import INGEST_SOURCE, INGESTION
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable,
    ConcordanceRefusal,
    Mode,
    Novel,
    SHAPE_STATE_PAGE,
    begin_identity_book,
    current_identity_book,
    end_identity_book,
    invalidate_proven_shape,
)


def _state(book, scope, value_id):
    row = (scope, value_id)
    return book.post(
        SHAPE_STATE_PAGE, row,
        ("resolved", ((1,), "int64", 1, "static", None), ("fixture",)),
        stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
        mode=Mode.CONCORD,
    )


def test_withdrawal_uses_exact_upstream_scope_despite_same_id_collision():
    _book, token = begin_identity_book()
    try:
        book = current_identity_book()
        _state(book, ("target",), 3)
        wrong = _state(book, ("target",), 7)
        right = _state(book, ("source",), 7)
        invalidate_proven_shape(
            ("target",), 3, 7, "repository_ssa_enrichment",
            source_scope=("source",),
        )
        row = (("target",), 3)
        withdrawn = book.latest_ref(SHAPE_STATE_PAGE, row)
        causes = {source for source, _stage in book.edges_into(withdrawn)}
        assert book.page(SHAPE_STATE_PAGE).latest(row) == (
            "invalidated", 7, "repository_ssa_enrichment",
        )
        assert right in causes
        assert wrong not in causes
    finally:
        end_identity_book(token)


def test_withdrawal_refuses_unknown_upstream_scope_instead_of_same_id_proxy():
    _book, token = begin_identity_book()
    try:
        book = current_identity_book()
        _state(book, ("target",), 3)
        _state(book, ("target",), 7)
        with pytest.raises(ConcordanceRefusal, match="no source cell"):
            invalidate_proven_shape(
                ("target",), 3, 7, "repository_ssa_enrichment",
                source_scope=("missing",),
            )
    finally:
        end_identity_book(token)


def test_owned_byte_loop_return_shape_transport_finishes():
    # root's tuple-built bytearray reaches append helper through a borrowed
    # formal. Its reverse shape edge must withdraw from the helper's scope.
    source = """def root(data: bytes) -> bytes:
    header = bytearray((2,))
    header.append(3)
    for _ in range(1):
        if not data:
            return bytes(header)
    return bytes(header) + bytes(data)
"""
    contract = ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml")
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "data", "storage": "span",
            "dtype": "int64", "rank": 1, "python_type": "builtins.bytes",
        }],
        "records": {}, "bindings": [],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="owned_byte_loop_return_shape_transport",
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert module.functions
    assert CorrelationTable.build(module).findings(module) == []
