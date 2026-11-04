"""An eager callee Add class reaches its exact authored call results."""

from pathlib import Path

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SEMANTIC_OUTPUT, SOURCE_CALL_RESULT_IDENTITY, SOURCE_VALUE_CLASS,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable


def test_two_reached_bytes_returns_use_callee_output_class():
    source = """def encode(data: bytes) -> bytes:
    return bytes(data) + bytes(data)
def root(data: bytes) -> bytes:
    return encode(data) + encode(data)
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "data",
            "storage": "span", "dtype": "int64", "rank": 1,
            "python_type": "builtins.bytes",
        }],
        "records": {}, "bindings": [],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = module.metadata["identity_book"]
    assert CorrelationTable._source_eager_bytes_return_findings(module) == []
    assert CorrelationTable._source_bytes_concat_class_findings(module) == []
    reached = [
        book.latest_ref(SOURCE_CALL_RESULT_IDENTITY, row)
        for row in book.page(SOURCE_CALL_RESULT_IDENTITY).rows()
        if any(parent.page == SEMANTIC_OUTPUT
               for parent, _ in book.edges_into(
                   book.latest_ref(SOURCE_CALL_RESULT_IDENTITY, row)
               ))
    ]
    assert len(reached) == 2
    assert all(book.page(SOURCE_CALL_RESULT_IDENTITY).cell(
        cell.row, cell.column,
    )[1:] == ("builtins.bytes", 1) for cell in reached)
    assert all(book.page(SOURCE_VALUE_CLASS).latest(cell.row)[:2]
               == ("builtins.bytes", 1) for cell in reached)

    # A forged class result retains its call/address/return edges and must
    # still fail the exact result audit.
    page = book.page(SOURCE_CALL_RESULT_IDENTITY)
    target = reached[0]
    key = (target.row, target.column)
    original = page.cells[key]
    page.cells[key] = (original[0], "builtins.int", 1)
    try:
        assert CorrelationTable._source_eager_bytes_return_findings(module)
    finally:
        page.cells[key] = original
