"""A bytearray constructor copies its sequence argument into owned storage."""

from dataclasses import replace
from pathlib import Path

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SOURCE_SEQUENCE_CONSTRUCTOR_SNAPSHOT,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable


def test_mutated_bytearray_constructor_keeps_its_own_sequence_identity():
    source = ("def root(data: bytes) -> bytes:\n"
              "    header = bytearray((2,))\n"
              "    header.append(3)\n"
              "    return bytes(header)\n")
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "data",
                    "storage": "span", "dtype": "int64", "rank": 1,
                    "python_type": "builtins.bytes"}],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="bytearray_constructor_snapshot",
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = module.metadata["identity_book"]
    page = book.page(SOURCE_SEQUENCE_CONSTRUCTOR_SNAPSHOT)
    rows = tuple(page.rows())
    assert len(rows) == 1
    row = rows[0]
    cell = book.latest_ref(SOURCE_SEQUENCE_CONSTRUCTOR_SNAPSHOT, row)
    fact = page.cell(cell.row, cell.column)
    assert fact.destination_cell.row[1] == 4
    assert fact.source_cell.row[1] == 1
    assert {fact.destination_cell, fact.source_cell, fact.operand_cell,
            fact.destination_class_cell, fact.source_class_cell} <= {
                parent for parent, _stage in book.edges_into(cell)
            }
    assert module.sequence_tables[exports[0]].by_id(4) is not None
    assert CorrelationTable.build(module).findings(module) == []

    key = (cell.row, cell.column)
    page.cells[key] = replace(fact, source_cell=fact.destination_cell)
    try:
        assert CorrelationTable._source_sequence_constructor_snapshot_findings(
            module,
        )
    finally:
        page.cells[key] = fact

    calls = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("source_sequence_snapshot_cell") == cell
    ]
    assert len(calls) == 1
    assert calls[0].attributes["ssa_sequence_operation"] == "replace"
    del calls[0].attributes["source_sequence_snapshot_cell"]
    try:
        assert CorrelationTable._source_sequence_constructor_snapshot_findings(
            module,
        )
    finally:
        calls[0].attributes["source_sequence_snapshot_cell"] = cell


def test_two_specialized_helpers_keep_each_live_snapshot_obligation():
    source = ("def first(data: bytes) -> int:\n"
              "    header = bytearray((2,))\n"
              "    header.append(3)\n"
              "    return len(header)\n\n"
              "def second(data: bytes) -> int:\n"
              "    tail = bytearray((4,))\n"
              "    tail.append(5)\n"
              "    return len(tail)\n\n"
              "def root(data: bytes) -> int:\n"
              "    return first(data) + second(data)\n")
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "data",
                    "storage": "span", "dtype": "int64", "rank": 1,
                    "python_type": "builtins.bytes"}],
        "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="two_constructor_snapshots",
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = module.metadata["identity_book"]
    page = book.page(SOURCE_SEQUENCE_CONSTRUCTOR_SNAPSHOT)
    assert len(tuple(page.rows())) == 4  # Two source forks and two live forks.
    assert CorrelationTable._source_sequence_constructor_snapshot_findings(
        module,
    ) == []
    calls = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("source_sequence_snapshot_cell")
        is not None
    ]
    assert len(calls) == 2
    removed = calls[0].attributes.pop("source_sequence_snapshot_cell")
    try:
        findings = CorrelationTable._source_sequence_constructor_snapshot_findings(
            module,
        )
        assert any(
            finding.function == str(removed.row[0])
            and finding.value_id == removed.row[1]
            for finding in findings
        )
    finally:
        calls[0].attributes["source_sequence_snapshot_cell"] = removed
