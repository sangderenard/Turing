"""A folded list row dtype keeps its exact callsite literal proof."""

import inspect
from pathlib import Path

from src.common.tensors.compression import octet_lz
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    NAME_BINDING, PLANNER_SPECIALIZATION_PAGE, PROCESS_GRAPH_OPERANDS,
    SEQUENCE_ROW_DTYPE, SOURCE_VALUE_CLASS, SpecializationFact,
    SpecializationSource,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book


def test_folded_list_row_rejects_wrong_literal_formal_class_and_operand():
    source = inspect.getsource(octet_lz._tensor) + (
        "\ndef root(packet: bytes, dictionary: bytes):\n"
        "    a = _tensor([0])\n"
        "    b = _tensor(packet + dictionary)\n"
        "    return a, b\n"
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": name, "storage": "span",
             "dtype": "int64", "rank": 1, "python_type": "builtins.bytes"}
            for name in ("packet", "dictionary")
        ], "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="specialized_list_row_audit",
        source_module=octet_lz.__name__, python_bindings=vars(octet_lz),
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None, progress_bars=False,
    )
    book = identity_book(module)
    page = book.page(SEQUENCE_ROW_DTYPE)
    row = next(row for row in page.rows()
               if page.latest(row)[1] == "specialized literal materialization")
    receipt = book.latest_ref(SEQUENCE_ROW_DTYPE, row)
    causes = tuple(source for source, _stage in book.edges_into(receipt))
    assert page.latest(row) == (("int64",), "specialized literal materialization")
    assert CorrelationTable._specialized_literal_materialized_row_findings(module) == []

    specialization = next(source for source in causes
                          if source.page == PLANNER_SPECIALIZATION_PAGE)
    spec_page = book.page(PLANNER_SPECIALIZATION_PAGE)
    spec_key = (specialization.row, specialization.column)
    original = spec_page.cells[spec_key]
    spec_page.cells[spec_key] = SpecializationFact([0.0], SpecializationSource.LITERAL)
    try:
        assert CorrelationTable._specialized_literal_materialized_row_findings(module)
    finally:
        spec_page.cells[spec_key] = original

    binding = next(source for source in causes if source.page == NAME_BINDING)
    bind_page = book.page(NAME_BINDING)
    bind_key = (binding.row, binding.column)
    original = bind_page.cells[bind_key]
    bind_page.cells[bind_key] = type(original)(
        row[1], original.authored, original.span_positions,
        original.context_sha256,
    )
    try:
        assert CorrelationTable._specialized_literal_materialized_row_findings(module)
    finally:
        bind_page.cells[bind_key] = original

    class_cell = next(source for source in causes
                      if source.page == SOURCE_VALUE_CLASS)
    class_page = book.page(SOURCE_VALUE_CLASS)
    class_key = (class_cell.row, class_cell.column)
    original = class_page.cells[class_key]
    class_page.cells[class_key] = ("builtins.bytes", *original[1:])
    try:
        assert CorrelationTable._specialized_literal_materialized_row_findings(module)
    finally:
        class_page.cells[class_key] = original

    operand = next(source for source in causes
                   if source.page == PROCESS_GRAPH_OPERANDS)
    operand_page = book.page(PROCESS_GRAPH_OPERANDS)
    operand_key = (operand.row, operand.column)
    original = operand_page.cells[operand_key]
    operand_page.cells[operand_key] = tuple(
        (receipt, role) if role == "arg:0" else (source, role)
        for source, role in original
    )
    try:
        assert CorrelationTable._specialized_literal_materialized_row_findings(module)
    finally:
        operand_page.cells[operand_key] = original
