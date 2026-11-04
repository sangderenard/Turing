"""Private bytearray heap admission follows its exact constructor and append."""

from dataclasses import replace
from pathlib import Path

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SEQUENCE_DESCRIPTOR, SOURCE_PRIVATE_DYNAMIC_SEQUENCE,
    SOURCE_SEQUENCE_MUTATION,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.identity_concordance import EDGE_PAGE
from src.transmogrifier.ssa import SSASequenceCapacityPolicy


def test_private_bytearray_admission_requires_exact_append_and_descriptor():
    source = ("def root(data: bytes) -> bytes:\n"
              "    header = bytearray()\n"
              "    header.append(2)\n"
              "    for _ in range(1):\n"
              "        if not data:\n"
              "            return bytes(header)\n"
              "    return bytes(header) + bytes(data)\n")
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
        source, "root", name="two_private_bytearrays",
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = module.metadata["identity_book"]
    page = book.page(SOURCE_PRIVATE_DYNAMIC_SEQUENCE)
    rows = tuple(page.rows())
    assert len(rows) == 1
    assert CorrelationTable._source_private_dynamic_sequence_findings(
        module,
    ) == []
    assert CorrelationTable.build(module).findings(module) == []

    left_ref = book.latest_ref(SOURCE_PRIVATE_DYNAMIC_SEQUENCE, rows[0])
    left = page.cell(left_ref.row, left_ref.column)
    mutation_page = book.page(SOURCE_SEQUENCE_MUTATION)
    mutation_ref = left.mutation_cell
    mutation = mutation_page.cell(mutation_ref.row, mutation_ref.column)
    mutation_key = (mutation_ref.row, mutation_ref.column)
    mutation_page.cells[mutation_key] = (
        int(left_ref.row[1]) + 1, *mutation[1:],
    )
    try:
        assert CorrelationTable._source_private_dynamic_sequence_findings(
            module,
        )
    finally:
        mutation_page.cells[mutation_key] = mutation
    key = (left_ref.row, left_ref.column)
    page.cells[key] = replace(left, mutation_cell=left.class_cell)
    try:
        assert CorrelationTable._source_private_dynamic_sequence_findings(
            module,
        )
    finally:
        page.cells[key] = left

    descriptor_page = book.page(SEQUENCE_DESCRIPTOR)
    descriptor_ref = next(
        ref
        for row in descriptor_page.rows()
        for ref in (book.latest_ref(SEQUENCE_DESCRIPTOR, row),)
        if left_ref in {cause for cause, _stage in book.edges_into(ref)}
    )
    descriptor = descriptor_page.cell(
        descriptor_ref.row, descriptor_ref.column,
    )
    assert descriptor.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC
    descriptor_key = (descriptor_ref.row, descriptor_ref.column)
    descriptor_page.cells[descriptor_key] = replace(
        descriptor, capacity_policy=SSASequenceCapacityPolicy.FIXED,
        heap_owner_value_id=None,
    )
    try:
        assert CorrelationTable._source_private_dynamic_sequence_findings(
            module,
        )
    finally:
        descriptor_page.cells[descriptor_key] = descriptor


def test_two_private_helper_admissions_keep_distinct_append_origins():
    source = ("def first(data: bytes) -> int:\n"
              "    header = bytearray()\n"
              "    header.append(2)\n"
              "    for _ in range(1):\n"
              "        if not data:\n"
              "            return len(header)\n"
              "    return len(header) + len(data)\n\n"
              "def second(data: bytes) -> int:\n"
              "    tail = bytearray()\n"
              "    tail.append(3)\n"
              "    for _ in range(1):\n"
              "        if not data:\n"
              "            return len(tail)\n"
              "    return len(tail) + len(data)\n\n"
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
        source, "root", name="two_private_scalar_helpers",
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = module.metadata["identity_book"]
    page = book.page(SOURCE_PRIVATE_DYNAMIC_SEQUENCE)
    rows = tuple(page.rows())
    assert len(rows) == 2
    assert CorrelationTable.build(module).findings(module) == []
    left_ref = book.latest_ref(SOURCE_PRIVATE_DYNAMIC_SEQUENCE, rows[0])
    right_ref = book.latest_ref(SOURCE_PRIVATE_DYNAMIC_SEQUENCE, rows[1])
    left = page.cell(left_ref.row, left_ref.column)
    right = page.cell(right_ref.row, right_ref.column)
    key = (left_ref.row, left_ref.column)
    page.cells[key] = replace(left, mutation_cell=right.mutation_cell)
    try:
        findings = CorrelationTable._source_private_dynamic_sequence_findings(
            module,
        )
        assert any(finding.value_id == left_ref.row[1] for finding in findings)
    finally:
        page.cells[key] = left

    # Both helpers mint sequence ID 8. A descriptor from the other helper
    # cannot discharge a missing direct admission cause for the first one.
    descriptor_page = book.page(SEQUENCE_DESCRIPTOR)
    descriptor_refs = {
        receipt: next(
            ref
            for descriptor_row in descriptor_page.rows()
            for ref in (book.latest_ref(SEQUENCE_DESCRIPTOR, descriptor_row),)
            if receipt in {cause for cause, _ in book.edges_into(ref)}
        )
        for receipt in (left_ref, right_ref)
    }
    a_descriptor = descriptor_refs[left_ref]
    b_descriptor = descriptor_refs[right_ref]
    assert a_descriptor.row[-1] == b_descriptor.row[-1]
    assert a_descriptor.row[0] != b_descriptor.row[0]
    edge_page = book.page(EDGE_PAGE)
    a_edge = next(
        (a_descriptor.key, cause.key, stage.name)
        for cause, stage in book.edges_into(a_descriptor)
        if cause == left_ref
    )
    b_stage = next(
        stage.name
        for cause, stage in book.edges_into(b_descriptor)
        if cause == right_ref
    )
    forged_b_edge = (b_descriptor.key, left_ref.key, b_stage)
    assert forged_b_edge not in edge_page.row_columns
    a_edge_cell = edge_page.cells.pop((a_edge, 0))
    a_edge_stamp = edge_page.stamps.pop((a_edge, 0))
    edge_page.row_columns.pop(a_edge)
    edge_page.scopes[a_descriptor.key].pop(a_edge)
    edge_page._stamp(forged_b_edge, 0, True)
    try:
        findings = CorrelationTable._source_private_dynamic_sequence_findings(
            module,
        )
        assert any(
            finding.function == str(left_ref.row[0])
            and finding.value_id == left_ref.row[1]
            for finding in findings
        )
    finally:
        edge_page.cells.pop((forged_b_edge, 0))
        edge_page.stamps.pop((forged_b_edge, 0))
        edge_page.row_columns.pop(forged_b_edge)
        edge_page.scopes[b_descriptor.key].pop(forged_b_edge)
        edge_page.cells[(a_edge, 0)] = a_edge_cell
        edge_page.stamps[(a_edge, 0)] = a_edge_stamp
        edge_page.row_columns[a_edge] = [0]
        edge_page.scopes[a_descriptor.key][a_edge] = None
