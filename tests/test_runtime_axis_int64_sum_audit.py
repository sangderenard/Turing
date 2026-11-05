"""The selected runtime axis sum keeps its int64 storage proof."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SSA_VALUE, TENSOR_OWNED_HEAP_BINDING, TENSOR_REDUCTION_DOMAIN,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, EDGE_PAGE, identity_book,
)
from src.compiler.ssa_record_return_state import (
    function_scope_of, ssa_value_identity_cell,
)


@pytest.fixture(scope="module")
def axis_product():
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "int64", "rank": 2,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = (
        "def root(values: AbstractTensor) -> int:\n"
        "    columns = values.sum(dim=0)\n"
        "    return columns.sum().item()\n"
    )
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="runtime_axis_int64_sum_audit",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    page = book.page(TENSOR_REDUCTION_DOMAIN)
    row, = (row for row in page.rows()
            if page.latest(row)[:3] == ("axis-runtime", 0, "sum"))
    function = module.functions[row[0]]
    call, = (instruction for block in function.blocks.values()
             for instruction in block.instrs if instruction.op == "Call"
             and instruction.res is not None
             and instruction.res.id == row[1]
             and instruction.attributes.get("callee") == "sum_dim_int64")
    assert CorrelationTable._tensor_reduction_domain_findings(module) == []
    return module, book, row, function, call


def test_runtime_axis_int64_sum_accepts_typed_span(axis_product):
    module, _, _, _, call = axis_product
    assert len(call.args) == 5
    assert CorrelationTable._tensor_reduction_domain_findings(module) == []


@pytest.mark.parametrize("value_kind", ("source", "result"))
def test_runtime_axis_int64_sum_rejects_wrong_physical_value(
    axis_product, value_kind,
):
    module, book, row, function, _ = axis_product
    fact = book.page(TENSOR_REDUCTION_DOMAIN).latest(row)
    value_id = fact[1] if value_kind == "source" else row[1]
    ref = ssa_value_identity_cell(function, value_id, book=book)
    page = book.page(SSA_VALUE)
    original = page.cell(ref.row, ref.column)
    page.cells[(ref.row, ref.column)] = replace(
        original, physical_dtype="float64",
    )
    try:
        assert CorrelationTable._tensor_reduction_domain_findings(module)
    finally:
        page.cells[(ref.row, ref.column)] = original


def test_runtime_axis_int64_sum_rejects_wrong_result_descriptor(axis_product):
    module, _, row, function, _ = axis_product
    table = module.tensor_tables[function.name]
    original = table.by_id(row[1])
    table.tensors[row[1]] = replace(original, dtype="float64")
    try:
        assert CorrelationTable._tensor_reduction_domain_findings(module)
    finally:
        table.tensors[row[1]] = original


@pytest.mark.parametrize("field,replacement", ((0, -1), (2, -1), (3, "float64")))
def test_runtime_axis_int64_sum_rejects_wrong_owner(
    axis_product, field, replacement,
):
    module, book, row, function, _ = axis_product
    owner_row = (function_scope_of(function), function.name, row[1])
    owner_ref = book.latest_ref(TENSOR_OWNED_HEAP_BINDING, owner_row)
    page = book.page(TENSOR_OWNED_HEAP_BINDING)
    original = page.cell(owner_ref.row, owner_ref.column)
    changed = list(original)
    changed[field] = replacement
    page.cells[(owner_ref.row, owner_ref.column)] = tuple(changed)
    try:
        assert CorrelationTable._tensor_reduction_domain_findings(module)
    finally:
        page.cells[(owner_ref.row, owner_ref.column)] = original


def test_runtime_axis_int64_sum_rejects_wrong_count(axis_product):
    module, book, row, _, _ = axis_product
    page = book.page(TENSOR_REDUCTION_DOMAIN)
    ref = book.latest_ref(TENSOR_REDUCTION_DOMAIN, row)
    original = page.cell(ref.row, ref.column)
    changed = list(original)
    changed[10] = -1
    page.cells[(ref.row, ref.column)] = tuple(changed)
    try:
        assert CorrelationTable._tensor_reduction_domain_findings(module)
    finally:
        page.cells[(ref.row, ref.column)] = original


def test_runtime_axis_int64_sum_rejects_wrong_callee(axis_product):
    module, _, _, _, call = axis_product
    original = call.attributes["callee"]
    call.attributes["callee"] = "reduce_dim_double"
    try:
        assert CorrelationTable._tensor_reduction_domain_findings(module)
    finally:
        call.attributes["callee"] = original


def test_runtime_axis_int64_sum_rejects_wrong_output_port(axis_product):
    module, _, _, _, call = axis_product
    original = call.attributes["ssa_output_argument"]
    call.attributes["ssa_output_argument"] = 0
    try:
        assert CorrelationTable._tensor_reduction_domain_findings(module)
    finally:
        call.attributes["ssa_output_argument"] = original


@pytest.mark.parametrize("formal_position", (0, 1))
def test_runtime_axis_int64_sum_requires_selected_formal_cause(
    axis_product, formal_position, monkeypatch,
):
    module, book, row, function, _ = axis_product
    result = ssa_value_identity_cell(function, row[1], book=book)
    selected = module.functions["sum_dim_int64"]
    formal = ssa_value_identity_cell(
        selected, selected.args[formal_position].id, book=book,
    )
    assert formal in {cell for cell, _ in book.edges_into(result)}
    edges = book.page(EDGE_PAGE)
    original = edges.scope_rows
    monkeypatch.setattr(edges, "scope_rows", lambda scope: tuple(
        edge for edge in original(scope)
        if not (scope == result.key and edge[1] == formal.key)
    ))
    assert CorrelationTable._tensor_reduction_domain_findings(module)


@pytest.mark.parametrize("formal_position", (0, 1))
def test_runtime_axis_int64_sum_rejects_wrong_typed_formal(
    axis_product, formal_position,
):
    module, book, _, _, _ = axis_product
    selected = module.functions["sum_dim_int64"]
    formal = ssa_value_identity_cell(
        selected, selected.args[formal_position].id, book=book,
    )
    page = book.page(SSA_VALUE)
    original = page.cell(formal.row, formal.column)
    page.cells[(formal.row, formal.column)] = replace(
        original, physical_dtype="float64",
    )
    try:
        assert CorrelationTable._tensor_reduction_domain_findings(module)
    finally:
        page.cells[(formal.row, formal.column)] = original
