"""A scatter values conversion must cover every runtime selector update."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    KERNEL_INPUT_CONVERSION, SSA_VALUE, TENSOR_OWNED_HEAP_BINDING,
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
def scatter_product():
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "octets", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = """def root(octets: AbstractTensor):
    counts = AbstractTensor.scatter(
        AbstractTensor.zeros((256,), cls=type(octets)),
        octets, octets + 1, dim=0,
    )
    return counts.sum().item()
"""
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="scatter_counted_conversion_audit",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    page = book.page(KERNEL_INPUT_CONVERSION)
    row, = (row for row in page.rows()
            if page.latest(row)[4] == "scatter_sum_double")
    function, = (function for function in module.functions.values()
                 if function_scope_of(function) == row[0]
                 and any(instruction.op == "Call"
                         and instruction.attributes.get("callee") == "scatter_sum_double"
                         and instruction.res is not None
                         and instruction.res.id == row[2]
                         for block in function.blocks.values()
                         for instruction in block.instrs))
    calls = tuple(instruction for block in function.blocks.values()
                  for instruction in block.instrs if instruction.op == "Call")
    cast, = (call for call in calls if call.attributes.get("callee")
             == "cast_int64_to_double_values")
    scatter, = (call for call in calls if call.attributes.get("callee")
                == "scatter_sum_double")
    assert CorrelationTable._physical_call_input_conversion_findings(module) == []
    return module, book, row, function, cast, scatter


def test_counted_scatter_conversion_accepts_exact_span(scatter_product):
    module, _, _, _, cast, scatter = scatter_product
    assert cast.args[2].id == scatter.args[6].id
    assert CorrelationTable._physical_call_input_conversion_findings(module) == []


@pytest.mark.parametrize("field,replacement", ((3, "int64"), (4, "binary_double")))
def test_counted_scatter_conversion_rejects_forged_receipt(
    scatter_product, field, replacement,
):
    module, book, row, _, _, _ = scatter_product
    receipt = book.latest_ref(KERNEL_INPUT_CONVERSION, row)
    page = book.page(KERNEL_INPUT_CONVERSION)
    original = page.cell(receipt.row, receipt.column)
    changed = list(original)
    changed[field] = replacement
    page.cells[(receipt.row, receipt.column)] = tuple(changed)
    try:
        assert CorrelationTable._physical_call_input_conversion_findings(module)
    finally:
        page.cells[(receipt.row, receipt.column)] = original


def test_counted_scatter_conversion_rejects_missing_receipt(
    scatter_product, monkeypatch,
):
    module, book, row, _, _, _ = scatter_product
    page = book.page(KERNEL_INPUT_CONVERSION)
    original = page.latest
    monkeypatch.setattr(
        page, "latest", lambda candidate, default=None: (
            None if candidate == row else original(candidate, default)
        ),
    )
    assert CorrelationTable._physical_call_input_conversion_findings(module)


@pytest.mark.parametrize("argument", ("cast", "scatter"))
def test_counted_scatter_conversion_rejects_wrong_count(
    scatter_product, argument,
):
    module, _, _, _, cast, scatter = scatter_product
    call, position = (cast, 2) if argument == "cast" else (scatter, 6)
    original = call.args[position]
    call.args[position] = scatter.args[4]  # axis is not the selector count
    try:
        assert CorrelationTable._physical_call_input_conversion_findings(module)
    finally:
        call.args[position] = original


def test_scalar_cast_cannot_replace_counted_conversion(scatter_product):
    module, _, _, _, cast, _ = scatter_product
    original_op, original_args = cast.op, cast.args
    cast.op, cast.args = "Cast", [cast.args[0]]
    try:
        assert CorrelationTable._physical_call_input_conversion_findings(module)
    finally:
        cast.op, cast.args = original_op, original_args


def test_counted_scatter_conversion_rejects_wrong_output_port(scatter_product):
    module, _, _, _, cast, _ = scatter_product
    original = cast.attributes["ssa_output_argument"]
    cast.attributes["ssa_output_argument"] = 0
    try:
        assert CorrelationTable._physical_call_input_conversion_findings(module)
    finally:
        cast.attributes["ssa_output_argument"] = original


def test_counted_scatter_conversion_rejects_missing_wrapper_forward(scatter_product):
    module, _, _, _, _, _ = scatter_product
    wrapper = module.functions["scatter_sum_double"]
    forward, = (instruction for block in wrapper.blocks.values()
                for instruction in block.instrs
                if instruction.op == "Call"
                and instruction.attributes.get("callee") == "scatter_reduce_double")
    original = forward.args[7]
    forward.args[7] = forward.args[0]
    try:
        assert CorrelationTable._physical_call_input_conversion_findings(module)
    finally:
        forward.args[7] = original


def test_counted_scatter_conversion_rejects_wrong_owned_carrier(scatter_product):
    module, book, row, function, cast, _ = scatter_product
    owner_row = (row[0], function.name, cast.res.id)
    owner_ref = book.latest_ref(TENSOR_OWNED_HEAP_BINDING, owner_row)
    page = book.page(TENSOR_OWNED_HEAP_BINDING)
    original = page.cell(owner_ref.row, owner_ref.column)
    page.cells[(owner_ref.row, owner_ref.column)] = (
        -1, *original[1:],
    )
    try:
        assert CorrelationTable._physical_call_input_conversion_findings(module)
    finally:
        page.cells[(owner_ref.row, owner_ref.column)] = original


def test_counted_scatter_conversion_rejects_scalar_converted_fact(scatter_product):
    module, book, _, function, cast, _ = scatter_product
    converted = ssa_value_identity_cell(function, cast.res.id, book=book)
    page = book.page(SSA_VALUE)
    original = page.cell(converted.row, converted.column)
    page.cells[(converted.row, converted.column)] = replace(
        original, storage="scalar", rank=0,
    )
    try:
        assert CorrelationTable._physical_call_input_conversion_findings(module)
    finally:
        page.cells[(converted.row, converted.column)] = original


@pytest.mark.parametrize("formal_position", (0, 1))
def test_counted_scatter_conversion_requires_typed_formal_cause(
    scatter_product, formal_position, monkeypatch,
):
    module, book, _, function, cast, _ = scatter_product
    converted = ssa_value_identity_cell(function, cast.res.id, book=book)
    selected = module.functions["cast_int64_to_double_values"]
    formal = ssa_value_identity_cell(
        selected, selected.args[formal_position].id, book=book,
    )
    assert formal in {cell for cell, _ in book.edges_into(converted)}
    edges = book.page(EDGE_PAGE)
    original = edges.scope_rows
    monkeypatch.setattr(edges, "scope_rows", lambda scope: tuple(
        edge for edge in original(scope)
        if not (scope == converted.key and edge[1] == formal.key)
    ))
    assert CorrelationTable._physical_call_input_conversion_findings(module)
