"""The live cumsum call and owned geometry must match the scan receipt."""

from pathlib import Path
from dataclasses import replace

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SSA_VALUE, TENSOR_RUNTIME_SCAN_GEOMETRY,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, EDGE_PAGE, identity_book,
)
from src.compiler.ssa_record_return_state import ssa_value_identity_cell


@pytest.fixture(scope="module", params=("float64", "int64"))
def scan_module(request):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": request.param, "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        "def root(values: AbstractTensor) -> float:\n"
        "    scanned = values.cumsum(dim=0)\n"
        "    return scanned.sum().item()\n",
        "root", name="dynamic_scan_audit",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    return module


def test_runtime_scan_has_exact_geometry_and_selected_call(scan_module):
    book = identity_book(scan_module)
    row, = book.page(TENSOR_RUNTIME_SCAN_GEOMETRY).rows()
    assert row[1].startswith("dynamic_scan_audit")
    assert CorrelationTable._tensor_runtime_scan_geometry_findings(scan_module) == []


@pytest.mark.parametrize("field,replacement", [
    (3, 1),  # wrong normalized axis
    (6, -1),  # wrong source/output count identity
    (7, -1),  # wrong owned output carrier
])
def test_runtime_scan_rejects_forged_geometry(
    scan_module, field, replacement,
):
    book = identity_book(scan_module)
    page = book.page(TENSOR_RUNTIME_SCAN_GEOMETRY)
    row, = page.rows()
    ref = book.latest_ref(TENSOR_RUNTIME_SCAN_GEOMETRY, row)
    original = page.cell(ref.row, ref.column)
    changed = list(original)
    changed[field] = replacement
    page.cells[(ref.row, ref.column)] = tuple(changed)
    try:
        assert CorrelationTable._tensor_runtime_scan_geometry_findings(scan_module)
    finally:
        page.cells[(ref.row, ref.column)] = original


def test_runtime_scan_rejects_wrong_selected_output_port(scan_module):
    book = identity_book(scan_module)
    row, = book.page(TENSOR_RUNTIME_SCAN_GEOMETRY).rows()
    function = scan_module.functions[row[1]]
    selected, = (
        instruction for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call" and instruction.res is not None
        and instruction.res.id == row[2]
    )
    original = selected.attributes["ssa_output_argument"]
    selected.attributes["ssa_output_argument"] = 0
    try:
        assert CorrelationTable._tensor_runtime_scan_geometry_findings(scan_module)
    finally:
        selected.attributes["ssa_output_argument"] = original


@pytest.mark.parametrize("formal_position", [0, 1])
def test_runtime_scan_requires_direct_selected_formal_cause(
    scan_module, formal_position,
):
    book = identity_book(scan_module)
    row, = book.page(TENSOR_RUNTIME_SCAN_GEOMETRY).rows()
    receipt = book.latest_ref(TENSOR_RUNTIME_SCAN_GEOMETRY, row)
    selected = scan_module.functions[
        book.page(TENSOR_RUNTIME_SCAN_GEOMETRY).latest(row)[8]
    ]
    formal = ssa_value_identity_cell(
        selected, int(selected.args[formal_position].id), book=book,
    )
    assert formal in {source for source, _ in book.edges_into(receipt)}
    edges = book.page(EDGE_PAGE)
    original = edges.scope_rows
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(edges, "scope_rows", lambda scope: tuple(
            edge for edge in original(scope)
            if not (scope == receipt.key and edge[1] == formal.key)
        ))
        assert CorrelationTable._tensor_runtime_scan_geometry_findings(scan_module)


@pytest.mark.parametrize("formal_position", [0, 1])
def test_runtime_scan_rejects_wrong_typed_kernel_formal(
    scan_module, formal_position,
):
    book = identity_book(scan_module)
    row, = book.page(TENSOR_RUNTIME_SCAN_GEOMETRY).rows()
    selected = scan_module.functions[
        book.page(TENSOR_RUNTIME_SCAN_GEOMETRY).latest(row)[8]
    ]
    formal = ssa_value_identity_cell(
        selected, selected.args[formal_position].id, book=book,
    )
    page = book.page(SSA_VALUE)
    original = page.cell(formal.row, formal.column)
    page.cells[(formal.row, formal.column)] = replace(
        original, physical_dtype=(
            "int64" if original.physical_dtype == "float64" else "float64"
        ),
    )
    try:
        assert CorrelationTable._tensor_runtime_scan_geometry_findings(scan_module)
    finally:
        page.cells[(formal.row, formal.column)] = original


@pytest.mark.parametrize("resident_dtype", ["int64", "ptr"])
def test_runtime_scan_resident_dtype_cannot_replace_pointer_origin(
    scan_module, resident_dtype,
):
    book = identity_book(scan_module)
    row, = book.page(TENSOR_RUNTIME_SCAN_GEOMETRY).rows()
    selected = scan_module.functions[
        book.page(TENSOR_RUNTIME_SCAN_GEOMETRY).latest(row)[8]
    ]
    formal = selected.args[0]
    original_dtype = formal.dtype
    symbol, origins = selected.metadata["llvm_repository_kernel"]
    # An ordinary scalar with the element dtype does not establish a pointer.
    formal.dtype = resident_dtype
    selected.metadata["llvm_repository_kernel"] = (
        symbol, {key: value for key, value in origins.items()
                 if key != int(formal.id)},
    )
    try:
        assert CorrelationTable._tensor_runtime_scan_geometry_findings(scan_module)
    finally:
        formal.dtype = original_dtype
        selected.metadata["llvm_repository_kernel"] = (symbol, origins)


@pytest.mark.parametrize("value_kind", ["source", "result"])
def test_runtime_scan_rejects_descriptor_with_wrong_element_type(
    scan_module, value_kind,
):
    book = identity_book(scan_module)
    row, = book.page(TENSOR_RUNTIME_SCAN_GEOMETRY).rows()
    record = book.page(TENSOR_RUNTIME_SCAN_GEOMETRY).latest(row)
    value_id = record[1] if value_kind == "source" else row[2]
    table = scan_module.tensor_tables[row[1]]
    original = table.by_id(value_id)
    table.tensors[value_id] = replace(
        original, dtype="int64" if original.dtype == "float64" else "float64",
    )
    try:
        assert CorrelationTable._tensor_runtime_scan_geometry_findings(scan_module)
    finally:
        table.tensors[value_id] = original
