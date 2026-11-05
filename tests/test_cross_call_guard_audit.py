"""A projected guard retains its exact tuple and call-target proof."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.concordance_declarations import (
    FUNCTION_ADDRESS, GUARDED_CALL_ACTUAL_RANGE, PROCESS_GRAPH_OPERANDS,
    SOURCE_AGGREGATE_LAYOUT, SOURCE_FUNCTION_REACHABILITY, SOURCE_LITERAL_VALUE,
    SOURCE_INTRINSIC_ADMISSION,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, EDGE_PAGE, identity_book


@pytest.fixture(scope="module")
def product():
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": [
        {"function": "root", "parameter": "values", "storage": "span",
         "dtype": "int64", "rank": 1, "shape": [3],
         "python_type": "src.common.tensors.abstraction.AbstractTensor"},
        {"function": "root", "parameter": "k", "storage": "scalar",
         "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
    ]})
    module, _, _ = lower_ast_source_to_ssa(
        "def bounded(k: int):\n"
        "    if k < 1:\n"
        "        raise ValueError('small')\n"
        "    return (0, 0, k)\n"
        "def root(values, k: int):\n"
        "    selected = bounded(k)[2]\n"
        "    if selected > 3:\n"
        "        raise ValueError('large')\n"
        "    return AbstractTensor.topk(values, k=selected, dim=0).indices\n",
        "root", name="cross_call_guard_audit", extraction_contract=policy,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    row, = (candidate for candidate in book.page(GUARDED_CALL_ACTUAL_RANGE).rows()
            if any(source.page == SOURCE_INTRINSIC_ADMISSION
                   for source, _ in book.edges_into(
                       book.latest_ref(GUARDED_CALL_ACTUAL_RANGE, candidate)))
            and book.page(GUARDED_CALL_ACTUAL_RANGE).latest(candidate) == (1, 3))
    ref = book.latest_ref(GUARDED_CALL_ACTUAL_RANGE, row)
    assert book.page(GUARDED_CALL_ACTUAL_RANGE).latest(row) == (1, 3)
    assert CorrelationTable.build(module).findings(module) == []
    return module, book, ref


def _rejected(module):
    return any(f.kind == "intrinsic-guarded-actual-origin"
               for f in CorrelationTable._operator_safety_findings(module))


@pytest.mark.parametrize("page", (SOURCE_FUNCTION_REACHABILITY, FUNCTION_ADDRESS,
                                  SOURCE_AGGREGATE_LAYOUT, SOURCE_LITERAL_VALUE))
def test_projected_guard_requires_each_authority(product, monkeypatch, page):
    module, book, ref = product
    removed = {cell for cell, _ in book.edges_into(ref) if cell.page == page}
    assert removed
    edges = book.page(EDGE_PAGE)
    original = edges.scope_rows
    monkeypatch.setattr(edges, "scope_rows", lambda scope: tuple(
        edge for edge in original(scope)
        if not (scope == ref.key and edge[1] in {cell.key for cell in removed})
    ))
    assert _rejected(module)


def test_projected_guard_rejects_wrong_tuple_member(product):
    module, book, ref = product
    layout, = (cell for cell, _ in book.edges_into(ref)
               if cell.page == SOURCE_AGGREGATE_LAYOUT)
    page = book.page(SOURCE_AGGREGATE_LAYOUT)
    original = page.cell(layout.row, layout.column)
    members = (*original.members[:2], original.members[0])
    page.cells[(layout.row, layout.column)] = replace(original, members=members)
    try:
        assert _rejected(module)
    finally:
        page.cells[(layout.row, layout.column)] = original


def test_projected_guard_rejects_unreachable_callee(product):
    module, book, ref = product
    reach, = (cell for cell, _ in book.edges_into(ref)
              if cell.page == SOURCE_FUNCTION_REACHABILITY)
    page = book.page(SOURCE_FUNCTION_REACHABILITY)
    original = page.cell(reach.row, reach.column)
    page.cells[(reach.row, reach.column)] = False
    try:
        assert _rejected(module)
    finally:
        page.cells[(reach.row, reach.column)] = original


def test_projected_guard_rejects_wrong_index_operand(product):
    module, book, ref = product
    projection, = (cell for cell, _ in book.edges_into(ref)
                   if cell.page == PROCESS_GRAPH_OPERANDS and cell.row[1] != ref.row[1])
    page = book.page(PROCESS_GRAPH_OPERANDS)
    original = page.cell(projection.row, projection.column)
    base = next(cell for cell, role in original if role == "base")
    page.cells[(projection.row, projection.column)] = ((base, "base"), (base, "index"))
    try:
        assert _rejected(module)
    finally:
        page.cells[(projection.row, projection.column)] = original
