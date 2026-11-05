"""A dead scalar reduction helper retains its typed literal proof."""

from pathlib import Path

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SSA_CONSTANT_MATERIALIZATION, TENSOR_REDUCTION_DOMAIN,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, EDGE_PAGE, identity_book,
)
from src.compiler.ssa_record_return_state import function_scope_of


@pytest.fixture(scope="module")
def scalar_product():
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        "def root(values: AbstractTensor) -> int:\n"
        "    return values.sum(dim=0).item()\n",
        "root", name="runtime_axis_int64_scalar_audit",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    domain_page = book.page(TENSOR_REDUCTION_DOMAIN)
    domain_row, = (
        row for row in domain_page.rows()
        if (fact := domain_page.latest(row))
        and fact[0] == "axis-runtime"
    )
    domain = domain_page.latest(domain_row)
    function = module.functions[domain_row[0]]
    materializations = book.page(SSA_CONSTANT_MATERIALIZATION)
    receipts = {}
    for value_id, literal in zip(domain[8:11], ((), 0, 1), strict=True):
        row, = (
            row for row in materializations.scope_rows(function_scope_of(function))
            if row[1] == function.name and row[3] == value_id
        )
        receipts[literal] = book.latest_ref(SSA_CONSTANT_MATERIALIZATION, row)
        assert materializations.latest(row) == ("Const", "int32", repr(literal))
    assert CorrelationTable._constant_materialization_findings(module) == []
    assert CorrelationTable._tensor_reduction_domain_findings(module) == []
    return module, book, domain_row, function, receipts


def test_scalar_axis_int64_keeps_dead_explicit_geometry_literals(scalar_product):
    module, _, domain_row, function, _ = scalar_product
    ids = set(identity_book(module).page(TENSOR_REDUCTION_DOMAIN).latest(
        domain_row)[8:11])
    assert not any(
        instruction.res is not None and instruction.res.id in ids
        for block in function.blocks.values() for instruction in block.instrs
    )
    assert CorrelationTable._constant_materialization_findings(module) == []
    assert CorrelationTable._tensor_reduction_domain_findings(module) == []


@pytest.mark.parametrize("literal", ((), 0, 1))
def test_scalar_axis_int64_rejects_forged_literal(scalar_product, literal):
    module, book, _, _, receipts = scalar_product
    ref = receipts[literal]
    page = book.page(SSA_CONSTANT_MATERIALIZATION)
    original = page.cell(ref.row, ref.column)
    page.cells[(ref.row, ref.column)] = (
        "Const", "int32", repr((9,) if literal == () else 9),
    )
    try:
        assert CorrelationTable._constant_materialization_findings(module)
        assert CorrelationTable._tensor_reduction_domain_findings(module)
    finally:
        page.cells[(ref.row, ref.column)] = original


def test_scalar_axis_int64_rejects_forged_literal_dtype(scalar_product):
    module, book, _, _, receipts = scalar_product
    ref = receipts[1]
    page = book.page(SSA_CONSTANT_MATERIALIZATION)
    original = page.cell(ref.row, ref.column)
    page.cells[(ref.row, ref.column)] = ("Const", "int64", "1")
    try:
        assert CorrelationTable._constant_materialization_findings(module)
        assert CorrelationTable._tensor_reduction_domain_findings(module)
    finally:
        page.cells[(ref.row, ref.column)] = original


@pytest.mark.parametrize("cause_kind", ("domain", "formal", "value"))
def test_scalar_axis_int64_requires_literal_causes(
    scalar_product, cause_kind, monkeypatch,
):
    module, book, domain_row, function, receipts = scalar_product
    ref = receipts[1]
    domain = book.latest_ref(TENSOR_REDUCTION_DOMAIN, domain_row)
    causes = {cell for cell, _ in book.edges_into(ref)}
    if cause_kind == "domain":
        removed = domain
    elif cause_kind == "formal":
        selected = module.functions["sum_dim_int64"]
        removed, = (
            cell for cell in causes
            if cell.row == (
                function_scope_of(selected), selected.args[0].id,
            )
        )
    else:
        removed, = (
            cell for cell in causes
            if cell.row == (function_scope_of(function), ref.row[3])
        )
    assert removed in causes
    edges = book.page(EDGE_PAGE)
    original = edges.scope_rows
    monkeypatch.setattr(edges, "scope_rows", lambda scope: tuple(
        edge for edge in original(scope)
        if not (scope == ref.key and edge[1] == removed.key)
    ))
    assert CorrelationTable._constant_materialization_findings(module)
    assert CorrelationTable._tensor_reduction_domain_findings(module)
