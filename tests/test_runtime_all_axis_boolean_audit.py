"""The runtime Boolean reduction audit checks its live counted kernel proof."""

from pathlib import Path

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import TENSOR_REDUCTION_DOMAIN
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, EDGE_PAGE, identity_book
from src.compiler.ssa_record_return_state import ssa_value_identity_cell


@pytest.fixture(scope="module", params=("all", "any"))
def reduction(request):
    operation = request.param
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "float64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        "def root(values: AbstractTensor) -> bool:\n"
        f"    return values.{operation}().item()\n",
        "root", name=f"runtime_{operation}_axis_audit",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    page = book.page(TENSOR_REDUCTION_DOMAIN)
    rows = tuple(row for row in page.rows()
                 if page.latest(row)[0] == "all-axis-runtime")
    assert len(rows) == 1
    assert CorrelationTable._tensor_reduction_domain_findings(module) == []
    return module, book, rows[0], page.latest(rows[0])


def _instructions(function):
    return tuple(instruction for block in function.blocks.values()
                 for instruction in block.instrs)


def test_runtime_boolean_reduction_accepts_selected_domain(reduction):
    module, _, row, fact = reduction
    assert fact[2] in {"all", "any"}
    assert row[1] in {instruction.res.id for instruction in
                      _instructions(module.functions[row[0]])
                      if instruction.res is not None}


def test_runtime_boolean_reduction_rejects_wrong_count(reduction):
    module, book, row, fact = reduction
    page = book.page(TENSOR_REDUCTION_DOMAIN)
    receipt = book.latest_ref(TENSOR_REDUCTION_DOMAIN, row)
    changed = list(fact)
    changed[5] = -1
    page.cells[(receipt.row, receipt.column)] = tuple(changed)
    try:
        assert CorrelationTable._tensor_reduction_domain_findings(module)
    finally:
        page.cells[(receipt.row, receipt.column)] = fact


def test_runtime_boolean_reduction_rejects_wrong_flat_shape_store(reduction):
    module, _, row, fact = reduction
    function = module.functions[row[0]]
    store, = (instruction for instruction in _instructions(function)
              if instruction.op == "Store" and len(instruction.args) == 2
              and instruction.args[0].id == fact[5])
    original = store.args[0]
    source = next(argument for argument in function.args if argument.id == fact[1])
    store.args[0] = source
    try:
        assert CorrelationTable._tensor_reduction_domain_findings(module)
    finally:
        store.args[0] = original


@pytest.mark.parametrize("field", (7, 8, 9))
def test_runtime_boolean_reduction_rejects_wrong_rank_axis_or_opcode(
    reduction, field,
):
    module, _, row, fact = reduction
    function = module.functions[row[0]]
    constant, = (instruction for instruction in _instructions(function)
                 if instruction.op == "Const" and instruction.res is not None
                 and instruction.res.id == fact[field])
    original = dict(constant.attributes)
    constant.attributes.clear()
    constant.attributes["constant"] = 13
    try:
        assert CorrelationTable._tensor_reduction_domain_findings(module)
    finally:
        constant.attributes.clear()
        constant.attributes.update(original)


def test_runtime_boolean_reduction_rejects_wrong_output_port(reduction):
    module, _, row, _ = reduction
    function = module.functions[row[0]]
    call, = (instruction for instruction in _instructions(function)
             if instruction.op == "Call" and instruction.res is not None
             and instruction.res.id == row[1])
    original = call.attributes["ssa_output_argument"]
    call.attributes["ssa_output_argument"] = 0
    try:
        assert CorrelationTable._tensor_reduction_domain_findings(module)
    finally:
        call.attributes["ssa_output_argument"] = original


@pytest.mark.parametrize("formal_position", (0, 1))
def test_runtime_boolean_reduction_requires_selected_formal_cause(
    reduction, formal_position, monkeypatch,
):
    module, book, row, _ = reduction
    receipt = book.latest_ref(TENSOR_REDUCTION_DOMAIN, row)
    selected = module.functions["reduce_dim_double"]
    formal = ssa_value_identity_cell(
        selected, selected.args[formal_position].id, book=book,
    )
    assert formal in {source for source, _ in book.edges_into(receipt)}
    edges = book.page(EDGE_PAGE)
    original = edges.scope_rows
    monkeypatch.setattr(edges, "scope_rows", lambda scope: tuple(
        edge for edge in original(scope)
        if not (scope == receipt.key and edge[1] == formal.key)
    ))
    assert CorrelationTable._tensor_reduction_domain_findings(module)


def test_runtime_boolean_result_physical_type_requires_output_formal(reduction, monkeypatch):
    module, book, row, _ = reduction
    function = module.functions[row[0]]
    result = ssa_value_identity_cell(function, row[1], book=book)
    selected = module.functions["reduce_dim_double"]
    formal = ssa_value_identity_cell(selected, selected.args[1].id, book=book)
    assert formal in {source for source, _ in book.edges_into(result)}
    edges = book.page(EDGE_PAGE)
    original = edges.scope_rows
    monkeypatch.setattr(edges, "scope_rows", lambda scope: tuple(
        edge for edge in original(scope)
        if not (scope == result.key and edge[1] == formal.key)
    ))
    assert CorrelationTable._tensor_reduction_domain_findings(module)
