"""The validated runtime reshape view has exact SSA and call provenance."""

from dataclasses import replace
from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SSA_VALUE, TENSOR_METADATA_RESULT_BINDING,
    TENSOR_RUNTIME_RESHAPE_VALIDITY,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, EDGE_PAGE, identity_book,
)
from src.compiler.ssa_record_return_state import function_scope_of


def _runtime_reshape_module():
    source = """def root(bits: AbstractTensor, width: int) -> AbstractTensor:
    shaped = bits.reshape(-1, width)
    return shaped + 1
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "bits", "storage": "span",
             "dtype": "int64", "rank": 2,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "width", "storage": "scalar",
             "dtype": "int64", "rank": 0, "shape": [],
             "python_type": "builtins.int"},
        ], "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="reshape_validity_audit",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    return module


def test_runtime_reshape_validity_rejects_wrong_geometry_and_guard():
    module = _runtime_reshape_module()
    book = identity_book(module)
    page = book.page(TENSOR_RUNTIME_RESHAPE_VALIDITY)
    row, = tuple(page.rows())
    fact = page.latest(row)
    function = module.functions[row[1]]
    descriptor = module.tensor_tables[function.name].by_id(row[2])
    assert not CorrelationTable._tensor_runtime_reshape_validity_findings(module)

    module.tensor_tables[function.name].tensors[row[2]] = replace(
        descriptor, shape_value_id=fact[3],
    )
    try:
        assert CorrelationTable._tensor_runtime_reshape_validity_findings(module)
    finally:
        module.tensor_tables[function.name].tensors[row[2]] = descriptor

    guard = next(instruction for block in function.blocks.values()
                 for instruction in block.instrs
                 if instruction.op == "CondBr" and instruction.args
                 and instruction.args[0].id == fact[9])
    original = guard.attributes["true_target"]
    guard.attributes["true_target"] = fact[11]
    try:
        assert CorrelationTable._tensor_runtime_reshape_validity_findings(module)
    finally:
        guard.attributes["true_target"] = original

    stores = tuple(instruction for block in function.blocks.values()
                   for instruction in block.instrs
                   if instruction.op == "Store"
                   and instruction.attributes.get("binding")
                   == "tensor-reshape-shape")
    assert len(stores) == 2
    original = stores[0].args[0]
    stores[0].args[0] = stores[1].args[0]
    try:
        assert CorrelationTable._tensor_runtime_reshape_validity_findings(module)
    finally:
        stores[0].args[0] = original


def test_runtime_reshape_metadata_result_cites_validated_view():
    module = _runtime_reshape_module()
    book = identity_book(module)
    validity_page = book.page(TENSOR_RUNTIME_RESHAPE_VALIDITY)
    row, = tuple(validity_page.rows())
    validity = book.latest_ref(TENSOR_RUNTIME_RESHAPE_VALIDITY, row)
    result_page = book.page(TENSOR_METADATA_RESULT_BINDING)
    bindings = tuple(binding for binding in result_page.rows()
                     if binding[2:] == (row[1], row[2]))
    assert len(bindings) == 1
    binding = bindings[0]
    ref = book.latest_ref(TENSOR_METADATA_RESULT_BINDING, binding)
    assert validity in {source for source, _stage in book.edges_into(ref)}
    assert not CorrelationTable._tensor_runtime_reshape_validity_findings(module)

    original = result_page.cells[(ref.row, ref.column)]
    result_page.cells[(ref.row, ref.column)] = (
        original[0], (original[1][1], original[1][0], original[1][2]),
        original[2],
    )
    try:
        assert CorrelationTable._tensor_runtime_reshape_validity_findings(module)
    finally:
        result_page.cells[(ref.row, ref.column)] = original

    edge_page = book.page(EDGE_PAGE)
    edge_row = next((ref.key, source.key, stage.name)
                    for source, stage in book.edges_into(ref)
                    if source == validity)
    edge_fact = edge_page.cells.pop((edge_row, 0))
    edge_stamp = edge_page.stamps.pop((edge_row, 0))
    edge_page.row_columns.pop(edge_row)
    edge_page.scopes[ref.key].pop(edge_row)
    try:
        assert CorrelationTable._tensor_runtime_reshape_validity_findings(module)
    finally:
        edge_page.cells[(edge_row, 0)] = edge_fact
        edge_page.stamps[(edge_row, 0)] = edge_stamp
        edge_page.row_columns[edge_row] = [0]
        edge_page.scopes[ref.key][edge_row] = None

    binding_cells = {key: value for key, value in result_page.cells.items()
                     if key[0] == binding}
    binding_stamps = {key: value for key, value in result_page.stamps.items()
                      if key[0] == binding}
    columns = result_page.row_columns.pop(binding)
    for key in binding_cells:
        result_page.cells.pop(key)
        result_page.stamps.pop(key)
    result_page._reindex()
    try:
        assert CorrelationTable._tensor_runtime_reshape_validity_findings(module)
    finally:
        result_page.cells.update(binding_cells)
        result_page.stamps.update(binding_stamps)
        result_page.row_columns[binding] = columns
        result_page._reindex()


def test_runtime_reshape_physical_view_cites_exact_source_and_receipt():
    module = _runtime_reshape_module()
    book = identity_book(module)
    validity_page = book.page(TENSOR_RUNTIME_RESHAPE_VALIDITY)
    row, = tuple(validity_page.rows())
    function = module.functions[row[1]]
    scope = function_scope_of(function)
    source_ref = book.latest_ref(SSA_VALUE, (scope, validity_page.latest(row)[0]))
    result_ref = book.latest_ref(SSA_VALUE, (scope, row[2]))
    validity_ref = book.latest_ref(TENSOR_RUNTIME_RESHAPE_VALIDITY, row)
    assert source_ref is not None and result_ref is not None
    assert validity_ref is not None
    source_fact = book.page(SSA_VALUE).cell(source_ref.row, source_ref.column)
    result_fact = book.page(SSA_VALUE).cell(result_ref.row, result_ref.column)
    assert (source_fact.physical_dtype, source_fact.storage) == (
        "int64", "span",
    )
    assert (result_fact.physical_dtype, result_fact.storage, result_fact.rank) == (
        "int64", "span", 2,
    )
    assert {source_ref, validity_ref} <= {
        cause for cause, _ in book.edges_into(result_ref)
    }
    assert not CorrelationTable._tensor_runtime_reshape_validity_findings(module)

    value_page = book.page(SSA_VALUE)
    value_page.cells[(result_ref.row, result_ref.column)] = replace(
        result_fact, physical_dtype=None,
    )
    try:
        assert CorrelationTable._tensor_runtime_reshape_validity_findings(module)
    finally:
        value_page.cells[(result_ref.row, result_ref.column)] = result_fact

    edge_page = book.page(EDGE_PAGE)
    edge_row = next(
        (result_ref.key, source.key, stage.name)
        for source, stage in book.edges_into(result_ref)
        if source == validity_ref
    )
    edge_fact = edge_page.cells.pop((edge_row, 0))
    edge_stamp = edge_page.stamps.pop((edge_row, 0))
    edge_page.row_columns.pop(edge_row)
    edge_page.scopes[result_ref.key].pop(edge_row)
    try:
        assert CorrelationTable._tensor_runtime_reshape_validity_findings(module)
    finally:
        edge_page.cells[(edge_row, 0)] = edge_fact
        edge_page.stamps[(edge_row, 0)] = edge_stamp
        edge_page.row_columns[edge_row] = [0]
        edge_page.scopes[result_ref.key][edge_row] = None
