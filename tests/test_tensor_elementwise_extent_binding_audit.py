"""A dynamic binary result retains the exact transported tensor geometry."""

from dataclasses import replace

from src.compiler.concordance_declarations import (
    SSA_VALUE, TENSOR_ELEMENTWISE_EXTENT_BINDING,
)
from src.compiler.identity_concordance import CorrelationTable, identity_book
from tests.test_runtime_reshape_validity_audit import _runtime_reshape_module


def test_dynamic_binary_result_requires_its_proved_extent_and_physical_span():
    module = _runtime_reshape_module()
    book = identity_book(module)
    page = book.page(TENSOR_ELEMENTWISE_EXTENT_BINDING)
    row, = tuple(page.rows())
    function = module.functions[row[1]]
    descriptor = module.tensor_tables[function.name].by_id(row[2])
    assert not CorrelationTable._tensor_elementwise_extent_binding_findings(module)

    module.tensor_tables[function.name].tensors[row[2]] = replace(
        descriptor, element_count_value_id=descriptor.shape_value_id,
    )
    try:
        assert CorrelationTable._tensor_elementwise_extent_binding_findings(module)
    finally:
        module.tensor_tables[function.name].tensors[row[2]] = descriptor

    ssa_ref = book.latest_ref(SSA_VALUE, (row[0], row[2]))
    ssa_page = book.page(SSA_VALUE)
    old_fact = ssa_page.cells[(ssa_ref.row, ssa_ref.column)]
    ssa_page.cells[(ssa_ref.row, ssa_ref.column)] = replace(
        old_fact, storage="scalar", rank=0,
    )
    try:
        findings = CorrelationTable._tensor_elementwise_extent_binding_findings(module)
        assert any(item.kind == "tensor-elementwise-result-geometry"
                   for item in findings)
    finally:
        ssa_page.cells[(ssa_ref.row, ssa_ref.column)] = old_fact


def test_dynamic_binary_extent_rejects_wrong_live_operand():
    module = _runtime_reshape_module()
    book = identity_book(module)
    page = book.page(TENSOR_ELEMENTWISE_EXTENT_BINDING)
    row, = tuple(page.rows())
    function = module.functions[row[1]]
    instruction = next(instruction for block in function.blocks.values()
                       for instruction in block.instrs
                       if instruction.res is not None
                       and int(instruction.res.id) == row[2])
    original = instruction.args[1]
    instruction.args[1] = instruction.args[0]
    try:
        assert any(item.kind == "tensor-elementwise-extent-binding"
                   for item in CorrelationTable._tensor_elementwise_extent_binding_findings(module))
    finally:
        instruction.args[1] = original
