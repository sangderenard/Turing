"""A dynamic binary result retains the exact transported tensor geometry."""

from dataclasses import replace
from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

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


def test_dynamic_scalar_kernel_extent_rejects_pointer_kernel_or_wrong_reverse():
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "data", "storage": "span",
                    "dtype": "float64", "rank": 1,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
        "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        "def root(data: AbstractTensor, multiplier: float) -> float:\n"
        "    return (data * multiplier).sum().item()\n",
        "root", name="scalar_kernel_extent_audit",
        extraction_contract=contract, python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert not CorrelationTable._tensor_elementwise_extent_binding_findings(module)
    page = identity_book(module).page(TENSOR_ELEMENTWISE_EXTENT_BINDING)
    row, = tuple(page.rows())
    function = module.functions[row[1]]
    call = next(instruction for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None and instruction.res.id == row[2])
    assert call.attributes["callee"] == "binary_scalar_double"
    call.attributes["callee"] = "binary_double"
    try:
        assert any(finding.kind == "tensor-elementwise-extent-binding"
                   for finding in CorrelationTable._tensor_elementwise_extent_binding_findings(module))
    finally:
        call.attributes["callee"] = "binary_scalar_double"
    reverse = next(instruction for block in function.blocks.values()
                   for instruction in block.instrs
                   if instruction.res is not None
                   and instruction.res.id == call.args[5].id)
    old = reverse.attributes["constant"]
    reverse.attributes["constant"] = 1 - old
    try:
        assert any(finding.kind == "tensor-elementwise-extent-binding"
                   for finding in CorrelationTable._tensor_elementwise_extent_binding_findings(module))
    finally:
        reverse.attributes["constant"] = old
