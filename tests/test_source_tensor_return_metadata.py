"""Computed source return geometry reaches a different caller data identity."""

from pathlib import Path

from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book


def test_computed_source_return_projects_runtime_geometry_to_caller():
    source = """\
def make(count: int) -> AbstractTensor:
    return AbstractTensor.arange(count).to_dtype('int64')

def root(count: int) -> int:
    values = make(count)
    return (values + 1).sum().item()
"""
    contract = ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml").resolve()
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "count", "storage": "scalar",
            "dtype": "int64", "rank": 0, "python_type": "builtins.int",
        }],
        "records": {}, "bindings": [],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="source_tensor_return_metadata",
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        extraction_contract=contract, progress_bars=False,
    )
    assert not CorrelationTable.build(module).findings(module)

    root = module.functions["source_tensor_return_metadata__root"]
    source_call = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("source_linked")
    )
    callee_name = source_call.attributes["callee"]
    callee = module.functions[callee_name]
    callee_data = int(source_call.attributes["callee_output_ids"][0])
    caller_data = int(source_call.attributes["output_ids"][0])
    assert callee_data != caller_data
    callee_return = next(
        instruction
        for block in callee.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )
    assert tuple(int(value.id) for value in callee_return.args) == tuple(
        int(value_id) for value_id in source_call.attributes["callee_output_ids"]
    )
    assert len(callee_return.args) == 4
    caller_descriptor = next(
        descriptor
        for table_name, table in module.tensor_tables.items()
        if table_name.startswith("source_tensor_return_metadata__root__planned_region_")
        for descriptor in table.tensors.values()
        if descriptor.metadata_state == "dynamic"
        and descriptor.element_count_value_id is not None
    )
    assert caller_descriptor.metadata_state == "dynamic"
    book = identity_book(module)
    result_page = book.page("tensor_metadata_result_binding")
    assert any(
        row[2] == callee_name
        and int(result_page.latest(row)[0]) == callee_data
        and tuple(map(int, result_page.latest(row)[2]))
            == tuple(map(int, source_call.attributes["output_ids"][1:]))
        for row in result_page.rows()
    )
    # The source Call has one callee result identity and a different caller
    # projection identity. Changing only that live projection must invalidate
    # the booked cross-frame relation.
    output_ids = source_call.attributes["output_ids"]
    source_call.attributes["output_ids"] = (
        max(int(value.id) for value in root.args) + 1,
        *output_ids[1:],
    )
    try:
        assert CorrelationTable._tensor_metadata_result_binding_findings(module)
    finally:
        source_call.attributes["output_ids"] = output_ids
