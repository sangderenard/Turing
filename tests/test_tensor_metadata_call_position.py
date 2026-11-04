"""Computed metadata binds to a stable Call result across insertions."""

from pathlib import Path
from dataclasses import replace

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import TENSOR_METADATA_CALL_BINDING
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable


def test_metadata_actual_binding_tracks_call_result_across_insertions():
    source = """def root(lengths: AbstractTensor, bit_count: int) -> AbstractTensor:
    return AbstractTensor.arange(bit_count, cls=type(lengths)).to_dtype("float64").sum().item()
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "lengths", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "bit_count", "storage": "scalar",
             "dtype": "int64", "rank": 0, "shape": [],
             "python_type": "builtins.int"},
        ],
        "records": {}, "bindings": [],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = module.metadata["identity_book"]
    rows = tuple(book.page(TENSOR_METADATA_CALL_BINDING).rows())
    assert len(rows) == 2
    assert CorrelationTable._tensor_metadata_call_binding_findings(module) == []
    row = rows[0]
    block = module.functions[row[1]].blocks[row[2]]
    calls = [instruction for instruction in block.instrs
             if instruction.op == "Call" and instruction.res is not None
             and int(instruction.res.id) == row[3]]
    assert len(calls) == 1
    call = calls[0]
    original_index = block.instrs.index(call)
    inserted = next(instruction for instruction in block.instrs
                    if instruction.op == "GetElementPtr")
    block.instrs.insert(original_index, inserted)
    try:
        assert CorrelationTable._tensor_metadata_call_binding_findings(module) == []
    finally:
        block.instrs.pop(original_index)

    original_result = call.res
    call.res = replace(original_result, id=int(original_result.id) + 10**9)
    try:
        assert CorrelationTable._tensor_metadata_call_binding_findings(module)
    finally:
        call.res = original_result
