"""Authored byte concatenation keeps helper storage identities physical."""

from pathlib import Path

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


def test_dynamic_append_slice_uses_its_pointer_storage_formals():
    source = """def compress_tail(header: bytes, encoded_length: bytes, coded_commands: bytes, coded_literals: bytes) -> bytes:
    return header + encoded_length + coded_commands + coded_literals
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "compress_tail", "parameter": parameter,
             "storage": "span", "dtype": "int64", "rank": 1,
             "python_type": "builtins.bytes"}
            for parameter in (
                "header", "encoded_length", "coded_commands", "coded_literals"
            )
        ],
        "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        source, "compress_tail", extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    helpers = [
        function for function in module.functions.values()
        if function.metadata.get("ssa_sequence_operation") == "append_slice"
        and "__reserve" not in function.name
    ]
    assert len(helpers) >= 3
    pointer_bases = 0
    for function in helpers:
        formals = {int(value.id): value for value in function.args}
        column_ids = {
            int(column_id)
            for descriptor in module.sequence_tables[function.name].sequences.values()
            for column_id in descriptor.column_value_ids
        }
        bases = [
            instruction.args[0]
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op == "GetElementPtr"
            and instruction.args
            and int(instruction.args[0].id) in column_ids
        ]
        assert len(bases) == 2
        assert all(
            base.dtype == formals[int(base.id)].dtype
            and base.accounting == formals[int(base.id)].accounting
            for base in bases
        ), [(base.id, base.dtype, formals[int(base.id)].dtype)
             for base in bases]
        pointer_bases += sum(base.dtype == "ptr" for base in bases)
    assert pointer_bases > 0
