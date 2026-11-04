"""An owned bytes result uses source operands before storage aliasing."""

from pathlib import Path

import numpy as np

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, PLANNING_VALUE, PROCESS_GRAPH_OPERANDS,
    SOURCE_OWNED_SEQUENCE_RETURN,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_storage_requirements import public_owned_sequence_returns


def test_returned_bytes_views_keep_source_operand_owner_proof(tmp_path):
    source = (
        "def root(data: bytes) -> bytes:\n"
        "    header = bytearray()\n"
        "    header.append(2)\n"
        "    for _ in range(1):\n"
        "        header.append(3)\n"
        "    return bytes(header) + bytes(data)\n"
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "data",
                    "storage": "span", "dtype": "int64", "rank": 1,
                    "python_type": "builtins.bytes"}],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="owned_bytes_source_operands",
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    function_name = exports[0]
    book = module.metadata["identity_book"]
    (ports,) = public_owned_sequence_returns(module, function_name)
    fact = book.page(SOURCE_OWNED_SEQUENCE_RETURN).cell(
        ports.source_cell.row, ports.source_cell.column,
    )
    assert len(fact.operand_cells) == 1
    operands = book.page(PROCESS_GRAPH_OPERANDS).cell(
        fact.operand_cells[0].row, fact.operand_cells[0].column,
    )
    assert tuple(role for _, role in operands) == ("lhs", "rhs")
    # Both semantic operands have distinct resident storage aliases. Their
    # exact source classes, not the arenas' classes, prove the bytes result.
    resident_operands = []
    for operand, role in operands:
        alias = book.latest_ref(PLANNING_VALUE, (
            function_name, int(operand.row[1]),
        ))
        assert alias is not None
        resident_id = book.page(PLANNING_VALUE).cell(alias.row, alias.column)
        assert resident_id != operand.row[1]
        resident_operands.append((book.latest_ref(
            CANONICAL_VALUE, (operand.row[0], resident_id),
        ), role))
    assert CorrelationTable.build(module).findings(module) == []
    operand_page = book.page(PROCESS_GRAPH_OPERANDS)
    key = (fact.operand_cells[0].row, fact.operand_cells[0].column)
    operand_page.cells[key] = tuple(resident_operands)
    try:
        assert CorrelationTable._source_owned_sequence_return_findings(module)
    finally:
        operand_page.cells[key] = operands
    artifact = emit_ssa_module_to_c(module, function_name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "owned_bytes_views")
    parameter_id = dict(module.functions[function_name].metadata[
        "parameter_names"
    ])["data"]
    namespace = {}
    exec(source, namespace)
    for data in (b"", b"\x00\xff", bytes(range(256))):
        execution = artifact.prepare_execution({
            parameter_id: np.frombuffer(data, dtype=np.uint8),
        })
        for _ in range(2):
            execution.run()
            assert bytes(execution.buffers[ports.data_id].tolist()) == (
                namespace["root"](data)
            )
