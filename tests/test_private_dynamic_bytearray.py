"""A fresh private bytearray uses the existing owned sequence ABI."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SOURCE_PRIVATE_DYNAMIC_SEQUENCE,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_storage_requirements import public_owned_sequence_returns


def test_private_fresh_bytearray_early_return_native(tmp_path):
    source = (
        "def root(data: bytes) -> bytes:\n"
        "    header = bytearray()\n"
        "    header.append(2)\n"
        "    for _ in range(1):\n"
        "        if not data:\n"
        "            return bytes(header)\n"
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
        source, "root", name="private_dynamic_bytearray",
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    function_name = exports[0]
    book = module.metadata["identity_book"]
    page = book.page(SOURCE_PRIVATE_DYNAMIC_SEQUENCE)
    rows = tuple(page.rows())
    assert len(rows) == 1
    cell = book.latest_ref(SOURCE_PRIVATE_DYNAMIC_SEQUENCE, rows[0])
    fact = page.cell(cell.row, cell.column)
    assert CorrelationTable._source_private_dynamic_sequence_findings(module) == []
    assert CorrelationTable.build(module).findings(module) == []
    key = (cell.row, cell.column)
    page.cells[key] = replace(fact, static_definition_cell=fact.mutation_cell)
    try:
        assert CorrelationTable._source_private_dynamic_sequence_findings(module)
    finally:
        page.cells[key] = fact

    (ports,) = public_owned_sequence_returns(module, function_name)
    artifact = emit_ssa_module_to_c(module, function_name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "private_dynamic_bytearray")
    parameter_id = dict(module.functions[function_name].metadata[
        "parameter_names"
    ])["data"]
    namespace = {}
    exec(source, namespace)
    for data in (b"", b"\x00\xff", bytes(range(32))):
        execution = artifact.prepare_execution({
            parameter_id: np.frombuffer(data, dtype=np.uint8),
        })
        for _ in range(2):
            execution.run()
            assert bytes(execution.buffers[ports.data_id].tolist()) == (
                namespace["root"](data)
            )
