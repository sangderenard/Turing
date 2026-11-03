"""A dynamic byte sequence retains its proved int64 storage across tensor views."""

import inspect
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.common.tensors.compression import octet_huffman
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.identity_concordance import identity_book
from src.compiler.concordance_declarations import VALUE_TYPE_ABI
from src.compiler.source_evidence_resolver import (
    PhysicalLayoutEvidence, UnresolvedSourceDemand, require_physical_layout,
)
from src.compiler.ssa_record_return_state import function_scope_of
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def test_public_octets_same_dtype_uses_proved_dynamic_sequence_view(tmp_path):
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{
            "function": "_octets", "parameter": "data", "storage": "span",
            "dtype": "int64", "rank": 1, "python_type": "builtins.bytes",
        }],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        textwrap.dedent(inspect.getsource(octet_huffman._octets)),
        "_octets", python_bindings=vars(octet_huffman),
        source_module=octet_huffman.__name__,
        name="dynamic_byte_view", extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    assert not any(
        instruction.op == "Call"
        and str(instruction.attributes.get("callee", "")).startswith("cast_")
        for function in module.functions.values()
        if function.name.startswith("dynamic_byte_view___octets")
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    root = module.functions["dynamic_byte_view___octets"]
    assert root.metadata["value_aliases"][8] == 0
    book = identity_book(module)
    scope = function_scope_of(root)
    physical = require_physical_layout(scope, 0, book=book)
    assert isinstance(physical, PhysicalLayoutEvidence)
    assert (physical.storage, physical.physical_dtype, physical.rank,
            physical.shape) == ("span", "int64", 1, ())
    abi_ref = next(ref for ref in physical.proof_cells
                   if ref.page == VALUE_TYPE_ABI)
    abi_page = book.page(VALUE_TYPE_ABI)
    original_abi = abi_page.cells[(abi_ref.row, abi_ref.column)]
    original_ssa = book.page(physical.subject_cell.page).cells[
        (physical.subject_cell.row, physical.subject_cell.column)
    ]
    try:
        # A static extent of four cannot prove the SSA's static extent three.
        book.page(physical.subject_cell.page).cells[
            (physical.subject_cell.row, physical.subject_cell.column)
        ] = replace(original_ssa, shape=(3,))
        abi_page.cells[(abi_ref.row, abi_ref.column)] = replace(
            original_abi, physical_alternatives=(
                replace(original_abi.physical_alternatives[0],
                        shape=(4,), metadata_state="static"),
            ),
        )
        mismatch = require_physical_layout(scope, 0, book=book)
        assert isinstance(mismatch, UnresolvedSourceDemand)
        assert mismatch.missing == "span_abi_disagreement"
        # Even an unknown extent does not erase a rank disagreement.
        abi_page.cells[(abi_ref.row, abi_ref.column)] = replace(
            original_abi, physical_alternatives=(
                replace(original_abi.physical_alternatives[0],
                        shape=(-1, -1), rank=2),
            ),
        )
        book.page(physical.subject_cell.page).cells[
            (physical.subject_cell.row, physical.subject_cell.column)
        ] = original_ssa
        mismatch = require_physical_layout(scope, 0, book=book)
        assert isinstance(mismatch, UnresolvedSourceDemand)
        assert mismatch.missing == "span_abi_disagreement"
    finally:
        abi_page.cells[(abi_ref.row, abi_ref.column)] = original_abi
        book.page(physical.subject_cell.page).cells[
            (physical.subject_cell.row, physical.subject_cell.column)
        ] = original_ssa
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    native = artifact.compile(tmp_path / "native_octets")
    abi = artifact.runtime_abi()
    sequence = module.sequence_tables[root.name].by_id(0)
    assert tuple(item["value_id"] for item in abi["buffers"]) == (
        0, sequence.length_address_id, sequence.capacity_value_id,
    )
    for payload in (b"", bytes([0, 127, 128, 255])):
        data = np.asarray(list(payload), dtype=np.int64)
        execution = native.prepare_execution({
            0: data,
        }).run()
        assert execution.buffers[0].tolist() == octet_huffman._octets(payload).tolist()
        assert execution.buffers[sequence.length_address_id].item() == len(payload)
        assert execution.buffers[sequence.capacity_value_id].item() == len(payload)
