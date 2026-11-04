"""A fixed weight view and a runtime tensor use one broadcast geometry proof."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    TENSOR_RUNTIME_BROADCAST_VALIDITY,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import identity_book


def test_dynamic_rows_broadcast_against_fixed_weight_view():
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "bits", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = """def root(bits: AbstractTensor):
    weights = bits.ensure_tensor((128, 64, 32, 16, 8, 4, 2, 1))
    return (bits.reshape(-1, 8) * weights.unsqueeze(0)).sum(dim=1)
"""
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="dynamic_static_broadcast_probe",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _event: None, progress_bars=False,
    )
    book = identity_book(module)
    rows = tuple(book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY).rows())
    assert len(rows) == 1
    fact = book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY).cell(
        rows[0], book.latest_ref(TENSOR_RUNTIME_BROADCAST_VALIDITY, rows[0]).column,
    )
    assert fact[0] == "mul"
    assert fact[1:3] == (16, 19)
    assert any(
        instruction.op == "Call"
        and instruction.attributes.get("callee") == fact[16]
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
