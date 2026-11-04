"""The C reader uses the owned heap carrier of a dynamic broadcast."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.concordance_declarations import (
    TENSOR_RUNTIME_BROADCAST_VALIDITY,
)
from src.compiler.identity_concordance import identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_storage_requirements import owned_tensor_data_bindings


def test_dynamic_broadcast_carriers_are_not_activation_temporaries():
    source = """def root(left: AbstractTensor, right: AbstractTensor) -> AbstractTensor:
    a = left.unsqueeze(1)
    b = right.unsqueeze(0)
    return a + b
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": name, "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"}
            for name in ("left", "right")
        ],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="c_owned_broadcast_carrier",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    owned = [
        binding
        for function in module.functions
        for binding in owned_tensor_data_bindings(module, function)
    ]
    assert len(owned) >= 2
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert not [
        shortfall for shortfall in artifact.shortfalls
        if shortfall.operation in {"temporary", "operand", "Add"}
    ]
    (broadcast_row,) = tuple(identity_book(module).page(
        TENSOR_RUNTIME_BROADCAST_VALIDITY
    ).rows())
    index = f"ew_i_{broadcast_row[2]}"
    (loop,) = tuple(
        line for line in artifact.source.splitlines()
        if f"for (ptrdiff_t {index}" in line
        and "__builtin_add_overflow" in line
    )
    assert loop.count(f"[{index}]") >= 3
    assert any(
        shortfall.operation == "Ret"
        and "no public input" in shortfall.reason
        for shortfall in artifact.shortfalls
    )
