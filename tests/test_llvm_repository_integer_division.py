"""Imported LLVM integer division retains its declared physical width."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SCALAR_INTEGER_WIDTH,
    SSA_VALUE,
    IntegerWidthDecision,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import identity_book


def test_imported_sdiv_keeps_int32_through_mixed_rank_broadcast():
    reference = c_backend_repository_ssa_reference()
    contract = ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml").resolve()
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "left", "storage": "span",
             "dtype": "int64", "rank": 2, "shape": [2, 1],
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "right", "storage": "span",
             "dtype": "int64", "rank": 1, "shape": [3],
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
        ],
        "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        "def root(left: AbstractTensor, right: AbstractTensor) -> AbstractTensor:\n"
        "    return left + right\n",
        "root", name="repository_integer_division", extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=reference, progress=lambda _phase: None,
        progress_bars=False,
    )
    kernel = module.functions["broadcast_int64"]
    divisions = [instruction for block in kernel.blocks.values()
                 for instruction in block.instrs
                 if instruction.attributes.get("llvm_opcode") == "sdiv"]
    assert divisions
    book = identity_book(module)
    for instruction in divisions:
        assert instruction.res.dtype == "int32"
        row = ("broadcast_int64", int(instruction.res.id))
        assert book.page(SSA_VALUE).latest(row).dtype == "int32"
        width = book.page(SCALAR_INTEGER_WIDTH).latest(row)
        assert width.decision is IntegerWidthDecision.KEPT_DECLARED
