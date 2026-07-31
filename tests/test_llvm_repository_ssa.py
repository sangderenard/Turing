from __future__ import annotations

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    LLVM_SSA_MODULE,
    TRANSLATIONS,
)
from src.common.tensors.accelerator_backends.llvm_repository_ssa import (
    import_llvm_to_repository_ssa,
)
from src.compiler.precompile_to_ssa import find_ssa_cycles
from src.transmogrifier.ssa_registry import Handler


def test_real_llvm_tensor_algorithms_import_to_fundamental_repository_ssa():
    result = import_llvm_to_repository_ssa(LLVM_SSA_MODULE)

    assert result.complete, result.shortfall_report()
    legal = {handler.value for handler in Handler}
    assert all(
        instruction.op in legal
        for function in result.module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    assert {
        translation.llvm_symbol
        for translation in TRANSLATIONS
    } <= set(result.module.functions)


def test_llvm_tensor_loops_retain_phi_cfg_cycles():
    result = import_llvm_to_repository_ssa(LLVM_SSA_MODULE)
    fill = result.module.functions["fill_double"]
    cycles = find_ssa_cycles(fill)

    assert len(cycles) == 1
    assert cycles[0].represented_by_phi


def test_llvm_switch_is_legalized_to_existing_compare_and_branch_ops():
    result = import_llvm_to_repository_ssa(LLVM_SSA_MODULE)
    binary = result.module.functions["binary_value"]
    instructions = [
        instruction
        for block in binary.blocks.values()
        for instruction in block.instrs
    ]

    assert "Switch" not in {instruction.op for instruction in instructions}
    assert any(
        instruction.op == Handler.Eq.value
        and instruction.attributes.get("llvm_opcode") == "switch"
        for instruction in instructions
    )
    assert any(
        instruction.op == Handler.CondBr.value
        and instruction.attributes.get("llvm_opcode") == "switch"
        for instruction in instructions
    )
