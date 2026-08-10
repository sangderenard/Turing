from __future__ import annotations

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    LLVM_SSA_MODULE,
    TRANSLATIONS,
)
from src.common.tensors.accelerator_backends.llvm_repository_ssa import (
    import_llvm_to_repository_ssa,
)
from src.compiler.precompile_to_ssa import find_ssa_cycles
from src.compiler.ssa_features import (
    RANDOM_SSA_MODULE,
    XOROSHIRO128SS_FILL,
    link_required_ssa_features,
)
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
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


def test_random_feature_imports_as_real_bitwise_repository_ssa():
    imported = import_llvm_to_repository_ssa(RANDOM_SSA_MODULE)

    assert imported.complete, imported.shortfall_report()
    function = imported.module.functions[XOROSHIRO128SS_FILL]
    operations = {
        instruction.op
        for block in function.blocks.values()
        for instruction in block.instrs
    }
    assert {
        Handler.Xor.value,
        Handler.Or.value,
        Handler.Shl.value,
        Handler.Shr.value,
    } <= operations


def test_random_feature_is_linked_only_when_called():
    output = SSAValue(0, dtype="float64", shape=(4,))
    caller = Function(
        "program",
        [],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr(
                        Handler.Call.value,
                        [],
                        output,
                        attributes={"callee": XOROSHIRO128SS_FILL},
                    )
                ],
            )
        },
    )

    assert XOROSHIRO128SS_FILL in link_required_ssa_features({"program": caller})
    assert XOROSHIRO128SS_FILL not in link_required_ssa_features({})
