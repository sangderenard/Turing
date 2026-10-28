from pathlib import Path

import numpy as np
import pytest

from src.compiler.ssa_llvm_backend import (
    emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution,
)
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule


@pytest.mark.parametrize("reverse", [False, True])
def test_internal_signed128_add_keeps_full_uint64_and_signed_domain(reverse, tmp_path):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report

    left, right = ("right", "left") if reverse else ("left", "right")
    source = f"def root(left, right):\n    return ({left} + {right}) > left\n"
    contract = ExtractionContract(Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml").with_program_abi({
        "records": {}, "bindings": [], "values": [
            {"function": "root", "parameter": name, "storage": "scalar", "dtype": dtype,
             "rank": 0, "python_type": "builtins.int"}
            for name, dtype in (("left", "uint64"), ("right", "int64"))
        ],
    })
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"internal128_{reverse}", extraction_contract=contract,
        progress=lambda _: None, progress_bars=False,
    )
    assert "0 finding(s)" in concordance_report(module)
    root = exports[0]
    artifact = emit_ssa_function_to_llvm(module, root)
    assert artifact.complete, artifact.shortfalls
    assert "zext i64" in artifact.llvm_ir and "sext i64" in artifact.llvm_ir
    assert "add i128" in artifact.llvm_ir
    native = compile_artifact(artifact, directory=tmp_path)
    arguments = module.functions[root].args
    feeds = {value.id: np.zeros(1, dtype=value.dtype) for value in arguments}
    execution = prepare_artifact_execution(native, feeds)
    for left_value, right_value in (
        (2**64 - 1, 2**63 - 1), (0, -(2**63)),
        (2**63 + 1, -1), (2**64 - 1, 0), (0, 1),
    ):
        execution.buffers[arguments[0].id][0] = left_value
        execution.buffers[arguments[1].id][0] = right_value
        execution.run()
        assert bool(execution.buffers[outputs[root][0].id].item()) == (left_value + right_value > left_value)


def test_internal_unsigned128_comparison_keeps_high_bit(tmp_path):
    left, right, result = SSAValue(0, "uint128"), SSAValue(1, "uint128"), SSAValue(2, "bool")
    module = IRModule({"root": Function("root", [], {"entry": BasicBlock("entry", [
        Instr("Const", [], left, attributes={"value": 2**128 - 1}),
        Instr("Const", [], right, attributes={"value": 1}),
        Instr("Gt", [left, right], result), Instr("Ret", [result], None),
    ])})})
    artifact = emit_ssa_function_to_llvm(module, "root")
    assert artifact.complete, artifact.shortfalls
    assert "icmp ugt i128" in artifact.llvm_ir
    native = compile_artifact(artifact, directory=tmp_path)
    assert prepare_artifact_execution(native, {}).run().buffers[result.id].item()


def test_internal128_call_result_keeps_pointer_abi_storage(tmp_path):
    left, right = SSAValue(0, "uint64"), SSAValue(1, "int64")
    total, extended, result = SSAValue(2, "int128"), SSAValue(3, "int128"), SSAValue(4, "bool")
    a, b = SSAValue(10, "uint64"), SSAValue(11, "int64")
    wide_a, wide_b, wide_sum = (SSAValue(i, "int128") for i in (12, 13, 14))
    module = IRModule({
        "root": Function("root", [left, right], {"entry": BasicBlock("entry", [
            Instr("Call", [left, right], total, attributes={"callee": "wide"}),
            Instr("Cast", [left], extended, attributes={"target_dtype": "int128"}),
            Instr("Gt", [total, extended], result), Instr("Ret", [result], None),
        ])}),
        "wide": Function("wide", [a, b], {"entry": BasicBlock("entry", [
            Instr("Cast", [a], wide_a, attributes={"target_dtype": "int128"}),
            Instr("Cast", [b], wide_b, attributes={"target_dtype": "int128"}),
            Instr("Add", [wide_a, wide_b], wide_sum), Instr("Ret", [wide_sum], None),
        ])}),
    })
    artifact = emit_ssa_function_to_llvm(module, "root")
    assert artifact.complete, artifact.shortfalls
    assert "alloca i128, i64 1, align 16" in artifact.llvm_ir
    native = compile_artifact(artifact, directory=tmp_path)
    execution = prepare_artifact_execution(native, {
        left.id: np.array([2**64 - 1], dtype="uint64"),
        right.id: np.array([2**63 - 1], dtype="int64"),
    })
    assert execution.run().buffers[result.id].item()


@pytest.mark.parametrize("public_input", [False, True])
def test_internal128_has_no_public_host_buffer_abi(public_input):
    value = SSAValue(0, "int128")
    instructions = ([] if public_input else [Instr("Const", [], value, attributes={"value": 1})])
    instructions.append(Instr("Ret", [value], None))
    module = IRModule({"root": Function("root", [value] if public_input else [],
                                       {"entry": BasicBlock("entry", instructions)})})
    artifact = emit_ssa_function_to_llvm(module, "root")
    assert not artifact.complete
    assert any("no public 128-bit buffer ABI" in item.reason for item in artifact.shortfalls)
