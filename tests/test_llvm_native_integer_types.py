"""Exact scalar/row ABI tests for the canonical LLVM module emitter."""
from pathlib import Path

import numpy as np
import pytest
from llvmlite import binding

from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule
from src.compiler.ssa_llvm_backend import (
    emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution,
)


def run(module, feeds, tmp_path):
    artifact = emit_ssa_function_to_llvm(module, "root")
    assert artifact.complete, artifact.shortfalls
    binding.parse_assembly(artifact.llvm_ir).verify()
    native = compile_artifact(artifact, directory=tmp_path)
    return artifact, prepare_artifact_execution(native, feeds).run().buffers


@pytest.mark.parametrize("bits", [8, 16, 32, 64])
def test_unsigned_operations_execute_in_declared_width(bits, tmp_path):
    dtype = f"uint{bits}"
    a, b = SSAValue(0, dtype), SSAValue(1, dtype)
    operations = ("Add", "Mod", "FloorDiv", "Lt", "ULt", "Shr", "BitAnd", "Min", "Max")
    results = [SSAValue(index + 2, "bool" if op in {"Lt", "ULt"} else dtype)
               for index, op in enumerate(operations)]
    module = IRModule({"root": Function("root", [a, b], {"entry": BasicBlock("entry", [
        *(Instr(op, [a, b], result) for op, result in zip(operations, results)),
        Instr("Ret", results, None),
    ])})})
    maximum = (1 << bits) - 1
    # Stay inside arithmetic range; still cross the signed high-bit boundary.
    value, divisor = maximum - 4, 3
    artifact, buffers = run(module, {0: np.array([value], dtype=dtype),
                                    1: np.array([divisor], dtype=dtype)}, tmp_path)
    expected = [value + divisor, value % divisor, value // divisor,
                value < divisor, value < divisor, value >> divisor,
                value & divisor, min(value, divisor), max(value, divisor)]
    for result, wanted in zip(results, expected):
        assert buffers[result.id].item() == wanted
        assert buffers[result.id].dtype == np.dtype(result.dtype)
    assert artifact.buffer_element_dtypes[:2] == (dtype, dtype)


def test_unsigned_casts_keep_sign_and_high_bits(tmp_path):
    narrow, wide = SSAValue(0, "uint32"), SSAValue(1, "uint64")
    extended, narrowed, floating = SSAValue(2, "uint64"), SSAValue(3, "uint32"), SSAValue(4, "float64")
    module = IRModule({"root": Function("root", [narrow, wide], {"entry": BasicBlock("entry", [
        Instr("Cast", [narrow], extended, attributes={"target_dtype": "uint64"}),
        Instr("Cast", [wide], narrowed, attributes={"target_dtype": "uint32"}),
        Instr("Cast", [wide], floating, attributes={"target_dtype": "float64"}),
        Instr("Ret", [extended, narrowed, floating], None),
    ])})})
    value = (1 << 63) + (1 << 12)
    artifact, buffers = run(module, {0: np.array([2**32 - 1], dtype="uint32"),
                                    1: np.array([value], dtype="uint64")}, tmp_path)
    assert buffers[2].item() == 2**32 - 1
    assert buffers[3].item() == 1 << 12
    assert buffers[4].item() == float(value)
    assert "zext i32" in artifact.llvm_ir
    assert "uitofp i64" in artifact.llvm_ir


@pytest.mark.parametrize("dtype,value", [("int8", -120), ("uint8", 250),
                                         ("int16", -32000), ("uint16", 65000),
                                         ("uint32", 2**32 - 1), ("uint64", 2**63 + 1)])
def test_integer_load_pins_before_aliasing_store(dtype, value, tmp_path):
    data = SSAValue(0, dtype, (2,))
    index, address = SSAValue(1, "int64"), SSAValue(2, "ptr")
    before, zero = SSAValue(3, dtype), SSAValue(4, dtype)
    module = IRModule({"root": Function("root", [data], {"entry": BasicBlock("entry", [
        Instr("Const", [], index, attributes={"value": 1}),
        Instr("GetElementPtr", [data, index], address),
        Instr("Load", [address], before),
        Instr("Const", [], zero, attributes={"value": 0}),
        Instr("Store", [zero, address], None),
        Instr("Ret", [before], None),
    ])})})
    _, buffers = run(module, {0: np.array([0, value], dtype=dtype)}, tmp_path)
    assert buffers[3].item() == value
    assert buffers[0].tolist() == [0, 0]


def test_float32_store_rounds_to_declared_row(tmp_path):
    data = SSAValue(0, "float32", (2,))
    value = SSAValue(1, "float64")
    index, address = SSAValue(2, "int64"), SSAValue(3, "ptr")
    module = IRModule({"root": Function("root", [data, value], {"entry": BasicBlock("entry", [
        Instr("Const", [], index, attributes={"value": 1}),
        Instr("GetElementPtr", [data, index], address),
        Instr("Store", [value, address], None), Instr("Ret", [data], None),
    ])})})
    _, buffers = run(module, {0: np.array([7, 0], dtype="float32"),
                              1: np.array([2**24 + 1], dtype="float64")}, tmp_path)
    assert buffers[0].dtype == np.dtype("float32")
    assert buffers[0].tolist() == [7, 2**24]


def test_mixed_signed_unsigned_arithmetic_requires_conversion():
    a, b, result = SSAValue(0, "uint64"), SSAValue(1, "int64"), SSAValue(2, "uint64")
    module = IRModule({"root": Function("root", [a, b], {"entry": BasicBlock("entry", [
        Instr("Add", [a, b], result), Instr("Ret", [result], None),
    ])})})
    artifact = emit_ssa_function_to_llvm(module, "root")
    assert not artifact.complete
    assert any("explicit common dtype conversion" in item.reason for item in artifact.shortfalls)


def test_uint64_call_phi_and_return_keep_exact_identity(tmp_path):
    a, b, choose = SSAValue(0, "uint64"), SSAValue(1, "uint64"), SSAValue(2, "bool")
    merged, result = SSAValue(3, "uint64"), SSAValue(4, "uint64")
    helper_arg = SSAValue(0, "uint64")
    helper = Function("identity", [helper_arg], {"entry": BasicBlock("entry", [Instr("Ret", [helper_arg], None)])})
    root = Function("root", [a, b, choose], {
        "entry": BasicBlock("entry", [Instr("CondBr", [choose], None, attributes={"true": "yes", "false": "no"})]),
        "yes": BasicBlock("yes", [Instr("Br", [], None, attributes={"target": "merge"})]),
        "no": BasicBlock("no", [Instr("Br", [], None, attributes={"target": "merge"})]),
        "merge": BasicBlock("merge", [
            Instr("Phi", [a, b], merged, attributes={"incoming_blocks": ("yes", "no")}),
            Instr("Call", [merged], result, attributes={"callee": "identity"}),
            Instr("Ret", [result], None),
        ]),
    })
    value = 2**63 + 3
    artifact, buffers = run(IRModule({"root": root, "identity": helper}), {
        0: np.array([value], dtype="uint64"), 1: np.array([2**64 - 1], dtype="uint64"),
        2: np.array([True], dtype="bool"),
    }, tmp_path)
    assert buffers[4].item() == value
    assert buffers[4].dtype == np.dtype("uint64")


@pytest.mark.parametrize("bits", [8, 16, 32, 64])
def test_unsigned_span_comparison_and_reduction(bits, tmp_path):
    dtype = f"uint{bits}"
    data = SSAValue(0, dtype, (3,))
    threshold = SSAValue(1, dtype)
    compared = SSAValue(2, "bool", (3,))
    minimum, maximum = SSAValue(3, dtype), SSAValue(4, dtype)
    module = IRModule({"root": Function("root", [data, threshold], {"entry": BasicBlock("entry", [
        Instr("Lt", [data, threshold], compared),
        Instr("min", [data], minimum), Instr("max", [data], maximum),
        Instr("Ret", [compared, minimum, maximum], None),
    ])})})
    high = (1 << bits) - 1
    _, buffers = run(module, {0: np.array([0, high, high - 1], dtype=dtype),
                              1: np.array([high], dtype=dtype)}, tmp_path)
    assert buffers[2].tolist() == [True, False, True]
    assert buffers[3].item() == 0
    assert buffers[4].item() == high


def test_pointer_call_abi_does_not_erase_integer_signedness():
    actual, formal = SSAValue(0, "uint64"), SSAValue(0, "int64")
    helper = Function("helper", [formal], {"entry": BasicBlock("entry", [Instr("Ret", [], None)])})
    root = Function("root", [actual], {"entry": BasicBlock("entry", [
        Instr("Call", [actual], None, attributes={"callee": "helper"}), Instr("Ret", [], None),
    ])})
    artifact = emit_ssa_function_to_llvm(IRModule({"root": root, "helper": helper}), "root")
    assert not artifact.complete
    assert any("call integer storage requires explicit conversion" in item.reason for item in artifact.shortfalls)


def test_float32_row_load_widens_after_reading_four_bytes(tmp_path):
    data = SSAValue(0, "float32", (3,))
    index, address, loaded = SSAValue(1, "int64"), SSAValue(2, "ptr"), SSAValue(3, "float32")
    module = IRModule({"root": Function("root", [data], {"entry": BasicBlock("entry", [
        Instr("Const", [], index, attributes={"value": 1}),
        Instr("GetElementPtr", [data, index], address),
        Instr("Load", [address], loaded), Instr("Ret", [loaded], None),
    ])})})
    artifact, buffers = run(module, {0: np.array([3, 1.5, 9], dtype="float32")}, tmp_path)
    assert buffers[3].item() == 1.5
    assert buffers[3].dtype == np.dtype("float64")
    assert "load float" in artifact.llvm_ir
    assert "fpext float" in artifact.llvm_ir


def test_elementwise_integer_sweep_keeps_common_dtype_refusal():
    data = SSAValue(0, "uint64", (3,))
    signed = SSAValue(1, "int64")
    result = SSAValue(2, "int64", (3,))
    module = IRModule({"root": Function("root", [data, signed], {"entry": BasicBlock("entry", [
        Instr("Add", [data, signed], result), Instr("Ret", [result], None),
    ])})})
    artifact = emit_ssa_function_to_llvm(module, "root")
    assert not artifact.complete
    assert any("explicit common dtype conversion" in item.reason for item in artifact.shortfalls)
