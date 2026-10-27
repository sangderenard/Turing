"""Exact C native values must retain their declared integer storage class."""
import numpy as np
import pytest

from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule


def run(module, feeds, tmp_path):
    artifact = emit_ssa_to_c(module, "root")
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    return artifact.prepare_execution(feeds).run().buffers


@pytest.mark.parametrize("dtype,value", [
    ("uint8", 255), ("uint16", 65535), ("uint32", 2**32 - 1),
    ("uint64", 2**53 + 1), ("uint64", 2**63 + 17), ("uint64", 2**64 - 1),
    ("int8", -128), ("int16", -32768),
])
def test_integer_scalar_add_zero_keeps_value_and_buffer_dtype(dtype, value, tmp_path):
    left, right, result = (SSAValue(index, dtype) for index in range(3))
    module = IRModule({"root": Function("root", [left, right], {
        "entry": BasicBlock("entry", [
            Instr("Add", [left, right], result), Instr("Ret", [result], None),
        ]),
    })})
    buffers = run(module, {0: np.array([value], dtype=dtype),
                           1: np.array([0], dtype=dtype)}, tmp_path)
    assert buffers[2].dtype == np.dtype(dtype)
    assert buffers[2].item() == value


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "uint64"])
def test_unsigned_memory_stride_and_internal_call_round_trip(dtype, tmp_path):
    maximum = int(np.iinfo(dtype).max)
    column, value = SSAValue(0, dtype, (3,)), SSAValue(1, dtype)
    index, address, loaded = SSAValue(2, "int64"), SSAValue(3, "ptr"), SSAValue(4, dtype)
    helper = Function("helper", [column, value], {"entry": BasicBlock("entry", [
        Instr("Const", [], index, attributes={"value": 1}),
        Instr("GetElementPtr", [column, index], address),
        Instr("Store", [value, address], None),
        Instr("Load", [address], loaded), Instr("Ret", [loaded], None),
    ])})
    data, incoming, result = SSAValue(0, dtype, (3,)), SSAValue(1, dtype), SSAValue(5, dtype)
    root = Function("root", [data, incoming], {"entry": BasicBlock("entry", [
        Instr("Call", [data, incoming], result, attributes={"callee": "helper"}),
        Instr("Ret", [result], None),
    ])})
    buffers = run(IRModule({"root": root, "helper": helper}), {
        0: np.array([7, 0, 11], dtype=dtype), 1: np.array([maximum], dtype=dtype),
    }, tmp_path)
    assert buffers[0].dtype == np.dtype(dtype)
    assert buffers[0].tolist() == [7, maximum, 11]
    assert buffers[5].dtype == np.dtype(dtype)
    assert buffers[5].item() == maximum


@pytest.mark.parametrize("operation,expected", [
    ("Mod", (2**63 + 17) % 7), ("FloorDiv", (2**63 + 17) // 7),
    ("Shr", (2**63 + 17) >> 7), ("BitAnd", (2**63 + 17) & 7),
    ("Lt", False), ("ULt", False), ("Max", 2**63 + 17),
])
def test_uint64_operations_keep_unsigned_domain(operation, expected, tmp_path):
    left, right = SSAValue(0, "uint64"), SSAValue(1, "uint64")
    result = SSAValue(2, "bool" if operation in {"Lt", "ULt"} else "uint64")
    module = IRModule({"root": Function("root", [left, right], {"entry": BasicBlock("entry", [
        Instr(operation, [left, right], result), Instr("Ret", [result], None),
    ])})})
    buffers = run(module, {0: np.array([2**63 + 17], dtype=np.uint64),
                           1: np.array([7], dtype=np.uint64)}, tmp_path)
    assert buffers[2].item() == expected


def test_uint64_constant_and_widening_are_exact(tmp_path):
    source = SSAValue(0, "uint32")
    widened, maximum, result = SSAValue(1, "uint64"), SSAValue(2, "uint64"), SSAValue(3, "bool")
    module = IRModule({"root": Function("root", [source], {"entry": BasicBlock("entry", [
        Instr("Cast", [source], widened, attributes={"target_dtype": "uint64"}),
        Instr("Const", [], maximum, attributes={"value": 2**64 - 1}),
        Instr("Lt", [widened, maximum], result), Instr("Ret", [widened, maximum, result], None),
    ])})})
    buffers = run(module, {0: np.array([2**32 - 1], dtype=np.uint32)}, tmp_path)
    assert buffers[1].item() == 2**32 - 1
    assert buffers[2].item() == 2**64 - 1
    assert buffers[3].item() == 1

@pytest.mark.parametrize("dtype,operation,left,right,expected", [
    ("uint16", "Mul", 65535, 65535, 1),
    ("uint8", "Mul", 255, 255, 1),
    ("int8", "Shr", -1, 1, 127),
    ("uint16", "Shl", 65535, 3, 65528),
])
def test_narrow_integer_operations_keep_declared_bit_width(dtype, operation, left, right, expected, tmp_path):
    a, b, result = (SSAValue(index, dtype) for index in range(3))
    module = IRModule({"root": Function("root", [a, b], {"entry": BasicBlock("entry", [
        Instr(operation, [a, b], result), Instr("Ret", [result], None),
    ])})})
    buffers = run(module, {0: np.array([left], dtype=dtype),
                           1: np.array([right], dtype=dtype)}, tmp_path)
    assert buffers[2].item() == expected


def test_mixed_sign_operands_refuse_implicit_c_promotion():
    left, right, result = SSAValue(0, "int64"), SSAValue(1, "uint64"), SSAValue(2, "bool")
    module = IRModule({"root": Function("root", [left, right], {"entry": BasicBlock("entry", [
        Instr("Lt", [left, right], result), Instr("Ret", [result], None),
    ])})})
    artifact = emit_ssa_to_c(module, "root")
    assert not artifact.complete
    assert any("mixed signed/unsigned" in failure.reason for failure in artifact.shortfalls)


@pytest.mark.parametrize("source_dtype,value,operation,expected", [
    ("uint32", 2**32 - 1, "SExt", -1),
    ("int32", -1, "ZExt", 2**32 - 1),
    ("uint16", 65535, "SExt", -1),
    ("bool", True, "SExt", -1),
])
def test_explicit_extension_opcode_owns_its_signedness(source_dtype, value, operation, expected, tmp_path):
    source, result = SSAValue(0, source_dtype), SSAValue(1, "int64")
    module = IRModule({"root": Function("root", [source], {"entry": BasicBlock("entry", [
        Instr(operation, [source], result), Instr("Ret", [result], None),
    ])})})
    buffers = run(module, {0: np.array([value], dtype=source_dtype)}, tmp_path)
    assert buffers[1].item() == expected

@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "uint64"])
def test_unsigned_elementwise_multiply_uses_declared_rows(dtype, tmp_path):
    maximum = int(np.iinfo(dtype).max)
    left, right, result = (SSAValue(index, dtype, (3,)) for index in range(3))
    module = IRModule({"root": Function("root", [left, right], {"entry": BasicBlock("entry", [
        Instr("Mul", [left, right], result), Instr("Ret", [result], None),
    ])})})
    buffers = run(module, {0: np.array([maximum, maximum, 7], dtype=dtype),
                           1: np.array([maximum, 0, 3], dtype=dtype)}, tmp_path)
    assert buffers[2].dtype == np.dtype(dtype)
    assert buffers[2].tolist() == [1, 0, 21]


def test_boolean_public_buffer_keeps_boolean_class(tmp_path):
    left, right, result = (SSAValue(index, "bool") for index in range(3))
    module = IRModule({"root": Function("root", [left, right], {"entry": BasicBlock("entry", [
        Instr("LAnd", [left, right], result), Instr("Ret", [result], None),
    ])})})
    buffers = run(module, {0: np.array([True], dtype=np.bool_),
                           1: np.array([False], dtype=np.bool_)}, tmp_path)
    assert buffers[2].dtype == np.dtype("bool")
    assert buffers[2].item() is False


def test_structural_pointer_table_preserves_address_bits_before_use(tmp_path):
    table = SSAValue(0, "ptrptr_float64", (2,))
    module = IRModule({"root": Function("root", [table], {
        "entry": BasicBlock("entry", [Instr("Ret", [], None)]),
    })})
    artifact = emit_ssa_to_c(module, "root")
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    # This is an opaque address-bit transport check. The impossible address
    # is deliberately never dereferenced by the native function.
    words = np.array([2**53 + 1, 0], dtype=np.uintp)
    execution = artifact.prepare_execution({0: words}).run()
    assert execution.buffers[0].dtype == np.dtype("uintp")
    assert execution.buffers[0].tolist() == [2**53 + 1, 0]


def test_integer_power_refuses_floating_libm_substitution():
    left, right, result = (SSAValue(index, "uint64") for index in range(3))
    module = IRModule({"root": Function("root", [left, right], {"entry": BasicBlock("entry", [
        Instr("Pow", [left, right], result), Instr("Ret", [result], None),
    ])})})
    artifact = emit_ssa_to_c(module, "root")
    assert not artifact.complete
    assert any("exact power reduction" in item.reason for item in artifact.shortfalls)
