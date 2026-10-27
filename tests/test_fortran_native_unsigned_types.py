"""Native GNU unsigned kinds retain exact width and value through the Fortran ABI."""
from pathlib import Path
import subprocess

import numpy as np
import pytest

from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
from src.compiler.ssa_fortran_backend import (
    FortranCoreNative, emit_module, compile_module, fortran_compiler,
)
from src.compiler.fortran_toolchain import native_fortran_flags, aggressive_fortran_flags
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule


@pytest.fixture(scope="module")
def native_fortran_compiler():
    compiler = fortran_compiler()
    if compiler is None:
        pytest.skip("native Fortran compiler unavailable")
    return compiler


@pytest.fixture(scope="module")
def native_unsigned_toolchain(native_fortran_compiler):
    # Probe a known-good native ISO-C declaration before generating the code
    # under test. Older GNU/other compilers may truthfully refuse the feature;
    # no exception from generated-source emission/compile/execution is caught.
    try:
        native_fortran_flags(native_fortran_compiler,
                             "unsigned(c_int64_t) :: capability_probe",
                             aggressive_fortran_flags(native_fortran_compiler))
    except ValueError as error:
        if str(error).startswith((
            "native unsigned Fortran requires a compiler supporting -funsigned;",
            "selected Fortran compiler/flags do not support exact native unsigned ISO C storage with -funsigned:",
        )):
            pytest.skip("selected Fortran toolchain lacks native unsigned ISO-C storage: " + str(error))
        raise
    return native_fortran_compiler


def native(module, tmp_path, name="root"):
    artifact = emit_module(module, name="unsigned_native", progress=lambda _: None)
    assert artifact.complete, artifact.shortfalls
    built = compile_fortran_module_c_shell(
        artifact, {}, tmp_path, entrypoint=name, library=True,
    )
    return FortranCoreNative(built.executable_path, artifact.api.entry_point(name), artifact.api), artifact


@pytest.mark.parametrize("dtype,width", [("uint8", 8), ("uint16", 16), ("uint32", 32), ("uint64", 64)])
def test_unsigned_native_scalar_operations_keep_full_width(dtype, width, tmp_path, native_unsigned_toolchain):
    x, y = SSAValue(0, dtype), SSAValue(1, dtype)
    operators = ("Add", "Sub", "Mul", "Div", "FloorDiv", "Mod", "Lt", "ULt", "Shr", "Shl", "BitAnd", "BitOr", "BitXor", "Max", "Min")
    results = [SSAValue(index, "bool" if operation in {"Lt", "ULt"} else dtype)
               for index, operation in enumerate(operators, 2)]
    function = Function("root", [x, y], {"entry": BasicBlock("entry", [
        *(Instr(operation, [x, y], result) for operation, result in zip(operators, results)),
        Instr("Ret", results, None),
    ])})
    core, artifact = native(IRModule({"root": function}), tmp_path)
    mask = (1 << width) - 1
    for left in (0, 1, (1 << (width-1)), mask):
        execution = core.prepare_execution({0: np.array([left], dtype=dtype), 1: np.array([2], dtype=dtype)}).run()
        expected = ((left+2)&mask, (left-2)&mask, (left*2)&mask, left//2, left//2, left%2,
                    left<2, left<2, left>>2, (left<<2)&mask, left&2, left|2, left^2, max(left,2), min(left,2))
        for result, operation, value in zip(results, operators, expected):
            actual = execution.buffers[result.id]
            assert actual.dtype == np.dtype(result.dtype)
            assert actual.item() == value, (dtype, operation, left, actual.item(), value)
    assert f"unsigned(c_int{width}_t)" in artifact.source


@pytest.mark.parametrize("dtype,width", [("uint8", 8), ("uint16", 16), ("uint32", 32), ("uint64", 64)])
def test_unsigned_native_array_stride_load_and_store(dtype, width, tmp_path, native_unsigned_toolchain):
    array = SSAValue(0, dtype, (3,))
    value = SSAValue(1, dtype)
    index, address, loaded = SSAValue(2, "int64"), SSAValue(3, "ptr"), SSAValue(4, dtype)
    function = Function("root", [array, value], {"entry": BasicBlock("entry", [
        Instr("Const", [], index, attributes={"value": 1}),
        Instr("GetElementPtr", [array, index], address),
        Instr("Store", [value, address], None), Instr("Load", [address], loaded),
        Instr("Ret", [loaded], None),
    ])})
    core, _ = native(IRModule({"root": function}), tmp_path)
    maximum = (1 << width) - 1
    execution = core.prepare_execution({0: np.array([1, 2, 3], dtype=dtype), 1: np.array([maximum], dtype=dtype)}).run()
    assert execution.buffers[0].tolist() == [1, maximum, 3]
    assert execution.buffers[loaded.id].dtype == np.dtype(dtype)
    assert execution.buffers[loaded.id].item() == maximum


def test_unsigned_native_literals_casts_calls_and_output(tmp_path, native_unsigned_toolchain):
    source = SSAValue(0, "uint8")
    wide, signed, real, maximum = SSAValue(1,"uint64"), SSAValue(2,"int64"), SSAValue(3,"float64"), SSAValue(4,"uint64")
    helper = Function("helper", [source], {"entry": BasicBlock("entry", [
        Instr("Cast", [source], wide, attributes={"target_dtype":"uint64"}),
        Instr("Ret", [wide], None),
    ])})
    arg = SSAValue(0,"uint8")
    root = Function("root", [arg], {"entry": BasicBlock("entry", [
        Instr("Call", [arg], wide, attributes={"callee":"helper"}),
        Instr("Const", [], maximum, attributes={"value":(1<<64)-1}),
        Instr("Cast", [maximum], signed, attributes={"target_dtype":"int64"}),
        Instr("Cast", [maximum], real, attributes={"target_dtype":"float64"}),
        Instr("Ret", [wide, maximum, signed, real], None),
    ])})
    core, _ = native(IRModule({"root":root,"helper":helper}), tmp_path)
    execution = core.prepare_execution({0:np.array([255],dtype=np.uint8)}).run()
    assert execution.buffers[1].dtype == np.dtype("uint64")
    assert execution.buffers[1].item() == 255
    assert execution.buffers[4].item() == (1<<64)-1
    assert execution.buffers[2].item() == -1
    assert execution.buffers[3].item() == float((1<<64)-1)


def test_direct_compile_module_enables_unsigned_feature(tmp_path, native_unsigned_toolchain):
    result = SSAValue(0,"uint64")
    function = Function("root", [], {"entry":BasicBlock("entry",[
        Instr("Const",[],result,attributes={"value":(1<<64)-1}), Instr("Ret",[result],None),
    ])})
    artifact=emit_module(IRModule({"root":function}),name="unsigned_direct",progress=lambda _:None)
    assert compile_module(artifact,directory=tmp_path).is_file()


def test_unsupported_compiler_has_named_unsigned_feature_refusal():
    with pytest.raises(ValueError, match="native unsigned Fortran requires"):
        native_fortran_flags("unverified-fortran", "unsigned(c_int64_t) :: x", ())


def test_unsigned_feature_checks_exact_compile_flags(monkeypatch):
    # Mandatory even on hosts without Fortran. The capability operation must
    # pass the exact requested flags to the probe, not trust a compiler name.
    calls = []
    def rejected(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "unsupported native feature flag")
    monkeypatch.setattr("src.compiler.fortran_toolchain.subprocess.run", rejected)
    with pytest.raises(ValueError, match="do not support exact native unsigned"):
        native_fortran_flags("gfortran", "unsigned(c_int64_t) :: x", ("-f-no-such-native-type-probe",))
    assert len(calls) == 1
    assert "-f-no-such-native-type-probe" in calls[0]
    assert "-funsigned" in calls[0]
    assert "-c" in calls[0]



def test_inlined_float32_cast_uses_the_callee_native_kind(tmp_path, native_fortran_compiler):
    formal = SSAValue(0, "float32")
    zero, copied = SSAValue(1,"float32"), SSAValue(2,"float32")
    helper = Function("helper", [formal], {"entry": BasicBlock("entry", [
        Instr("Const",[],zero,attributes={"value":0.0}),
        Instr("Add",[formal,zero],copied), Instr("Ret", [copied], None),
    ])})
    integer, narrow, result = SSAValue(0,"int64"), SSAValue(1,"float32"), SSAValue(2,"float32")
    root = Function("root", [integer], {"entry": BasicBlock("entry", [
        Instr("Cast", [integer], narrow, attributes={"target_dtype":"float32"}),
        Instr("Call", [narrow], result, attributes={"callee":"helper"}), Instr("Ret", [result], None),
    ])})
    core, artifact = native(IRModule({"root":root,"helper":helper}), tmp_path)
    execution = core.prepare_execution({0:np.array([16777217],dtype=np.int64)}).run()
    assert execution.buffers[2].dtype == np.dtype("float32")
    assert execution.buffers[2].item() == 16777216
    assert "real(t0, c_float)" in artifact.source


def test_unsigned_select_and_truth_cast_keep_native_value(tmp_path, native_unsigned_toolchain):
    flag, left, right = SSAValue(0,"bool"), SSAValue(1,"uint64"), SSAValue(2,"uint64")
    result, truth = SSAValue(3,"uint64"), SSAValue(4,"bool")
    function = Function("root", [flag,left,right], {"entry":BasicBlock("entry",[
        Instr("Select", [flag,left,right], result),
        Instr("Cast", [result], truth, attributes={"target_dtype":"bool"}),
        Instr("Ret", [result,truth], None),
    ])})
    core, _ = native(IRModule({"root":function}), tmp_path)
    for flag_value, expected in ((False,0),(True,(1<<64)-1)):
        execution = core.prepare_execution({0:np.array([flag_value],dtype=np.bool_),
            1:np.array([(1<<64)-1],dtype=np.uint64),2:np.array([0],dtype=np.uint64)}).run()
        assert execution.buffers[3].dtype == np.dtype("uint64")
        assert execution.buffers[3].item() == expected
        assert execution.buffers[4].item() == bool(expected)


@pytest.mark.parametrize("operation,source_dtype,target_dtype,value,expected", [
    ("UiToFp","int64","float64",-1,float((1<<64)-1)),
    ("ZExt","int8","int64",-1,255),
    ("SExt","uint8","int64",255,-1),
    ("FpToUi","float64","int32",float((1<<32)-1),-1),
])
def test_explicit_integer_conversion_opcode_owns_signedness(operation,source_dtype,target_dtype,value,expected,tmp_path,native_unsigned_toolchain):
    source, result = SSAValue(0,source_dtype), SSAValue(1,target_dtype)
    function=Function("root",[source],{"entry":BasicBlock("entry",[
        Instr(operation,[source],result), Instr("Ret",[result],None),
    ])})
    core, _ = native(IRModule({"root":function}),tmp_path)
    execution=core.prepare_execution({0:np.array([value],dtype=source_dtype)}).run()
    assert execution.buffers[1].dtype == np.dtype(target_dtype)
    assert execution.buffers[1].item() == expected
