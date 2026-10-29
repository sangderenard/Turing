"""Wide arithmetic is internal; public buffers retain their portable ABI."""
import numpy as np
import pytest

from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule


def _module(dtype, boundary=False):
    left, right = SSAValue(0, "int64"), SSAValue(1, "uint64")
    a, b, total, expected = [SSAValue(i, dtype) for i in range(2, 6)]
    equal, result = SSAValue(6, "bool"), SSAValue(7, "int64")
    signed = dtype == "int128"
    code = [Instr("Cast", [left], a),
            (Instr("Const", [], b, attributes={"value": 2**(127 if signed else 128) - 1})
             if boundary else Instr("Cast", [right], b)),
            Instr("Add", [a, b], total),
            Instr("Const", [], expected, attributes={"value":
                (-2**127 if signed else 0) if boundary else 2**64 - 2}),
            Instr("Eq", [total, expected], equal), Instr("Cast", [equal], result),
            Instr("Ret", [result], None)]
    return IRModule({"root": Function("root", [left, right], {"entry": BasicBlock("entry", code)})}), result


@pytest.mark.parametrize("lane", ["C", "Fortran"])
@pytest.mark.parametrize("dtype", ["int128", "uint128"])
@pytest.mark.parametrize("boundary", [False, True], ids=["mixed-64-inputs", "wide-wrap-boundary"])
def test_internal_wide_add_and_comparison_native(tmp_path, lane, dtype, boundary):
    module, result = _module(dtype, boundary)
    if lane == "C":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, "root")
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path)
        prepare = artifact.prepare_execution
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        if fortran_compiler() is None:
            pytest.skip("native Fortran compiler unavailable")
        artifact = emit_module(module, outputs={"root": [result]}, extra_roots=("root",), progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        native = compile_fortran_module_c_shell(artifact, {}, tmp_path, entrypoint="root", library=True)
        prepare = FortranCoreNative(native.executable_path, artifact.api.entry_point("root"), artifact.api).prepare_execution
    for left, expected in (((1, 1), (0, 0)) if boundary else ((-1, 1), (0, 0))):
        execution = prepare({0: np.array([left], dtype=np.int64),
                             1: np.array([2**64 - 1], dtype=np.uint64)})
        assert execution.run().buffers[result.id].item() == expected


@pytest.mark.parametrize("lane", ["C", "Fortran"])
@pytest.mark.parametrize("dtype", ["int128", "uint128"])
@pytest.mark.parametrize("surface", ["input", "output"])
def test_wide_integer_public_export_refuses(lane, dtype, surface):
    value = SSAValue(0, dtype)
    inputs = [value] if surface == "input" else []
    instructions = [] if inputs else [Instr("Const", [], value, attributes={"value": 0})]
    instructions.append(Instr("Ret", [value], None))
    module = IRModule({"root": Function("root", inputs, {"entry": BasicBlock("entry", instructions)})})
    if lane == "C":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        with pytest.raises(ValueError, match="no public C buffer ABI"):
            emit_ssa_to_c(module, "root")
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranEmissionError
        with pytest.raises(FortranEmissionError, match="no public Fortran buffer ABI"):
            emit_module(module, outputs={"root": [value]}, extra_roots=("root",), progress=lambda _: None)


@pytest.mark.parametrize("dtype,width", [("int8", 8), ("int16", 16), ("int32", 32), ("int64", 64)])
@pytest.mark.parametrize("operation", ["Add", "Sub", "Mul"])
def test_fortran_signed_arithmetic_wraps_at_every_catalogue_width(tmp_path, dtype, width, operation):
    from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
    from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
    if fortran_compiler() is None:
        pytest.skip("native Fortran compiler unavailable")
    maximum, minimum = 2**(width - 1) - 1, -2**(width - 1)
    constant, runtime, wanted = ((minimum, 1, maximum) if operation == "Sub" else
                                 (maximum, 2, -2) if operation == "Mul" else
                                 (maximum, 1, minimum))
    argument = SSAValue(0, dtype)
    bound, arithmetic, expected = [SSAValue(i, dtype) for i in range(1, 4)]
    equal, result = SSAValue(4, "bool"), SSAValue(5, "int64")
    module = IRModule({"root": Function("root", [argument], {"entry": BasicBlock("entry", [
        Instr("Const", [], bound, attributes={"value": constant}),
        Instr(operation, [bound, argument], arithmetic),
        Instr("Const", [], expected, attributes={"value": wanted}),
        Instr("Eq", [arithmetic, expected], equal), Instr("Cast", [equal], result),
        Instr("Ret", [result], None),
    ])})})
    artifact = emit_module(module, outputs={"root": [result]}, extra_roots=("root",), progress=lambda _: None)
    assert artifact.complete, artifact.shortfalls
    native = compile_fortran_module_c_shell(artifact, {}, tmp_path, entrypoint="root", library=True)
    execution = FortranCoreNative(native.executable_path, artifact.api.entry_point("root"), artifact.api).prepare_execution(
        {argument.id: np.array([runtime], dtype=dtype)},
    )
    assert execution.run().buffers[result.id].item() == 1
