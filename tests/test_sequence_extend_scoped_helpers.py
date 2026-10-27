"""Nested sequence helpers retain the owning operation's function identity."""
import numpy as np
import pytest

from src.compiler.ir_sequence_tables import lower_sequence_extend
from src.transmogrifier.ssa import IRModule, SSASequenceDescriptor


def _two_scopes():
    first = lower_sequence_extend(
        SSASequenceDescriptor(75, (62,), 63, 64, column_dtypes=("int64",)),
        SSASequenceDescriptor(39, (39,), 40, 41, column_dtypes=("int64",)),
        function_name="owner_a_extend",
    )
    second = lower_sequence_extend(
        SSASequenceDescriptor(75, (62,), 73, 74, column_dtypes=("int64",)),
        SSASequenceDescriptor(39, (39,), 50, 51, column_dtypes=("int64",)),
        function_name="owner_b_extend",
    )
    assert first.complete and second.complete
    return first, second


def test_nested_insert_has_owning_function_identity():
    first, second = _two_scopes()
    assert not ({function.name for function in first.functions}
                & {function.name for function in second.functions})


@pytest.mark.parametrize("lane", ["C", "LLVM", "Fortran"])
def test_two_scoped_extend_helpers_execute_native(tmp_path, lane):
    first, second = _two_scopes()
    module = IRModule({function.name: function
                       for lowering in (first, second) for function in lowering.functions})
    for number, lowering in enumerate((first, second)):
        entry = lowering.entry
        values = (2**53 + 1 + number * 2, 7 + number)
        feeds = {
            entry.args[0].id: np.array([99, 0, 0, 77], dtype=np.int64),
            entry.args[1].id: np.array([1], dtype=np.int64),
            entry.args[2].id: np.array([4], dtype=np.int64),
            entry.args[3].id: np.array(values, dtype=np.int64),
            entry.args[4].id: np.array([2], dtype=np.int64),
            entry.args[5].id: np.array([2], dtype=np.int64),
        }
        if lane == "C":
            from src.compiler.ssa_c_backend import emit_ssa_to_c
            artifact = emit_ssa_to_c(module, entry.name)
            assert artifact.complete, artifact.shortfalls
            artifact.compile(tmp_path / str(number))
            prepare = artifact.prepare_execution
        elif lane == "LLVM":
            from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
            artifact = emit_ssa_function_to_llvm(module, entry.name)
            assert artifact.complete, artifact.shortfalls
            native = compile_artifact(artifact, directory=tmp_path / str(number))
            prepare = lambda supplied: prepare_artifact_execution(native, supplied)
        else:
            from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
            from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
            if fortran_compiler() is None:
                pytest.skip("native Fortran compiler unavailable")
            returned = next(instruction.args for block in entry.blocks.values()
                            for instruction in block.instrs if instruction.op == "Ret")
            artifact = emit_module(module, outputs={entry.name: returned},
                                   extra_roots=(entry.name,), progress=lambda _: None)
            assert artifact.complete, artifact.shortfalls
            native = compile_fortran_module_c_shell(artifact, {}, tmp_path / str(number),
                                                   entrypoint=entry.name, library=True)
            prepare = FortranCoreNative(native.executable_path,
                                       artifact.api.entry_point(entry.name),
                                       artifact.api).prepare_execution
        result = prepare(feeds).run().buffers
        assert result[entry.args[0].id].tolist() == [99, *values, 77]
        assert result[entry.args[1].id].item() == 3
