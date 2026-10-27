"""Native proofs for the existing context-owned, typed child-pool ABI."""
import ctypes

import numpy as np
import pytest

from src.compiler.ir_sequence_tables import lower_sequence_allocate, lower_sequence_reserve
from src.compiler.ssa_heap_allocation import heap_release
from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAChildTablePoolDescriptor, SSASequenceCapacityPolicy, SSASequenceDescriptor, SSAValue


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
def test_owned_pool_growth_preserves_handles_rows_and_context(backend, tmp_path):
    pool = SSAChildTablePoolDescriptor(1, (20,), 21, 22, 23,
        status_value_id=24, column_dtypes=("int64",), key_columns=(), writable=True)
    descriptor = SSASequenceDescriptor(0, (10, 11), 2, 3, status_address_id=4,
        column_dtypes=("int64", "int64"), key_columns=(0,), child_table_pool=pool,
        heap_owner_value_id=6, capacity_policy=SSASequenceCapacityPolicy.DYNAMIC)
    initialize = lower_sequence_allocate(descriptor).entry
    reserve = lower_sequence_reserve(descriptor).entry
    owner = SSAValue(6, "ptr", shape=(1,), accounting={"heap_owner": True})
    release = Function("release", [owner], {"entry": BasicBlock("entry", [heap_release(owner), Instr("Ret", [], None)])})
    module = IRModule(functions={function.name: function for function in (initialize, reserve, release)})
    natives = {}
    def prepare(native, feeds):
        return prepare_artifact_execution(native, feeds) if backend == "llvm" else native.prepare_execution(feeds)
    for function in (initialize, reserve, release):
        if backend == "c":
            from src.compiler.ssa_c_backend import emit_ssa_to_c
            artifact = emit_ssa_to_c(module, function.name)
            assert artifact.complete, artifact.shortfalls
            artifact.compile(tmp_path / function.name)
            natives[function.name] = artifact
        elif backend == "llvm":
            artifact = emit_ssa_function_to_llvm(module, function.name)
            assert artifact.complete, artifact.shortfalls
            natives[function.name] = compile_artifact(artifact, directory=tmp_path / function.name)
        else:
            from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
            from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
            if fortran_compiler() is None:
                pytest.skip("Fortran compiler unavailable")
            result_values = next(instruction.args for block in function.blocks.values()
                                 for instruction in block.instrs if instruction.op == "Ret")
            artifact = emit_module(module, name="owned_pool", outputs={function.name: tuple(result_values)}, extra_roots=(function.name,), progress=lambda _: None)
            assert artifact.complete, artifact.shortfalls
            built = compile_fortran_module_c_shell(artifact, {}, tmp_path / function.name, entrypoint=function.name, library=True)
            natives[function.name] = FortranCoreNative(built.executable_path, artifact.api.entry_point(function.name), artifact.api)
    buffers = {value_id: np.zeros(1, dtype="uint64") for value_id in (6, 10, 11, 20, 21, 24)}
    buffers.update({value_id: np.zeros(1, dtype="int64") for value_id in (2, 3, 22, 23)})
    required = reserve.metadata["required_capacity_value_id"]
    child_required = reserve.metadata["required_child_capacity_value_id"]
    status = dict(reserve.metadata["named_outputs"])["status"]
    def rows(value_id, ctype, count):
        return (ctype * count).from_address(int(buffers[value_id][0]))
    try:
        prepare(natives[initialize.name], buffers).run()
        for outer, stride in ((1, 1), (2, 1), (2, 3), (5, 9)):
            buffers[required] = np.array([outer], dtype="int64")
            buffers[child_required] = np.array([stride], dtype="int64")
            old_length = int(buffers[2][0])
            execution = prepare(natives[reserve.name], buffers).run()
            assert execution.buffers[status].item() == 1, (outer, stride, {key: buffers[key].tolist() for key in (2, 3, 22, 23, required, child_required)})
            assert buffers[2][0] == old_length
            assert buffers[22][0] == buffers[3][0] * buffers[23][0]
            actual_stride = int(buffers[23][0])
            data = rows(20, ctypes.c_int64, int(buffers[22][0]))
            lengths = rows(21, ctypes.c_int64, int(buffers[3][0]))
            keys, handles = rows(10, ctypes.c_int64, int(buffers[3][0])), rows(11, ctypes.c_int64, int(buffers[3][0]))
            for row in range(old_length):
                assert keys[row] == row + 77 and handles[row] == row
                assert lengths[row] == 1
                assert data[row * actual_stride] == 2**60 + 3 + row
            for row in range(min(outer, 2)):
                keys[row], handles[row], lengths[row] = row + 77, row, 1
                data[row * actual_stride] = 2**60 + 3 + row
            buffers[2][0] = min(outer, 2)
        for outer, stride in ((-1, 1), (2, 2**63-1)):
            before = {value_id: value.copy() for value_id, value in buffers.items()
                      if value_id not in (required, child_required)}
            buffers[required] = np.array([outer], dtype="int64")
            buffers[child_required] = np.array([stride], dtype="int64")
            execution = prepare(natives[reserve.name], buffers).run()
            assert execution.buffers[status].item() == 2
            assert all(np.array_equal(buffers[value_id], value) for value_id, value in before.items())
        # The flattened bound is part of the same descriptor. A conflicting
        # public bound cannot authorize a row copy beyond that bound.
        buffers[required] = np.array([2], dtype="int64")
        buffers[child_required] = np.array([1], dtype="int64")
        buffers[22][0] = 0
        before = {value_id: value.copy() for value_id, value in buffers.items()}
        execution = prepare(natives[reserve.name], buffers).run()
        assert execution.buffers[status].item() == 2
        assert all(np.array_equal(buffers[value_id], value) for value_id, value in before.items())
    finally:
        prepare(natives[release.name], {6: buffers[6]}).run()
    assert buffers[6][0] == 0
