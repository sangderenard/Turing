"""Explicit fixed caller arenas retain the incumbent validation-75 boundary."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def _exercise(backend, directory, exhaustion):
    from src.compiler.control_source import ControlProgram, SequenceBlock, SequenceMutationBlock, ControlSequenceMutation
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa, _sequence_artifacts_from_control, Meta
    from src.compiler.ssa_reachability import hoist_nondominating_constants
    from src.transmogrifier.ssa import IRModule
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book
    book, token = begin_identity_book()
    program = ControlProgram(SequenceBlock((
        SequenceMutationBlock(ControlSequenceMutation(0, 'setdefault', (1, 5), 3,
            policy='unique', argument_kind='mapping_setdefault')),
        SequenceMutationBlock(ControlSequenceMutation(3, 'append', (2,), 4, policy='duplicates')),
    )))
    function, shortfalls = lower_control_program_to_ssa(program, function_name='fixed_selected_child',
        first_value_id=20, sequence_declarations=((0, 'unique', 2, True),),
        sequence_column_dtypes={0: ('int64', 'int64')},
        sequence_child_table_layouts={0: (1, ('int64',), (), True)}, nested_sequence_ids=(0,),
        region_value_meta={0: Meta((4,), 'int64'), 1: Meta((), 'int64'), 2: Meta((), 'int64')},
        parameter_names=('table', 'key', 'value'),
        value_name_histories={'table': (0,), 'key': (1,), 'value': (2,)})
    assert not shortfalls
    helpers, tables = _sequence_artifacts_from_control(function)
    module = IRModule(functions={function.name: function, **helpers}, sequence_tables=tables,
        metadata={'identity_book': book})
    for member in module.functions.values():
        hoist_nondominating_constants(member)
    from src.compiler.ir_identities import apply_numeric_feature_pipeline, apply_precision_pipeline
    apply_numeric_feature_pipeline(module)
    apply_precision_pipeline(module)
    from src.compiler.ssa_call_input_adapters import adapt_physical_call_inputs, physical_call_input_conflicts
    adapt_physical_call_inputs(module.functions)
    assert not physical_call_input_conflicts(module.functions)
    end_identity_book(token)
    descriptor = tables[function.name].by_id(0)
    pool = descriptor.child_table_pool
    assert descriptor.capacity_policy.value == 'fixed' and descriptor.heap_owner_value_id is None
    feeds = {1: np.array([3], dtype='int64'), 2: np.array([99], dtype='int64')}
    for value_id in descriptor.column_value_ids:
        feeds[value_id] = np.full(4, 123, dtype='int64')
    feeds[descriptor.length_address_id] = np.array([0], dtype='int64')
    feeds[descriptor.capacity_value_id] = np.array([0 if exhaustion == 'outer' else 4], dtype='int64')
    # Lookup publishes absence status0 before the allocation guard; start
    # at that incumbent status so the no-mutation check covers storage.
    feeds[descriptor.status_address_id] = np.zeros(1, dtype='int32')
    feeds[pool.column_value_ids[0]] = np.full(16, 111, dtype='int64')
    feeds[pool.length_value_id] = np.full(4, 77, dtype='int64')
    feeds[pool.capacity_value_id] = np.array([3 if exhaustion == 'pool' else 16], dtype='int64')
    feeds[pool.row_stride_value_id] = np.array([4], dtype='int64')
    feeds[pool.status_value_id] = np.full(4, 91, dtype='int32')
    if backend == 'c':
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, function.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(directory)
        execution = artifact.prepare_execution(feeds)
    elif backend == 'llvm':
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        artifact = emit_ssa_function_to_llvm(module, function.name)
        assert artifact.complete, artifact.shortfalls
        execution = prepare_artifact_execution(compile_artifact(artifact, directory=directory), feeds)
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        if fortran_compiler() is None:
            pytest.skip('Fortran compiler unavailable')
        artifact = emit_module(module, name='fixed_pool', outputs={function.name: ()},
            extra_roots=(function.name,), progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(artifact, {}, directory, entrypoint=function.name, library=True)
        execution = FortranCoreNative(built.executable_path, artifact.api.entry_point(function.name), artifact.api).prepare_execution(feeds)
    before = {key: execution.buffers[key].copy() for key in feeds}
    print('BOUNDS_EXECUTION_BEGIN', backend, flush=True)
    if exhaustion:
        if backend == 'fortran':
            execution.run()
            raise AssertionError('missing bounds refusal')
        with pytest.raises(RuntimeError, match='75'):
            execution.run()
        for key, value in before.items():
            assert np.array_equal(execution.buffers[key], value), key
        print('BOUNDS_NO_MUTATION', backend, flush=True)
    else:
        execution.run()
        assert execution.buffers[descriptor.length_address_id].item() == 1
        assert execution.buffers[pool.length_value_id][0] == 1
        assert execution.buffers[pool.column_value_ids[0]][0] == 99
        assert execution.buffers[pool.column_value_ids[0]][1] == 111


@pytest.mark.parametrize('backend', ['c', 'llvm', 'fortran'])
@pytest.mark.parametrize('exhaustion', [None, 'pool', 'outer'])
def test_fixed_caller_child_pool_bounds(backend, exhaustion, tmp_path):
    command = ('from pathlib import Path; from tests.test_fixed_child_pool_bounds import _exercise; '
               f'_exercise({backend!r}, Path({str(tmp_path)!r}), {exhaustion!r})')
    result = subprocess.run([sys.executable, '-u', '-c', command],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    if backend == 'fortran' and exhaustion:
        assert result.returncode != 0, result.stdout + result.stderr
        assert 'BOUNDS_EXECUTION_BEGIN fortran' in result.stdout
        assert 'ERROR STOP 75' in result.stderr, result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
