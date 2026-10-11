"""Child-copy indices have the same width on every loop edge (N4)."""
import pickle
import subprocess
import sys

import pytest

from src.compiler.ir_sequence_tables import lower_record_sequence_append_with_child_copy
from src.transmogrifier.ssa import (
    IRModule, SSASequenceDescriptor, SSASequenceTable, SSAChildTablePoolDescriptor,
)


@pytest.mark.parametrize('backend', ['c', 'llvm'])
@pytest.mark.parametrize('live_flags', [False, True])
def test_native_child_append_preserves_snapshot_and_capacity(tmp_path, backend, live_flags):
    pool = SSAChildTablePoolDescriptor(
        handle_column=2, column_value_ids=(90,), length_value_id=91,
        capacity_value_id=92, row_stride_value_id=93, status_value_id=94,
        column_dtypes=('int64',), key_columns=(),
        live_flags_value_id=95 if live_flags else None,
    )
    destination = SSASequenceDescriptor(
        80, (81, 82, 83), 84, 85,
        column_dtypes=('float64', 'bool', 'int64'), child_table_pool=pool,
        live_flags_value_id=86 if live_flags else None,
    )
    source = SSASequenceDescriptor(
        100, (101,), 102, 103, column_dtypes=('int64',),
        live_flags_value_id=104 if live_flags else None,
    )
    lowering = lower_record_sequence_append_with_child_copy(destination, source, function_name='append')
    assert lowering.complete, lowering.shortfalls
    fn = lowering.functions[0]
    phi = next(i for i in fn.blocks['copy_header'].instrs if i.op == 'Phi')
    assert phi.res.dtype == 'int64'
    assert {arg.dtype for arg in phi.args} == {'int64'}
    module = IRModule({'append': fn}, sequence_tables={
        'append': SSASequenceTable({80: destination, 100: source}),
    })
    if backend == 'llvm':
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact
        artifact = emit_ssa_function_to_llvm(module, 'append')
    else:
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c
        artifact = emit_ssa_module_to_c(module, 'append')
    assert artifact.complete, artifact.shortfalls
    if backend == 'llvm':
        artifact = compile_artifact(artifact, directory=tmp_path / 'native')
    else:
        artifact.compile(tmp_path / 'native')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, tuple(v.id for v in fn.args[-2:]),
                                   fn.metadata['named_outputs'][0][1], live_flags)))
    result = subprocess.run([sys.executable, '-u', '-c', '''
import pickle, sys
import numpy as np
artifact, row_ids, status, live = pickle.load(open(sys.argv[1], 'rb'))
feeds = {81: np.zeros(3), 82: np.zeros(3, dtype=np.bool_), 83: np.zeros(3, dtype=np.int64),
         84: np.array([0], dtype=np.int64), 85: np.array([3], dtype=np.int64),
         90: np.full(12, -1, dtype=np.int64), 91: np.zeros(3, dtype=np.int64),
         92: np.array([12], dtype=np.int64), 93: np.array([4], dtype=np.int64),
         94: np.full(3, -1, dtype=np.int32), 101: np.array([7, 9, 11, 13], dtype=np.int64),
         102: np.array([2], dtype=np.int64), 103: np.array([4], dtype=np.int64),
         row_ids[0]: np.array([3.5]), row_ids[1]: np.array([True], dtype=np.bool_)}
if live:
    feeds.update({86: np.zeros(3, dtype=np.bool_), 95: np.zeros(12, dtype=np.bool_),
                  104: np.array([True, False, True, True], dtype=np.bool_)})
if hasattr(artifact, 'prepare_execution'):
    execution = artifact.prepare_execution(feeds)
else:
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution
    execution = prepare_artifact_execution(artifact, feeds)
b = execution.buffers
for row, length in enumerate((2, 0, 3)):
    b[102][0] = length
    execution.run()
    assert b[status].item() == 1
    assert b[84].item() == row + 1
    assert b[91][row] == length and b[94][row] == 0
    assert b[83][row] == row and b[81][row] == 3.5 and b[82][row]
    np.testing.assert_array_equal(b[90][row*4:row*4+length], b[101][:length])
    if live:
        assert b[86][row]
        np.testing.assert_array_equal(b[95][row*4:row*4+length], b[104][:length])
snapshot = b[90].copy()
b[101][:] = 123
execution.run()
assert b[status].item() == 2 and b[84].item() == 3
np.testing.assert_array_equal(b[90], snapshot)
''', str(saved)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
