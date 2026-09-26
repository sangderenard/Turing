"""Resident replacement must not destroy old contents on a failed copy."""
import pickle
import subprocess
import sys

import pytest

from src.compiler.ir_sequence_tables import lower_sequence_replace
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.transmogrifier.ssa import IRModule, SSASequenceDescriptor, SSASequenceTable


@pytest.mark.parametrize('self_alias', [False, True])
def test_native_keyed_replace_preserves_storage_and_failure_contents(tmp_path, self_alias):
    destination = SSASequenceDescriptor(10, (10, 11), 12, 13,
                                       column_dtypes=('int64', 'float64'), key_columns=(0,))
    source = destination if self_alias else SSASequenceDescriptor(
        20, (20, 21), 22, 23, column_dtypes=('int64', 'float64'), key_columns=(0,))
    lowering = lower_sequence_replace(destination, source, function_name='replace')
    assert lowering.complete, lowering.shortfalls
    function = lowering.functions[0]
    module = IRModule({'replace': function})
    module.sequence_tables['replace'] = SSASequenceTable(sequences={
        destination.sequence_id: destination, source.sequence_id: source})
    artifact = emit_ssa_module_to_c(module, 'replace')
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((artifact, self_alias, function.metadata['named_outputs'][0][1])))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, self_alias, status = pickle.load(open(sys.argv[1], 'rb'))
feeds = {10: np.array([3, 5, 7, 99], dtype=np.int64),
         11: np.array([1.5, 2.5, 3.5, 99.0]),
         12: np.array([3], dtype=np.int64), 13: np.array([4], dtype=np.int64)}
if not self_alias:
    feeds.update({20: np.array([8, 9, 10, 11, 12], dtype=np.int64),
                  21: np.array([8.5, 9.5, 10.5, 11.5, 12.5]),
                  22: np.array([2], dtype=np.int64), 23: np.array([5], dtype=np.int64)})
execution = artifact.prepare_execution(feeds)
b = execution.buffers
if self_alias:
    for _ in range(3):
        execution.run()
        assert b[12].item() == 3
        assert b[10][:3].tolist() == [3, 5, 7]
        assert b[11][:3].tolist() == [1.5, 2.5, 3.5]
        assert b[status].item() == 1
else:
    for length in (2, 0, 3, 5, -1, 2):
        before = (b[10].copy(), b[11].copy(), b[12].copy())
        b[22][0] = length
        execution.run()
        if length < 0 or length > 4:
            assert b[status].item() == 2
            for actual, expected in zip((b[10], b[11], b[12]), before):
                np.testing.assert_array_equal(actual, expected)
        else:
            assert b[status].item() == 1
            assert b[12].item() == length
            np.testing.assert_array_equal(b[10][:length], b[20][:length])
            np.testing.assert_array_equal(b[11][:length], b[21][:length])
    assert b[20].tolist() == [8, 9, 10, 11, 12]
    assert b[21].tolist() == [8.5, 9.5, 10.5, 11.5, 12.5]
    before = (b[10].copy(), b[11].copy(), b[12].copy())
    b[23][0] = 1  # A corrupt source extent must not read past its capacity.
    b[22][0] = 2
    execution.run()
    assert b[status].item() == 2
    for actual, expected in zip((b[10], b[11], b[12]), before):
        np.testing.assert_array_equal(actual, expected)
''', str(payload)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_replace_rejects_incompatible_key_layout():
    destination = SSASequenceDescriptor(10, (10, 11), 12, 13,
                                       column_dtypes=('int64', 'float64'), key_columns=(0,))
    source = SSASequenceDescriptor(20, (20, 21), 22, 23,
                                  column_dtypes=('int64', 'float64'))
    with pytest.raises(ValueError, match='matching row storage and key policy'):
        lower_sequence_replace(destination, source)
