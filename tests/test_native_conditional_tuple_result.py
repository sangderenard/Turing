import pickle
import runpy
import subprocess
import sys
import numpy as np
import pytest
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize('call_arms', [False, True])
def test_conditional_tuple_members_reach_native_consumer(tmp_path, call_arms):
    lower = runpy.run_path('tests/test_aggregate_call_identity.py')['_lower']
    source = '''
def capture(array):
    return (array.copy(), array[0, 0])
def recover(saved):
    return saved[0] * saved[1]
def tick(left, right, enabled):
    saved = ''' + ('capture(left) if enabled else capture(right)' if call_arms
                   else '(left, left[0, 0]) if enabled else (right, right[0, 0])') + '''
    return recover(saved)
'''
    module, outputs, _ = lower(source, 'tuple_merge', {
        'left': np.zeros((2, 3)), 'right': np.ones((2, 3)), 'enabled': True,
    })
    root = module.functions['tuple_merge__tick']
    names = dict(root.metadata['parameter_names'])
    result = outputs[root.name][0]
    assert len(root.args) == 3
    assert tuple(result.shape) == (2, 3)
    merges = [i for b in root.blocks.values() for i in b.instrs if i.op == 'Phi']
    assert sorted(tuple(merge.res.shape) for merge in merges) == [(), (2, 3)]
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'tuple_merge')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((artifact, names, result.id, source)))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
with open(sys.argv[1], 'rb') as stream:
    artifact, names, result_id, source = pickle.load(stream)
scope = {}
exec(source, scope)
left = np.array([[2.0, -3.0, 4.0], [11.0, 17.0, -23.0]])
right = np.array([[-7.0, 9.0, 12.0], [31.0, -41.0, 53.0]])
for enabled in (False, True):
    expected = scope['tick'](left.copy(), right.copy(), enabled)
    result = artifact.prepare_execution({
        names['left']: left.copy(), names['right']: right.copy(),
        names['enabled']: np.array([enabled], dtype=np.bool_),
    }).run()
    actual = np.asarray(result.buffers[result_id])
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.array_equal(actual, expected), (enabled, actual, expected)
''', str(payload)], capture_output=True, text=True, timeout=20)
    assert probe.returncode == 0, probe.stdout + probe.stderr
