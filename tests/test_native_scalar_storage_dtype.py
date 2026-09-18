import pickle
import subprocess
import sys
import pytest

from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule


@pytest.mark.parametrize('copied_operand', [False, True])
def test_scalar_bool_keeps_its_storage_type_when_helper_also_reads_arrays(tmp_path, copied_operand):
    data = SSAValue(0, 'float64', accounting={'physical_dtype': 'float64'})
    index, address, loaded = SSAValue(1, 'int64'), SSAValue(2, 'float64'), SSAValue(3, 'float64')
    read = Function('read', [data], {'entry': BasicBlock('entry', [
        Instr('Const', [], index, attributes={'value': 0}),
        Instr('GetElementPtr', [data, index], address),
        Instr('Load', [address], loaded), Instr('Ret', [loaded], None),
    ])})
    boolean, converted = SSAValue(0, 'bool'), SSAValue(1, 'float64')
    use = SSAValue(0, 'float64', accounting={'ssa_call_rank': 3}) if copied_operand else boolean
    bridge = Function('bridge', [boolean], {'entry': BasicBlock('entry', [
        Instr('Call', [use], converted, attributes={'callee': 'read'}),
        Instr('Ret', [converted], None),
    ])})
    array, flag = SSAValue(0, 'float64', (2,)), SSAValue(1, 'bool')
    first, second = SSAValue(2, 'float64'), SSAValue(3, 'float64')
    root = Function('root', [array, flag], {'entry': BasicBlock('entry', [
        Instr('Call', [array], first, attributes={'callee': 'read'}),
        Instr('Call', [flag], second, attributes={'callee': 'bridge'}),
        Instr('Ret', [first, second], None),
    ])})
    artifact = emit_ssa_to_c(IRModule({'root': root, 'bridge': bridge, 'read': read}), 'root')
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps(artifact))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact = pickle.load(open(sys.argv[1], 'rb'))
for flag in (False, True, False, True):
    execution = artifact.prepare_execution({0: np.array([200.0, 42.0]), 1: np.array([flag], dtype=np.bool_)}).run()
    assert execution.buffers[2].item() == 200.0
    assert execution.buffers[3].item() == float(flag), execution.buffers[3]
''', str(saved)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
