import pickle
import subprocess
import sys

from src.compiler.fortran_c_shell import _retain_forwarded_aggregate_storage
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule


def test_forwarded_field_frame_keeps_call_effects_without_new_results(tmp_path):
    left, right = SSAValue(100, 'float64'), SSAValue(101, 'float64')
    constant = SSAValue(102, 'float64')
    callee = Function('forward', [left, right], {'entry': BasicBlock('entry', [
        Instr('Const', [], constant, attributes={'value': 17.0}),
        Instr('Store', [constant, left], None),
        Instr('Ret', [left, right], None),
    ])})
    first, second = SSAValue(0, 'float64'), SSAValue(1, 'float64')
    call = Instr('Call', [first, second], SSAValue(3, 'ssa.aggregate'), attributes={
        'callee': 'forward', 'callee_input_ids': (100, 101),
        'callee_output_ids': (100, 101), 'output_ids': (10, 11),
        'result_convention': 'ssa.aggregate',
    })
    # Equal arity alone does not establish shared caller storage.
    assert not _retain_forwarded_aggregate_storage(call, callee)
    call.attributes['output_ids'] = (0, 1)
    assert _retain_forwarded_aggregate_storage(call, callee)
    assert call.res is None
    assert call.attributes['forwarded_output_bindings'] == ((100, 0), (101, 1))
    output = SSAValue(2, 'float64')
    root = Function('root', [first, second], {'entry': BasicBlock('entry', [
        call, Instr('Sub', [first, second], output), Instr('Ret', [output], None),
    ])})
    artifact = emit_ssa_to_c(IRModule({'root': root, 'forward': callee}), 'root')
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps(artifact))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact = pickle.load(open(sys.argv[1], 'rb'))
execution = artifact.prepare_execution({0: np.array([3.0]), 1: np.array([2.0])}).run()
assert execution.buffers[0].item() == 17.0
assert execution.buffers[2].item() == 15.0
''', str(saved)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
