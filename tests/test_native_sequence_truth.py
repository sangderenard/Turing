"""A sequence's truth value observes mutations, never its first element."""

import pickle
import subprocess
import sys

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c


def test_conditional_append_of_zero_has_native_sequence_truth(tmp_path):
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file('extraction_contracts/vehicle_full_native_execution.yaml')
    module, _, exports = lower_ast_source_to_ssa('''
def root(flag):
    reasons = []
    if flag:
        reasons.append(0.0)
    return bool(reasons)
''', 'root', name='sequence_truth', extraction_contract=policy)
    artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    root = module.functions[exports[0]]
    flag = next(value_id for name, value_id in root.metadata['parameter_names'] if name == 'flag')
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, flag, output)))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, flag, output = pickle.load(open(sys.argv[1], 'rb'))
execution = artifact.prepare_execution({flag: np.array([0.0])})
# Reuse the same storage: local sequence length must reset each invocation.
for enabled in (False, True, False, True):
    execution.buffers[flag][0] = enabled
    execution.run()
    assert execution.buffers[output].item() == enabled, (enabled, execution.buffers[output])
''', str(saved)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
