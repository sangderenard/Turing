import pickle
import subprocess
import sys

from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c


def test_nested_native_call_reads_current_loop_carried_value(tmp_path):
    policy = ExtractionContract('extraction_contracts/program_extraction.yaml').with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml')
    source = '''
def next_cap(value):
    return value * 2.0

def root(dt):
    cap = AbstractTensor.tensor(dt * 0.25)
    total = AbstractTensor.tensor(0.0)
    while total < dt:
        used = cap
        proposal = next_cap(cap)
        total = total + used
        cap = proposal
    return total
'''
    module, _, exports = lower_ast_source_to_ssa(
        source, 'root', name='carried_call', extraction_contract=policy,
        python_bindings={'AbstractTensor': AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference())
    root = module.functions[exports[0]]
    parameter = dict(root.metadata['parameter_names'])['dt']
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    artifact = emit_ssa_to_c(module, exports[0], watch=(output,))
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, parameter, output)))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, parameter, output = pickle.load(open(sys.argv[1], 'rb'))
for dt in (0.5, 1.0, 2.0):
    execution = artifact.prepare_execution({parameter: np.array([dt])}).run()
    # Three steps use dt/4, dt/2, dt. Reusing the initial call argument
    # instead produces dt/4, dt/2, dt/2, which sums to the wrong total.
    assert execution.buffers[output].item() == dt * 1.75, execution.buffers[output]
''', str(saved)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
