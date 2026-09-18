"""A continue on a rejected attempt precedes every accept-side mutation."""

import pickle
import subprocess
import sys

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference


def test_rejected_attempt_never_calls_accept_mutation(tmp_path):
    source = '''
class Material:
    def accept(self):
        self.state[0] = self.state[0] + 1.0

def root(material, reject, count):
    for attempt in range(3):
        if attempt >= count:
            break
        if reject:
            continue
        material.accept()
    return material.state[0]
'''
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({
        'records': {'Material': {'identity': 'Material', 'fields': {
            'state': {'storage': 'span', 'dtype': 'float64', 'rank': 1,
                      'shape': [2], 'mutable': True},
        }}},
        'bindings': [{'function': 'root', 'parameter': 'material', 'record': 'Material'}],
        'values': [{'function': 'root', 'parameter': 'reject', 'storage': 'scalar',
                    'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'},
                   {'function': 'root', 'parameter': 'count', 'storage': 'scalar',
                    'dtype': 'int64', 'rank': 0, 'python_type': 'builtins.int'}],
    })
    module, _, _ = lower_ast_source_to_ssa(
        source, 'root', name='rejection_effect_order', extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    root = module.functions['rejection_effect_order__root']
    names = dict(root.metadata['parameter_names'])
    assert 'reject' in names, 'The authored rejection guard must remain live'
    reject = names['reject']
    state = next(v.id for v in root.args if v.accounting.get('program_abi_field') == 'state')
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, state, reject, names['count'], source)))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, state, reject, count, source = pickle.load(open(sys.argv[1], 'rb'))
namespace = {}
exec(source, namespace)
execution = artifact.prepare_execution({state: np.array([2.0, 3.0]),
                                       reject: np.array([False], dtype=np.bool_),
                                       count: np.array([0], dtype=np.int64)})
for rejected, iterations in ((False, 0), (False, 1), (True, 3), (False, 3), (True, 0)):
    material = namespace['Material']()
    material.state = np.array([2.0, 3.0])
    namespace['root'](material, rejected, iterations)
    execution.buffers[state][:] = [2.0, 3.0]
    execution.buffers[reject][0] = rejected
    execution.buffers[count][0] = iterations
    execution.run()
    actual = execution.buffers[state].tolist()
    assert actual == material.state.tolist(), (rejected, actual, material.state)
''', str(saved)], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stdout + probe.stderr
