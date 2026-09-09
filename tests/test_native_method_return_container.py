import pickle
import subprocess
import sys

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def test_record_method_tuple_result_preserves_member_natively(tmp_path):
    source = '''
class Material:
    def capture(self):
        return (self.state.copy(),)
    def recover(self, saved):
        return saved[0]

def root(material):
    saved = material.capture()
    return material.recover(saved)
'''
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({
        'records': {'Material': {'identity': 'Material', 'fields': {
            'state': {'storage': 'span', 'dtype': 'float64', 'rank': 2,
                      'shape': [2, 3], 'mutable': True},
        }}},
        'bindings': [{'function': 'root', 'parameter': 'material', 'record': 'Material'}],
        'values': [],
    })
    module, outputs, _ = lower_ast_source_to_ssa(
        source, 'root', name='method_tuple', extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    root = module.functions['method_tuple__root']
    result = outputs[root.name][0]
    assert len(root.args) == 1
    assert root.args[0].accounting['program_abi_field'] == 'state'
    assert result.dtype == 'float64'
    assert tuple(result.shape) == (2, 3)
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'method_tuple')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((artifact, root.args[0].id, result.id)))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
with open(sys.argv[1], 'rb') as stream:
    artifact, state_id, result_id = pickle.load(stream)
expected = np.array([[2.0, -3.0, 4.0], [11.0, 17.0, -23.0]])
result = artifact.prepare_execution({state_id: expected.copy()}).run()
actual = np.asarray(result.buffers[result_id])
assert actual.shape == expected.shape, (actual.shape, expected.shape)
assert np.array_equal(actual, expected), (actual, expected)
''', str(payload)], capture_output=True, text=True, timeout=20)
    assert probe.returncode == 0, probe.stdout + probe.stderr
