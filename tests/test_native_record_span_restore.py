import pickle
import subprocess
import sys

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference


def test_record_span_restore_writes_full_slice(tmp_path):
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
        'values': [{'function': 'root', 'parameter': 'source', 'storage': 'span',
                    'dtype': 'float64', 'rank': 2, 'shape': [2, 3],
                    'python_type': 'numpy.ndarray'}],
    })
    module, _, _ = lower_ast_source_to_ssa(
        'def root(material, source):\n    material.state[...] = source\n    return 0.0\n',
        'root', name='span_restore', extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    root = module.functions['span_restore__root']
    state = next(argument.id for argument in root.args
                 if argument.accounting.get('program_abi_field') == 'state')
    source = dict(root.metadata['parameter_names'])['source']
    assert {argument.id for argument in root.args} == {source, state}
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'span_restore')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((artifact, state, source)))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
with open(sys.argv[1], 'rb') as stream:
    artifact, state_id, source_id = pickle.load(stream)
expected = np.array([[2.0, -3.0, 4.0], [11.0, 17.0, -23.0]])
result = artifact.prepare_execution({state_id: np.zeros((2, 3)), source_id: expected.copy()}).run()
actual = np.asarray(result.buffers[state_id])
assert np.array_equal(actual, expected), (actual, expected)
assert np.array_equal(result.buffers[source_id], expected)
''', str(payload)], capture_output=True, text=True, timeout=20)
    assert probe.returncode == 0, probe.stdout + probe.stderr
