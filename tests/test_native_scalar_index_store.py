import pickle
import subprocess
import sys

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference


def test_repository_tensor_provider_preserves_scalar_index_assignment(tmp_path):
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file('extraction_contracts/vehicle_full_native_execution.yaml').with_program_abi({
        'records': {}, 'bindings': [], 'values': [{
            'function': 'root', 'parameter': 'array', 'storage': 'span',
            'python_type': 'numpy.ndarray', 'dtype': 'float64',
            'rank': 1, 'shape': [2], 'mutable': True,
        }],
    })
    module, _, exports = lower_ast_source_to_ssa(
        'def root(array, value):\n    array[0] = value\n    return array\n',
        'root', name='scalar_index_store', extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    parameters = dict(module.functions[exports[0]].metadata['parameter_names'])
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, parameters)))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, parameters = pickle.load(open(sys.argv[1], 'rb'))
for value in (-3.5, 0.0, 12.0):
    execution = artifact.prepare_execution({
        parameters['array']: np.array([91.0, 42.0]),
        parameters['value']: np.array([value]),
    }).run()
    np.testing.assert_array_equal(execution.buffers[parameters['array']], [value, 42.0])
''', str(saved)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
