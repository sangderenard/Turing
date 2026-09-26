import pickle
import subprocess
import sys
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c


@pytest.mark.parametrize('repository_provider', [False, True])
@pytest.mark.parametrize('store_result', [False, True])
def test_boolean_local_is_produced_before_numeric_region_consumer(tmp_path, repository_provider, store_result):
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
    options = ({'tensor_ssa_reference': c_backend_repository_ssa_reference()}
               if repository_provider else {})
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file('extraction_contracts/vehicle_full_native_execution.yaml')
    source = '''
def root(advanced, duration, hard_failure):
    completed = float(advanced) >= float(duration) - 1.0e-15 and not bool(hard_failure)
    return float(completed)
'''
    if store_result:
        policy = policy.with_program_abi({'records': {}, 'bindings': [], 'values': [{
            'function': 'root', 'parameter': 'telemetry', 'storage': 'span',
            'dtype': 'float64', 'rank': 1, 'shape': [2], 'mutable': True,
            'python_type': 'numpy.ndarray',
        }]})
        source = source.replace('hard_failure):', 'hard_failure, telemetry):').replace(
            'return float(completed)', 'telemetry[0] = float(completed)\n    return telemetry')
    module, _, exports = lower_ast_source_to_ssa(source, 'root', name='completed_window', extraction_contract=policy, **options)
    root = module.functions[exports[0]]
    parameters = dict(root.metadata['parameter_names'])
    assert {value.id for value in root.args} == set(parameters.values())
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, parameters, output, store_result)))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, parameters, output, store_result = pickle.load(open(sys.argv[1], 'rb'))
for advanced in (0.0, 0.25, 0.5):
    for failure in (0.0, 1.0):
        feeds = {
            parameters['advanced']: np.array([advanced]),
            parameters['duration']: np.array([0.25]),
            parameters['hard_failure']: np.array([failure]),
        }
        if store_result:
            feeds[parameters['telemetry']] = np.array([91.0, 42.0])
        execution = artifact.prepare_execution(feeds).run()
        expected = float(advanced >= 0.25 - 1e-15 and not bool(failure))
        if store_result:
            np.testing.assert_array_equal(execution.buffers[output], [expected, 42.0])
        else:
            assert execution.buffers[output].item() == expected, (advanced, failure, execution.buffers[output])
''', str(saved)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
