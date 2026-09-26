import pickle
import subprocess
import sys

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def test_record_span_copy_preserves_shape(tmp_path):
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
        'def root(material):\n    return material.state.copy()\n',
        'root', name='span_copy', extraction_contract=policy,
    )
    root = module.functions['span_copy__root']
    result = outputs[root.name][0]
    assert tuple(result.shape) == (2, 3)
    state = next(argument.id for argument in root.args
                 if argument.accounting.get('program_abi_field') == 'state')
    assert len(root.args) == 1
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'span_copy')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((artifact, state, result.id)))
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
