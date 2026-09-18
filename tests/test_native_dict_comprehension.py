"""One compiled artifact exercises dictionary row publication and overwrite."""
import pickle
import subprocess
import sys

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def test_native_dictionary_comprehensions_publish_owned_rows(tmp_path):
    source = '''
def root(scale, keys):
    values = {key: item * scale for key, item in ((1, 1), (2, 2), (1, 4), (3, 3))}
    empty = {key: key * scale for key in ()}
    filtered = {key: key * scale for key in (1, 2, 3) if key != 2}
    runtime = {key: key * scale for key in keys if key > 0}
    return values.get(1, -1.0), values.get(3, -1.0), empty.get(1, -1.0), filtered.get(2, -1.0), filtered.get(3, -1.0), runtime.get(1, -1.0), runtime.get(3, -1.0)
'''
    policy = ExtractionContract('extraction_contracts/program_extraction.yaml').with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml').with_program_abi({
            'bindings': [], 'values': [{'function': 'root', 'parameter': 'scale',
                'storage': 'scalar', 'dtype': 'float64', 'rank': 0, 'python_type': 'builtins.float'},
                {'function': 'root', 'parameter': 'keys', 'storage': 'span', 'dtype': 'float64',
                 'rank': 1, 'shape': [4], 'python_type': 'builtins.list'}]})
    module, outputs, _ = lower_ast_source_to_ssa(source, 'root', name='dict_rows', extraction_contract=policy)
    root = module.functions['dict_rows__root']
    assert len(outputs[root.name]) == 7
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((artifact, dict(root.metadata['parameter_names']),
                                    tuple(v.id for v in outputs[root.name]), source)))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, parameters, outputs, source = pickle.load(open(sys.argv[1], 'rb'))
scale_id, keys_id = parameters['scale'], parameters['keys']
scope = {}
exec(source, scope)
execution = artifact.prepare_execution({scale_id: np.array([0.0]), keys_id: np.zeros(4)})
for scale, keys in ((0.0, [1,2,3,4]), (2.5, [3,1,1,2]), (-3.0, [-1,-2,-3,-4]), (1.0, [1,3,2,4])):
    execution.buffers[scale_id][0] = scale
    execution.buffers[keys_id][:] = keys
    execution.run()
    actual = tuple(execution.buffers[value].item() for value in outputs)
    expected = scope['root'](scale, keys)
    assert actual == expected, (scale, actual, expected)
''', str(payload)], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stdout + probe.stderr
