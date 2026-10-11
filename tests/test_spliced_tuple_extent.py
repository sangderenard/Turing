"""Iteration cardinality is projected from exact tuple/star operand edges."""
import pickle
import subprocess
import sys
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c

SOURCE = '''
def root(rejected):
    failures = []
    if rejected:
        failures.append((1.0, 2.0, (3.0,)))
    count = 0.0
    for index, (dt_f, m, why) in enumerate((*failures, (4.0, 5.0, (6.0,))), 1):
        count += 1.0
    return count
'''

def test_spliced_tuple_count_uses_resident_length_without_a_tuple_buffer(tmp_path):
    policy = ExtractionContract('extraction_contracts/program_extraction.yaml').with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml').with_program_abi({
        'bindings': [], 'values': [{'function': 'root', 'parameter': 'rejected',
        'storage': 'scalar', 'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'}]})
    module, outputs, exports = lower_ast_source_to_ssa(SOURCE, 'root', name='spliced_extent',
        extraction_contract=policy, progress=lambda message: None)
    function = module.functions[exports[0]]
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    parameter = dict(function.metadata['parameter_names'])['rejected']
    result = outputs[function.name][0].id
    saved = tmp_path / 'proof.pkl'
    saved.write_bytes(pickle.dumps((artifact, parameter, result)))
    probe = subprocess.run([sys.executable, '-u', '-c', '''
import pickle, sys
import numpy as np
artifact, parameter, result = pickle.load(open(sys.argv[1], 'rb'))
execution = artifact.prepare_execution({parameter: np.array([False], dtype=np.bool_)})
for rejected, expected in ((False, 1.0), (True, 2.0), (False, 1.0), (True, 2.0)):
    execution.buffers[parameter][0] = rejected
    execution.run()
    actual = execution.buffers[result].item()
    print('SPLICE_NATIVE', rejected, actual, expected, flush=True)
    assert actual == expected, (rejected, actual, expected)
''', str(saved)], capture_output=True, text=True)
    print(probe.stdout, end='')
    assert probe.returncode == 0, probe.stdout + probe.stderr
