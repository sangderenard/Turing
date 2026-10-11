"""A retained loop has its closest authored owner, not every ancestor."""
import pickle
import subprocess
import sys
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

SOURCE = '''
def root(rejected):
    failures = []
    lines = [10.0]
    while True:
        if rejected:
            failures.append((1.0, 2.0, (3.0,)))
        if rejected:
            for index, (dt_f, m, why) in enumerate(failures, 1):
                lines.append(20.0)
        break
    return len(lines)
'''

def lower():
    policy = ExtractionContract('extraction_contracts/program_extraction.yaml').with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml').with_program_abi({
        'bindings': [], 'values': [{'function': 'root', 'parameter': 'rejected',
        'storage': 'scalar', 'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'}]})
    return lower_ast_source_to_ssa(SOURCE, 'root', name='direct_control_owner',
                                 extraction_contract=policy, progress=lambda message: None)

def test_loop_inside_conditional_inside_while_is_lowered_once():
    module, _, exports = lower()
    function = module.functions[exports[0]]
    headers = [instruction for block in function.blocks.values() for instruction in block.instrs
               if instruction.op == 'Phi' and instruction.attributes.get('source_loop_node_id') is not None]
    assert len(headers) == 1, [(instruction.attributes['source_loop_node_id'], instruction.res.id)
                               for instruction in headers]

def test_guarded_loop_executes_once_and_only_in_its_authored_arm(tmp_path):
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    module, outputs, exports = lower()
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
for rejected, expected in ((False, 1), (True, 2), (False, 1), (True, 2)):
    execution.buffers[parameter][0] = rejected
    execution.run()
    actual = execution.buffers[result].item()
    print('GUARDED_NATIVE', rejected, actual, expected, flush=True)
    assert actual == expected, (rejected, actual, expected)
''', str(saved)], capture_output=True, text=True)
    print(probe.stdout, end='')
    assert probe.returncode == 0, probe.stdout + probe.stderr
