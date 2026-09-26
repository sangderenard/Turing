import pickle
import subprocess
import sys

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def test_conditional_call_result_has_producer(tmp_path):
    source = '''
def left(x):
    return x + 1.0

def right(x):
    return x - 1.0

def root(x, enabled):
    selected = left(x) if enabled else right(x)
    return selected * 2.0
'''
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({'bindings': [], 'values': [
        {'function': '*', 'parameter': 'x', 'storage': 'scalar',
         'dtype': 'float64', 'rank': 0, 'python_type': 'builtins.float'},
        {'function': 'root', 'parameter': 'enabled', 'storage': 'scalar',
         'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'},
    ]})
    module, outputs, _ = lower_ast_source_to_ssa(
        source, 'root', name='conditional_call', extraction_contract=policy,
    )
    root = module.functions['conditional_call__root']
    names = dict(root.metadata['parameter_names'])
    assert {argument.id for argument in root.args} == {names['x'], names['enabled']}
    merges = [instruction for block in root.blocks.values()
              for instruction in block.instrs
              if instruction.op == 'Phi'
              and instruction.attributes.get('binding') == 'conditional_result']
    assert len(merges) == 1
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'conditional_call')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((artifact, names, outputs[root.name][0].id)))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
with open(sys.argv[1], 'rb') as stream:
    artifact, names, result_id = pickle.load(stream)
for enabled, expected in ((False, 4.0), (True, 8.0)):
    feeds = {names['x']: np.array([3.0]),
             names['enabled']: np.array([enabled], dtype=np.bool_)}
    result = artifact.prepare_execution(feeds).run()
    actual = float(np.asarray(result.buffers[result_id]).reshape(-1)[0])
    assert actual == expected, (enabled, actual, expected)
''', str(payload)], capture_output=True, text=True, timeout=20)
    assert probe.returncode == 0, probe.stdout + probe.stderr
