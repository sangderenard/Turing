import pickle
import subprocess
import sys
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c


@pytest.mark.parametrize('guard', ['value is not None', 'callable(None)', 'sequential', 'zero'])
def test_none_returning_call_precedes_its_branch_and_executes_natively(tmp_path, guard):
    source = f'''
def helper():
    return None
def root(x):
    value = helper()
    if {guard}:
        x = x * 2.0
    return x
'''
    if guard == 'zero':
        source = source.replace('return None', 'return 0.0').replace('if zero:', 'if value is not None:')
    if guard == 'sequential':
        source = '''
def helper():
    return None
def positive(x):
    return x > 0.0
def root(x, flag):
    value = helper()
    if value is not None:
        x = x * 2.0
    if positive(flag):
        x = x + 1.0
    value = helper()
    if value is not None:
        x = x * 3.0
    return x
'''
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file('extraction_contracts/vehicle_full_native_execution.yaml')
    module, _, exports = lower_ast_source_to_ssa(
        source, 'root', name='none_predicate', extraction_contract=policy,
    )
    artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps(artifact))
    root = module.functions[exports[0]]
    input_id = root.args[0].id
    flag_id = root.args[1].id if guard == 'sequential' else input_id
    output_id = next(i.args[0].id for b in root.blocks.values()
                     for i in b.instrs if i.op == 'Ret')
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact = pickle.load(open(sys.argv[1], 'rb'))
for value in (-3.0, 0.0, 2.5):
    execution = artifact.prepare_execution({int(sys.argv[2]): np.array([value]),
                                           int(sys.argv[5]): np.array([value])}).run()
    expected = value + (1.0 if sys.argv[4] == 'sequential' and value > 0 else 0.0)
    if sys.argv[4] == 'zero':
        expected = value * 2.0
    actual = execution.buffers[int(sys.argv[3])].item()
    assert actual == expected, (value, actual, expected)
''', str(saved), str(input_id), str(output_id), guard, str(flag_id)],
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
