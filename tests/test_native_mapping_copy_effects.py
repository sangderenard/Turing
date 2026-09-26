import pickle
import subprocess
import sys
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize('variant', ['ordinary', 'empty', 'overwrite'])
def test_materialized_mapping_copy_and_conditional_writes(tmp_path, variant):
    source = '''
def root(change):
    original = {1: 2.0}
    copied = dict(original)
    if change:
        copied[3] = 4.0
        copied[5] = 6.0
    return copied.get(1, 0.0), copied.get(3, 0.0), copied.get(5, 0.0), original.get(3, 0.0)
'''
    if variant == 'empty':
        source = source.replace('{1: 2.0}', '{}')
    elif variant == 'overwrite':
        source = source.replace('        copied[5] = 6.0',
                                '        copied[3] = 9.0\n        copied[5] = 6.0')
    policy = ExtractionContract('extraction_contracts/program_extraction.yaml').with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml').with_program_abi({
            'bindings': [], 'values': [{'function': 'root', 'parameter': 'change',
                'storage': 'scalar', 'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'}]})
    module, outputs, _ = lower_ast_source_to_ssa(source, 'root', name='mapping_copy', extraction_contract=policy)
    root = module.functions['mapping_copy__root']
    assert len(outputs[root.name]) == 4
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((artifact, dict(root.metadata['parameter_names'])['change'],
                                    tuple(v.id for v in outputs[root.name]), source)))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, change_id, outputs, source = pickle.load(open(sys.argv[1], 'rb'))
scope = {}
exec(source, scope)
execution = artifact.prepare_execution({change_id: np.array([False], dtype=np.bool_)})
for change in (False, True, False, True):
    execution.buffers[change_id][0] = change
    execution.run()
    actual = tuple(execution.buffers[value].item() for value in outputs)
    expected = scope['root'](change)
    assert actual == expected, (change, actual, expected)
''', str(payload)], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stdout + probe.stderr


def test_dynamic_mapping_literal_clears_before_all_authored_rows(tmp_path):
    source = '''
def root(value, flag):
    early = value + 1.0
    if flag:
        middle = value + 2.0
    else:
        middle = value + 3.0
    result = {
        "first": early,
        "second": middle,
        "third": value * 4.0,
    }
    return (
        result.get("first", 0.0),
        result.get("second", 0.0),
        result.get("third", 0.0),
    )
'''
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({'bindings': [], 'values': [
        {
            'function': 'root', 'parameter': 'value',
            'storage': 'scalar', 'dtype': 'float64', 'rank': 0,
            'python_type': 'builtins.float',
        },
        {
            'function': 'root', 'parameter': 'flag',
            'storage': 'scalar', 'dtype': 'bool', 'rank': 0,
            'python_type': 'builtins.bool',
        },
    ]})
    module, outputs, _ = lower_ast_source_to_ssa(
        source, 'root', name='mapping_literal_effect_order',
        extraction_contract=policy,
    )
    root = module.functions['mapping_literal_effect_order__root']
    effects = [
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if (
            instruction.attributes.get('binding') == 'ssa_sequence_clear'
            or instruction.attributes.get('ssa_sequence_operation')
            == 'table_store'
        )
    ]
    assert effects[0].attributes.get('binding') == 'ssa_sequence_clear'
    assert sum(
        instruction.attributes.get('ssa_sequence_operation') == 'table_store'
        for instruction in effects
    ) == 3

    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native', optimization='O0')
    payload = tmp_path / 'artifact.pkl'
    names = dict(root.metadata['parameter_names'])
    payload.write_bytes(pickle.dumps((
        artifact, names, tuple(value.id for value in outputs[root.name]), source,
    )))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, names, outputs, source = pickle.load(open(sys.argv[1], 'rb'))
scope = {}
exec(source, scope)
execution = artifact.prepare_execution({
    names['value']: np.array([0.0], dtype=np.float64),
    names['flag']: np.array([False], dtype=np.bool_),
})
for value, flag in ((2.0, False), (5.0, True), (-3.0, False)):
    execution.buffers[names['value']][0] = value
    execution.buffers[names['flag']][0] = flag
    execution.run()
    actual = tuple(execution.buffers[item].item() for item in outputs)
    expected = scope['root'](value, flag)
    assert actual == expected, ((value, flag), actual, expected)
''', str(payload)], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stdout + probe.stderr
