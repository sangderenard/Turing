"""An authored field read must not observe a later conditional write."""

import pickle
import subprocess
import sys
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference


@pytest.mark.parametrize('variant', ['capture', 'first_write', 'nested', 'increment'])
def test_scalar_field_capture_precedes_conditional_write(tmp_path, variant):
    source = '''
def root(state, change):
    before = bool(state.flag)
    if change:
        state.flag = False
    return before, state.flag
'''
    if variant == 'first_write':
        source = source.replace('    before = bool(state.flag)\n', '')
        source = source.replace('    return before, state.flag',
                                '    return bool(state.flag), state.flag')
    elif variant == 'nested':
        source = source.replace('        state.flag = False',
                                '        if bool(state.flag):\n            state.flag = False')
    elif variant == 'increment':
        source = source.replace('state.flag = False', 'state.flag += 1.0')
    dtype = 'float64' if variant == 'increment' else 'bool'
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({
        'records': {'State': {'identity': 'State', 'fields': {
            'flag': {'storage': 'scalar', 'dtype': dtype, 'rank': 0, 'mutable': True},
        }}},
        'bindings': [{'function': 'root', 'parameter': 'state', 'record': 'State'}],
        'values': [{'function': 'root', 'parameter': 'change', 'storage': 'scalar',
                    'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'}],
    })
    module, outputs, _ = lower_ast_source_to_ssa(
        source, 'root', name='record_read_order', extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    root = module.functions['record_read_order__root']
    flag = next(argument.id for argument in root.args
                if argument.accounting.get('program_abi_field') == 'flag')
    names = dict(root.metadata['parameter_names'])
    assert 'change' in names, 'Authored scalar write and guard must remain live'
    change = names['change']
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, flag, change, tuple(value.id for value in outputs[root.name]), source, dtype)))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
from types import SimpleNamespace
import numpy as np
artifact, flag, change, output, source, dtype = pickle.load(open(sys.argv[1], 'rb'))
namespace = {}
exec(source, namespace)
execution = artifact.prepare_execution({flag: np.array([False], dtype=dtype),
                                       change: np.array([False], dtype=np.bool_)})
for initial in (False, True):
    for enabled in (False, True):
        state = SimpleNamespace(flag=initial)
        expected = namespace['root'](state, enabled)
        execution.buffers[flag][0] = initial
        execution.buffers[change][0] = enabled
        execution.run()
        actual = tuple(execution.buffers[value].item() for value in output)
        assert actual == expected, (initial, enabled, actual, expected)
        assert execution.buffers[flag].item() == state.flag
''', str(saved)], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stdout + probe.stderr
