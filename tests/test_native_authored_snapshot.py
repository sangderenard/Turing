import pickle
import subprocess
import sys
import pytest

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize('mode', ['direct', 'then', 'else', 'early_return'])
def test_native_snapshot_restores_only_authored_fields(tmp_path, mode):
    optional = mode != 'direct'
    source = '''
class Material:
    def copy_shallow(self):
        return (self.state.copy(),)
    def restore(self, saved):
        self.state[...] = saved[0]

def root(material):
    saved = material.copy_shallow()
    material.state[0] = 7.0
    material.telemetry[0] = material.telemetry[0] + 1.0
    material.restore(saved)
    return material.state[0]
'''
    if optional:
        source = source.replace('def root(material):', 'def root(material, rollback):')
        source = source.replace('saved = material.copy_shallow()',
                                'saved = material.copy_shallow() if rollback else None'
                                if mode != 'else' else 'saved = None if rollback else material.copy_shallow()')
        if mode == 'early_return':
            source = source.replace('    material.restore(saved)',
                                    '    if not rollback:\n        return material.state[0]\n'
                                    '    material.restore(saved)')
        else:
            source = source.replace('    material.restore(saved)',
                                    ('    if rollback:\n' if mode == 'then' else '    if not rollback:\n')
                                    + '        material.restore(saved)')
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({
        'records': {'Material': {'identity': 'Material', 'fields': {
            name: {'storage': 'span', 'dtype': 'float64', 'rank': 1,
                   'shape': [2], 'mutable': True}
            for name in ('state', 'telemetry')
        }}},
        'bindings': [{'function': 'root', 'parameter': 'material', 'record': 'Material'}],
        'values': ([{'function': 'root', 'parameter': 'rollback', 'storage': 'scalar',
                     'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'}]
                   if optional else []),
    })
    module, outputs, _ = lower_ast_source_to_ssa(
        source, 'root', name='authored_snapshot', extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    root = module.functions['authored_snapshot__root']
    fields = {argument.accounting['program_abi_field']: argument.id for argument in root.args
              if 'program_abi_field' in argument.accounting}
    assert set(fields) == {'state', 'telemetry'}
    assert len(root.args) == 2 + int(optional)
    rollback = dict(root.metadata['parameter_names']).get('rollback')
    if optional:
        # Inactive span payloads are null pointers, never scalar addresses.
        inactive = {value for receipt in root.metadata['source_optional_values']
                    for value in receipt['inactive_value_ids']}
        constants = {instruction.res.id: instruction.res.dtype
                     for block in root.blocks.values() for instruction in block.instrs
                     if instruction.res is not None and instruction.op == 'Const'}
        assert all(constants[value] == 'ptr' for value in inactive)
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'authored_snapshot')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((artifact, fields, outputs[root.name][0].id, source, rollback)))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
with open(sys.argv[1], 'rb') as stream:
    artifact, fields, result_id, source, rollback_id = pickle.load(stream)
scope = {}
exec(source, scope)
for rollback in ((False, True) if rollback_id is not None else (True,)):
    material = scope['Material']()
    material.state = np.array([2.0, 3.0])
    material.telemetry = np.array([10.0, 20.0])
    expected_return = (scope['root'](material, rollback) if rollback_id is not None
                       else scope['root'](material))
    feeds = {fields['state']: np.array([2.0, 3.0]), fields['telemetry']: np.array([10.0, 20.0])}
    if rollback_id is not None:
        feeds[rollback_id] = np.array([rollback], dtype=np.bool_)
    result = artifact.prepare_execution(feeds).run()
    state = np.asarray(result.buffers[fields['state']])
    telemetry = np.asarray(result.buffers[fields['telemetry']])
    assert np.array_equal(state, material.state), (rollback, state, material.state)
    assert np.array_equal(telemetry, material.telemetry), (telemetry, material.telemetry)
    assert float(np.asarray(result.buffers[result_id]).reshape(-1)[0]) == expected_return
''', str(payload)], capture_output=True, text=True, timeout=20)
    assert probe.returncode == 0, probe.stdout + probe.stderr
