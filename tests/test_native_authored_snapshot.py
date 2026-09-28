import pickle
import subprocess
import sys
import numpy as np
import pytest

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.common.dt_system.dt_scaler import Metrics
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import (
    _reconcile_singleton_destructured_call_results,
    lower_ast_source_to_ssa,
)
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue


def test_final_singleton_concordance_rebinds_restored_plan_call_result():
    temporary = SSAValue(19, dtype='float64')
    projected = SSAValue(20, dtype='float64', accounting={
        'compiler_frame_storage': True,
    })
    call = Instr('Call', [], temporary, attributes={
        'callee': 'linked_law',
        'source_linked': True,
        'plan_callsite_id': 7,
    })
    consumer = Instr('Call', [projected], None, attributes={
        'callee': 'assign_result',
    })
    caller = Function('caller', [projected], {
        'entry': BasicBlock('entry', [call, consumer]),
    }, metadata={'singleton_call_result_concordance': [{
        'callsite_id': 7,
        'callee': 'linked_law',
        'temporary_id': 19,
        'projected_result_id': 20,
    }]})

    receipts = _reconcile_singleton_destructured_call_results({
        'caller': caller,
    })

    assert int(call.res.id) == int(consumer.args[0].id) == 20
    assert receipts[0]['reason'] == (
        'final_singleton_destructuring_concordance')


def test_singleton_destructured_call_result_is_not_dead_frame_storage(tmp_path):
    source = '''
class Material:
    pass

def produce(value):
    return (value + 2.0,)

def root(material):
    produced, = produce(material.state)
    material.state[...] = produced
'''
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_program_abi({
        'records': {'Material': {'identity': 'Material', 'fields': {
            'state': {'storage': 'span', 'dtype': 'float64', 'rank': 1,
                      'shape': [2], 'mutable': True},
        }}},
        'bindings': [
            {'function': '*', 'parameter': 'material', 'record': 'Material'},
        ],
        'values': [],
    })
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, 'root', name='singleton_result', extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    root = module.functions[exports[0]]
    state = next(
        argument for argument in root.args
        if (argument.accounting or {}).get('program_abi_field') == 'state'
        and (argument.accounting or {}).get('program_abi_field_written')
    )
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'singleton_result')
    execution = artifact.prepare_execution({
        int(state.id): np.array([1.0, 4.0]),
    })
    execution.run()

    np.testing.assert_allclose(
        execution.buffers[int(state.id)], np.array([3.0, 6.0]),
        rtol=0.0, atol=0.0)


def test_native_snapshot_state_is_resident_across_while_call(tmp_path):
    """A mutable ProgramABI field is the loop's resident state, not a frame copy."""
    source = '''
class Material:
    def copy_shallow(self):
        return (self.state.copy(),)
    def restore(self, saved):
        self.state[...] = saved[0]

class StepResult:
    def __init__(self, dt_next):
        self.dt_next = dt_next

def advance(material, dt):
    material.state[...] = material.state + dt
    return StepResult(dt * 4.0), dt * 0.5, dt

def root(material, round_dt, dt_initial):
    advanced = 0.0
    dt_cap = dt_initial
    last_dt_next = dt_cap
    while advanced < round_dt:
        saved = material.copy_shallow()
        used = min(dt_cap, round_dt - advanced)
        outcome, dt_next, dt_used = advance(material, used)
        if used < 0.0:
            material.restore(saved)
        advanced = advanced + dt_used
        dt_cap = min(dt_cap, dt_next)
        last_dt_next = dt_next
    return advanced, last_dt_next
'''
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({
        'records': {'Material': {'identity': 'Material', 'fields': {
            'state': {'storage': 'span', 'dtype': 'float64', 'rank': 1,
                      'shape': [2], 'mutable': True},
        }}},
        'bindings': [
            {'function': '*', 'parameter': 'material', 'record': 'Material'},
        ],
        'values': [
            {'function': 'root', 'parameter': name, 'storage': 'scalar',
             'dtype': 'float64', 'rank': 0, 'python_type': 'builtins.float'}
            for name in ('round_dt', 'dt_initial')
        ],
    })
    module, outputs, _ = lower_ast_source_to_ssa(
        source, 'root', name='resident_snapshot', extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    assert module.metadata['frontend_representations_detached'] is True
    assert module.metadata['full_native_link_gate'][
        'frontend_representations_detached'] is True
    assert all(entry.graph is None for entry in module.function_table)
    pickle.dumps(module)
    root = module.functions['resident_snapshot__root']
    fields = {
        argument.accounting['program_abi_field']: argument.id
        for argument in root.args
        if 'program_abi_field' in argument.accounting
    }
    parameters = dict(root.metadata['parameter_names'])
    artifact = emit_ssa_module_to_c(module, root.name)
    aliased_calls = [instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == 'Call'
        and instruction.attributes.get('result_aliases_frame')
    ]
    for instruction in aliased_calls:
        position = int(instruction.attributes['ssa_output_argument'])
        callee = module.functions[instruction.attributes['callee']]
        assert int(callee.args[position].id) == int(
            instruction.attributes['ssa_output_formal_id'])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'resident_snapshot')
    execution = artifact.prepare_execution({
        fields['state']: np.array([1.0, 4.0]),
        parameters['round_dt']: np.array([0.1875]),
        parameters['dt_initial']: np.array([0.1]),
    })
    execution.run()

    np.testing.assert_allclose(
        execution.buffers[fields['state']], np.array([1.1875, 4.1875]),
        rtol=0.0, atol=2.0e-15)
    assert float(execution.buffers[outputs[root.name][0].id][0]) == pytest.approx(0.1875)
    assert float(execution.buffers[outputs[root.name][1].id][0]) == pytest.approx(0.00625)


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
