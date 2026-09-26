import pickle
import subprocess
import sys
import pytest

from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c


@pytest.mark.parametrize('tuple_return', [False, True])
@pytest.mark.parametrize('dynamic_guard', [False, True, 'direct'])
@pytest.mark.parametrize('pruned_return', [False, True])
def test_return_after_conditional_and_source_call_uses_produced_value(tmp_path, tuple_return, dynamic_guard, pruned_return):
    policy = ExtractionContract('extraction_contracts/program_extraction.yaml').with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml').with_program_abi({
            'records': {'Limits': {'identity': 'Limits', 'fields': {
                'dt_max': {'storage': 'scalar', 'dtype': 'float64', 'mutable': False},
            }}}, 'bindings': [{'function': 'root', 'parameter': 'ctrl', 'record': 'Limits'}], 'values': []})
    source = '''
def propose(value):
    return value * 2.0

def finish(value):
    return value * 0.5

def root(dt, ctrl):
    while dt > 0.0:
        proposal = propose(dt)
        if ctrl.dt_max is not None:
            proposal = AbstractTensor.minimum(proposal, ctrl.dt_max)
        proposal = finish(proposal)
        return float(proposal)
    return float(dt)
'''
    if tuple_return:
        source = source.replace('return float(proposal)', 'return finish(proposal), float(dt)').replace('return float(dt)', 'return float(dt), float(dt)')
    if dynamic_guard:
        source = source.replace('ctrl.dt_max is not None',
            'ctrl.dt_max' if dynamic_guard == 'direct' else 'ctrl.dt_max > 0.0')
    if pruned_return:
        dead_value = '(dt, finish(dt * 3.0))' if tuple_return else 'finish(dt * 3.0)'
        source = source.replace('        proposal = propose(dt)',
            f'        if False:\n            return {dead_value}\n        proposal = propose(dt)')
    module, _, exports = lower_ast_source_to_ssa(source, 'root', name='return_call',
        extraction_contract=policy, python_bindings={'AbstractTensor': AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference())
    root = module.functions[exports[0]]
    parameter = dict(root.metadata['parameter_names'])['dt']
    limit = tuple(v.id for v in root.args if v.accounting.get('program_abi_field') == 'dt_max')
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    artifact = emit_ssa_to_c(module, exports[0], watch=(output,))
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, parameter, limit, output, tuple_return, dynamic_guard)))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, parameter, limit, output, tuple_return, dynamic_guard = pickle.load(open(sys.argv[1], 'rb'))
for dt in (0.5, 1.0, 2.0):
    for cap in (float('inf'), dt * 0.5, 0.0):
        feeds = {value_id: np.array([cap]) for value_id in limit}
        feeds[parameter] = np.array([dt])
        execution = artifact.prepare_execution(feeds).run()
        bounded = dt * 2 if dynamic_guard and cap <= 0.0 else min(dt * 2, cap)
        assert execution.buffers[output].item() == bounded * (0.25 if tuple_return else 0.5), (dt, cap, execution.buffers[output])
''', str(saved)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr

