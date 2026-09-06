import pickle
import subprocess
import sys
import pytest

from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c


@pytest.mark.parametrize('through_loop', [False, True])
@pytest.mark.parametrize('runtime_limits', [False, True])
@pytest.mark.parametrize('repeat_bounds', [False, True])
def test_pruned_optional_cap_keeps_initial_symbolic_value(tmp_path, through_loop, runtime_limits, repeat_bounds):
    if repeat_bounds and not (through_loop and runtime_limits):
        pytest.skip('repeated runtime bounds require the loop and record')
    policy = ExtractionContract('extraction_contracts/program_extraction.yaml').with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml')
    source = '''
def root(dt):
    cap = AbstractTensor.tensor(dt)
    limit = None
    if limit is not None:
        cap = AbstractTensor.minimum(cap, limit)
    return cap
'''
    if through_loop:
        source = 'def next_cap(value):\n    return value * 0.5\n' + source.replace(
            '    return cap',
            '    total = AbstractTensor.tensor(0.0)\n    while total < dt:\n'
            '        used = cap\n        proposal = next_cap(cap)\n'
            '        total = total + used\n        cap = proposal\n    return total')
    if runtime_limits:
        policy = policy.with_program_abi({'records': {'Limits': {'identity': 'Limits', 'fields': {
            'dt_min': {'storage': 'scalar', 'dtype': 'float64', 'mutable': False},
            'dt_max': {'storage': 'scalar', 'dtype': 'float64', 'mutable': False},
        }}}, 'bindings': [{'function': 'root', 'parameter': 'ctrl', 'record': 'Limits'}], 'values': []})
        source = source.replace('def root(dt):', 'def root(dt, ctrl):').replace(
            '    limit = None\n    if limit is not None:\n        cap = AbstractTensor.minimum(cap, limit)',
            '    if ctrl.dt_min is not None:\n        cap = AbstractTensor.maximum(cap, ctrl.dt_min)\n'
            '    if ctrl.dt_max is not None:\n        cap = AbstractTensor.minimum(cap, ctrl.dt_max)')
        if repeat_bounds:
            source = source.replace('        cap = proposal',
                '        cap = proposal\n'
                '        if ctrl.dt_min is not None:\n'
                '            cap = AbstractTensor.maximum(ctrl.dt_min, cap)\n'
                '        if ctrl.dt_max is not None:\n'
                '            cap = AbstractTensor.minimum(ctrl.dt_max, cap)')
    module, _, exports = lower_ast_source_to_ssa(source, 'root', name='pruned_cap', extraction_contract=policy,
        python_bindings={'AbstractTensor': AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference())
    root = module.functions[exports[0]]
    parameter = dict(root.metadata['parameter_names'])['dt']
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    limits = {v.id: (0.0 if v.accounting.get('program_abi_field') == 'dt_min' else float('inf'))
              for v in root.args if v.accounting.get('program_abi_field') in ('dt_min', 'dt_max')}
    artifact = emit_ssa_to_c(module, exports[0], watch=(output,))
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, parameter, output, limits)))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, parameter, output, limits = pickle.load(open(sys.argv[1], 'rb'))
for dt in (0.5, 1.0, 2.0):
    feeds = {k: np.array([v]) for k, v in limits.items()}
    feeds[parameter] = np.array([dt])
    execution = artifact.prepare_execution(feeds).run()
    assert execution.buffers[output].item() == dt, execution.buffers[output]
''', str(saved)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
