import pytest
from src.compiler.fortran_c_shell import _full_native_link_failures
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule


@pytest.mark.parametrize('payload_kind', ['none', 'scalar', 'span', 'unknown'])
def test_native_gate_rejects_unrepresented_optional_phi(payload_kind):
    absent = SSAValue(1, 'none')
    payload = absent if payload_kind == 'none' else SSAValue(
        0, None if payload_kind == 'unknown' else 'float64',
        (2, 3) if payload_kind == 'span' else (),
    )
    merged = SSAValue(2, payload.dtype, payload.shape)
    function = Function('root', [] if payload is absent else [payload], {
        'entry': BasicBlock('entry', [Instr('NoneValue', [], absent),
            Instr('Phi', [payload, absent], merged), Instr('Ret', [merged], None)])
    })
    failures = _full_native_link_failures((), (), (), module=IRModule({'root': function}))
    assert bool(failures['unrepresented_optional_merges']) == (payload_kind != 'none')


def test_python_optional_scalar_fails_at_native_representation_boundary():
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_fortran_backend import FortranEmissionError

    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({'bindings': [], 'values': [
        {'function': 'root', 'parameter': 'value', 'storage': 'scalar',
         'dtype': 'float64', 'rank': 0, 'python_type': 'builtins.float'},
        {'function': 'root', 'parameter': 'enabled', 'storage': 'scalar',
         'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'},
    ]})
    with pytest.raises(FortranEmissionError, match='explicit presence and payload'):
        lower_ast_source_to_ssa(
            'def root(value, enabled):\n    return value if enabled else None\n',
            'root', name='optional_scalar', extraction_contract=policy,
        )

@pytest.mark.parametrize('guard', ['unguarded', 'absent', 'changed'])
def test_optional_tuple_payload_requires_matching_presence_guard(guard):
    import runpy
    import numpy as np
    lower = runpy.run_path('tests/test_aggregate_call_identity.py')['_lower']
    consumer = {
        'unguarded': '    return recover(saved)\n',
        'absent': '    if not enabled:\n        return recover(saved)\n    return array\n',
        'changed': '    enabled = not enabled\n    if enabled:\n        return recover(saved)\n    return array\n',
    }[guard]
    source = '''
def capture(array):
    return (array.copy(),)
def recover(saved):
    return saved[0]
def tick(array, enabled):
    saved = capture(array) if enabled else None
''' + consumer
    with pytest.raises(ValueError, match='no proven presence guard'):
        lower(source, 'optional_guard', {'array': np.zeros((2, 3)), 'enabled': True})
