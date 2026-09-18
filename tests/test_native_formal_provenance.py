import pytest
from src.compiler.fortran_c_shell import _full_native_link_failures, _undefined_repository_ssa_operands
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule


@pytest.mark.parametrize('accounting', ['invented', 'record', 'workspace', 'member', 'local_member', 'local_view', 'zero_args'])
def test_full_native_gate_checks_formal_provenance(accounting):
    parameter = SSAValue(0, 'float64')
    saved = SSAValue(46, 'float64')
    metadata = {'parameter_names': (('dt', 0),), 'value_names': (('saved', 46),)}
    if accounting == 'record':
        saved.accounting = {'program_abi_parameter': 'material', 'program_abi_field': 'state'}
    elif accounting == 'workspace':
        metadata['storage_formals'] = ({'value_id': 46},)
    elif accounting in {'member', 'local_member'}:
        metadata['authored_parameters'] = ('dt', 'history')
        metadata['parameter_member_formals'] = ({
            'value_id': 46,
            'parameter': 'history' if accounting == 'member' else 'saved',
            'path': (0,),
        },)
    elif accounting == 'local_view':
        saved.accounting = {'source_parameter_view_root': 'step'}
    elif accounting == 'zero_args':
        metadata['parameter_names'] = ()
        metadata['authored_parameters'] = ()
    function = Function('root', [saved] if accounting == 'zero_args' else [parameter, saved], {
        'entry': BasicBlock('entry', [Instr('Ret', [saved], None)]),
    }, metadata=metadata)
    module = IRModule({'root': function})
    # An invented argument passes definedness; provenance must reject it.
    assert _undefined_repository_ssa_operands(module) == ()
    failures = _full_native_link_failures((), (), (), module=module)
    rejected = accounting in {'invented', 'local_member', 'local_view', 'zero_args'}
    assert bool(failures['unaccounted_formals']) == rejected
    assert bool(any(failures.values())) == rejected


@pytest.mark.parametrize('key', ['structural_output_shortfalls', 'unresolved_required_source_values'])
def test_full_native_gate_rejects_missing_returns_without_undefined_operands(key):
    function = Function('root', [], {
        'entry': BasicBlock('entry', [Instr('Ret', [], None)]),
    }, metadata={'parameter_names': (), key: ((25, 'get', 'operator'),)})
    module = IRModule({'root': function})
    assert _undefined_repository_ssa_operands(module) == ()
    failures = _full_native_link_failures((), (), (), module=module)
    assert failures['structural_outputs']
