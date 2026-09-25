import pytest

from src.compiler.identity_concordance import (
    begin_identity_book, current_identity_book, end_identity_book,
)
from src.compiler.fortran_c_shell import (
    _complete_propagated_frame_tails, _propagate_record_field_demand,
    _harmonize_call_argument_shapes, _prune_unused_callee_formals,
    _intern_writable_region_outputs,
)
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock
from src.transmogrifier.ssa import SSATensorDescriptor, SSATensorTable


@pytest.fixture(autouse=True)
def _isolated_identity_book():
    _book, token = begin_identity_book()
    try:
        yield
    finally:
        end_identity_book(token)


def test_exact_binding_prevents_duplicate_field_demand_and_survives_storage_growth():
    # Two distinct receivers can have the same field name. The caller's
    # computed operands need not retain that field-name metadata at all.
    fields = [SSAValue(i, 'float64', accounting={
        'program_abi_record': 'Metrics', 'program_abi_field': 'max_vel',
        'program_abi_parameter': name,
    }) for i, name in ((5, 'left'), (6, 'right'))]
    values = [SSAValue(20, 'float64'), SSAValue(21, 'float64')]
    call = Instr('Call', values.copy(), None, attributes={
        'callee': 'callee', 'callee_input_ids': (5, 6), 'plan_callsite_id': 9,
    })
    caller = Function('caller', values.copy(), {'entry': BasicBlock('entry', [call])})
    callee = Function('callee', fields.copy(), {'entry': BasicBlock('entry', [])})
    functions = {'caller': caller, 'callee': callee}
    _propagate_record_field_demand(functions)
    assert call.args == values
    assert caller.args == values
    storage = SSAValue(7, 'float64', shape=(4,), accounting={'compiler_frame_storage': 'callee'})
    callee.args.append(storage)
    assert _complete_propagated_frame_tails(functions) == 1
    assert call.attributes['callee_input_ids'] == (5, 6, 7)
    assert call.args[-1].accounting['propagated_formal_id'] == 7
    # Formal identity, not ambiguous field names, restores declaration order.
    callee.args = [fields[1], fields[0], storage]
    _harmonize_call_argument_shapes(functions)
    assert call.args[:2] == values[::-1]
    assert call.attributes['callee_input_ids'] == (6, 5, 7)


def test_exact_binding_propagates_late_returned_record_storage():
    value = SSAValue(5, 'float64')
    returned = SSAValue(7, 'float64', accounting={
        'returned_record_storage': 'Metrics',
    })
    actual = SSAValue(20, 'float64')
    call = Instr('Call', [actual], None, attributes={
        'callee': 'callee', 'callee_input_ids': (5,), 'plan_callsite_id': 9,
    })
    caller = Function(
        'caller', [actual], {'entry': BasicBlock('entry', [call])},
    )
    callee = Function(
        'callee', [value, returned], {'entry': BasicBlock('entry', [])},
    )

    assert _complete_propagated_frame_tails({
        'caller': caller, 'callee': callee,
    }) == 1
    assert call.attributes['callee_input_ids'] == (5, 7)
    assert call.args[-1] is caller.args[-1]
    assert call.args[-1].accounting['propagated_formal_id'] == 7


def test_exact_binding_reuses_concorded_late_sequence_storage():
    value = SSAValue(5, 'float64')
    sequence_column = SSAValue(7, 'float64')
    actual = SSAValue(20, 'float64')
    caller_storage = SSAValue(21, 'float64', accounting={
        'compiler_frame_storage': 'caller',
    })
    call = Instr('Call', [actual], None, attributes={
        'callee': 'callee', 'callee_input_ids': (5,), 'plan_callsite_id': 9,
    })
    caller = Function(
        'caller', [actual, caller_storage],
        {'entry': BasicBlock('entry', [call])},
    )
    callee = Function(
        'callee', [value, sequence_column],
        {'entry': BasicBlock('entry', [])},
    )
    current_identity_book().page('argument_binding').set(
        ('callee', 7, 'binding'), 9, ('caller_storage', 21),
    )

    assert _complete_propagated_frame_tails({
        'caller': caller, 'callee': callee,
    }) == 1
    assert caller.args == [actual, caller_storage]
    assert call.args == [actual, caller_storage]
    assert call.attributes['callee_input_ids'] == (5, 7)
    assert current_identity_book().page(
        'propagated_frame_tail_concordance'
    ).latest(('caller', 9, 'callee', 7)) == (
        21, 'argument_binding:caller_storage',
    )


def test_exact_binding_restores_removed_caller_storage_identity():
    value = SSAValue(5, 'float64')
    sequence_length = SSAValue(7, 'int64', shape=(1,))
    actual = SSAValue(20, 'float64')
    call = Instr('Call', [actual], None, attributes={
        'callee': 'callee', 'callee_input_ids': (5,), 'plan_callsite_id': 9,
    })
    caller = Function(
        'caller', [actual], {'entry': BasicBlock('entry', [call])},
    )
    callee = Function(
        'callee', [value, sequence_length],
        {'entry': BasicBlock('entry', [])},
    )
    current_identity_book().page('argument_binding').set(
        ('callee', 7, 'binding'), 9, ('caller_storage', 21),
    )

    assert _complete_propagated_frame_tails({
        'caller': caller, 'callee': callee,
    }) == 1
    restored = caller.args[-1]
    assert restored.id == 21
    assert restored.shape == (1,)
    assert restored.accounting['restored_argument_binding'] is True
    assert call.args == [actual, restored]
    assert call.attributes['callee_input_ids'] == (5, 7)
    assert current_identity_book().page(
        'propagated_frame_tail_concordance'
    ).latest(('caller', 9, 'callee', 7)) == (
        21, 'argument_binding:restored_caller_storage',
    )


def test_pruning_updates_exact_call_input_receipt_with_operands():
    dead, live = SSAValue(1, 'float64'), SSAValue(2, 'float64')
    callee = Function('callee', [dead, live], {'entry': BasicBlock('entry', [Instr('Ret', [live], None)])})
    args = [SSAValue(10, 'float64'), SSAValue(11, 'float64')]
    call = Instr('Call', args.copy(), SSAValue(12, 'float64'),
                 attributes={'callee': 'callee', 'callee_input_ids': (1, 2)})
    caller = Function('caller', args.copy(), {'entry': BasicBlock('entry', [call])})
    assert _prune_unused_callee_formals({'caller': caller, 'callee': callee}) == 1
    assert call.args == [args[1]]
    assert call.attributes['callee_input_ids'] == (2,)


def test_tensor_output_ownership_survives_loss_of_record_field_name():
    formal = SSAValue(7, 'float64', shape=(20,), accounting={
        'program_abi_mutable': True, 'program_abi_storage': 'span',
    })
    result = SSAValue(7, 'float64', shape=(20,))
    producer = Instr('Call', [result], result, attributes={'ssa_output_argument': 0})
    ret = Instr('Ret', [formal], None)
    function = Function('region', [formal], {'entry': BasicBlock('entry', [producer, ret])})
    # Mutability alone does not prove a duplicate ID is the same output.
    _intern_writable_region_outputs(function)
    assert producer.res is result
    table = SSATensorTable(tensors={7: SSATensorDescriptor(
        tensor_id=7, data_value_id=7, dtype='float64', shape=(20,),
        storage='output', writable=True,
    )})
    _intern_writable_region_outputs(function, table)
    assert producer.res is formal
    assert producer.args[0] is formal
    assert ret.args[0] is formal
    assert table.tensors[7].data_value_id == formal.id


def test_declared_region_inout_interns_its_producer_without_field_metadata():
    formal = SSAValue(7, 'float64')
    result = SSAValue(7, 'float64')
    producer = Instr('max', [SSAValue(6, 'float64', shape=(4,))], result)
    ret = Instr('Ret', [formal], None)
    function = Function(
        'region', [formal], {'entry': BasicBlock('entry', [producer, ret])},
        metadata={'source_region_integral': {
            'capture_value_ids': (6, 7),
            'output_value_ids': (7,),
        }},
    )

    _intern_writable_region_outputs(function)

    assert producer.res is formal
    assert ret.args[0] is formal
    assert function.metadata['interned_program_abi_output_ids'] == (7,)
