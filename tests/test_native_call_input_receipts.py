from src.compiler.fortran_c_shell import (
    _complete_propagated_frame_tails, _propagate_record_field_demand,
    _harmonize_call_argument_shapes, _prune_unused_callee_formals,
    _intern_writable_region_outputs,
)
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock
from src.transmogrifier.ssa import SSATensorDescriptor, SSATensorTable


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
