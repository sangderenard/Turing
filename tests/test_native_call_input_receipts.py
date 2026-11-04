import pytest

from src.compiler.identity_concordance import (
    Derived, Mode, Novel, begin_identity_book, current_identity_book,
    end_identity_book,
)
from src.compiler.concordance_declarations import (
    ARGUMENT_BINDING, ArgumentBindingFact, CALL_RECORD, CONTROL_SSA,
    FRAME_BINDING, INGEST_SOURCE, SSA_VALUE, SSAValueFact, SSAValueOrigin,
)
from src.compiler.fortran_c_shell import (
    _complete_propagated_frame_tails, _propagate_record_field_demand,
    _harmonize_call_argument_shapes, _prune_unused_callee_formals,
    _intern_writable_region_outputs,
)
from src.transmogrifier.ssa import (
    SSAValue, Instr, Function, BasicBlock, SSACallRecord, SSACallTable,
)
from src.transmogrifier.ssa import SSATensorDescriptor, SSATensorTable


@pytest.fixture(autouse=True)
def _isolated_identity_book():
    _book, token = begin_identity_book()
    try:
        yield
    finally:
        end_identity_book(token)


def _post_value_roots(scope, *values):
    """Declared inputs of the fixture program: each value's ``ssa_value`` row
    is a ``Novel(INGEST_SOURCE, ())`` root, as the real builder posts it."""

    book = current_identity_book()
    return tuple(
        book.post(
            SSA_VALUE, (scope, value.id),
            SSAValueFact(value.dtype, (), SSAValueOrigin.ADOPTED_GRAPH_ID),
            stage=CONTROL_SSA, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        for value in values
    )


def _post_caller_storage_binding(callee, formal, callsite, caller, storage, actual):
    """Publish the exact call record and its caller-qualified storage edge."""

    book = current_identity_book()
    formal_cell = book.latest_ref(SSA_VALUE, (callee, formal.id))
    storage_cell = book.latest_ref(SSA_VALUE, (caller, storage.id))
    actual_cell = book.latest_ref(SSA_VALUE, (caller, actual.id))
    call_table = SSACallTable(owner='caller-storage-fixture', mutable=True)
    call_record = SSACallRecord(
        caller=caller,
        callsite_id=callsite,
        callee_reference=None,
        callee_name=callee,
        callee_symbol=callee,
        argument_bindings=((actual.id, formal.id),),
        callee_storage_value_ids=(formal.id,),
        frame_bindings=((formal.id, 'caller_storage', storage.id),),
        resolution='native_call',
    )
    record_cell = call_table.assign(
        caller, (call_record,),
        sources=(formal_cell, storage_cell, actual_cell),
        stage=FRAME_BINDING,
    )
    cells = (formal_cell, storage_cell, actual_cell, record_cell)
    return book.post(
        ARGUMENT_BINDING, (callee, caller, formal.id, callsite),
        ArgumentBindingFact('caller_storage', storage.id),
        stage=FRAME_BINDING, provenance=Derived(cells), mode=Mode.CONCORD,
    )


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


def test_late_owned_heap_tail_uses_declared_binding_schema_and_reaches_root():
    # A nested tensor result acquires caller-owned storage after source-link
    # Calls were built. The source helper already carries two such formals;
    # the root Call still names only its original data input.
    root_data = SSAValue(20, 'int64', shape=(4,))
    helper_data = SSAValue(1, 'int64', shape=(4,))
    first = SSAValue(7, 'ptr', shape=(1,), accounting={
        'heap_owner': True, 'compiler_frame_storage': 'helper',
    })
    second = SSAValue(8, 'ptr', shape=(1,), accounting={
        'heap_owner': True, 'compiler_frame_storage': 'helper',
    })
    call = Instr('Call', [root_data], None, attributes={
        'callee': 'helper', 'callee_input_ids': (1,),
        'plan_callsite_id': 9, 'source_linked': True,
    })
    root = Function('root', [root_data], {
        'entry': BasicBlock('entry', [call]),
    })
    helper = Function('helper', [helper_data, first, second], {
        'entry': BasicBlock('entry', []),
    })
    _post_value_roots('root', root_data)
    _post_value_roots('helper', helper_data, first, second)
    # A different formal's booked argument binding makes the page present.
    # The late owner has no caller-qualified binding and must use the
    # declared page schema, then the incumbent frame-tail mint.
    _post_caller_storage_binding(
        'helper', helper_data, 3, 'root', root_data, root_data,
    )

    assert _complete_propagated_frame_tails({
        'root': root, 'helper': helper,
    }) == 2
    assert call.attributes['callee_input_ids'] == (1, 7, 8)
    assert tuple(value.id for value in call.args[1:]) == tuple(
        value.id for value in root.args[1:]
    )
    assert all(value.accounting['propagated_formal_id'] == formal
               for value, formal in zip(call.args[1:], (7, 8)))
    page = current_identity_book().page('propagated_frame_tail_concordance')
    assert all(page.latest(('root', 9, 'helper', formal))[0] == actual.id
               for formal, actual in zip((7, 8), call.args[1:]))


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
    # Program: two-formal callee whose tail formal 7 is bound to caller
    # storage 21.  The wrong write was a raw page set (refused by the book);
    # the rule is that the binding posts DERIVED from the formal's and the
    # storage's own cells, which are posted as Novel roots first.
    _post_value_roots('callee', value, sequence_column)
    _post_value_roots('caller', actual, caller_storage)
    _post_caller_storage_binding(
        'callee', sequence_column, 9, 'caller', caller_storage, actual,
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
    # Program: callee tail formal 7 is bound to caller storage 21, which the
    # caller no longer carries.  Raw page set was refused; post the binding
    # DERIVED from the formal's cell and the storage's cell (both Novel
    # roots; the removed storage keeps its own ssa_value row).
    removed_storage = SSAValue(21, 'int64', shape=(1,))
    _post_value_roots('callee', value, sequence_length)
    _post_value_roots('caller', actual, removed_storage)
    _post_caller_storage_binding(
        'callee', sequence_length, 9, 'caller', removed_storage, actual,
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
    # Program: callee with an unread formal 1.  The pass derives the pruned
    # formal from the formal's ssa_value cell, which this fixture never
    # posted; post the declared formals as Novel roots (as the sibling test).
    _post_value_roots('callee', dead, live)
    assert _prune_unused_callee_formals({'caller': caller, 'callee': callee}) == 1
    assert call.args == [args[1]]
    assert call.attributes['callee_input_ids'] == (2,)


def test_pruning_revises_book_call_record_from_exact_record_and_formal_cells():
    book = current_identity_book()
    dead, live = SSAValue(1, 'float64'), SSAValue(2, 'float64')
    for value in (dead, live):
        book.post(
            SSA_VALUE, ('callee', value.id),
            SSAValueFact(value.dtype, (), SSAValueOrigin.ADOPTED_GRAPH_ID),
            stage=CONTROL_SSA,
            provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
        )
    callee = Function(
        'callee', [dead, live],
        {'entry': BasicBlock('entry', [Instr('Ret', [live], None)])},
    )
    actual_dead, actual_live = SSAValue(10), SSAValue(11)
    caller = Function(
        'caller', [actual_dead, actual_live],
        {'entry': BasicBlock('entry', [Instr(
            'Call', [actual_dead, actual_live], None,
            attributes={'callee': 'callee', 'callee_input_ids': (1, 2)},
        )])},
    )
    table = SSACallTable(owner='prune-test', mutable=True)
    source_cell = book.latest_ref(SSA_VALUE, ('callee', dead.id))
    table.assign(
        'caller', (SSACallRecord(
            caller='caller', callsite_id=20, callee_reference=None,
            callee_name='callee', callee_symbol='callee',
            argument_bindings=((10, 1), (11, 2)),
            frame_bindings=((1, 'caller_value', 10),
                            (2, 'caller_value', 11)),
            unresolved_frame_value_ids=(1, 2),
        ),), sources=(source_cell,), stage=FRAME_BINDING,
    )
    incumbent = book.latest_ref(CALL_RECORD, (table.owner, 'caller'))

    assert _prune_unused_callee_formals(
        {'caller': caller, 'callee': callee}, table,
    ) == 1

    record = table['caller'][0]
    assert record.argument_bindings == ((11, 2),)
    assert record.frame_bindings == ((2, 'caller_value', 11),)
    assert record.unresolved_frame_value_ids == (2,)
    revised = book.latest_ref(CALL_RECORD, (table.owner, 'caller'))
    assert revised is not None and revised != incumbent
    assert incumbent in tuple(source for source, _stage in book.edges_into(revised))


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


def test_late_frame_tail_publishes_call_record_and_posted_physical_layout():
    from src.compiler.concordance_declarations import FRAME_TAIL

    book = current_identity_book()
    actual = SSAValue(20, 'int64')
    formal = SSAValue(1, 'int64')
    owner = SSAValue(7, 'ptr', shape=(1,), accounting={
        'heap_owner': True, 'compiler_frame_storage': 'helper',
    })
    roots = []
    for scope, value, physical in (
        ('root', actual, {'physical_dtype': 'int64', 'storage': 'scalar', 'rank': 0}),
        ('helper', formal, {'physical_dtype': 'int64', 'storage': 'scalar', 'rank': 0}),
        ('helper', owner, {'physical_dtype': 'ptr', 'storage': 'span', 'rank': 1}),
    ):
        roots.append(book.post(
            SSA_VALUE, (scope, value.id),
            SSAValueFact(value.dtype, tuple(value.shape or ()),
                         SSAValueOrigin.ADOPTED_GRAPH_ID, **physical),
            stage=CONTROL_SSA, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        ))
    call = Instr('Call', [actual], None, attributes={
        'callee': 'helper', 'callee_input_ids': (1,),
        'plan_callsite_id': 9, 'source_linked': True,
    })
    root = Function('root', [actual], {'entry': BasicBlock('entry', [call])})
    helper = Function('helper', [formal, owner], {'entry': BasicBlock('entry', [])})
    records = SSACallTable(owner='late-owner-record-fixture')
    original = records.assign('root', (SSACallRecord(
        caller='root', callsite_id=9, callee_reference=None,
        callee_name='helper', callee_symbol='helper',
        argument_bindings=((20, 1),), callee_storage_value_ids=(1,),
        frame_bindings=((1, 'caller_value', 20),), resolution='native_call',
    ),), sources=tuple(roots), stage=FRAME_BINDING)
    functions = {'root': root, 'helper': helper}

    assert _complete_propagated_frame_tails(functions, call_records=records) == 1
    slot = call.args[-1]
    assert slot is root.args[-1]
    physical = book.page(SSA_VALUE).latest(('root', slot.id))
    assert (physical.physical_dtype, physical.storage, physical.rank) == ('ptr', 'span', 1)
    assert slot.accounting['physical_dtype'] == 'ptr'
    record = tuple(records['root'])[0]
    assert record.callee_storage_value_ids == (1, 7)
    assert record.frame_bindings == ((1, 'caller_value', 20), (7, 'caller_storage', slot.id))
    assert record.unresolved_frame_value_ids == ()
    argument_ref = book.latest_ref(ARGUMENT_BINDING, ('helper', 'root', 7, 9))
    assert book.page(ARGUMENT_BINDING).latest(argument_ref.row) == ('caller_storage', slot.id)
    tail_ref = book.latest_ref(FRAME_TAIL, ('root', 9, 'helper', 7))
    current = book.latest_ref(CALL_RECORD, (records.owner, 'root'))
    assert {original, argument_ref, tail_ref} <= {
        source for source, _stage in book.edges_into(current)
    }
    assert _complete_propagated_frame_tails(functions, call_records=records) == 0
    assert book.latest_ref(CALL_RECORD, (records.owner, 'root')) == current

    page = book.page(FRAME_TAIL)
    key = (tail_ref.row, tail_ref.column)
    fact = page.cells[key]
    try:
        page.cells[key] = (slot.id + 1, fact[1])
        with pytest.raises(ValueError, match='disagrees with its live Call'):
            _complete_propagated_frame_tails(functions, call_records=records)
    finally:
        page.cells[key] = fact
