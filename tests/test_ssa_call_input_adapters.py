from src.compiler.ssa_call_input_adapters import (
    _adaptation_state_signature,
    adapt_physical_call_inputs,
    physical_call_input_conflicts,
)
from src.compiler.identity_concordance import begin_identity_book, end_identity_book
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock


def fixture():
    field = SSAValue(0, 'bool', accounting={'program_abi_storage': 'scalar', 'physical_dtype': 'bool'})
    formal = SSAValue(10, 'float64', accounting={'physical_dtype': 'bool'})
    region = Function('region', [formal], {'entry': BasicBlock('entry', [Instr('Ret', [formal], None)])},
                      metadata={'source_region_integral': {'owner': 'root'}})
    output = SSAValue(1, 'float64')
    call = Instr('Call', [field], output, attributes={'callee': 'region'})
    root = Function('root', [field], {'entry': BasicBlock('entry', [call, Instr('Ret', [output, field], None)])})
    return {'root': root, 'region': region}, field, formal, call


def test_region_gets_conversion_without_retyping_shared_physical_field():
    functions, field, formal, call = fixture()
    assert adapt_physical_call_inputs(functions) == 1
    cast = functions['root'].blocks['entry'].instrs[0]
    assert cast.op == 'Cast' and cast.args == [field]
    assert cast.res is call.args[0] and cast.res.dtype == 'float64'
    assert field.dtype == 'bool' and field.accounting['physical_dtype'] == 'bool'
    assert functions['root'].blocks['entry'].instrs[-1].args[-1] is field
    assert formal.accounting['physical_dtype'] == 'float64'
    assert adapt_physical_call_inputs(functions) == 0


def test_writable_region_feed_cannot_be_replaced_by_temporary():
    functions, field, formal, call = fixture()
    functions['region'].blocks['entry'].instrs.insert(0, Instr('Store', [formal, formal], None))
    assert adapt_physical_call_inputs(functions) == 0
    assert call.args[0] is field
    assert physical_call_input_conflicts(functions) == (('root', 0, 'bool', 'region', 10, 'float64'),)


def test_authored_callee_contract_is_not_a_generated_region_adapter():
    functions, field, formal, call = fixture()
    functions['region'].metadata.clear()
    assert adapt_physical_call_inputs(functions) == 0
    assert call.args[0] is field


def test_generated_scalar_call_input_gets_concordant_numeric_conversion():
    index = SSAValue(3, 'int64')
    formal = SSAValue(10, 'float64')
    result = SSAValue(4, 'float64')
    helper = Function('helper', [formal], {
        'entry': BasicBlock('entry', [Instr('Ret', [formal], None)]),
    })
    call = Instr('Call', [index], result, attributes={'callee': 'helper'})
    caller = Function('caller', [], {
        'entry': BasicBlock('entry', [call, Instr('Ret', [result], None)]),
    })
    functions = {'caller': caller, 'helper': helper}
    book, token = begin_identity_book()
    try:
        assert adapt_physical_call_inputs(functions) == 1
        cast, rewritten_call, _ret = caller.blocks['entry'].instrs
        assert cast.op == 'Cast'
        assert cast.args == [index]
        assert cast.res is rewritten_call.args[0]
        assert cast.res.dtype == 'float64'
        assert cast.attributes == {
            'target_dtype': 'float64',
            'source_dtype': 'int64',
            'concordant_call_input_conversion': True,
        }
        row = ('caller', 3, 'helper', 10)
        converted_id, source, target, kind = book.page(
            'call_input_conversion'
        ).latest(row)
        assert converted_id == int(cast.res.id)
        assert (source, target, kind) == (
            'int64', 'float64', 'read_only_scalar_numeric',
        )
        assert adapt_physical_call_inputs(functions) == 0
    finally:
        end_identity_book(token)


def test_settled_integer_scalar_is_adapted_to_broadcast_double_abi():
    index = SSAValue(3, 'int64')
    output = SSAValue(4, 'float64', (2,))
    shape = SSAValue(5, 'int32', (1,))
    rank = SSAValue(6, 'int32')
    call = Instr(
        'Call', [index, output, shape, rank, shape, rank], output,
        attributes={'callee': 'broadcast_double', 'ssa_output_argument': 1},
    )
    caller = Function('caller', [index], {
        'entry': BasicBlock('entry', [call, Instr('Ret', [output], None)]),
    })
    book, token = begin_identity_book()
    try:
        assert adapt_physical_call_inputs({'caller': caller}) == 1
        cast, rewritten_call, _ret = caller.blocks['entry'].instrs
        assert cast.op == 'Cast'
        assert cast.args == [index]
        assert cast.res is rewritten_call.args[0]
        assert cast.res.dtype == 'float64'
        row = ('caller', 'entry', output.id, 0)
        assert book.page('kernel_input_conversion').latest(row) == (
            index.id, cast.res.id, 'int64', 'float64', 'broadcast_double',
        )
        assert adapt_physical_call_inputs({'caller': caller}) == 0
    finally:
        end_identity_book(token)


def test_stale_same_id_region_capture_uses_incumbent_formal_type():
    field = SSAValue(47, 'bool', accounting={
        'ssa_call_dtype': 'bool',
        'ssa_call_result_source': ('controller', 65),
    })
    stale_capture = SSAValue(47, 'float64')
    helper_input = SSAValue(950, 'ptr', accounting={
        'physical_dtype': 'float64',
        'physical_dtype_provenance': 'typed_memory_operations',
    })
    helper = Function('cast_double_to_bool_values', [helper_input], {
        'entry': BasicBlock('entry', [Instr('Ret', [helper_input], None)]),
    })
    call = Instr('Call', [stale_capture], SSAValue(50, 'bool'), attributes={
        'callee': helper.name,
    })
    region = Function('region', [field], {
        'entry': BasicBlock('entry', [call, Instr('Ret', [call.res], None)]),
    }, metadata={'source_region_integral': {'owner': 'root'}})
    functions = {region.name: region, helper.name: helper}

    assert adapt_physical_call_inputs(functions) == 1

    cast, rewritten_call, _ret = region.blocks['entry'].instrs
    assert cast.op == 'Cast'
    assert cast.args == [field]
    assert cast.attributes == {
        'target_dtype': 'float64',
        'source_dtype': 'bool',
        'physical_region_input_conversion': True,
    }
    assert rewritten_call is call
    assert call.args[0] is cast.res
    assert region.metadata['canonicalized_formal_use_ids'] == (47,)
    assert adapt_physical_call_inputs(functions) == 0


def test_exact_region_feed_restores_physical_call_view_after_projection():
    produced = SSAValue(30, 'bool', (2,), accounting={
        'physical_dtype': 'float64',
        'physical_dtype_provenance': ('callee_output', 'producer', 30),
    })
    stale_view = SSAValue(30, 'bool', (2,), accounting={
        'ssa_storage_alias': 30,
        'ssa_region_feed': (15, 0),
    })
    formal = SSAValue(40, 'bool', (2,))
    region = Function('region', [formal], {
        'entry': BasicBlock('entry', [Instr('Ret', [formal], None)]),
    }, metadata={'source_region_integral': {'owner': 'root'}})
    call = Instr('Call', [stale_view], SSAValue(50, 'ssa.aggregate'), attributes={
        'callee': region.name,
        'feed_ids': (37,),
        'feed_dtypes': ('float64',),
        'result_convention': 'ssa.aggregate',
    })
    caller = Function('caller', [], {
        'entry': BasicBlock('entry', [
            Instr('Identity', [], produced),
            call,
            Instr('Ret', [], None),
        ]),
    })
    functions = {caller.name: caller, region.name: region}
    book, token = begin_identity_book()
    try:
        assert adapt_physical_call_inputs(functions) == 2
        exact_view = call.args[0]
        assert exact_view is not stale_view
        assert exact_view.id == produced.id
        assert exact_view.dtype == 'float64'
        assert exact_view.accounting['exact_region_feed_dtype'] == 'float64'
        assert formal.dtype == 'float64'
        assert formal.accounting['exact_region_feed_dtype'] == 'float64'
        assert produced.dtype == 'bool'
        assert adapt_physical_call_inputs(functions) == 0
        assert book.page('exact_region_feed_dtype').latest(
            ('feed', 'caller', 37)
        ) == ('float64',)
        assert book.page('exact_region_feed_dtype').latest(
            ('formal', 'region', 40)
        ) == ('float64',)
    finally:
        end_identity_book(token)


def test_exact_region_feed_view_survives_formal_use_interning():
    """The concorded feed view must be installed after storage interning."""

    storage = SSAValue(30, 'bool', (2,), accounting={
        'physical_dtype': 'float64',
        'program_abi_storage': 'span',
    })
    stale_view = SSAValue(30, 'bool', (2,))
    formal = SSAValue(40, 'bool', (2,))
    region = Function('region', [formal], {
        'entry': BasicBlock('entry', [Instr('Ret', [formal], None)]),
    }, metadata={'source_region_integral': {'owner': 'root'}})
    call = Instr('Call', [stale_view], None, attributes={
        'callee': region.name,
        'feed_ids': (37,),
        'feed_dtypes': ('float64',),
    })
    caller = Function('caller', [storage], {
        'entry': BasicBlock('entry', [call, Instr('Ret', [], None)]),
    })
    functions = {caller.name: caller, region.name: region}

    assert adapt_physical_call_inputs(functions) == 2
    assert call.args[0].id == storage.id
    assert call.args[0].dtype == 'float64'
    assert call.args[0].accounting['exact_region_feed_dtype'] == 'float64'
    assert adapt_physical_call_inputs(functions) == 0


def test_cycle_signature_quotients_fresh_conversion_temporary_ids():
    source = SSAValue(1, 'bool', accounting={'physical_dtype': 'bool'})
    first = SSAValue(2, 'float64', accounting={'physical_dtype': 'float64'})
    formal = SSAValue(10, 'float64')
    cast = Instr('Cast', [source], first, attributes={
        'physical_region_input_conversion': True,
    })
    call = Instr('Call', [first], None, attributes={'callee': 'callee'})
    caller = Function('caller', [source], {
        'entry': BasicBlock('entry', [cast, call, Instr('Ret', [], None)]),
    })
    callee = Function('callee', [formal], {
        'entry': BasicBlock('entry', [Instr('Ret', [formal], None)]),
    })
    functions = {caller.name: caller, callee.name: callee}
    first_signature = _adaptation_state_signature(functions)

    second = SSAValue(3, 'float64', accounting={'physical_dtype': 'float64'})
    nested = Instr('Cast', [first], second, attributes={
        'physical_region_input_conversion': True,
    })
    caller.blocks['entry'].instrs.insert(1, nested)
    call.args[0] = second

    assert _adaptation_state_signature(functions) == first_signature


def test_repository_output_position_outvotes_semantic_output_identity():
    """A semantic publication does not make a different kernel input writable."""
    mask = SSAValue(49, 'bool', (15,), accounting={
        'program_abi_storage': 'span',
        'physical_dtype': 'bool',
    })
    condition = SSAValue(334, 'ptr', accounting={
        'physical_dtype': 'float64',
    })
    output = SSAValue(337, 'ptr', accounting={
        'physical_dtype': 'float64',
    })
    count = SSAValue(338, 'int32')
    condition_ptr = SSAValue(339, 'ptr')
    condition_value = SSAValue(340, 'float64')
    helper = Function('where_double', [condition, output, count], {
        'entry': BasicBlock('entry', [
            Instr('GetElementPtr', [condition, count], condition_ptr),
            Instr('Load', [condition_ptr], condition_value),
            Instr('Store', [condition_value, output], None),
            Instr('Ret', [], None),
        ]),
    })
    result = SSAValue(50, 'float64', (15,))
    call = Instr('Call', [mask, result, count], result, attributes={
        'callee': helper.name,
        # This is source/publication provenance copied through tensor
        # lowering.  It is deliberately the mask identity from the observed
        # failure; the physical kernel output is argument one.
        'output_ids': (mask.id,),
        'ssa_output_argument': 1,
    })
    caller = Function('planned_region', [mask], {
        'entry': BasicBlock('entry', [call, Instr('Ret', [result], None)]),
    }, metadata={'source_region_integral': {'owner': 'step_2'}})
    functions = {caller.name: caller, helper.name: helper}
    book, token = begin_identity_book()
    try:
        assert adapt_physical_call_inputs(functions) == 1
        cast, rewritten_call, _ret = caller.blocks['entry'].instrs
        assert cast.op == 'Cast'
        assert cast.args == [mask]
        assert rewritten_call.args[0] is cast.res
        assert rewritten_call.args[1] is result
        row = (caller.name, mask.id, helper.name, condition.id)
        converted_id, source, target, kind = book.page(
            'call_input_conversion'
        ).latest(row)
        assert converted_id == cast.res.id
        assert (source, target, kind) == (
            'bool', 'float64', 'physical_region',
        )
        assert physical_call_input_conflicts(functions) == ()
    finally:
        end_identity_book(token)


def test_pointer_contract_compares_element_type_instead_of_address_marker():
    functions, field, formal, _ = fixture()
    formal.dtype = 'ptr'
    formal.accounting['physical_dtype'] = 'bool'
    assert physical_call_input_conflicts(functions) == ()
    formal.accounting['physical_dtype'] = 'float64'
    assert physical_call_input_conflicts(functions) == (('root', 0, 'bool', 'region', 10, 'float64'),)
    field.accounting['physical_dtype'] = 'float64'
    assert physical_call_input_conflicts(functions) == ()
    assert field.dtype == 'bool'


def test_native_region_conversion_preserves_boolean_input_buffer(tmp_path):
    import pickle
    import subprocess
    import sys
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.transmogrifier.ssa import IRModule

    functions, field, _, _ = fixture()
    mask = SSAValue(4, 'bool', (4,), accounting={'program_abi_storage': 'span', 'physical_dtype': 'bool'})
    vector_formal = SSAValue(10, 'bool', (4,), accounting={'physical_dtype': 'float64'})
    vector_result = SSAValue(5, 'float64', (4,))
    functions['vector'] = Function('vector', [vector_formal], {'entry': BasicBlock('entry', [
        Instr('Ret', [vector_formal], None),
    ])})
    functions['root'].args.append(mask)
    body = functions['root'].blocks['entry'].instrs
    body.insert(-1, Instr('Call', [mask], vector_result, attributes={'callee': 'vector'}))
    body[-1].args.append(vector_result)
    adapt_physical_call_inputs(functions)
    artifact = emit_ssa_module_to_c(IRModule(functions), 'root')
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native', optimization='O0')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps(artifact))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact = pickle.load(open(sys.argv[1], 'rb'))
execution = artifact.prepare_execution({0: np.array([False], dtype=np.bool_), 4: np.zeros(4, dtype=np.bool_)})
for value in (False, True, False, True):
    execution.buffers[0][0] = value
    mask = np.array([value, not value, value, not value])
    execution.buffers[4][:] = mask
    execution.run()
    assert execution.buffers[0].dtype == np.bool_
    assert execution.buffers[0].item() == value
    assert execution.buffers[1].item() == float(value)
    assert execution.buffers[4].dtype == np.bool_
    np.testing.assert_array_equal(execution.buffers[4], mask)
    np.testing.assert_array_equal(execution.buffers[5], mask.astype(np.float64))
''', str(payload)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
