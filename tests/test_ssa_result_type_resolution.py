from src.compiler.ssa_result_type_resolution import settle_call_result_types, equivalent_physical_layout
from src.compiler.ssa_self_check import check_structural_outputs
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule


def values(function):
    result = {value.id: value for value in function.args}
    for block in function.blocks.values():
        for instruction in block.instrs:
            for value in instruction.args:
                result[value.id] = value
            if instruction.res is not None:
                result[instruction.res.id] = instruction.res
    return result


def outputs(name, function):
    return next(instruction.args for block in function.blocks.values()
                for instruction in block.instrs if instruction.op == 'Ret')


def function(name, value, calls=()):
    instructions = [Instr('Call', [], value, attributes={
        'callee': callee, 'result_convention': 'ssa.aggregate', 'output_ids': (value.id,),
    }) for callee in calls]
    return Function(name, [], {'entry': BasicBlock('entry', [*instructions, Instr('Ret', [value], None)])})


def test_equal_priority_producer_keeps_incumbent_and_records_conflict():
    result = SSAValue(0, 'unknown')
    fs = {'root': function('root', result, ('first', 'second')),
          'first': function('first', SSAValue(1, 'float64')),
          'second': function('second', SSAValue(2, 'int64'))}
    _, events, conflicts, rounds = settle_call_result_types(fs, outputs, values)
    assert result.dtype == 'float64'
    assert len(conflicts) == 1 and conflicts[0]['callee'] == 'second'
    assert any(event.get('reason') == 'incumbent_tie' for event in events)
    module = IRModule(fs, metadata={'call_result_type_conflicts': conflicts})
    assert check_structural_outputs(module)[0].check == 'call_result_contract'


def test_stronger_physical_proof_propagates_through_reverse_order_chain():
    physical = SSAValue(3, 'bool', accounting={'physical_dtype': 'bool'})
    fs = {'outer': function('outer', SSAValue(0, 'float64'), ('middle',)),
          'middle': function('middle', SSAValue(1, 'float64'), ('inner',)),
          'inner': function('inner', physical)}
    _, _, conflicts, _ = settle_call_result_types(fs, outputs, values)
    assert not conflicts
    assert outputs('outer', fs['outer'])[0].dtype == 'bool'


def test_contiguous_result_views_keep_incumbent_shape_without_cycle():
    result = SSAValue(0, 'float64', (8, 1), accounting={'physical_dtype': 'float64'})
    fs = {'root': function('root', result, ('producer',)),
          'producer': function('producer', SSAValue(1, 'float64', (8,)))}
    _, _, conflicts, _ = settle_call_result_types(fs, outputs, values)
    assert result.shape == (8, 1) and not conflicts
    assert not equivalent_physical_layout(('float64', (8,), None), ('float64', (9,), None))


def test_runtime_recursion_does_not_make_type_propagation_oscillate():
    fs = {'a': function('a', SSAValue(0, 'float64'), ('b',)),
          'b': function('b', SSAValue(1, 'int64'), ('a',))}
    _, _, conflicts, _ = settle_call_result_types(fs, outputs, values)
    assert not conflicts
    assert outputs('a', fs['a'])[0].dtype == outputs('b', fs['b'])[0].dtype


def test_storage_label_without_known_dtype_is_not_a_physical_type_proof():
    result = SSAValue(0, 'unknown', accounting={'program_abi_storage': 'scalar'})
    fs = {'root': function('root', result, ('producer',)),
          'producer': function('producer', SSAValue(1, 'bool', accounting={'physical_dtype': 'bool'}))}
    _, _, conflicts, _ = settle_call_result_types(fs, outputs, values)
    assert result.dtype == 'bool' and not conflicts


def test_direct_c_emission_keeps_result_contract_failure_visible():
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    result = SSAValue(0, 'float64')
    root = Function('root', [], {'entry': BasicBlock('entry', [
        Instr('Const', [], result, attributes={'value': 0.0}), Instr('Ret', [result], None),
    ])})
    module = IRModule({'root': root}, metadata={'call_result_type_conflicts': ({
        'caller': 'root', 'value_id': 0, 'retained': ('float64', (), None),
        'callee': 'producer', 'callee_value_id': 1, 'proposed': ('none', (), None),
    },)})
    artifact = emit_ssa_module_to_c(module, 'root')
    assert not artifact.complete
    assert any(item.operation == 'call_result_contract' for item in artifact.shortfalls)


def test_rejected_physical_incumbent_does_not_claim_callee_provenance():
    result = SSAValue(0, 'bool', accounting={'physical_dtype': 'bool'})
    fs = {'root': function('root', result, ('producer',)),
          'producer': function('producer', SSAValue(1, 'float64'))}
    exact, _, conflicts, _ = settle_call_result_types(fs, outputs, values)
    assert result.dtype == 'bool' and conflicts
    assert id(result) not in exact
    assert 'ssa_call_result_from' not in result.accounting


def test_inferred_projection_inherits_proven_buffer_representation():
    result = SSAValue(0, 'bool', (4,))
    source = SSAValue(1, 'bool', (4,), accounting={'physical_dtype': 'float64'})
    fs = {'root': function('root', result, ('producer',)),
          'producer': function('producer', source)}
    _, _, conflicts, _ = settle_call_result_types(fs, outputs, values)
    assert result.dtype == 'bool'
    assert result.accounting['physical_dtype'] == 'float64'
    assert not conflicts
