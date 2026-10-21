from src.compiler.ssa_result_type_resolution import settle_call_result_types, equivalent_physical_layout
from src.compiler.ssa_self_check import check_structural_outputs
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule


import pytest

from src.compiler.transformation_priority import TransformationConflict

from tests.compiler_provenance_inputs import publish_authored_ssa
from src.compiler.identity_concordance import begin_identity_book as _begin_book, end_identity_book as _end_book

@pytest.fixture(autouse=True)
def _fresh_compile_book():
    # One compile book per test, as the real pipeline opens one per compile.
    _book, token = _begin_book()
    try:
        yield
    finally:
        _end_book(token)

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


def test_equal_priority_producer_disagreement_refuses_as_conflict():
    result = SSAValue(0, 'unknown')
    fs = {'root': function('root', result, ('first', 'second')),
          'first': function('first', SSAValue(1, 'float64')),
          'second': function('second', SSAValue(2, 'int64'))}
    # stands in for precompile_to_ssa's _mint_ssa_id/_value_cell rows
    publish_authored_ssa(fs)
    # Rule: an equal-priority disagreement between producers is a discordance; arrival order never picks a winner.
    with pytest.raises(TransformationConflict) as refusal:
        settle_call_result_types(fs, outputs, values)
    message = str(refusal.value)
    assert "('root', 0)" in message
    assert "('float64', (), None)" in message and "('int64', (), None)" in message
    assert "'first'" in message and "'second'" in message


def test_stronger_physical_proof_propagates_through_reverse_order_chain():
    physical = SSAValue(3, 'bool', accounting={'physical_dtype': 'bool'})
    fs = {'outer': function('outer', SSAValue(0, 'float64'), ('middle',)),
          'middle': function('middle', SSAValue(1, 'float64'), ('inner',)),
          'inner': function('inner', physical)}
    # stands in for precompile_to_ssa's _mint_ssa_id/_value_cell rows
    publish_authored_ssa(fs)
    _, _, conflicts, _ = settle_call_result_types(fs, outputs, values)
    assert not conflicts
    assert outputs('outer', fs['outer'])[0].dtype == 'bool'


def test_contiguous_result_views_keep_incumbent_shape_without_cycle():
    result = SSAValue(0, 'float64', (8, 1), accounting={'physical_dtype': 'float64'})
    fs = {'root': function('root', result, ('producer',)),
          'producer': function('producer', SSAValue(1, 'float64', (8,)))}
    # stands in for precompile_to_ssa's _mint_ssa_id/_value_cell rows
    publish_authored_ssa(fs)
    _, _, conflicts, _ = settle_call_result_types(fs, outputs, values)
    assert result.shape == (8, 1) and not conflicts
    assert not equivalent_physical_layout(('float64', (8,), None), ('float64', (9,), None))


def test_runtime_recursion_does_not_make_type_propagation_oscillate():
    fs = {'a': function('a', SSAValue(0, 'float64'), ('b',)),
          'b': function('b', SSAValue(1, 'int64'), ('a',))}
    # stands in for precompile_to_ssa's _mint_ssa_id/_value_cell rows
    publish_authored_ssa(fs)
    _, _, conflicts, _ = settle_call_result_types(fs, outputs, values)
    assert not conflicts
    assert outputs('a', fs['a'])[0].dtype == outputs('b', fs['b'])[0].dtype


def test_storage_label_without_known_dtype_is_not_a_physical_type_proof():
    result = SSAValue(0, 'unknown', accounting={'program_abi_storage': 'scalar'})
    fs = {'root': function('root', result, ('producer',)),
          'producer': function('producer', SSAValue(1, 'bool', accounting={'physical_dtype': 'bool'}))}
    # stands in for precompile_to_ssa's _mint_ssa_id/_value_cell rows
    publish_authored_ssa(fs)
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
    # stands in for precompile_to_ssa's _mint_ssa_id/_value_cell rows
    publish_authored_ssa(fs)
    exact, _, conflicts, _ = settle_call_result_types(fs, outputs, values)
    assert result.dtype == 'bool' and conflicts
    assert id(result) not in exact
    assert 'ssa_call_result_from' not in result.accounting


def test_inferred_projection_inherits_proven_buffer_representation():
    result = SSAValue(0, 'bool', (4,))
    source = SSAValue(1, 'bool', (4,), accounting={'physical_dtype': 'float64'})
    fs = {'root': function('root', result, ('producer',)),
          'producer': function('producer', source)}
    # stands in for precompile_to_ssa's _mint_ssa_id/_value_cell rows
    publish_authored_ssa(fs)
    _, _, conflicts, _ = settle_call_result_types(fs, outputs, values)
    assert result.dtype == 'bool'
    assert result.accounting['physical_dtype'] == 'float64'
    assert not conflicts


def test_repeated_region_projection_receives_its_own_physical_contract():
    source = SSAValue(4, 'bool', (1,), accounting={'physical_dtype': 'float64'})
    first, repeated = SSAValue(4, 'bool', (1,)), SSAValue(8, 'bool', (1,))
    aggregate, pointer = SSAValue(9, 'ssa.aggregate'), SSAValue(10, 'ptr')
    index = SSAValue(11, 'int64')
    root = Function('root', [first], {'entry': BasicBlock('entry', [
        Instr('Call', [], aggregate, attributes={
            'callee': 'producer', 'result_convention': 'ssa.aggregate', 'output_ids': (4,),
        }),
        Instr('Const', [], index, attributes={'value': 0}),
        Instr('GetElementPtr', [aggregate, index], pointer, attributes={'source_output_id': 4}),
        Instr('Load', [pointer], repeated), Instr('Ret', [repeated], None),
    ])})
    fs = {'root': root, 'producer': function('producer', source)}
    # stands in for precompile_to_ssa's _mint_ssa_id/_value_cell rows
    publish_authored_ssa(fs)
    _, _, conflicts, _ = settle_call_result_types(fs, outputs, values)
    assert not conflicts
    assert repeated.accounting['physical_dtype'] == 'float64'
    assert repeated.accounting['ssa_call_result_from'] == ('producer', 4)


def test_projection_uses_caller_output_identity_after_dead_output_removal():
    left, right = SSAValue(21, 'float64', (3,)), SSAValue(22, 'float64', (7,))
    first, second = SSAValue(4, 'unknown'), SSAValue(5, 'unknown')
    repeated = SSAValue(8, 'unknown')
    aggregate, pointer, index = SSAValue(9, 'ssa.aggregate'), SSAValue(10, 'ptr'), SSAValue(11, 'int64')
    root = Function('root', [first, second], {'entry': BasicBlock('entry', [
        Instr('Call', [], aggregate, attributes={
            'callee': 'producer', 'result_convention': 'ssa.aggregate',
            'output_ids': (4, 5), 'callee_output_ids': (21, 22),
        }),
        Instr('Const', [], index, attributes={'value': 1}),
        # Slot zero was removed. Original slot one still names caller output 4.
        Instr('GetElementPtr', [aggregate, index], pointer,
              attributes={'source_output_id': 4, 'aggregate_index': 1}),
        Instr('Load', [pointer], repeated), Instr('Ret', [repeated], None),
    ])})
    producer = Function('producer', [left, right], {'entry': BasicBlock('entry', [
        Instr('Ret', [left, right], None),
    ])})
    # stands in for precompile_to_ssa's _mint_ssa_id/_value_cell rows
    publish_authored_ssa({'root': root, 'producer': producer})
    _, _, conflicts, _ = settle_call_result_types({'root': root, 'producer': producer}, outputs, values)
    assert not conflicts
    assert repeated.shape == (3,)
    assert repeated.accounting['ssa_call_result_from'] == ('producer', 21)


def test_accepted_component_is_a_source_not_an_ambient_settlement_target():
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book

    accepted = function('accepted', SSAValue(1, 'int64'))
    owner, owner_token = begin_identity_book()
    try:
        accepted.metadata['accepted_ssa_component'] = {
            'schema': 'turing.accepted-ssa-component.v1',
            'artifact_sha256': 'accepted-fixture',
            'identity_book': owner,
        }
    finally:
        end_identity_book(owner_token)

    ambient, ambient_token = begin_identity_book()
    try:
        settle_call_result_types({'accepted': accepted}, outputs, values)
        assert not ambient.page('transformation_decision').rows()
    finally:
        end_identity_book(ambient_token)

def test_caller_derives_from_accepted_components_imported_proof():
    import hashlib
    digest = hashlib.sha256(b'accepted-fixture').hexdigest()
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book

    source = SSAValue(1, 'bool', accounting={'physical_dtype': 'bool'})
    accepted = function('accepted', source)
    owner, owner_token = begin_identity_book()
    try:
        # The accepted body's rows live on its owner book, as when it was compiled.
        # stands in for precompile_to_ssa's _mint_ssa_id/_value_cell rows
        publish_authored_ssa({'accepted': accepted})
        accepted.metadata['accepted_ssa_component'] = {
            'schema': 'turing.accepted-ssa-component.v1',
            'artifact_sha256': digest,
            'identity_book': owner,
        }
    finally:
        end_identity_book(owner_token)
    result = SSAValue(0, 'float64')
    caller = function('caller', result, ('accepted',))

    ambient, ambient_token = begin_identity_book()
    try:
        # stands in for precompile_to_ssa's _mint_ssa_id/_value_cell rows
        publish_authored_ssa({'caller': caller})
        _, _, conflicts, _ = settle_call_result_types(
            {'caller': caller, 'accepted': accepted}, outputs, values,
        )
        assert result.dtype == 'bool'
        assert not conflicts
        # The accepted body's output reaches the caller through the digest-qualified import edge.
        receipts = ambient.page('concordance_imported_cell').rows()
        assert receipts and all(row[0] == digest for row in receipts)
    finally:
        end_identity_book(ambient_token)
