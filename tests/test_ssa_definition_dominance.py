from src.compiler.ssa_reachability import hoist_nondominating_constants
from src.compiler.ssa_self_check import check_definition_dominance, check_phi_cfg_contract
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule


def module(blocks, args=()):
    return IRModule({'root': Function('root', list(args), blocks)})


def test_return_before_its_producer_is_reported():
    value = SSAValue(1, 'float64')
    result = module({
        'entry': BasicBlock('entry', [Instr('Ret', [value], None)]),
        'unreachable': BasicBlock('unreachable', [Instr('Const', [], value, attributes={'value': 1.0})]),
    })
    findings = check_definition_dominance(result)
    assert len(findings) == 1
    assert 'reads %1' in findings[0].detail


def test_named_entry_wins_over_an_earlier_inserted_dead_block():
    value = SSAValue(1, 'float64')
    result = module({
        'dead': BasicBlock('dead', [Instr('Const', [], value)]),
        'entry': BasicBlock('entry', [Instr('Ret', [value], None)]),
    })
    findings = check_definition_dominance(result)
    assert len(findings) == 1
    assert 'reads %1' in findings[0].detail


def test_phi_contract_requires_exact_operand_predecessor_correspondence():
    left, right, merged = (SSAValue(i, 'float64') for i in (1, 2, 3))
    result = module({
        'entry': BasicBlock('entry', [], successors=['yes', 'no']),
        'yes': BasicBlock('yes', [Instr('Const', [], left)], successors=['merge']),
        'no': BasicBlock('no', [Instr('Const', [], right)], successors=['merge']),
        'merge': BasicBlock('merge', [
            Instr('Phi', [left, right], merged, attributes={'incoming_blocks': ('yes',)}),
            Instr('Ret', [merged], None),
        ]),
    })
    findings = check_phi_cfg_contract(result)
    assert len(findings) == 1
    assert '2 operands but 1 incoming labels' in findings[0].detail
    assert "['yes'] do not equal reachable predecessors ['no', 'yes']" in findings[0].detail


def test_phi_reads_its_arm_on_incoming_edge_but_direct_read_fails():
    left, right, merged = (SSAValue(i, 'float64') for i in (1, 2, 3))
    result = module({
        'entry': BasicBlock('entry', [], successors=['yes', 'no']),
        'yes': BasicBlock('yes', [Instr('Const', [], left)], successors=['merge']),
        'no': BasicBlock('no', [Instr('Const', [], right)], successors=['merge']),
        'merge': BasicBlock('merge', [
            Instr('Phi', [left, right], merged, attributes={'incoming_blocks': ('yes', 'no')}),
            Instr('Ret', [merged], None),
        ]),
    })
    assert check_definition_dominance(result) == []
    result.functions['root'].blocks['merge'].instrs[-1].args = [left]
    assert len(check_definition_dominance(result)) == 1


def test_same_block_read_before_definition_is_not_hidden_by_later_definition():
    value, output = SSAValue(1, 'float64'), SSAValue(2, 'float64')
    result = module({'entry': BasicBlock('entry', [
        Instr('Add', [value, value], output), Instr('Const', [], value), Instr('Ret', [output], None),
    ])})
    assert len(check_definition_dominance(result)) == 2


def test_operand_free_constant_is_hoisted_from_late_exit_with_provenance():
    value, output = SSAValue(1, 'int64'), SSAValue(2, 'int64')
    function = Function('root', [], {
        'entry': BasicBlock('entry', [], successors=['body']),
        'body': BasicBlock('body', [Instr('Add', [value, value], output)], successors=['exit']),
        'exit': BasicBlock('exit', [
            Instr('Const', [], value, attributes={'value': []}),
            Instr('Ret', [output], None),
        ]),
    })
    result = module(function.blocks)

    assert len(check_definition_dominance(result)) == 2
    from tests.compiler_provenance_inputs import publish_authored_ssa
    publish_authored_ssa({function.name: function})
    receipts = hoist_nondominating_constants(function)

    assert check_definition_dominance(result) == []
    assert function.blocks['entry'].instrs[0].res is value
    assert receipts == ({
        'value_id': 1,
        'from_block': 'exit',
        'from_index': 0,
        'to_block': 'entry',
        'priority': 'operand_free_immutable_definition',
        'tie_policy': 'incumbent',
    },)
    assert hoist_nondominating_constants(function) == ()


def test_accepted_component_constant_hoist_uses_its_provenance_owner():
    from src.compiler.concordance_declarations import (
        SSA_BLOCK_GEOMETRY,
        SSA_CONSTANT_HOIST,
    )
    from src.compiler.identity_concordance import (
        IdentityBook,
        begin_identity_book,
        end_identity_book,
    )
    from src.compiler.ssa_record_return_state import function_scope_of
    from tests.compiler_provenance_inputs import publish_authored_ssa

    value, output = SSAValue(11, 'int64'), SSAValue(12, 'int64')
    function = Function('linked', [], {
        'entry': BasicBlock('entry', [], successors=['body']),
        'body': BasicBlock('body', [
            Instr('Add', [value, value], output),
        ], successors=['exit']),
        'exit': BasicBlock('exit', [
            Instr('Const', [], value, attributes={'value': 1}),
            Instr('Ret', [output], None),
        ]),
    })
    owner, owner_token = begin_identity_book(IdentityBook())
    try:
        publish_authored_ssa({function.name: function})
    finally:
        end_identity_book(owner_token)
    function.metadata['accepted_ssa_component'] = {
        'schema': 'turing.accepted-ssa-component.v1',
        'artifact_sha256': 'dependency-digest',
        'unit_index': 44,
        'identity_book': owner,
    }

    ambient, ambient_token = begin_identity_book(IdentityBook())
    try:
        receipts = hoist_nondominating_constants(function)
    finally:
        end_identity_book(ambient_token)

    scope = function_scope_of(function)
    assert receipts[0]['value_id'] == 11
    assert owner.page(SSA_BLOCK_GEOMETRY).latest(
        (scope, function.name, 'entry')
    ) is not None
    assert tuple(owner.page(SSA_CONSTANT_HOIST).scope_rows(
        (scope, function.name)
    ))
    assert ambient.page(SSA_BLOCK_GEOMETRY).latest(
        (scope, function.name, 'entry')
    ) is None
    assert tuple(ambient.page(SSA_CONSTANT_HOIST).scope_rows(
        (scope, function.name)
    )) == ()


def test_source_literal_constant_hoist_uses_canonical_node_identity():
    import networkx as nx

    from src.compiler.concordance_declarations import (
        CANONICAL_VALUE,
        INGEST_SOURCE,
        INGESTION,
        SSA_CONSTANT_HOIST,
    )
    from src.compiler.identity_concordance import (
        IdentityBook,
        Mode,
        Novel,
        begin_identity_book,
        end_identity_book,
    )
    from src.compiler.ssa_reachability import record_constant_hoists
    from src.compiler.ssa_record_return_state import function_scope_of
    from tests.compiler_provenance_inputs import publish_authored_ssa

    function = Function('root', [], {
        'entry': BasicBlock('entry', [], successors=['exit']),
        'exit': BasicBlock('exit', [Instr('Ret', [], None)]),
    })
    book, token = begin_identity_book(IdentityBook())
    try:
        publish_authored_ssa({function.name: function})
        scope = function_scope_of(function)
        book.post(
            CANONICAL_VALUE, (scope, 23), (), stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
        )
        source_graph = nx.DiGraph()
        source_graph.add_node(23, value_id=23, type='Constant')
        source_graph.graph.update({
            'canonical_value_ids': True,
            'lexical_read_scope': scope,
        })
        literal = SSAValue(23, 'bool')
        instruction = Instr('Const', [], literal, attributes={'value': False})
        function.blocks['exit'].instrs.insert(0, instruction)

        record_constant_hoists(
            function, (('exit', 0, instruction, 'entry', 0),), source_graph,
        )

        assert tuple(book.page(SSA_CONSTANT_HOIST).scope_rows(
            (scope, function.name)
        ))
    finally:
        end_identity_book(token)
