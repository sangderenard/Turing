"""Source-conditional membership must use authored syntax, not AST tokens."""

from __future__ import annotations

import ast
import copy
from types import SimpleNamespace

import networkx as nx

from src.compiler.glsl_deployment_strategy import (
    _optional_presence_control_expression,
    _repair_missing_phi_initial_identities,
    _branch_compartments,
    _record_field_state_keys,
)
from src.compiler.glsl_deployment_strategy import _retain_source_sequence_mutation_records
from src.compiler.fortran_c_shell import (
    _apply_phi_initial_identity_repairs_to_control,
)


def _optional_field_graph(*, duplicate_presence=False):
    graph = nx.DiGraph()
    graph.graph['identity_table'] = {'limit': (2,)}
    graph.add_node(1, type='Input', value_id=1, parents=[], attributes={
        'binding_name': 'self', 'binding_kind': 'parameter',
    })
    graph.add_node(2, type='GetAttr', value_id=2, parents=[(1, 'value')],
                   attributes={'attribute': 'limit'})
    graph.add_node(3, type='Input', value_id=3, parents=[], attributes={
        'program_abi_optional_presence': True,
        'program_abi_optional_present_when': True,
        'program_abi_parameter': 'self',
        'program_abi_field': 'limit',
    })
    if duplicate_presence:
        graph.add_node(4, type='Input', value_id=4, parents=[], attributes={
            'program_abi_optional_presence': True,
            'program_abi_optional_present_when': True,
            'program_abi_parameter': 'self',
            'program_abi_field': 'limit',
        })
    return graph


def test_erased_optional_none_test_uses_exact_presence_slot():
    graph = _optional_field_graph()
    is_present = ast.parse('limit is not None', mode='eval').body
    is_absent = ast.parse('limit is None', mode='eval').body

    present = _optional_presence_control_expression(graph, is_present)
    absent = _optional_presence_control_expression(graph, is_absent)

    assert (present.op, present.value_id) == ('value', 3)
    assert absent.op == 'not'
    assert absent.operands[0].value_id == 3
    assert graph.graph['optional_presence_control_receipts'] == ({
        'name': 'limit',
        'payload_value_id': 2,
        'presence_value_id': 3,
        'parameter': 'self',
        'field': 'limit',
        'comparison': 'is_not_none',
        'predicate': 'presence',
        'priority': 'exact_program_abi_optional_presence',
        'tie_policy': 'incumbent',
    }, {
        'name': 'limit',
        'payload_value_id': 2,
        'presence_value_id': 3,
        'parameter': 'self',
        'field': 'limit',
        'comparison': 'is_none',
        'predicate': 'not_presence',
        'priority': 'exact_program_abi_optional_presence',
        'tie_policy': 'incumbent',
    })


def test_ambiguous_optional_presence_keeps_unresolved_incumbent():
    graph = _optional_field_graph(duplicate_presence=True)
    expression = ast.parse('limit is not None', mode='eval').body

    assert _optional_presence_control_expression(graph, expression) is None
    assert 'optional_presence_control_receipts' not in graph.graph


def test_missing_phi_initial_uses_unique_nearest_identity_ancestor():
    graph = nx.DiGraph()
    graph.graph['identity_table'] = {'value': (9, 10, 20, 40)}
    graph.add_node(9, value_id=9, type='Input', parents=[])
    graph.add_node(10, value_id=10, type='LoopResult', parents=[(9, 'value')])
    graph.add_node(20, value_id=20, type='maximum', parents=[(10, 'arg:0')])
    graph.add_node(30, value_id=30, type='LoopResult', parents=[(10, 'value')])
    graph.add_node(40, value_id=40, type='Phi', parents=[
        (20, 'body'), (30, 'orelse'),
    ], attributes={
        'binding_name': 'value', 'initial_value_id': 99,
    })

    receipts = _repair_missing_phi_initial_identities(graph)

    assert receipts == ({
        'phi_value_id': 40,
        'binding_name': 'value',
        'missing_initial_value_id': 99,
        'resident_initial_value_id': 10,
        'dataflow_distance': 2,
        'priority': 'exact_common_dataflow_ancestor',
        'tie_policy': 'incumbent',
    },)
    assert graph.nodes[40]['attributes']['initial_value_id'] == 10
    assert _repair_missing_phi_initial_identities(graph) == ()


def test_equal_phi_initial_candidates_do_not_replace_missing_incumbent():
    graph = nx.DiGraph()
    graph.graph['identity_table'] = {'value': (10, 11, 20, 40)}
    graph.add_node(10, value_id=10, type='Input', parents=[])
    graph.add_node(11, value_id=11, type='Input', parents=[])
    graph.add_node(20, value_id=20, type='Add', parents=[
        (10, 'arg:0'), (11, 'arg:1'),
    ])
    graph.add_node(30, value_id=30, type='Add', parents=[
        (10, 'arg:0'), (11, 'arg:1'),
    ])
    graph.add_node(40, value_id=40, type='Phi', parents=[
        (20, 'body'), (30, 'orelse'),
    ], attributes={
        'binding_name': 'value', 'initial_value_id': 99,
    })

    assert _repair_missing_phi_initial_identities(graph) == ()
    assert graph.nodes[40]['attributes']['initial_value_id'] == 99
    assert graph.graph['unresolved_phi_initial_identities'] == [{
        'phi_value_id': 40,
        'binding_name': 'value',
        'missing_initial_value_id': 99,
        'candidate_value_ids': (10, 11),
        'reason': 'equal_priority_candidates',
        'tie_policy': 'incumbent',
    }]
    assert _repair_missing_phi_initial_identities(graph) == ()
    assert len(graph.graph['unresolved_phi_initial_identities']) == 1


def test_phi_initial_repair_updates_already_planned_control_identity():
    from src.compiler.control_source import (
        ConditionalBlock, ControlProgram, StatementBlock,
    )

    conditional = ConditionalBlock(
        1,
        StatementBlock(('body',)),
        StatementBlock(('orelse',)),
        carried_aliases=((261, 298, 260, 262),),
    )
    control = ControlProgram(conditional)
    repaired = _apply_phi_initial_identity_repairs_to_control(control, ({
        'phi_value_id': 262,
        'missing_initial_value_id': 260,
        'resident_initial_value_id': 256,
    },))

    assert control.root.carried_aliases == ((261, 298, 260, 262),)
    assert repaired.root.carried_aliases == ((261, 298, 256, 262),)


def test_locationless_ast_helpers_do_not_claim_unrelated_branch_work():
    conditional = ast.parse("if ready:\n    value = source\n").body[0]
    guarded_name = conditional.body[0].value
    outside_name = ast.parse("other = source\n").body[0].value

    graph = nx.DiGraph()
    graph.add_node(10, expr_obj=conditional)
    graph.add_node(11, expr_obj=guarded_name)
    graph.add_node(12, expr_obj=outside_name)
    # Context/operator helpers have no source position.  In CPython these are
    # commonly shared singleton-shaped nodes; matching one to a branch used
    # to pollute arbitrary scheduled regions with that branch membership.
    graph.add_node(13, expr_obj=ast.Load())

    memberships = _branch_compartments(SimpleNamespace(G=graph))

    assert memberships[11] == frozenset({(10, "body")})
    assert (10, "body") not in memberships.get(12, frozenset())
    assert (10, "body") not in memberships.get(13, frozenset())


def test_copied_source_and_folded_provenance_require_structural_identity():
    conditional = ast.parse("if ready:\n    tick()\n").body[0]
    call = conditional.body[0].value
    copied = copy.deepcopy(call)
    different = copy.deepcopy(call)
    different.func.id = 'another'
    graph = nx.DiGraph()
    graph.add_node(10, expr_obj=conditional)
    graph.add_node(11, expr_obj=copied)
    graph.add_node(12, expr_obj=different)
    graph.add_node(13, expr_obj=None, authored_expr_obj=copy.deepcopy(call))
    memberships = _branch_compartments(SimpleNamespace(G=graph))
    assert memberships[11] == memberships[13] == frozenset({(10, 'body')})
    assert 12 not in memberships
    copied.func.id = 'changed_between_passes'
    assert 11 not in _branch_compartments(SimpleNamespace(G=graph))


def test_sequence_effect_index_preserves_nested_arm_ownership():
    outer = ast.parse('if ready:\n    if enabled:\n        log.append(1)\n    else:\n        log.append(2)\n').body[0]
    inner = outer.body[0]
    graph = nx.DiGraph()
    graph.graph['source_control_records'] = {
        10: {'expression': outer}, 20: {'expression': inner},
    }
    records = {
        30: {'expression': copy.deepcopy(inner.body[0].value)},
        40: {'expression': copy.deepcopy(inner.orelse[0].value)},
    }
    graph.graph['source_sequence_mutation_records'] = records
    _retain_source_sequence_mutation_records(graph)
    assert set(records[30]['branch_memberships']) == {(10, 'body'), (20, 'body')}
    assert set(records[40]['branch_memberships']) == {(10, 'body'), (20, 'orelse')}


def test_record_field_state_key_correlates_phi_and_setattr_history():
    graph = nx.DiGraph()
    graph.add_node(7, value_id=7, type="Input")
    graph.add_node(
        40,
        value_id=40,
        type="SetAttr",
        parents=[(7, "object"), (12, "value")],
        attributes={"attribute": "flag"},
    )
    graph.add_node(
        41,
        value_id=41,
        type="Phi",
        attributes={
            "binding_name": "field:7.flag",
            "record_field_state": (7, "flag"),
        },
    )
    node_by_value = {7: 7, 40: 40, 41: 41}

    assert _record_field_state_keys(graph, node_by_value, (40,)) == {
        (7, "flag")
    }
    assert _record_field_state_keys(graph, node_by_value, (41,)) == {
        (7, "flag")
    }
