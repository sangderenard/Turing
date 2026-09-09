import pickle
import subprocess
import sys
import pytest

import networkx as nx

from src.compiler.ssa_record_return_state import (
    freshen_redefined_ssa_objects,
    publish_inout_scalar_return_snapshots,
    reconcile_conditional_phi_continuations,
    reconcile_nondominating_identity_cast_results,
    repair_non_dominating_return_phi_inputs,
    scalar_return_field_versions,
)
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_self_check import check_definition_dominance
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


def test_conditional_phi_replaces_only_dominated_continuation_uses():
    initial = SSAValue(0, "float64")
    changed = SSAValue(1, "float64")
    predicate = SSAValue(2, "bool")
    merged = SSAValue(3, "float64")
    before = SSAValue(4, "float64")
    after = SSAValue(5, "float64")
    function = Function("root", [initial, changed, predicate], {
        "entry": BasicBlock("entry", [
            Instr("Copy", [changed], before),
            Instr("CondBr", [predicate], None, attributes={
                "true_target": "changed", "false_target": "unchanged",
            }),
        ], ["changed", "unchanged"]),
        "changed": BasicBlock("changed", [
            Instr("Br", [], None, attributes={"target": "merge"}),
        ], ["merge"]),
        "unchanged": BasicBlock("unchanged", [
            Instr("Br", [], None, attributes={"target": "merge"}),
        ], ["merge"]),
        "merge": BasicBlock("merge", [
            Instr("Phi", [changed, initial], merged, attributes={
                "binding": "conditional_carried",
                "incoming_blocks": ("changed", "unchanged"),
            }),
            Instr("Copy", [changed], after),
            Instr("Ret", [after], None),
        ]),
    })
    module = IRModule({"root": function})

    assert reconcile_conditional_phi_continuations(module) == 1
    assert reconcile_conditional_phi_continuations(module) == 0
    assert function.blocks["entry"].instrs[0].args[0] is changed
    assert function.blocks["merge"].instrs[1].args[0] is merged
    [receipt] = module.metadata["conditional_phi_continuation_receipts"]
    assert receipt["priority"] == "unique_dominating_conditional_phi"
    assert receipt["tie_policy"] == "incumbent"


def test_redefined_value_object_is_freshened_per_dominating_edge():
    initial = SSAValue(0, "float64")
    predicate = SSAValue(1, "bool")
    reused = SSAValue(7, "float64")
    source = SSAValue(8, "float64")
    final = SSAValue(9, "float64")
    first_phi = Instr("Phi", [initial, initial], reused, attributes={
        "binding": "conditional_carried",
        "incoming_blocks": ("first_true", "first_false"),
    })
    second_definition = Instr("Copy", [source], reused)
    final_phi = Instr("Phi", [reused, reused], final, attributes={
        "binding": "conditional_carried",
        "incoming_blocks": ("second_true", "second_false"),
    })
    function = Function("root", [initial, predicate, source], {
        "entry": BasicBlock("entry", [
            Instr("CondBr", [predicate], None),
        ], ["first_true", "first_false"]),
        "first_true": BasicBlock("first_true", [Instr("Br", [], None)], ["merge"]),
        "first_false": BasicBlock("first_false", [Instr("Br", [], None)], ["merge"]),
        "merge": BasicBlock("merge", [
            first_phi, Instr("CondBr", [predicate], None),
        ], ["second_true", "second_false"]),
        "second_true": BasicBlock("second_true", [
            second_definition, Instr("Br", [], None),
        ], ["final_merge"]),
        "second_false": BasicBlock("second_false", [
            Instr("Br", [], None),
        ], ["final_merge"]),
        "final_merge": BasicBlock("final_merge", [final_phi, Instr("Ret", [final], None)]),
    })
    module = IRModule({"root": function})

    assert freshen_redefined_ssa_objects(module) == 1
    assert freshen_redefined_ssa_objects(module) == 0
    assert first_phi.res is reused
    assert second_definition.res is not reused
    assert second_definition.res.id != reused.id
    assert final_phi.args[0] is second_definition.res
    assert final_phi.args[1] is reused
    [receipt] = module.metadata["redefined_ssa_object_receipts"]
    assert receipt["priority"] == "later_definition_requires_unique_identity"


def test_nondominating_identity_cast_result_uses_dominating_actual():
    callee_input = SSAValue(10, "float64")
    normalized = SSAValue(11, "float64")
    callee = Function("normalize", [callee_input], {
        "entry": BasicBlock("entry", [
            Instr("Cast", [callee_input], normalized, attributes={
                "ensures_schema_type": "AbstractTensor",
            }),
            Instr("Ret", [normalized], None),
        ]),
    })
    actual = SSAValue(0, "float64")
    predicate = SSAValue(1, "bool")
    aggregate = SSAValue(2, "ssa.aggregate")
    index = SSAValue(3, "int64")
    pointer = SSAValue(4, "ptr")
    projected = SSAValue(11, "float64")
    consumed = SSAValue(5, "float64")
    caller = Function("root", [actual, predicate], {
        "entry": BasicBlock("entry", [
            Instr("CondBr", [predicate], None, attributes={
                "true_target": "first_true", "false_target": "first_false",
            }),
        ], ["first_true", "first_false"]),
        "first_true": BasicBlock("first_true", [
            Instr("Call", [actual], aggregate, attributes={
                "callee": "normalize", "output_ids": (11,),
            }),
            Instr("GetElementPtr", [aggregate, index], pointer,
                  attributes={"source_output_id": 11}),
            Instr("Load", [pointer], projected,
                  attributes={"source_output_id": 11}),
            Instr("Br", [], None, attributes={"target": "merge"}),
        ], ["merge"]),
        "first_false": BasicBlock("first_false", [
            Instr("Br", [], None, attributes={"target": "merge"}),
        ], ["merge"]),
        "merge": BasicBlock("merge", [
            Instr("CondBr", [predicate], None, attributes={
                "true_target": "second_true", "false_target": "second_false",
            }),
        ], ["second_true", "second_false"]),
        "second_true": BasicBlock("second_true", [
            Instr("Copy", [projected], consumed),
            Instr("Ret", [consumed], None),
        ]),
        "second_false": BasicBlock("second_false", [
            Instr("Ret", [actual], None),
        ]),
    })
    module = IRModule({"root": caller, "normalize": callee})

    assert reconcile_nondominating_identity_cast_results(module) == 1
    assert reconcile_nondominating_identity_cast_results(module) == 0
    assert caller.blocks["second_true"].instrs[0].args[0] is actual
    [receipt] = module.metadata["identity_cast_result_receipts"]
    assert receipt["priority"] == "exact_physical_identity_cast_result"
    assert receipt["tie_policy"] == "incumbent"


def test_inout_scalar_return_uses_a_distinct_value_snapshot():
    from src.transmogrifier.ssa import (
        SSARecordDescriptor,
        SSARecordFieldDescriptor,
        SSARecordTable,
    )

    resident = SSAValue(18, "float64")
    written = SSAValue(28, "float64", accounting={
        "source_value_id": 18,
        "ssa_inout_write_version": True,
    })
    callee = Function(
        "update",
        [resident],
        {"entry": BasicBlock("entry", [
            Instr("Identity", [resident], written),
            Instr("Ret", [resident, resident], None),
        ])},
        metadata={"record_return_layouts": ((100, (18, 18)),)},
    )
    aggregate = SSAValue(40, "ssa.aggregate")
    caller = Function("root", [SSAValue(7, "float64")], {
        "entry": BasicBlock("entry", [
            Instr("Call", [SSAValue(7, "float64")], aggregate, attributes={
                "callee": "update",
                "result_convention": "ssa.aggregate",
                "output_ids": (48, 48),
                "callee_output_ids": (18, 18),
                "native_result_contract": (
                    (18, "float64", ()), (18, "float64", ()),
                ),
            }),
            Instr("Ret", [], None),
        ]),
    })
    module = IRModule({"root": caller, "update": callee})
    module.record_tables["update"] = SSARecordTable(records={
        100: SSARecordDescriptor(100, "Metrics", (
            SSARecordFieldDescriptor(
                "max_vel", "scalar", value_ids=(18,), dtype="float64",
            ),
            SSARecordFieldDescriptor(
                "max_flux", "scalar", value_ids=(18,), dtype="float64",
            ),
        )),
    })

    assert publish_inout_scalar_return_snapshots(module) == 2
    assert publish_inout_scalar_return_snapshots(module) == 0
    assert [value.id for value in callee.blocks["entry"].instrs[-1].args] == [28, 28]
    assert callee.metadata["record_return_layouts"] == ((100, (28, 28)),)
    assert [
        field.value_ids
        for field in module.record_tables["update"].records[100].fields
    ] == [(28,), (28,)]
    call = caller.blocks["entry"].instrs[0]
    assert call.attributes["callee_output_ids"] == (28, 28)
    assert tuple(
        item[0] for item in call.attributes["native_result_contract"]
    ) == (28, 28)
    [receipt, duplicate_receipt] = module.metadata[
        "inout_scalar_return_snapshot_receipts"
    ]
    assert receipt["priority"] == "exact_dominating_inout_write"
    assert duplicate_receipt["tie_policy"] == "incumbent"


def test_inout_scalar_return_keeps_incumbent_for_branch_tie():
    resident = SSAValue(18, "float64")
    predicate = SSAValue(19, "bool")
    left = SSAValue(28, "float64", accounting={
        "source_value_id": 18, "ssa_inout_write_version": True,
    })
    right = SSAValue(29, "float64", accounting={
        "source_value_id": 18, "ssa_inout_write_version": True,
    })
    function = Function("branch_update", [resident, predicate], {
        "entry": BasicBlock("entry", [
            Instr("CondBr", [predicate], None, attributes={
                "true_target": "left", "false_target": "right",
            }),
        ], ["left", "right"]),
        "left": BasicBlock("left", [
            Instr("Identity", [resident], left),
            Instr("Br", [], None, attributes={"target": "exit"}),
        ], ["exit"]),
        "right": BasicBlock("right", [
            Instr("Identity", [resident], right),
            Instr("Br", [], None, attributes={"target": "exit"}),
        ], ["exit"]),
        "exit": BasicBlock("exit", [Instr("Ret", [resident], None)]),
    })
    module = IRModule({function.name: function})

    assert publish_inout_scalar_return_snapshots(module) == 0
    assert function.blocks["exit"].instrs[-1].args == [resident]


def test_return_edge_keeps_source_slots_after_physical_alias_resolution():
    from src.compiler.control_source import ControlProgram, LoopControlBlock
    from src.compiler.precompile_to_ssa import _ControlSSABuilder

    for predicate in (None, 20):
        block = LoopControlBlock('return', predicate_value_id=predicate,
                                 return_value_ids=(10,))
        builder = _ControlSSABuilder(
            ControlProgram(block), function_name='root', first_value_id=100,
            region_callees={}, region_signatures={}, value_aliases={10: 11},
        )
        builder.lower(block, path='root')
        [(edge, values)] = builder.function_return_edges
        assert values[0].id == 11
        assert edge.instrs[-1].attributes['return_source_value_ids'] == (10,)


def test_pure_return_projection_is_recomputed_on_its_physical_edge():
    source = SSAValue(0, 'float64')
    predicate = SSAValue(1, 'bool')
    aggregate = SSAValue(10, 'ssa.aggregate')
    index = SSAValue(11, 'int64')
    pointer = SSAValue(12, 'ptr')
    projected = SSAValue(13, 'float64')
    fallback = SSAValue(14, 'float64')
    returned = SSAValue(15, 'float64')
    function = Function('root', [source, predicate], {
        'entry': BasicBlock('entry', [
            Instr('Const', [], index, attributes={'value': 0}),
            Instr('Const', [], fallback, attributes={'value': -1.0}),
            Instr('CondBr', [predicate], None, attributes={
                'true_target': 'calculated', 'false_target': 'skipped',
            }),
        ], ['calculated', 'skipped']),
        'calculated': BasicBlock('calculated', [
            Instr('Call', [source], aggregate, attributes={
                'callee': 'root__planned_region_0',
                'region_index': 0,
                'output_ids': (13,),
                'result_convention': 'ssa.aggregate',
            }),
            Instr('GetElementPtr', [aggregate, index], pointer, attributes={
                'region_index': 0,
                'source_output_id': 13,
            }),
            Instr('Load', [pointer], projected, attributes={
                'region_index': 0,
                'source_output_id': 13,
            }),
            Instr('Br', [], None, attributes={'target': 'guard'}),
        ], ['guard']),
        'skipped': BasicBlock('skipped', [
            Instr('Br', [], None, attributes={'target': 'guard'}),
        ], ['guard']),
        'guard': BasicBlock('guard', [
            Instr('CondBr', [predicate], None, attributes={
                'true_target': 'return_edge', 'false_target': 'fallthrough',
            }),
        ], ['return_edge', 'fallthrough']),
        'return_edge': BasicBlock('return_edge', [
            Instr('Br', [], None, attributes={
                'target': 'exit', 'return_source_value_ids': (13,),
            }),
        ], ['exit']),
        'fallthrough': BasicBlock('fallthrough', [
            Instr('Br', [], None, attributes={
                'target': 'exit', 'return_source_value_ids': (14,),
            }),
        ], ['exit']),
        'exit': BasicBlock('exit', [
            Instr('Phi', [projected, fallback], returned, attributes={
                'incoming_blocks': ('return_edge', 'fallthrough'),
                'binding': 'return_merge',
                'return_slot_index': 0,
            }),
            Instr('Ret', [returned], None),
        ]),
    })

    before = check_definition_dominance(IRModule({'root': function}))
    assert len(before) == 1
    receipts = repair_non_dominating_return_phi_inputs(function)
    assert len(receipts) == 1
    assert receipts[0]['source_value_id'] == 13
    assert receipts[0]['operation_count'] == 3
    assert receipts[0]['tie_policy'] == 'incumbent'
    assert check_definition_dominance(IRModule({'root': function})) == []
    assert repair_non_dominating_return_phi_inputs(function) == ()
    edge_operations = function.blocks['return_edge'].instrs
    assert [item.op for item in edge_operations] == [
        'Call', 'GetElementPtr', 'Load', 'Br',
    ]
    assert function.blocks['exit'].instrs[0].args[0] is edge_operations[2].res


def test_record_projection_alias_moves_return_receipt_with_its_slot():
    from types import SimpleNamespace
    from src.compiler.glsl_deployment_strategy import _alias_projection_to_member

    graph = nx.DiGraph()
    graph.add_node(10, parents=(), children=())
    graph.add_node(11, parents=(), children=())
    graph.add_node(3, parents=(), children=())
    span = (10, 0, 10, 10)
    graph.graph['return_slot_values'] = {span: (10,)}
    graph.graph['return_record_field_states'] = {span: ((10, 'flag', 3),)}
    process = SimpleNamespace(G=graph, roots=[10])
    _alias_projection_to_member(process, 10, 11)
    assert 10 not in graph
    assert graph.graph['return_slot_values'][span] == (11,)
    assert graph.graph['return_record_field_states'][span] == ((11, 'flag', 3),)


def fixture():
    initial, change, false, merged, returned = [SSAValue(i, 'bool') for i in range(5)]
    function = Function('root', [initial, change], {
        'entry': BasicBlock('entry', [
            Instr('Const', [], false, attributes={'value': False}),
            Instr('CondBr', [change], None, attributes={
                'true_target': 'changed', 'false_target': 'unchanged'})]),
        'changed': BasicBlock('changed', [Instr('Br', [], None, attributes={'target': 'merge'})]),
        'unchanged': BasicBlock('unchanged', [Instr('Br', [], None, attributes={'target': 'merge'})]),
        'merge': BasicBlock('merge', [
            Instr('Phi', [false, initial], merged, attributes={
                'incoming_blocks': ('changed', 'unchanged'), 'binding': 'conditional_carried'}),
            Instr('Br', [], None, attributes={'target': 'exit', 'return_source_value_ids': (100,)})]),
        'exit': BasicBlock('exit', [
            Instr('Phi', [initial], returned, attributes={'incoming_blocks': ('merge',)}),
            Instr('Ret', [initial, returned], None)]),
    })
    graph = nx.DiGraph()
    graph.graph['return_record_field_states'] = {(10, 0, 10, 10): ((100, 'flag', merged.id),)}
    graph.graph['return_slot_values'] = {(10, 0, 10, 10): (100,)}
    return function, graph, initial, merged, returned


def test_scalar_receipt_requires_exact_receiver_and_dominance():
    function, graph, initial, merged, _ = fixture()
    function.blocks['changed'].instrs[-1].attributes['return_source_value_ids'] = (100,)
    resolve = scalar_return_field_versions(function, graph)
    assert resolve(100, 'flag', 'merge', initial) is merged
    assert resolve(100, 'flag', 'changed', initial) is initial
    assert resolve(101, 'flag', 'merge', initial) is initial
    assert resolve(100, 'other', 'merge', initial) is initial
    graph.graph['return_record_field_states'][(20, 0, 20, 10)] = ((100, 'flag', 2),)
    graph.graph['return_slot_values'][(20, 0, 20, 10)] = (100,)
    assert scalar_return_field_versions(function, graph)(100, 'flag', 'merge', initial) is initial


def test_return_edge_selects_its_own_site_among_distinct_return_slots():
    function, graph, initial, merged, _ = fixture()
    graph.graph['return_slot_values'][(20, 0, 20, 10)] = (100, 200)
    graph.graph['return_record_field_states'][(20, 0, 20, 10)] = ((100, 'flag', 2),)
    assert scalar_return_field_versions(function, graph)(100, 'flag', 'merge', initial) is merged
    function.blocks['merge'].instrs[-1].attributes['return_source_value_ids'] = (100, 200)
    # Terminal constants are exact edge state, just like conditional versions.
    assert scalar_return_field_versions(function, graph)(100, 'flag', 'merge', initial).id == 2


@pytest.mark.parametrize('provisional_dtype', ['bool', 'float64'])
def test_scalar_return_receipt_matches_both_native_outcomes(tmp_path, provisional_dtype):
    function, graph, initial, merged, returned = fixture()
    merged.dtype = provisional_dtype
    function.blocks['merge'].instrs[0].attributes['binding'] = 'conditional_carried'
    phi = function.blocks['exit'].instrs[0]
    from src.compiler.ssa_record_return_state import publish_scalar_record_return_fields
    from src.transmogrifier.ssa import SSARecordTable, SSARecordDescriptor, SSARecordFieldDescriptor
    scalar_return_field_versions(function, graph)  # Retain the source receipts.
    phi.attributes.update(record_return_scalar=True, record_return_receivers=(100,),
                          return_slot_index=0, record_field='flag')
    module = IRModule({'root': function})
    module.record_tables['root'] = SSARecordTable(records={100: SSARecordDescriptor(
        100, 'State', (SSARecordFieldDescriptor('flag', 'scalar', value_ids=(0,), dtype='bool'),))})
    assert publish_scalar_record_return_fields(module) == 1
    assert publish_scalar_record_return_fields(module) == 0
    assert phi.res is returned
    assert phi.args[0].dtype == 'bool'
    artifact = emit_ssa_module_to_c(module, 'root')
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps(artifact))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact = pickle.load(open(sys.argv[1], 'rb'))
execution = artifact.prepare_execution({0: np.array([False], dtype=np.bool_),
                                       1: np.array([False], dtype=np.bool_)})
for initial in (False, True, False, True):
    for change in (False, True, False):
        execution.buffers[0][0] = initial
        execution.buffers[1][0] = change
        execution.run()
        assert execution.buffers[4].item() == (False if change else initial)
        assert execution.buffers[0].item() == initial
''', str(payload)], capture_output=True, text=True, timeout=20)
    assert probe.returncode == 0, probe.stdout + probe.stderr


def test_return_receipt_does_not_treat_mutable_formal_as_immutable_definition():
    function, graph, initial, merged, _ = fixture()
    function.args.append(merged)
    assert scalar_return_field_versions(function, graph)(100, 'flag', 'merge', initial) is initial


def test_boolean_return_conversion_requires_boolean_leaves():
    function, graph, initial, merged, _ = fixture()
    merged.dtype = 'float64'
    function.blocks['merge'].instrs[0].attributes['binding'] = 'conditional_carried'
    # A numeric source assignment is not evidence of a Boolean field value.
    function.blocks['entry'].instrs[0].res.dtype = 'float64'
    assert scalar_return_field_versions(function, graph)(100, 'flag', 'merge', initial) is initial


def test_later_call_through_field_storage_prevents_stale_return_selection():
    function, graph, initial, merged, _ = fixture()
    unrelated = SSAValue(9, 'float64')
    function.blocks['merge'].instrs.insert(-1, Instr('Call', [unrelated], None,
                                                   attributes={'callee': 'other'}))
    assert scalar_return_field_versions(function, graph)(100, 'flag', 'merge', initial) is merged
    function.blocks['merge'].instrs.insert(-1, Instr('Call', [initial], None,
                                                   attributes={'callee': 'mutate'}))
    assert scalar_return_field_versions(function, graph)(100, 'flag', 'merge', initial) is initial


def test_transitive_readonly_call_may_copy_field_value_but_not_write_field_storage():
    function, graph, initial, merged, _ = fixture()
    flag, output = SSAValue(10, 'bool'), SSAValue(11, 'ptr')
    leaf = Function('leaf', [flag, output], {'entry': BasicBlock('entry', [
        Instr('Store', [flag, output], None), Instr('Ret', [], None)])})
    wrapper = Function('wrapper', [flag, output], {'entry': BasicBlock('entry', [
        Instr('Call', [flag, output], None, attributes={'callee': 'leaf'}),
        Instr('Ret', [], None)])})
    destination = SSAValue(9, 'ptr')
    function.args.append(destination)
    function.blocks['merge'].instrs.insert(-1, Instr('Call', [initial, destination], None,
                                                   attributes={'callee': 'wrapper'}))
    functions = {'root': function, 'wrapper': wrapper, 'leaf': leaf}
    assert scalar_return_field_versions(function, graph, functions)(100, 'flag', 'merge', initial) is merged
    leaf.blocks['entry'].instrs[0] = Instr('Store', [flag, flag], None)
    assert scalar_return_field_versions(function, graph, functions)(100, 'flag', 'merge', initial) is initial


def test_loop_reexecution_kills_the_previous_field_version():
    function, graph, initial, merged, _ = fixture()
    function.blocks['entry'].instrs.insert(0, Instr('Call', [initial], None,
                                                  attributes={'callee': 'mutate_before_definition'}))
    function.blocks['merge'].instrs[-1] = Instr('CondBr', [function.args[1]], None,
        attributes={'true_target': 'entry', 'false_target': 'return_edge'})
    function.blocks['return_edge'] = BasicBlock('return_edge', [Instr('Br', [], None,
        attributes={'target': 'exit', 'return_source_value_ids': (100,)})])
    function.blocks['exit'].instrs[0].attributes['incoming_blocks'] = ('return_edge',)
    assert scalar_return_field_versions(function, graph)(100, 'flag', 'return_edge', initial) is merged


def test_other_record_field_projection_is_not_the_selected_fields_storage():
    function, graph, initial, merged, _ = fixture()
    record, index, other_field = SSAValue(100, 'ssa.aggregate'), SSAValue(8, 'int64'), SSAValue(9, 'ptr')
    function.blocks['merge'].instrs[-1:-1] = [
        Instr('GetElementPtr', [record, index], other_field),
        Instr('Call', [other_field], None, attributes={'callee': 'other_field_writer'}),
    ]
    assert scalar_return_field_versions(function, graph)(100, 'flag', 'merge', initial) is merged
    function.blocks['merge'].instrs.insert(-1, Instr('Call', [record], None,
                                                   attributes={'callee': 'unknown_record_writer'}))
    assert scalar_return_field_versions(function, graph)(100, 'flag', 'merge', initial) is initial
