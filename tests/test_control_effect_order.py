from src.compiler.control_source import (
    ConditionalBlock, ControlProgram, ControlSequenceMutation, LoopControlBlock,
    SequenceBlock, SequenceMutationBlock, SequenceQueryBlock, StatementBlock,
)
from src.compiler.hierarchical_plan import PlanCall, PlanClosure
from src.compiler.precompile_to_ssa import _schedule_loop_callsites
import pytest


def test_query_waits_for_earlier_conditional_effect_and_its_predicate():
    append = SequenceMutationBlock(ControlSequenceMutation(30, 'append', (3,), 40))
    conditional = ConditionalBlock(10, SequenceBlock((append,)), source_node_id=11)
    query = SequenceQueryBlock(50, 30, 'truth', source_call_node_id=51)
    producer = StatementBlock(('__scheduled_region_0__',))
    scheduled, _ = _schedule_loop_callsites(
        ControlProgram(SequenceBlock((conditional, query, producer)), region_indices=(0,)),
        PlanClosure('root', (), ()), {0: ((1,), (10,))},
    )
    assert scheduled.root.blocks == (producer, conditional, query)


def test_initial_sequence_read_does_not_depend_on_later_clear():
    query = SequenceQueryBlock(50, 30, 'truth', source_call_node_id=51)
    clear = SequenceMutationBlock(ControlSequenceMutation(30, 'clear', (), 40))
    scheduled, _ = _schedule_loop_callsites(
        ControlProgram(SequenceBlock((query, clear))), PlanClosure('root', (), ()), {},
    )
    assert scheduled.root.blocks == (query, clear)


def test_snapshot_reads_source_before_later_mutation():
    snapshot = SequenceMutationBlock(ControlSequenceMutation(31, 'replace', (30,), 31))
    query = SequenceQueryBlock(50, 31, 'length', source_call_node_id=51)
    append = SequenceMutationBlock(ControlSequenceMutation(30, 'append', (3,), 40))
    scheduled, _ = _schedule_loop_callsites(
        ControlProgram(SequenceBlock((snapshot, query, append))),
        PlanClosure('root', (), ()), {},
    )
    assert scheduled.root.blocks == (snapshot, query, append)


def test_accept_call_does_not_pass_continue_waiting_for_predicate():
    terminal = LoopControlBlock('continue', predicate_value_id=10, site_node_id=12)
    call = StatementBlock(('__plan_callsite_20__',))
    producer = StatementBlock(('__scheduled_region_0__',))
    hierarchy = PlanClosure('root', (), (
        PlanCall(20, PlanClosure('mutate', (), ()), argument_value_ids=(), result_value_ids=()),
    ))
    scheduled, _ = _schedule_loop_callsites(
        ControlProgram(SequenceBlock((terminal, call, producer)), region_indices=(0,)),
        hierarchy, {0: ((1,), (10,))},
    )
    assert scheduled.root.blocks == (producer, terminal, call)


def test_pure_hierarchy_rank_cannot_force_predicate_after_accept_call():
    append = SequenceMutationBlock(ControlSequenceMutation(30, 'append', (3,), 40))
    conditional = ConditionalBlock(10, SequenceBlock((append,)), source_node_id=11)
    query = SequenceQueryBlock(50, 30, 'truth', source_call_node_id=51)
    terminal = LoopControlBlock('continue', predicate_value_id=50, site_node_id=12)
    call = StatementBlock(('__plan_callsite_20__',))
    accept = StatementBlock(('__scheduled_region_1__',))
    predicate = StatementBlock(('__scheduled_region_0__',))
    hierarchy = PlanClosure('root', (), (
        PlanCall(20, PlanClosure('mutate', (), ()), result_value_ids=(21,)),
        PlanClosure('region_1', (), ()), PlanClosure('region_0', (), ()),
    ))
    scheduled, _ = _schedule_loop_callsites(
        ControlProgram(SequenceBlock((conditional, query, terminal, call, accept, predicate)),
                       region_indices=(0, 1)),
        hierarchy, {0: ((1,), (10,)), 1: ((21,), (22,))},
        pure_region_indices=frozenset({0, 1}),
    )
    assert scheduled.root.blocks == (predicate, conditional, query, terminal, call, accept)


def test_conflicting_effect_and_value_orders_are_a_diagnostic():
    append = SequenceMutationBlock(ControlSequenceMutation(30, 'append', (3,), 40))
    conditional = ConditionalBlock(50, SequenceBlock((append,)), source_node_id=11)
    query = SequenceQueryBlock(50, 30, 'truth', source_call_node_id=51)
    with pytest.raises(ValueError, match='control effect order conflicts'):
        _schedule_loop_callsites(
            ControlProgram(SequenceBlock((conditional, query))), PlanClosure('root', (), ()), {},
        )
