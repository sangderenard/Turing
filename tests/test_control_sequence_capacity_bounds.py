"""Finite control proofs size arenas; unknown backedges remain unknown."""
from src.compiler.control_source import (ControlProgram, ControlSequenceMutation,
    LoopBlock, LoopControlBlock, SequenceBlock, SequenceMutationBlock, WhileBlock)
from src.compiler.resident_sequence_materialization import _control_sequence_capacity_bounds


def append(sequence):
    return SequenceMutationBlock(ControlSequenceMutation(sequence_value_id=sequence,
        operator="append", argument_value_ids=(100,), effect_node_id=101))


def test_nested_constant_loops_multiply_growth_without_unrolling():
    inner = LoopBlock("j", "0", "3", "1", SequenceBlock((append(1),)))
    outer = LoopBlock("i", "0", "1000000", "1", SequenceBlock((inner,)))
    assert _control_sequence_capacity_bounds(ControlProgram(
        SequenceBlock((append(1), outer))), (1,)) == {1: 3000001}


def test_single_entry_while_is_bounded_by_its_unconditional_exit():
    loop = WhileBlock(10, SequenceBlock(()), SequenceBlock((append(1), LoopControlBlock("break"))))
    assert _control_sequence_capacity_bounds(ControlProgram(loop), (1,)) == {1: 1}


def test_unknown_while_does_not_fabricate_a_capacity():
    loop = WhileBlock(10, SequenceBlock(()), SequenceBlock((append(1),)))
    assert _control_sequence_capacity_bounds(ControlProgram(loop), (1,)) == {}


def test_mutating_the_iterated_sequence_does_not_freeze_its_bound():
    loop = LoopBlock("i", "0", "__iterable_extent_1__", "1", SequenceBlock((append(1),)))
    assert _control_sequence_capacity_bounds(ControlProgram(SequenceBlock((append(1), loop))), (1,)) == {}
