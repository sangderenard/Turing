import pytest

from src.compiler.control_source import (
    ControlProgram, SequenceBlock, StatementBlock, WhileBlock,
    order_control_region_dependencies,
)


def region(index):
    return StatementBlock((f'__scheduled_region_{index}__',))


def loop(*indices):
    return WhileBlock(99, SequenceBlock(()), SequenceBlock(tuple(map(region, indices))))


def test_initial_cap_precedes_atomic_loop_without_hoisting_post_loop_effect():
    stepping = loop(3, 4, 7)
    report = StatementBlock(('report_after_loop();',))
    original = ControlProgram(SequenceBlock((region(0), stepping, report, region(6))), (0, 3, 4, 6, 7))
    result = order_control_region_dependencies(original, ((0, 6), (6, 7)))
    assert result.root.blocks == (region(0), region(6), stepping, report)
    assert original.root.blocks[1] == stepping


def test_nested_loop_prerequisite_stays_inside_outer_loop():
    inner = loop(3)
    outer = WhileBlock(99, SequenceBlock(()), SequenceBlock((inner, region(2))))
    result = order_control_region_dependencies(ControlProgram(outer, (2, 3)), ((2, 3),))
    assert result.root.body.blocks == (region(2), inner)


def test_region_bracketed_by_one_loop_is_sunk_into_that_loop():
    program = ControlProgram(SequenceBlock((loop(1, 3), region(2))), (1, 2, 3))
    result = order_control_region_dependencies(program, ((1, 2), (2, 3)))
    assert result.root.blocks == (loop(1, 2, 3),)


def test_cycle_between_two_atomic_loops_is_still_a_refusal():
    program = ControlProgram(
        SequenceBlock((loop(1, 3), loop(2, 4))), (1, 2, 3, 4))
    with pytest.raises(ValueError, match='atomic control boundaries cyclically'):
        order_control_region_dependencies(
            program, ((1, 2), (2, 3), (3, 4), (4, 1)))
