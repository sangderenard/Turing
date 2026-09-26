import pytest

from src.compiler.control_source import ControlProgram, ConditionalBlock, LoopControlBlock, SequenceBlock, StatementBlock
from src.compiler.hierarchical_plan import PlanCall, PlanClosure
from src.compiler.precompile_to_ssa import _schedule_loop_callsites


@pytest.mark.parametrize('boundary_kind', ['conditional', 'continue', 'break'])
def test_call_dependency_is_not_hidden_by_conditional_region_anchor(boundary_kind):
    # The overlay moved the producer's numerical anchor after a conditional,
    # while the authored hierarchy still orders advance before normalization.
    region = lambda i: StatementBlock((f'__scheduled_region_{i}__',))
    plan = PlanClosure('root', (), (
        PlanCall(326, PlanClosure('advance', (), ()), argument_bindings=((0, 0),), result_bindings=((1, 459),)),
        PlanClosure('region_1', (), ()),
        PlanCall(460, PlanClosure('normalize', (), ()), argument_bindings=((459, 0),), result_bindings=((0, 460),)),
        PlanClosure('region_2', (), ()),
        PlanClosure('region_3', (), ()),
    ))
    conditional = ConditionalBlock(461, SequenceBlock((region(3),)))
    if boundary_kind != 'conditional':
        conditional = LoopControlBlock(boundary_kind, predicate_value_id=461)
    result, _ = _schedule_loop_callsites(
        ControlProgram(SequenceBlock((region(2), conditional, region(1)))),
        plan, {1: ((459,), (462,)), 2: ((460,), (461,)), 3: ((460,), (463,))},
    )
    blocks = result.root.blocks
    producer = StatementBlock(('__plan_callsite_326__',))
    consumer = StatementBlock(('__plan_callsite_460__',))
    assert blocks.index(producer) < blocks.index(consumer) < blocks.index(conditional)
