"""A Phi selects a representation; it does not reinterpret its operands."""
from src.compiler.ir_indexing import _propagate_scalar_dtypes
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue
import pytest


def test_boolean_merge_chain_uses_exact_incoming_storage_types():
    left = SSAValue(1, 'bool')
    right = SSAValue(2, 'bool')
    inner = SSAValue(3, 'float64')
    outer = SSAValue(4, 'float64')
    function = Function('boolean_chain', [left, right], {'entry': BasicBlock('entry', [
        Instr('Phi', [inner, right], outer),
        Instr('Phi', [left, right], inner),
        Instr('Ret', [outer], None),
    ])})
    _propagate_scalar_dtypes({function.name: function})
    assert inner.dtype == outer.dtype == 'bool'


def test_integer_merge_tracks_settled_backedge_and_attribute_occurrences():
    seed = SSAValue(1, 'int64')
    backedge = SSAValue(2, 'int64')
    merged = SSAValue(3, 'int')
    attribute_seed = SSAValue(1, 'int64')
    attribute_backedge = SSAValue(2, 'int64')
    function = Function('integer_merge', [seed, backedge], {'entry': BasicBlock('entry', [
        Instr('Phi', [], merged, attributes={'incoming': (('start', attribute_seed), ('latch', attribute_backedge))}),
        Instr('Ret', [merged], None),
    ])})
    _propagate_scalar_dtypes({function.name: function})
    assert merged.dtype == 'int64'


def test_mixed_representation_phi_is_not_implicitly_reinterpreted():
    seed = SSAValue(1, 'float64')
    backedge = SSAValue(2, 'int64')
    merged = SSAValue(3, 'float64')
    function = Function('bad_merge', [seed, backedge], {'entry': BasicBlock('entry', [
        Instr('Phi', [seed, backedge], merged), Instr('Ret', [merged], None),
    ])})
    _propagate_scalar_dtypes({function.name: function})
    assert merged.dtype == 'float64'
    assert backedge.dtype == 'int64'


@pytest.mark.parametrize('literal,dtype', [(0, 'int64'), (False, 'bool'), (0.0, 'float64')])
def test_folded_while_seed_retains_authored_literal_representation(literal, dtype):
    from src.compiler.control_source import ControlProgram, ControlExpression, SequenceBlock, WhileBlock
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa
    control = ControlProgram(WhileBlock(
        predicate_value_id=10, condition=SequenceBlock(()), body=SequenceBlock(()),
        predicate_expression=ControlExpression('const', literal=False),
        carried_aliases=((21, 20),), carried_seeds=((20, literal),), source_loop_node_id=30))
    function, shortfalls = lower_control_program_to_ssa(control)
    assert not shortfalls
    carried = next(i for b in function.blocks.values() for i in b.instrs
                   if i.op == 'Phi' and i.attributes.get('binding') == 'loop_carried')
    seed = next(i for b in function.blocks.values() for i in b.instrs
                if i.op == 'Const' and i.res.id == carried.args[0].id)
    assert seed.res.dtype == dtype
    assert type(seed.attributes['value']) is type(literal)
    assert seed.attributes['value'] == literal
