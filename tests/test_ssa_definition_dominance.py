from src.compiler.ssa_self_check import check_definition_dominance
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
