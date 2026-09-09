from src.compiler.ssa_reachability import prune_constant_control_flow
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue


def test_dead_edge_removed_even_when_both_predecessor_blocks_remain_live():
    flag, literal, left, right, merged = [SSAValue(i, 'bool') for i in range(5)]
    function = Function('root', [flag, left, right], {
        'entry': BasicBlock('entry', [Instr('CondBr', [flag], None, attributes={
            'true_target': 'a', 'false_target': 'b'})]),
        'a': BasicBlock('a', [Instr('Const', [], literal, attributes={'value': False}),
            Instr('CondBr', [literal], None, attributes={
                'true_target': 'merge', 'false_target': 'b'})]),
        'b': BasicBlock('b', [Instr('Br', [], None, attributes={'target': 'merge'})]),
        'merge': BasicBlock('merge', [Instr('Phi', [left, right], merged,
            attributes={'incoming_blocks': ('a', 'b'), 'record_return_scalar': True,
                        'record_return_receivers': (10, 20), 'initial_value_id': left.id}),
            Instr('Ret', [merged], None)]),
        'orphan': BasicBlock('orphan', [Instr('Call', [left], None,
            attributes={'callee': 'effect'}), Instr('Ret', [], None)]),
    })
    assert prune_constant_control_flow(function) == 3
    assert set(function.blocks) == {'entry', 'a', 'b', 'merge'}
    phi = function.blocks['merge'].instrs[0]
    assert phi.args == [right]
    assert phi.attributes['incoming_blocks'] == ('b',)
    assert phi.attributes['record_return_receivers'] == (20,)
    assert phi.attributes['initial_value_id'] == right.id
    assert prune_constant_control_flow(function) == 0


def test_absorbing_boolean_does_not_delete_reachable_effects_or_mutable_reads():
    literal, unknown, predicate = [SSAValue(i, 'bool') for i in range(3)]
    call = Instr('Call', [unknown], None, attributes={'callee': 'effect'})
    function = Function('root', [unknown], {
        'entry': BasicBlock('entry', [call,
            Instr('Const', [], literal, attributes={'value': False}),
            Instr('LAnd', [literal, unknown], predicate),
            Instr('CondBr', [predicate], None, attributes={
                'true_target': 'dead', 'false_target': 'live'})]),
        'dead': BasicBlock('dead', [Instr('Ret', [unknown], None)]),
        'live': BasicBlock('live', [Instr('Ret', [literal], None)]),
    })
    prune_constant_control_flow(function)
    assert 'dead' not in function.blocks
    assert function.blocks['entry'].instrs[0] is call
    assert unknown in call.args


def test_dynamic_loop_phi_is_not_folded_from_its_initial_value():
    initial, carried, updated = [SSAValue(i, 'bool') for i in range(3)]
    function = Function('root', [], {
        'entry': BasicBlock('entry', [Instr('Const', [], initial, attributes={'value': True}),
            Instr('Br', [], None, attributes={'target': 'header'})]),
        'header': BasicBlock('header', [Instr('Phi', [initial, updated], carried,
            attributes={'incoming_blocks': ('entry', 'body')}),
            Instr('CondBr', [carried], None, attributes={
                'true_target': 'body', 'false_target': 'exit'})]),
        'body': BasicBlock('body', [Instr('Call', [], updated, attributes={'callee': 'read'}),
            Instr('Br', [], None, attributes={'target': 'header'})]),
        'exit': BasicBlock('exit', [Instr('Ret', [carried], None)]),
    })
    assert prune_constant_control_flow(function) == 0
    assert function.blocks['header'].instrs[-1].op == 'CondBr'


def test_pruning_updates_caller_and_callee_without_exempting_live_formals():
    from src.compiler.fortran_c_shell import (
        _prune_unused_callee_formals, _undefined_repository_ssa_operands,
    )
    from src.compiler.ssa_self_check import check_formal_parity
    from src.transmogrifier.ssa import IRModule

    parameter, fabricated, literal, output = [SSAValue(i, 'float64') for i in range(4)]
    child = Function('child', [parameter, fabricated], {
        'entry': BasicBlock('entry', [Instr('Const', [], literal, attributes={'value': False}),
            Instr('CondBr', [literal], None, attributes={
                'true_target': 'dead', 'false_target': 'live'})]),
        'dead': BasicBlock('dead', [Instr('Ret', [fabricated], None)]),
        'live': BasicBlock('live', [Instr('Ret', [parameter], None)]),
    }, metadata={'parameter_names': (('x', parameter.id),), 'authored_parameters': ('x',)})
    call = Instr('Call', [parameter, fabricated], output, attributes={
        'callee': 'child', 'callee_input_ids': (parameter.id, fabricated.id)})
    caller = Function('root', [parameter, fabricated], {
        'entry': BasicBlock('entry', [call, Instr('Ret', [output], None)]),
    })
    module = IRModule({'root': caller, 'child': child})
    assert check_formal_parity(module)
    prune_constant_control_flow(child)
    assert _prune_unused_callee_formals(module.functions) == 1
    assert child.args == call.args == [parameter]
    assert call.attributes['callee_input_ids'] == (parameter.id,)
    assert not check_formal_parity(module)
    assert not _undefined_repository_ssa_operands(module)


def test_mutable_abi_slot_is_not_a_constant_even_with_a_literal_write():
    slot = SSAValue(0, 'bool', accounting={'program_abi_mutable': True})
    function = Function('root', [slot], {
        'entry': BasicBlock('entry', [Instr('Const', [], slot, attributes={'value': True}),
            Instr('Call', [slot], None, attributes={'callee': 'mutate'}),
            Instr('CondBr', [slot], None, attributes={
                'true_target': 'a', 'false_target': 'b'})]),
        'a': BasicBlock('a', [Instr('Ret', [slot], None)]),
        'b': BasicBlock('b', [Instr('Ret', [slot], None)]),
    })
    assert prune_constant_control_flow(function) == 0


def test_pruned_cfg_preserves_both_native_return_paths(tmp_path):
    import pickle
    import subprocess
    import sys

    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.transmogrifier.ssa import IRModule

    flag, false = SSAValue(0, 'bool'), SSAValue(1, 'bool')
    left, right, result = [SSAValue(i, 'float64') for i in range(2, 5)]
    function = Function('root', [flag], {
        'entry': BasicBlock('entry', [
            Instr('Const', [], false, attributes={'value': False}),
            Instr('Const', [], left, attributes={'value': 10.0}),
            Instr('Const', [], right, attributes={'value': 20.0}),
            Instr('CondBr', [flag], None, attributes={
                'true_target': 'left', 'false_target': 'right'})]),
        'left': BasicBlock('left', [Instr('CondBr', [false], None, attributes={
            'true_target': 'dead', 'false_target': 'merge'})]),
        'right': BasicBlock('right', [Instr('Br', [], None, attributes={'target': 'merge'})]),
        'dead': BasicBlock('dead', [Instr('Call', [], None,
            attributes={'callee': 'unavailable'}), Instr('Ret', [left], None)]),
        'merge': BasicBlock('merge', [Instr('Phi', [left, right], result,
            attributes={'incoming_blocks': ('left', 'right')}), Instr('Ret', [result], None)]),
    })
    prune_constant_control_flow(function)
    artifact = emit_ssa_module_to_c(IRModule({'root': function}), 'root')
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps(artifact))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact = pickle.load(open(sys.argv[1], 'rb'))
execution = artifact.prepare_execution({0: np.array([False], dtype=np.bool_)})
for enabled, expected in ((False, 20.0), (True, 10.0), (False, 20.0)):
    execution.buffers[0][0] = enabled
    execution.run()
    assert execution.buffers[4].item() == expected
''', str(saved)], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stdout + probe.stderr
