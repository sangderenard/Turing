from __future__ import annotations
from pathlib import Path
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.compiler import fortran_c_shell as shell
from src.compiler.ssa_self_check import run_all
original_unary = shell._recover_late_source_unary_operations
original_pure = shell._recover_late_source_pure_expressions

def report(kind, function, graph):
    if 'step_with_dt_control_used__specialized' not in function.name or '__planned_region_' in function.name:
        return
    print('DIAG', kind, function.name, 'args', [a.id for a in function.args if a.id in {147,149,231}], 'graph', graph is not None, flush=True)
    if graph is not None and kind == 'unary-before':
        produced = {
            int(instruction.res.id): (name, index, instruction.res)
            for name, block in function.blocks.items()
            for index, instruction in enumerate(block.instrs)
            if instruction.res is not None
        }
        for value_id in (147, 149, 231):
            data = graph.nodes.get(value_id, {})
            operation = str(data.get('op') or data.get('type') or '').casefold()
            operands = tuple(int(parent) for parent, role in data.get('parents') or () if str(role) in {'operand','arg:0','value'})
            carried = tuple(
                (name, index, instruction.res.id, instruction.res.shape)
                for name, block in function.blocks.items()
                for index, instruction in enumerate(block.instrs)
                if instruction.op == 'Phi' and instruction.res is not None
                and int((instruction.attributes or {}).get('initial_value_id', -1)) == (operands[0] if operands else -2)
                and (instruction.attributes or {}).get('binding') == 'loop_carried'
            )
            consumers = tuple(name for name, block in function.blocks.items() for instruction in block.instrs if any(int(argument.id)==value_id for argument in instruction.args))
            print('CASE',value_id,'op',operation,'operands',operands,'self-produced',value_id in produced,'source',produced.get(operands[0])[:2] if operands and produced.get(operands[0]) else None,'shape',produced.get(operands[0])[2].shape if operands and produced.get(operands[0]) else None,'carried',carried,'consumers',consumers,flush=True)

def unary(function, graph):
    report('unary-before', function, graph)
    result = original_unary(function, graph)
    report('unary-after', function, graph)
    print('UNARY-RESULT', result, flush=True) if 'step_with_dt_control_used__specialized' in function.name and '__planned_region_' not in function.name else None
    return result

def pure(function, graph):
    report('pure-before', function, graph)
    result = original_pure(function, graph)
    report('pure-after', function, graph)
    print('PURE-RESULT', result, flush=True) if 'step_with_dt_control_used__specialized' in function.name and '__planned_region_' not in function.name else None
    return result

shell._recover_late_source_unary_operations = unary
shell._recover_late_source_pure_expressions = pure
checkpoint=Path('build/patch_sequence_replay_v76/pre-frame-link.pkl')
positional, keywords = pickle.loads(checkpoint.read_bytes())
keywords = {**keywords, 'progress': lambda message: print(message, flush=True)}
module, outputs, exports = shell._class_surface_ssa_program(*positional, **keywords)
print('FINDINGS', len(run_all(module)), flush=True)
