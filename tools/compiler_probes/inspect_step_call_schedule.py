import pickle
from pathlib import Path
from src.compiler.control_source import SequenceBlock, StatementBlock, ConditionalBlock, WhileBlock, LoopBlock, CallBlock
from src.compiler.precompile_to_ssa import _schedule_loop_callsites
control,plan,signatures,dependencies,saved=pickle.loads(Path('build/step-call-schedule.pkl').read_bytes())
print('PLAN',plan.name)
for position,item in enumerate(plan.items):
    if type(item).__name__=='PlanCall':
        print(position, 'CALL',item.callsite_id,'args',item.argument_bindings,'results',item.result_bindings,'loops',item.enclosing_loop_ids)
    else:
        print(position,type(item).__name__,getattr(item,'name',None))
def walk(block,path='root'):
    if isinstance(block, StatementBlock):
        for line in block.lines:
            if '__plan_callsite_' in line: print(path,line)
    elif isinstance(block, SequenceBlock):
        for i,child in enumerate(block.blocks):walk(child,f'{path}/{i}')
    elif isinstance(block, ConditionalBlock):
        walk(block.body,path+'/true')
        if block.orelse:walk(block.orelse,path+'/false')
    elif isinstance(block,(WhileBlock,LoopBlock)):
        walk(block.body,path+f'/loop{block.source_loop_node_id}')
        if isinstance(block,WhileBlock):walk(block.condition,path+'/predicate')
    elif isinstance(block,CallBlock):walk(block.callee,path+'/callee')
print('SAVED BINDING460',saved[1].get(460))
walk(saved[0].root)
replayed=_schedule_loop_callsites(control,plan,signatures,dependencies)
print('REPLAY MATCHES',replayed==saved)
walk(replayed[0].root)
