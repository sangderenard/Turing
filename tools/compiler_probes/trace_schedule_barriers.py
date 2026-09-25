import pickle,sys
from pathlib import Path
from src.compiler.precompile_to_ssa import _schedule_loop_callsites
from src.compiler.control_source import WhileBlock, SequenceBlock
p=pickle.loads(Path('build/step-call-schedule.pkl').read_bytes())
def trace(f,event,arg):
 if event=='return' and f.f_code.co_name=='dependency_signature' and arg is None:
  b=f.f_locals['block']
  print('BARRIER',type(b).__name__,str(b)[:320])
 return trace
sys.settrace(trace)
r=_schedule_loop_callsites(*p[:4])
sys.settrace(None)
def walk(b):
 if isinstance(b,WhileBlock) and b.source_loop_node_id==616:
  for i,c in enumerate(b.body.blocks):print('BODY',i,type(c).__name__,str(c)[:190])
 elif isinstance(b,SequenceBlock):
  for c in b.blocks:walk(c)
walk(r[0].root)
