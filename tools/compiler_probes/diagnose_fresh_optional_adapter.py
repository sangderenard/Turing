from __future__ import annotations
from pathlib import Path
import pickle,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.compiler import fortran_c_shell as shell
from src.compiler import ssa_call_input_adapters as adapters
original=adapters.adapt_physical_call_inputs

def snapshots(functions,label):
 for name,function in functions.items():
  if 'step_with_dt_control_used__specialized' not in name or '__planned_region_' in name: continue
  formal=next((x for x in function.args if int(x.id)==109),None)
  print(label,'ROOTFORMAL',id(formal),formal and formal.dtype,formal and formal.accounting,flush=True)
  for bn,b in function.blocks.items():
   for i,ins in enumerate(b.instrs):
    if ins.op!='Call': continue
    callee=functions.get(str(ins.attributes.get('callee','')))
    if callee is None: continue
    for pos,(actual,calleeformal) in enumerate(zip(ins.args,callee.args)):
     if int(actual.id)==109:
      print(label,'EDGE',bn,i,ins.attributes.get('callee'),pos,'actualobj',id(actual),actual.dtype,actual.accounting,'formalobj',id(calleeformal),calleeformal.id,calleeformal.dtype,calleeformal.accounting,'same',actual is calleeformal,flush=True)

def wrapped(functions):
 snapshots(functions,'BEFORE')
 result=original(functions)
 snapshots(functions,'AFTER')
 print('ADAPTED',result,flush=True)
 return result
adapters.adapt_physical_call_inputs=wrapped
positional,keywords=pickle.loads(Path('build/patch_sequence_fresh_v89/pre-frame-link.pkl').read_bytes())
keywords={**keywords,'progress':lambda message: print(message,flush=True)}
try: shell._class_surface_ssa_program(*positional,**keywords)
except Exception as error: print('STOP',type(error).__name__,error,flush=True)
