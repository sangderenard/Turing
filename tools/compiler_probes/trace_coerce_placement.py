import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def local_trace(frame,event,arg):
    if event=='return':
        v=frame.f_locals
        print('MARKER RETURN',arg,'marker',getattr(v.get('block'),'name',None),'index',v.get('index'),'relocated',v.get('relocated'),'consumers',v.get('consumers'),flush=True)
        print('INPUTS',[(i.op,[a.id for a in i.args]) for i in v.get('sequence',())],flush=True)
    return local_trace
def trace(frame,event,arg):
    if event=='call' and frame.f_code.co_name in {'replace_at_callsite_marker','insert_at_loop_anchor'}:
        record=frame.f_locals.get('record')
        if record and record.callsite_id==460 and 'step_with_dt_control_used' in record.caller:
            print('LINK TRACE',frame.f_code.co_name,record.caller,flush=True)
            return local_trace
    return None
from src.compiler.vehicle_python_compilation import lower_balloon_tire_managed_python_ssa
sys.settrace(trace)
try:
    result=lower_balloon_tire_managed_python_ssa(batch_size=8,window_duration=2**-20,dt_initial=2**-20,progress=lambda message: print(message,flush=True))
finally:
    sys.settrace(None)
print('LOWERING COMPLETE',flush=True)
