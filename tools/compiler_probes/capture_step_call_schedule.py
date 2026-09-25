import pickle
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.compiler import precompile_to_ssa as pre
original = pre._schedule_loop_callsites
def capture(control, hierarchy_plan, region_signatures, region_dependency_signatures=None):
    result=original(control,hierarchy_plan,region_signatures,region_dependency_signatures)
    if hierarchy_plan is not None and 'step_with_dt_control_used' in str(hierarchy_plan.name):
        Path('build/step-call-schedule.pkl').write_bytes(pickle.dumps((control,hierarchy_plan,region_signatures,region_dependency_signatures,result)))
        print('SAVED STEP SCHEDULE',hierarchy_plan.name,flush=True)
    return result
pre._schedule_loop_callsites=capture
from src.compiler.vehicle_python_compilation import lower_balloon_tire_managed_python_ssa
result=lower_balloon_tire_managed_python_ssa(batch_size=8,window_duration=2**-20,dt_initial=2**-20,progress=lambda message: print(message,flush=True))
print('LOWERING COMPLETE',flush=True)
