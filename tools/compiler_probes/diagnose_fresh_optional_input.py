from __future__ import annotations
from pathlib import Path
import pickle, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.compiler import fortran_c_shell as shell
from src.compiler import ssa_call_input_adapters as adapters
original=adapters.physical_call_input_conflicts

def diagnose(functions):
    for name,function in functions.items():
        if 'step_with_dt_control_used__specialized' not in name or '__planned_region_' in name:
            continue
        print('DIAG-CALLER',name,flush=True)
        for position,arg in enumerate(function.args):
            if int(arg.id)==109:
                print('FORMAL109',position,arg.dtype,arg.shape,arg.accounting,flush=True)
        for block_name,block in function.blocks.items():
            for index,instruction in enumerate(block.instrs):
                if (instruction.res is not None and int(instruction.res.id)==109) or any(int(arg.id)==109 for arg in instruction.args):
                    print('OCC109',block_name,index,instruction.op,[f'{arg.id}:{arg.dtype}:{arg.shape}:{arg.accounting}' for arg in instruction.args],None if instruction.res is None else f'{instruction.res.id}:{instruction.res.dtype}:{instruction.res.shape}:{instruction.res.accounting}',instruction.attributes,flush=True)
                if instruction.op=='Call' and 'pi_update__specialized' in str(instruction.attributes.get('callee')):
                    callee=functions[str(instruction.attributes['callee'])]
                    print('PI-CALL',block_name,index,flush=True)
                    for pos,(actual,formal) in enumerate(zip(instruction.args,callee.args)):
                        print('PAIR',pos,actual.id,actual.dtype,actual.shape,actual.accounting,'=>',formal.id,formal.dtype,formal.shape,formal.accounting,flush=True)
    result=original(functions)
    print('CONFLICTS',result,flush=True)
    return result
adapters.physical_call_input_conflicts=diagnose
positional,keywords=pickle.loads(Path('build/patch_sequence_fresh_v89/pre-frame-link.pkl').read_bytes())
keywords={**keywords,'progress':lambda message: print(message,flush=True)}
try:
    shell._class_surface_ssa_program(*positional,**keywords)
except Exception as error:
    print('STOP',type(error).__name__,error,flush=True)
