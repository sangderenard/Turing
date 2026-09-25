from src.compiler import fortran_c_shell as shell
from src.compiler.extraction_contract import ExtractionContract
SOURCE='''
class Material:
    def copy_shallow(self):
        return (self.state.copy(),)
    def restore(self, saved):
        self.state[...] = saved[0]
def root(material, rollback):
    saved = material.copy_shallow()
    material.state[0] = 7.0
    material.telemetry[0] = material.telemetry[0] + 1.0
    material.restore(saved)
    return material.state[0]
'''
policy=ExtractionContract('extraction_contracts/program_extraction.yaml').with_execution_file('extraction_contracts/vehicle_full_native_execution.yaml').with_program_abi({
 'records':{'Material':{'identity':'Material','fields':{name:{'storage':'span','dtype':'float64','rank':1,'shape':[2],'mutable':True} for name in ('state','telemetry')}}},
 'bindings':[{'function':'root','parameter':'material','record':'Material'}],
 'values':[{'function':'root','parameter':'rollback','storage':'scalar','dtype':'bool','rank':0,'python_type':'builtins.bool'}]})
# Diagnostic only: ask ordinary source ingestion to handle the real methods.
# The production all-mutable-fields rewrite is intentionally bypassed in memory.
shell._normalize_record_snapshot_methods=lambda tree, abi: ()
import pickle
original_gate=shell._full_native_link_failures
def capture(*args,**kwargs):
    Path('build/authored-snapshot-failed.pkl').write_bytes(pickle.dumps(kwargs['module']))
    return original_gate(*args,**kwargs)
from pathlib import Path
shell._full_native_link_failures=capture
m,o,e=shell.lower_ast_source_to_ssa(SOURCE,'root',name='snapshot_authored',extraction_contract=policy)
print('LOWERED',e)
for n,f in m.functions.items():
 if '__planned_region' not in n:print(n, [(a.id,a.dtype,a.accounting) for a in f.args])
