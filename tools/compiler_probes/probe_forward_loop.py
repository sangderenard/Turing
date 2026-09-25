import pickle
from pathlib import Path
from src.common.dt_system.dt_scaler import Metrics, coerce_metrics
from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_self_check import check_definition_dominance
policy=ExtractionContract('extraction_contracts/program_extraction.yaml').with_execution_file('extraction_contracts/vehicle_full_native_execution.yaml')
abi=policy.program_abi.receipt()
abi['bindings'].append({'function':'normalize','parameter':'value','record':'Metrics'})
policy=policy.with_program_abi(abi)
source='''
def advance(dt):
    return Metrics(dt * 2.0, dt, 0.0, 0.0, error_channels={'error': dt})
def normalize(value):
    value.max_vel = value.max_vel * 0.5
    return value
def root(dt):
    while dt > 0.0:
        metrics = advance(dt)
        metrics = coerce_metrics(metrics)
        return float(metrics.max_vel)
    return dt
'''
m,o,e=lower_ast_source_to_ssa(source,'root',name='forward_loop',extraction_contract=policy,python_bindings={'Metrics':Metrics,'coerce_metrics':coerce_metrics,'AbstractTensor':AbstractTensor},tensor_ssa_reference=c_backend_repository_ssa_reference())
Path('build/forward-loop.pkl').write_bytes(pickle.dumps((m,o,e)))
for finding in check_definition_dominance(m): print(finding)
for n,f in m.functions.items():
 if n.endswith('__root'):
  for b in f.blocks.values():
   for i in b.instrs:
    print(b.name,i.op, i.res.id if i.res else None,[v.id for v in i.args], i.attributes if i.op=='Call' else '')



