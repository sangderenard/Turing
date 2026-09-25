import runpy
from pathlib import Path
import numpy as np
lower=runpy.run_path('tests/test_aggregate_call_identity.py')['_lower']
source='''
def recover(saved):
    return saved[0]
def tick(left, right, enabled):
    saved = (left,) if enabled else (right,)
    return recover(saved)
'''
m,o,e=lower(source,'tuple_merge',{'left':np.zeros((2,3)), 'right':np.ones((2,3)), 'enabled':True})
print('OUTPUTS',o,flush=True)
for n,f in m.functions.items():
    if n.startswith('tuple_merge'):
        print(n,[(a.id,a.shape) for a in f.args],flush=True)
        print([(i.op, None if i.res is None else (i.res.id,i.res.dtype,i.res.shape),[(a.id,a.dtype,a.shape) for a in i.args]) for b in f.blocks.values() for i in b.instrs if i.op in ('Phi','Call','Ret')],flush=True)
