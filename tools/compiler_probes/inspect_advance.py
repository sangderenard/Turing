import pickle
m=pickle.load(open('build/full_formal_diagnostic/repository-ssa.pkl','rb'))
f=m.functions['balloon_tire_managed_python__balloon_tire_managed_advance']
for vid in (127,130,132,133):
 print('\nVALUE',vid,'arg',[(a.id,a.dtype,a.shape,a.accounting) for a in f.args if a.id==vid])
 for bn,b in f.blocks.items():
  for i,x in enumerate(b.instrs):
   if x.res is not None and x.res.id==vid or any(a.id==vid for a in x.args):
    print(bn,i,x.op,[a.id for a in x.args],None if x.res is None else x.res.id,x.attributes)
