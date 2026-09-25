import pickle
m=pickle.load(open('build/full_formal_diagnostic/repository-ssa.pkl','rb'))
f=m.functions['balloon_tire_managed_python__step_with_dt_control_used__specialized_d8d399b621bf']
print('args43/24',[(a.id,a.dtype,a.accounting) for a in f.args if a.id in (24,43)])
for bn,b in f.blocks.items():
 for i,x in enumerate(b.instrs):
  if x.op=='Phi' and (x.attributes.get('binding') in {'loop_carried','loop_result_port'}):
   print(bn,i,[a.id for a in x.args],x.res.id,x.attributes)
for vid in (43,566,24,280,167,169,273):
 print('\nVALUE',vid)
 for bn,b in f.blocks.items():
  for i,x in enumerate(b.instrs):
   if (x.res is not None and x.res.id==vid) or any(a.id==vid for a in x.args):
    print(bn,i,x.op,[a.id for a in x.args],None if x.res is None else x.res.id,x.attributes)
