import pickle
m=pickle.load(open('build/full_formal_diagnostic/repository-ssa.pkl','rb'))
for fn in ['balloon_tire_managed_python__run_superstep__specialized_b0c1d57d7251','balloon_tire_managed_python__step_with_dt_control_used__specialized_d8d399b621bf']:
 f=m.functions[fn]; print('\n',fn)
 for bn,b in f.blocks.items():
  for i,x in enumerate(b.instrs):
   if x.op=='Phi': print(bn,i,[a.id for a in x.args],x.res.id,x.attributes)
