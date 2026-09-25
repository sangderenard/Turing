import pickle
m=pickle.load(open('build/full_formal_diagnostic/repository-ssa.pkl','rb'))
for n in ['balloon_tire_managed_python__run_superstep__specialized_b0c1d57d7251','balloon_tire_managed_python__step_with_dt_control_used__specialized_d8d399b621bf']:
 f=m.functions[n]
 print(n)
 for k,v in sorted(f.metadata.items()):
  if any(x in k.lower() for x in ('id','source','rebind','alias','collision','required')):
   s=repr(v)
   print(k, s[:1200])
