import pickle
m=pickle.load(open('build/full_formal_diagnostic/repository-ssa.pkl','rb'))
f=m.functions['balloon_tire_managed_python__step_with_dt_control_used__specialized_d8d399b621bf']
for k,v in sorted(f.metadata.items()):
 if any(s in k.lower() for s in ('loop','control','identity','carried')):
  print(k,repr(v)[:8000])
