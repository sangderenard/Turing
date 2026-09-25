import pickle
m=pickle.load(open('build/full_formal_diagnostic/repository-ssa.pkl','rb'))
for fn, ids in [('balloon_tire_managed_python__run_superstep__specialized_b0c1d57d7251',(83,)),('balloon_tire_managed_python__step_with_dt_control_used__specialized_d8d399b621bf',(169,273))]:
 f=m.functions[fn]; print('\n',fn)
 print('recovered',f.metadata.get('recovered_late_source_reductions'))
 for vid in ids:
  print('formal',vid,any(a.id==vid for a in f.args))
  for bn,b in f.blocks.items():
   for i,x in enumerate(b.instrs):
    if (x.res is not None and x.res.id==vid) or any(a.id==vid for a in x.args):
     print(vid,bn,i,x.op,[a.id for a in x.args],None if x.res is None else x.res.id,x.attributes)
