import pickle
m=pickle.load(open('build/full_formal_diagnostic/repository-ssa.pkl','rb'))
items=[('balloon_tire_managed_python__run_superstep__specialized_b0c1d57d7251',(41,83)),('balloon_tire_managed_python__step_with_dt_control_used__specialized_d8d399b621bf',(43,165,166,167,169,273))]
for fn,ids in items:
 f=m.functions[fn]; print('\nFUNCTION',fn)
 for vid in ids:
  print(' VALUE',vid)
  for a in f.args:
   if a.id==vid: print('  ARG',a.dtype,a.shape,a.accounting)
  for bn,b in f.blocks.items():
   for i,x in enumerate(b.instrs):
    if (x.res is not None and x.res.id==vid) or any(a.id==vid for a in x.args):
     print(' ',bn,i,x.op,[a.id for a in x.args],None if x.res is None else x.res.id, None if x.res is None else (x.res.dtype,x.res.shape,x.res.accounting),{k:x.attributes.get(k) for k in ('source_output_id','region_index','binding','feed_ids') if k in x.attributes})
