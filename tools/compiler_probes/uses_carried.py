import pickle
m=pickle.load(open('build/full_formal_diagnostic/repository-ssa.pkl','rb'))
f=m.functions['balloon_tire_managed_python__step_with_dt_control_used__specialized_d8d399b621bf']
for vid in (713,714,568):
 print('\nVALUE',vid)
 for bn,b in f.blocks.items():
  for i,x in enumerate(b.instrs):
   if (x.res is not None and x.res.id==vid) or any(a.id==vid for a in x.args):
    print(bn,i,x.op,[a.id for a in x.args],None if x.res is None else x.res.id,{k:x.attributes.get(k) for k in ('region_index','feed_ids','binding','initial_value_id','updated_value_id') if k in x.attributes})
