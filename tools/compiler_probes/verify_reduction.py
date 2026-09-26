import pickle
m=pickle.load(open('build/full_formal_diagnostic/repository-ssa.pkl','rb'))
f=m.functions['balloon_tire_managed_python__balloon_tire_managed_advance']
print('formal132', [a for a in f.args if a.id == 132])
for name,b in f.blocks.items():
 for i,x in enumerate(b.instrs):
  if (x.res is not None and x.res.id == 132) or any(a.id == 132 for a in x.args):
   print(name,i,x.op,[a.id for a in x.args],None if x.res is None else x.res.id,x.res.dtype if x.res else None,x.attributes)
print('metadata',f.metadata.get('recovered_late_source_reductions'))
