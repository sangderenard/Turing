import pickle
m=pickle.load(open('build/full_formal_diagnostic/repository-ssa.pkl','rb'))
f=m.functions['balloon_tire_managed_python__balloon_tire_managed_advance']
for vid in (127,130,132):
 vals=[]
 vals += [a for a in f.args if a.id==vid]
 vals += [x.res for b in f.blocks.values() for x in b.instrs if x.res is not None and x.res.id==vid]
 for v in vals: print(vid,v.dtype,v.shape,v.accounting)
