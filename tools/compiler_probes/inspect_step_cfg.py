import pickle
m=pickle.load(open('build/full_formal_diagnostic/repository-ssa.pkl','rb'))
f=m.functions['balloon_tire_managed_python__step_with_dt_control_used__specialized_d8d399b621bf']
for bn,b in f.blocks.items():
 term=b.instrs[-1] if b.instrs else None
 print(bn,'succ',b.successors,'term',None if term is None else (term.op,[a.id for a in term.args],term.attributes))
