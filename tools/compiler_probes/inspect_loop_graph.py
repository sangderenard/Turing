import pickle,ast,pprint
x=pickle.load(open('build/full_formal_diagnostic/resolved-process-graph.pkl','rb'))
for e in x.function_table:
 q=str(getattr(e,'qualified_name','') or getattr(e,'name',''))
 if 'step_with_dt_control_used' not in q: continue
 g=e.graph.G
 for vid in (592,43,566,268,276,353,553,44):
  d=g.nodes.get(vid)
  print('\nNODE',vid)
  if d:
   for k,v in d.items():
    if k not in ('expr_obj',): print(k,repr(v)[:4000])
   ex=d.get('expr_obj'); print('source',ast.unparse(ex) if isinstance(ex,ast.AST) else None)
