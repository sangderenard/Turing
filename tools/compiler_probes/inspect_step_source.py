import pickle,ast
x=pickle.load(open('build/full_formal_diagnostic/resolved-process-graph.pkl','rb'))
for e in x.function_table:
 q=str(getattr(e,'qualified_name','') or getattr(e,'name',''))
 if 'step_with_dt_control_used' not in q: continue
 g=e.graph.G
 print('ENTRY',q,'identity',g.graph.get('identity_table'))
 for vid in (43,165,166,167,169,273,301,554):
  if vid in g:
   d=g.nodes[vid]
   ex=d.get('expr_obj')
   print(vid,d.get('type'),d.get('op'),ast.unparse(ex) if isinstance(ex,ast.AST) else None,d.get('parents'),d.get('attributes'))
