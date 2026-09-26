import json,pickle
from pathlib import Path
p=Path('build/full_formal_diagnostic')
module=pickle.loads((p/'repository-ssa.pkl').read_bytes())
reports=json.loads((p/'formals.json').read_text())
for row in reports:
 f=module.functions[row['function']]
 v=next(x for x in f.args if x.id==row['value_id'])
 print('\n',row['function'],row['value_id'],'acct=',v.accounting)
 print('metadata keys:', {k:f.metadata.get(k) for k in ('required_source_value_ids','ssa_value_id_rebindings','value_id_rebindings','source_value_id_aliases','source_output_value_ids') if f.metadata.get(k)})
 for bn,b in f.blocks.items():
  for i,ins in enumerate(b.instrs):
   if any(a is v or a.id==v.id for a in ins.args):
    print(' use',bn,i,ins.op,'args',[a.id for a in ins.args],'attrs',{k:ins.attributes.get(k) for k in ('feed_ids','output_ids','region_index','binding','callee') if k in ins.attributes})
