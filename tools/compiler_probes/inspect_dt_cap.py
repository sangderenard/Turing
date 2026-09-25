import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
m, _, _ = pickle.loads(Path('build/managed_dt_forwarded_marker_20260905/repository-ssa.pkl').read_bytes())
n = 'balloon_tire_managed_python__run_superstep__specialized_b0c1d57d7251'
f = m.functions[n]
for value in f.args:
    if value.id in (7, 16, 175):
        print('FORMAL', value)
for block in f.blocks.values():
    for i in block.instrs:
        if any(v.id == 175 for v in i.args) or (i.res is not None and i.res.id == 175):
            print('USE', block.name, i)
for owner, caller in m.functions.items():
    for block in caller.blocks.values():
        for i in block.instrs:
            if i.op == 'Call' and i.attributes.get('callee') == n:
                receipt = i.attributes.get('callee_input_ids', [v.id for v in f.args])
                actual = next(v for v, formal_id in zip(i.args, receipt) if formal_id == 175)
                print('BINDING', owner, actual)
                for b in caller.blocks.values():
                    for producer in b.instrs:
                        if producer.res is not None and producer.res.id == actual.id:
                            print('PRODUCER', producer)
print('METADATA KEYS', f.metadata.keys())
print('ALIASES', f.metadata.get('value_aliases'))
import pprint
for key in ('value_names', 'carried_port_values', 'specialized_conditional_node_ids', 'deployment_regions', 'control_ir'):
    print('META', key)
    pprint.pprint(f.metadata.get(key), depth=4, width=120)
region = m.functions.get(n+'__planned_region_5')
print('REGION5', region)
