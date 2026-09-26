import ast,contextlib,io
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
s=ast.parse('''\ndef retry(value, stable, rejected):\n    while True:\n        if rejected:\n            value, stable = value - 1, stable\n            continue\n        return stable\n''')
g=ProcessGraph(materialize_memory=False)
with contextlib.redirect_stdout(io.StringIO()): g.build_from_ast(s)
reduce_abstract_tensor_topology(g)
e=g.function_table.entry('retry').graph.G
loop=next(d for _,d in e.nodes(data=True) if d.get('type')=='While')
print(loop['attributes']['loop_carried_bindings'])
for name,(a,b) in loop['attributes']['loop_carried_bindings'].items():
 print(name,a,e.nodes.get(a),b,e.nodes.get(b))
print(e.graph.get('loop_control_site_bindings'))
