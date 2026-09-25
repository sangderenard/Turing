import runpy
from src.compiler import glsl_deployment_strategy as s
original=s._propagate_callsite_tensor_specializations

def inspect(graph):
    for entry in graph.function_table:
        g=entry.graph
        if g is None: continue
        if g.G.graph.get('function_name') in ('copy_shallow','restore','root'):
            print('GRAPH',entry.reference.address,g.G.graph.get('function_name'),'ABI',g.G.graph.get('parameter_record_abi'),flush=True)
            print('NODES',[(n,d.get('type'),d.get('attributes'),d.get('tensor')) for n,d in g.G.nodes(data=True)],flush=True)
    return original(graph)
s._propagate_callsite_tensor_specializations=inspect
runpy.run_module('build.probe_authored_snapshot',run_name='__main__')
