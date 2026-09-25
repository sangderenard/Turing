import runpy
from src.compiler import glsl_deployment_strategy as s
original=s._ordinary_conditional_control_programs

def inspect(graph, retained, subgraphs):
    retained=tuple(retained); subgraphs=tuple(subgraphs)
    controls=original(graph,retained,subgraphs)
    if graph.G.graph.get('function_name')=='root':
        print('CONTROLS',[(c.region_indices,c.anchor_region,c.root) for c in controls],flush=True)
        for i,g in enumerate(subgraphs):
            print('REGION',i,[(n,graph.G.nodes[n].get('source_span'),graph.G.nodes[n].get('type')) for n in g.G.graph.get('deployment_nodes',())],flush=True)
    return controls
s._ordinary_conditional_control_programs=inspect
runpy.run_module('build.probe_authored_snapshot',run_name='__main__')
