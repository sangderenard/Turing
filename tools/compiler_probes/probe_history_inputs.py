import runpy
from src.compiler import glsl_deployment_strategy as strategy
original = strategy.strategize_shell_deployment

def inspect(graph, *args, **kwargs):
    if graph.G.graph.get('function_name') == 'inner_recurrence':
        print('INPUT NODES', [(i,d.get('attributes')) for i,d in graph.G.nodes(data=True) if d.get('type') == 'Input'], flush=True)
        print('LOOP NODES', [(i,d.get('attributes')) for i,d in graph.G.nodes(data=True) if d.get('type') in ('For','Loop')], flush=True)
    return original(graph, *args, **kwargs)
strategy.strategize_shell_deployment = inspect
runpy.run_module('build.probe_history_formals', run_name='__main__')
