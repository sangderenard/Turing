import runpy
from src.compiler import glsl_deployment_strategy as strategy
original = strategy._callsite_specialized_shell_type
original_strategy = strategy.strategize_shell_deployment
def plan(graph, *args, **kwargs):
    if graph.G.graph.get('function_name') == 'recover':
        print('RECOVER GRAPH', {key: graph.G.graph.get(key) for key in ('identity_table', 'function_outputs', 'return_slot_values', 'planner_tensor_descriptors')}, flush=True)
        print('RECOVER NODES', [(key, value.get('type'), value.get('attributes'), value.get('tensor')) for key, value in graph.G.nodes(data=True)], flush=True)
    return original_strategy(graph, *args, **kwargs)
strategy.strategize_shell_deployment = plan
def inspect(owner, node_id, reference, fallback, max_nodes_per_dispatch):
    graph = owner.process_graph
    if graph.G.graph.get('function_name') == 'tick':
        print('CALL', node_id, graph.G.nodes[node_id], flush=True)
        callee = graph.function_table.entry(reference).graph
        print('CALLEE PARAMS', strategy._method_parameter_layout(callee.G), callee.G.graph.get('identity_table'), flush=True)
        for parent, role in graph.G.nodes[node_id].get('parents', ()):
            if str(role).startswith('arg'):
                print('ARG', parent, graph.G.nodes[parent], flush=True)
    return original(owner, node_id, reference, fallback, max_nodes_per_dispatch)
strategy._callsite_specialized_shell_type = inspect
runpy.run_module('build.probe_singleton_snapshot', run_name='__main__')
