import runpy
import ast
from src.compiler import fortran_c_shell as shell
from src.compiler import glsl_deployment_strategy as strategy

original = shell.lower_ast_source_to_ssa
def inspect_graph(graph):
    for entry in graph.function_table:
        function = getattr(entry, 'graph', None)
        if getattr(function, 'G', None) is None:
            continue
        metadata = function.G.graph
        print('GRAPH', metadata.get('function_name'),
              'owner', metadata.get('method_owner'),
              'records', metadata.get('parameter_record_abi'), flush=True)
        if metadata.get('function_name') == 'restore':
            print('BODY', '\n'.join(ast.unparse(item) for item in metadata.get('function_body', ())), flush=True)
            for key, data in function.G.nodes(data=True):
                print('RESTORE', key, data.get('op'), data.get('type'), data.get('parents'), data.get('attributes'), flush=True)
        for key, data in function.G.nodes(data=True):
            if str(data.get('op', '')).lower() in ('getattr', 'copy', 'clone'):
                print('NODE', key, data.get('op'), data.get('attributes'),
                      'tensor', data.get('tensor'),
                      'descriptor', strategy._tensor_descriptor(function, key), flush=True)
def capture(*args, **kwargs):
    kwargs['resolved_process_graph_sink'] = inspect_graph
    return original(*args, **kwargs)
shell.lower_ast_source_to_ssa = capture
runpy.run_module('build.probe_authored_snapshot', run_name='__main__')
