from pathlib import Path
import runpy
from src.compiler import fortran_c_shell as shell

original = shell.lower_ast_source_to_ssa

def inspect_graph(graph):
    lines = []
    for entry in graph.function_table:
        function = getattr(entry, 'graph', None)
        if getattr(function, 'G', None) is None:
            continue
        lines.append(repr(entry))
        lines.append(repr(function.G.graph))
        lines.extend(f'{key}: {value!r}' for key, value in function.G.nodes(data=True))
    Path('build/snapshot-graph.txt').write_text('\n'.join(lines), encoding='utf-8')

def capture(*args, **kwargs):
    kwargs['resolved_process_graph_sink'] = inspect_graph
    return original(*args, **kwargs)

shell.lower_ast_source_to_ssa = capture
runpy.run_module('build.probe_authored_snapshot', run_name='__main__')
