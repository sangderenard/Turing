import ast
import pickle
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.compiler.glsl_deployment_strategy as strategy
original = strategy._overlay_control_or_require_subdivision
def capture(graph, *args, **kwargs):
    result = original(graph, *args, **kwargs)
    if graph.G.graph.get('function_name') == 'root':
        Path('build/return-overlay.pkl').write_bytes(pickle.dumps((graph.G, graph.roots, args, result)))
        print('ROOTS', graph.roots, 'RETURNS', graph.G.graph.get('return_slot_values'), flush=True)
        print('LOOPS', args[2], 'RESULT', result, flush=True)
        for n,d in graph.G.nodes(data=True):
            e = d.get('expr_obj')
            if isinstance(e, (ast.Tuple, ast.Return, ast.While)):
                print('NODE', n, ast.unparse(e), d.get('value_id'), flush=True)
        raise SystemExit(0)
    return result
strategy._overlay_control_or_require_subdivision = capture
ns=runpy.run_path('tests/test_native_return_after_call.py')
ns['test_return_after_conditional_and_source_call_uses_produced_value'](Path('build'), True)
