import ast
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.compiler.glsl_deployment_strategy as strategy
from src.compiler.vehicle_python_compilation import lower_balloon_tire_managed_python_ssa

original = strategy._overlay_control_or_require_subdivision

def capture(graph, regions, reductions, loops, conditionals, nesting, **kwargs):
    result = original(graph, regions, reductions, loops, conditionals, nesting, **kwargs)
    if graph.G.graph.get('function_name') == 'step_with_dt_control_used':
        Path('build/step-overlay.pkl').write_bytes(pickle.dumps(
            (graph.G, regions, reductions, loops, conditionals, nesting, result, kwargs)))
        print('Saved step_with_dt_control_used overlay', flush=True)
        print('REGIONS', regions, 'LOOPS', [c.region_indices for c in loops],
              'IFS', [c.region_indices for c in conditionals], flush=True)
        print('CONTROL', result, flush=True)
        raise SystemExit(0)
    return result

strategy._overlay_control_or_require_subdivision = capture
lower_balloon_tire_managed_python_ssa(batch_size=8, window_duration=2**-20,
                                     dt_initial=2**-20, progress=print)
