"""Differential and bounded timing check, not a replacement graph/backend.

Requires Spectral's optional AbstractTensor backend. Does not start services,
render shaders or change production state. Output belongs in user scratch.
"""
import os, sys, inspect, time, json, statistics
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description='Validate existing NumPy lowering on actual Spectral signal graphs')
parser.add_argument('--spectral-root', required=True)
parser.add_argument('--output-dir', required=True, help='Untracked user-owned scratch directory')
args = parser.parse_args()
r = Path(args.output_dir)
r.mkdir(parents=True, exist_ok=True)
os.environ['SPECTRAL_TENSOR_BACKEND'] = 'abstract'
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(args.spectral_root).resolve())]
import graph_solver as g
from src.compiler.fused_program_python_backend import lower_functions_to_numpy
from src.common.tensors import AbstractTensor as A
import numpy as np
g.PROFILE_TIMING = False
g.TRACE_EVENTS = False
p = lower_functions_to_numpy([], namespace=vars(g), owned=lambda v: getattr(v, '__module__', None) == 'graph_solver', subclass_methods={g.GraphSolver: tuple((n for n, v in g.GraphSolver.__dict__.items() if inspect.isfunction(v))), g.TensorEdge: ('apply',)})
(r / 'actual_graph_solver_numpy.py').write_text(p.source, encoding='utf-8')
ns = {n: vars(g)[n] for n in p.injected if n in vars(g)}
exec(p.source, ns)
results = []
for label, links in [('chain', [('radio', 'rack', 0.5), ('rack', 'light', 0.5)]), ('cycle', [('radio', 'rack', 0.5), ('rack', 'light', 0.5), ('light', 'rack', 0.2)])]:
    nodes = [g.TensorNode(key=k, layer='signal') for k in ('radio', 'rack', 'light')]
    edges = [g.TensorEdge(src_key=a, dst_key=b, weight=w) for a, b, w in links]
    interpreted = g.GraphSolver(nodes, edges)
    lowered = ns['GraphSolver'](nodes, edges)
    raw = np.linspace(0.1, 0.8, 256).astype('complex128')
    wrapped = A.tensor(raw)
    a = interpreted.run_schedule({'radio': wrapped}, n_frames=256)['light'].data
    b = lowered.run_schedule({'radio': raw}, n_frames=256)['light']
    assert isinstance(b, np.ndarray)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    timings = {'abstract': [], 'numpy': []}
    cases = [('abstract', interpreted, wrapped), ('numpy', lowered, raw)]
    for _, solver, feed in cases:
        for _ in range(5):
            solver.run_schedule({'radio': feed}, n_frames=256)
    for batch in range(7):
        for name, solver, feed in cases if batch % 2 == 0 else cases[::-1]:
            start = time.perf_counter()
            for _ in range(50):
                solver.run_schedule({'radio': feed}, n_frames=256)
            timings[name].append((time.perf_counter() - start) / 50)
    timings = {name: {'median': statistics.median(samples), 'min': min(samples), 'max': max(samples)} for name, samples in timings.items()}
    results.append({'graph': label, 'frames': 256, 'repeat': 50, 'seconds_per_window': timings, 'max_error': float(np.max(np.abs(a - b)))})
print(json.dumps(results, indent=2))
(r / 'numpy_graph_comparison_7batches.json').write_text(json.dumps(results, indent=2))
g._T._events.clear()
for label, delay, transform in [('delayed', 1, None), ('nonlinear', 0, lambda x: x * x)]:
    nodes = [g.TensorNode(key='radio', layer='input'), g.TensorNode(key='light', layer='output', transform=transform)]
    edges = [g.TensorEdge(src_key='radio', dst_key='light', weight=0.5, delay_samples=delay)]
    a = g.GraphSolver(nodes, edges)
    b = ns['GraphSolver'](nodes, edges)
    raw = np.array([0.4, 0.8], dtype='complex128')
    expected = a.run_schedule({'radio': A.tensor(raw)}, n_frames=2)['light'].data
    actual = b.run_schedule({'radio': raw}, n_frames=2)['light']
    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    print(label, 'PASS', actual)
g._T._events.clear()
from camera_software.trigger_grid_bridge import CameraTriggerGrid
panel = CameraTriggerGrid(['shop:17', 'studio:0'], ['rack:center'])
measurement = {'status': 'live', 'rms': 0.4, 'bands': [0.25, 0.75], 'frame': '1024'}
panel.radio(measurement)
panel.graph.set_weight('radio:rms', 'rack:center', 0.5)
panel.graph.set_weight('rack:center', 'shop:17', 0.5)
expected = panel.radio(measurement)
original = panel._radio_runtime.solver
lowered = ns['GraphSolver'](original.nodes, original.edges, sample_rate=original.sample_rate)
actual = lowered.run_schedule({'radio:rms': np.array([0.4], dtype='complex128'), 'radio:band:1': np.array([0.25], dtype='complex128'), 'radio:band:2': np.array([0.75], dtype='complex128')}, n_frames=1)
for name in panel.receivers:
    assert isinstance(actual[name], np.ndarray)
    np.testing.assert_allclose(abs(actual[name].reshape(-1)[0]), expected['gains'][name], atol=1e-12, rtol=1e-12)
print('PASS actual full camera grid graph', expected['gains'])
g._T._events.clear()
