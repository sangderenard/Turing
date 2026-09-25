import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['TURING_DEBUG_CONTROL_OVERLAY'] = '1'
os.environ['TURING_DEBUG_REGION_ORDER'] = '1'
ns = runpy.run_path('tests/test_native_pruned_cap_initialization.py')
test = ns['test_pruned_optional_cap_keeps_initial_symbolic_value']
def inspect(module, root):
    import pickle
    Path('build/pruned-cap-ssa.pkl').write_bytes(pickle.dumps((module, root)))
    return SimpleNamespace(complete=True, compile=lambda *args: sys.exit(0))
test.__globals__['emit_ssa_to_c'] = inspect
test(Path('build'), through_loop=True, runtime_limits=True)
