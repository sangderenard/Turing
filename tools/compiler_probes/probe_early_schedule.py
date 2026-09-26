from pathlib import Path
from src.compiler import precompile_to_ssa as p
original = p._schedule_loop_callsites
def inspect(control, plan, *args, **kwargs):
    result = original(control, plan, *args, **kwargs)
    if plan is not None and 'root' in plan.name:
        print('PLAN', plan, flush=True)
        print('BEFORE', control, flush=True)
        print('AFTER', result[0], flush=True)
    return result
p._schedule_loop_callsites = inspect
source = Path('tools/compiler_probes/probe_authored_snapshot.py').read_text()
source = source.replace('    if rollback:\n        material.restore(saved)', '    if not rollback:\n        return material.state[0]\n    material.restore(saved)')
exec(compile(source, 'probe', 'exec'))
