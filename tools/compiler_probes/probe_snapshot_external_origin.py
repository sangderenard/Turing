import runpy
import traceback
from src.compiler import precompile_to_ssa as lowering
from pathlib import Path

original_lower = lowering.lower_control_sections_to_ssa
def inspect_control(control, **kwargs):
    if kwargs.get('control_name', '').endswith('__root'):
        Path('build/snapshot-control.txt').write_text(repr(control), encoding='utf-8')
    return original_lower(control, **kwargs)
lowering.lower_control_sections_to_ssa = inspect_control

for candidate in vars(lowering).values():
    if isinstance(candidate, type) and 'external_value' in vars(candidate):
        original = candidate.external_value
        def capture(self, value_id, *, dtype=None):
            if int(value_id) == 5 and value_id not in self.external_values:
                print('NEW EXTERNAL 5', flush=True)
                traceback.print_stack(limit=7)
            return original(self, value_id, dtype=dtype)
        candidate.external_value = capture
        break

runpy.run_module('build.probe_authored_snapshot', run_name='__main__')
