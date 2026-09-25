import runpy
from pathlib import Path
from src.compiler import ssa_c_backend as backend
from src.compiler import precompile_to_ssa as lowering
original_lower = lowering.lower_control_sections_to_ssa
def show_control(control, **kwargs):
    if kwargs.get('control_name', '').endswith('__root'):
        print('CONTROL', repr(control), flush=True)
    return original_lower(control, **kwargs)
lowering.lower_control_sections_to_ssa = show_control

original = backend.emit_ssa_module_to_c
def capture(module, entry, **kwargs):
    for name, block in module.functions[entry].blocks.items():
        print(name, repr(block), flush=True)
    result = original(module, entry, **kwargs)
    Path('build/conditional-result.c').write_text(result.source, encoding='utf-8')
    return result
backend.emit_ssa_module_to_c = capture
runpy.run_path('tests/test_native_conditional_call_result.py')['test_conditional_call_result_has_producer'](Path('build'))
