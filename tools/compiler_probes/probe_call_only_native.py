import runpy
import tempfile
from pathlib import Path
import numpy as np
from src.compiler import fortran_c_shell as shell
from src.compiler.ssa_c_backend import emit_ssa_module_to_c

original = shell.lower_ast_source_to_ssa
captured = []
def capture(*args, **kwargs):
    result = original(*args, **kwargs)
    captured.append(result)
    return result
shell.lower_ast_source_to_ssa = capture
test = runpy.run_path('tests/test_native_call_only_conditional.py')
test['test_record_method_call_retains_runtime_conditional']()
module, outputs, exports = captured[0]
entry = 'call_only__root'
function = module.functions[entry]
enabled_id = dict(function.metadata['parameter_names'])['enabled']
state_id = next(a.id for a in function.args if a.accounting.get('program_abi_field') == 'state')
artifact = emit_ssa_module_to_c(module, entry)
assert artifact.complete, artifact.shortfalls
artifact.compile(Path(tempfile.mkdtemp(prefix='conditional_method_')) / 'conditional_method')
for enabled in (False, True):
    feeds = {state_id: np.array([2.0, 3.0]), enabled_id: np.array([enabled], dtype=np.bool_)}
    result = artifact.prepare_execution(feeds).run()
    state = np.asarray(result.buffers[state_id]).reshape(-1)
    expected = 7.0 if enabled else 2.0
    assert state.tolist() == [expected, 3.0], (enabled, state)
    print('NATIVE', enabled, state.tolist(), flush=True)
