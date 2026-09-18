import pickle
import subprocess
import sys
import pytest

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize('scale', [False, True])
def test_conditional_tensor_result_preserves_selected_span(tmp_path, scale):
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({'bindings': [], 'values': [
        *({'function': 'root', 'parameter': name, 'storage': 'span',
           'dtype': 'float64', 'rank': 2, 'shape': [2, 3],
           'python_type': 'src.common.tensors.abstraction.AbstractTensor'}
          for name in ('left', 'right')),
        {'function': 'root', 'parameter': 'enabled', 'storage': 'scalar',
         'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'},
    ]})
    module, outputs, _ = lower_ast_source_to_ssa(
        'def root(left, right, enabled):\n    selected = left if enabled else right\n'
        + ('    return selected * 2.0\n' if scale else '    return selected\n'),
        'root', name='conditional_span', extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    root = module.functions['conditional_span__root']
    result = outputs[root.name][0]
    names = dict(root.metadata['parameter_names'])
    assert len(root.args) == 3
    assert tuple(result.shape) == (2, 3)
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'conditional_span')
    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((artifact, names, result.id, scale)))
    probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
with open(sys.argv[1], 'rb') as stream:
    artifact, names, result_id, scale = pickle.load(stream)
left = np.array([[2.0, -3.0, 4.0], [11.0, 17.0, -23.0]])
right = np.array([[-7.0, 9.0, 12.0], [31.0, -41.0, 53.0]])
for enabled in (False, True):
    feeds = {names['left']: left.copy(), names['right']: right.copy(),
             names['enabled']: np.array([enabled], dtype=np.bool_)}
    result = artifact.prepare_execution(feeds).run()
    actual = np.asarray(result.buffers[result_id])
    expected = left if enabled else right
    if scale:
        expected = expected * 2.0
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.array_equal(actual, expected), (enabled, actual, expected)
''', str(payload)], capture_output=True, text=True, timeout=20)
    assert probe.returncode == 0, probe.stdout + probe.stderr
