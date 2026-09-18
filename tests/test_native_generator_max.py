"""One native artifact exercises ordered starred-generator reductions."""

import pickle
import subprocess
import sys

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def test_native_starred_generator_max_preserves_order_and_empty_filter(tmp_path):
    source = '''
def root(initial, values, suffix, cutoff):
    complete = max(initial, *(value for value in values), suffix)
    filtered = max(initial, *(value for value in values if value > cutoff), suffix)
    return complete, filtered
'''
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({'bindings': [], 'values': [
        {
            'function': 'root', 'parameter': name, 'storage': 'scalar',
            'dtype': 'float64', 'rank': 0,
            'python_type': 'builtins.float',
        }
        for name in ('initial', 'suffix', 'cutoff')
    ] + [{
        'function': 'root', 'parameter': 'values', 'storage': 'span',
        'dtype': 'float64', 'rank': 1, 'shape': [4],
        'python_type': 'builtins.list',
    }]})
    module, outputs, _ = lower_ast_source_to_ssa(
        source,
        'root',
        name='generator_max',
        extraction_contract=policy,
    )
    root = module.functions['generator_max__root']
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')

    payload = tmp_path / 'artifact.pkl'
    payload.write_bytes(pickle.dumps((
        artifact,
        dict(root.metadata['parameter_names']),
        tuple(value.id for value in outputs[root.name]),
        source,
    )))
    probe = subprocess.run(
        [sys.executable, '-c', '''
import math, pickle, sys
import numpy as np
artifact, parameters, outputs, source = pickle.load(open(sys.argv[1], 'rb'))
namespace = {}
exec(source, namespace)
feeds = {
    parameters['initial']: np.array([0.0]),
    parameters['values']: np.zeros(4),
    parameters['suffix']: np.array([0.0]),
    parameters['cutoff']: np.array([0.0]),
}
execution = artifact.prepare_execution(feeds)
cases = (
    (-0.0, [0.0, -0.0, 0.0, -0.0], -0.0, 1.0),
    (1.0, [4.0, 2.0, 3.0, -5.0], 6.0, 10.0),
    (float('nan'), [8.0, 9.0, 10.0, 11.0], 12.0, 20.0),
    (2.0, [float('nan'), 7.0, 6.0, 5.0], 4.0, 10.0),
)
for initial, values, suffix, cutoff in cases:
    execution.buffers[parameters['initial']][0] = initial
    execution.buffers[parameters['values']][:] = values
    execution.buffers[parameters['suffix']][0] = suffix
    execution.buffers[parameters['cutoff']][0] = cutoff
    execution.run()
    actual = tuple(execution.buffers[value].item() for value in outputs)
    expected = namespace['root'](initial, values, suffix, cutoff)
    for got, want in zip(actual, expected):
        if math.isnan(want):
            assert math.isnan(got), (initial, values, suffix, cutoff, actual, expected)
        else:
            assert got == want, (initial, values, suffix, cutoff, actual, expected)
            if got == 0.0:
                assert math.copysign(1.0, got) == math.copysign(1.0, want)
''', str(payload)],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
