"""Sequence observations see exactly their authored prefix of effects."""

import pickle
import subprocess
import sys
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize('snapshot', [False, True])
def test_sequence_truth_between_conditional_appends_and_clear(tmp_path, snapshot):
    source = '''
def root(first, second, clear):
    reasons = []
    if first:
        reasons.append(0.0)
    if second:
        reasons.append(1.0)
    before = bool(reasons)
    if clear:
        reasons.clear()
    return before, bool(reasons)
'''
    if snapshot:
        source = source.replace('before = bool(reasons)', 'saved = tuple(reasons)\n    before = len(saved)')
        source = source.replace('return before, bool(reasons)',
                                'reasons.append(3.0)\n    return before, len(saved), len(reasons)')
    policy = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        'extraction_contracts/vehicle_full_native_execution.yaml'
    ).with_program_abi({'bindings': [], 'values': [
        {'function': 'root', 'parameter': name, 'storage': 'scalar',
         'dtype': 'bool', 'rank': 0, 'python_type': 'builtins.bool'}
        for name in ('first', 'second', 'clear')
    ]})
    module, outputs, _ = lower_ast_source_to_ssa(
        source, 'root', name='sequence_effect_order', extraction_contract=policy,
    )
    root = module.functions['sequence_effect_order__root']
    names = dict(root.metadata['parameter_names'])
    snapshot_column = None
    if snapshot:
        first_output = outputs[root.name][0].id
        length_source = next(i.args[0].id for b in root.blocks.values() for i in b.instrs
                             if i.op == 'Load' and i.attributes.get('source_call_node_id') == first_output)
        descriptor = next(d for d in module.sequence_tables[root.name].sequences.values()
                          if d.length_address_id == length_source)
        snapshot_column = descriptor.column_value_ids[0]
        assert descriptor.column_dtypes == ('float64',)
    # The backend's watch surface exposes private state for diagnostics.
    artifact = emit_ssa_module_to_c(module, root.name, watch=(() if snapshot_column is None else (snapshot_column,)))
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native')
    saved = tmp_path / 'artifact.pkl'
    saved.write_bytes(pickle.dumps((artifact, names, tuple(v.id for v in outputs[root.name]), source, snapshot_column)))
    probe = subprocess.run([sys.executable, '-c', '''
import itertools, pickle, sys
import numpy as np
artifact, names, outputs, source, snapshot_column = pickle.load(open(sys.argv[1], 'rb'))
namespace = {}
exec(source, namespace)
execution = artifact.prepare_execution({v: np.array([False], dtype=np.bool_) for v in names.values()})
for args in itertools.product((False, True), repeat=3):
    for name, value in zip(('first', 'second', 'clear'), args):
        execution.buffers[names[name]][0] = value
    execution.run()
    actual = tuple(execution.buffers[v].item() for v in outputs)
    expected = namespace['root'](*args)
    assert actual == expected, (args, actual, expected)
    if snapshot_column is not None:
        contents = ([0.0] if args[0] else []) + ([1.0] if args[1] else [])
        assert execution.buffers[snapshot_column].reshape(-1)[:len(contents)].tolist() == contents
''', str(saved)], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stdout + probe.stderr
