"""Keyed membership at the public source boundary, including captured memos."""
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.compiler.authored_parameter_abi import _authored_parameter_sequence_contracts
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.ssa_llvm_backend import (
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)

ROOT = Path(__file__).resolve().parents[1]


def policy(identity=False):
    return ExtractionContract(ROOT / 'extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {}, 'bindings': [], 'values': [
            {'function': 'root', 'parameter': 'memo', 'storage': 'keyed',
             'dtype': 'opaque_ref', 'mutable': True, 'key_encoding': 'integer_identity',
             'python_type': 'builtins.dict'},
            {'function': 'root', 'parameter': 'value' if identity else 'key',
             **({'storage': 'reference', 'python_type': 'builtins.object'} if identity else
                {'storage': 'scalar', 'dtype': 'int64', 'python_type': 'builtins.int'})},
        ],
    }).with_execution_file(ROOT / 'extraction_contracts/vehicle_full_native_execution.yaml')


@pytest.mark.parametrize('mode', ['direct', 'negated', 'conditional', 'captured', 'identity'])
def test_native_membership_keeps_exact_parameter_arenas(tmp_path, mode):
    if mode == 'captured':
        source = ('def root(memo,key):\n'
                  '    def visit(value):\n        return value in memo\n'
                  '    return visit(key)\n')
    elif mode == 'identity':
        source = 'def root(memo,value):\n    identity=id(value)\n    return identity in memo\n'
    elif mode == 'conditional':
        source = ('def root(memo,key):\n    if key in memo:\n        result=1.0\n'
                  '    else:\n        result=0.0\n    return result\n')
    else:
        source = f'def root(memo,key):\n    return key {"not in" if mode == "negated" else "in"} memo\n'
    module, _, exports = lower_ast_source_to_ssa(source, 'root', name='memo_membership',
        extraction_contract=policy(mode == 'identity'), runtime_closure_only=True,
        progress=lambda _: None, progress_bars=False)
    root = module.functions[exports[0]]
    assert concordance_report(module).splitlines()[0].endswith('0 finding(s)')
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / mode)
    descriptor = next(iter(module.sequence_tables[root.name].sequences.values()))
    assert descriptor.column_dtypes == ('int64', 'opaque_ref')
    assert set(descriptor.column_value_ids) <= {v.id for v in root.args}
    output = next(i.args[0].id for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
    parameters = dict(root.metadata['parameter_names'])
    keys = np.array([0, -7, 2**60 + 3, 41], dtype=np.int64)
    live = np.array([True, True, True, False], dtype=np.bool_)
    buffers = {
        descriptor.column_value_ids[0]: keys,
        descriptor.column_value_ids[1]: np.zeros(4, dtype=np.int64),
        descriptor.length_address_id: np.array([4], dtype=np.int64),
        descriptor.capacity_value_id: np.array([4], dtype=np.int64),
        descriptor.status_address_id: np.zeros(1, dtype=np.int32),
        descriptor.live_flags_value_id: live,
    }
    for count in (0, 4):
        buffers[descriptor.length_address_id][0] = count
        for query in (0, -7, 2**60 + 3, 15, 41):
            buffers[parameters['value' if mode == 'identity' else 'key']] = np.array([query], dtype=np.int64)
            execution = prepare_artifact_execution(native, buffers).run()
            expected = count != 0 and query in (0, -7, 2**60 + 3)
            assert execution.buffers[output].item() == (not expected if mode == 'negated' else expected)


def test_linked_capture_contract_uses_exact_input_identity():
    graph = nx.DiGraph()
    graph.add_node(7, type='Input', attributes={'binding_name': 'memo', 'binding_kind': 'external'})
    graph.add_node(8, type='Call', attributes={'binding_name': 'unrelated'})
    graph.graph['linked_value_abi'] = {
        7: {'storage': 'keyed', 'dtype': 'opaque_ref', 'mutable': True},
        8: {'storage': 'keyed', 'dtype': 'float64', 'mutable': False},
    }
    assert _authored_parameter_sequence_contracts(graph) == {
        'memo': ('unique', 2, True, ('int64', 'opaque_ref')),
    }


def test_linked_capture_cannot_override_declared_row_layout():
    graph = nx.DiGraph()
    graph.add_node(7, type='Input', attributes={'binding_name': 'memo'})
    graph.graph['parameter_value_abi'] = {'memo': {'storage': 'keyed', 'dtype': 'opaque_ref'}}
    graph.graph['linked_value_abi'] = {7: {'storage': 'keyed', 'dtype': 'float64'}}
    with pytest.raises(ValueError, match='declared and linked keyed contracts disagree'):
        _authored_parameter_sequence_contracts(graph)


@pytest.mark.parametrize('corruption', ['storage', 'query', 'target'])
def test_membership_receipt_detects_stale_parameter_binding(corruption):
    module, _, exports = lower_ast_source_to_ssa('def root(memo,key):\n    return key in memo\n',
        'root', extraction_contract=policy(), runtime_closure_only=True,
        progress=lambda _: None, progress_bars=False)
    root = module.functions[exports[0]]
    call = next(i for b in root.blocks.values() for i in b.instrs
                if i.attributes.get('structural_membership_parameter'))
    if corruption == 'storage':
        call.args[0] = call.args[1]
    elif corruption == 'query':
        call.args[-1] = call.args[0]
    else:
        call.attributes['structural_membership_parameter']['source_target_id'] += 1
    assert '[membership-parameter-contract]' in concordance_report(module)
