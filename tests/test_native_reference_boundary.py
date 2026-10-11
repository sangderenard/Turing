"""Public source compiler regressions for reference-valued keyed storage."""
from pathlib import Path

import numpy as np
import pytest

from src.compiler.authored_parameter_abi import (
    _authored_sequence_annotation_contract, _authored_parameter_sequence_contracts,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution,
)

ROOT = Path(__file__).resolve().parents[1]


def boundary_policy(annotated=False):
    values = [
        {'function': 'root', 'parameter': 'key', 'storage': 'scalar',
         'dtype': 'int64', 'python_type': 'builtins.int'},
        {'function': 'root', 'parameter': 'value', 'storage': 'reference',
         'python_type': 'builtins.object'},
    ]
    if not annotated:
        values.append({'function': 'root', 'parameter': 'memo', 'storage': 'keyed',
            'dtype': 'opaque_ref', 'mutable': True, 'key_encoding': 'integer_identity',
            'python_type': 'builtins.dict'})
    return ExtractionContract(ROOT / 'extraction_contracts/program_extraction.yaml').with_program_abi(
        {'records': {}, 'bindings': [], 'values': values}
    ).with_execution_file(ROOT / 'extraction_contracts/vehicle_full_native_execution.yaml')


@pytest.mark.parametrize('annotated', [False, True])
def test_memo_rows_keep_handle_layout_and_execute_native_store_lookup(annotated, tmp_path):
    annotation = ': dict[int, object]' if annotated else ''
    module, _, exports = lower_ast_source_to_ssa(
        f'def root(memo{annotation}, key, value):\n'
        '    memo[key] = value\n'
        '    return memo[key]\n',
        'root', name='reference_boundary', extraction_contract=boundary_policy(annotated),
        runtime_closure_only=True, progress=lambda _: None, progress_bars=False,
    )
    function = module.functions[exports[0]]
    descriptors = tuple(module.sequence_tables[function.name].sequences.values())
    assert len(descriptors) == 1, 'the declared mapping must not become a flat indexed pointer'
    descriptor = descriptors[0]
    assert descriptor.column_dtypes == ('int64', 'opaque_ref')
    assert descriptor.key_columns == (0,)
    assert all(value_id in {value.id for value in function.args}
               for value_id in (*descriptor.column_value_ids, descriptor.length_address_id,
                                descriptor.capacity_value_id, descriptor.status_address_id))
    calls = [i.attributes.get('callee', '') for b in function.blocks.values() for i in b.instrs if i.op == 'Call']
    assert any(str(name).endswith('_store') for name in calls)
    assert any(str(name).endswith('_lookup') for name in calls)
    artifact = emit_ssa_function_to_llvm(module, function.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'memo')
    parameters = dict(function.metadata['parameter_names'])
    output = next(i.args[0].id for b in function.blocks.values() for i in b.instrs if i.op == 'Ret')
    # Allocation is deliberately supplied by the caller; storage semantics
    # and native execution must not depend on a chosen arena placement.
    capacity = 4
    keys, handles = np.zeros(capacity, dtype=np.int64), np.zeros(capacity, dtype=np.int64)
    length, status = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int32)
    buffers = {
        descriptor.column_value_ids[0]: keys, descriptor.column_value_ids[1]: handles,
        descriptor.length_address_id: length,
        descriptor.capacity_value_id: np.array([capacity], dtype=np.int64),
        descriptor.status_address_id: status,
    }
    if descriptor.live_flags_value_id is not None:
        buffers[descriptor.live_flags_value_id] = np.zeros(capacity, dtype=np.bool_)
    for key, handle in ((2**60 + 3, 2**60 + 17), (23, 2**60 + 29), (2**60 + 3, 31)):
        buffers[parameters['key']] = np.array([key], dtype=np.int64)
        buffers[parameters['value']] = np.array([handle], dtype=np.int64)
        execution = prepare_artifact_execution(native, buffers).run()
        assert execution.buffers[output].item() == handle
    assert length[0] == 2
    assert list(keys[:2]) == [2**60 + 3, 23]
    assert list(handles[:2]) == [31, 2**60 + 29]


def test_object_annotation_proves_handle_slots_not_object_payload():
    assert _authored_sequence_annotation_contract('dict[int, object]') == (
        'unique', 2, True, ('int64', 'opaque_ref'))
    assert _authored_sequence_annotation_contract('list[object]') == (
        'duplicates', 1, True, ('opaque_ref',))


def test_conflicting_boundary_row_contract_is_not_silently_overridden():
    import networkx as nx
    graph = nx.DiGraph()
    graph.graph['function_parameter_annotations'] = {'memo': 'dict[int, float]'}
    graph.graph['parameter_value_abi'] = {'memo': {
        'storage': 'keyed', 'dtype': 'opaque_ref', 'mutable': True,
    }}
    with pytest.raises(ValueError, match='disagree about sequence row layout'):
        _authored_parameter_sequence_contracts(graph)


def test_existing_keyed_record_view_exposes_runtime_kind_natively(tmp_path):
    # This uses the incumbent keyed value_record ABI, not a second object
    # registry. Caller-owned descriptor columns provide the runtime facts;
    # this does NOT claim an opaque Python handle supplies those columns.
    kinds = ('builtins.dict', 'builtins.list', 'builtins.tuple',
             'builtins.set', 'builtins.frozenset', 'leaf')
    policy = ExtractionContract(ROOT / 'extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {
            'Book': {'identity': 'metadata.Book', 'fields': {'objects': {
                'storage': 'keyed', 'dtype': 'int64', 'key_encoding': 'integer_identity',
                'value_record': 'Object',
            }}},
            'Object': {'identity': 'metadata.Object', 'fields': {'kind': {
                'storage': 'scalar', 'dtype': 'int64', 'token_vocabulary': kinds,
            }}},
        },
        'bindings': [{'function': 'root', 'parameter': 'book', 'record': 'Book'}],
        'values': [{'function': 'root', 'parameter': 'handle', 'storage': 'reference',
                    'python_type': 'builtins.object'}],
    }).with_execution_file(ROOT / 'extraction_contracts/vehicle_full_native_execution.yaml')
    module, _, exports = lower_ast_source_to_ssa(
        'def root(book, handle):\n    item = book.objects[handle]\n    return item.kind\n',
        'root', name='reference_kind', extraction_contract=policy,
        runtime_closure_only=True, progress=lambda _: None, progress_bars=False,
    )
    function = module.functions[exports[0]]
    artifact = emit_ssa_function_to_llvm(module, function.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'kind')
    facts = {
        'objects.length': np.array([3], dtype=np.int64),
        'objects.keys': np.array([2**60 + 3, 17, 29], dtype=np.int64),
        'objects.values': np.array([0, 1, 2], dtype=np.int64),
        'objects[].kind.column': np.array([1, 2, 6], dtype=np.int64),
    }
    buffers = {v.id: facts[v.accounting['program_abi_field']]
               for v in function.args if v.accounting.get('program_abi_field') in facts}
    assert len(buffers) == len(facts)
    parameter = dict(function.metadata['parameter_names'])['handle']
    output = next(i.args[0].id for b in function.blocks.values() for i in b.instrs if i.op == 'Ret')
    for handle, kind in ((2**60 + 3, 1), (17, 2), (29, 6)):
        buffers[parameter] = np.array([handle], dtype=np.int64)
        execution = prepare_artifact_execution(native, buffers).run()
        assert execution.buffers[output].item() == kind


def test_sequence_call_binding_keeps_owner_and_refuses_storage_kind_change():
    from src.compiler.sequence_storage_binding import _bind_sequence_storage_members
    from src.transmogrifier.ssa import SSASequenceDescriptor
    callee = SSASequenceDescriptor(1, (2,), 3, 4,
        column_dtypes=('opaque_ref',), heap_owner_value_id=5)
    caller = SSASequenceDescriptor(11, (12,), 13, 14,
        column_dtypes=('opaque_ref',), heap_owner_value_id=15)
    bindings = {}
    assert _bind_sequence_storage_members(bindings, callee, caller)
    assert bindings == {1: 11, 2: 12, 3: 13, 4: 14, 5: 15}
    direct = SSASequenceDescriptor(11, (12,), 13, 14, column_dtypes=('opaque_ref',))
    unchanged = {99: 100}
    assert not _bind_sequence_storage_members(unchanged, callee, direct)
    assert unchanged == {99: 100}
