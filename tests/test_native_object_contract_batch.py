"""Shared storage, type evidence, and payload views use one admitted context."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.test_source_object_context import context
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.monotonic_ids import GLOBAL_MONOTONIC_IDS
from src.compiler.shell_external_references import ExternalReferenceValueCodec
from src.compiler.ssa_object_context import lower_object_context
from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution


def typed_context():
    scope = context()
    objects = replace(scope.objects,
        column_value_ids=(*scope.objects.column_value_ids, GLOBAL_MONOTONIC_IDS.mint()),
        column_dtypes=('opaque_ref', 'int64', 'int64'), column_shapes=((), (), ()))
    return replace(scope, objects=objects,
        type_vocabulary=(*scope.kind_vocabulary, 'builtins.bool', 'builtins.int',
                         'builtins.float', 'builtins.str', 'builtins.bytes', 'builtins.NoneType'))


def policy(memo=False):
    repository = Path(__file__).resolve().parents[1]
    values = [{'function': 'root', 'parameter': 'value', 'storage': 'reference', 'python_type': 'builtins.object'}]
    if memo:
        values.append({'function': 'root', 'parameter': 'memo', 'storage': 'keyed',
            'dtype': 'opaque_ref', 'mutable': True, 'key_encoding': 'integer_identity', 'python_type': 'builtins.dict'})
    return ExtractionContract(repository / 'extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {}, 'bindings': [], 'values': values,
    }).with_execution_file(repository / 'extraction_contracts/vehicle_full_native_execution.yaml')


def compile_source(source, scope, tmp_path, *, memo=False):
    module, _, exports = lower_ast_source_to_ssa(source, 'root', extraction_contract=policy(memo),
        native_object_context=scope, progress=lambda _: None, progress_bars=False)
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    function = module.functions[exports[0]]
    artifact = emit_ssa_function_to_llvm(module, function.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'source')
    release_module = lower_object_context(scope, prefix='release_contract_batch')
    release = compile_artifact(emit_ssa_function_to_llvm(release_module, 'release_contract_batch_release'), directory=tmp_path / 'release')
    output = next(i.args[0] for b in function.blocks.values() for i in b.instrs if i.op == 'Ret' and i.args)
    return module, function, native, release, output


class TupleLeaf(tuple):
    pass


def test_type_tuple_admission_requires_the_existing_static_type_receipt():
    import networkx as nx
    from src.compiler.ssa_object_context import source_object_type_tests
    graph = nx.DiGraph()
    graph.add_node(1, type='Input', value_id=1)
    attributes = {'extraction_identity': 'builtins.isinstance',
                  'static_call_values': {'arg:1': (int, tuple)}}
    graph.add_node(2, type='Call', value_id=2, parents=((1, 'arg:0'),), attributes=attributes)
    tests = source_object_type_tests(graph, {1}, typed_context())
    assert tests[2]['accepted_types'] == ('builtins.int', 'builtins.tuple')
    # Whole-source tuple targets currently lose this static receipt upstream.
    # An empty aggregate or absent type receipt is NOT an accepted type union.
    attributes['static_call_values']['arg:1'] = ()
    assert source_object_type_tests(graph, {1}, typed_context()) == {}
    attributes.pop('static_call_values')
    assert source_object_type_tests(graph, {1}, typed_context()) == {}


@pytest.mark.parametrize('target', ['dict', 'list', 'tuple', 'set', 'frozenset', 'int', 'bool', 'object'])
def test_native_builtin_guards_read_type_evidence_not_storage_kind(tmp_path, target):
    scope = typed_context()
    source = (f'def classify(item):\n    if isinstance(item,{target}):\n'
              '        result=1.0\n    else:\n        result=0.0\n    return result\n'
              'def root(value):\n    return classify(value)\n')
    module, function, native, release, output = compile_source(source, scope, tmp_path)
    assert any(f.metadata.get('native_object_type_tests') for f in module.functions.values())
    assert not function.metadata['native_object_requires_member_storage']
    accepted = eval(target, {'int': int, 'bool': bool, 'tuple': tuple, 'dict': dict,
                             'list': list, 'set': set, 'frozenset': frozenset})
    for original in ({}, [], (), set(), frozenset(), 42, True, 1.25, 'text', b'bytes', None, object(), TupleLeaf((1,))):
        boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (original,))
        try:
            parameter = dict(function.metadata['parameter_names'])['value']
            invocation = boundary.bind_execution(native, function, {parameter: np.array([boundary.roots[0]], dtype=np.int64)})
            invocation.run()
            assert invocation.native_invocations == 1
            assert invocation.buffers[output.id].item() == float(isinstance(original, accepted))
            result, = boundary.export()
            assert result is original
        finally:
            prepare_artifact_execution(release, boundary.buffers).run()
            boundary.close()


def test_new_native_birth_publishes_the_same_type_evidence(tmp_path):
    scope = typed_context()
    source = ('def classify(item):\n    if isinstance(item,list):\n'
              '        result=1.0\n    else:\n        result=0.0\n    return result\n'
              'def root(value):\n    created=[]\n    return classify(created)\n')
    _, function, native, release, output = compile_source(source, scope, tmp_path)
    boundary = ExternalReferenceValueCodec().bind_native_objects(scope, ())
    try:
        feeds = {v.id: np.zeros(1, dtype=np.int64) for v in function.args if v.id not in boundary.buffers}
        invocation = boundary.bind_execution(native, function, feeds)
        invocation.run()
        assert invocation.buffers[output.id].item() == 1.0
        rows = boundary._rows(scope.objects)
        assert len(rows) == 1
        assert rows[0][2] == ((1 << scope.type_vocabulary.index('builtins.list')) |
                              (1 << scope.type_vocabulary.index('builtins.object')))
    finally:
        prepare_artifact_execution(release, boundary.buffers).run()
        boundary.close()


def test_payload_requirement_follows_native_call_graph_and_refuses_opaque_subclass(tmp_path):
    scope = typed_context()
    source = 'def copy(item):\n    return list(item)\ndef root(value):\n    return copy(value)\n'
    _, function, native, release, _ = compile_source(source, scope, tmp_path)
    assert function.metadata['native_object_requires_member_storage']
    boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (TupleLeaf((1,)),))
    try:
        parameter = dict(function.metadata['parameter_names'])['value']
        with pytest.raises(ValueError, match='type evidence is not payload storage'):
            boundary.bind_execution(native, function, {parameter: np.array([boundary.roots[0]], dtype=np.int64)})
    finally:
        prepare_artifact_execution(release, boundary.buffers).run()
        boundary.close()


@pytest.mark.parametrize('guard', ['isinstance(item,tuple)', 'len(item)>0'])
def test_recursive_mutation_capture_binds_all_incumbent_memo_columns(tmp_path, guard):
    scope = typed_context()
    source = (f'def root(value,memo):\n    def visit(item):\n        if {guard}:\n'
              '            result=[]\n            memo[id(item)]=result\n'
              '            return tuple(visit(child) for child in item)\n'
              '        return item\n    return visit(value)\n')
    module, function, native, release, output = compile_source(source, scope, tmp_path, memo=True)
    descriptor = next(d for d in module.sequence_tables[function.name].sequences.values()
                      if d.column_dtypes == ('int64', 'opaque_ref'))
    assert {*descriptor.column_value_ids, descriptor.length_address_id,
            descriptor.capacity_value_id, descriptor.status_address_id,
            descriptor.live_flags_value_id} <= set(native.buffer_order)
    for depth in (0, 1, 7, 32):
        original = ()
        for _ in range(depth):
            original = (original,)
        boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (original,))
        feeds = {descriptor.column_value_ids[0]: np.zeros(64, dtype=np.int64),
                 descriptor.column_value_ids[1]: np.zeros(64, dtype=np.int64),
                 descriptor.length_address_id: np.zeros(1, dtype=np.int64),
                 descriptor.capacity_value_id: np.array([64], dtype=np.int64),
                 descriptor.status_address_id: np.zeros(1, dtype=np.int32),
                 descriptor.live_flags_value_id: np.zeros(64, dtype=np.bool_)}
        parameter = dict(function.metadata['parameter_names'])['value']
        feeds[parameter] = np.array([boundary.roots[0]], dtype=np.int64)
        expected = {handle for handle in boundary.retained
                    if guard.startswith('isinstance') or boundary.codec.object(handle)}
        try:
            invocation = boundary.bind_execution(native, function, feeds)
            invocation.run()
            assert invocation.native_invocations == 1
            result, = boundary.export((int(invocation.buffers[output.id].item()),))
            for _ in range(depth):
                assert type(result) is tuple and len(result) == 1
                result, = result
            assert result == ()
            count = depth + int(guard.startswith('isinstance'))
            assert feeds[descriptor.length_address_id][0] == count
            assert set(feeds[descriptor.column_value_ids[0]][:count]) == expected
            assert all(type(item) is list and item == [] for item in
                       boundary.export(feeds[descriptor.column_value_ids[1]][:count]))
        finally:
            prepare_artifact_execution(release, boundary.buffers).run()
            boundary.close()


@pytest.mark.parametrize('fault', ['mask', 'reference', 'column', 'requirements'])
def test_type_and_payload_receipts_refuse_stale_contracts(tmp_path, fault):
    scope = typed_context()
    source = ('def root(value):\n    if isinstance(value,tuple):\n'
              '        result=1.0\n    else:\n        result=0.0\n    return result\n')
    module, function, _, _, _ = compile_source(source, scope, tmp_path)
    test = next(i for b in function.blocks.values() for i in b.instrs
                if i.attributes.get('source_object_type_test'))
    if fault == 'mask':
        test.attributes['source_object_type_test']['mask'] ^= 1
    elif fault == 'reference':
        test.attributes['source_object_type_test']['reference_value_id'] = -1
    elif fault == 'column':
        function.metadata['native_object_context'] = {**function.metadata['native_object_context'],
                                                     'type_evidence_column': 1}
    else:
        function.metadata['native_object_requires_member_storage'] = True
    assert ('payload requirements disagree' if fault == 'requirements'
            else 'source-object-contract') in concordance_report(module)
    with pytest.raises(ValueError):
        emit_ssa_function_to_llvm(module, function.name)
