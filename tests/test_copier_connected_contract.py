"""Connected copier operations, through the public compiler and native ABI."""
import numpy as np
import pytest

from tests.test_native_object_contract_batch import typed_context, compile_source
from src.compiler.shell_external_references import ExternalReferenceValueCodec
from src.compiler.ssa_llvm_backend import prepare_artifact_execution
from tools.compiler_probes.probe_native_metadata_copy import authored_program


@pytest.mark.parametrize('source,target', [
    ('def root(value):\n    result=[]\n    result.extend(item for item in value)\n    return result\n', list),
    ('def root(value):\n    result=set()\n    result.update(item for item in value)\n    return result\n', set),
    ('def root(value):\n    return frozenset(value)\n', frozenset),
])
def test_native_connected_container_operations(tmp_path, source, target):
    scope = typed_context()
    _, function, native, release, output = compile_source(source, scope, tmp_path)
    for count in (0, 1, 3, 64):
        leaves = [object() for _ in range(count)]
        original = tuple(leaves + leaves)
        boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (original,))
        try:
            parameter = dict(function.metadata['parameter_names'])['value']
            invocation = boundary.bind_execution(native, function,
                {parameter: np.array([boundary.roots[0]], dtype=np.int64)}).run()
            assert invocation.native_invocations == 1
            result, = boundary.export((int(invocation.buffers[output.id].item()),))
            assert type(result) is target
            assert result == target(original)
            if target is list:
                assert all(a is b for a, b in zip(result, original, strict=True))
        finally:
            prepare_artifact_execution(release, boundary.buffers).run()
            boundary.close()


def test_recursive_dictionary_calls_use_recorded_result_edges(tmp_path):
    from tools.compiler_probes.observe_copier_contract_batch import SOURCES
    scope = typed_context()
    _, function, native, release, output = compile_source(SOURCES['recursive_dict'], scope, tmp_path)
    leaf = object()
    original = {'first': {'second': {'third': leaf}}, 'leaf': leaf}
    boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (original,))
    try:
        parameter = dict(function.metadata['parameter_names'])['value']
        invocation = boundary.bind_execution(native, function,
            {parameter: np.array([boundary.roots[0]], dtype=np.int64)}).run()
        result, = boundary.export((int(invocation.buffers[output.id].item()),))
        assert invocation.native_invocations == 1
        assert result == original and result is not original
        assert result['first'] is not original['first']
        assert result['first']['second'] is not original['first']['second']
        assert result['first']['second']['third'] is leaf and result['leaf'] is leaf
    finally:
        prepare_artifact_execution(release, boundary.buffers).run()
        boundary.close()


def test_identity_reduction_is_scalar_and_uses_reference_identity(tmp_path):
    from tools.compiler_probes.observe_copier_contract_batch import SOURCES
    scope = typed_context()
    module, function, native, release, output = compile_source(SOURCES['identity_all'], scope, tmp_path)
    assert any(i.attributes.get('source_object_identity_reduction')
               for block in function.blocks.values() for i in block.instrs)
    assert all(d.column_dtypes != ('bool',) for d in module.sequence_tables[function.name].sequences.values())
    for count in (0, 1, 3, 64):
        original = tuple(object() for _ in range(count))
        boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (original,))
        try:
            parameter = dict(function.metadata['parameter_names'])['value']
            invocation = boundary.bind_execution(native, function,
                {parameter: np.array([boundary.roots[0]], dtype=np.int64)}).run()
            assert invocation.native_invocations == 1
            assert invocation.buffers[output.id].item() == 1
        finally:
            prepare_artifact_execution(release, boundary.buffers).run()
            boundary.close()


@pytest.fixture(scope='module')
def compiled_whole_copier(tmp_path_factory):
    from pathlib import Path
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_object_context import lower_object_context
    from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm
    scope = typed_context()
    repository = Path(__file__).resolve().parents[1]
    contract = ExtractionContract(repository / 'extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {}, 'bindings': [], 'values': [
            {'function': 'metadata_copy_program', 'parameter': 'value', 'storage': 'reference',
             'python_type': 'builtins.object'},
            {'function': 'metadata_copy_program', 'parameter': 'metadata_memo', 'storage': 'keyed',
             'dtype': 'opaque_ref', 'mutable': True, 'key_encoding': 'integer_identity', 'python_type': 'builtins.dict'}]
    }).with_execution_file(repository / 'extraction_contracts/vehicle_full_native_execution.yaml')
    module, _, exports = lower_ast_source_to_ssa(authored_program(), 'metadata_copy_program',
        extraction_contract=contract, native_object_context=scope, progress=lambda _: None,
        progress_bars=False, runtime_closure_only=True)
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    function = module.functions[exports[0]]
    import os
    if os.environ.get('TURING_DEBUG_COPIER'):
        for f in module.functions.values():
            if f.metadata.get('native_object_collections'):
                for block in f.blocks.values():
                    print('COPIER_BLOCK', block.name, flush=True)
                    for instruction in block.instrs:
                        print('COPIER_SSA', instruction.op, tuple((v.id,v.dtype,v.shape) for v in instruction.args),
                            None if instruction.res is None else instruction.res.id,
                            {key:value for key,value in instruction.attributes.items()
                             if key in {'callee','value','binding','true_target','false_target','target'}}, flush=True)
    artifact = emit_ssa_function_to_llvm(module, function.name)
    assert artifact.complete, artifact.shortfalls
    directory = tmp_path_factory.mktemp('whole_copier')
    native = compile_artifact(artifact, directory=directory / 'copier')
    release_module = lower_object_context(scope, prefix='whole_copier_release')
    release = compile_artifact(emit_ssa_function_to_llvm(release_module, 'whole_copier_release_release'),
                               directory=directory / 'release')
    descriptor = next(d for d in module.sequence_tables[function.name].sequences.values()
                      if d.column_dtypes == ('int64', 'opaque_ref'))
    output = next(i.args[0] for block in function.blocks.values() for i in block.instrs if i.op == 'Ret' and i.args)
    return scope, function, native, release, descriptor, output


@pytest.mark.parametrize('kind', ['leaf', 'dict', 'list', 'tuple', 'set', 'frozenset', 'cycle', 'mixed_cycle'])
def test_entire_authored_copier_executes_native(compiled_whole_copier, kind):
    scope, function, native, release, memo, output = compiled_whole_copier
    leaf = object()
    if kind == 'leaf':
        original = leaf
    elif kind == 'dict':
        shared = [leaf]
        original = {'left': shared, 'right': shared}
    elif kind == 'list':
        shared = {'leaf': leaf}
        original = [shared, shared]
    elif kind == 'tuple':
        original = (leaf, 7, None)
    elif kind == 'set':
        original = {leaf, 7}
    elif kind == 'frozenset':
        original = frozenset((leaf, 7))
    elif kind == 'cycle':
        original = []
        original.append(original)
    else:
        child = []
        original = (child, leaf)
        child.append(original)
    boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (original,))
    feeds = {memo.column_value_ids[0]: np.zeros(256, dtype=np.int64),
             memo.column_value_ids[1]: np.zeros(256, dtype=np.int64),
             memo.length_address_id: np.zeros(1, dtype=np.int64),
             memo.capacity_value_id: np.array([256], dtype=np.int64),
             memo.status_address_id: np.zeros(1, dtype=np.int32),
             memo.live_flags_value_id: np.zeros(256, dtype=np.bool_)}
    parameter = dict(function.metadata['parameter_names'])['value']
    feeds[parameter] = np.array([boundary.roots[0]], dtype=np.int64)
    try:
        invocation = boundary.bind_execution(native, function, feeds).run()
        assert invocation.native_invocations == 1
        result, = boundary.export((int(invocation.buffers[output.id].item()),))
        assert type(result) is type(original)
        if kind in {'leaf', 'tuple', 'frozenset'}:
            assert result is original
        elif kind == 'cycle':
            assert result is not original and result[0] is result
        elif kind == 'mixed_cycle':
            assert result is not original and result[0] is not original[0]
            assert result[0][0] is result and result[1] is leaf
        else:
            assert result == original and result is not original
            if kind == 'dict':
                assert result['left'] is result['right'] and result['left'] is not original['left']
            elif kind == 'list':
                assert result[0] is result[1] and result[0] is not original[0]
        size = int(feeds[memo.length_address_id][0])
        assert 0 <= size <= 256
        if kind != 'leaf':
            assert boundary.roots[0] in feeds[memo.column_value_ids[0]][:size]
    finally:
        prepare_artifact_execution(release, boundary.buffers).run()
        boundary.close()


def test_installed_copier_loads_without_compilation_and_batches_real_extraction(tmp_path, monkeypatch):
    from src.compiler import native_compiler_accelerators as accelerators
    from src.compiler.process_graph_fusion import extract_clean_process_subgraph
    from src.transmogrifier.graph.graph_express2 import ProcessGraph
    from tests.test_scheduled_process_graph_dispatches import _add
    registry = accelerators.compiler_accelerators
    previous = registry._native.get(accelerators.METADATA_COPY)
    try:
        provider = accelerators.compile_and_register_metadata_copy_accelerator(tmp_path)
        registry.unregister_native(accelerators.METADATA_COPY)
        def forbidden(*args, **kwargs):
            raise AssertionError('installed code must load without lowering/emission/build')
        monkeypatch.setattr('src.compiler.fortran_c_shell.lower_ast_source_to_ssa', forbidden)
        monkeypatch.setattr('src.compiler.ssa_llvm_backend.emit_ssa_function_to_llvm', forbidden)
        monkeypatch.setattr('src.compiler.ssa_llvm_backend.compile_artifact', forbidden)
        loaded = accelerators.load_metadata_copy_accelerator(tmp_path)
        assert loaded.metadata['loaded_without_compilation']
        calls = []
        native = loaded.run
        def observe(values, *, copier):
            calls.append(len(values))
            return native(values, copier=forbidden)
        loaded.run = observe
        graph = ProcessGraph(materialize_memory=False)
        _add(graph, 1, 'input')
        shared = {'leaf': object()}
        shared['backedge'] = shared
        graph.G.graph['metadata'] = shared
        graph.G.nodes[1]['attributes']['same'] = shared
        result = extract_clean_process_subgraph(graph, (1,))
        copied = result.G.graph['metadata']
        assert len(calls) == 1 and calls[0] > 1
        assert copied is not shared and copied['backedge'] is copied
        assert result.G.nodes[1]['attributes']['same'] is copied
        assert copied['leaf'] is shared['leaf']
        assert graph.G.graph['metadata'] is shared
        # One admission decision for the WHOLE batch, before any native call.
        class ListSubclass(list):
            pass
        legacy = ListSubclass([shared['leaf']])
        fallbacks = []
        result = native((legacy,), copier=lambda value: fallbacks.append(value) or list(value))
        assert result[0] == legacy and type(result[0]) is list and fallbacks == [legacy]
        # Matching checkpoints with altered DLL bytes are rejected at load.
        import hashlib
        original_sha = hashlib.sha256
        class WrongDigest:
            def hexdigest(self):
                return 'invalid'
        # Alter only the verification answer, not the installation or user data.
        library_bytes = provider.library_path.read_bytes()
        monkeypatch.setattr(hashlib, 'sha256', lambda value=b'': WrongDigest()
                            if value == library_bytes else original_sha(value))
        with pytest.raises(ValueError, match='digest changed'):
            accelerators.load_metadata_copy_accelerator(tmp_path)
    finally:
        registry.unregister_native(accelerators.METADATA_COPY)
        if previous is not None:
            registry._native[accelerators.METADATA_COPY] = previous


def test_source_nesting_does_not_equate_spanless_sibling_comprehensions():
    import ast
    from types import SimpleNamespace
    import networkx as nx
    from src.compiler.glsl_deployment_strategy import _source_control_nesting_hints
    expressions = [ast.parse('(item for item in value)').body[0].value,
                   ast.parse('(item for item in value)').body[0].value]
    graph = nx.DiGraph()
    for index, expression in enumerate(expressions):
        clause = expression.generators[0]
        graph.add_node(index, expr_obj=clause)
    from src.compiler.glsl_deployment_strategy import _ast_source_signature
    assert _ast_source_signature(expressions[0].generators[0]) == _ast_source_signature(expressions[1].generators[0])
    assert expressions[0].generators[0] is not expressions[1].generators[0]
    reductions = tuple(SimpleNamespace(loop_node_id=index, control_program=None, region_indices=())
                       for index in range(2))
    assert _source_control_nesting_hints(reductions, (), (), SimpleNamespace(G=graph)) == {}
