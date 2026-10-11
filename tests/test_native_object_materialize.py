"""Bulk shallow object stamping: one native call, no per-member dispatch."""
import numpy as np
import pytest

from tests.test_source_object_context import context
from src.compiler.identity_concordance import concordance_report
from src.compiler.shell_external_references import ExternalReferenceValueCodec
from src.compiler.ssa_object_context import lower_object_context
from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution
from pathlib import Path
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


@pytest.fixture
def compiled(tmp_path):
    scope = context()
    module = lower_object_context(scope, prefix='bulk')
    assert concordance_report(module).splitlines()[0].endswith('0 finding(s)')
    artifacts = {label: compile_artifact(emit_ssa_function_to_llvm(module, 'bulk_' + label),
        directory=tmp_path / label) for label in ('materialize', 'release')}
    return scope, module, artifacts


@pytest.mark.parametrize('target', [list, tuple])
def test_single_native_materialization_preserves_order_and_shallow_identity(compiled, target):
    scope, module, artifacts = compiled
    leaf, shared = object(), []
    sources = ([leaf, shared, shared], (leaf, shared, shared),
               {'second': shared, 'first': leaf}, {1, 3, 2}, frozenset({2, 4}), [])
    for source in sources:
        boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (source,))
        function = module.functions['bulk_materialize']
        receipt = function.metadata['opaque_object_helper']
        feeds = {**boundary.buffers,
            receipt['reference_formal_id']: np.array([boundary.roots[0]], dtype=np.int64),
            receipt['operand_formal_ids'][0]: np.array([scope.kind_vocabulary.index('builtins.' + target.__name__) + 1], dtype=np.int64)}
        try:
            execution = prepare_artifact_execution(artifacts['materialize'], feeds).run()
            handle = int(execution.buffers[function.metadata['return_value'].id].item())
            result, = boundary.export((handle,))
            expected = target(source)
            assert type(result) is target and len(result) == len(expected)
            assert all(a is b for a, b in zip(result, expected))
            if target is tuple and type(source) is tuple:
                assert result is source
            else:
                assert result is not source
        finally:
            prepare_artifact_execution(artifacts['release'], boundary.buffers).run()
            boundary.close()


def test_large_materialization_does_not_call_member_lookup_or_store(compiled):
    scope, module, artifacts = compiled
    function = module.functions['bulk_materialize']
    callees = {i.attributes['callee'] for b in function.blocks.values() for i in b.instrs if i.op == 'Call'}
    assert callees == {'bulk_kind', 'bulk_materialize_reserve', 'bulk_construct'}
    leaf = object()
    source = [leaf] * 4097
    boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (source,))
    receipt = function.metadata['opaque_object_helper']
    feeds = {**boundary.buffers, receipt['reference_formal_id']: np.array([boundary.roots[0]], dtype=np.int64),
        receipt['operand_formal_ids'][0]: np.array([scope.kind_vocabulary.index('builtins.tuple') + 1], dtype=np.int64)}
    try:
        execution = prepare_artifact_execution(artifacts['materialize'], feeds).run()
        result, = boundary.export((int(execution.buffers[function.metadata['return_value'].id].item()),))
        assert len(result) == 4097 and all(item is leaf for item in result)
    finally:
        prepare_artifact_execution(artifacts['release'], boundary.buffers).run()
        boundary.close()


def test_bulk_refusal_and_prepared_boundary_contract(compiled):
    scope, module, artifacts = compiled
    function = module.functions['bulk_materialize']
    receipt = function.metadata['opaque_object_helper']
    source = np.array([0], dtype=np.int64)
    kind = np.array([scope.kind_vocabulary.index('builtins.tuple') + 1], dtype=np.int64)
    boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (object(),))
    feeds = {receipt['reference_formal_id']: source, receipt['operand_formal_ids'][0]: kind}
    try:
        with pytest.raises(ValueError, match='unbound input'):
            boundary.bind_execution(artifacts['materialize'], function, {})
        with pytest.raises(ValueError, match='replace leased'):
            boundary.bind_execution(artifacts['materialize'], function,
                {**feeds, scope.next_handle_value_id: np.array([1], dtype=np.int64)})
        invocation = boundary.bind_execution(artifacts['materialize'], function, feeds)
        counter = boundary.buffers[scope.next_handle_value_id].item()
        for handle, token in ((0, kind[0]), (9999, kind[0]), (boundary.roots[0], kind[0]),
                              (boundary.roots[0], 0)):
            source[0], kind[0] = handle, token
            invocation.run()
            assert invocation.buffers[function.metadata['return_value'].id].item() == 0
            assert boundary.buffers[scope.next_handle_value_id].item() == counter
    finally:
        prepare_artifact_execution(artifacts['release'], boundary.buffers).run()
        boundary.close()


def test_dispatch_validation_capability_does_not_scan_ir_per_call(compiled):
    from dataclasses import replace
    _scope, _module, artifacts = compiled
    artifact = artifacts['materialize']
    class CountedText(str):
        searches = 0
        def __contains__(self, item):
            self.searches += 1
            return super().__contains__(item)
    artifact.llvm_ir = CountedText(artifact.llvm_ir)
    for _ in range(100):
        artifact.reset_validation_error()
        assert artifact.take_validation_error() == 0
    assert artifact.llvm_ir.searches == 1
    changed = replace(artifact, llvm_ir=CountedText('declare void @turing_validation_error(i32)'))
    assert changed._uses_validation_runtime()
    assert changed.llvm_ir.searches == 1
    changed.llvm_ir = CountedText('')
    assert not changed._uses_validation_runtime()
    changed.needs_validation_runtime = True
    assert changed._uses_validation_runtime()


def test_cached_dispatch_still_reports_real_native_validation_errors(tmp_path):
    from src.transmogrifier.ssa import Function, BasicBlock, Instr, IRModule
    function = Function('dispatch_validation', [], {
        'entry': BasicBlock('entry', [Instr('Br', [], None, attributes={'target': 'guard'})]),
        'guard': BasicBlock('guard', [
        Instr('Call', [], None, attributes={'callee': 'turing_validation_error', 'error_code': 7}),
        Instr('Ret', [], None)])}, metadata={'control_ir': True})
    artifact = emit_ssa_function_to_llvm(IRModule({function.name: function}), function.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path)
    execution = prepare_artifact_execution(native, {})
    for _ in range(2):
        with pytest.raises(RuntimeError, match='code 7'):
            execution.run()


@pytest.mark.parametrize('corruption', ['position', 'extent'])
def test_bad_positional_edge_refuses_before_invalid_address(compiled, corruption):
    scope, module, artifacts = compiled
    function = module.functions['bulk_materialize']
    receipt = function.metadata['opaque_object_helper']
    boundary = ExternalReferenceValueCodec().bind_native_objects(scope, ([object()],))
    if corruption == 'position':
        boundary._initial_columns[scope.members.column_value_ids[1]][0] = -1
    else:
        boundary.buffers[scope.members.length_address_id][0] = 2
    feeds = {receipt['reference_formal_id']: np.array([boundary.roots[0]], dtype=np.int64),
        receipt['operand_formal_ids'][0]: np.array([scope.kind_vocabulary.index('builtins.tuple') + 1], dtype=np.int64)}
    try:
        invocation = boundary.bind_execution(artifacts['materialize'], function, feeds).run()
        assert invocation.buffers[function.metadata['return_value'].id].item() == 0
        assert boundary.buffers[scope.members.length_address_id].item() == (1 if corruption == 'position' else 2)
    finally:
        prepare_artifact_execution(artifacts['release'], boundary.buffers).run()
        boundary.close()


@pytest.mark.parametrize('expression', ['tuple(value)', 'list(value)', 'tuple(list(value))'])
def test_ordinary_source_materialization_has_real_input_and_one_native_invocation(tmp_path, expression):
    scope = context()
    repository = Path(__file__).resolve().parents[1]
    policy = ExtractionContract(repository / 'extraction_contracts/program_extraction.yaml').with_program_abi(
        {'records': {}, 'bindings': [], 'values': [{'function': 'root', 'parameter': 'value',
         'storage': 'reference', 'python_type': 'builtins.object'}]}).with_execution_file(
        repository / 'extraction_contracts/vehicle_full_native_execution.yaml')
    module, _, exports = lower_ast_source_to_ssa('def root(value):\n    return ' + expression + '\n',
        'root', name='authored_materialize', extraction_contract=policy, native_object_context=scope,
        progress=lambda _: None, progress_bars=False)
    assert concordance_report(module).splitlines()[0].endswith('0 finding(s)')
    root = module.functions[exports[0]]
    native = compile_artifact(emit_ssa_function_to_llvm(module, exports[0]), directory=tmp_path / 'source')
    release_module = lower_object_context(scope, prefix='release_bulk')
    release = compile_artifact(emit_ssa_function_to_llvm(release_module, 'release_bulk_release'),
        directory=tmp_path / 'release')
    leaf, shared = object(), []
    original = [leaf, shared, shared]
    boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (original,))
    input_id = dict(root.metadata['parameter_names'])['value']
    feeds = {input_id: np.array([boundary.roots[0]], dtype=np.int64)}
    try:
        invocation = boundary.bind_execution(native, root, feeds)
        execution = invocation.run()
        output = next(i.args[0] for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
        result, = boundary.export((int(execution.buffers[output.id].item()),))
        assert type(result) is (list if expression == 'list(value)' else tuple)
        assert len(result) == 3 and all(a is b for a, b in zip(result, original))
        pointers = invocation.execution.pointers
        invocation.run()
        assert invocation.native_invocations == 2
        assert invocation.execution.pointers is pointers
        repeated, = boundary.export((int(execution.buffers[output.id].item()),))
        assert all(a is b for a, b in zip(repeated, original))
    finally:
        prepare_artifact_execution(release, boundary.buffers).run()
        boundary.close()
    with pytest.raises(RuntimeError, match='lease is closed'):
        invocation.run()
