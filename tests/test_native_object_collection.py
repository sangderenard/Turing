"""Authored generator loops own dense native rows, not Python dispatch."""
from pathlib import Path
import numpy as np
import pytest

from tests.test_source_object_context import context
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.shell_external_references import ExternalReferenceValueCodec
from src.compiler.ssa_object_context import lower_object_context
from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution


@pytest.mark.parametrize('target', ['tuple', 'list'])
@pytest.mark.parametrize('body', ['item', 'identity(item)', 'fresh(item)', 'copy(item)', 'direct'])
def test_native_generator_dynamic_extent_and_heap_relocation(tmp_path, target, body):
    repository = Path(__file__).resolve().parents[1]
    policy = ExtractionContract(repository / 'extraction_contracts/program_extraction.yaml').with_program_abi(
        {'records': {}, 'bindings': [], 'values': [{'function': 'root', 'parameter': 'value',
         'storage': 'reference', 'python_type': 'builtins.object'},
        {'function': 'copy', 'parameter': 'item', 'storage': 'reference', 'python_type': 'builtins.object'}]}).with_execution_file(
        repository / 'extraction_contracts/vehicle_full_native_execution.yaml')
    scope = context()
    prefix = '    members=value\n' if body == 'direct' else '    members=tuple(value)\n'
    expression = 'item' if body == 'direct' else body
    source = ('def fresh(item):\n    result=[]\n    return result\n'
              'def identity(item):\n    return item\n'
              'def copy(item):\n    return list(item)\n'
              'def root(value):\n' + prefix + '    return ' + target + '(' + expression + ' for item in members)\n')
    module, _, exports = lower_ast_source_to_ssa(source, 'root', name='native_generator',
        extraction_contract=policy, native_object_context=scope, progress=lambda _: None, progress_bars=False)
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    root = module.functions[exports[0]]
    assert root.metadata['native_object_collections']
    assert all(value.dtype not in {None, 'unknown'} for value in root.args)
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'source')
    release_module = lower_object_context(scope, prefix='collection_release')
    release = compile_artifact(emit_ssa_function_to_llvm(release_module, 'collection_release_release'), directory=tmp_path / 'release')
    parameters = dict(root.metadata['parameter_names'])
    output = next(i.args[0] for b in root.blocks.values() for i in b.instrs if i.op == 'Ret' and i.args)
    leaf = object()
    for size in ((0, 1, 17, 257, 4097) if body in {'item', 'identity(item)'} else (0, 1, 17, 257)):
        original = (([leaf, leaf, leaf],) * size if body == 'copy(item)' else (leaf,) * size)
        boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (original,))
        try:
            invocation = boundary.bind_execution(native, root, {parameters['value']: np.array([boundary.roots[0]], dtype=np.int64)})
            invocation.run()
            assert invocation.native_invocations == 1
            result, = boundary.export((int(invocation.buffers[output.id].item()),))
            assert type(result).__name__ == target and len(result) == size
            if body in {'item', 'identity(item)', 'direct'}:
                assert all(item is leaf for item in result)
            else:
                assert all(type(item) is list and item == ([] if body == 'fresh(item)' else [leaf]*3) for item in result), (size, [(type(item).__name__, id(item)) for item in result[:5]])
                assert len({id(item) for item in result}) == size
        finally:
            prepare_artifact_execution(release, boundary.buffers).run()
            boundary.close()


def test_scalar_int64_load_is_snapshot_not_mutable_address(tmp_path):
    from src.transmogrifier.ssa import SSAValue, Function, IRModule, BasicBlock, Instr
    cell, zero, replacement = SSAValue(1, 'int64', (1,)), SSAValue(2, 'int64'), SSAValue(3, 'int64')
    address, before, after, difference = SSAValue(4, 'ptr'), SSAValue(5, 'int64'), SSAValue(6, 'int64'), SSAValue(7, 'int64')
    entry = BasicBlock('entry', [Instr('Const', [], zero, attributes={'value': 0}),
        Instr('Const', [], replacement, attributes={'value': 19}),
        Instr('GetElementPtr', [cell, zero], address), Instr('Load', [address], before),
        Instr('Store', [replacement, address], None), Instr('Br', [], None, attributes={'target': 'done'})], ['done'])
    done = BasicBlock('done', [Instr('Load', [address], after), Instr('Sub', [after, before], difference), Instr('Ret', [difference], None)])
    function = Function('wide_load_snapshot', [cell], {'entry': entry, 'done': done}, metadata={'control_ir': True})
    native = compile_artifact(emit_ssa_function_to_llvm(IRModule({function.name: function}), function.name), directory=tmp_path)
    execution = prepare_artifact_execution(native, {cell.id: np.array([7], dtype=np.int64)}).run()
    assert int(execution.buffers[difference.id].item()) == 12


@pytest.mark.parametrize('corruption', ['column', 'index', 'kind', 'member'])
def test_collection_concordance_refuses_stale_live_producers(corruption):
    from src.compiler.ssa_object_operations import validate_object_operations
    from src.transmogrifier.ssa import SSAValue
    repository = Path(__file__).resolve().parents[1]
    policy = ExtractionContract(repository / 'extraction_contracts/program_extraction.yaml').with_program_abi(
        {'records': {}, 'bindings': [], 'values': [{'function': 'root', 'parameter': 'value',
         'storage': 'reference', 'python_type': 'builtins.object'}]}).with_execution_file(
        repository / 'extraction_contracts/vehicle_full_native_execution.yaml')
    scope = context()
    module, _, exports = lower_ast_source_to_ssa('def root(value):\n    members=tuple(value)\n    return tuple(item for item in members)\n',
        'root', name='stale_collection', extraction_contract=policy, native_object_context=scope, progress=lambda _: None, progress_bars=False)
    root = module.functions[exports[0]]
    instructions = [i for b in root.blocks.values() for i in b.instrs]
    address = next(i for i in instructions if i.attributes.get('native_collection_row', {}).get('action') == 'address')
    if corruption == 'column':
        address.args[0] = next(value for value in root.args if value.id == scope.members.column_value_ids[0])
    elif corruption == 'index':
        address.attributes['native_collection_row']['index_value_id'] = -1
    elif corruption == 'kind':
        call = next(i for i in instructions if i.attributes.get('source_object_collection'))
        receipt = call.attributes['opaque_object_operation']
        operand, = receipt['operand_formal_ids']
        token = dict(call.attributes['opaque_object_bindings'])[operand]
        next(i for i in instructions if i.res is not None and i.res.id == token).attributes['value'] = 0
    else:
        publication = next(i for i in instructions if i.attributes.get('native_collection_row', {}).get('action') == 'publish')
        publication.args[0] = SSAValue(publication.args[0].id, 'int64')
    with pytest.raises(ValueError, match='collection'):
        validate_object_operations(module, exports[0])
    assert 'source-object-contract' in concordance_report(module)
    with pytest.raises(ValueError, match='collection'):
        emit_ssa_function_to_llvm(module, exports[0])
