"""Ordinary authored object births through the public source compiler."""
from pathlib import Path
import numpy as np
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.monotonic_ids import GLOBAL_MONOTONIC_IDS
from src.compiler.shell_external_references import ExternalReferenceValueCodec
from src.compiler.ssa_object_context import lower_object_context
from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution
from src.transmogrifier.ssa import SSAObjectContext, SSASequenceDescriptor, SSASequenceCapacityPolicy


def context():
    mint = GLOBAL_MONOTONIC_IDS.mint
    owner = mint()
    def table(dtypes, keys):
        columns = tuple(mint() for _ in dtypes)
        return SSASequenceDescriptor(columns[0], columns, mint(), mint(),
            status_address_id=mint(), column_dtypes=dtypes, key_columns=keys,
            capacity_policy=SSASequenceCapacityPolicy.DYNAMIC, heap_owner_value_id=owner)
    return SSAObjectContext('source.context', table(('opaque_ref', 'int64'), (0,)),
        table(('opaque_ref', 'opaque_ref', 'opaque_ref'), (0, 1)), mint(),
        ('builtins.dict', 'builtins.list', 'builtins.tuple', 'builtins.set',
         'builtins.frozenset', 'builtins.object'))


@pytest.mark.parametrize('literal, expected', [('{}', dict), ('[]', list)])
@pytest.mark.parametrize('mode', ['direct', 'forwarded', 'recursive', 'loop', 'conditional'])
def test_source_constructor_has_handle_not_private_recursive_columns(tmp_path, literal, expected, mode):
    if mode == 'direct':
        source = f'def root(n):\n    item={literal}\n    return item\n'
    elif mode == 'forwarded':
        source = (f'def make(n):\n    item={literal}\n    return item\n'
                  'def relay(n):\n    return make(n)\ndef root(n):\n    return relay(n)\n')
    elif mode == 'recursive':
        source = (f'def visit(n):\n    item={literal}\n'
                  '    if n>0.0:\n        result=visit(n-1.0)\n'
                  '    else:\n        result=item\n    return result\n'
                  'def root(n):\n    return visit(n)\n')
    elif mode == 'loop':
        source = (f'def root(n):\n    item={literal}\n'
                  f'    while n>0.0:\n        item={literal}\n        n=n-1.0\n'
                  '    return item\n')
    else:
        source = (f'def root(n):\n    if n>0.0:\n        item={literal}\n'
                  f'    else:\n        item={literal}\n    return item\n')
    repository = Path(__file__).resolve().parents[1]
    policy = ExtractionContract(repository / 'extraction_contracts/program_extraction.yaml').with_program_abi(
        {'records': {}, 'bindings': [], 'values': [{'function': '*', 'parameter': 'n',
         'storage': 'scalar', 'dtype': 'float64', 'python_type': 'builtins.float'}]}).with_execution_file(
        repository / 'extraction_contracts/vehicle_full_native_execution.yaml')
    scope = context()
    module, _, exports = lower_ast_source_to_ssa(source, 'root', name='authored_birth',
        extraction_contract=policy, native_object_context=scope, progress=lambda _: None, progress_bars=False)
    constructors = [instruction for function in module.functions.values()
                    for block in function.blocks.values() for instruction in block.instrs
                    if instruction.attributes.get('source_object_constructor')]
    assert len(constructors) == (2 if mode in {'loop', 'conditional'} else 1)
    assert constructors[0].res.dtype == 'opaque_ref' and constructors[0].res.shape == ()
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'source')
    release_module = lower_object_context(scope, prefix='release_source')
    release = compile_artifact(emit_ssa_function_to_llvm(release_module, 'release_source_release'),
                               directory=tmp_path / 'release')
    root = module.functions[exports[0]]
    outputs = [instruction.args[0] for block in root.blocks.values()
               for instruction in block.instrs if instruction.op == 'Ret' and instruction.args]
    for depth in ((0, 1, 7, 32) if mode in {'recursive', 'loop', 'conditional'} else (0,)):
        boundary = ExternalReferenceValueCodec().bind_native_objects(scope, ())
        feeds = {}
        for value in root.args:
            role = value.accounting.get('native_object_context_value_id')
            if role is not None:
                feeds[value.id] = boundary.buffers[role]
        parameters = dict(root.metadata['parameter_names'])
        if 'n' in parameters:
            feeds[parameters['n']] = np.array([depth], dtype=np.float64)
        try:
            execution = prepare_artifact_execution(native, feeds)
            execution.run()
            handle = int(execution.buffers[outputs[0].id].item())
            result, = boundary.export((handle,))
            assert type(result) is expected and len(result) == 0
            assert boundary.buffers[scope.objects.length_address_id][0] == (depth + 1 if mode in {'recursive', 'loop'} else 1)
        finally:
            prepare_artifact_execution(release, boundary.buffers).run()
            boundary.close()


@pytest.mark.parametrize('expression', ['item', 'item[key]', 'len(item)'])
def test_source_dictionary_reference_store_lookup_and_length(tmp_path, expression):
    repository = Path(__file__).resolve().parents[1]
    policy = ExtractionContract(repository / 'extraction_contracts/program_extraction.yaml').with_program_abi(
        {'records': {}, 'bindings': [], 'values': [{'function': 'root', 'parameter': name,
         'storage': 'reference', 'python_type': 'builtins.object'} for name in ('key', 'value')]}).with_execution_file(
        repository / 'extraction_contracts/vehicle_full_native_execution.yaml')
    scope = context()
    source = 'def root(key,value):\n    item={}\n    item[key]=value\n    return ' + expression + '\n'
    module, _, exports = lower_ast_source_to_ssa(source, 'root', name='authored_member',
        extraction_contract=policy, native_object_context=scope, progress=lambda _: None, progress_bars=False)
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'source')
    release_module = lower_object_context(scope, prefix='release_member')
    release = compile_artifact(emit_ssa_function_to_llvm(release_module, 'release_member_release'),
                               directory=tmp_path / 'release')
    root = module.functions[exports[0]]
    parameters = dict(root.metadata['parameter_names'])
    key, value = object(), object()
    boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (key, value))
    feeds = {argument.id: boundary.buffers[argument.accounting['native_object_context_value_id']]
             for argument in root.args if 'native_object_context_value_id' in argument.accounting}
    for name, handle in zip(('key', 'value'), boundary.roots):
        if name in parameters:
            feeds[parameters[name]] = np.array([handle], dtype=np.int64)
    try:
        execution = prepare_artifact_execution(native, feeds).run()
        output = next(instruction.args[0] for block in root.blocks.values()
                      for instruction in block.instrs if instruction.op == 'Ret')
        returned = int(execution.buffers[output.id].item())
        if expression == 'len(item)':
            assert returned == 1
        else:
            result, = boundary.export((returned,))
            assert result is value if expression == 'item[key]' else result == {key: value}
    finally:
        prepare_artifact_execution(release, boundary.buffers).run()
        boundary.close()


@pytest.mark.parametrize('corruption', ['role', 'token', 'context'])
def test_source_context_audit_and_emission_refuse_stale_receipts(corruption):
    from src.compiler.ssa_object_operations import validate_object_operations
    repository = Path(__file__).resolve().parents[1]
    scope = context()
    policy = ExtractionContract(repository / 'extraction_contracts/program_extraction.yaml').with_execution_file(
        repository / 'extraction_contracts/vehicle_full_native_execution.yaml')
    module, _, exports = lower_ast_source_to_ssa('def root():\n    item={}\n    return item\n',
        'root', name='stale_birth', extraction_contract=policy, native_object_context=scope,
        progress=lambda _: None, progress_bars=False)
    root = module.functions[exports[0]]
    if corruption == 'role':
        formal = next(value for value in root.args if value.id == scope.next_handle_value_id)
        formal.accounting['native_object_context_value_id'] = -1
    elif corruption == 'token':
        token = next(i for b in root.blocks.values() for i in b.instrs if i.op == 'Const')
        token.attributes['value'] = 2
    else:
        module.metadata['native_object_context'] = dict(scope.to_mapping(), storage_identity='other')
    assert 'object-context-contract' in concordance_report(module) or 'source-object-contract' in concordance_report(module)
    with pytest.raises(ValueError):
        validate_object_operations(module, exports[0])
