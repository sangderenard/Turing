"""Boundary conversion tests; not proof of automatic copier-source lowering."""
import numpy as np
import pytest

from src.compiler.monotonic_ids import GLOBAL_MONOTONIC_IDS
from src.compiler.shell_external_references import ExternalReferenceValueCodec
from src.transmogrifier.ssa import SSAObjectContext, SSASequenceDescriptor, SSASequenceCapacityPolicy


def context():
    mint = GLOBAL_MONOTONIC_IDS.mint
    owner = mint()
    def table(dtypes, keys):
        columns = tuple(mint() for _ in dtypes)
        return SSASequenceDescriptor(columns[0], columns, mint(), mint(),
            status_address_id=mint(), column_dtypes=dtypes, key_columns=keys,
            capacity_policy=SSASequenceCapacityPolicy.DYNAMIC, heap_owner_value_id=owner)
    return SSAObjectContext('boundary.test', table(('opaque_ref', 'int64'), (0,)),
        table(('opaque_ref', 'opaque_ref', 'opaque_ref'), (0, 1)), mint(),
        ('builtins.dict', 'builtins.list', 'builtins.tuple', 'builtins.set',
         'builtins.frozenset', 'builtins.object'))


def test_conversion_preserves_graph_aliases_cycles_keys_and_leaf_identity():
    codec = ExternalReferenceValueCodec()
    leaf = object()
    shared = [leaf]
    ring = (shared,)
    shared.extend((shared, ring))
    key = ('key', 123)
    source = {key: shared, 'ring': ring, 'set': {key}, 'frozen': frozenset((key,)),
              'equal': {1: 'a', True: 'b'}, 'none': None}
    boundary = codec.bind_native_objects(context(), (source, shared, leaf, source))
    result, copied, returned_leaf, alias = boundary.export()
    assert result is alias and result is source and copied is shared
    assert returned_leaf is leaf and copied[0] is leaf
    assert copied[1] is copied and copied[2][0] is copied
    assert result[key] is copied and result['ring'] is copied[2]
    assert next(iter(result)) is next(iter(result['set']))
    assert next(iter(result['frozen'])) is next(iter(result))
    assert result['equal'] == {1: 'b'} and result['none'] is None
    boundary.close()
    assert not codec._objects
    with pytest.raises(RuntimeError, match='closed'):
        boundary.export()


def test_deep_conversion_is_iterative_and_domain_lease_is_exclusive():
    source = object()
    leaf = source
    for _ in range(2500):
        source = (source,)
    codec = ExternalReferenceValueCodec()
    boundary = codec.bind_native_objects(context(), (source,))
    with pytest.raises(RuntimeError, match='leased'):
        codec.retain(object())
    with pytest.raises(RuntimeError, match='leased'):
        codec.bind_native_objects(context(), ())
    result, = boundary.export()
    for _ in range(2500):
        result, = result
    assert result is leaf
    boundary.close()
    assert not codec._objects


def test_subclasses_stay_opaque_and_none_retention_can_release():
    class Custom(list):
        def __iter__(self):
            raise AssertionError('must not infer custom container semantics')
    value = Custom()
    codec = ExternalReferenceValueCodec()
    boundary = codec.bind_native_objects(context(), (value, None))
    exported = boundary.export()
    assert exported[0] is value and exported[1] is None
    boundary.close()


def test_invalid_native_graph_refuses_without_collapsing_dictionary_semantics():
    scope = context()
    codec = ExternalReferenceValueCodec()
    original = {1: 'first', 2: 'second'}
    boundary = codec.bind_native_objects(scope, (original, True))
    true_handle = boundary.roots[1]
    key_column = boundary._initial_columns[scope.members.column_value_ids[1]]
    key_column[1] = true_handle  # 1 and True distinct refs, Python-equal keys.
    with pytest.raises(ValueError, match='collapse'):
        boundary.export()
    assert original == {1: 'first', 2: 'second'}
    boundary.close()


def test_immutable_mutation_and_impossible_immutable_cycle_refuse():
    scope = context()
    codec = ExternalReferenceValueCodec()
    original = (object(),)
    replacement = object()
    boundary = codec.bind_native_objects(scope, (original, replacement))
    column = boundary._initial_columns[scope.members.column_value_ids[2]]
    column[0] = boundary.roots[1]
    with pytest.raises(ValueError, match='immutable object'):
        boundary.export()
    column[0] = boundary.roots[0]
    with pytest.raises(ValueError, match='immutable-only cycle'):
        boundary.export()
    boundary.close()
    boundary = codec.bind_native_objects(scope, ([1],))
    boundary._initial_columns[scope.members.column_value_ids[1]][0] = 7
    with pytest.raises(ValueError, match='contiguous'):
        boundary.export()
    boundary.close()


def test_invalid_extent_is_rejected_before_dereferencing_pointer():
    scope = context()
    boundary = ExternalReferenceValueCodec().bind_native_objects(scope, ([],))
    boundary.buffers[scope.objects.length_address_id][0] = 10
    with pytest.raises(ValueError, match='extents'):
        boundary.export()
    boundary.buffers[scope.objects.length_address_id][0] = 1
    boundary.close()


@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int', 'int64'])
def test_raw_memory_load_keeps_declared_integer_width(dtype):
    from src.compiler.ir_indexing import lower_indexing_to_ssa_addressing
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
    mint = GLOBAL_MONOTONIC_IDS.mint
    storage = SSAValue(mint(), dtype, shape=(1,))
    zero, address, loaded = SSAValue(mint(), 'int64'), SSAValue(mint(), 'ptr'), SSAValue(mint(), dtype)
    block = BasicBlock('entry', [Instr('Const', [], zero, attributes={'value': 0}),
        Instr('GetElementPtr', [storage, zero], address), Instr('Load', [address], loaded),
        Instr('Ret', [loaded], None)])
    function = Function('typed_memory_read', [storage], {'entry': block})
    lower_indexing_to_ssa_addressing({function.name: function})
    assert loaded.dtype == dtype


def test_abstract_call_projection_is_not_a_raw_memory_width_declaration():
    from src.compiler.ir_indexing import lower_indexing_to_ssa_addressing
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
    mint = GLOBAL_MONOTONIC_IDS.mint
    aggregate, zero, address, projected = (SSAValue(mint(), dtype)
        for dtype in ('ssa.aggregate', 'int64', 'ptr', 'int'))
    block = BasicBlock('entry', [
        Instr('Call', [], aggregate, attributes={'callee': 'declared_region', 'result_convention': 'ssa.aggregate'}),
        Instr('Const', [], zero, attributes={'value': 0}),
        Instr('GetElementPtr', [aggregate, zero], address),
        Instr('Load', [address], projected, attributes={'aggregate_index': 0, 'source_output_id': projected.id}),
        Instr('Ret', [projected], None)])
    function = Function('abstract_projection', [], {'entry': block})
    lower_indexing_to_ssa_addressing({function.name: function})
    assert projected.dtype == 'int64'  # Incumbent Python-integer projection rule.


def test_native_growth_and_new_container_export_use_existing_context(tmp_path):
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_object_context import lower_object_context
    from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution
    scope = context()
    module = lower_object_context(scope, prefix='boundary_native')
    assert concordance_report(module).splitlines()[0].endswith('0 finding(s)')
    natives = {label: compile_artifact(emit_ssa_function_to_llvm(module, 'boundary_native_' + label),
        directory=tmp_path / label) for label in ('construct', 'set', 'release')}
    leaf = object()
    source = [leaf]
    codec = ExternalReferenceValueCodec()
    boundary = codec.bind_native_objects(scope, (source, leaf))
    original_arrays = {cell: array.copy() for cell, array in boundary._initial_columns.items()}

    def run(label, *operands):
        function = module.functions['boundary_native_' + label]
        receipt = function.metadata.get('opaque_object_helper', {})
        ids = (() if receipt.get('reference_formal_id') is None else (receipt['reference_formal_id'],))
        ids += receipt.get('operand_formal_ids', ())
        bound = {**boundary.buffers, **{key: np.array([value], dtype=np.int64)
                 for key, value in zip(ids, operands, strict=True)}}
        execution = prepare_artifact_execution(natives[label], bound)
        for key in boundary.buffers:
            if key in execution.buffers:
                assert np.shares_memory(boundary.buffers[key], execution.buffers[key])
        execution.run()
        returned = function.metadata.get('return_value')
        return None if returned is None else execution.buffers[returned.id].item()

    try:
        # SSA ABI integration, not compilation of a substitute metadata copier.
        new = run('construct', scope.kind_vocabulary.index('builtins.dict') + 1)
        assert new > max(boundary.retained)
        assert run('set', new, boundary.roots[1], boundary.roots[0])
        assert run('set', boundary.roots[0], 2, new)  # list -> new dict -> list cycle
        assert run('set', boundary.roots[0], 3, boundary.roots[1])
        assert source == [leaf]  # Host mutations publish at export, not per native op.
        result, fresh = boundary.export((boundary.roots[0], new))
        assert result[0] is leaf and result[2] is leaf
        assert result[1] is fresh and fresh[leaf] is result
        assert result is source
        again, = boundary.export((new,))
        assert again is fresh and source[1] is fresh
        assert boundary.buffers[scope.objects.heap_owner_value_id][0]
        with pytest.raises(RuntimeError, match='release'):
            boundary.close()
    finally:
        run('release')
        boundary.close()
    # Growth writes replacement arenas; borrowed initial arrays remain intact.
    for cell, original in original_arrays.items():
        np.testing.assert_array_equal(boundary._initial_columns[cell], original)
    handle = codec.retain(fresh)
    assert codec.object(handle) is fresh
    codec.release(handle)


def test_public_source_loop_accepts_python_graph_and_exports_native_objects(tmp_path):
    from pathlib import Path
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_object_context import context_values, lower_object_context
    from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution
    scope = context()
    native_module = lower_object_context(scope, prefix='source_boundary')
    cells = context_values(scope)
    names = [f'cell_{value.id}' for value in cells]
    signature = ','.join(names)
    declared = {label: ','.join(name for name, _ in native_module.functions[
        'source_boundary_' + label].metadata['parameter_names']) for label in ('construct', 'set')}
    source = (f'def construct({declared["construct"]}):\n    return 0\n'
              f'def store({declared["set"]}):\n    return 0\n'
              f'def root({signature},root_ref,leaf_ref,requested_kind,count):\n'
              f'    for index in range(count):\n'
              f'        item=construct({signature},requested_kind)\n'
              f'        store({signature},item,leaf_ref,root_ref)\n'
              f'        store({signature},root_ref,index+2,item)\n'
              f'    return root_ref\n')
    values = []
    for name, value in zip(names, cells, strict=True):
        receipt = {'function': '*', 'parameter': name, 'dtype': value.dtype, 'mutable': True,
                   'storage': 'span' if value.shape else 'scalar',
                   'python_type': 'numpy.ndarray' if value.shape else 'builtins.int'}
        if value.shape:
            receipt.update(rank=1, shape=list(value.shape))
        values.append(receipt)
    for name in ('root_ref', 'leaf_ref'):
        values.append({'function': '*', 'parameter': name, 'dtype': 'opaque_ref',
            'storage': 'reference', 'python_type': 'builtins.object'})
    for name in ('requested_kind', 'count'):
        values.append({'function': '*', 'parameter': name, 'dtype': 'int64',
            'storage': 'scalar', 'python_type': 'builtins.int'})
    repository = Path(__file__).resolve().parents[1]
    policy = ExtractionContract(repository / 'extraction_contracts/program_extraction.yaml').with_program_abi(
        {'records': {}, 'bindings': [], 'values': values}).with_execution_file(
        repository / 'extraction_contracts/vehicle_full_native_execution.yaml')
    links = {}
    for declaration, label in (('construct', 'construct'), ('store', 'set')):
        name = 'source_boundary_' + label
        links[declaration] = (native_module, name, {name: (native_module.functions[name].metadata['return_value'],)})
    module, _, exports = lower_ast_source_to_ssa(source, 'root', name='python_graph_boundary',
        extraction_contract=policy, linked_repository_ssa=links, progress=lambda _: None, progress_bars=False)
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'source')
    destructor = compile_artifact(emit_ssa_function_to_llvm(native_module, 'source_boundary_release'),
        directory=tmp_path / 'release')
    root = module.functions[exports[0]]
    parameters = dict(root.metadata['parameter_names'])
    output = next(instruction.args[0] for block in root.blocks.values()
                  for instruction in block.instrs if instruction.op == 'Ret')
    codec = ExternalReferenceValueCodec()
    for count in (1, 17, 257):
        leaf = object()
        original = [leaf]
        boundary = codec.bind_native_objects(scope, (original, leaf))
        feeds = {parameters[name]: boundary.buffers[value.id] for name, value in zip(names, cells)
                 if name in parameters}
        for name, value in (('root_ref', boundary.roots[0]), ('leaf_ref', boundary.roots[1]),
                            ('requested_kind', 1), ('count', count)):
            feeds[parameters[name]] = np.array([value], dtype=np.int64)
        try:
            execution = prepare_artifact_execution(native, feeds)
            for name, value in zip(names, cells):
                if name in parameters:
                    assert np.shares_memory(execution.buffers[parameters[name]], boundary.buffers[value.id])
            execution.run()  # ONE invocation for the entire source loop.
            assert original == [leaf]
            result, = boundary.export((int(execution.buffers[output.id][0]),))
            assert len(result) == count + 1 and result[0] is leaf
            assert len({id(item) for item in result[1:]}) == count
            assert all(item[leaf] is result for item in result[1:])
            assert result is original
        finally:
            prepare_artifact_execution(destructor, boundary.buffers).run()
            boundary.close()
        assert not codec._objects
