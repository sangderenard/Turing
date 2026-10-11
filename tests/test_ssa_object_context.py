"""Native context ABI tests, distinct from automatic Python-container lowering."""
from dataclasses import replace

import numpy as np
import pytest

from src.compiler.identity_concordance import concordance_report
from src.compiler.monotonic_ids import GLOBAL_MONOTONIC_IDS
from src.compiler.ssa_object_context import context_values, lower_object_context, validate_context_function
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
    return SSAObjectContext('test.context', table(('opaque_ref', 'int64'), (0,)),
        table(('opaque_ref', 'opaque_ref', 'opaque_ref'), (0, 1)), mint(),
        ('builtins.dict', 'builtins.list', 'builtins.object'))


def test_context_rejects_unrelated_owners_and_overlapping_storage():
    scope = context()
    with pytest.raises(ValueError, match='shared owner'):
        replace(scope, members=replace(scope.members, heap_owner_value_id=GLOBAL_MONOTONIC_IDS.mint()))
    with pytest.raises(ValueError, match='distinct storage'):
        replace(scope, next_handle_value_id=scope.objects.length_address_id)


def test_context_receipt_is_checked_against_live_descriptors():
    scope = context()
    module = lower_object_context(scope)
    name = 'native_object_construct'
    assert concordance_report(module).splitlines()[0].endswith('0 finding(s)')
    table = module.sequence_tables[name]
    table.sequences[scope.objects.sequence_id] = replace(scope.objects,
        capacity_value_id=GLOBAL_MONOTONIC_IDS.mint())
    with pytest.raises(ValueError, match='live storage'):
        validate_context_function(module, name)
    assert 'object-context-contract' in concordance_report(module)
    with pytest.raises(ValueError, match='live storage'):
        emit_ssa_function_to_llvm(module, name)


def test_native_context_constructs_grows_accesses_and_releases(tmp_path):
    scope = context()
    module = lower_object_context(scope)
    natives = {}
    for label in ('initialize', 'construct', 'kind', 'set', 'get', 'length', 'release'):
        name = 'native_object_' + label
        artifact = emit_ssa_function_to_llvm(module, name)
        assert artifact.complete, artifact.shortfalls
        natives[label] = compile_artifact(artifact, directory=tmp_path / label)
    buffers = {value.id: np.zeros(1, dtype=np.uint64 if value.dtype == 'ptr' else
        np.int32 if value.dtype == 'int' else np.int64) for value in context_values(scope)}
    def run(label, *operands):
        function = module.functions['native_object_' + label]
        receipt = function.metadata.get('opaque_object_helper', {})
        ids = (() if receipt.get('reference_formal_id') is None else (receipt['reference_formal_id'],))
        ids += receipt.get('operand_formal_ids', ())
        bound = {**buffers, **{key: np.array([value], dtype=np.int64)
                             for key, value in zip(ids, operands, strict=True)}}
        execution = prepare_artifact_execution(natives[label], bound).run()
        returned = function.metadata.get('return_value')
        return None if returned is None else execution.buffers[returned.id].item()
    run('initialize')
    try:
        # Handles and all rows live in caller-owned storage after helper return.
        handles = [run('construct', 1 + index % 3) for index in range(17)]
        assert handles == list(range(1, 18))
        for index, handle in enumerate(handles):
            assert run('kind', handle) == 1 + index % 3
            assert run('set', handle, 2**60 + 3, 2**60 + 17 + index)
            assert run('get', handle, 2**60 + 3) == 2**60 + 17 + index
            assert run('length', handle) == 1
        assert run('set', handles[0], 2**60 + 3, handles[0])
        assert run('get', handles[0], 2**60 + 3) == handles[0]  # self-edge
        assert run('set', handles[0], 2**60 + 4, handles[-1])
        assert run('set', handles[-1], 2**60 + 4, handles[0])  # backedge
        assert run('length', handles[0]) == 2
        assert run('kind', 9999) == 0
        assert run('get', handles[0], 9999) == 0
        assert not run('set', 9999, 1, 2)
        before = buffers[scope.next_handle_value_id].item()
        assert run('construct', 0) == 0
        assert buffers[scope.next_handle_value_id].item() == before
        buffers[scope.next_handle_value_id][0] = 1
        assert run('construct', 2) == 0  # collision cannot change kind
        assert run('kind', 1) == 1
        buffers[scope.next_handle_value_id][0] = 2**63 - 1
        assert run('construct', 1) == 0  # no signed overflow
    finally:
        run('release')
    assert all(not buffer.any() for buffer in buffers.values())
    run('release')
    run('initialize')
    assert run('construct', 3) == 1
    run('release')


@pytest.mark.parametrize('forwarded', [False, True])
def test_public_source_links_native_context_constructor_and_kind(tmp_path, forwarded):
    from pathlib import Path
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    scope = context()
    native_module = lower_object_context(scope, prefix='linked_context')
    cells = context_values(scope)
    names = [f'cell_{value.id}' for value in cells]
    signature = ','.join(names)
    declared = {label: ','.join(name for name, _ in native_module.functions[
        'linked_context_' + label].metadata['parameter_names']) for label in ('construct', 'kind')}
    # These are link SIGNATURE declarations, not implementation bodies. The
    # native roots supply the accepted physical results; the declaration must
    # retain an authored return edge so the caller has a result projection.
    source = (f'def construct({declared["construct"]}):\n    return 0\n'
              f'def inspect({declared["kind"]}):\n    return 0\n'
              f'def execute({signature},requested_kind):\n'
              f'    handle=construct({signature},requested_kind)\n'
              f'    return inspect({signature},handle)\n')
    if forwarded:
        source += (f'def relay({signature},requested_kind):\n'
                   f'    return execute({signature},requested_kind)\n')
    target = 'relay' if forwarded else 'execute'
    source += (f'def root({signature},requested_kind,count):\n'
               f'    total=0\n    for index in range(count):\n'
               f'        total=total+{target}({signature},requested_kind)\n'
               f'    return total\n')
    values = []
    for name, value in zip(names, cells, strict=True):
        receipt = {'function': '*', 'parameter': name, 'dtype': value.dtype, 'mutable': True,
                   'storage': 'span' if value.shape else 'scalar',
                   'python_type': 'numpy.ndarray' if value.shape else 'builtins.int'}
        if value.shape:
            receipt.update(rank=1, shape=list(value.shape))
        values.append(receipt)
    values.append({'function': '*', 'parameter': 'requested_kind', 'storage': 'scalar',
                   'dtype': 'int64', 'python_type': 'builtins.int'})
    values.append({'function': '*', 'parameter': 'count', 'storage': 'scalar',
                   'dtype': 'int64', 'python_type': 'builtins.int'})
    policy = ExtractionContract(Path(__file__).resolve().parents[1] /
        'extraction_contracts/program_extraction.yaml').with_program_abi(
        {'records': {}, 'bindings': [], 'values': values}).with_execution_file(
        Path(__file__).resolve().parents[1] / 'extraction_contracts/vehicle_full_native_execution.yaml')
    module, _, exports = lower_ast_source_to_ssa(source, 'root', name='context_source',
        extraction_contract=policy, linked_repository_ssa={
            'construct': (native_module, 'linked_context_construct', {
                'linked_context_construct': (native_module.functions['linked_context_construct'].metadata['return_value'],)}),
            'inspect': (native_module, 'linked_context_kind', {
                'linked_context_kind': (native_module.functions['linked_context_kind'].metadata['return_value'],)}),
        }, progress=lambda _: None, progress_bars=False)
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    native = compile_artifact(artifact, directory=tmp_path / 'source')
    root = module.functions[exports[0]]
    parameters = dict(root.metadata['parameter_names'])
    buffers = {parameters[name]: np.zeros(1, dtype=np.uint64 if value.dtype == 'ptr' else
        np.int32 if value.dtype == 'int' else np.int64) for name, value in zip(names, cells)
        if name in parameters}
    buffers[parameters[f'cell_{scope.next_handle_value_id}']][0] = 1
    try:
        for kind, count in ((1, 1), (3, 17), (2, 257)):
            buffers[parameters['requested_kind']] = np.array([kind], dtype=np.int64)
            buffers[parameters['count']] = np.array([count], dtype=np.int64)
            execution = prepare_artifact_execution(native, buffers)
            for name, value in zip(names, cells):
                if value.dtype == 'ptr' and name in parameters:
                    actual_id = parameters[name]
                    assert dict(zip(artifact.buffer_order, artifact.buffer_dtypes))[actual_id] == 'ptr'
                    assert np.shares_memory(execution.buffers[actual_id], buffers[actual_id])
            execution.run()  # Whole count-row loop is ONE native invocation.
            output = next(i.args[0] for b in root.blocks.values() for i in b.instrs if i.op == 'Ret')
            assert execution.buffers[output.id].item() == kind * count
    finally:
        release = emit_ssa_function_to_llvm(native_module, 'linked_context_release')
        destructor = compile_artifact(release, directory=tmp_path / 'release_source')
        release_buffers = {value.id: buffers[parameters[name]] for name, value in zip(names, cells)
                           if name in parameters}
        for value in cells:
            release_buffers.setdefault(value.id, np.zeros(1, dtype=np.uint64 if value.dtype == 'ptr' else np.int64))
        prepare_artifact_execution(destructor, release_buffers).run()
