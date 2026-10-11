"""Native allocation ABI unit tests; not claims of a compiled metadata copier."""
import ctypes

import numpy as np
import pytest

from src.compiler.ssa_heap_allocation import heap_allocate, heap_release
from src.compiler.ssa_llvm_backend import (
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


def owner(value_id):
    return SSAValue(value_id, 'ptr', shape=(1,), accounting={'heap_owner': True})


def make_allocation_module(dtype='int64'):
    count, scope, arena = SSAValue(0, 'int64'), owner(1), SSAValue(2, 'ptr')
    allocate = Function('allocate', [count, scope], {'entry': BasicBlock('entry', [
        heap_allocate(count, scope, arena, element_dtype=dtype, zero_initialize=True),
        Instr('Ret', [arena], None),
    ])})
    scope2 = owner(10)
    release = Function('release', [scope2], {'entry': BasicBlock('entry', [
        heap_release(scope2), Instr('Ret', [], None),
    ])})
    return IRModule(functions={'allocate': allocate, 'release': release})


@pytest.mark.parametrize('dtype, ctype', [('int64', ctypes.c_int64),
                                       ('int32', ctypes.c_int32),
                                       ('float64', ctypes.c_double),
                                       ('float32', ctypes.c_float),
                                       ('opaque_ref', ctypes.c_uint64)])
def test_runtime_sized_arena_survives_return_until_owner_release(dtype, ctype, tmp_path):
    module = make_allocation_module(dtype)
    allocated = emit_ssa_function_to_llvm(module, 'allocate')
    released = emit_ssa_function_to_llvm(module, 'release')
    assert allocated.complete, allocated.shortfalls
    assert released.complete, released.shortfalls
    native = compile_artifact(allocated, directory=tmp_path / 'allocate')
    destructor = compile_artifact(released, directory=tmp_path / 'release')
    scope = np.zeros(1, dtype=np.uint64)
    kept = []
    try:
        for n in (0, 1, 7, 4097):
            execution = prepare_artifact_execution(native, {
                0: np.array([n], dtype=np.int64), 1: scope,
            }).run()
            address = int(execution.buffers[2].item())
            assert address and scope[0]
            # Read AFTER native helper returned. Prior allocations must also
            # survive later invocations through the same owner.
            values = (ctype * n).from_address(address)
            assert list(values) == [0] * n
            if n:
                values[0], values[n - 1] = 17, 29
                kept.append((values, n))
            for old, size in kept:
                assert old[size - 1] == 29
                if size > 1:
                    assert old[0] == 17
    finally:
        prepare_artifact_execution(destructor, {10: scope}).run()
    assert scope[0] == 0
    prepare_artifact_execution(destructor, {10: scope}).run()  # Empty release is safe.


def test_native_caller_reads_returned_arena_without_host_materialization(tmp_path):
    module = make_allocation_module()
    count, scope = SSAValue(20, 'int64'), owner(21)
    address = SSAValue(22, 'ptr', accounting={'heap_element_dtype': 'int64'})
    zero, number = SSAValue(23, 'int64'), SSAValue(24, 'int64')
    slot, loaded = SSAValue(25, 'ptr'), SSAValue(26, 'int64')
    module.functions['root'] = Function('root', [count, scope], {'entry': BasicBlock('entry', [
        Instr('Call', [count, scope], address, attributes={'callee': 'allocate'}),
        Instr('Const', [], zero, attributes={'value': 0}),
        Instr('Const', [], number, attributes={'value': 2**60 + 3}),
        Instr('GetElementPtr', [address, zero], slot),
        Instr('Store', [number, slot], None),
        Instr('Load', [slot], loaded),
        heap_release(scope), Instr('Ret', [loaded], None),
    ])})
    artifact = emit_ssa_function_to_llvm(module, 'root')
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'caller')
    scope_buffer = np.zeros(1, dtype=np.uint64)
    execution = prepare_artifact_execution(native, {
        20: np.array([4097], dtype=np.int64), 21: scope_buffer,
    }).run()
    assert execution.buffers[26].item() == 2**60 + 3
    assert scope_buffer[0] == 0


def test_unknown_element_layout_is_not_guessed():
    with pytest.raises(KeyError, match='no dtype layout'):
        heap_allocate(SSAValue(0, 'int64'), owner(1), SSAValue(2, 'ptr'), element_dtype='unknown')


@pytest.mark.parametrize('dtype, width', [('int64', 8), ('opaque_ref', 8),
                                        ('float64', 8), ('float32', 4), ('bool', 1)])
def test_dynamic_sequence_reserve_preserves_bytes_and_owner(dtype, width, tmp_path):
    # SSA allocation-contract proof, not automatic Python object construction.
    from src.compiler.ir_sequence_tables import lower_sequence_allocate, lower_sequence_reserve
    from src.transmogrifier.ssa import SSASequenceDescriptor, SSASequenceCapacityPolicy
    descriptor = SSASequenceDescriptor(0, (10, 11), 2, 3,
        column_dtypes=(dtype, dtype), key_columns=(0,), heap_owner_value_id=6,
        capacity_policy=SSASequenceCapacityPolicy.DYNAMIC)
    initialize = lower_sequence_allocate(descriptor).functions[0]
    reserve = lower_sequence_reserve(descriptor).functions[0]
    release = Function('release', [owner(6)], {'entry': BasicBlock('entry', [
        heap_release(owner(6)), Instr('Ret', [], None)])})
    module = IRModule(functions={f.name: f for f in (initialize, reserve, release)})
    artifacts = {f.name: emit_ssa_function_to_llvm(module, f.name) for f in (initialize, reserve, release)}
    for artifact in artifacts.values():
        assert artifact.complete, artifact.shortfalls
    natives = {name: compile_artifact(artifact, directory=tmp_path / name)
               for name, artifact in artifacts.items()}
    buffers = {10: np.zeros(1, dtype=np.uint64), 11: np.zeros(1, dtype=np.uint64),
               2: np.zeros(1, dtype=np.int64), 3: np.zeros(1, dtype=np.int64),
               6: np.zeros(1, dtype=np.uint64)}
    required_id = reserve.metadata['required_capacity_value_id']
    status_id = dict(reserve.metadata['named_outputs'])['status']
    expected = {10: b'', 11: b''}
    try:
        prepare_artifact_execution(natives[initialize.name], buffers).run()
        for required in (1, 2, 7, 4097):
            old_length = int(buffers[2][0])
            old_addresses = {column: int(buffers[column][0]) for column in (10, 11)}
            buffers[required_id] = np.array([required], dtype=np.int64)
            execution = prepare_artifact_execution(natives[reserve.name], buffers).run()
            assert execution.buffers[status_id].item() == 1
            capacity = int(buffers[3][0])
            assert capacity >= required and buffers[2][0] == old_length
            for column in (10, 11):
                data = (ctypes.c_ubyte * (capacity * width)).from_address(int(buffers[column][0]))
                assert bytes(data[:old_length*width]) == expected[column]
                assert not any(data[old_length*width:])
                # Retired arenas remain valid for their owner's lifetime.
                retired = (ctypes.c_ubyte * (old_length * width)).from_address(old_addresses[column])
                assert bytes(retired) == expected[column]
                payload = bytes((position*17+column) % 256 for position in range(required*width))
                data[:required*width] = payload
                expected[column] = payload
            buffers[2][0] = required
            before = {key: value.copy() for key, value in buffers.items() if key != required_id}
            assert prepare_artifact_execution(natives[reserve.name], buffers).run().buffers[status_id].item() == 0
            for key, value in before.items():
                assert np.array_equal(buffers[key], value)
        for invalid in (-1, 2**63-1):
            buffers[required_id] = np.array([invalid], dtype=np.int64)
            before = {key: value.copy() for key, value in buffers.items() if key != required_id}
            assert prepare_artifact_execution(natives[reserve.name], buffers).run().buffers[status_id].item() == 2
            for key, value in before.items():
                assert np.array_equal(buffers[key], value)
    finally:
        prepare_artifact_execution(natives['release'], {6: buffers[6]}).run()
    assert buffers[6][0] == 0


def test_heap_copy_layout_disagreement_is_audited():
    from src.compiler.ssa_heap_allocation import heap_copy
    from src.compiler.identity_concordance import concordance_report
    source = SSAValue(1, 'ptr', accounting={'heap_element_dtype': 'int64', 'heap_owner_value_id': 7})
    destination = SSAValue(2, 'ptr', accounting={'heap_element_dtype': 'int64', 'heap_owner_value_id': 7})
    count = SSAValue(3, 'int64')
    copy = heap_copy(count, source, destination)
    destination.accounting['heap_element_dtype'] = 'float64'
    function = Function('copy', [source, destination, count],
        {'entry': BasicBlock('entry', [copy, Instr('Ret', [], None)])})
    module = IRModule(functions={'copy': function})
    artifact = emit_ssa_function_to_llvm(module, 'copy')
    assert not artifact.complete
    assert any('element layouts disagree' in finding.reason for finding in artifact.shortfalls)
    assert 'heap-copy-contract' in concordance_report(module)


def test_owned_sequence_contract_rejects_aliased_counters_and_pointee_mismatch():
    from src.compiler.sequence_storage_binding import _bind_sequence_storage_members
    from src.transmogrifier.ssa import SSASequenceDescriptor
    with pytest.raises(ValueError, match='length/capacity must be distinct'):
        SSASequenceDescriptor(0, (10,), 2, 2, column_dtypes=('int64',), heap_owner_value_id=6)
    with pytest.raises(ValueError, match='aliased heap columns disagree'):
        SSASequenceDescriptor(0, (10, 10), 2, 3,
            column_dtypes=('int64', 'float64'), heap_owner_value_id=6)
    with pytest.raises(ValueError, match='heap owner must be a distinct'):
        SSASequenceDescriptor(0, (10,), 2, 3, status_address_id=6,
            column_dtypes=('int64',), heap_owner_value_id=6)
    source = SSASequenceDescriptor(0, (10,), 2, 3,
        column_dtypes=('int64',), heap_owner_value_id=6)
    target = SSASequenceDescriptor(20, (30,), 22, 23,
        column_dtypes=('float64',), heap_owner_value_id=26)
    bindings = {}
    assert not _bind_sequence_storage_members(bindings, source, target)
    assert bindings == {}
    from dataclasses import replace
    matching = replace(target, column_dtypes=('int64',), column_shapes=((),))
    assert _bind_sequence_storage_members(bindings, source, matching)
    assert bindings == {0: 20, 10: 30, 2: 22, 3: 23, 6: 26}


def test_heap_copy_does_not_accept_an_opaque_handle_as_an_address():
    from src.compiler.ssa_heap_allocation import heap_copy
    source = SSAValue(1, 'opaque_ref')
    destination = SSAValue(2, 'ptr', accounting={'heap_element_dtype': 'int64', 'heap_owner_value_id': 7})
    with pytest.raises(ValueError, match='not opaque handles'):
        heap_copy(SSAValue(3, 'int64'), source, destination)


def test_heap_copy_has_defined_overlap_semantics(tmp_path):
    from src.compiler.ssa_heap_allocation import heap_copy
    source = SSAValue(1, 'ptr', accounting={'heap_element_dtype': 'int64', 'heap_owner_value_id': 7})
    destination = SSAValue(2, 'ptr', accounting={'heap_element_dtype': 'int64', 'heap_owner_value_id': 7})
    count = SSAValue(3, 'int64')
    function = Function('copy', [source, destination, count],
        {'entry': BasicBlock('entry', [heap_copy(count, source, destination), Instr('Ret', [], None)])})
    native = compile_artifact(emit_ssa_function_to_llvm(IRModule(functions={'copy': function}), 'copy'),
                              directory=tmp_path / 'overlap')
    module = make_allocation_module()
    allocator = compile_artifact(emit_ssa_function_to_llvm(module, 'allocate'), directory=tmp_path / 'allocate')
    destructor = compile_artifact(emit_ssa_function_to_llvm(module, 'release'), directory=tmp_path / 'release')
    scope = np.zeros(1, dtype=np.uint64)
    try:
        execution = prepare_artifact_execution(allocator, {0: np.array([4], dtype=np.int64), 1: scope}).run()
        address = int(execution.buffers[2].item())
        rows = (ctypes.c_int64 * 4).from_address(address)
        rows[:] = [1, 2, 3, 4]
        prepare_artifact_execution(native, {1: np.array([address], dtype=np.uint64),
            2: np.array([address+8], dtype=np.uint64), 3: np.array([3], dtype=np.int64)}).run()
        assert list(rows) == [1, 1, 2, 3]
    finally:
        prepare_artifact_execution(destructor, {10: scope}).run()
    assert scope[0] == 0


def test_dynamic_insert_rejects_invalid_extents_before_touching_null_columns(tmp_path):
    from src.compiler.ir_sequence_tables import lower_sequence_insert
    from src.transmogrifier.ssa import SSASequenceDescriptor, SSASequenceCapacityPolicy
    descriptor = SSASequenceDescriptor(0, (10, 11), 2, 3,
        column_dtypes=('opaque_ref', 'opaque_ref'), key_columns=(0,), heap_owner_value_id=6,
        capacity_policy=SSASequenceCapacityPolicy.DYNAMIC)
    lowering = lower_sequence_insert(descriptor)
    function = lowering.functions[0]
    module = IRModule(functions={f.name: f for f in lowering.functions})
    artifact = emit_ssa_function_to_llvm(module, function.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'invalid_extents')
    status_id = dict(function.metadata['named_outputs'])['status']
    physical_ids = {0, 10, 11, 2, 3, 6}
    rows = {value.id: np.array([17], dtype=np.int64) for value in function.args
            if value.id not in physical_ids}
    for length, capacity in ((-1, 0), (1, 0), (0, -1), (2**63-1, 2**63-1)):
        buffers = {**rows, 10: np.zeros(1, dtype=np.uint64), 11: np.zeros(1, dtype=np.uint64),
                   2: np.array([length], dtype=np.int64), 3: np.array([capacity], dtype=np.int64),
                   6: np.zeros(1, dtype=np.uint64)}
        before = {key: value.copy() for key, value in buffers.items()}
        assert prepare_artifact_execution(native, buffers).run().buffers[status_id].item() == 2
        for key, value in before.items():
            assert np.array_equal(buffers[key], value)


@pytest.mark.parametrize('dtype', ['int64', 'opaque_ref'])
@pytest.mark.parametrize('operation', ['insert', 'store'])
def test_native_insert_loop_grows_keyed_columns_without_python_dispatch(dtype, operation, tmp_path):
    from src.compiler.ir_sequence_tables import lower_sequence_insert, lower_table_lookup, lower_table_store
    from src.transmogrifier.ssa import SSASequenceDescriptor, SSASequenceCapacityPolicy
    descriptor = SSASequenceDescriptor(0, (10, 11), 2, 3, status_address_id=4,
        column_dtypes=(dtype, dtype), key_columns=(0,), heap_owner_value_id=6,
        capacity_policy=SSASequenceCapacityPolicy.DYNAMIC)
    lowering = (lower_table_store(descriptor) if operation == 'store'
                else lower_sequence_insert(descriptor))
    assert lowering.complete, lowering.shortfalls
    insert = lowering.functions[0]
    lookup = lower_table_lookup(descriptor).functions[0]
    physical_ids = {0, 10, 11, 2, 3, 4, 6}
    key_formal, value_formal = [v for v in insert.args if v.id not in physical_ids]
    query_formal = next(v for v in lookup.args if v.id not in physical_ids)
    storage = {v.id: v for v in (*insert.args, *lookup.args) if v.id in physical_ids}
    count, zero, one = SSAValue(20, 'int64'), SSAValue(21, 'int64'), SSAValue(22, 'int64')
    index, following = SSAValue(23, 'int64'), SSAValue(24, 'int64')
    predicate, base, offset = SSAValue(25, 'bool'), SSAValue(26, 'int64'), SSAValue(27, 'int64')
    number, key, value = SSAValue(28, 'int64'), SSAValue(29, dtype), SSAValue(30, dtype)
    insert_status = SSAValue(31, 'int')
    last, query, answer = SSAValue(32, 'int64'), SSAValue(33, dtype), SSAValue(34, dtype)
    bindings = {**storage, key_formal.id: key, value_formal.id: value}
    root = Function('root', [*storage.values(), count], {
        'entry': BasicBlock('entry', [
            Instr('Const', [], zero, attributes={'value': 0}),
            Instr('Const', [], one, attributes={'value': 1}),
            Instr('Const', [], base, attributes={'value': 2**60+3}),
            Instr('Const', [], offset, attributes={'value': 1000000}),
            Instr('Br', [], None, attributes={'target': 'header'})], successors=['header']),
        'header': BasicBlock('header', [
            Instr('Phi', [zero, following], index, attributes={'incoming_blocks': ('entry', 'body')}),
            Instr('Lt', [index, count], predicate),
            Instr('CondBr', [predicate], None, attributes={'true_target': 'body', 'false_target': 'exit'})],
            successors=['body', 'exit']),
        'body': BasicBlock('body', [
            Instr('Add', [base, index], number), Instr('Cast', [number], key),
            Instr('Add', [key, offset], value),
            Instr('Call', [bindings[v.id] for v in insert.args], None if operation == 'store' else insert_status,
                  attributes={'callee': insert.name}),
            Instr('Add', [index, one], following),
            Instr('Br', [], None, attributes={'target': 'header'})], successors=['header']),
        'exit': BasicBlock('exit', [
            Instr('Sub', [count, one], last), Instr('Add', [base, last], query),
            Instr('Call', [query if v.id == query_formal.id else storage[v.id] for v in lookup.args],
                  answer, attributes={'callee': lookup.name}), Instr('Ret', [answer], None)]),
    })
    release = Function('release', [owner(6)], {'entry': BasicBlock('entry', [
        heap_release(owner(6)), Instr('Ret', [], None)])})
    module = IRModule(functions={f.name: f for f in (*lowering.functions, lookup, root, release)})
    artifact = emit_ssa_function_to_llvm(module, 'root')
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'loop')
    destructor = compile_artifact(emit_ssa_function_to_llvm(module, 'release'), directory=tmp_path / 'release')
    for size in (1, 257, 4097):
        buffers = {10: np.zeros(1, dtype=np.uint64), 11: np.zeros(1, dtype=np.uint64),
                   2: np.zeros(1, dtype=np.int64), 3: np.zeros(1, dtype=np.int64),
                   4: np.zeros(1, dtype=np.int32), 6: np.zeros(1, dtype=np.uint64),
                   20: np.array([size], dtype=np.int64)}
        try:
            execution = prepare_artifact_execution(native, buffers).run()
            assert execution.buffers[answer.id].item() == 2**60+3+size-1+1000000
            assert buffers[2][0] == size
            assert size <= buffers[3][0] < 2*size
            for column, delta in ((10, 0), (11, 1000000)):
                rows = (ctypes.c_uint64 * size).from_address(int(buffers[column][0]))
                assert list(rows) == [2**60+3+i+delta for i in range(size)]
            # Re-inserting all keys must neither grow nor change the columns.
            saved = {key: value.copy() for key, value in buffers.items()}
            assert prepare_artifact_execution(native, buffers).run().buffers[answer.id].item() == execution.buffers[answer.id].item()
            for key in (10, 11, 2, 3, 6):
                assert np.array_equal(buffers[key], saved[key])
        finally:
            prepare_artifact_execution(destructor, {6: buffers[6]}).run()
        assert buffers[6][0] == 0


def test_native_extend_keeps_full_dynamic_helper_closure_and_owner(tmp_path):
    from src.compiler.ir_sequence_tables import lower_sequence_extend
    from src.transmogrifier.ssa import SSASequenceDescriptor, SSASequenceCapacityPolicy
    destination = SSASequenceDescriptor(0, (10, 11), 2, 3,
        column_dtypes=('opaque_ref', 'opaque_ref'), key_columns=(0,), heap_owner_value_id=6,
        capacity_policy=SSASequenceCapacityPolicy.DYNAMIC)
    source = SSASequenceDescriptor(100, (110, 111), 102, 103,
        column_dtypes=('opaque_ref', 'opaque_ref'), key_columns=(0,))
    lowering = lower_sequence_extend(destination, source)
    assert lowering.complete, lowering.shortfalls
    function = lowering.functions[-1]
    assert any(v.id == 6 and v.accounting.get('heap_owner') for v in function.args)
    module = IRModule(functions={f.name: f for f in lowering.functions})
    artifact = emit_ssa_function_to_llvm(module, function.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'extend')
    release = Function('release', [owner(6)], {'entry': BasicBlock('entry', [
        heap_release(owner(6)), Instr('Ret', [], None)])})
    destructor = compile_artifact(emit_ssa_function_to_llvm(
        IRModule(functions={'release': release}), 'release'), directory=tmp_path / 'release')
    size = 257
    buffers = {10: np.zeros(1, dtype=np.uint64), 11: np.zeros(1, dtype=np.uint64),
               2: np.zeros(1, dtype=np.int64), 3: np.zeros(1, dtype=np.int64),
               6: np.zeros(1, dtype=np.uint64), 102: np.array([size], dtype=np.int64),
               103: np.array([size], dtype=np.int64),
               110: np.arange(size, dtype=np.int64)+(2**60+3),
               111: np.arange(size, dtype=np.int64)+(2**60+43)}
    status = dict(function.metadata['named_outputs'])['status']
    original = {column: buffers[column].copy() for column in (110, 111)}
    try:
        for _ in range(2):
            assert prepare_artifact_execution(native, buffers).run().buffers[status].item() == 1
            assert buffers[2][0] == size
            for destination_id, source_id in ((10, 110), (11, 111)):
                rows = (ctypes.c_int64 * size).from_address(int(buffers[destination_id][0]))
                assert list(rows) == original[source_id].tolist()
                assert np.array_equal(buffers[source_id], original[source_id])
    finally:
        prepare_artifact_execution(destructor, {6: buffers[6]}).run()
    assert buffers[6][0] == 0


def test_owner_disagreement_is_a_named_emission_shortfall():
    module = make_allocation_module()
    module.functions['allocate'].blocks['entry'].instrs[0].res.accounting['heap_owner_value_id'] = 99
    artifact = emit_ssa_function_to_llvm(module, 'allocate')
    assert not artifact.complete
    assert any('disagree about allocation owner' in s.reason for s in artifact.shortfalls)


def test_existing_sequence_helpers_use_runtime_heap_columns_in_native_caller(tmp_path):
    from src.compiler.ir_sequence_tables import (
        lower_sequence_allocate, lower_sequence_insert, lower_table_lookup,
    )
    from src.transmogrifier.ssa import SSASequenceDescriptor

    descriptor = SSASequenceDescriptor(0, (10, 11), 2, 3, status_address_id=4,
        column_dtypes=('int64', 'int64'), key_columns=(0,), heap_owner_value_id=6)
    allocate = lower_sequence_allocate(descriptor).functions[0]
    insert = lower_sequence_insert(descriptor).functions[0]
    lookup = lower_table_lookup(descriptor).functions[0]
    values = {value.id: value for value in allocate.args}
    scope, capacity, length = values[6], values[3], values[2]
    status = next(value for value in lookup.args if value.id == 4)
    zero, one, last = SSAValue(50, 'int64'), SSAValue(51, 'int64'), SSAValue(52, 'int64')
    length_slot = SSAValue(53, 'ptr')
    key, value = insert.args[-2:]
    lookup_result = next(i for b in lookup.blocks.values() for i in b.instrs if i.op == 'Ret').args[0]
    answer = SSAValue(54, lookup_result.dtype)
    root = Function('root', [capacity, scope, length, status], {'entry': BasicBlock('entry', [
        Instr('Call', list(allocate.args), None, attributes={'callee': allocate.name}),
        Instr('Const', [], zero, attributes={'value': 0}),
        Instr('Const', [], one, attributes={'value': 1}),
        Instr('Sub', [capacity, one], last),
        Instr('GetElementPtr', [length, zero], length_slot),
        # The initializer zeroes every row; write at the final runtime row
        # to exercise addressing well beyond a scalar or small static frame.
        Instr('Store', [last, length_slot], None),
        Instr('Const', [], key, attributes={'value': 2**60 + 3}),
        Instr('Const', [], value, attributes={'value': 2**60 + 17}),
        Instr('Call', [values.get(v.id, v) for v in insert.args], None,
              attributes={'callee': insert.name}),
        Instr('Call', [key if v is lookup.args[-1] else values.get(v.id, v) for v in lookup.args],
              answer, attributes={'callee': lookup.name}),
        heap_release(scope), Instr('Ret', [answer], None),
    ])})
    module = IRModule(functions={f.name: f for f in (root, allocate, insert, lookup)})
    artifact = emit_ssa_function_to_llvm(module, 'root')
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'table')
    for n in (1, 7, 4097):
        scope_buffer = np.zeros(1, dtype=np.uint64)
        length_buffer = np.zeros(1, dtype=np.int64)
        status_buffer = np.zeros(1, dtype=np.int32)
        execution = prepare_artifact_execution(native, {
            3: np.array([n], dtype=np.int64), 6: scope_buffer,
            2: length_buffer, 4: status_buffer,
        }).run()
        assert execution.buffers[54].item() == 2**60 + 17
        assert length_buffer[0] == n
        assert status_buffer[0] == 1
        assert scope_buffer[0] == 0


def test_heap_owner_is_a_sequence_member_on_the_incumbent_book():
    from src.transmogrifier.ssa import SSASequenceDescriptor, SSASequenceTable
    descriptor = SSASequenceDescriptor(0, (10,), 2, 3,
        column_dtypes=('int64',), heap_owner_value_id=6)
    table = SSASequenceTable()
    table.register(descriptor)
    assert (0, ('heap_owner_value_id',)) in table.member_claims(6)
    assert descriptor.to_mapping()['heap_owner_value_id'] == 6


def test_heap_allocation_disagreement_is_visible_to_concordance():
    from src.compiler.identity_concordance import concordance_report
    module = make_allocation_module()
    module.functions['allocate'].blocks['entry'].instrs[0].res.accounting['heap_owner_value_id'] = 99
    assert 'heap-allocation-contract' in concordance_report(module)


def test_invalid_allocation_does_not_rewrite_result_identity():
    result = SSAValue(2, 'ptr', accounting={'heap_owner_value_id': 99})
    with pytest.raises(ValueError, match='disagree about allocation owner'):
        heap_allocate(SSAValue(0, 'int64'), owner(1), result, element_dtype='int64')
    assert result.accounting == {'heap_owner_value_id': 99}


def test_direct_column_insert_preserves_integer_length_cell(tmp_path):
    from src.compiler.ir_sequence_tables import lower_sequence_insert
    from src.transmogrifier.ssa import SSASequenceDescriptor
    descriptor = SSASequenceDescriptor(0, (10, 11), 2, 3,
        column_dtypes=('int64', 'int64'), key_columns=(0,))
    function = lower_sequence_insert(descriptor).functions[0]
    artifact = emit_ssa_function_to_llvm(IRModule(functions={function.name: function}), function.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'direct')
    length = np.zeros(1, dtype=np.int64)
    key_column, value_column = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
    prepare_artifact_execution(native, {
        10: key_column, 11: value_column, 2: length, 3: np.array([1], dtype=np.int64),
        function.args[-2].id: np.array([17], dtype=np.int64),
        function.args[-1].id: np.array([29], dtype=np.int64),
    }).run()
    assert length[0] == 1
    assert (key_column[0], value_column[0]) == (17, 29)
