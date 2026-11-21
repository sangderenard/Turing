"""Owned content-key lookup follows existing pointer cells through reserve."""
import ctypes
from dataclasses import replace
import numpy as np
import pytest
from src.compiler.ir_sequence_tables import lower_sequence_allocate, lower_sequence_reserve, lower_table_lookup, hash_index_carrier
from src.compiler.ssa_heap_allocation import heap_release
from src.compiler.string_table import string_token
from src.transmogrifier.ssa import SSAValue, Function, BasicBlock, Instr, IRModule, SSASequenceCapacityPolicy
from tests.test_fixed_glyph_key_lookup import descriptors


def test_hash_carrier_excludes_only_child_handle_keys():
    table, query = descriptors()
    owned = replace(table, writable=True, heap_owner_value_id=6, capacity_policy=SSASequenceCapacityPolicy.DYNAMIC)
    assert hash_index_carrier(owned) is None
    assert hash_index_carrier(replace(owned, key_columns=(1,))) == 1
    # Query data cannot alias an owned pointer CELL under a fixed array role.
    with pytest.raises(ValueError, match='conflicting roles'):
        lower_table_lookup(owned, query_sequence=replace(query, column_value_ids=(20,)))


@pytest.mark.parametrize('backend', ['c', 'llvm'])
def test_dynamic_glyph_probe_across_reserve_clear_and_release(backend, tmp_path):
    table, query = descriptors()
    table = replace(table, writable=True, heap_owner_value_id=6,
        capacity_policy=SSASequenceCapacityPolicy.DYNAMIC,
        child_table_pool=replace(table.child_table_pool, writable=True, status_value_id=24))
    init = lower_sequence_allocate(table).entry
    reserve_lowering = lower_sequence_reserve(table)
    reserve = reserve_lowering.entry
    lookup_lowering = lower_table_lookup(table, query_sequence=query)
    lookup = lookup_lowering.entry
    owner = SSAValue(6, 'ptr', shape=(1,), accounting={'heap_owner': True})
    release = Function('release', [owner], {'entry': BasicBlock('entry', [heap_release(owner), Instr('Ret', [], None)])})
    functions = (init, *reserve_lowering.functions, *lookup_lowering.functions, release)
    module = IRModule(functions={f.name:f for f in functions})
    def build(function):
        if backend == 'llvm':
            from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact
            emitted = emit_ssa_function_to_llvm(module, function.name)
            assert emitted.complete, emitted.shortfalls
            return compile_artifact(emitted, directory=tmp_path/function.name)
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, function.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path/function.name)
        return artifact
    artifacts = {f.name: build(f) for f in (init, reserve, lookup, release)}
    def run(function, feeds):
        artifact = artifacts[function.name]
        if backend == 'llvm':
            from src.compiler.ssa_llvm_backend import prepare_artifact_execution
            return prepare_artifact_execution(artifact, feeds).run()
        return artifact.prepare_execution(feeds).run()
    buffers = {i:np.zeros(1,dtype=np.uint64) for i in (6,10,11,20,21,24)}
    buffers.update({i:np.zeros(1,dtype=np.int64) for i in (2,3,4,22,23,31,32)})
    def array(member, count, ctype=ctypes.c_int64):
        assert count >= 0 and (count == 0 or buffers[member][0] != 0)
        return np.ctypeslib.as_array((ctype*count).from_address(int(buffers[member][0]))) if count else np.zeros(0,dtype=np.int64)
    result_id = dict(lookup.metadata['named_outputs'])['value']
    expected = []
    def check(needle, query_view=None):
        glyphs = (np.array([string_token(c) for c in needle],dtype=np.int64)
                  if query_view is None else query_view)
        assert glyphs.tolist() == [string_token(c) for c in needle]
        feeds = {**buffers,30:glyphs,31:np.array([len(glyphs)],dtype=np.int64),32:np.array([len(glyphs)],dtype=np.int64)}
        execution = run(lookup, feeds)
        matches=[i for i,w in enumerate(expected) if w == needle]
        assert int(execution.buffers[result_id].reshape(-1)[0]) == (100+matches[0] if matches else 0)
        assert int(execution.buffers[4].reshape(-1)[0]) == bool(matches)
    try:
        run(init, buffers)
        check('')
        for word in ('', '\u00e9', '\u00e9x', '\u00e9', '\u03bb'*129, None, 'new', '\u03bb'*257):
            if word is None:
                buffers[2][0]=0;expected.clear()
                check('');check('\u00e9');check('\u03bb'*129)
                continue
            needed = len(expected)+1
            feeds = {**buffers,
                reserve.metadata['required_capacity_value_id']:np.array([needed],dtype=np.int64),
                reserve.metadata['required_child_capacity_value_id']:np.array([len(word)],dtype=np.int64)}
            execution = run(reserve, feeds)
            assert int(execution.buffers[dict(reserve.metadata['named_outputs'])['status']].reshape(-1)[0]) < 2
            capacity, stride, flat = (int(buffers[i][0]) for i in (3,23,22))
            assert needed <= capacity and len(word) <= stride and capacity*stride <= flat
            # The existing reserve must preserve every previously published row.
            for i,old in enumerate(expected):
                assert array(10,capacity)[i] == i
                n = int(array(21,capacity)[i])
                assert array(20,flat)[i*stride:i*stride+n].tolist() == [string_token(c) for c in old]
            # Host fixture population, not an implementation of insertion:
            # all writes are to established allocated extents; length is last.
            row=len(expected)
            array(10,capacity)[row]=row;array(11,capacity)[row]=100+row
            array(20,flat)[row*stride:row*stride+len(word)]=[string_token(c) for c in word]
            array(21,capacity)[row]=len(word)
            array(24,capacity,ctypes.c_int32)[row]=1
            buffers[2][0]=needed
            expected.append(word)
            # Reacquire the current child pointer after every reserve. This
            # readonly query may alias payload, never its pointer cell.
            check(word, array(20,flat)[row*stride:row*stride+len(word)])
            for needle in ('', '\u00e9', '\u00e9x', '\u00e9xx', '\u03bb'*129, 'absent'):
                check(needle)
        # No deletion/compaction is claimed: reset logical visibility in this
        # low-level fixture, then read the empty table without touching payload.
        buffers[2][0]=0;expected.clear()
        check('\u00e9');check('')
    finally:
        run(release,{6:buffers[6]})
    assert buffers[6][0] == 0
