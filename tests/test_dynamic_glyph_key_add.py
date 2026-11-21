"""Absent-only set insertion copies complete existing glyph-token spans."""
import ctypes
from dataclasses import replace
import numpy as np
import pytest
from src.compiler.ir_sequence_tables import lower_glyph_key_add, lower_sequence_contains, lower_sequence_allocate, lower_record_sequence_append_with_child_copy
from src.compiler.ssa_heap_allocation import heap_release
from src.compiler.string_table import string_token
from src.transmogrifier.ssa import SSAValue, Function, BasicBlock, Instr, IRModule
from tests.test_dynamic_child_span_copy import descriptors


def test_keyed_copy_requires_dedup_admission():
    table, source = descriptors()
    table = replace(table, key_columns=(0,))
    with pytest.raises(ValueError, match="outer key dedup"):
        lower_record_sequence_append_with_child_copy(table, source)
    with pytest.raises(ValueError, match="explicit resident query"):
        lower_sequence_contains(table)
    for changed in (replace(table, writable=False),
                    replace(table, child_table_pool=replace(table.child_table_pool, writable=False)),
                    replace(table, column_value_ids=(10,11), column_dtypes=("int64","int64"))):
        with pytest.raises(ValueError, match="owned writable single-key"):
            lower_glyph_key_add(changed, source)
    with pytest.raises(ValueError):
        lower_glyph_key_add(table, replace(source, column_value_ids=(20,)))


def test_content_key_delete_compaction_and_scalar_insert_refuse():
    from src.compiler.ir_sequence_tables import (lower_sequence_insert,
        lower_table_store, lower_table_delete, lower_table_delete_first,
        lower_sequence_delete_at, lower_child_table_delete)
    from src.transmogrifier.ssa import SSASequenceCapacityPolicy
    table, source = descriptors()
    table = replace(table, key_columns=(0,))
    for candidate in (table, replace(table, heap_owner_value_id=None,
                                    capacity_policy=SSASequenceCapacityPolicy.FIXED)):
        for lower in (lower_sequence_insert, lower_table_store,
                      lower_table_delete, lower_table_delete_first):
            with pytest.raises(ValueError, match="child-content keys or child-slot reuse"):
                lower(candidate)
        assert not lower_sequence_delete_at(candidate).complete
        with pytest.raises(ValueError, match="live/status"):
            lower_child_table_delete(candidate.child_table_pool, function_name="unsupported_delete")


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_absent_only_glyph_set_copy(backend, tmp_path):
    table, source = descriptors()
    table = replace(table, key_columns=(0,))
    init = lower_sequence_allocate(table).entry
    add = lower_glyph_key_add(table, source)
    contains = lower_sequence_contains(table, query_sequence=source)
    owner = SSAValue(6, "ptr", shape=(1,), accounting={"heap_owner": True})
    release = Function("release", [owner], {"entry": BasicBlock("entry", [heap_release(owner), Instr("Ret", [], None)])})
    functions = (init, *add.functions, *contains.functions, release)
    module = IRModule(functions={f.name:f for f in functions})
    def compile_one(function):
        if backend == "llvm":
            from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact
            emitted = emit_ssa_function_to_llvm(module, function.name)
            assert emitted.complete, emitted.shortfalls
            return compile_artifact(emitted, directory=tmp_path/function.name)
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, function.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path/function.name)
        return artifact
    artifacts = {f.name:compile_one(f) for f in (init, add.entry, contains.entry, release)}
    def run(function, feeds):
        artifact = artifacts[function.name]
        if backend == "llvm":
            from src.compiler.ssa_llvm_backend import prepare_artifact_execution
            return prepare_artifact_execution(artifact, feeds).run()
        return artifact.prepare_execution(feeds).run()
    buffers = {i:np.zeros(1,dtype=np.uint64) for i in (6,10,20,21,24)}
    buffers.update({i:np.zeros(1,dtype=np.int64) for i in (2,3,4,22,23)})
    expected = []
    def words():
        n, capacity, stride, flat = (int(buffers[i][0]) for i in (2,3,23,22))
        assert 0 <= n <= capacity and capacity*stride <= flat
        def array(member,count):
            return np.ctypeslib.as_array((ctypes.c_int64*count).from_address(int(buffers[member][0]))) if count else []
        handles, lengths, glyphs = array(10,capacity), array(21,capacity), array(20,flat)
        result = []
        for i in range(n):
            handle = int(handles[i]); assert 0 <= handle < capacity
            length = int(lengths[handle]); assert 0 <= length <= stride
            result.append(list(glyphs[handle*stride:handle*stride+length]))
        return result
    def feeds_for(data, length=None, capacity=None):
        return {**buffers,30:data,31:np.array([len(data) if length is None else length],dtype=np.int64),
                32:np.array([len(data) if capacity is None else capacity],dtype=np.int64)}
    def output(execution, function):
        return int(execution.buffers[dict(function.metadata["named_outputs"])["status"]].reshape(-1)[0])
    try:
        run(init,buffers)
        for word in ("", "é", "éx", "é", "λ"*129, "", "λ"*257, "è"):
            data = np.array([string_token(c) for c in word],dtype=np.int64)
            snapshot = {i:buf.copy() for i,buf in buffers.items()}
            old = words()
            result = output(run(add.entry,feeds_for(data)),add.entry)
            if word in expected:
                assert result == 0
                for i in (6,10,20,21,24,2,3,22,23):
                    np.testing.assert_array_equal(buffers[i],snapshot[i])
                assert words() == old
            else:
                assert result == 1
                expected.append(word)
            assert words() == [[string_token(c) for c in w] for w in expected]
            # The source buffer may be reused immediately after copied insertion.
            data[:] = string_token("?")
            assert words() == [[string_token(c) for c in w] for w in expected]
            for query in ("", "é", "éx", "éxx", "λ"*129, "missing"):
                glyphs=np.array([string_token(c) for c in query],dtype=np.int64)
                ex=run(contains.entry,feeds_for(glyphs))
                assert bool(ex.buffers[dict(contains.entry.metadata["named_outputs"])["contains"]].reshape(-1)[0]) == (query in expected)
        # Invalid source count refuses before membership dereferences payload.
        for count in (-1,2):
            snapshot=words(); length=int(buffers[2][0])
            assert output(run(add.entry,feeds_for(np.array([1],dtype=np.int64),count,1)),add.entry)==2
            assert words()==snapshot and buffers[2][0]==length
        # Logical reset/refill, not delete/compaction or slot-reuse semantics.
        buffers[2][0]=0;expected.clear()
        for word in ("new", "", "new"):
            data=np.array([string_token(c) for c in word],dtype=np.int64)
            assert output(run(add.entry,feeds_for(data)),add.entry)==(0 if word in expected else 1)
            if word not in expected:expected.append(word)
        assert words()==[[string_token(c) for c in w] for w in expected]
    finally:
        run(release,{6:buffers[6]})
    assert buffers[6][0]==0


def test_composed_add_and_copy_formals_have_independent_accounting():
    table, source = descriptors()
    lowering = lower_glyph_key_add(replace(table, key_columns=(0,)), source,
                                  function_name="accounting_probe")
    copied = next(f for f in lowering.functions if f.name == "accounting_probe__copy")
    caller_args = {v.id:v for v in lowering.entry.args}
    for formal in copied.args:
        actual = caller_args[formal.id]
        assert actual is not formal
        assert actual.accounting is not formal.accounting
        actual.accounting["scope_test_marker"] = "caller"
        assert "scope_test_marker" not in formal.accounting
