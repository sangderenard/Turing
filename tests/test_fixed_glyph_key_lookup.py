from dataclasses import replace
import numpy as np
import pytest
from src.compiler.string_table import string_token
from src.compiler.ir_sequence_tables import lower_table_lookup, hash_index_carrier
from src.transmogrifier.ssa import SSAChildTablePoolDescriptor, SSASequenceDescriptor, SSASequenceCapacityPolicy, IRModule


def descriptors():
    pool = SSAChildTablePoolDescriptor(0, (20,), 21, 22, 23,
        column_dtypes=('int64',), key_columns=(), writable=False)
    table = SSASequenceDescriptor(1, (10, 11), 2, 3, status_address_id=4,
        column_dtypes=('int64', 'int64'), key_columns=(0,),
        child_table_pool=pool, writable=False)
    query = SSASequenceDescriptor(30, (30,), 31, 32,
        column_dtypes=('int64',), writable=False)
    return table, query


def test_reject_unsupported_glyph_key_contracts():
    d, q = descriptors()
    for changed in (replace(d, key_columns=(0,1)),
                    replace(d, key_columns=(1,)),
                    replace(d, child_table_pool=None),
                    replace(d, column_dtypes=()),
                    replace(d, child_table_pool=replace(d.child_table_pool, status_value_id=47)),
                    replace(d, live_flags_value_id=40),
                    replace(d, column_shapes=((), (2,))),
                    replace(d, capacity_policy=SSASequenceCapacityPolicy.DYNAMIC),
                    replace(d, child_table_pool=replace(d.child_table_pool, key_columns=(0,))),
                    replace(d, child_table_pool=replace(d.child_table_pool, shape_value_id=42,rank_value_id=43,shape_stride_value_id=44))):
        with pytest.raises(ValueError):
            lower_table_lookup(changed, query_sequence=q)
    with pytest.raises(ValueError, match='explicit resident query'):
        lower_table_lookup(d)
    for role in (d.child_table_pool.row_stride_value_id, d.child_table_pool.capacity_value_id,
                 d.column_value_ids[0]):
        with pytest.raises(ValueError, match='conflicting roles'):
            lower_table_lookup(d, query_sequence=replace(q, length_address_id=role))
    # Identical readonly glyph-data roles can share a raw arena; other roles
    # must remain independently declared (no dict overwrite by value ID).
    assert lower_table_lookup(d, query_sequence=replace(q,
        column_value_ids=d.child_table_pool.column_value_ids)).complete
    for changed in (replace(q, writable=True), replace(q, column_dtypes=('float64',)),
                    replace(q, key_columns=(0,)), replace(q, heap_owner_value_id=44),
                    replace(q, column_shapes=((2,),)), replace(q,live_flags_value_id=45),
                    replace(q, status_address_id=46),
                    replace(q, capacity_policy=SSASequenceCapacityPolicy.DYNAMIC)):
        with pytest.raises(ValueError):
            lower_table_lookup(d, query_sequence=changed)
    with pytest.raises(ValueError, match='handle column is outside'):
        replace(d, child_table_pool=replace(d.child_table_pool, handle_column=99))
    # A content-key child handle is not a scalar hash key, even when owned.
    # Unrelated scalar keys retain the existing hash carrier contract.
    carrier = replace(d, heap_owner_value_id=41, capacity_policy=SSASequenceCapacityPolicy.DYNAMIC)
    assert hash_index_carrier(carrier) is None
    assert lower_table_lookup(carrier, query_sequence=q).complete


@pytest.mark.parametrize('backend', ['c','llvm'])
@pytest.mark.parametrize('alias_data', [False, True])
def test_fixed_glyph_key_lookup_matches_full_contents(backend, alias_data, tmp_path):
    d, q = descriptors()
    if alias_data:
        q = replace(q, column_value_ids=d.child_table_pool.column_value_ids)
    lowered = lower_table_lookup(d, query_sequence=q)
    assert lowered.complete
    root = lowered.entry
    assert any(f.metadata.get('ssa_sequence_operation') == 'equal' for f in lowered.functions)
    assert not any('hash' in str(op.attributes.get('callee', ''))
                   for f in lowered.functions for b in f.blocks.values() for op in b.instrs)
    module = IRModule(functions={f.name:f for f in lowered.functions})
    if backend == 'llvm':
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        emitted = emit_ssa_function_to_llvm(module, root.name)
        assert emitted.complete, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path/'llvm')
        execute = lambda feeds: prepare_artifact_execution(artifact, feeds).run()
    else:
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, root.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path/'c')
        execute = lambda feeds: artifact.prepare_execution(feeds).run()
    result = dict(root.metadata['named_outputs'])['value']
    # Raw low-level fixtures explicitly establish valid handle/length extents.
    # No adversarial native pointer/handle test is authorized by this fixture.
    cases = ((['\u00e9', 'z', '\u00e9'], [1, 2, 0]),) if alias_data else (
        ([], []), ([''], [0]),
        (['', '\u00e9', '\u00e9x', '\u00e9', '\u00e8', '\u03bb'*129], [3,0,2,1,5,4]))
    for words, handles in cases:
        stride = max([1, *map(len, words)])
        assert all(0 <= handle < len(words) for handle in handles)
        assert all(0 <= len(word) <= stride for word in words)
        glyphs = np.zeros(len(words)*stride, dtype=np.int64)
        for i, word in enumerate(words):
            glyphs[i*stride:i*stride+len(word)] = [string_token(ch) for ch in word]
        assert glyphs.size == len(words)*stride
        values = np.arange(len(handles), dtype=np.int64) + 100
        queries = (words[0],) * 3 if alias_data else (
            '', '\u00e9', '\u00e9x', '\u00e9xx', '\u00e8', '\u03bb'*129, 'missing')
        for query in queries:
            feeds = {10:np.array(handles,dtype=np.int64), 11:values,
                     2:np.array([len(handles)],dtype=np.int64), 3:np.array([len(handles)],dtype=np.int64),
                     4:np.array([-1],dtype=np.int64),20:glyphs,
                     21:np.array(list(map(len,words)),dtype=np.int64),
                     22:np.array([glyphs.size],dtype=np.int64),23:np.array([stride],dtype=np.int64),
                     30:np.array([string_token(c) for c in query],dtype=np.int64),
                     31:np.array([len(query)],dtype=np.int64),32:np.array([len(query)],dtype=np.int64)}
            if alias_data:
                del feeds[30]
                assert glyphs[:len(query)].tolist() == [string_token(c) for c in query]
            matches=[i for i,handle in enumerate(handles) if words[handle] == query]
            execution=execute(feeds)
            assert int(execution.buffers[result].reshape(-1)[0]) == (100+matches[0] if matches else 0)
            assert int(execution.buffers[4].reshape(-1)[0]) == bool(matches)
