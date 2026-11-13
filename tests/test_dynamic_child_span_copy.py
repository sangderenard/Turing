"""Runtime glyph spans are copied into the existing owned child pool."""
import ctypes
import numpy as np
import pytest
from src.compiler.ir_sequence_tables import lower_sequence_allocate,lower_record_sequence_append_with_child_copy
from src.compiler.ssa_heap_allocation import heap_release
from src.transmogrifier.ssa import SSAChildTablePoolDescriptor,SSASequenceDescriptor,SSASequenceCapacityPolicy,SSAValue,Function,BasicBlock,Instr,IRModule

def descriptors():
    p=SSAChildTablePoolDescriptor(0,(20,),21,22,23,status_value_id=24,column_dtypes=('int64',),key_columns=())
    d=SSASequenceDescriptor(1,(10,),2,3,status_address_id=4,column_dtypes=('int64',),key_columns=(),child_table_pool=p,heap_owner_value_id=6,capacity_policy=SSASequenceCapacityPolicy.DYNAMIC)
    s=SSASequenceDescriptor(30,(30,),31,32,column_dtypes=('int64',),key_columns=(),writable=False)
    return d,s

def test_reject_owned_or_overlapping_source():
    from dataclasses import replace
    d,s=descriptors()
    for changed in (replace(s,heap_owner_value_id=6),replace(s,heap_owner_value_id=66),replace(s,column_value_ids=(20,))):
        with pytest.raises(ValueError,match='disjoint borrowed'):
            lower_record_sequence_append_with_child_copy(d,changed)

def test_reject_shaped_or_sparse_source():
    from dataclasses import replace
    d,s=descriptors()
    for changed in (replace(s,column_shapes=((2,),)),replace(s,live_flags_value_id=33)):
        with pytest.raises(ValueError,match='dense scalar leaf'):
            lower_record_sequence_append_with_child_copy(d,changed)

@pytest.mark.parametrize('backend',['c','llvm'])
def test_runtime_glyph_copy(backend,tmp_path):
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm,compile_artifact,prepare_artifact_execution
    d,s=descriptors();init=lower_sequence_allocate(d).entry
    lowering=lower_record_sequence_append_with_child_copy(d,s)
    assert lowering.complete,lowering.shortfalls
    append=lowering.entry
    owner=SSAValue(6,'ptr',shape=(1,),accounting={'heap_owner':True})
    release=Function('release',[owner],{'entry':BasicBlock('entry',[heap_release(owner),Instr('Ret',[],None)])})
    module=IRModule(functions={f.name:f for f in (init,*lowering.functions,release)})
    def build(f):
        if backend=='llvm':
            a=emit_ssa_function_to_llvm(module,f.name)
            assert a.complete,a.shortfalls
            return compile_artifact(a,directory=tmp_path/f.name)
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        a=emit_ssa_to_c(module,f.name);assert a.complete,a.shortfalls
        a.compile(tmp_path/f.name);return a
    native={f.name:build(f) for f in (init,append,release)}
    def run(f,feeds):
        a=native[f.name]
        return (prepare_artifact_execution(a,feeds) if backend=='llvm' else a.prepare_execution(feeds)).run()
    b={i:np.zeros(1,dtype='uint64') for i in (6,10,20,21,24)}
    b.update({i:np.zeros(1,dtype='int64') for i in (2,3,4,22,23,31,32)})
    status=dict(append.metadata['named_outputs'])['status']
    def rows(i,n):return list((ctypes.c_int64*n).from_address(int(b[i][0]))) if n else []
    expected=[]
    try:
        run(init,b)
        for word in ('','a',chr(955)+chr(128578)+'abc','x','a much longer glyph span',chr(32066)):
            glyphs=np.array([ord(c) for c in word],dtype='int64')
            b[30]=glyphs;b[31][0]=len(glyphs);b[32][0]=len(glyphs)
            e=run(append,b);assert e.buffers[status].item()==1
            expected.append(glyphs.tolist())
            glyphs[:] = -999  # Reusing the borrowed source must not mutate snapshots.
            count=int(b[2][0]);stride=int(b[23][0])
            assert count==len(expected)
            assert rows(10,count)==list(range(count))
            lengths=rows(21,count);data=rows(20,int(b[22][0]))
            assert lengths==list(map(len,expected))
            assert [data[i*stride:i*stride+n] for i,n in enumerate(lengths)]==expected
        for count,capacity in ((-1,1),(2,1),(2**63-1,2**63-1)):
            b[30]=np.array([123],dtype='int64');b[31][0]=count;b[32][0]=capacity
            before={i:b[i].copy() for i in (6,10,20,21,24,2,3,4,22,23)}
            e=run(append,b);assert e.buffers[status].item()==2
            for i,data in before.items():np.testing.assert_array_equal(b[i],data)
        saved={i:b[i].copy() for i in (2,3,22,23)}
        try:
            b[2][0]=b[3][0]=2**63-1;b[22][0]=b[23][0]=0
            b[31][0]=b[32][0]=0
            before={i:b[i].copy() for i in (6,10,20,21,24,2,3,22,23)}
            e=run(append,b);assert e.buffers[status].item()==2
            for i,data in before.items():np.testing.assert_array_equal(b[i],data)
        finally:
            for i,data in saved.items():b[i][:]=data
    finally:run(release,{6:b[6]})
    assert b[6][0]==0
