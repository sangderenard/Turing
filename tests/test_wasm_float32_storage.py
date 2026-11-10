import numpy as np
from src.compiler.identity_concordance import begin_identity_book,end_identity_book,CorrelationTable
from src.compiler.ssa_wasm_backend import emit_ssa_module_to_wasm_core,prepare_wasm_core_execution
from src.transmogrifier.ssa import SSAValue,Instr,BasicBlock,Function,IRModule
from tests.compiler_provenance_inputs import publish_authored_ssa

def test_float32_indexed_span_preserves_layout_and_rounding():
    a=SSAValue(0,'float32',(4,),accounting={'physical_dtype':'float32','program_abi_storage':'span','program_abi_rank':1})
    o=SSAValue(1,'float32',(4,),accounting={'physical_dtype':'float32','program_abi_storage':'span','program_abi_rank':1})
    ins=[]
    for k in range(4):
        i=SSAValue(2+k*5,'int32');p=SSAValue(3+k*5,'ptr');v=SSAValue(4+k*5,'float32',accounting={'physical_dtype':'float32'});q=SSAValue(5+k*5,'ptr');r=SSAValue(6+k*5,'float32',accounting={'physical_dtype':'float32'})
        ins.extend([Instr('Const',[],i,attributes={'value':k}),Instr('GetElementPtr',[a,i],p),Instr('Load',[p],v),Instr('Add',[v,v],r),Instr('GetElementPtr',[o,i],q),Instr('Store',[r,q],None)])
    ins.append(Instr('Ret',[o],None));f=Function('copy_f32',[a,o],{'entry':BasicBlock('entry',ins)});m=IRModule({'copy_f32':f})
    book,token=begin_identity_book()
    try:
        publish_authored_ssa(m.functions);m.metadata['identity_book']=book
        assert not CorrelationTable.build(m).findings(m)
        artifact=emit_ssa_module_to_wasm_core(m,'copy_f32')
        assert artifact.complete,artifact.shortfalls
        assert artifact.buffer_dtypes==('float32','float32')
        values=np.array([0.1,-7.25,1e-20,16777215],dtype=np.float32)
        ex=prepare_wasm_core_execution(artifact,{0:values,1:np.zeros(4,dtype=np.float32)})
        try:
            result=ex.run().buffers[1]
            assert result.dtype==np.dtype('float32')
            np.testing.assert_array_equal(result,values+values)
        finally:ex.close()
    finally:end_identity_book(token)


def test_explicit_float32_integer_and_boolean_casts():
    source=SSAValue(0,'float32');integer=SSAValue(1,'int32');restored=SSAValue(2,'float32');truth=SSAValue(3,'bool')
    f=Function('casts',[source],{'entry':BasicBlock('entry',[Instr('Cast',[source],integer),Instr('Cast',[integer],restored),Instr('Cast',[source],truth),Instr('Ret',[integer,restored,truth],None)])})
    m=IRModule({'casts':f});book,token=begin_identity_book()
    try:
        publish_authored_ssa(m.functions);m.metadata['identity_book']=book
        artifact=emit_ssa_module_to_wasm_core(m,'casts');assert artifact.complete,artifact.shortfalls
        for value in (-2.75,0.0,7.99):
            ex=prepare_wasm_core_execution(artifact,{0:value})
            try:
                ex.run()
                assert int(ex.buffers[1][0])==int(np.float32(value))
                assert ex.buffers[2][0]==np.float32(int(np.float32(value)))
                assert bool(ex.buffers[3][0])==bool(value)
            finally:ex.close()
    finally:end_identity_book(token)
