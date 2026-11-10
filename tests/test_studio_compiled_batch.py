"""Exact Studio GLSL law, called once per element inside one Wasm batch."""
import numpy as np
import pytest
from examples.meowdoku.studio_triangle_compute import studio_source
from src.compiler.glsl_function_ingestion import lower_glsl_function_to_ssa
from src.compiler.ssa_scalar_batch import add_scalar_batch
from src.compiler.ssa_wasm_backend import emit_ssa_module_to_wasm_core,prepare_wasm_core_execution
from src.compiler.identity_concordance import CorrelationTable


def source_module():
    return lower_glsl_function_to_ssa(studio_source(),function_name='hit_tri',struct_fields={'TriGeom':{'v0':'vec3','e1':'vec3','e2':'vec3'}}).module


def test_original_triangle_batch_matches_scalar_on_varying_rays():
    module=source_module();scalar=emit_ssa_module_to_wasm_core(module,'hit_tri')
    f,inputs,outputs=add_scalar_batch(module,'hit_tri',capacity=24)
    assert not CorrelationTable.build(module).findings(module)
    batch=emit_ssa_module_to_wasm_core(module,f.name)
    assert batch.complete,batch.shortfalls
    rng=np.random.default_rng(125)
    values=np.zeros((15,24),np.float32)
    values[0:2]=rng.uniform(-.25,1.25,(2,24))
    values[2]=rng.uniform(.01,3,24)
    values[5]=-rng.uniform(.2,2,24)  # Non-unit directions matter.
    values[9]=rng.uniform(.5,2,24);values[13]=rng.uniform(.5,2,24)
    feeds={0:24,**dict(zip(inputs,values)),outputs[0]:np.zeros(24,np.int32),outputs[1]:np.zeros(24,np.float32)}
    execution=prepare_wasm_core_execution(batch,feeds)
    try:
        execution.run();actual=[execution.buffers[i] for i in outputs]
    finally:execution.close()
    assert np.any(actual[0]) and np.any(actual[0]==0)
    for j in range(24):
        execution=prepare_wasm_core_execution(scalar,dict(zip(range(15),values[:,j])))
        try:
            execution.run();expected=[execution.buffers[i][0] for i in scalar.output_order]
        finally:execution.close()
        assert actual[0][j]==expected[0]
        assert actual[1][j]==expected[1]


@pytest.mark.parametrize('count',[0,-1,25])
def test_batch_count_outside_capacity_never_writes(count):
    module=source_module();f,inputs,outputs=add_scalar_batch(module,'hit_tri',capacity=24)
    artifact=emit_ssa_module_to_wasm_core(module,f.name)
    feeds={0:count,**{i:np.zeros(24,np.float32) for i in inputs},outputs[0]:np.full(24,71,np.int32),outputs[1]:np.full(24,13,np.float32)}
    execution=prepare_wasm_core_execution(artifact,feeds)
    try:
        execution.run()
        assert np.all(execution.buffers[outputs[0]]==71)
        assert np.all(execution.buffers[outputs[1]]==13)
    finally:execution.close()

def test_shared_scalar_arguments_are_not_repacked_per_ray():
    module=source_module()
    shared=tuple(range(6,15))
    f,inputs,outputs=add_scalar_batch(module,'hit_tri',capacity=8,shared_argument_ids=shared)
    artifact=emit_ssa_module_to_wasm_core(module,f.name)
    assert artifact.complete,artifact.shortfalls
    assert not CorrelationTable.build(module).findings(module)
    feeds={0:8,**{i:np.zeros(8,np.float32) for i in inputs[:6]},**{i:0.0 for i in inputs[6:]},outputs[0]:np.zeros(8,np.int32),outputs[1]:np.zeros(8,np.float32)}
    feeds[inputs[2]]=np.arange(1,9,dtype=np.float32)
    feeds[inputs[5]]=np.full(8,-2,np.float32)
    feeds[inputs[9]]=1.;feeds[inputs[13]]=1.
    execution=prepare_wasm_core_execution(artifact,feeds)
    try:
        execution.run()
        np.testing.assert_array_equal(execution.buffers[outputs[0]],np.ones(8,np.int32))
        np.testing.assert_array_equal(execution.buffers[outputs[1]],np.arange(1,9,dtype=np.float32)/2)
    finally:execution.close()

def test_shared_float32_span_is_one_buffer_for_whole_batch():
    from src.compiler.identity_concordance import begin_identity_book,end_identity_book
    from src.transmogrifier.ssa import SSAValue,Instr,BasicBlock,Function,IRModule
    from tests.compiler_provenance_inputs import publish_authored_ssa
    i=SSAValue(0,'int32');a=SSAValue(1,'float32',(4,),accounting={'physical_dtype':'float32','program_abi_storage':'span','program_abi_rank':1})
    p=SSAValue(2,'ptr');v=SSAValue(3,'float32',accounting={'physical_dtype':'float32'})
    callee=Function('lookup',[i,a],{'entry':BasicBlock('entry',[Instr('GetElementPtr',[a,i],p),Instr('Load',[p],v),Instr('Ret',[v],None)])})
    module=IRModule({'lookup':callee});book,token=begin_identity_book()
    try:
        publish_authored_ssa(module.functions);module.metadata['identity_book']=book
        f,inputs,outputs=add_scalar_batch(module,'lookup',capacity=8,shared_argument_ids=(1,))
        artifact=emit_ssa_module_to_wasm_core(module,f.name)
        assert artifact.complete,artifact.shortfalls
        assert not CorrelationTable.build(module).findings(module)
        indexes=np.array([0,3,1,2,1,3,0,2],np.int32);data=np.array([.1,2.7,-3.5,9.1],np.float32)
        execution=prepare_wasm_core_execution(artifact,{0:8,inputs[0]:indexes,inputs[1]:data,outputs[0]:np.zeros(8,np.float32)})
        try:
            execution.run();np.testing.assert_array_equal(execution.buffers[outputs[0]],data[indexes])
        finally:execution.close()
    finally:end_identity_book(token)


def test_built_slab_batch_inout_and_artifact_hashes(tmp_path):
    import hashlib,json
    from examples.meowdoku.studio_batch_compute import build
    from examples.meowdoku.studio_bvh_compute import query_identity
    from src.compiler.ssa_wasm_backend import WasmCoreArtifact
    records=build(tmp_path,capacity=8)
    for name,record in records.items():
        assert record['source']==query_identity()[1]['laws'][name]
        for extension in ('wasm','wgsl'):
            assert hashlib.sha256((tmp_path/f'studio_{name}_batch.{extension}').read_bytes()).hexdigest()==record['artifacts'][extension]
    source=records['slab_axis']['source']
    module=lower_glsl_function_to_ssa(source,function_name='slab_axis').module
    f,inputs,outputs=add_scalar_batch(module,'slab_axis',capacity=8)
    artifact=emit_ssa_module_to_wasm_core(module,f.name)
    # Three items in an eight-element capacity. Tail sentinels must survive.
    values=np.zeros((6,8),np.float32)
    values[:,:3]=np.array([[0,1,2,4,.01,10],[0,0,-1,1,.25,9],[2,0,-1,1,.25,9]],np.float32).T
    feeds={0:3,**dict(zip(inputs,values)),outputs[0]:np.full(8,71,np.int32),outputs[1]:np.full(8,13,np.float32),outputs[2]:np.full(8,17,np.float32)}
    execution=prepare_wasm_core_execution(artifact,feeds)
    try:
        execution.run()
        np.testing.assert_array_equal(execution.buffers[outputs[0]],np.array([1,1,0,71,71,71,71,71]))
        np.testing.assert_array_equal(execution.buffers[outputs[1]],np.array([2,.25,.25,13,13,13,13,13],np.float32))
        np.testing.assert_array_equal(execution.buffers[outputs[2]],np.array([4,9,9,17,17,17,17,17],np.float32))
    finally:execution.close()


def test_original_full_traversal_single_batch_varying_rays(tmp_path):
    from examples.meowdoku.studio_traversal_compute import build
    module,artifact,payload=build(tmp_path,node_capacity=3,triangle_capacity=2,batch_capacity=4)
    assert artifact.complete and not CorrelationTable.build(module).findings(module)
    nodes=np.zeros((3,12),np.float32)
    nodes[:,0:3]=(-1,-1,1);nodes[:,4:7]=(1,1,1)
    nodes[0,3]=1;nodes[0,7]=2;nodes[1:,3]=-1;nodes[1:,9]=1;nodes[2,8]=1
    geom=np.zeros((2,16),np.float32)
    geom[:,:3]=(-1,-1,1);geom[:,4:7]=(2,0,0);geom[:,8:11]=(0,2,0)
    named={'ro_x':np.array([0,4,0,0],np.float32),'ro_y':np.array([0,4,0,0],np.float32),'ro_z':np.array([0,0,2,0],np.float32),
           'rd_x':np.zeros(4,np.float32),'rd_y':np.zeros(4,np.float32),'rd_z':np.array([1,1,-1,2],np.float32),
           'uNodeCount':3,'nodes':nodes.ravel(),'tri_ids':np.array([0,1],np.int32),'geom_tris':geom.ravel()}
    ids,distance=payload['output_ids']
    feeds={0:4,**{i:named[n] for i,n in zip(payload['input_ids'],payload['argument_names'])},ids:np.zeros(4,np.int32),distance:np.zeros(4,np.float32)}
    ex=prepare_wasm_core_execution(artifact,feeds)
    try:
        for _ in range(2):
            ex.run()  # one exported execution for all rays, including full BVH
            np.testing.assert_array_equal(ex.buffers[ids],np.array([1,-1,1,1]))
            np.testing.assert_array_equal(ex.buffers[distance],np.array([1,1e30,1,.5],np.float32))
    finally:ex.close()
