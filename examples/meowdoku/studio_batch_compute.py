"""Build single-execution candidate batches, retaining original GLSL law identity.

This removes host calls per candidate. BVH traversal is not compiled here.
Wasm is a compiled scalar loop; WGSL is a parallel invocation per candidate.
"""
import hashlib
import json
from pathlib import Path
from examples.meowdoku.studio_triangle_compute import studio_source
from examples.meowdoku.studio_slab_compute import source as slab_source
from examples.meowdoku.studio_region_compute import region_sources
from examples.meowdoku.studio_bvh_compute import query_identity as plain_query_identity
from src.compiler.glsl_function_ingestion import lower_glsl_function_to_ssa
from src.compiler.ssa_scalar_batch import add_scalar_batch
from src.compiler.ssa_wasm_backend import emit_ssa_module_to_wasm_core
from src.compiler.ssa_webgpu_backend import emit_module
from src.compiler.identity_concordance import CorrelationTable


def build(directory, *, capacity=16384, region=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    laws=region_sources() if region else plain_query_identity()[1]['laws']
    records={}
    for name in ('hit_tri','slab_axis'):
        options={}
        if name=='hit_tri':
            options['struct_fields']={'TriGeom':{'v0':'vec3','e1':'vec3','e2':'vec3'}}
            if region:options['pure_helpers']={'studio_region_epsilon':laws['studio_region_epsilon']}
        module=lower_glsl_function_to_ssa(laws[name],function_name=name,**options).module
        scalar=module.functions[name]
        gpu=emit_module(module,name=name,outputs={name:scalar.blocks['entry'].instrs[-1].args},count=capacity)
        wrapper,inputs,outputs=add_scalar_batch(module,name,capacity=capacity)
        findings=CorrelationTable.build(module).findings(module)
        if findings:raise RuntimeError(findings)
        wasm=emit_ssa_module_to_wasm_core(module,wrapper.name)
        if not wasm.complete or not gpu.complete:raise RuntimeError((wasm.shortfalls,gpu.shortfalls))
        stem='studio_'+name+'_batch'
        (directory/(stem+'.wasm')).write_bytes(wasm.binary)
        (directory/(stem+'.wgsl')).write_bytes(gpu.source.encode())
        m={'schema':'studio.compiled-law-batch.v1','entry':wrapper.name,'capacity':capacity,
           'wasm_parameters':wasm.parameters,'input_ids':inputs,'output_ids':outputs,
           'buffer_order':wasm.buffer_order,'buffer_dtypes':wasm.buffer_dtypes,
           'reserved_bytes':wasm.reserved_bytes,'wgsl_api':gpu.api.to_mapping(),
           'source':laws[name],'pure_helpers':options.get('pure_helpers',{}),
           'source_sha256':hashlib.sha256(laws[name].encode()).hexdigest(),
           'scope':'candidate-law-only; BVH traversal remains host-managed',
           'wasm_parallelism':'single compiled scalar loop; no SIMD claim',
           'artifacts':{'wasm':hashlib.sha256(wasm.binary).hexdigest(),'wgsl':hashlib.sha256(gpu.source.encode()).hexdigest()}}
        (directory/(stem+'.json')).write_text(json.dumps(m,indent=2),encoding='utf-8')
        records[name]=m
    return records

if __name__=='__main__':
    root=Path('build/dot-client-ray-20261109')
    for directory,region in [('studio-compute',False),('studio-region-compute',True)]:
        build(root/directory,region=region)
        print('Built original single-entry candidate batches:',directory,flush=True)
