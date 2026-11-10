"""Compile actual Studio hit_tri through common SSA to existing compute backends."""
import json,hashlib
from pathlib import Path
from examples.meowdoku.client_ray_build import SPECTRAL
from src.compiler.glsl_function_ingestion import lower_glsl_function_to_ssa
from src.compiler.ssa_webgpu_backend import emit_module
from src.compiler.ssa_wasm_backend import emit_ssa_module_to_wasm_core

def studio_source():
    text=(SPECTRAL/'demo_pluck_gl.py').read_text(encoding='utf-8')
    start=text.index('bool hit_tri(vec3 ro, vec3 rd, TriGeom t, out float hit_t)')
    end=text.index('\n}',start)+2
    return text[start:end]

def build(directory,count=256,source=None,pure_helpers=None):
    source=studio_source() if source is None else source
    result=lower_glsl_function_to_ssa(source,function_name='hit_tri',struct_fields={'TriGeom':{'v0':'vec3','e1':'vec3','e2':'vec3'}},pure_helpers=pure_helpers)
    f=result.module.functions['hit_tri'];outputs=f.blocks['entry'].instrs[-1].args
    w=emit_module(result.module,name=f.name,outputs={f.name:outputs},count=count)
    a=emit_ssa_module_to_wasm_core(result.module,f.name,entry_name=f.name)
    if not w.complete or not a.complete:raise RuntimeError(str((w.shortfalls,a.shortfalls)))
    directory=Path(directory);directory.mkdir(exist_ok=True,parents=True)
    (directory/'studio_hit_tri.wgsl').write_bytes(w.source.encode())
    (directory/'studio_hit_tri.wasm').write_bytes(a.binary)
    manifest={'pure_helpers':pure_helpers or {},'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'source_owner':'Spectral demo_pluck_gl.py hit_tri','source':source,'wasm_parameters':a.parameters,'wasm_outputs':a.output_order,'wasm_output_dtypes':a.output_dtypes,'wgsl_api':w.api.to_mapping(),'input_order':['ro_x','ro_y','ro_z','rd_x','rd_y','rd_z','t_v0_x','t_v0_y','t_v0_z','t_e1_x','t_e1_y','t_e1_z','t_e2_x','t_e2_y','t_e2_z'],'output_order':['valid','distance'],'count':count,'artifacts':{'wasm':hashlib.sha256(a.binary).hexdigest(),'wgsl':hashlib.sha256(w.source.encode()).hexdigest()}}
    (directory/'studio_hit_tri.json').write_text(json.dumps(manifest,indent=2))
    return result,a,w
if __name__=='__main__':
    build('build/dot-client-ray-20261109/studio-compute')
    print('Studio hit_tri common SSA -> WASM + WGSL built')
