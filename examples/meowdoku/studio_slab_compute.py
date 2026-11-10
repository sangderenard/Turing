"""Original Studio slab-axis law compiled for temporary host BVH orchestration."""
import hashlib,json
from pathlib import Path
from examples.meowdoku.client_ray_build import SPECTRAL
from src.compiler.glsl_function_ingestion import lower_glsl_function_to_ssa
from src.compiler.ssa_webgpu_backend import emit_module
from src.compiler.ssa_wasm_backend import emit_ssa_module_to_wasm_core

def source():
    text=(SPECTRAL/'demo_pluck_gl.py').read_text(encoding='utf-8')
    start=text.index('bool slab_axis(float ro, float rd, float lo, float hi,')
    return text[start:text.index('\n}',start)+2]

def build(directory,count=256):
    src=source();result=lower_glsl_function_to_ssa(src,function_name='slab_axis')
    f=result.module.functions['slab_axis'];outputs=f.blocks['entry'].instrs[-1].args
    w=emit_module(result.module,name=f.name,outputs={f.name:outputs},count=count)
    a=emit_ssa_module_to_wasm_core(result.module,f.name,entry_name=f.name)
    if not w.complete or not a.complete:raise RuntimeError(str((w.shortfalls,a.shortfalls)))
    directory=Path(directory);directory.mkdir(exist_ok=True,parents=True)
    (directory/'studio_slab_axis.wgsl').write_bytes(w.source.encode());(directory/'studio_slab_axis.wasm').write_bytes(a.binary)
    m={'schema':'studio.scalar-law.v1','entry':'slab_axis','source':src,'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'input_order':['ro','rd','lo','hi','nt','ft'],'output_order':['valid','nt','ft'],'count':count,'artifacts':{'wasm':hashlib.sha256(a.binary).hexdigest(),'wgsl':hashlib.sha256(w.source.encode()).hexdigest()}}
    (directory/'studio_slab_axis.json').write_text(json.dumps(m,indent=2));return result,a,w
if __name__=='__main__':build('build/dot-client-ray-20261109/studio-compute');print('Original slab_axis compiled')
