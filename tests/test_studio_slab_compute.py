import json, shutil, subprocess
import pytest
from examples.meowdoku.studio_slab_compute import build

@pytest.mark.skipif(shutil.which('node') is None, reason='Node required')
def test_original_slab_preserves_inout_and_parallel_guard(tmp_path):
    _,a,w=build(tmp_path)
    assert a.complete and w.complete
    cases=[([0,1,2,4,.01,10],[1,2,4]),([0,-1,-4,-2,.01,10],[1,2,4]),([0,0,-1,1,.25,9],[1,.25,9]),([2,0,-1,1,.25,9],[0,.25,9]),([0,1,2,4,.01,1],[0,2,1]),([1,1e-12,-1,1,.25,9],[1,.25,9])]
    script="""const fs=require('fs');WebAssembly.instantiate(fs.readFileSync(process.argv[1])).then(({instance:i})=>{let out=[];for(const c of JSON.parse(process.argv[2])){i.exports.slab_axis(...c[0],16,24,32);out.push([Number(new BigInt64Array(i.exports.memory.buffer,16,1)[0]),new Float64Array(i.exports.memory.buffer,24,1)[0],new Float64Array(i.exports.memory.buffer,32,1)[0]]);}console.log(JSON.stringify(out));});"""
    values=json.loads(subprocess.check_output(['node','-e',script,str(tmp_path/'studio_slab_axis.wasm'),json.dumps(cases)],text=True))
    for value,case in zip(values,cases):
        assert value==pytest.approx(case[1],abs=1e-7)

@pytest.mark.skipif(shutil.which('node') is None, reason='Node required')
def test_host_bvh_right_first_ties_and_validation(tmp_path):
    from pathlib import Path
    from examples.meowdoku.studio_bvh_compute import build as build_bvh
    build_bvh(tmp_path)
    runtime=(Path(__file__).resolve().parents[1]/'examples/meowdoku/web/studio_bvh_compute.js').as_uri()
    script=tmp_path/'test.mjs'
    script.write_text('''import {readFile} from 'node:fs/promises';
import {StudioBvhCompute} from %s;
globalThis.fetch=async u=>{const b=await readFile(new URL(u));return {ok:true,arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)}};
const q=await StudioBvhCompute.create(new URL('./',import.meta.url),{backend:'wasm'});
const tri=owner=>({owner,v0:[0,0,0],e1:[1,0,0],e2:[0,1,0]});
const leaf=start=>[-1,-1,-.001,-1,2,2,.001,-1,start,1,0,0];
const scene={revision:'v1',triangles:[tri(5),tri(9)],bvh_epsilon:1e-7,bvh_ids:[5,9],bvh_nodes:[[-1,-1,-.001,1,2,2,.001,2,0,0,0,0],leaf(0),leaf(1)]};
q.setScene(scene);
const r=await q.query(new Float32Array([.25,.25,-1,0,0,0,1,10, 4,4,-1,0,0,0,1,10]),'v1');
if(JSON.stringify(Array.from(r.hits))!==JSON.stringify([1,9,1,9,0,-1,0,-1]))throw Error('Tie/miss mismatch '+r.hits);
for(const bad of [{...scene,bvh_ids:[5,5]},{...scene,bvh_epsilon:1e-6},{...scene,bvh_nodes:[scene.bvh_nodes[0]]}]){let rejected=false;try{q.setScene(bad)}catch{rejected=true}if(!rejected)throw Error('Invalid scene admitted');}
let stale=false;try{await q.query(new Float32Array([0,0,0,0,0,0,1,10]),'bad')}catch{stale=true}if(!stale)throw Error('Stale accepted');
q.close();console.log('pass');
''' % json.dumps(runtime),encoding='utf-8')
    assert subprocess.check_output(['node',str(script)],text=True).strip()=='pass'

@pytest.mark.skipif(shutil.which('node') is None, reason='Node required')
def test_region_original_epsilon_changes_small_feature_threshold(tmp_path):
    from examples.meowdoku.studio_region_compute import region_sources
    from examples.meowdoku.studio_triangle_compute import build as triangle_build
    laws=region_sources()
    triangle_build(tmp_path,source=laws['hit_tri'],pure_helpers={'studio_region_epsilon':laws['studio_region_epsilon']})
    cases=[([.25,.25,-5e-7],[0,0,1],[0,0,0],[1,0,0],[0,1,0],1),([.25,.25,-1e-8],[0,0,1],[0,0,0],[1,0,0],[0,1,0],0),([.25,.25,-5e-7],[0,0,10],[0,0,0],[1,0,0],[0,1,0],1)]
    script="""const fs=require('fs');WebAssembly.instantiate(fs.readFileSync(process.argv[1])).then(({instance:i})=>{let out=[];for(const c of JSON.parse(process.argv[2])){i.exports.hit_tri(...c.slice(0,5).flat(),16,24);out.push(Number(new BigInt64Array(i.exports.memory.buffer,16,1)[0]));}console.log(JSON.stringify(out));});"""
    assert json.loads(subprocess.check_output(['node','-e',script,str(tmp_path/'studio_hit_tri.wasm'),json.dumps(cases)],text=True))==[c[-1] for c in cases]

@pytest.mark.parametrize('helper',[
 'float helper(float y){return x;}',
 'float helper(float y){return helper(y);}',
 'float helper(float y){float z=y;return z;}',
 'float helper(float y){return float(1/2);}',
])
def test_helper_ingestion_fails_closed(helper):
    from src.compiler.glsl_function_ingestion import lower_glsl_function_to_ssa
    with pytest.raises(ValueError):
        lower_glsl_function_to_ssa('float main(float x){return helper(x);}',function_name='main',pure_helpers={'helper':helper})
