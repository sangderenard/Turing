import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateStudioSubjectPacket,studioFlatSamplerPacket,validateStudioLightTransform} from '../examples/meowdoku/web/marble_studio_subject_renderer.js';
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],face=id=>[[0,0,0],[1,0,0],[0,1,0],id,[[0,0,1],[0,0,1],[0,0,1]],7];
const uniforms=[],scalar=(name,value,type=5124)=>uniforms.push({name,value,type,size:1});
for(const n of ['uFieldGain','uShadowEnabled','uShadowLightCount','uThermalEmission','uThermalGain','uRenderPass','uEnableEmissionDirect','uNumLights'])scalar(n,0);
const sampler=studioFlatSamplerPacket();
for(const s of sampler.textures)scalar(s.name,s.unit,s.target==='TEXTURE_2D'?35678:s.target==='TEXTURE_3D'?35679:36289);
const materials={};
for(const [i,[name,stride]]of [['pbr',16],['phong',8],['enamel',8],['texstack',16]].entries()) {
 materials[name]={layout:{floats_per_record:stride},data:Array(stride*2).fill(0)};scalar('turing_ssbo_'+name,i,35678);
}
materials.pbr.data[7]=1;materials.pbr.data[21]=1;
const packet={schema:'turing.studio-class-world-import.v1',render:{uniforms,materials,referenceCamera:{view:I},
 shaders:{vertex:'#version 300 es\nvoid main(){}',fragment:'#version 300 es\n#define PLUCK_TRANSMISSION 1\n#define PLUCK_PREMULTIPLIED_GLASS 1\nvoid main(){}'}},
 scene:{channel:'studio',ray:{pbr:[...materials.pbr.data]},subject:{id:'subject',model:I,faces:[face(0),face(1),face(1)],
 phong:{interiorCount:2,glassCount:1,bubbles:[{start:1,count:1}]}}}};
const before=structuredClone(packet),result=validateStudioSubjectPacket(packet,sampler);
assert.deepEqual(validateStudioLightTransform(I),{scale:1,areaScale:1});
assert.deepEqual(validateStudioLightTransform([0,0,-4,0,0,4,0,0,4,0,0,0,9,8,7,1]),{scale:4,areaScale:16});
for(const mutate of [m=>m[0]=-1,m=>m[0]=2,m=>m[4]=.5,m=>{m[0]=0;m[5]=0;m[10]=0;}]) {
 const invalid=[...I];mutate(invalid);assert.throws(()=>validateStudioLightTransform(invalid));
}
assert.equal(result.opaque.length,1);assert.equal(result.transparent.length,1);assert.equal(result.omittedBubbleTriangles,1);
assert.deepEqual(packet,before);assert.equal(result.opaque[0],packet.scene.subject.faces[0]);
for(const mutate of [p=>p.render.uniforms.find(u=>u.name==='uFieldGain').value=1,
 p=>p.render.uniforms.find(u=>u.name==='uShadowEnabled').value=1,
 p=>p.render.uniforms.find(u=>u.name==='uThermalEmission').value=1,
 p=>p.render.uniforms.find(u=>u.name==='uColorUv').value=0,
 p=>p.render.uniforms.push(p.render.uniforms[0]),p=>p.render.materials.phong.data.pop(),
 p=>p.scene.ray.pbr[0]=1,p=>p.scene.subject.faces[0][3]=2,p=>p.scene.subject.faces[0][4][0][0]=NaN,
 p=>p.scene.subject.phong.glassCount=0,p=>p.scene.subject.phong.bubbles[0].count=3,
 p=>p.render.referenceCamera.view=Array(16).fill(0),p=>p.render.shaders.fragment=p.render.shaders.fragment.replace('PLUCK_PREMULTIPLIED_GLASS','OTHER')]) {
 const invalid=structuredClone(packet);mutate(invalid);assert.throws(()=>validateStudioSubjectPacket(invalid,sampler));
}
for(const mutate of [s=>s.textures.pop(),s=>s.textures[0].data[0]=1,s=>s.textures[0].depth=3,s=>s.textures[0].unit=7]) {
 const invalid=structuredClone(sampler);mutate(invalid);assert.throws(()=>validateStudioSubjectPacket(packet,invalid));
}
let actual=null;
if(process.argv[2]) {
 const input=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),r=validateStudioSubjectPacket(input,sampler);
 actual={subject:input.scene.subject.id,opaque:r.opaque.length,transparent:r.transparent.length,lights:r.lightCount,materials:r.materialCount};
}
console.log(JSON.stringify({pass:true,negativeContracts:21,uniformLightAreaScaling:true,actual,browserRendering:false}));
