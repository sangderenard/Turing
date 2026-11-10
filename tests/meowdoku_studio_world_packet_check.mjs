// Detached authoring/render-state packet contracts, not a draw/import claim.
import assert from 'node:assert/strict';
import {studioWorldPacket,captureStudioProgramState} from '../examples/meowdoku/web/marble_studio_world_packet.js';
import {packStudioScene} from '../examples/meowdoku/web/marble_studio_scene.js';
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const face=[[0,0,0],[1,0,0],[0,1,0],0,[[0,0,1],[0,0,1],[0,0,1]],7];
const pbr=Array.from({length:16},(_,i)=>i/16);
const capture={revision:5,state:packStudioScene({channel:'studio',unitM:.02,rayOriginM:[0,0,.18],
  parts:[],pbr:{stride:16,data:pbr},subject:{id:'marble',faces:[face],model:I},
  camera:{view:I,fovDegrees:50},room:{leftX:-4,rightX:4,backZ:-4,frontZ:4,floorY:-1,ceilingY:4}})};
const resources=Object.fromEntries(['pbr','phong','enamel','texstack'].map(name=>[name,
  {table:{floats_per_record:16},data:new Float32Array(name==='pbr'?pbr:Array(16).fill(.5))}]));
const gl={LINK_STATUS:1,ACTIVE_UNIFORMS:2,SHADER_TYPE:3,VERTEX_SHADER:4,FRAGMENT_SHADER:5,
  isContextLost:()=>false,getProgramParameter:(p,type)=>type===1?true:2,
  getAttachedShaders:()=>['vertex','fragment'],getShaderParameter:s=>s==='vertex'?4:5,
  getShaderSource:s=>`#version 300 es\n// ${s}\nvoid main(){}`,
  getActiveUniform:(p,i)=>({name:i?'uView':'uExposure',type:i?35676:5126,size:1}),
  getUniformLocation:(p,name)=>name,getUniform:(p,name)=>name==='uView'?new Float32Array(I):.75};
const render=captureStudioProgramState(gl,{},resources);
assert.deepEqual(render.uniforms[1].value,I);
assert.deepEqual(render.materials.pbr.data,pbr);
const arrayRender=captureStudioProgramState({...gl,
 getProgramParameter:(p,type)=>type===1?true:1,
 getActiveUniform:()=>({name:'uLights[0]',type:35665,size:2}),
 getUniform:(p,name)=>new Float32Array(name==='uLights[0]'?[1,2,3]:[4,5,6])},{},resources);
assert.deepEqual(arrayRender.uniforms.map(x=>[x.name,x.value]),[['uLights[0]',[1,2,3]],['uLights[1]',[4,5,6]]]);
const options={classIdentity:'python:camera_software.exposure_timing.SceneCameraCoordinator',placementM:[20,0,30]};
const before=structuredClone(capture),packet=await studioWorldPacket(capture,render,options);
assert.deepEqual(capture,before);
assert.equal(packet.ready,false);
assert.equal(packet.ownership.photographicCamera,'remains-linked-until-explicitly-unhooked');
assert.deepEqual(packet.scene.subject.faces,[face]);
assert.equal(packet.render.shaderHashes.vertex.length,64);
assert(packet.remaining.some(x=>x.includes('sampler')));
render.materials.pbr.data[0]=99;capture.state.subject.faces[0][0][0]=99;
assert.equal(packet.render.materials.pbr.data[0],0);assert.equal(packet.scene.subject.faces[0][0][0],0);
await assert.rejects(()=>studioWorldPacket(before,render,options),/materials changed/);
render.materials.pbr.data[0]=0;
for(const [key,value]of [['classIdentity','invented'],['placementM',[0,NaN,0]]])
  await assert.rejects(()=>studioWorldPacket(before,render,{...options,[key]:value}));
for(const edit of [r=>r.materials.phong.data.push(1),r=>r.materials.enamel.data[0]=NaN,
  r=>r.shaders.fragment='not shader',r=>r.uniforms[0].value=NaN]){
  const invalid=structuredClone(render);edit(invalid);
  await assert.rejects(()=>studioWorldPacket(before,invalid,options));
}
assert.throws(()=>captureStudioProgramState({...gl,isContextLost:()=>true},{},resources),/live linked/);
console.log(JSON.stringify({passed:true,actualSceneImport:false,actualRendering:false,preservesExactTables:true}));
