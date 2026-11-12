import assert from 'node:assert/strict';
import {hydrated} from './meowdoku_studio_scene_asset_check.mjs';
import {createStudioSceneRenderer} from '../examples/meowdoku/web/marble_studio_scene_asset.js';
// Actual scene and subject renderers, instrumented WebGL commands. This proves
// attribute bytes, sorting, allocation and state ownership, not GPU pixels.
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const asset=structuredClone(hydrated);
for(const name of ['uMV','uMVP'])asset.packet.render.uniforms.push({name,type:35676,size:1,value:I});
function recorder(){
 const constants={INT:5124,SAMPLER_2D:35678,SAMPLER_2D_ARRAY:36289,SAMPLER_3D:35679,FLOAT_MAT4:35676};
 let serial=50000,resource=0,array=null,vao=null;const buffers=new Map(),vaos=new Map(),matrices=new Map(),calls=[],writes=[];
 const api={buffers,calls,writes,isContextLost:()=>false,
  getParameter:k=>k===gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS?16:k===gl.MAX_TEXTURE_SIZE?4096:k===gl.ARRAY_BUFFER_BINDING?array:
   k===gl.VERTEX_ARRAY_BINDING?vao:k===gl.DEPTH_RANGE?[0,1]:k===gl.COLOR_WRITEMASK?[true,true,true,true]:0,
  isEnabled:()=>false,getVertexAttrib:()=>new Float32Array(4),
  getShaderParameter:()=>true,getProgramParameter:(_p,k)=>k===gl.LINK_STATUS?true:k===gl.ACTIVE_UNIFORMS?asset.packet.render.uniforms.length:0,
  getActiveUniform:(_p,i)=>asset.packet.render.uniforms[i],getUniformLocation:(_p,n)=>n,
  createBuffer:()=>{const b=++resource;buffers.set(b,null);return b;},
  createVertexArray:()=>{const v=++resource;vaos.set(v,{attributes:{},index:null});return v;},
  bindVertexArray:v=>{vao=v;},bindBuffer:(target,b)=>{if(target===gl.ARRAY_BUFFER)array=b;else if(target===gl.ELEMENT_ARRAY_BUFFER)vaos.get(vao).index=b;},
  bufferData:(target,data)=>{const b=target===gl.ARRAY_BUFFER?array:vaos.get(vao).index;buffers.set(b,new data.constructor(data));},
  bufferSubData:(target,offset,data)=>{assert.equal(target,gl.ARRAY_BUFFER);buffers.get(array).set(data,offset/data.BYTES_PER_ELEMENT);writes.push({buffer:array,offset,data:Array.from(data)});},
  vertexAttribPointer:i=>{vaos.get(vao).attributes[i]=array;},vertexAttribIPointer:i=>{vaos.get(vao).attributes[i]=array;},
  uniformMatrix4fv:(name,_t,value)=>matrices.set(name,Array.from(value)),
  drawArrays:(_mode,_first,count)=>record(count,false),drawElements:(_mode,count)=>record(count,true),
 };
 function record(count,indexed){const v=vaos.get(vao);calls.push({count,modelView:matrices.get('uMV'),attributes:Object.fromEntries(Object.entries(v.attributes).map(([k,b])=>[k,Array.from(buffers.get(b))])),indices:indexed?Array.from(buffers.get(v.index)):null});}
 const gl=new Proxy(api,{get:(o,k)=>k in o?o[k]:/^[A-Z_0-9]+$/.test(k)?constants[k]??(constants[k]=serial++):k.startsWith('create')?()=>++resource:()=>{}});
 return gl;
}
const gl=recorder(),renderer=createStudioSceneRenderer({gl,asset});
renderer.draw({view:I,projection:I});const initial=structuredClone(gl.calls),bufferCount=gl.buffers.size;
const pose=[0,0,-.5,0,0,.5,0,0,.5,0,0,0,2,3,-5,1];
const foreign=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,foreign);
assert(renderer.setSubjectPose(pose));assert.equal(gl.getParameter(gl.ARRAY_BUFFER_BINDING),foreign);
assert.equal(gl.buffers.size,bufferCount+1,'no buffers rebuilt for pose');
assert.equal(gl.writes.length,2,'only subject position/normal ranges uploaded');
assert(gl.writes.every(w=>w.offset===36&&w.data.length===9),'static furniture prefix untouched');
assert.equal(renderer.setSubjectPose(pose),false);assert.equal(gl.writes.length,2,'unchanged pose is no-op');
gl.calls.length=0;renderer.draw({view:I,projection:I,phase:'opaque'});renderer.draw({view:I,projection:I,phase:'transparent'});
const moved=structuredClone(gl.calls),freshAsset=structuredClone(asset);freshAsset.packet.scene.subject.model=pose;
const freshGl=recorder(),fresh=createStudioSceneRenderer({gl:freshGl,asset:freshAsset});fresh.draw({view:I,projection:I});
assert.deepEqual(renderer.subjectBounds,fresh.subjectBounds);assert.deepEqual(renderer.bounds,fresh.bounds);
assert.deepEqual(moved,freshGl.calls,'streamed bytes/model/sort must equal full native plan at same pose');
assert.deepEqual(moved.slice(0,2),initial.slice(0,2),'static and fixture passes remain identical');
assert.notDeepEqual(moved.at(-1).indices,initial.at(-1).indices,'fixed camera must resort moving subject across furniture');
assert.deepEqual(moved.at(-1).attributes[2],initial.at(-1).attributes[2],'material IDs preserved');
assert.deepEqual(moved.at(-1).attributes[4],initial.at(-1).attributes[4],'group IDs preserved');
assert.throws(()=>renderer.setSubjectPose(Array(16).fill(NaN)));assert.throws(()=>renderer.setSubjectPose([2,...I.slice(1)]));
const overflow=[...I];overflow[12]=1e100;assert.throws(()=>renderer.validateSubjectPose(overflow),/finite/);assert.throws(()=>renderer.setSubjectPose(overflow),/finite/);
assert.equal(gl.writes.length,2,'invalid pose cannot write buffers');
renderer.dispose();fresh.dispose();assert.throws(()=>renderer.setSubjectPose(pose),/closed/);
console.log(JSON.stringify({pass:true,actualRenderers:true,sharedTransparentRange:true,fixedCameraResort:true,noMeshRebuild:true,gpuPixelsTested:false}));
