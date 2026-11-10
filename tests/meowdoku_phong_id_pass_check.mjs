// Shader instrumentation/uniform transport contracts. Actual pixel visibility
// is tested separately in meowdoku_phong_id_gpu_check.html, never simulated here.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {instrumentPhongTriangleIds,copyPhongUniforms} from '../examples/meowdoku/web/marble_phong_id_pass.js';
const vertex=await readFile(new URL('../build/meowdoku/web/shaders/base_material.webgl.vert.glsl',import.meta.url),'utf8');
const rawFragment=await readFile(new URL('../build/meowdoku/web/shaders/base_material.webgl.shadow.frag.glsl',import.meta.url),'utf8');
const fragment=rawFragment.replace('#version 300 es','#version 300 es\n#define PLUCK_TRANSMISSION 1\n#define PLUCK_PREMULTIPLIED_GLASS 1');
const patched=instrumentPhongTriangleIds(vertex,fragment);
const vertexInsertion='flat out highp uint dotPrimaryTriangleId;\nvoid main() {\n'+
  'dotPrimaryTriangleId = uint(gl_VertexID / 3) + 1u;';
const fragmentInsertion='flat in highp uint dotPrimaryTriangleId;\nuniform bool dotPrimaryTriangleEnabled;\n'+
  'layout(location=1) out highp uint dotPrimaryTriangleOutput;\nvoid main() {\n'+
  'dotPrimaryTriangleOutput = dotPrimaryTriangleEnabled ? dotPrimaryTriangleId : 0u;';
assert.equal(patched.vertex.replace(vertexInsertion,'void main() {'),vertex,
  'Every original vertex expression is byte-preserved');
assert.equal(patched.fragment.replace(fragmentInsertion,'void main() {')
  .replace('layout(location=0) out vec4 FragColor;','out vec4 FragColor;'),fragment,
  'Every original material/discard/early-return expression is byte-preserved');
assert.equal((patched.fragment.match(/\bdiscard\s*;/g)||[]).length,(fragment.match(/\bdiscard\s*;/g)||[]).length);
assert(patched.fragment.indexOf('dotPrimaryTriangleOutput =')<patched.fragment.indexOf('bool front_facing'));
assert.throws(()=>instrumentPhongTriangleIds(vertex.replace('#version 300 es','#version 430 core'),fragment),/WebGL2/);
assert.throws(()=>instrumentPhongTriangleIds(patched.vertex,fragment),/unused/);
assert.throws(()=>instrumentPhongTriangleIds(vertex+'\nvoid main() {}',fragment),/one original/);
assert.throws(()=>instrumentPhongTriangleIds(vertex,fragment.replace('out vec4 FragColor;','out vec4 Other;')),/FragColor/);

// Recorded GL-type transport fixture, not a graphics or visibility emulator.
const calls=[],rows=[
  ['table',0x8B5E,0],['array',0x8DC1,4],['volume',0x8B5F,8],['shadowArray',0x8DC1,9],['thermal',0x8B5E,10],
  ['enabled',0x8B56,true],['gain',0x1406,1.2],['position',0x8B51,new Float32Array([1,2,3])],
  ['matrix',0x8B5C,new Float32Array(16)],['counter',0x1405,12],
].map(([name,type,value])=>({name,type,value,size:1}));
const gl={ACTIVE_UNIFORMS:0x8B86,FLOAT:0x1406,INT:0x1404,BOOL:0x8B56,UNSIGNED_INT:0x1405,
  FLOAT_VEC2:0x8B50,FLOAT_VEC3:0x8B51,FLOAT_VEC4:0x8B52,INT_VEC2:0x8B53,INT_VEC3:0x8B54,INT_VEC4:0x8B55,
  BOOL_VEC2:0x8B57,BOOL_VEC3:0x8B58,BOOL_VEC4:0x8B59,UNSIGNED_INT_VEC2:0x8DC6,UNSIGNED_INT_VEC3:0x8DC7,UNSIGNED_INT_VEC4:0x8DC8,
  FLOAT_MAT2:0x8B5A,FLOAT_MAT3:0x8B5B,FLOAT_MAT4:0x8B5C,FLOAT_MAT2x3:0x8B65,FLOAT_MAT2x4:0x8B66,
  FLOAT_MAT3x2:0x8B67,FLOAT_MAT3x4:0x8B68,FLOAT_MAT4x2:0x8B69,FLOAT_MAT4x3:0x8B6A,
  SAMPLER_2D:0x8B5E,SAMPLER_3D:0x8B5F,SAMPLER_CUBE:0x8B60,SAMPLER_2D_ARRAY:0x8DC1,
  getProgramParameter:()=>rows.length,getActiveUniform:(_,i)=>rows[i],
  getUniformLocation:(_,name)=>name,getUniform:(_,name)=>rows.find(row=>row.name===name).value};
for(const method of ['uniform1iv','uniform1fv','uniform1uiv','uniform3fv','uniformMatrix4fv'])
  gl[method]=(...args)=>calls.push({method,args});
copyPhongUniforms(gl,'original','instrumented');
assert.deepEqual(calls.slice(0,5).map(call=>[call.method,call.args[1]]),
  [['uniform1iv',[0]],['uniform1iv',[4]],['uniform1iv',[8]],['uniform1iv',[9]],['uniform1iv',[10]]]);
assert.deepEqual(calls[5].args,['enabled',[1]]);
assert.equal(calls[8].method,'uniformMatrix4fv');assert.equal(calls[8].args[1],false);
assert.equal(calls[9].method,'uniform1uiv');
// WebGL getUniform returns ONE array element, even for an active [0] record
// whose size is greater than one. Distinct values catch omitted later lights.
const arrayRows=[
  {name:'uLightPos[0]',type:gl.FLOAT_VEC3,size:3,values:[
    new Float32Array([1,2,3]),new Float32Array([4,5,6]),new Float32Array([7,8,9])]},
  {name:'uShadowMatrices[0]',type:gl.FLOAT_MAT4,size:2,values:[
    Float32Array.from({length:16},(_,i)=>i),Float32Array.from({length:16},(_,i)=>i+16)]},
  {name:'textures[0]',type:gl.SAMPLER_2D_ARRAY,size:2,values:[4,9]},
  {name:'weights[0]',type:gl.FLOAT,size:2,values:[.25,.75]},
  {name:'enabledLights[0]',type:gl.BOOL,size:2,values:[false,true]},
  {name:'counters[0]',type:gl.UNSIGNED_INT,size:2,values:[2,42]},
  {name:'lights[0].enabled',type:gl.INT,size:2,values:[0,1]},
];
const values=new Map(arrayRows.flatMap(row=>row.values.map((value,j)=>[row.name.replace('[0]',`[${j}]`),value])));
const arrayCalls=[],reads=[];
const arrayGl={...gl,getProgramParameter:()=>arrayRows.length,getActiveUniform:(_,i)=>arrayRows[i],
  getUniform:(_,name)=>{reads.push(name);assert(values.has(name));return values.get(name);}};
for(const method of ['uniform1iv','uniform1fv','uniform1uiv','uniform3fv','uniformMatrix4fv'])
  arrayGl[method]=(...args)=>arrayCalls.push({method,args});
copyPhongUniforms(arrayGl,'original','instrumented');
assert.deepEqual(reads,[...values.keys()]);
assert.equal(arrayCalls.length,values.size);
for(const [name,value]of values) {
  const call=arrayCalls.find(c=>c.args[0]===name);assert(call,`Missing array element ${name}`);
  if(name.startsWith('uShadowMatrices')) {
    assert.equal(call.method,'uniformMatrix4fv');assert.equal(call.args[1],false);
    assert.deepEqual(call.args[2],value);
  }else if(name.startsWith('uLightPos')) {
    assert.equal(call.method,'uniform3fv');assert.deepEqual(call.args[1],value);
  }else {
    assert.equal(call.method,name.startsWith('weights')?'uniform1fv':name.startsWith('counters')?'uniform1uiv':'uniform1iv');
    assert.deepEqual(call.args[1],[Number(value)]);
  }
}
rows[0].type=999;assert.throws(()=>copyPhongUniforms(gl,'original','instrumented'),/Unsupported/);
console.log(JSON.stringify({passed:true,scope:'original shader byte-preservation and typed uniform-copy contracts',
  actualPixelReadback:false,originalDiscardCount:(fragment.match(/\bdiscard\s*;/g)||[]).length}));
