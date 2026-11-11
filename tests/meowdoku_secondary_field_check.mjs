import assert from 'node:assert/strict';
import {createStudioSecondaryField,secondaryFieldFragment,SECONDARY_FIELD_PASSES} from '../examples/meowdoku/web/marble_secondary_field.js';

function fakeGL() {
  const gl={},names=['VERTEX_ARRAY_BINDING','ARRAY_BUFFER_BINDING','STATIC_DRAW','ARRAY_BUFFER','FLOAT','TEXTURE_2D','RGBA16F',
    'DEPTH_COMPONENT24','COLOR_ATTACHMENT0','DEPTH_ATTACHMENT','FRAMEBUFFER','DRAW_FRAMEBUFFER','READ_FRAMEBUFFER',
    'TEXTURE_MIN_FILTER','TEXTURE_MAG_FILTER','TEXTURE_WRAP_S','TEXTURE_WRAP_T','NEAREST','CLAMP_TO_EDGE','FRAMEBUFFER_COMPLETE',
    'DRAW_FRAMEBUFFER_BINDING','READ_FRAMEBUFFER_BINDING','VIEWPORT','CURRENT_PROGRAM','ACTIVE_TEXTURE','TEXTURE_BINDING_2D','SAMPLER_BINDING',
    'DEPTH_FUNC','DEPTH_WRITEMASK','BLEND_SRC_RGB','BLEND_DST_RGB','BLEND_SRC_ALPHA','BLEND_DST_ALPHA','BLEND_EQUATION_RGB','BLEND_EQUATION_ALPHA',
    'DEPTH_TEST','BLEND','CULL_FACE','MAX_TEXTURE_SIZE','ACTIVE_UNIFORMS','EQUAL','LEQUAL','LESS','ONE','ZERO','ONE_MINUS_SRC_ALPHA','FUNC_ADD',
    'DEPTH_BUFFER_BIT','TRIANGLES','CURRENT_VERTEX_ATTRIB'];
  names.forEach((n,i)=>gl[n]=i+1);gl.TEXTURE0=1000;
  let id=0;const params=new Map([[gl.VIEWPORT,[0,0,100,80]],[gl.CURRENT_PROGRAM,'primary'],[gl.ACTIVE_TEXTURE,gl.TEXTURE0+3],
    [gl.DEPTH_FUNC,gl.LESS],[gl.DEPTH_WRITEMASK,true],[gl.MAX_TEXTURE_SIZE,4096]]),enabled=new Set([gl.DEPTH_TEST]),uniforms=new Map(),events=[];
  const textures=new Map(),samplers=new Map();
  gl.getParameter=n=>n===gl.TEXTURE_BINDING_2D?textures.get(params.get(gl.ACTIVE_TEXTURE))??null:n===gl.SAMPLER_BINDING?
    samplers.get(params.get(gl.ACTIVE_TEXTURE))??null:params.get(n)??null;
  gl.getUniformLocation=(p,n)=>`${p}:${n}`;gl.getUniform=(p,n)=>uniforms.get(n)??(n.endsWith('uEnableSpecular')?true:0);
  gl.uniform1i=(n,v)=>{uniforms.set(n,v);events.push(['uniform',n,v]);};gl.uniform1fv=gl.uniform1iv=gl.uniform1i;
  gl.uniformMatrix4fv=()=>{};gl.getExtension=()=>({});gl.isContextLost=()=>false;
  gl.getVertexAttrib=()=>new Float32Array([0,0,0,1]);gl.vertexAttribI4i=gl.vertexAttribI4iv=gl.vertexAttribI4uiv=gl.vertexAttrib4fv=()=>{};
  gl.getProgramParameter=()=>0;gl.isEnabled=n=>enabled.has(n);
  gl.enable=n=>enabled.add(n);gl.disable=n=>enabled.delete(n);
  for(const n of ['VertexArray','Buffer','Framebuffer','Texture']){gl[`create${n}`]=()=>`${n}${++id}`;gl[`delete${n}`]=v=>events.push(['delete',v]);}
  gl.bindVertexArray=v=>params.set(gl.VERTEX_ARRAY_BINDING,v);gl.bindBuffer=(t,v)=>params.set(gl.ARRAY_BUFFER_BINDING,v);
  gl.bufferData=gl.enableVertexAttribArray=gl.vertexAttribPointer=gl.texStorage2D=gl.texParameteri=gl.framebufferTexture2D=gl.drawBuffers=()=>{};
  gl.activeTexture=v=>params.set(gl.ACTIVE_TEXTURE,v);gl.bindTexture=(t,v)=>textures.set(params.get(gl.ACTIVE_TEXTURE),v);
  gl.bindSampler=(unit,v)=>samplers.set(gl.TEXTURE0+unit,v);
  gl.bindFramebuffer=(t,v)=>{if(t!==gl.READ_FRAMEBUFFER)params.set(gl.DRAW_FRAMEBUFFER_BINDING,v);if(t!==gl.DRAW_FRAMEBUFFER)params.set(gl.READ_FRAMEBUFFER_BINDING,v);};
  gl.checkFramebufferStatus=()=>gl.FRAMEBUFFER_COMPLETE;gl.viewport=(...v)=>params.set(gl.VIEWPORT,v);gl.useProgram=v=>params.set(gl.CURRENT_PROGRAM,v);
  gl.depthFunc=v=>params.set(gl.DEPTH_FUNC,v);gl.depthMask=v=>params.set(gl.DEPTH_WRITEMASK,v);
  gl.blendFuncSeparate=(...v)=>['BLEND_SRC_RGB','BLEND_DST_RGB','BLEND_SRC_ALPHA','BLEND_DST_ALPHA'].forEach((n,i)=>params.set(gl[n],v[i]));
  gl.blendEquation=v=>{params.set(gl.BLEND_EQUATION_RGB,v);params.set(gl.BLEND_EQUATION_ALPHA,v);};
  gl.blendEquationSeparate=(a,b)=>{params.set(gl.BLEND_EQUATION_RGB,a);params.set(gl.BLEND_EQUATION_ALPHA,b);};
  gl.drawArrays=()=>events.push(['resolve',uniforms.get('secondary:uSecondaryPass')]);gl.blitFramebuffer=()=>events.push(['blit']);
  return {gl,events,uniforms,params};
}
assert.deepEqual(SECONDARY_FIELD_PASSES,['volumetric','reflectance','glass']);
assert.match(secondaryFieldFragment('#version 300 es\nuniform int uSecondaryPass;'),/#define PLUCK_SECONDARY_FIELD 1/);
assert.throws(()=>secondaryFieldFragment('#version 300 es\nold export'),/Export the current/);
const {gl,events,uniforms,params}=fakeGL(),pipeline=createStudioSecondaryField({gl,primary:'primary',secondary:'secondary'});
pipeline.begin(100,80);assert.equal(uniforms.get('primary:uLinearOutput'),1);assert.equal(pipeline.active,true);
const draws=[],callback=(program,pass)=>{draws.push([program,pass,gl.getParameter(gl.DEPTH_FUNC),gl.getParameter(gl.DEPTH_WRITEMASK),gl.getParameter(gl.BLEND_DST_RGB)]);};
assert.deepEqual(pipeline.finish({bind:()=>true,drawOpaque:callback,drawTransparent:callback}).passes,SECONDARY_FIELD_PASSES);
assert.deepEqual(draws,[['secondary','volumetric',gl.EQUAL,false,gl.ONE],['secondary','reflectance',gl.EQUAL,false,gl.ONE],['secondary','glass',gl.LEQUAL,false,gl.ONE_MINUS_SRC_ALPHA]]);
assert.deepEqual(events.filter(x=>x[0]==='resolve'),[['resolve',3]]);assert.equal(events.some(x=>x[0]==='blit'),false);
assert.equal(gl.getParameter(gl.CURRENT_PROGRAM),'primary');assert.equal(uniforms.get('primary:uLinearOutput'),0);
assert.equal(gl.getParameter(gl.DEPTH_FUNC),gl.LESS);assert.equal(gl.getParameter(gl.DEPTH_WRITEMASK),true);
assert.equal(gl.getParameter(gl.ACTIVE_TEXTURE),gl.TEXTURE0+3);assert.equal(pipeline.active,false);
// Stale bind and callback failure restore the caller's state; no partial resolve.
pipeline.begin(100,80);assert.throws(()=>pipeline.finish({bind:()=>false,drawOpaque:callback,drawTransparent:callback}),/became stale/);
assert.equal(pipeline.active,false);assert.equal(uniforms.get('primary:uLinearOutput'),0);
pipeline.begin(100,80);assert.throws(()=>pipeline.finish({bind:()=>true,drawOpaque:()=>Promise.resolve(),drawTransparent:callback}),/synchronously/);
assert.equal(pipeline.active,false);
// A resize reallocates only owned attachments, then DOF receives opaque depth.
params.set(gl.DRAW_FRAMEBUFFER_BINDING,'dof-target');params.set(gl.READ_FRAMEBUFFER_BINDING,'dof-target');
pipeline.begin(120,90);pipeline.finish({bind:()=>true,drawOpaque:callback,drawTransparent:callback});
assert.equal(events.filter(x=>x[0]==='blit').length,1);assert.equal(gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING),'dof-target');
pipeline.begin(120,90);pipeline.cancel();assert.equal(pipeline.active,false);
assert.throws(()=>pipeline.begin(0,90),/dimensions/);pipeline.dispose();assert.throws(()=>pipeline.begin(100,80),/closed/);
console.log('Secondary field orchestration passed: same-frame order, visibility, linear resolve, resize, DOF, cancellation and state restoration.');
