import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createStudioSecondaryField,secondaryFieldFragment,SECONDARY_FIELD_PASSES,SECONDARY_TARGET_BUDGET,secondaryTargetBytes} from '../examples/meowdoku/web/marble_secondary_field.js';
import {responseApplicationTexture,currentResponseQuery,matchResponseQuery} from '../examples/meowdoku/web/marble_response_application.js';

function fakeGL() {
  const gl={},names=['VERTEX_ARRAY_BINDING','ARRAY_BUFFER_BINDING','STATIC_DRAW','ARRAY_BUFFER','FLOAT','TEXTURE_2D','RGBA32F',
    'DEPTH_COMPONENT24','COLOR_ATTACHMENT0','DEPTH_ATTACHMENT','FRAMEBUFFER','DRAW_FRAMEBUFFER','READ_FRAMEBUFFER',
    'TEXTURE_MIN_FILTER','TEXTURE_MAG_FILTER','TEXTURE_WRAP_S','TEXTURE_WRAP_T','NEAREST','CLAMP_TO_EDGE','FRAMEBUFFER_COMPLETE',
    'DRAW_FRAMEBUFFER_BINDING','READ_FRAMEBUFFER_BINDING','VIEWPORT','CURRENT_PROGRAM','ACTIVE_TEXTURE','TEXTURE_BINDING_2D','SAMPLER_BINDING',
    'DEPTH_FUNC','DEPTH_WRITEMASK','BLEND_SRC_RGB','BLEND_DST_RGB','BLEND_SRC_ALPHA','BLEND_DST_ALPHA','BLEND_EQUATION_RGB','BLEND_EQUATION_ALPHA',
    'DEPTH_TEST','BLEND','CULL_FACE','MAX_TEXTURE_SIZE','ACTIVE_UNIFORMS','EQUAL','LEQUAL','LESS','ONE','ZERO','ONE_MINUS_SRC_ALPHA','FUNC_ADD',
    'DEPTH_BUFFER_BIT','TRIANGLES','CURRENT_VERTEX_ATTRIB','COLOR','RG32UI','COLOR_ATTACHMENT1'];
  names.forEach((n,i)=>gl[n]=i+1);gl.TEXTURE0=1000;
  let id=0;const params=new Map([[gl.VIEWPORT,[0,0,100,80]],[gl.CURRENT_PROGRAM,'primary'],[gl.ACTIVE_TEXTURE,gl.TEXTURE0+3],
    [gl.DEPTH_FUNC,gl.LESS],[gl.DEPTH_WRITEMASK,true],[gl.MAX_TEXTURE_SIZE,4096]]),enabled=new Set([gl.DEPTH_TEST]),uniforms=new Map(),events=[];
  const textures=new Map(),samplers=new Map();
  gl.getParameter=n=>n===gl.TEXTURE_BINDING_2D?textures.get(params.get(gl.ACTIVE_TEXTURE))??null:n===gl.SAMPLER_BINDING?
    samplers.get(params.get(gl.ACTIVE_TEXTURE))??null:params.get(n)??null;
  gl.getUniformLocation=(p,n)=>`${p}:${n}`;gl.getUniform=(p,n)=>uniforms.get(n)??(n.endsWith('uEnableSpecular')?true:0);
  gl.uniform1i=(n,v)=>{uniforms.set(n,v);events.push(['uniform',n,v]);};gl.uniform1fv=gl.uniform1iv=gl.uniform1i;
  gl.uniformMatrix4fv=()=>{};gl.getExtension=()=>({});gl.isContextLost=()=>false;
  gl.getVertexAttrib=()=>new Float32Array([0,0,0,1]);gl.vertexAttribI4i=gl.vertexAttribI4ui=gl.vertexAttribI4iv=gl.vertexAttribI4uiv=gl.vertexAttrib4fv=()=>{};
  gl.clearBufferfv=(...args)=>events.push(['clear-volume',...args]);
  gl.clearBufferuiv=(...args)=>events.push(['clear-recipient',...args]);
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
uniforms.set('secondary:uEnableSpecular',0);
pipeline.begin(100,80);assert.equal(uniforms.get('primary:uLinearOutput'),1);assert.equal(pipeline.active,true);
const draws=[],targets=[],callback=(program,pass)=>{targets.push(gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING));
 draws.push([program,pass,gl.getParameter(gl.DEPTH_FUNC),gl.getParameter(gl.DEPTH_WRITEMASK),gl.isEnabled(gl.BLEND),gl.getParameter(gl.BLEND_DST_RGB)]);};
assert.deepEqual(pipeline.finish({bind:()=>true,drawOpaque:callback,drawTransparent:callback}).passes,SECONDARY_FIELD_PASSES);
assert.equal(uniforms.get('secondary:uEnableSpecular'),0,'application orchestration must not override copied ordinary material controls');
assert.deepEqual(draws,[['secondary','volumetric',gl.EQUAL,false,false,null],['secondary','volumetric',gl.LEQUAL,true,false,null],
 ['secondary','reflectance',gl.EQUAL,false,true,gl.ONE_MINUS_SRC_ALPHA],['secondary','glass',gl.LEQUAL,false,true,gl.ONE_MINUS_SRC_ALPHA]]);
assert.deepEqual(events.filter(x=>x[0]==='resolve'),[['resolve',3]]);assert.equal(events.filter(x=>x[0]==='blit').length,1,'volume target receives opaque depth separately');
assert.equal(events.filter(x=>x[0]==='clear-volume').length,1);
assert.equal(events.filter(x=>x[0]==='clear-recipient').length,1);
assert.equal(targets[0],targets[1]);assert.equal(targets[2],targets[3]);assert.notEqual(targets[0],targets[2],
 'pre-first-hit volume cannot be erased by downstream alpha replacement');
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
const previousBlits=events.filter(x=>x[0]==='blit').length;
pipeline.begin(120,90);pipeline.finish({bind:()=>true,drawOpaque:callback,drawTransparent:callback});
assert.equal(events.filter(x=>x[0]==='blit').length,previousBlits+2);assert.equal(gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING),'dof-target');
pipeline.begin(120,90);pipeline.cancel();assert.equal(pipeline.active,false);
assert.throws(()=>pipeline.begin(0,90),/dimensions/);pipeline.dispose();assert.throws(()=>pipeline.begin(100,80),/closed/);
const unsupported=fakeGL().gl;unsupported.getExtension=name=>name==='EXT_float_blend'?null:{};
assert.throws(()=>createStudioSecondaryField({gl:unsupported,primary:'primary',secondary:'secondary'}),/float32 response blending/);
assert(secondaryTargetBytes(1920,1080)<=SECONDARY_TARGET_BUDGET);
assert(secondaryTargetBytes(3840,2160)>SECONDARY_TARGET_BUDGET);
const bounded=fakeGL().gl,boundedPipeline=createStudioSecondaryField({gl:bounded,primary:'primary',secondary:'secondary'});
bounded.createTexture=()=>{throw Error('Must not allocate beyond admitted budget');};
assert.throws(()=>boundedPipeline.begin(3840,2160),/working-memory budget/);assert.equal(boundedPipeline.active,false);boundedPipeline.dispose();
console.log('Secondary field orchestration passed: same-frame order, visibility, linear resolve, resize, DOF, cancellation and state restoration.');

// Exercise the actual GPU upload layout on CPU. No renderer or context.
{
 const rows=new Uint32Array(36),floats=new Float32Array(rows.buffer);
 for(const [i,id,available,original]of [[0,0x20000002,1,4],[1,0x10000001,1,1],[2,0x10000002,0,7]]) {
  const at=i*12;rows.set([id,2,3,4,5,6,1,available,original],at);
  floats.set(i===1?[0,-0,0]:[-.25,2,3],at+9);
 }
 const packed=responseApplicationTexture(rows,6);
 assert.equal(packed.count,3);assert.equal(packed.width,6);assert.equal(packed.height,2);
 assert.deepEqual(packed.words.slice(0,12),rows.slice(12,24),'sorting never changes original conditional RGB or weights');
 assert.equal(packed.words[10],0x80000000,'signed zero is retained, not used to infer availability');
 assert.deepEqual(packed.words.slice(12,24),rows.slice(24,36));
 assert.deepEqual(packed.words.slice(24,36),rows.slice(0,12));
 assert.throws(()=>responseApplicationTexture(new Uint32Array([...rows.slice(0,12),...rows.slice(0,12)]),6),/Duplicate/);
 const bad=rows.slice();bad[7]=5;assert.throws(()=>responseApplicationTexture(bad,6),/weights/);
 bad.set(rows);bad[6]=3;assert.throws(()=>responseApplicationTexture(bad,6),/Unsupported/,'total alternative cannot be added to components');
 const query={id:6,origin_m:[1,2,3],lens:{aperture_radius:0,focus_distance:4},film:{gain:1},layer_specs:[{}, {}, {}],
  transport:{transport_scene_sha256:'a'.repeat(64)}};
 const application={query_contexts:[query],current_query_context_ids:[6],current_camera_revision:'b'.repeat(64),
  current_live_revision:2,current_transport_scene_sha256:'a'.repeat(64)};
 assert.equal(currentResponseQuery(application,[1,2,3],2),6);
 assert.equal(currentResponseQuery(application,[1,2,3.001],2),-1,'different eye or lens-sampled origin has no match');
 assert.equal(currentResponseQuery(application,[1,2,3],1),-1,'stale native camera context has no match');
 assert.equal(currentResponseQuery({...application,current_query_context_ids:[]},[1,2,3],2),-1,'cache order cannot infer current context');
 assert.equal(matchResponseQuery(application,query),6);
 assert.equal(matchResponseQuery(application,{...query,lens:{...query.lens,aperture_radius:.01}}),-1);
 const shader=readFileSync(new URL('../../spectral-analyzer/csrc/shaders/base_material.frag.glsl',import.meta.url),'utf8');
 assert(!shader.includes('direction_moment')&&!shader.includes('field_material_mask'));
 assert.match(shader,/float\(b\.w\) \/ float\(c\.x\)/);
 assert.match(shader,/linear_color\.rgb \+= texelFetch\(uFieldResolvedVolume/);
 assert.match(shader,/nearest\.x != vResponseRecipient \|\| nearest\.y != floatBitsToUint\(gl_FragCoord\.z\)/);
 assert.match(shader,/ResponseRecipientDepth = uvec2\(vResponseRecipient, floatBitsToUint\(gl_FragCoord\.z\)\)/);
 assert.match(shader,/float reciprocal = 1\.0 \/ x;/,'final canonical ACES avoids overflow of finite float32 HDR values');
 assert.match(shader,/FragColor = mix\(vec4\(baseline_color, alpha\),\s*vec4\(copied_response\.rgb, 1\.0\), copied_response\.a\)/);
 // The two-target blend, with independently observed component availability.
 for(const c of [0,.25,1]) {
  const P=.2,A=.4,B=.6,F=.8,V=.1;
  const result=V+c*F+(1-c)*P+(1-(c+(1-c)*A))*B;
  assert(Math.abs(result-(V+c*F+(1-c)*(P+(1-A)*B)))<1e-12);
 }
}
console.log('Native response application CPU PASS: exact keys/bits, independent sample availability, strict query matching and separate-volume composition');
