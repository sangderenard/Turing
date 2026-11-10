import {createStudioSubjectRenderer,studioFlatSamplerPacket,validateStudioSubjectPacket} from '../examples/meowdoku/web/marble_studio_subject_renderer.js';
import {triangleCentres,transparentTriangleOrder} from '../examples/meowdoku/web/marble_glass_preview.js';
const out=document.querySelector('#result'),canvas=document.querySelector('#view'),assert=(x,m)=>{if(!x)throw Error(m);};
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const multiply=(a,b)=>{const m=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)m[c*4+r]+=a[k*4+r]*b[c*4+k];return m;};
try {
 const asset=new URLSearchParams(window.location.search).get('asset')||'../build/dot-studio-game-transfer-20261110/studio-marble-appearance.json';
 const response=await fetch(asset);assert(response.ok,'Asset HTTP '+response.status);const packet=await response.json();
 const gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true,alpha:false});assert(gl,'WebGL2 unavailable');
 const sampler=studioFlatSamplerPacket(),plan=validateStudioSubjectPacket(packet,sampler);
 // Baseline uses the real current Studio upload/sort/draw functions verbatim,
 // scoped to this independent test context, not a reimplementation of materials.
 const source=await fetch('../examples/meowdoku/web/marble_studio.js').then(r=>r.text());
 const extract=name=>{const start=source.indexOf(`function ${name}(`);assert(start>=0,'Missing Studio '+name);
  const open=source.indexOf('{',start);let depth=1,i=open+1;for(;depth;i++){if(source[i]==='{')depth++;if(source[i]==='}')depth--;}
  return source.slice(start,i);};
 const canonical=Function('gl','triangleCentres','transparentTriangleOrder',
  [extract('mesh'),extract('transparentMesh'),extract('drawTransparent')].join('\n')+'\nreturn {mesh,transparentMesh,drawTransparent};')(gl,triangleCentres,transparentTriangleOrder);
 const program=gl.createProgram();
 for(const [name,type]of [['vertex',gl.VERTEX_SHADER],['fragment',gl.FRAGMENT_SHADER]]) {
  const s=gl.createShader(type);gl.shaderSource(s,packet.render.shaders[name]);gl.compileShader(s);
  assert(gl.getShaderParameter(s,gl.COMPILE_STATUS),gl.getShaderInfoLog(s));gl.attachShader(program,s);
 }
 gl.linkProgram(program);assert(gl.getProgramParameter(program,gl.LINK_STATUS),gl.getProgramInfoLog(program));gl.useProgram(program);
 const location=n=>gl.getUniformLocation(program,n);
 for(const u of packet.render.uniforms) {
  const loc=location(u.name),v=u.value;
  switch(u.type) {
   case gl.FLOAT:gl.uniform1f(loc,v);break;
   case gl.FLOAT_VEC2:gl.uniform2fv(loc,v);break;
   case gl.FLOAT_VEC3:gl.uniform3fv(loc,v);break;
   case gl.FLOAT_VEC4:gl.uniform4fv(loc,v);break;
   case gl.FLOAT_MAT3:gl.uniformMatrix3fv(loc,false,v);break;
   case gl.FLOAT_MAT4:gl.uniformMatrix4fv(loc,false,v);break;
   case gl.BOOL:case gl.INT:case gl.SAMPLER_2D:case gl.SAMPLER_2D_ARRAY:case gl.SAMPLER_3D:gl.uniform1i(loc,Number(v));break;
   default:throw Error('Fixture unsupported actual uniform '+u.name+' type '+u.type);
  }
 }
 const textures=[];
 const tex=row=>{const target=gl[row.target],texture=gl.createTexture();textures.push({target,texture,unit:row.unit});
  gl.activeTexture(gl.TEXTURE0+row.unit);gl.bindTexture(target,texture);
  gl.texParameteri(target,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(target,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  if(row.data===null)gl.texStorage3D(target,1,gl[row.internalFormat],row.width,row.height,row.depth);
  else {const data=row.type==='FLOAT'?new Float32Array(row.data):new Uint8Array(row.data);
   if(row.target==='TEXTURE_2D')gl.texImage2D(target,0,gl[row.internalFormat],row.width,row.height,0,gl[row.format],gl[row.type],data);
   else gl.texImage3D(target,0,gl[row.internalFormat],row.width,row.height,row.depth,0,gl[row.format],gl[row.type],data);}
 };
 for(const [unit,name]of ['pbr','phong','enamel','texstack'].entries()) {
  const values=packet.render.materials[name].data,width=256,height=Math.ceil(values.length/256),data=new Float32Array(width*height);data.set(values);
  tex({unit,target:'TEXTURE_2D',internalFormat:'R32F',format:'RED',type:'FLOAT',width,height,data});
 }
 for(const row of sampler.textures)tex(row);
 const opaque=canonical.mesh(plan.opaque),transparent=canonical.transparentMesh(plan.transparent);
 const renderer=createStudioSubjectRenderer({gl,packet,samplers:sampler});
 const view=packet.render.referenceCamera.view,fov=packet.render.referenceCamera.fovDegrees||50,f=1/Math.tan(fov*Math.PI/360),
  projection=[f,0,0,0,0,f,0,0,0,0,-30.01/29.99,-1,0,0,-.6/29.99,0],
  model=[.8,0,0,0,0,.8,0,0,0,0,.8,0,0,-.65,0,1],mv=multiply(view,model),mvp=multiply(projection,mv);
 const clean=()=>{gl.viewport(0,0,canvas.width,canvas.height);gl.disable(gl.SCISSOR_TEST);gl.disable(gl.STENCIL_TEST);
  gl.disable(gl.BLEND);gl.disable(gl.CULL_FACE);gl.disable(gl.POLYGON_OFFSET_FILL);gl.disable(gl.RASTERIZER_DISCARD);
  gl.disable(gl.SAMPLE_ALPHA_TO_COVERAGE);gl.disable(gl.SAMPLE_COVERAGE);gl.enable(gl.DITHER);gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LESS);gl.depthRange(0,1);gl.frontFace(gl.CCW);gl.colorMask(true,true,true,true);gl.depthMask(true);
  gl.clearColor(.035,.06,.09,1);gl.clearDepth(1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);};
 const pixels=()=>{const p=new Uint8Array(canvas.width*canvas.height*4);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,p);return p;};
 clean();gl.useProgram(program);gl.uniformMatrix4fv(location('uMV'),false,mv);gl.uniformMatrix4fv(location('uMVP'),false,mvp);
 gl.vertexAttrib2f(3,0,0);gl.vertexAttribI4i(5,1,0,0,0);gl.vertexAttrib3f(6,0,0,0);
 gl.bindVertexArray(opaque.vao);gl.drawArrays(gl.TRIANGLES,0,opaque.count);
 gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.blendEquation(gl.FUNC_ADD);gl.depthMask(false);
 canonical.drawTransparent(transparent,mv);const baseline=pixels();assert(gl.getError()===gl.NO_ERROR,'Baseline GL error');
 // Contaminate state as an incumbent engine can. Adapter must temporarily own
 // its required pass state and restore program, buffers, textures and constants.
 const incumbent=gl.createVertexArray(),buffer=gl.createBuffer(),unpack=gl.createBuffer(),foreignSampler=gl.createSampler();
 const poison=()=>{gl.useProgram(program);gl.bindVertexArray(incumbent);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
  gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER,unpack);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);gl.pixelStorei(gl.UNPACK_ROW_LENGTH,3);
  gl.enable(gl.BLEND);gl.blendFuncSeparate(gl.SRC_ALPHA,gl.ONE,gl.ZERO,gl.ONE);gl.blendEquation(gl.FUNC_REVERSE_SUBTRACT);
  gl.disable(gl.DEPTH_TEST);gl.depthMask(false);gl.depthFunc(gl.GREATER);gl.depthRange(.1,.9);gl.frontFace(gl.CW);
  gl.colorMask(true,false,true,false);gl.enable(gl.SCISSOR_TEST);gl.scissor(3,4,9,8);gl.enable(gl.STENCIL_TEST);
  gl.enable(gl.CULL_FACE);gl.enable(gl.POLYGON_OFFSET_FILL);gl.disable(gl.DITHER);gl.enable(gl.SAMPLE_ALPHA_TO_COVERAGE);
  gl.enable(gl.SAMPLE_COVERAGE);gl.enable(gl.RASTERIZER_DISCARD);gl.vertexAttrib4f(3,8,9,10,11);gl.vertexAttribI4i(5,8,7,6,5);gl.vertexAttrib4f(6,4,3,2,1);
  gl.activeTexture(gl.TEXTURE0+5);gl.bindSampler(5,foreignSampler);};
 const snapshot=()=>{
  const names=['CURRENT_PROGRAM','VERTEX_ARRAY_BINDING','ARRAY_BUFFER_BINDING','PIXEL_UNPACK_BUFFER_BINDING','ACTIVE_TEXTURE',
   'UNPACK_ALIGNMENT','UNPACK_ROW_LENGTH','DEPTH_WRITEMASK','DEPTH_FUNC','DEPTH_RANGE','FRONT_FACE','COLOR_WRITEMASK',
   'BLEND_SRC_RGB','BLEND_DST_RGB','BLEND_SRC_ALPHA','BLEND_DST_ALPHA','BLEND_EQUATION_RGB','BLEND_EQUATION_ALPHA',
   'VIEWPORT','SCISSOR_BOX','DRAW_FRAMEBUFFER_BINDING','READ_FRAMEBUFFER_BINDING'];
  const state=names.map(n=>{const v=gl.getParameter(gl[n]);return ArrayBuffer.isView(v)?Array.from(v):v;});
  for(const n of ['BLEND','DEPTH_TEST','CULL_FACE','SCISSOR_TEST','STENCIL_TEST','DITHER','POLYGON_OFFSET_FILL',
   'SAMPLE_ALPHA_TO_COVERAGE','SAMPLE_COVERAGE','RASTERIZER_DISCARD'])state.push(gl.isEnabled(gl[n]));
  for(const i of [3,5,6])state.push(Array.from(gl.getVertexAttrib(i,gl.CURRENT_VERTEX_ATTRIB)));
  for(let unit=0;unit<11;unit++){gl.activeTexture(gl.TEXTURE0+unit);for(const n of ['TEXTURE_BINDING_2D','TEXTURE_BINDING_2D_ARRAY','TEXTURE_BINDING_3D','SAMPLER_BINDING'])state.push(gl.getParameter(gl[n]));}
  gl.activeTexture(state[4]);return state;
 };
 const same=(a,b)=>a.length===b.length&&a.every((x,i)=>Array.isArray(x)?JSON.stringify(x)===JSON.stringify(b[i]):x===b[i]);
 clean();poison();const before=snapshot();renderer.draw({view,projection,model});assert(same(before,snapshot()),'GL state not restored');
 const candidate=pixels();let changed=0,maxDelta=0,lit=0;
 for(let i=0;i<baseline.length;i++){const d=Math.abs(baseline[i]-candidate[i]);if(d)changed++;maxDelta=Math.max(maxDelta,d);}
 for(let i=0;i<baseline.length;i+=4)if(baseline[i]!==baseline[0]||baseline[i+1]!==baseline[1]||baseline[i+2]!==baseline[2])lit++;
 assert(lit>1000,'Subject not visibly rendered');assert(changed===0,`Canonical Studio pixel mismatch ${changed}, max ${maxDelta}`);
 clean();poison();renderer.draw({view,projection,model,phase:'opaque'});renderer.draw({view,projection,model,phase:'transparent'});
 assert(same(before,snapshot()),'Split phases failed state restoration');const split=pixels();
 assert(split.every((v,i)=>v===candidate[i]),'Split phases differ');
 // Repeat after camera/model rotation, checking no stale transparency sort or
 // target game matrix ownership; returning to exact pose must match bitwise.
 const rotated=[0,0,-.8,0,0,.8,0,0,.8,0,0,0,.1,-.65,0,1];clean();renderer.draw({view,projection,model:rotated});
 clean();renderer.draw({view,projection,model});assert(pixels().every((v,i)=>v===candidate[i]),'Repeated draw/sort changed output');
 // Observe real GPU uniforms after one scaled rig draw, with original GL calls
 // intact. This proves scale is carried by emitter area, not a visual gain.
 let rendererProgram=null;const realUniform1f=gl.uniform1f;
 try {
  gl.uniform1f=function(loc,value){rendererProgram=gl.getParameter(gl.CURRENT_PROGRAM);return realUniform1f.call(gl,loc,value);};
  renderer.draw({view,projection,model,lightTransform:[2,0,0,0,0,2,0,0,0,0,2,0,1,2,3,1]});
 }finally{gl.uniform1f=realUniform1f;}
 assert(rendererProgram&&rendererProgram!==program,'Scaled renderer program was not observed');
 for(let i=0;i<plan.lightCount;i++) {
  const area=gl.getUniform(rendererProgram,gl.getUniformLocation(rendererProgram,`uLightIntensity[${i}]`));
  assert(area===Math.fround(plan.uniforms.get(`uLightIntensity[${i}]`).value*4),'Emitter area must scale by s squared');
  const direction=gl.getUniform(rendererProgram,gl.getUniformLocation(rendererProgram,`uLightSpotDir[${i}]`)),length=Math.hypot(...direction);
  assert(length===0||Math.abs(length-1)<1e-6,'Transformed spotlight direction is not unit length');
 }
 clean();renderer.draw({view,projection,model});assert(pixels().every((v,i)=>v===candidate[i]),'Rig scale leaked into later draws');
 assert(gl.getError()===gl.NO_ERROR,'Adapter GL error');renderer.dispose();renderer.dispose();assert(!renderer.ready,'Dispose failed');
 out.textContent=JSON.stringify({pass:true,asset,renderer:gl.getParameter(gl.RENDERER),counts:renderer.counts,
  canonicalStudioPixelDifferences:changed,maxDelta,visiblePixels:lit,sharedContextStateRestored:true,
  splitPhasesMatch:true,repeatedDraw:true,scaledEmitterAreas:true,normalizedSpotDirections:true,
  exactShaders:true,exactMaterialTables:true,rayTraceBake:false},null,2);
}catch(error){out.textContent=JSON.stringify({pass:false,error:String(error),stack:error.stack},null,2);}
