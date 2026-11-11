// Synchronous primary -> material volume -> reflectance -> sorted glass.
// All four stages share one linear target. The canonical material shader does
// the only display conversion. No CPU visibility readback or second frame.
import {copyPhongUniforms} from './marble_phong_id_pass.js';

const IDENTITY=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
export const SECONDARY_FIELD_PASSES=Object.freeze(['volumetric','reflectance','glass']);
export function secondaryFieldFragment(source) {
  if(typeof source!=='string'||!/^#version\s+300\s+es\b/.test(source)||!source.includes('uSecondaryPass'))
    throw Error('Export the current canonical material shader before using the secondary field.');
  return source.replace(/^(#version[^\n]*\n)/,'$1#define PLUCK_SECONDARY_FIELD 1\n');
}

export function createStudioSecondaryField({gl,primary,secondary}) {
  if(!gl.getExtension('EXT_color_buffer_float'))throw Error('Linear floating-point render targets are unavailable.');
  const location=(p,n)=>gl.getUniformLocation(p,n),uniform=(n,v)=>gl.uniform1i(location(secondary,n),v);
  if(location(primary,'uLinearOutput')===null||location(secondary,'uSecondaryPass')===null)
    throw Error('The primary and secondary material shader exports do not match.');
  let target=null,saved=null,closed=false;
  const vao=gl.createVertexArray(),buffer=gl.createBuffer();
  const oldVao=gl.getParameter(gl.VERTEX_ARRAY_BINDING),oldBuffer=gl.getParameter(gl.ARRAY_BUFFER_BINDING);
  try {
    if(!vao||!buffer)throw Error('Cannot allocate field resolve geometry.');
    gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,0,3,-1,0,-1,3,0]),gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
  }finally {gl.bindVertexArray(oldVao);gl.bindBuffer(gl.ARRAY_BUFFER,oldBuffer);}
  const drop=()=>{if(!target)return;gl.deleteFramebuffer(target.fbo);gl.deleteTexture(target.color);gl.deleteTexture(target.depth);target=null;};
  function allocate(width,height) {
    if(target?.width===width&&target?.height===height)return;
    drop();target={width,height,fbo:gl.createFramebuffer(),color:gl.createTexture(),depth:gl.createTexture()};
    if(!target.fbo||!target.color||!target.depth){drop();throw Error('Cannot allocate linear field target.');}
    gl.bindFramebuffer(gl.FRAMEBUFFER,target.fbo);
    for(const [texture,internalFormat,attachment]of [[target.color,gl.RGBA16F,gl.COLOR_ATTACHMENT0],[target.depth,gl.DEPTH_COMPONENT24,gl.DEPTH_ATTACHMENT]]) {
      gl.bindTexture(gl.TEXTURE_2D,texture);gl.texStorage2D(gl.TEXTURE_2D,1,internalFormat,width,height);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
      gl.framebufferTexture2D(gl.FRAMEBUFFER,attachment,gl.TEXTURE_2D,texture,0);
    }
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE){drop();throw Error('Linear field framebuffer is incomplete.');}
  }
  const state=()=>{
    const names=['DRAW_FRAMEBUFFER_BINDING','READ_FRAMEBUFFER_BINDING','VIEWPORT','CURRENT_PROGRAM','VERTEX_ARRAY_BINDING',
      'ACTIVE_TEXTURE','DEPTH_FUNC','DEPTH_WRITEMASK','BLEND_SRC_RGB','BLEND_DST_RGB','BLEND_SRC_ALPHA','BLEND_DST_ALPHA',
      'BLEND_EQUATION_RGB','BLEND_EQUATION_ALPHA'];
    const s=Object.fromEntries(names.map(n=>[n,gl.getParameter(gl[n])]));
    s.materialAttribute=gl.getVertexAttrib(2,gl.CURRENT_VERTEX_ATTRIB);
    s.enabled=Object.fromEntries(['DEPTH_TEST','BLEND','CULL_FACE'].map(n=>[n,gl.isEnabled(gl[n])]));
    gl.activeTexture(gl.TEXTURE0+14);s.texture=gl.getParameter(gl.TEXTURE_BINDING_2D);s.sampler=gl.getParameter(gl.SAMPLER_BINDING);
    gl.activeTexture(s.ACTIVE_TEXTURE);s.specular=gl.getUniform(primary,location(primary,'uEnableSpecular'));return s;
  };
  function restore() {
    if(!saved)return;const s=saved;saved=null;
    gl.useProgram(primary);gl.uniform1i(location(primary,'uLinearOutput'),0);gl.uniform1i(location(primary,'uEnableSpecular'),Number(s.specular));
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,s.DRAW_FRAMEBUFFER_BINDING);gl.bindFramebuffer(gl.READ_FRAMEBUFFER,s.READ_FRAMEBUFFER_BINDING);
    gl.viewport(...s.VIEWPORT);gl.depthFunc(s.DEPTH_FUNC);gl.depthMask(s.DEPTH_WRITEMASK);
    gl.blendFuncSeparate(s.BLEND_SRC_RGB,s.BLEND_DST_RGB,s.BLEND_SRC_ALPHA,s.BLEND_DST_ALPHA);
    gl.blendEquationSeparate(s.BLEND_EQUATION_RGB,s.BLEND_EQUATION_ALPHA);
    for(const [n,on]of Object.entries(s.enabled))gl[on?'enable':'disable'](gl[n]);
    gl.activeTexture(gl.TEXTURE0+14);gl.bindTexture(gl.TEXTURE_2D,s.texture);gl.bindSampler(14,s.sampler);gl.activeTexture(s.ACTIVE_TEXTURE);
    if(s.materialAttribute instanceof Int32Array)gl.vertexAttribI4iv(2,s.materialAttribute);
    else if(s.materialAttribute instanceof Uint32Array)gl.vertexAttribI4uiv(2,s.materialAttribute);
    else gl.vertexAttrib4fv(2,s.materialAttribute);
    gl.bindVertexArray(s.VERTEX_ARRAY_BINDING);gl.useProgram(s.CURRENT_PROGRAM);
  }
  function begin(width,height) {
    if(closed||saved||gl.isContextLost())throw Error('Secondary field is closed, busy or context lost.');
    if(!Number.isSafeInteger(width)||!Number.isSafeInteger(height)||width<1||height<1||width>gl.getParameter(gl.MAX_TEXTURE_SIZE)||height>gl.getParameter(gl.MAX_TEXTURE_SIZE))
      throw Error('Invalid secondary target dimensions.');
    saved=state();
    try {
      gl.activeTexture(gl.TEXTURE0+14);gl.bindSampler(14,null);allocate(width,height);
      gl.bindTexture(gl.TEXTURE_2D,saved.texture);gl.activeTexture(saved.ACTIVE_TEXTURE);
      gl.bindFramebuffer(gl.FRAMEBUFFER,target.fbo);gl.viewport(0,0,width,height);
      gl.useProgram(primary);gl.uniform1i(location(primary,'uLinearOutput'),1);
      return true;
    }catch(error){restore();throw error;}
  }
  function finish({bind,drawOpaque,drawTransparent,phongSpecular=false,specular=true}) {
    if(!saved)throw Error('Begin the primary linear frame before its secondary passes.');
    if([bind,drawOpaque,drawTransparent].some(fn=>typeof fn!=='function'))throw Error('Explicit field and scene draw callbacks are required.');
    try {
      gl.useProgram(secondary);copyPhongUniforms(gl,primary,secondary,{allowInactive:true});
      if(!bind(secondary))throw Error('The directional field became stale before its secondary draw.');
      // The resolve sampler remains active in this linked program even when
      // its branch is not taken. Bind the complete material 2-D texture here,
      // never the color attachment currently being written (feedback loop).
      uniform('uFieldResolvedColor',0);
      gl.uniform1i(location(secondary,'uEnableSpecular'),Number(specular));uniform('uSecondaryPhongSpecular',Number(phongSpecular));
      gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.depthMask(false);gl.enable(gl.BLEND);gl.blendEquation(gl.FUNC_ADD);
      for(let pass=0;pass<3;pass++) {
        uniform('uSecondaryPass',pass);
        gl.depthFunc(pass===2?gl.LEQUAL:gl.EQUAL);
        gl.blendFuncSeparate(gl.ONE,pass===2?gl.ONE_MINUS_SRC_ALPHA:gl.ONE,gl.ONE,pass===2?gl.ONE_MINUS_SRC_ALPHA:gl.ONE);
        const outcome=(pass===2?drawTransparent:drawOpaque)(secondary,SECONDARY_FIELD_PASSES[pass]);
        if(outcome&&typeof outcome.then==='function')throw Error('Secondary draws must finish synchronously in the primary frame.');
      }
      // DOF owns a single-sample offscreen depth target. The default canvas
      // may be multisampled, so never attempt a single->multisample depth blit.
      if(saved.DRAW_FRAMEBUFFER_BINDING) {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER,target.fbo);gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,saved.DRAW_FRAMEBUFFER_BINDING);
        gl.blitFramebuffer(0,0,target.width,target.height,0,0,target.width,target.height,gl.DEPTH_BUFFER_BIT,gl.NEAREST);
      }
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,saved.DRAW_FRAMEBUFFER_BINDING);gl.viewport(0,0,target.width,target.height);
      gl.disable(gl.DEPTH_TEST);gl.disable(gl.BLEND);gl.depthMask(false);gl.bindVertexArray(vao);
      // The shared vertex shader still declares integer aMatId on the resolve
      // triangle, even though the fragment resolve skips material fetches.
      gl.vertexAttribI4i(2,0,0,0,0);
      uniform('uSecondaryPass',3);uniform('uFieldResolvedColor',14);
      gl.uniformMatrix4fv(location(secondary,'uMV'),false,IDENTITY);gl.uniformMatrix4fv(location(secondary,'uMVP'),false,IDENTITY);
      gl.activeTexture(gl.TEXTURE0+14);gl.bindSampler(14,null);gl.bindTexture(gl.TEXTURE_2D,target.color);
      gl.drawArrays(gl.TRIANGLES,0,3);
      return {passes:[...SECONDARY_FIELD_PASSES],linear:true,resolveCount:1};
    }finally {restore();}
  }
  return {begin,finish,cancel:restore,dispose(){restore();drop();gl.deleteBuffer(buffer);gl.deleteVertexArray(vao);closed=true;},
    get active(){return !!saved;}};
}
