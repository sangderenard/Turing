// Isolated primary-visibility instrumentation of an actual linked Phong program.
// No visibility estimator, new material law, animation loop or physics path.
import {bindPhongVisibilityViews,phongPublicationDigest} from './marble_phong_visibility.js';
import {isOpaquePreviewMaterial} from './marble_glass_preview.js';

const PREFIX='dotPrimaryTriangle';
const MAIN=/void\s+main\s*\(\s*(?:void)?\s*\)\s*\{/g;
export function instrumentPhongTriangleIds(vertex,fragment) {
  for(const source of [vertex,fragment])if(typeof source!=='string'||!/^#version\s+300\s+es\b/.test(source)||
      source.includes(PREFIX)||[...source.matchAll(MAIN)].length!==1)
    throw Error('ID instrumentation requires one original WebGL2 main and unused private names.');
  const output=/\bout\s+(?:(?:lowp|mediump|highp)\s+)?vec4\s+FragColor\s*;/g;
  if([...fragment.matchAll(output)].length!==1||/layout\s*\([^)]*location\s*=/.test(fragment))
    throw Error('Expected the original single unlocated FragColor output.');
  return {
    vertex:vertex.replace(MAIN,`flat out highp uint ${PREFIX}Id;\nvoid main() {\n`+
      `${PREFIX}Id = uint(gl_VertexID / 3) + 1u;`),
    fragment:fragment.replace(output,'layout(location=0) out vec4 FragColor;')
      .replace(MAIN,`flat in highp uint ${PREFIX}Id;\nuniform bool ${PREFIX}Enabled;\n`+
        `layout(location=1) out highp uint ${PREFIX}Output;\nvoid main() {\n`+
        `${PREFIX}Output = ${PREFIX}Enabled ? ${PREFIX}Id : 0u;`),
  };
}

function linkedSources(gl,program) {
  if(!program||!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error('A live linked Phong program is required.');
  const source={};
  for(const shader of gl.getAttachedShaders(program)||[]) {
    const type=gl.getShaderParameter(shader,gl.SHADER_TYPE),key=type===gl.VERTEX_SHADER?'vertex':
      type===gl.FRAGMENT_SHADER?'fragment':null;
    if(!key||source[key])throw Error('Ambiguous original Phong shader ownership.');
    source[key]=gl.getShaderSource(shader);
  }
  if(!source.vertex||!source.fragment)throw Error('Original linked shader sources are unavailable.');
  return source;
}

function link(gl,sources) {
  const program=gl.createProgram(),shaders=[];
  if(!program)throw Error('Cannot allocate the isolated ID program.');
  try {
    for(const [kind,source]of [[gl.VERTEX_SHADER,sources.vertex],[gl.FRAGMENT_SHADER,sources.fragment]]) {
      const shader=gl.createShader(kind);if(!shader)throw Error('Cannot allocate the ID shader.');
      shaders.push(shader);gl.shaderSource(shader,source);gl.compileShader(shader);
      if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader)||'ID shader failed.');
      gl.attachShader(program,shader);
    }
    gl.linkProgram(program);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program)||'ID program failed.');
    return program;
  }catch(error){gl.deleteProgram(program);throw error;}
  finally{for(const shader of shaders)gl.deleteShader(shader);}
}

// Copy by declared GL type, never by numeric value (sampler types must not alias).
export function copyPhongUniforms(gl,from,to) {
  const samplers=new Set(['SAMPLER_2D','SAMPLER_3D','SAMPLER_CUBE','SAMPLER_2D_SHADOW',
    'SAMPLER_2D_ARRAY','SAMPLER_2D_ARRAY_SHADOW','SAMPLER_CUBE_SHADOW','INT_SAMPLER_2D',
    'INT_SAMPLER_3D','INT_SAMPLER_CUBE','INT_SAMPLER_2D_ARRAY','UNSIGNED_INT_SAMPLER_2D',
    'UNSIGNED_INT_SAMPLER_3D','UNSIGNED_INT_SAMPLER_CUBE','UNSIGNED_INT_SAMPLER_2D_ARRAY'].map(k=>gl[k]));
  const methods=new Map([
    [gl.FLOAT_VEC2,'uniform2fv'],[gl.FLOAT_VEC3,'uniform3fv'],[gl.FLOAT_VEC4,'uniform4fv'],
    [gl.INT_VEC2,'uniform2iv'],[gl.INT_VEC3,'uniform3iv'],[gl.INT_VEC4,'uniform4iv'],
    [gl.BOOL_VEC2,'uniform2iv'],[gl.BOOL_VEC3,'uniform3iv'],[gl.BOOL_VEC4,'uniform4iv'],
    [gl.UNSIGNED_INT_VEC2,'uniform2uiv'],[gl.UNSIGNED_INT_VEC3,'uniform3uiv'],[gl.UNSIGNED_INT_VEC4,'uniform4uiv'],
    [gl.FLOAT_MAT2,'uniformMatrix2fv'],[gl.FLOAT_MAT3,'uniformMatrix3fv'],[gl.FLOAT_MAT4,'uniformMatrix4fv'],
    [gl.FLOAT_MAT2x3,'uniformMatrix2x3fv'],[gl.FLOAT_MAT2x4,'uniformMatrix2x4fv'],
    [gl.FLOAT_MAT3x2,'uniformMatrix3x2fv'],[gl.FLOAT_MAT3x4,'uniformMatrix3x4fv'],
    [gl.FLOAT_MAT4x2,'uniformMatrix4x2fv'],[gl.FLOAT_MAT4x3,'uniformMatrix4x3fv'],
  ]);
  for(let i=0;i<gl.getProgramParameter(from,gl.ACTIVE_UNIFORMS);i++) {
    const info=gl.getActiveUniform(from,i);
    if(info.size>1&&!info.name.includes('[0]'))throw Error(`Cannot identify Phong uniform array ${info.name}.`);
    // Active [0] records describe the whole array, but getUniform returns only
    // the addressed element. Read every light/matrix/sampler by its own key.
    for(let j=0;j<info.size;j++) {
      const name=info.size>1?info.name.replace('[0]',`[${j}]`):info.name,
        originalLocation=gl.getUniformLocation(from,name),location=gl.getUniformLocation(to,name);
      if(originalLocation===null||location===null)throw Error(`Cannot preserve active Phong uniform ${name}.`);
      const value=gl.getUniform(from,originalLocation);
      if(value===null)throw Error(`Cannot preserve active Phong uniform ${name}.`);
      if(info.type===gl.FLOAT)gl.uniform1fv(location,typeof value==='number'?[value]:value);
      else if(info.type===gl.INT||info.type===gl.BOOL||samplers.has(info.type))
        gl.uniform1iv(location,typeof value==='number'||typeof value==='boolean'?[Number(value)]:value);
      else if(info.type===gl.UNSIGNED_INT)gl.uniform1uiv(location,typeof value==='number'?[value]:value);
      else {
        const method=methods.get(info.type);if(!method)throw Error(`Unsupported Phong uniform type ${info.type}.`);
        if(method.startsWith('uniformMatrix'))gl[method](location,false,value);else gl[method](location,value);
      }
    }
  }
}

function saveState(gl) {
  const names=['CURRENT_PROGRAM','VERTEX_ARRAY_BINDING','ARRAY_BUFFER_BINDING','RENDERBUFFER_BINDING',
    'DRAW_FRAMEBUFFER_BINDING','READ_FRAMEBUFFER_BINDING','ACTIVE_TEXTURE','VIEWPORT','SCISSOR_BOX',
    'DEPTH_WRITEMASK','DEPTH_FUNC','DEPTH_RANGE','COLOR_WRITEMASK','FRONT_FACE','CULL_FACE_MODE',
    'BLEND_SRC_RGB','BLEND_DST_RGB','BLEND_SRC_ALPHA','BLEND_DST_ALPHA','BLEND_EQUATION_RGB','BLEND_EQUATION_ALPHA'];
  const state=Object.fromEntries(names.map(name=>[name,gl.getParameter(gl[name])]));
  state.enabled=Object.fromEntries(['BLEND','DEPTH_TEST','CULL_FACE','SCISSOR_TEST','DITHER',
    'SAMPLE_ALPHA_TO_COVERAGE','SAMPLE_COVERAGE','RASTERIZER_DISCARD','POLYGON_OFFSET_FILL'].map(name=>[name,gl.isEnabled(gl[name])]));
  state.attributes=Array.from({length:gl.getParameter(gl.MAX_VERTEX_ATTRIBS)},(_,i)=>gl.getVertexAttrib(i,gl.CURRENT_VERTEX_ATTRIB));
  state.textures=[];
  for(let unit=0;unit<gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS);unit++) {
    gl.activeTexture(gl.TEXTURE0+unit);
    state.textures.push({bindings:['TEXTURE_BINDING_2D','TEXTURE_BINDING_2D_ARRAY','TEXTURE_BINDING_3D','TEXTURE_BINDING_CUBE_MAP']
      .map(name=>gl.getParameter(gl[name])),sampler:gl.getParameter(gl.SAMPLER_BINDING)});
  }
  gl.activeTexture(state.ACTIVE_TEXTURE);return state;
}
function restoreState(gl,s) {
  for(const [name,enabled]of Object.entries(s.enabled))gl[enabled?'enable':'disable'](gl[name]);
  gl.useProgram(s.CURRENT_PROGRAM);gl.bindVertexArray(s.VERTEX_ARRAY_BINDING);
  gl.bindBuffer(gl.ARRAY_BUFFER,s.ARRAY_BUFFER_BINDING);gl.bindRenderbuffer(gl.RENDERBUFFER,s.RENDERBUFFER_BINDING);
  gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,s.DRAW_FRAMEBUFFER_BINDING);gl.bindFramebuffer(gl.READ_FRAMEBUFFER,s.READ_FRAMEBUFFER_BINDING);
  gl.viewport(...s.VIEWPORT);gl.scissor(...s.SCISSOR_BOX);gl.depthMask(s.DEPTH_WRITEMASK);
  gl.depthFunc(s.DEPTH_FUNC);gl.depthRange(...s.DEPTH_RANGE);gl.colorMask(...s.COLOR_WRITEMASK);
  gl.frontFace(s.FRONT_FACE);gl.cullFace(s.CULL_FACE_MODE);
  gl.blendFuncSeparate(s.BLEND_SRC_RGB,s.BLEND_DST_RGB,s.BLEND_SRC_ALPHA,s.BLEND_DST_ALPHA);
  gl.blendEquationSeparate(s.BLEND_EQUATION_RGB,s.BLEND_EQUATION_ALPHA);
  s.attributes.forEach((value,i)=>{
    if(value instanceof Int32Array)gl.vertexAttribI4iv(i,value);
    else if(value instanceof Uint32Array)gl.vertexAttribI4uiv(i,value);
    else gl.vertexAttrib4fv(i,value);
  });
  s.textures.forEach((row,unit)=>{
    gl.activeTexture(gl.TEXTURE0+unit);
    [gl.TEXTURE_2D,gl.TEXTURE_2D_ARRAY,gl.TEXTURE_3D,gl.TEXTURE_CUBE_MAP].forEach((target,i)=>gl.bindTexture(target,row.bindings[i]));
    gl.bindSampler(unit,row.sampler);
  });
  gl.activeTexture(s.ACTIVE_TEXTURE);
}

function expandedOpaqueMesh(gl,source,indices,resources) {
  const positions=[],normals=[],materials=[],groups=[];
  for(const index of indices) {
    const face=source.faces[index];
    for(let corner=0;corner<3;corner++) {
      positions.push(...face[corner]);normals.push(...face[4][corner]);materials.push(face[3]);groups.push(face[5]??0);
    }
  }
  const vao=gl.createVertexArray();resources.vao=vao;gl.bindVertexArray(vao);
  for(const [location,data,integer]of [[0,new Float32Array(positions),false],[1,new Float32Array(normals),false],
    [2,new Int32Array(materials),true],[4,new Int32Array(groups),true]]) {
    const buffer=gl.createBuffer();resources.buffers.push(buffer);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
    gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);gl.enableVertexAttribArray(location);
    if(integer)gl.vertexAttribIPointer(location,1,gl.INT,0,0);else gl.vertexAttribPointer(location,3,gl.FLOAT,false,0,0);
  }
  return vao;
}

export function createPhongPrimaryIdPass({gl,program:originalProgram,maxPixels=1048576}) {
  if(!gl||gl.isContextLost()||typeof gl.drawBuffers!=='function'||
      !Number.isSafeInteger(maxPixels)||maxPixels<1||maxPixels>4194304)
    throw Error('A live WebGL2 context and bounded pixel budget are required.');
  const original=linkedSources(gl,originalProgram),instrumented=instrumentPhongTriangleIds(original.vertex,original.fragment),
    program=link(gl,instrumented);
  const enabled=gl.getUniformLocation(program,`${PREFIX}Enabled`),renderPass=gl.getUniformLocation(program,'uRenderPass');
  if(enabled===null||renderPass===null){gl.deleteProgram(program);throw Error('Original Phong opaque-pass uniform is absent.');}
  let closed=false,busy=false,deleted=false;
  const drop=()=>{if(closed&&!busy&&!deleted){gl.deleteProgram(program);deleted=true;}};
  function dispose(){closed=true;drop();}
  async function sample({source,view,width,height,configure,drawOpaqueOccluders=null,signal}={}) {
    if(closed||busy)throw Error('ID pass is closed or busy.');
    if(!Number.isSafeInteger(width)||!Number.isSafeInteger(height)||width<1||height<1||width*height>maxPixels||
        typeof configure!=='function'||(drawOpaqueOccluders!==null&&typeof drawOpaqueOccluders!=='function'))
      throw Error('Bounded dimensions and an explicit view/material configure callback are required.');
    busy=true;
    const resources={framebuffer:null,renderbuffers:[],buffers:[],vao:null,fence:null};
    const check=()=>{if(closed||signal?.aborted||gl.isContextLost())throw Error('ID pass canceled or graphics context lost.');};
    try {
      check();
      if(source?.schema!=='turing.phong-triangle-source.v1'||!Number.isSafeInteger(source.revision)||
          !Array.isArray(source.faces)||!source.faces.length||source.faces.length>200000)
        throw Error('Use an exact captured Phong triangle source.');
      const meshSha256=await phongPublicationDigest({id:source.id,faces:source.faces,model:source.model,frame:source.frame});
      if(meshSha256!==source.meshSha256||await phongPublicationDigest(source.pbr)!==source.materialSha256)
        throw Error('Source changed after its immutable capture.');
      const [bound]=await bindPhongVisibilityViews([view]);
      const opaque=[],retainedTransparentSourceIndices=[];
      source.faces.forEach((face,index)=>{
        const material=face[3],p=material*16;
        if(!Number.isSafeInteger(material)||material<0||p+15>=source.pbr.length||
            face.slice(0,3).some(v=>!Array.isArray(v)||v.length!==3||!v.every(Number.isFinite))||
            !Array.isArray(face[4])||face[4].length!==3||face[4].some(v=>v.length!==3||!v.every(Number.isFinite))||
            (face[5]!==undefined&&(!Number.isInteger(face[5])||face[5]<-2147483648||face[5]>2147483647)))
          throw Error('Source face attributes cannot be replayed exactly.');
        (isOpaquePreviewMaterial(source.pbr[p+5],source.pbr[p+7])?opaque:retainedTransparentSourceIndices).push(index);
      });
      const shaderSha256=await phongPublicationDigest(original);check();
      const state=saveState(gl);
      try {
        resources.framebuffer=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,resources.framebuffer);
        for(const [attachment,format]of [[gl.COLOR_ATTACHMENT0,gl.RGBA8],[gl.COLOR_ATTACHMENT1,gl.R32UI],
          [gl.DEPTH_ATTACHMENT,gl.DEPTH_COMPONENT24]]) {
          const buffer=gl.createRenderbuffer();resources.renderbuffers.push(buffer);gl.bindRenderbuffer(gl.RENDERBUFFER,buffer);
          gl.renderbufferStorage(gl.RENDERBUFFER,format,width,height);
          gl.framebufferRenderbuffer(gl.FRAMEBUFFER,attachment,gl.RENDERBUFFER,buffer);
        }
        gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1]);
        if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('Integer ID framebuffer is incomplete.');
        gl.viewport(0,0,width,height);gl.colorMask(true,true,true,true);gl.depthMask(true);
        for(const name of ['BLEND','SCISSOR_TEST','DITHER','SAMPLE_ALPHA_TO_COVERAGE','SAMPLE_COVERAGE',
          'RASTERIZER_DISCARD','POLYGON_OFFSET_FILL'])gl.disable(gl[name]);
        gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);
        gl.clearBufferfv(gl.COLOR,0,new Float32Array(4));gl.clearBufferuiv(gl.COLOR,1,new Uint32Array(4));
        gl.clearBufferfv(gl.DEPTH,0,new Float32Array([1]));
        gl.useProgram(program);copyPhongUniforms(gl,originalProgram,program);
        const configureSubject=()=>{
          const outcome=configure(program,{gl,view:bound,source});
          if(outcome&&typeof outcome.then==='function')throw Error('Configure must be synchronous on the shared GL context.');
          if(gl.getParameter(gl.CURRENT_PROGRAM)!==program||gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING)!==resources.framebuffer)
            throw Error('Configure changed the ID pass program/framebuffer owner.');
          gl.uniform1i(renderPass,1);gl.disable(gl.BLEND);gl.depthMask(true);
        };
        configureSubject();
        if(drawOpaqueOccluders) {
          gl.uniform1i(enabled,0);
          const outcome=drawOpaqueOccluders({gl,program,view:bound,source});
          if(outcome&&typeof outcome.then==='function')throw Error('Occluder replay must be synchronous.');
          configureSubject();
        }
        gl.uniform1i(enabled,1);expandedOpaqueMesh(gl,source,opaque,resources);
        gl.drawArrays(gl.TRIANGLES,0,opaque.length*3);
        resources.fence=gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE,0);gl.flush();
        if(!resources.fence||gl.getError()!==gl.NO_ERROR)throw Error('Actual Phong ID draw failed.');
      }finally{restoreState(gl,state);}
      // The caller's render loop may continue while only our fence is pending.
      const start=performance.now();
      while(true) {
        check();const ready=gl.clientWaitSync(resources.fence,0,0);
        if(ready===gl.WAIT_FAILED)throw Error('ID pass GPU fence failed.');
        if(ready!==gl.TIMEOUT_EXPIRED)break;
        if(performance.now()-start>5000)throw Error('ID pass exceeded its five-second readback budget.');
        await new Promise(resolve=>setTimeout(resolve,2));
      }
      check();
      const oldRead=gl.getParameter(gl.READ_FRAMEBUFFER_BINDING),oldPack=gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING),
        packNames=['PACK_ALIGNMENT','PACK_ROW_LENGTH','PACK_SKIP_PIXELS','PACK_SKIP_ROWS'],
        packValues=packNames.map(name=>gl.getParameter(gl[name])),pixels=new Uint32Array(width*height);
      try {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER,resources.framebuffer);gl.readBuffer(gl.COLOR_ATTACHMENT1);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER,null);
        packNames.forEach((name,i)=>gl.pixelStorei(gl[name],i===0?4:0));
        gl.readPixels(0,0,width,height,gl.RED_INTEGER,gl.UNSIGNED_INT,pixels);
        if(gl.getError()!==gl.NO_ERROR)throw Error('Integer ID readback failed.');
      }finally {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER,oldRead);gl.bindBuffer(gl.PIXEL_PACK_BUFFER,oldPack);
        packNames.forEach((name,i)=>gl.pixelStorei(gl[name],packValues[i]));
      }
      const counts=new Map();
      for(const value of pixels)if(value) {
        if(value>opaque.length)throw Error('GPU returned an out-of-range source triangle ID.');
        const id=opaque[value-1];counts.set(id,(counts.get(id)||0)+1);
      }
      const triangleIds=[...counts.keys()].sort((a,b)=>a-b);
      return {schema:'turing.phong-primary-visibility-sample.v1',revision:source.revision,
        meshSha256,materialSha256:source.materialSha256,viewIndex:bound.index,cameraSha256:bound.cameraSha256,
        triangleIds,sampleCount:width*height,width,height,retainedTransparentSourceIndices,
        observedPixelCounts:triangleIds.map(triangle=>({triangle,pixels:counts.get(triangle)})),
        producer:{kind:'primary-raster-triangle-id',id:`actual-phong:${shaderSha256}`,readbackComplete:true,
          sourceShaderSha256:shaderSha256,opaqueClassifier:'marble_glass_preview.isOpaquePreviewMaterial',
          occluderScope:drawOpaqueOccluders?'caller-replayed-opaque-scene':'subject-self-occlusion-only',
          multisample:false,depthFunction:state.DEPTH_FUNC},
        scope:'Actual opaque-pass pixel IDs only; retain every nonopaque face, full physics/shadow/refraction geometry and unseen views.'};
    }finally {
      if(resources.fence)gl.deleteSync(resources.fence);
      if(resources.vao)gl.deleteVertexArray(resources.vao);
      for(const buffer of resources.buffers)gl.deleteBuffer(buffer);
      for(const buffer of resources.renderbuffers)gl.deleteRenderbuffer(buffer);
      if(resources.framebuffer)gl.deleteFramebuffer(resources.framebuffer);
      busy=false;drop();
    }
  }
  return {sample,dispose,get busy(){return busy;},get closed(){return closed;},instrumentedSources:instrumented};
}
