// Exact Studio raster subject on an incumbent WebGL2 context. No world, loop,
// physics, shader rewrite, generated palette, lightfield bake or ray renderer.
import {isOpaquePreviewMaterial,triangleCentres,transparentTriangleOrder} from './marble_glass_preview.js';
import {phongInterior} from './marble_bubbles.js';

const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const arrays=x=>Array.isArray(x)||ArrayBuffer.isView(x);
const finite=x=>arrays(x)&&Array.from(x).every(Number.isFinite);
const matrix=(m,name)=>{if(!finite(m)||m.length!==16)throw Error(`${name} needs 16 finite values.`);return m;};
const affine=(m,name)=>{matrix(m,name);if(m[3]!==0||m[7]!==0||m[11]!==0||m[15]!==1)throw Error(`${name} must be affine.`);return m;};
const mul=(a,b)=>{const m=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)
  for(let k=0;k<4;k++)m[c*4+r]+=a[k*4+r]*b[c*4+k];return m;};
function inverseAffine(m) {
  affine(m,'Reference view');
  const a=m[0],b=m[4],c=m[8],d=m[1],e=m[5],f=m[9],g=m[2],h=m[6],i=m[10];
  const det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
  if(!Number.isFinite(det)||det===0)throw Error('Reference view is singular.');
  const o=[(e*i-f*h)/det,(f*g-d*i)/det,(d*h-e*g)/det,0,
    (c*h-b*i)/det,(a*i-c*g)/det,(b*g-a*h)/det,0,
    (b*f-c*e)/det,(c*d-a*f)/det,(a*e-b*d)/det,0,0,0,0,1];
  for(let r=0;r<3;r++)o[12+r]=-(o[r]*m[12]+o[4+r]*m[13]+o[8+r]*m[14]);
  return o;
}
const transform=(m,v,w)=>[0,1,2].map(r=>m[r]*v[0]+m[4+r]*v[1]+m[8+r]*v[2]+m[12+r]*w);

// Studio's uLightIntensity is emitter area, not an arbitrary gain. A uniformly
// resized imported rig must transform area by s^2 as well as its positions.
// This bound admits Float32 rigid-transform roundoff, not anisotropic rigs.
export function validateStudioLightTransform(m) {
  affine(m,'Light transform');
  const columns=[0,4,8].map(k=>[m[k],m[k+1],m[k+2]]),dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0),
    squared=columns.map(c=>dot(c,c)),areaScale=squared.reduce((a,b)=>a+b,0)/3;
  if(!(areaScale>0)||!Number.isFinite(areaScale)||squared.some(x=>Math.abs(x-areaScale)>areaScale*1e-6)||
      [[0,1],[0,2],[1,2]].some(([a,b])=>Math.abs(dot(columns[a],columns[b]))>areaScale*1e-6))
    throw Error('Light transform must have a uniform positive scale and no shear.');
  const [a,b,c]=columns,det=a[0]*(b[1]*c[2]-b[2]*c[1])-b[0]*(a[1]*c[2]-a[2]*c[1])+c[0]*(a[1]*b[2]-a[2]*b[1]);
  if(!(det>0))throw Error('Light transform cannot reflect the captured rig.');
  return {scale:Math.sqrt(areaScale),areaScale};
}

// This is the current marble_studio.js boot resource recipe, NOT a GPU capture.
// Supply it only when the captured runtime uses that exact unchanged recipe.
// Field/shadow/thermal are disabled, not a fabricated baked environment.
export function studioFlatSamplerPacket() {
  return {schema:'turing.studio-flat-samplers.v1',textures:[
    ...[['uEmitUv',4,[0,255,128,255]],['uColorUv',5,[255,255,255,255]],
      ['uDepthUv',6,[128,255,255,255]],['uRemitUv',7,[0,0,0,0]]].map(([name,unit,pixel])=>
      ({name,unit,target:'TEXTURE_2D_ARRAY',internalFormat:'RGBA8',format:'RGBA',type:'UNSIGNED_BYTE',
        width:1,height:1,depth:2,data:[...pixel,...pixel]})),
    {name:'uFieldVolume',unit:8,target:'TEXTURE_3D',internalFormat:'RGBA8',format:'RGBA',type:'UNSIGNED_BYTE',
      width:1,height:1,depth:1,data:[0,0,0,0]},
    {name:'uShadowMaps',unit:9,target:'TEXTURE_2D_ARRAY',internalFormat:'DEPTH_COMPONENT24',
      width:1,height:1,depth:1,data:null},
    {name:'turing_ssbo_thermal_k',unit:10,target:'TEXTURE_2D',internalFormat:'R32F',format:'RED',type:'FLOAT',
      width:256,height:1,data:Array(256).fill(0)}]};
}

export function validateStudioSubjectPacket(packet,samplers) {
  if(packet?.schema==='turing.living-data-studio-composition.v1'&&
    (packet.provenance?.schema!==packet.schema||!packet.provenance.rig_identity?.startsWith('studio-scene:')||
      !/^[a-f0-9]{64}$/.test(packet.provenance.source_sha256)||!packet.provenance.world_identity))
    throw Error('Explicit Living Data composition provenance is required.');
  if(!['turing.studio-class-world-import.v1','turing.studio-subject-appearance.v1','turing.living-data-studio-composition.v1'].includes(packet?.schema)||packet.scene?.channel!=='studio')
    throw Error('An exact Studio world packet is required.');
  const {render,scene}=packet;
  for(const stage of ['vertex','fragment'])if(typeof render?.shaders?.[stage]!=='string'||
      !/^#version\s+300\s+es\b/.test(render.shaders[stage]))throw Error('Actual WebGL2 shader source is required.');
  for(const define of ['PLUCK_TRANSMISSION','PLUCK_PREMULTIPLIED_GLASS'])
    if(!new RegExp(`^\\s*#define\\s+${define}\\s+1\\b`,'m').test(render.shaders.fragment))
      throw Error(`Expected Studio ${define} shader variant.`);
  if(!Array.isArray(render.uniforms))throw Error('Captured uniform state is required.');
  const uniforms=new Map();
  for(const u of render.uniforms) {
    if(typeof u.name!=='string'||uniforms.has(u.name)||!Number.isInteger(u.type)||u.size!==1||
        !(typeof u.value==='boolean'||typeof u.value==='number'&&Number.isFinite(u.value)||finite(u.value)))
      throw Error('Uniform state is incomplete or ambiguous.');
    uniforms.set(u.name,u);
  }
  for(const name of ['uFieldGain','uShadowEnabled','uShadowLightCount','uThermalEmission','uThermalGain'])
    if(!uniforms.has(name)||Number(uniforms.get(name).value)!==0)
      throw Error(`Unsupported active Studio resource: ${name}.`);
  if(Number(uniforms.get('uRenderPass')?.value)!==0||Number(uniforms.get('uEnableEmissionDirect')?.value)!==0)
    throw Error('Capture Studio subject state after its ordinary all-material draw.');
  const expected=studioFlatSamplerPacket();
  if(samplers?.schema!==expected.schema||samplers.textures?.length!==expected.textures.length)
    throw Error('Explicit current Studio flat sampler payload is required.');
  for(const wanted of expected.textures) {
    const rows=samplers.textures.filter(x=>x?.name===wanted.name),row=rows[0];
    if(rows.length!==1||Object.keys(wanted).some(k=>JSON.stringify(row[k])!==JSON.stringify(wanted[k])))
      throw Error(`Unsupported sampler payload: ${wanted.name}.`);
    if(uniforms.get(wanted.name)?.value!==wanted.unit)throw Error(`Wrong captured sampler slot: ${wanted.name}.`);
  }
  const samplerNames=new Set([...expected.textures.map(x=>x.name),...['pbr','phong','enamel','texstack'].map(x=>'turing_ssbo_'+x)]);
  for(const u of uniforms.values())if([35678,35679,35680,35682,36289,36292,36293,36298,36299,36300,36303,36306,36307,36308,36311].includes(u.type)&&
      !samplerNames.has(u.name))throw Error(`Unsupported captured sampler ${u.name}.`);
  let count;
  for(const [name,stride]of [['pbr',16],['phong',8],['enamel',8],['texstack',16]]) {
    const table=render.materials?.[name];
    if(table?.layout?.floats_per_record!==stride||!finite(table.data)||!table.data.length||table.data.length%stride)
      throw Error(`Incomplete actual ${name} material table.`);
    const rows=table.data.length/stride;
    if(count!==undefined&&count!==rows)throw Error('Material row identities differ.');count=rows;
    if(uniforms.get(`turing_ssbo_${name}`)?.value!==['pbr','phong','enamel','texstack'].indexOf(name))
      throw Error(`Wrong material sampler slot: ${name}.`);
  }
  const subject=scene.subject,layout=subject?.phong;
  if(!Array.isArray(subject?.faces)||!subject.faces.length||subject.faces.length>200000||
      !Number.isSafeInteger(layout?.interiorCount)||layout.interiorCount<0||
      !Number.isSafeInteger(layout?.glassCount)||layout.glassCount<0||
      subject.faces.length!==layout.interiorCount+layout.glassCount)
    throw Error('Exact full subject faces and Studio interior/glass partition are required.');
  for(const face of subject.faces) {
    if(!Array.isArray(face)||face.slice(0,3).length!==3||face.slice(0,3).some(p=>!finite(p)||p.length!==3)||
        !Number.isSafeInteger(face[3])||face[3]<0||face[3]>=count||
        !Array.isArray(face[4])||face[4].length!==3||face[4].some(n=>!finite(n)||n.length!==3)||
        face[5]!==undefined&&(!Number.isInteger(face[5])||face[5]<-2147483648||face[5]>2147483647))
      throw Error('Subject attributes cannot be replayed exactly.');
  }
  for(const bubble of layout.bubbles||[])if(!Number.isSafeInteger(bubble.start)||bubble.start<0||
      !Number.isSafeInteger(bubble.count)||bubble.count<0||bubble.start+bubble.count>layout.interiorCount)
    throw Error('Invalid Studio bubble partition.');
  const pbr=render.materials.pbr.data;
  if(!finite(scene.ray?.pbr)||scene.ray.pbr.length!==pbr.length||pbr.some((x,i)=>x!==scene.ray.pbr[i]))
    throw Error('Scene and captured material identities differ.');
  inverseAffine(render.referenceCamera?.view);
  affine(subject.model,'Subject model');
  const lightCount=uniforms.get('uNumLights')?.value;
  if(!Number.isInteger(lightCount)||lightCount<0||lightCount>64)throw Error('Invalid captured light count.');
  for(let i=0;i<lightCount;i++)for(const stem of ['uLightPos','uLightSpotDir']) {
    const v=uniforms.get(`${stem}[${i}]`)?.value;
    if(!finite(v)||v.length!==3)throw Error('Captured view-space light data is missing.');
  }
  for(let i=0;i<lightCount;i++)if(!Number.isFinite(uniforms.get(`uLightIntensity[${i}]`)?.value))
    throw Error('Captured emitter area is missing.');
  const opaque=[],transparent=[];
  // Same authored bubble omission as Studio: Phong has no cavity refraction.
  const faces=[...phongInterior({interior:subject.faces.slice(0,layout.interiorCount),bubbles:layout.bubbles}),
    ...subject.faces.slice(layout.interiorCount)];
  for(const face of faces)(isOpaquePreviewMaterial(pbr[face[3]*16+5],pbr[face[3]*16+7])?opaque:transparent).push(face);
  return {opaque,transparent,uniforms,lightCount,materialCount:count,
    omittedBubbleTriangles:subject.faces.length-faces.length};
}

// Type-driven replay matches marble_phong_id_pass.copyPhongUniforms. A detached
// packet has no source GL program to pass to that live-program-only helper.
function replayUniform(gl,location,u) {
  const v=u.value,scalar=typeof v==='number'||typeof v==='boolean'?[Number(v)]:v;
  if(u.type===gl.FLOAT)return gl.uniform1fv(location,scalar);
  if([gl.INT,gl.BOOL,gl.SAMPLER_2D,gl.SAMPLER_2D_ARRAY,gl.SAMPLER_3D].includes(u.type))return gl.uniform1iv(location,scalar);
  if(u.type===gl.UNSIGNED_INT)return gl.uniform1uiv(location,scalar);
  const methods={FLOAT_VEC2:'uniform2fv',FLOAT_VEC3:'uniform3fv',FLOAT_VEC4:'uniform4fv',
    INT_VEC2:'uniform2iv',INT_VEC3:'uniform3iv',INT_VEC4:'uniform4iv',BOOL_VEC2:'uniform2iv',BOOL_VEC3:'uniform3iv',BOOL_VEC4:'uniform4iv',
    UNSIGNED_INT_VEC2:'uniform2uiv',UNSIGNED_INT_VEC3:'uniform3uiv',UNSIGNED_INT_VEC4:'uniform4uiv',
    FLOAT_MAT2:'uniformMatrix2fv',FLOAT_MAT3:'uniformMatrix3fv',FLOAT_MAT4:'uniformMatrix4fv',
    FLOAT_MAT2x3:'uniformMatrix2x3fv',FLOAT_MAT2x4:'uniformMatrix2x4fv',FLOAT_MAT3x2:'uniformMatrix3x2fv',
    FLOAT_MAT3x4:'uniformMatrix3x4fv',FLOAT_MAT4x2:'uniformMatrix4x2fv',FLOAT_MAT4x3:'uniformMatrix4x3fv'};
  const name=Object.keys(methods).find(n=>gl[n]===u.type),method=methods[name];
  if(!method)throw Error(`Unsupported Studio uniform type ${u.type}.`);
  if(method.startsWith('uniformMatrix'))gl[method](location,false,v);else gl[method](location,v);
}

// Only states this module changes. FBO, viewport, scissor box, clear values,
// incumbent VAO storage and incumbent program uniforms are never touched.
function save(gl) {
  const names=['CURRENT_PROGRAM','VERTEX_ARRAY_BINDING','ARRAY_BUFFER_BINDING','ACTIVE_TEXTURE',
    'PIXEL_UNPACK_BUFFER_BINDING','DEPTH_WRITEMASK','DEPTH_FUNC','DEPTH_RANGE','COLOR_WRITEMASK','FRONT_FACE',
    'BLEND_SRC_RGB','BLEND_DST_RGB','BLEND_SRC_ALPHA','BLEND_DST_ALPHA','BLEND_EQUATION_RGB','BLEND_EQUATION_ALPHA'];
  const s=Object.fromEntries(names.map(n=>[n,gl.getParameter(gl[n])]));
  s.enabled=Object.fromEntries(['BLEND','DEPTH_TEST','CULL_FACE','SCISSOR_TEST','STENCIL_TEST','DITHER',
    'SAMPLE_ALPHA_TO_COVERAGE','SAMPLE_COVERAGE','RASTERIZER_DISCARD','POLYGON_OFFSET_FILL'].map(n=>[n,gl.isEnabled(gl[n])]));
  s.unpack=Object.fromEntries(['UNPACK_ALIGNMENT','UNPACK_ROW_LENGTH','UNPACK_IMAGE_HEIGHT','UNPACK_SKIP_PIXELS',
    'UNPACK_SKIP_ROWS','UNPACK_SKIP_IMAGES','UNPACK_FLIP_Y_WEBGL','UNPACK_PREMULTIPLY_ALPHA_WEBGL','UNPACK_COLORSPACE_CONVERSION_WEBGL']
    .map(n=>[n,gl.getParameter(gl[n])]));
  s.attributes=[3,5,6].map(i=>gl.getVertexAttrib(i,gl.CURRENT_VERTEX_ATTRIB));
  s.textures=Array.from({length:11},(_,unit)=>{gl.activeTexture(gl.TEXTURE0+unit);return {
    texture2D:gl.getParameter(gl.TEXTURE_BINDING_2D),array:gl.getParameter(gl.TEXTURE_BINDING_2D_ARRAY),
    volume:gl.getParameter(gl.TEXTURE_BINDING_3D),sampler:gl.getParameter(gl.SAMPLER_BINDING)};});
  gl.activeTexture(s.ACTIVE_TEXTURE);return s;
}
function restore(gl,s) {
  gl.useProgram(s.CURRENT_PROGRAM);gl.bindVertexArray(s.VERTEX_ARRAY_BINDING);gl.bindBuffer(gl.ARRAY_BUFFER,s.ARRAY_BUFFER_BINDING);
  gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER,s.PIXEL_UNPACK_BUFFER_BINDING);
  for(const [name,v]of Object.entries(s.unpack))gl.pixelStorei(gl[name],v);
  for(const [name,on]of Object.entries(s.enabled))gl[on?'enable':'disable'](gl[name]);
  gl.depthMask(s.DEPTH_WRITEMASK);gl.depthFunc(s.DEPTH_FUNC);gl.depthRange(...s.DEPTH_RANGE);gl.colorMask(...s.COLOR_WRITEMASK);gl.frontFace(s.FRONT_FACE);
  gl.blendFuncSeparate(s.BLEND_SRC_RGB,s.BLEND_DST_RGB,s.BLEND_SRC_ALPHA,s.BLEND_DST_ALPHA);
  gl.blendEquationSeparate(s.BLEND_EQUATION_RGB,s.BLEND_EQUATION_ALPHA);
  s.attributes.forEach((v,k)=>{const i=[3,5,6][k];
    if(v instanceof Int32Array)gl.vertexAttribI4iv(i,v);else if(v instanceof Uint32Array)gl.vertexAttribI4uiv(i,v);else gl.vertexAttrib4fv(i,v);});
  s.textures.forEach((t,unit)=>{gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,t.texture2D);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY,t.array);gl.bindTexture(gl.TEXTURE_3D,t.volume);gl.bindSampler(unit,t.sampler);});
  gl.activeTexture(s.ACTIVE_TEXTURE);
}

export function createStudioSubjectRenderer({gl,packet,samplers}={}) {
  if(!gl||gl.isContextLost()||typeof gl.createVertexArray!=='function'||gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS)<11)
    throw Error('The original game must provide its live WebGL2 context.');
  if(gl.getParameter(gl.TRANSFORM_FEEDBACK_ACTIVE)&&!gl.getParameter(gl.TRANSFORM_FEEDBACK_PAUSED))
    throw Error('Finish the incumbent transform feedback pass before creating the subject renderer.');
  // Detach so later edits to source packets cannot silently invalidate upload.
  packet=structuredClone(packet);samplers=structuredClone(samplers);
  const plan=validateStudioSubjectPacket(packet,samplers),resources={program:null,shaders:[],buffers:[],vaos:[],textures:[]},
    bindings=[],locations=new Map(),inverseReference=inverseAffine(packet.render.referenceCamera.view);
  let closed=false;
  const dispose=()=>{if(closed)return;closed=true;
    for(const x of resources.vaos)gl.deleteVertexArray(x);for(const x of resources.buffers)gl.deleteBuffer(x);
    for(const x of resources.textures)gl.deleteTexture(x);for(const x of resources.shaders)gl.deleteShader(x);
    if(resources.program)gl.deleteProgram(resources.program);};
  const allocate=(kind,list)=>{const x=gl[`create${kind}`]();if(!x)throw Error(`Cannot allocate Studio ${kind}.`);resources[list].push(x);return x;};
  const uploadMesh=(faces,transparent)=>{
    const result={vao:allocate('VertexArray','vaos'),count:faces.length*3,centres:triangleCentres(faces),sortAxis:null,attributes:[]};
    gl.bindVertexArray(result.vao);
    const values=[[],[],[],[]];
    for(const [a,b,c,id,n,gid=0]of faces)for(const [k,p]of [a,b,c].entries()) {
      values[0].push(...p);values[1].push(...n[k]);values[2].push(id);values[3].push(gid);
    }
    for(let k=0;k<4;k++) {
      const index=[0,1,2,4][k],integer=k>=2,data=integer?new Int32Array(values[k]):new Float32Array(values[k]);
      const buffer=allocate('Buffer','buffers');result.attributes.push(buffer);
      gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);
      gl.enableVertexAttribArray(index);
      if(integer)gl.vertexAttribIPointer(index,1,gl.INT,0,0);else gl.vertexAttribPointer(index,3,gl.FLOAT,false,0,0);
    }
    if(transparent){result.indexBuffer=allocate('Buffer','buffers');gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,result.indexBuffer);}
    return result;
  };
  const uploadTexture=row=>{
    const target=gl[row.target],texture=allocate('Texture','textures');gl.activeTexture(gl.TEXTURE0+row.unit);gl.bindSampler(row.unit,null);
    gl.bindTexture(target,texture);gl.texParameteri(target,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(target,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    if(row.data===null)gl.texStorage3D(target,1,gl[row.internalFormat],row.width,row.height,row.depth);
    else {const data=row.type==='FLOAT'?new Float32Array(row.data):new Uint8Array(row.data);
      if(target===gl.TEXTURE_2D)gl.texImage2D(target,0,gl[row.internalFormat],row.width,row.height,0,gl[row.format],gl[row.type],data);
      else gl.texImage3D(target,0,gl[row.internalFormat],row.width,row.height,row.depth,0,gl[row.format],gl[row.type],data);}
    bindings.push({unit:row.unit,target,texture});
  };
  let opaque,transparent;
  const prior=save(gl);
  try {
    resources.program=gl.createProgram();if(!resources.program)throw Error('Cannot allocate Studio program.');
    for(const [stage,type]of [['vertex',gl.VERTEX_SHADER],['fragment',gl.FRAGMENT_SHADER]]) {
      const shader=gl.createShader(type);if(!shader)throw Error('Cannot allocate Studio shader.');resources.shaders.push(shader);
      gl.shaderSource(shader,packet.render.shaders[stage]);gl.compileShader(shader);
      if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader)||'Studio shader compile failed.');
      gl.attachShader(resources.program,shader);
    }
    gl.linkProgram(resources.program);
    if(!gl.getProgramParameter(resources.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(resources.program)||'Studio link failed.');
    if(gl.getProgramParameter(resources.program,gl.ACTIVE_UNIFORM_BLOCKS)!==0)throw Error('Unsupported Studio uniform block.');
    gl.useProgram(resources.program);
    const active=new Set();
    for(let i=0;i<gl.getProgramParameter(resources.program,gl.ACTIVE_UNIFORMS);i++) {
      const info=gl.getActiveUniform(resources.program,i);
      if(info.size>1&&!info.name.includes('[0]'))throw Error('Ambiguous Studio uniform array.');
      for(let j=0;j<info.size;j++) {
        const name=info.size>1?info.name.replace('[0]',`[${j}]`):info.name,u=plan.uniforms.get(name),location=gl.getUniformLocation(resources.program,name);
        if(!u||u.type!==info.type||location===null)throw Error(`Captured uniform does not match linked shader: ${name}.`);
        locations.set(name,location);active.add(name);replayUniform(gl,location,u);
      }
    }
    if(active.size!==plan.uniforms.size)throw Error('Captured and linked active uniform sets differ.');
    for(const name of ['uMV','uMVP'])if(!locations.has(name))throw Error(`Missing Studio transform ${name}.`);
    const attribs=new Map([['aPos',[0,gl.FLOAT_VEC3]],['aNorm',[1,gl.FLOAT_VEC3]],['aMatId',[2,gl.INT]],
      ['aUv',[3,gl.FLOAT_VEC2]],['aGroupId',[4,gl.INT]],['aCullImmune',[5,gl.INT]],['aVelocity',[6,gl.FLOAT_VEC3]]]);
    for(let i=0;i<gl.getProgramParameter(resources.program,gl.ACTIVE_ATTRIBUTES);i++) {
      const a=gl.getActiveAttrib(resources.program,i),expected=attribs.get(a.name);
      if(!expected||a.type!==expected[1]||a.size!==1||gl.getAttribLocation(resources.program,a.name)!==expected[0])
        throw Error(`Unsupported Studio attribute ${a.name}.`);
    }
    gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER,null);
    for(const name of Object.keys(prior.unpack))gl.pixelStorei(gl[name],name==='UNPACK_ALIGNMENT'?4:name==='UNPACK_COLORSPACE_CONVERSION_WEBGL'?gl.NONE:0);
    for(const [unit,name]of ['pbr','phong','enamel','texstack'].entries()) {
      const source=packet.render.materials[name].data,width=256,height=Math.ceil(source.length/width),data=new Float32Array(width*height);data.set(source);
      if(height>gl.getParameter(gl.MAX_TEXTURE_SIZE))throw Error('Studio material table exceeds this GPU.');
      uploadTexture({unit,target:'TEXTURE_2D',internalFormat:'R32F',format:'RED',type:'FLOAT',width,height,data});
    }
    for(const row of samplers.textures)uploadTexture(row);
    opaque=uploadMesh(plan.opaque,false);transparent=uploadMesh(plan.transparent,true);
  }catch(error){dispose();throw error;}finally{restore(gl,prior);}

  // model is the complete subject-local -> game-world transform, including
  // the caller's chosen scale. lightTransform maps captured Studio-world lights
  // into game world; omitted means they remain at captured world coordinates.
  // It permits translation/rotation/uniform positive scale. Emitter area follows
  // scale squared, and nonzero spot directions remain unit vectors.
  // Caller owns current framebuffer, viewport and draw order. Existing opaque
  // game depth is respected. Cross-object transparent sorting remains the host's
  // responsibility, exactly as with other separately drawn glass objects.
  function draw({view,projection,model,lightTransform=I,phase='all',emissionDirect=false}={}) {
    if(closed||gl.isContextLost())throw Error('Studio subject renderer is closed or context lost.');
    if(!['all','opaque','transparent'].includes(phase))throw Error('Draw phase must be all, opaque or transparent.');
    if(typeof emissionDirect!=='boolean')throw Error('Studio direct emission must be explicit boolean state.');
    if(gl.getParameter(gl.TRANSFORM_FEEDBACK_ACTIVE)&&!gl.getParameter(gl.TRANSFORM_FEEDBACK_PAUSED))
      throw Error('Finish the incumbent transform feedback pass before drawing the subject.');
    affine(view,'Game view');matrix(projection,'Game projection');affine(model,'Game model');
    const {areaScale}=validateStudioLightTransform(lightTransform);
    const mv=mul(view,model),mvp=mul(projection,mv),lightView=mul(mul(view,lightTransform),inverseReference),before=save(gl),
      originalLightView=Array.from(view).every((v,i)=>v===packet.render.referenceCamera.view[i])&&
        Array.from(lightTransform).every((v,i)=>v===I[i]);
    try {
      gl.useProgram(resources.program);
      for(const {unit,target,texture}of bindings){gl.activeTexture(gl.TEXTURE0+unit);gl.bindSampler(unit,null);gl.bindTexture(target,texture);}
      gl.uniformMatrix4fv(locations.get('uMV'),false,mv);gl.uniformMatrix4fv(locations.get('uMVP'),false,mvp);
      // Studio enables this only for its actual opaque fixture/shop meshes.
      // It is per draw, so ordinary subject and sorted glass remain unchanged.
      gl.uniform1i(locations.get('uEnableEmissionDirect'),Number(emissionDirect));
      for(let i=0;i<plan.lightCount;i++) {
        for(const [stem,w]of [['uLightPos',1],['uLightSpotDir',0]]) {
          const name=`${stem}[${i}]`,value=plan.uniforms.get(name).value;
          let transformed=originalLightView?value:transform(lightView,value,w);
          if(w===0&&!originalLightView) {const length=Math.hypot(...transformed);
            if(length>0)transformed=transformed.map(x=>x/length);}
          gl.uniform3fv(locations.get(name),transformed);
        }
        const name=`uLightIntensity[${i}]`,area=plan.uniforms.get(name).value*areaScale;
        if(!Number.isFinite(area))throw Error('Transformed emitter area is nonfinite.');
        gl.uniform1f(locations.get(name),area);
      }
      gl.vertexAttrib2f(3,0,0);gl.vertexAttribI4i(5,1,0,0,0);gl.vertexAttrib3f(6,0,0,0);
      for(const name of ['CULL_FACE','SCISSOR_TEST','STENCIL_TEST','SAMPLE_ALPHA_TO_COVERAGE','SAMPLE_COVERAGE',
        'RASTERIZER_DISCARD','POLYGON_OFFSET_FILL'])gl.disable(gl[name]);
      gl.enable(gl.DEPTH_TEST);gl.enable(gl.DITHER);gl.depthFunc(gl.LESS);gl.depthRange(0,1);gl.frontFace(gl.CCW);gl.colorMask(true,true,true,true);
      if(phase!=='transparent') {
        gl.disable(gl.BLEND);gl.depthMask(true);gl.bindVertexArray(opaque.vao);gl.drawArrays(gl.TRIANGLES,0,opaque.count);
      }
      if(phase!=='opaque'&&transparent.count) {
        gl.enable(gl.BLEND);gl.blendEquation(gl.FUNC_ADD);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
        gl.bindVertexArray(transparent.vao);const axis=[mv[2],mv[6],mv[10]];
        if(!transparent.sortAxis||axis.some((v,k)=>v!==transparent.sortAxis[k])) {
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,transparent.indexBuffer);
          gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,transparentTriangleOrder(transparent.centres,mv),gl.DYNAMIC_DRAW);transparent.sortAxis=axis;
        }
        gl.drawElements(gl.TRIANGLES,transparent.count,gl.UNSIGNED_INT,0);
      }
      return {opaqueTriangles:opaque.count/3,transparentTriangles:transparent.count/3,
        omittedBubbleTriangles:plan.omittedBubbleTriangles};
    }finally{restore(gl,before);}
  }
  // Update only an existing transparent range. Keep the shared VAO, material
  // ownership and original centroid sorting; pose changes invalidate that sort.
  function updateTransparentRange(first,faces) {
    if(closed||gl.isContextLost())throw Error('Studio subject renderer is closed or context lost.');
    if(gl.getParameter(gl.TRANSFORM_FEEDBACK_ACTIVE)&&!gl.getParameter(gl.TRANSFORM_FEEDBACK_PAUSED))
      throw Error('Finish the incumbent transform feedback pass before updating the subject.');
    if(!Number.isSafeInteger(first)||first<0||first+faces.length>plan.transparent.length)
      throw Error('Transparent update must preserve the existing triangle range.');
    const positions=new Float32Array(faces.length*9),normals=new Float32Array(faces.length*9);
    for(let i=0;i<faces.length;i++) {
      const f=faces[i],old=plan.transparent[first+i];
      if(f[3]!==old[3]||(f[5]??0)!==(old[5]??0))throw Error('Transparent material ownership cannot change with pose.');
      for(let k=0;k<3;k++) {
        if(!finite(f[k])||f[k].length!==3||!finite(f[4]?.[k])||f[4][k].length!==3)throw Error('Nonfinite transparent pose.');
        positions.set(f[k],i*9+k*3);normals.set(f[4][k],i*9+k*3);
      }
    }
    const before=gl.getParameter(gl.ARRAY_BUFFER_BINDING);
    try {
      gl.bindBuffer(gl.ARRAY_BUFFER,transparent.attributes[0]);gl.bufferSubData(gl.ARRAY_BUFFER,first*36,positions);
      gl.bindBuffer(gl.ARRAY_BUFFER,transparent.attributes[1]);gl.bufferSubData(gl.ARRAY_BUFFER,first*36,normals);
    }finally{gl.bindBuffer(gl.ARRAY_BUFFER,before);}
    const centres=triangleCentres(faces);
    for(let i=0;i<faces.length;i++)transparent.centres[first+i]=centres[i];
    transparent.sortAxis=null;
  }
  return {draw,dispose,updateTransparentRange,subjectId:packet.scene.subject.id,revision:packet.revision,
    counts:{opaqueTriangles:plan.opaque.length,transparentTriangles:plan.transparent.length,omittedBubbleTriangles:plan.omittedBubbleTriangles},
    mode:'exact-Studio-Phong-premultiplied-raster',get ready(){return !closed&&!gl.isContextLost();}};
}
