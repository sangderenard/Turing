// A detached rendering/authoring packet for importing Studio into the original
// generated class world. This neither builds a second world nor changes physics.
// Scene triangles, source identities, material tables and the actual compiled
// Phong program are retained. The class-world consumer must explicitly realize
// placement, camera/light transforms, renderer ownership and contact semantics.
const clone=value=>structuredClone(value);
const finite=a=>(Array.isArray(a)||ArrayBuffer.isView(a))&&Array.from(a).every(Number.isFinite);
const scalar=value=>ArrayBuffer.isView(value)?Array.from(value):Array.isArray(value)?value.map(scalar):value;
const hash=async text=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))),x=>x.toString(16).padStart(2,'0')).join('');

export function captureStudioProgramState(gl,program,tables) {
  if(!gl||gl.isContextLost()||!program||!gl.getProgramParameter(program,gl.LINK_STATUS))
    throw Error('A live linked Studio Phong program is required.');
  const shaders={};
  for(const shader of gl.getAttachedShaders(program)||[]) {
    const type=gl.getShaderParameter(shader,gl.SHADER_TYPE),name=type===gl.VERTEX_SHADER?'vertex':type===gl.FRAGMENT_SHADER?'fragment':null;
    if(!name||shaders[name])throw Error('Ambiguous Studio shader ownership.');
    shaders[name]=gl.getShaderSource(shader);
  }
  if(!shaders.vertex||!shaders.fragment)throw Error('Actual Studio shader source is unavailable.');
  const uniforms=[];
  for(let i=0;i<gl.getProgramParameter(program,gl.ACTIVE_UNIFORMS);i++) {
    const info=gl.getActiveUniform(program,i);
    if(info.size>1&&!info.name.includes('[0]'))throw Error('Ambiguous active Studio uniform array.');
    for(let j=0;j<info.size;j++) {
      const name=info.size>1?info.name.replace('[0]',`[${j}]`):info.name;
      const value=gl.getUniform(program,gl.getUniformLocation(program,name));
      if(value===null)throw Error(`Studio uniform ${name} is unavailable.`);
      uniforms.push({name,type:info.type,size:1,value:scalar(value)});
    }
  }
  const materials={};
  for(const [name,resource]of Object.entries(tables||{}))materials[name]={layout:clone(resource.table),data:Array.from(resource.data)};
  return {shaders,uniforms,materials};
}

export async function studioWorldPacket(capture,programState,{classIdentity,placementM=[0,0,0]}={}) {
  const state=capture?.state;
  if(state?.schema!=='turing.studio-visible-scene.v1'||state.channel!=='studio'||
      !Number.isSafeInteger(capture.revision)||!state.room||!state.subject||
      !finite(state.camera?.view)||state.camera.view.length!==16)
    throw Error('Import needs the exact displayed Studio scene, room, subject and camera.');
  if(typeof classIdentity!=='string'||!classIdentity.startsWith('python:')||classIdentity.length>512)
    throw Error('Import needs an explicit real class identity from the original class interpreter.');
  if(!finite(placementM)||placementM.length!==3||placementM.some(x=>Math.abs(x)>1e6))
    throw Error('Invalid placement in the original world.');
  const render=clone(programState),names=['pbr','phong','enamel','texstack'];let count=null;
  for(const name of names) {
    const table=render?.materials?.[name],stride=table?.layout?.floats_per_record;
    if(!Number.isSafeInteger(stride)||stride<1||!finite(table.data)||!table.data.length||table.data.length%stride)
      throw Error(`Actual Studio ${name} material table is incomplete.`);
    const rows=table.data.length/stride;if(count!==null&&rows!==count)throw Error('Studio material table identities do not align.');count=rows;
  }
  const pbr=render.materials.pbr,stride=pbr.layout.floats_per_record;
  if(stride<16||state.ray.pbr.length!==count*16)throw Error('Captured scene and material table counts differ.');
  for(let row=0;row<count;row++)for(let col=0;col<16;col++)
    if(state.ray.pbr[row*16+col]!==pbr.data[row*stride+col])throw Error('Studio materials changed after the displayed capture.');
  if(!Array.isArray(render.uniforms)||render.uniforms.some(x=>typeof x.name!=='string'||
      !Number.isInteger(x.type)||!Number.isInteger(x.size)||
      (typeof x.value==='number'?!Number.isFinite(x.value):typeof x.value==='boolean'?false:!finite(x.value))))
    throw Error('Studio uniform state is incomplete.');
  for(const name of ['vertex','fragment'])if(typeof render.shaders?.[name]!=='string'||
      !render.shaders[name].startsWith('#version 300 es'))throw Error('Expected the actual Studio WebGL2 shader.');
  const scene=clone(state),shaderHashes={vertex:await hash(render.shaders.vertex),fragment:await hash(render.shaders.fragment)};
  return {schema:'turing.studio-class-world-import.v1',revision:capture.revision,
    classIdentity,placement:{translationM:Array.from(placementM),axes:'Y-up',scale:1},
    sourceFrame:clone(state.frame),scene,render:{...render,shaderHashes,referenceCamera:clone(state.camera)},
    ownership:{world:'existing-MechanicalCreature-world',engine:'existing-generated-worker-and-matched-compiled-products',
      scene:'exact-displayed-Studio-snapshot',contactAdmission:'consumer-must-declare-existing-engine-shapes',
      photographicCamera:'remains-linked-until-explicitly-unhooked'},
    ready:false,remaining:['class-world insertion','placed material renderer',
      'sampler resource and opaque/transparent draw-pass realization',
      'world contact realization','browser validation']};
}
