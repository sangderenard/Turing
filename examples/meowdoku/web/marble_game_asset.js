// An appearance asset for the existing game's sphere body. This carries exact
// Studio GPU geometry and materials, not another physics system or light bake.
// The 76-byte triangle layout is the incumbent marble_object.geometrySha256
// layout: six float32 vectors followed by one uint32 material ID, little-endian.
import {studioFlatSamplerPacket,validateStudioSubjectPacket} from './marble_studio_subject_renderer.js';

const clone=x=>JSON.parse(JSON.stringify(x,(_key,v)=>ArrayBuffer.isView(v)?Array.from(v):v));
const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
const encode=bytes=>{let out='';for(let at=0;at<bytes.length;at+=32768)out+=String.fromCharCode(...bytes.subarray(at,at+32768));return btoa(out);};
const decode=text=>Uint8Array.from(atob(text),c=>c.charCodeAt(0));
const utf8=text=>new TextEncoder().encode(text);
const assert=(ok,message)=>{if(!ok)throw Error(message);};
function admitSphere(subject,object) {
  assert(object?.design?.['base-shape']==='marble','The existing launcher accepts spherical marble appearance, not die geometry.');
  assert(Number(object.design.coverage)===0&&Number(object.design['band-count'])===0,
    'This transfer currently requires the original uncoated spherical outer shell.');
  assert(Number.isSafeInteger(subject.phong?.glassCount)&&subject.phong.glassCount>0,
    'A nonempty authored spherical exterior is required.');
  for(const face of subject.faces.slice(subject.phong.interiorCount))for(const p of face.slice(0,3))
    assert(Math.abs(Math.hypot(...p)-1)<1e-6,'The displayed exterior is not the existing unit-radius sphere.');
}

export async function createStudioGameAsset(packet,{name='Studio marble'}={}) {
  const samplers=studioFlatSamplerPacket();validateStudioSubjectPacket(packet,samplers);
  const object=packet.scene.authoring?.object;
  assert(typeof name==='string'&&name.trim()&&name.length<=120,'A bounded asset name is required.');
  const subject=packet.scene.subject,faces=subject.faces;
  // Admission is deliberately bounded to the existing uncoated sphere shell.
  // The game keeps its own radius, mass, contacts, inventory and state tensors.
  admitSphere(subject,object);
  const bytes=new Uint8Array(faces.length*76),view=new DataView(bytes.buffer),groups=[];let at=0;
  for(const face of faces){
    for(const vector of [...face.slice(0,3),...face[4]])for(const value of vector){
      assert(Number.isFinite(Math.fround(value)),'Studio coordinate exceeds its float32 GPU format.');
      view.setFloat32(at,value,true);at+=4;}
    view.setUint32(at,face[3],true);at+=4;groups.push(face[5]??0);
  }
  const body={schema:'turing.studio-game-asset.v1',name:name.trim(),
    mesh:{format:'meowdoku-geometry-sha256-f32le-76',triangles:faces.length,
      sha256:await digest(bytes),bytesBase64:encode(bytes),groups,
      phong:clone(subject.phong)},
    render:clone(packet.render),samplers,
    source:{revision:packet.revision,subjectId:subject.id,frame:clone(packet.scene.frame),
      model:clone(subject.model),camera:clone(packet.scene.camera),object:clone(object)},
    contract:{physics:'existing-game-sphere-unchanged',radius:'existing-projectile-archetype',
      orientation:'fixed-authored-pose-no-engine-spin-published',geometry:'exact-Studio-float32-GPU-attributes',
      lighting:'captured-Studio-Phong-rig',rayTracedBake:false}};
  return {...body,identity:'studio-marble:'+await digest(utf8(JSON.stringify(body)))};
}

export async function hydrateStudioGameAsset(asset) {
  assert(asset?.schema==='turing.studio-game-asset.v1','Unknown Studio game asset.');
  const {identity,...body}=asset;
  assert(identity==='studio-marble:'+await digest(utf8(JSON.stringify(body))),'Studio asset identity does not match its contents.');
  const m=asset.mesh;
  assert(m?.format==='meowdoku-geometry-sha256-f32le-76'&&Number.isSafeInteger(m.triangles)&&m.triangles>0&&m.triangles<=200000,
    'Invalid bounded Studio mesh.');
  assert(typeof m.bytesBase64==='string'&&m.bytesBase64.length<=Math.ceil(m.triangles*76/3)*4,
    'Invalid Studio mesh payload size.');
  const bytes=decode(m.bytesBase64);
  assert(bytes.length===m.triangles*76&&await digest(bytes)===m.sha256,'Studio mesh bytes changed after capture.');
  assert(Array.isArray(m.groups)&&m.groups.length===m.triangles&&m.groups.every(x=>Number.isInteger(x)&&x>=-2147483648&&x<=2147483647),
    'Studio semantic groups are incomplete.');
  assert(asset.source.object?.design?.['base-shape']==='marble'&&asset.contract?.physics==='existing-game-sphere-unchanged',
    'The asset cannot change the original game body contract.');
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),faces=[];let at=0;
  for(let i=0;i<m.triangles;i++){
    const vectors=[];for(let j=0;j<6;j++){const p=[];for(let k=0;k<3;k++){p.push(view.getFloat32(at,true));at+=4;}vectors.push(p);}
    const id=view.getUint32(at,true);at+=4;faces.push([...vectors.slice(0,3),id,vectors.slice(3),m.groups[i]]);
  }
  const packet={schema:'turing.studio-subject-appearance.v1',revision:asset.source.revision,
    scene:{channel:'studio',frame:clone(asset.source.frame),camera:clone(asset.source.camera),
      subject:{id:asset.source.subjectId,faces,model:clone(asset.source.model),phong:clone(m.phong)},
      ray:{pbr:clone(asset.render.materials.pbr.data)},authoring:{object:clone(asset.source.object)}},render:clone(asset.render)};
  validateStudioSubjectPacket(packet,asset.samplers);
  admitSphere(packet.scene.subject,asset.source.object);
  return {identity:asset.identity,name:asset.name,packet,samplers:clone(asset.samplers),
    source:{designedMesh:{center:[0,0,0],scale:1},geometry:{faces},materials:packet.render.materials},
    contract:clone(asset.contract),meshSha256:m.sha256};
}
