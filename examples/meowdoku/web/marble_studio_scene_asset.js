// Lossless transfer of the displayed Studio into the incumbent game's GL.
// No second world/loop/physics, no mesh reduction, and no synthetic materials.
import {createStudioSubjectRenderer,studioFlatSamplerPacket,validateStudioSubjectPacket,
  validateStudioLightTransform} from './marble_studio_subject_renderer.js';
import {isOpaquePreviewMaterial} from './marble_glass_preview.js';

const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const clone=x=>JSON.parse(JSON.stringify(x,(_k,v)=>ArrayBuffer.isView(v)?Array.from(v):v));
const assert=(ok,message)=>{if(!ok)throw Error(message);};
const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
const textBytes=x=>new TextEncoder().encode(JSON.stringify(x));
const encode=bytes=>{let s='';for(let i=0;i<bytes.length;i+=32768)s+=String.fromCharCode(...bytes.subarray(i,i+32768));return btoa(s);};
const decode=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const mul=(a,b)=>{const m=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)
  for(let k=0;k<4;k++)m[c*4+r]+=a[k*4+r]*b[c*4+k];return m;};
const point=(m,v)=>[0,1,2].map(r=>m[r]*v[0]+m[4+r]*v[1]+m[8+r]*v[2]+m[12+r]);
const without=(o,keys)=>Object.fromEntries(Object.entries(o).filter(([k])=>!keys.includes(k)));

// JSON.stringify of a typed array from the original capture produces numeric
// object keys. Admit exactly 0..15, not a guessed Object.values ordering.
export function studioSceneMatrix(value,name='Studio matrix') {
  if(value&&typeof value==='object'&&!Array.isArray(value)&&!ArrayBuffer.isView(value)) {
    assert(Object.keys(value).length===16&&Array.from({length:16},(_,i)=>Object.hasOwn(value,String(i))).every(Boolean),`${name} has incomplete numeric keys.`);
    value=Array.from({length:16},(_,i)=>value[i]);
  }
  assert((Array.isArray(value)||ArrayBuffer.isView(value))&&value.length===16&&Array.from(value).every(Number.isFinite),`${name} needs 16 finite values.`);
  return Array.from(value);
}
function normalisedPacket(packet) {
  assert(packet?.schema==='turing.studio-class-world-import.v1','An actual full Studio capture is required.');
  const render={...packet.render,referenceCamera:{...packet.render?.referenceCamera,
    view:studioSceneMatrix(packet.render?.referenceCamera?.view,'Reference camera')}};
  const scene={...packet.scene,camera:{...packet.scene?.camera,view:studioSceneMatrix(packet.scene?.camera?.view,'Scene camera')},
    subject:{...packet.scene?.subject,model:studioSceneMatrix(packet.scene?.subject?.model,'Subject pose')}};
  return {...packet,scene,render};
}
const packetFor=(packet,faces,id)=>({schema:'turing.studio-subject-appearance.v1',revision:packet.revision,
  render:packet.render,scene:{channel:'studio',ray:{pbr:packet.render.materials.pbr.data},
    subject:{id,faces,model:I,phong:{interiorCount:faces.length,glassCount:0,bubbles:[]}}}});
function validateScene(packet,samplers) {
  const subject=validateStudioSubjectPacket(packet,samplers),scene=packet.scene;
  assert(Number.isFinite(scene.frame?.unitM)&&scene.frame.unitM>0,'A positive Studio metre conversion is required.');
  assert(Array.isArray(scene.parts)&&scene.parts.length>0&&scene.parts.length<=10000,'Bounded authored Studio parts are required.');
  const ids=new Set();let offset=0;
  for(const p of scene.parts) {
    assert(typeof p.id==='string'&&p.id&&!ids.has(p.id)&&typeof p.bodyId==='string'&&p.bodyId,'Unique source part/body IDs are required.');
    ids.add(p.id);
    assert(p.firstTriangle===offset&&p.triangleCount===p.faces?.length,'Studio part source ranges changed.');
    assert(/^(room|backdrop|table|gobo|fixture|stand|shop)\//.test(p.id),`Unsupported Studio draw owner: ${p.id}.`);
    validateStudioSubjectPacket(packetFor(packet,p.faces,p.id),samplers);
    if(scene.triangleSources)for(let j=0;j<p.faces.length;j++) {
      const s=scene.triangleSources[offset+j];
      assert(s?.partId===p.id&&s.bodyId===p.bodyId&&s.triangle===j,'Studio triangle source ownership changed.');
    }
    offset+=p.faces.length;
  }
  assert(offset<=200000&&(!scene.triangleSources||scene.triangleSources.length===offset),'Invalid static source triangle count.');
  return {subject,staticTriangles:offset};
}

// Source doubles must survive: Studio transparentSet transforms the subject
// before float32 upload. Rounding before that CPU transform changes attributes.
// This 152-byte layout retains 18 float64 values, uint32 material and int32 group.
// It deliberately omits both enormous derived ray-geometry copies.
export async function createStudioSceneAsset(source,{name='Marble Studio',samplers=studioFlatSamplerPacket()}={}) {
  const packet=normalisedPacket(source),plan=validateScene(packet,samplers);
  assert(typeof name==='string'&&name.trim()&&name.length<=120,'A bounded Studio asset name is required.');
  const scene=packet.scene,faces=[...scene.parts.flatMap(p=>p.faces),...scene.subject.faces],bytes=new Uint8Array(faces.length*152),
    groupPresence=new Uint8Array(Math.ceil(faces.length/8)),v=new DataView(bytes.buffer);let at=0;
  for(const [index,face]of faces.entries()) {
    assert(face.length===5||face.length===6,'Unsupported extra Studio face attributes.');
    if(face.length===6)groupPresence[index>>3]|=1<<(index&7);
    for(const vector of [...face.slice(0,3),...face[4]])for(const x of vector){v.setFloat64(at,x,true);at+=8;}
    v.setUint32(at,face[3],true);v.setInt32(at+4,face[5]??0,true);at+=8;
  }
  const body={schema:'turing.studio-scene-asset.v1',name:name.trim(),revision:packet.revision,
    geometry:{format:'studio-source-f64le-152',triangles:faces.length,staticTriangles:plan.staticTriangles,
      sha256:await digest(bytes),bytesBase64:encode(bytes),groupPresenceBase64:encode(groupPresence)},
    render:clone(packet.render),samplers:clone(samplers),
    scene:{...clone(without(scene,['parts','subject','ray','triangleSources'])),
      parts:scene.parts.map(p=>clone(without(p,['faces']))),
      subject:clone(without(scene.subject,['faces','ray']))},
    source:{classIdentity:packet.classIdentity,ownership:clone(packet.ownership||{}),sourceFrame:clone(packet.sourceFrame||scene.frame)},
    contract:{geometry:'complete-captured-source-faces',physics:'host-owned-static-contact-admission',
      subject:'fixed-captured-pose',lighting:'captured-Studio-Phong-rig',rayTracedBake:false,
      drawOrder:'Studio-opaque-fixture-emission-subject-shared-transparent'}};
  return {...body,identity:'studio-scene:'+await digest(textBytes(body))};
}

export async function verifyStudioSceneAssetIdentity(asset) {
  assert(asset?.schema==='turing.studio-scene-asset.v1','Unknown Studio scene asset.');
  const {identity,...body}=asset;
  assert(identity==='studio-scene:'+await digest(textBytes(body)),'Studio scene asset identity does not match its contents.');
  return identity;
}

export async function hydrateStudioSceneAsset(asset) {
  // Detach the serialized strings/metadata before the first asynchronous hash.
  // Otherwise a caller can change material/range state between verification
  // and decode. This is one binary-string copy, not a decoded-mesh clone.
  asset=structuredClone(asset);
  const identity=await verifyStudioSceneAssetIdentity(asset),g=asset.geometry;
  assert(g?.format==='studio-source-f64le-152'&&Number.isSafeInteger(g.triangles)&&g.triangles>0&&g.triangles<=400000&&
    Number.isSafeInteger(g.staticTriangles)&&g.staticTriangles>0&&g.staticTriangles<g.triangles,'Invalid bounded Studio scene geometry.');
  assert(typeof g.bytesBase64==='string'&&g.bytesBase64.length===Math.ceil(g.triangles*152/3)*4,'Invalid Studio geometry payload size.');
  const bytes=decode(g.bytesBase64);
  assert(bytes.length===g.triangles*152&&await digest(bytes)===g.sha256,'Studio source geometry bytes changed.');
  const presence=decode(g.groupPresenceBase64);
  assert(presence.length===Math.ceil(g.triangles/8),'Studio source group-presence flags are incomplete.');
  const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),faces=[];let at=0;
  for(let i=0;i<g.triangles;i++) {
    const vectors=[];for(let n=0;n<6;n++){const p=[];for(let k=0;k<3;k++){p.push(v.getFloat64(at,true));at+=8;}vectors.push(p);}
    const material=v.getUint32(at,true),group=v.getInt32(at+4,true);at+=8;
    faces.push([...vectors.slice(0,3),material,vectors.slice(3),...(presence[i>>3]&(1<<(i&7))?[group]:[])]);
  }
  const scene=clone(asset.scene),render=clone(asset.render);
  scene.parts=scene.parts.map(p=>({...p,faces:faces.slice(p.firstTriangle,p.firstTriangle+p.triangleCount)}));
  scene.subject.faces=faces.slice(g.staticTriangles);scene.ray={pbr:render.materials.pbr.data};
  scene.triangleSources=scene.parts.flatMap(p=>p.faces.map((_,triangle)=>({partId:p.id,bodyId:p.bodyId,triangle})));
  const packet=normalisedPacket({schema:'turing.studio-class-world-import.v1',revision:asset.revision,scene,render,...clone(asset.source)});
  const counts=validateScene(packet,asset.samplers);assert(counts.staticTriangles===g.staticTriangles,'Studio source ranges do not cover the static geometry.');
  return {identity,name:asset.name,packet,samplers:clone(asset.samplers),contract:clone(asset.contract),geometrySha256:g.sha256};
}

const bounds=(faces,model)=>{
  const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
  for(const face of faces)for(const p of face.slice(0,3)){const q=point(model,p);for(let k=0;k<3;k++){min[k]=Math.min(min[k],q[k]);max[k]=Math.max(max[k],q[k]);}}
  return {min,max};
};
const union=values=>({min:[0,1,2].map(k=>Math.min(...values.map(v=>v.min[k]))),max:[0,1,2].map(k=>Math.max(...values.map(v=>v.max[k])))});
function transformedFace(face,model) {
  const normal=n=>{const p=[0,1,2].map(r=>model[r]*n[0]+model[4+r]*n[1]+model[8+r]*n[2]),length=Math.hypot(...p);
    assert(length>0,'Studio transformed transparent normal is zero.');return p.map(x=>x/length);};
  return [...face.slice(0,3).map(p=>point(model,p)),face[3],face[4].map(normal),face[5]??0];
}

// The host's normal box opening removes a rectangle from its front wall.
// Apply that same explicit opening only to a derived rendering copy; capture,
// asset identity, source part/triangle IDs and every other face remain intact.
export function studioSceneEntranceParts(scene,entrance=null) {
  if(entrance===null)return {parts:scene.parts,entrance:null};
  assert(entrance&&entrance.partId==='room/front'&&Number.isFinite(entrance.widthM)&&entrance.widthM>0&&
    Number.isFinite(entrance.heightM)&&entrance.heightM>0,'An explicit positive front doorway is required.');
  const unit=scene.frame.unitM,centerX=entrance.centerX??0,bottom=scene.room.floorY,
    left=centerX-entrance.widthM/(2*unit),right=centerX+entrance.widthM/(2*unit),top=bottom+entrance.heightM/unit;
  assert(Number.isFinite(centerX)&&left>scene.room.leftX&&right<scene.room.rightX&&top<scene.room.ceilingY,'Doorway must fit inside the source front wall.');
  const part=scene.parts.find(p=>p.id===entrance.partId);assert(part,'Source front wall is absent.');
  for(const f of part.faces)for(const p of f.slice(0,3))assert(p[2]===scene.room.frontZ,'Doorway requires the actual flat front wall.');
  const clip=(poly,axis,edge,sign)=>{
    const result=[];
    for(let i=0;i<poly.length;i++) {
      const a=poly[i],b=poly[(i+1)%poly.length],da=sign*(a.p[axis]-edge),db=sign*(b.p[axis]-edge);
      if(da>=0)result.push(a);
      if((da<0&&db>0)||(da>0&&db<0)) {
        const t=da/(da-db);result.push({p:a.p.map((v,k)=>v+t*(b.p[k]-v)),n:a.n.map((v,k)=>v+t*(b.n[k]-v))});
      }
    }
    return result;
  };
  const faces=[],sources=[];let changedSourceTriangles=0;
  for(const [triangle,face]of part.faces.entries()) {
    const poly=face.slice(0,3).map((p,k)=>({p,n:face[4][k]}));
    if(poly.every(v=>v.p[0]<=left)||poly.every(v=>v.p[0]>=right)||poly.every(v=>v.p[1]>=top)||poly.every(v=>v.p[1]<=bottom)) {
      faces.push(face);sources.push({partId:part.id,bodyId:part.bodyId,triangle,derivedTriangle:0,clipped:false});continue;
    }
    changedSourceTriangles++;
    // Disjoint rectangle-complement regions avoid duplicate triangles.
    const middle=clip(clip(poly,0,left,1),0,right,-1),regions=[clip(poly,0,left,-1),clip(poly,0,right,1),clip(middle,1,top,1),clip(middle,1,bottom,-1)];
    let derivedTriangle=0;
    for(const region of regions)for(let i=1;i+1<region.length;i++) {
      const vs=[region[0],region[i],region[i+1]],a=vs[0].p,b=vs[1].p,c=vs[2].p;
      if((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])===0)continue;
      faces.push([...vs.map(v=>v.p),face[3],vs.map(v=>v.n),face[5]??0]);
      sources.push({partId:part.id,bodyId:part.bodyId,triangle,derivedTriangle:derivedTriangle++,clipped:true});
    }
  }
  return {parts:scene.parts.map(p=>p===part?{...p,faces}:p),entrance:{...clone(entrance),centerX,
    sourceBounds:{min:[left,bottom,scene.room.frontZ],max:[right,top,scene.room.frontZ]},
    sourceTriangles:part.faces.length,derivedTriangles:faces.length,changedSourceTriangles,sources}};
}

export function studioSceneDrawPlan(source,{placement=I,samplers=studioFlatSamplerPacket(),entrance=null}={}) {
  const packet=normalisedPacket(source),validated=validateScene(packet,samplers),scene=packet.scene;
  placement=studioSceneMatrix(placement,'World placement');const rigid=validateStudioLightTransform(placement);
  assert(Math.abs(rigid.scale-1)<1e-6,'Scene placement must be rigid; source unitM supplies the scale.');
  const unit=scene.frame.unitM,scale=[unit,0,0,0,0,unit,0,0,0,0,unit,0,0,0,0,1],model=mul(placement,scale),subjectModel=mul(model,scene.subject.model);
  const opaque=face=>isOpaquePreviewMaterial(packet.render.materials.pbr.data[face[3]*16+5],packet.render.materials.pbr.data[face[3]*16+7]);
  const ordinary=[],emissive=[],transparent=[];
  // Match current Studio render(): table/gobos, stands, room, fixtures/shop,
  // subject; furniture/shop + subject glass then shares one world-space sort.
  const order=p=>/^(table|gobo)\//.test(p.id)?0:p.id.startsWith('stand/')?1:/^(room|backdrop)\//.test(p.id)?2:p.id.startsWith('fixture/')?3:4;
  const derived=studioSceneEntranceParts(scene,entrance),parts=[...derived.parts].sort((a,b)=>order(a)-order(b));
  for(const part of parts)for(const face of part.faces) {
    const forcedOpaque=/^(room|backdrop|fixture)\//.test(part.id),emission=/^(fixture|shop)\//.test(part.id);
    assert(!forcedOpaque||opaque(face),`Studio ${part.id} uses unconditional opaque draw; this captured material needs a dedicated pass.`);
    if(opaque(face))(emission?emissive:ordinary).push(face);else transparent.push(face);
  }
  transparent.push(...validated.subject.transparent.map(f=>transformedFace(f,scene.subject.model)));
  const groups=[{id:'studio/static-opaque',faces:ordinary,model,phase:'opaque',emissionDirect:false},
    {id:'studio/fixture-emission',faces:emissive,model,phase:'opaque',emissionDirect:true},
    {id:scene.subject.id,faces:validated.subject.opaque,model:subjectModel,phase:'opaque',emissionDirect:false},
    {id:'studio/shared-transparent',faces:transparent,model,phase:'transparent',emissionDirect:false}];
  const partBounds=scene.parts.map(p=>({id:p.id,bodyId:p.bodyId,bounds:bounds(p.faces,model),firstTriangle:p.firstTriangle,triangleCount:p.faces.length}));
  const subjectBounds=bounds(scene.subject.faces,subjectModel);
  return {packet,groups,placement,model,subjectModel,subjectBounds,partBounds,entrance:derived.entrance,bounds:union([...partBounds.map(p=>p.bounds),subjectBounds]),
    counts:{parts:scene.parts.length,staticTriangles:validated.staticTriangles,subjectTriangles:scene.subject.faces.length,
      renderedStaticTriangles:derived.parts.reduce((n,p)=>n+p.faces.length,0),
      opaqueTriangles:ordinary.length+emissive.length+validated.subject.opaque.length,transparentTriangles:transparent.length,
      omittedBubbleTriangles:validated.subject.omittedBubbleTriangles}};
}

// Pass a hydrated asset. Hydration is async/hash-checked once, never per frame.
// The original game owns FBO, viewport, loop, contacts and visible pass timing.
export function createStudioSceneRenderer({gl,asset,samplers=asset?.samplers,placement=I,entrance=null}={}) {
  assert(asset?.packet&&asset.identity?.startsWith('studio-scene:'),'Hydrate the Studio scene asset before renderer creation.');
  const plan=studioSceneDrawPlan(asset.packet,{placement,samplers,entrance}),draws=[];let closed=false;
  const dispose=()=>{if(closed)return;closed=true;for(const d of draws)d.renderer.dispose();};
  try {for(const group of plan.groups)if(group.faces.length)draws.push({...group,
    renderer:createStudioSubjectRenderer({gl,packet:packetFor(plan.packet,group.faces,group.id),samplers})});}
  catch(error){dispose();throw error;}
  function draw({view,projection,phase='all'}={}) {
    assert(!closed,'Studio scene renderer is closed.');assert(['all','opaque','transparent'].includes(phase),'Invalid Studio scene draw phase.');
    for(const d of draws)if(phase==='all'||phase===d.phase)d.renderer.draw({view,projection,model:d.model,
      lightTransform:plan.model,phase:d.phase,emissionDirect:d.emissionDirect});
    return {...plan.counts};
  }
  return {draw,dispose,identity:asset.identity,bounds:plan.bounds,partBounds:plan.partBounds,
    subjectPose:Array.from(plan.subjectModel),subjectBounds:plan.subjectBounds,placement:Array.from(plan.placement),
    entrance:plan.entrance,counts:plan.counts,mode:'exact-Studio-static-scene-Phong',get ready(){return !closed&&draws.every(d=>d.renderer.ready);}};
}
