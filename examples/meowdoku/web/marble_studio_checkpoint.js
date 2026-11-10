// A forward-replay checkpoint envelope, NOT a recording/movie exporter.
// Reuses the exact existing MRP1 ray-scene upload and named native state ABI.
// No camera, physics, worker, scene authoring or file writes happen here.
import {decodeStateStart,decodeStateOrientations} from "./compute_state.js";

const schema="turing.studio-physics-checkpoint.v1";
const encoder=new TextEncoder(),decoder=new TextDecoder("utf-8",{fatal:true});
const digestPattern=/^[0-9a-f]{64}$/;
const maxBytes=64*1024*1024;
const canonical=value=>{
  if(value===null||typeof value==="string"||typeof value==="boolean")return JSON.stringify(value);
  if(typeof value==="number"){
    if(!Number.isFinite(value))throw new TypeError("Checkpoint contains a nonfinite number");
    return JSON.stringify(value);
  }
  if(Array.isArray(value))return `[${value.map(canonical).join(",")}]`;
  if(value&&Object.getPrototypeOf(value)===Object.prototype)
    return `{${Object.keys(value).sort().map(key=>`${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
  throw new TypeError("Checkpoint metadata must be plain finite JSON");
};
const sha=async bytes=>[...new Uint8Array(await crypto.subtle.digest("SHA-256",bytes))]
  .map(x=>x.toString(16).padStart(2,"0")).join("");
const base64=bytes=>{
  let text="";for(let i=0;i<bytes.length;i+=0x8000)text+=String.fromCharCode(...bytes.subarray(i,i+0x8000));
  return btoa(text);
};
const unbase64=text=>Uint8Array.from(atob(text),c=>c.charCodeAt(0));
function vector(value,name) {
  if(!Array.isArray(value)||value.length!==3||!value.every(Number.isFinite))throw new TypeError(`Invalid ${name}`);
  return [...value];
}

// The existing photo packet uses structure-of-arrays, while saved geometry's
// existing SHA receipt is triangle-interleaved. Reorder bytes ONLY to verify
// that receipt. The archived MRP1 payload itself is retained byte-for-byte.
async function inspectScene(bytes,savedObject,objectDigest) {
  if(bytes.length<16||bytes.length>maxBytes||decoder.decode(bytes.subarray(0,4))!=="MRP1")
    throw new Error("Expected a bounded existing MRP1 scene packet");
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  const jsonBytes=view.getUint32(4,true),count=view.getUint32(8,true),offset=16+jsonBytes;
  if(!count||view.getUint32(12,true)!==0||offset+76*count!==bytes.length)throw new Error("Incomplete MRP1 triangle payload");
  const config=JSON.parse(decoder.decode(bytes.subarray(16,offset)));
  if(!digestPattern.test(objectDigest)||config.object_digest!==objectDigest)
    throw new Error("Ray packet does not reference the saved subject");
  const fixtures=config.fixtures;
  if(!fixtures||!Array.isArray(fixtures.vertices)||!Array.isArray(fixtures.material_ids)||!Array.isArray(fixtures.pbr)||
      fixtures.vertices.length%9||fixtures.material_ids.length!==fixtures.vertices.length/9||fixtures.pbr.length%16)
    throw new Error("Checkpoint needs the complete existing ray-scene triangles and PBR table");
  if(!fixtures.vertices.every(Number.isFinite)||!fixtures.pbr.every(Number.isFinite)||
      !fixtures.material_ids.every(id=>Number.isInteger(id)&&id>=0&&id<fixtures.pbr.length/16))
    throw new Error("Invalid ray-scene geometry or material identity");
  const interleaved=new Uint8Array(count*76);
  const pbr=savedObject.materials?.tables?.pbr;
  if(!pbr||!Number.isInteger(pbr.stride)||pbr.stride<16||!Array.isArray(pbr.data)||pbr.data.length%pbr.stride||
      !pbr.data.every(Number.isFinite))
    throw new Error("Checkpoint needs the saved subject's material table");
  for(let i=0;i<count;i++) {
    interleaved.set(bytes.subarray(offset+36*i,offset+36*(i+1)),76*i);
    interleaved.set(bytes.subarray(offset+36*count+36*i,offset+36*count+36*(i+1)),76*i+36);
    interleaved.set(bytes.subarray(offset+72*count+4*i,offset+72*count+4*(i+1)),76*i+72);
    if(view.getUint32(offset+72*count+4*i,true)>=pbr.data.length/pbr.stride)
      throw new Error("Subject triangle references an absent saved material");
    for(let k=0;k<18;k++)if(!Number.isFinite(view.getFloat32(offset+(k<9?36*i+4*k:36*count+36*i+4*(k-9)),true)))
      throw new Error("Nonfinite subject triangle or normal");
  }
  if(!digestPattern.test(savedObject.geometry_sha256)||await sha(interleaved)!==savedObject.geometry_sha256)
    throw new Error("Ray subject triangles differ from the saved geometry receipt");
  return {subjectTriangles:count,sceneTriangles:fixtures.material_ids.length,packetSha256:await sha(bytes)};
}

function validateClock(handoff,manifest) {
  const rec=manifest.sizes?.[handoff.size],clock=handoff.clock;
  if(!rec||!clock||!Number.isFinite(clock.dt)||clock.dt<=0||clock.dt!==manifest.law.dt||
      !Number.isSafeInteger(clock.snapEvery)||clock.snapEvery<1||
      !Number.isSafeInteger(handoff.frame)||handoff.frame<0||!Number.isSafeInteger(handoff.step)||handoff.step<0||
      handoff.step!==handoff.frame*clock.snapEvery||handoff.seconds!==handoff.step*clock.dt)
    throw new Error("Checkpoint must retain the exact native step clock");
  if(handoff.product?.count!==rec.n||handoff.product?.pad!==rec.pad||
      handoff.product?.sourceDigest!==(rec.source_digest??null))throw new Error("Checkpoint product identity changed");
  const columns=decodeStateStart(handoff.start,rec.pad,manifest.law.state);
  if(["wx","wy","wz"].every(name=>manifest.law.state.includes(name))) {
    if(!Array.isArray(handoff.start.orientations_xyzw)||handoff.start.orientations_xyzw.length!==rec.n)
      throw new Error("Checkpoint needs every active body's recorded orientation");
    decodeStateOrientations(handoff.start,rec.n,columns);
  }
  return rec;
}

export function validateStudioRigidBinding({handoff,manifest,rigidScene,rayOriginM}) {
  const frame=rigidScene?.frame;
  if(rigidScene?.schema!=="turing.studio-rigid-scene.v1"||manifest.scene_schema!==rigidScene.schema||
      !Number.isFinite(frame?.unitM)||frame.unitM<=0||canonical(frame)!==canonical(manifest.source_frame))
    throw new Error("Checkpoint is not bound to this shared native scene frame");
  const nativeOrigin=vector(frame.originM,"native origin"),rayOrigin=vector(rayOriginM,"ray origin");
  for(const side of ["low","high"])for(const [axis,name] of [..."xyz"].entries())
    if(Math.fround(vector(rigidScene.room?.[side],"room bound")[axis])!==handoff.uniforms[`studio_room_${side}_${name}`])
      throw new Error("Checkpoint room differs from the recorded native inputs");
  if(!Array.isArray(rigidScene.parts))throw new Error("Missing authored rigid parts");
  const topology=rigidScene.parts.map((part,i)=>{
    for(const [j,point] of part.vertices.entries())for(const [axis,name] of [..."xyz"].entries())
      if(Math.fround(vector(point,"native vertex")[axis])!==handoff.uniforms[`studio_p${i}_v${j}_${name}`])
        throw new Error("Checkpoint vertices differ from the recorded native inputs");
    return {id:part.id,bodyId:part.bodyId??part.id,closed:part.closed,vertices:part.vertices.length,
      triangles:part.triangles,materialIds:part.materialIds??part.triangles.map(()=>part.materialId)};
  });
  if(canonical(topology)!==canonical(manifest.source_topology))throw new Error("Checkpoint native topology differs from the product");
  if(rigidScene.contact_model!==undefined||manifest.geometry_contract!==undefined) {
    const contract=manifest.geometry_contract;
    if(rigidScene.contact_model!=="native-local-features-v1"||contract?.contact_model!==rigidScene.contact_model||
        contract.snapshot_vertices!=="immutable-native-f32"||!Array.isArray(contract.boundaries)||contract.boundaries.length!==rigidScene.parts.length)
      throw new Error("Checkpoint mesh contact model differs from the built native product");
    rigidScene.parts.forEach((part,i)=>{
      const boundary=contract.boundaries[i];
      if(boundary.id!==part.id||boundary.bodyId!==(part.bodyId??part.id)||
          canonical(boundary.sourceTriangleIds)!==canonical(part.sourceTriangleIds)||
          (boundary.embeddedBoundaryEvidence??null)!==(part.embeddedBoundaryEvidence??null)||
          canonical(boundary.sourceProvenance??null)!==canonical(part.sourceProvenance??null))
        throw new Error("Checkpoint mesh source ownership differs from its admitted native boundary");
    });
  }
  return rayOrigin.map((v,i)=>v-nativeOrigin[i]);
}

async function validateObjectReceipt({savedObject,objectDigest,objectIdentityJson}) {
  // Python's canonical number spelling is not JavaScript's. Preserve the
  // authoritative source_receipt bytes rather than inventing a second digest.
  if(typeof objectIdentityJson!=="string"||!digestPattern.test(objectDigest)||
      await sha(encoder.encode(objectIdentityJson))!==objectDigest)
    throw new Error("Authoritative saved-object identity receipt is required");
  const receipt=JSON.parse(objectIdentityJson);
  const identity=Object.fromEntries(["v","generator","design","materials","physical","geometry_sha256"]
    .map(key=>[key,savedObject[key]]));
  if(canonical(receipt)!==canonical(identity))throw new Error("Saved subject differs from its authoritative identity receipt");
}

export async function createStudioCheckpoint({handoff,manifest,photoPacket,savedObject,objectDigest,objectIdentityJson,rigidScene,rayOriginM}) {
  // Capture all mutable inputs before the first await. Editing the set while
  // hashing cannot pair old state with new triangles, materials or camera.
  const captured=structuredClone({handoff,manifest,savedObject,objectDigest,objectIdentityJson,rigidScene,rayOriginM});
  const bytes=new Uint8Array(photoPacket).slice();
  canonical(captured);
  validateClock(captured.handoff,captured.manifest);
  const translation=validateStudioRigidBinding(captured);
  await validateObjectReceipt(captured);
  const geometry=await inspectScene(bytes,captured.savedObject,captured.objectDigest);
  const payload={schema,scope:"checkpoint-forward",...captured,
    rayScene:{mrp1_base64:base64(bytes),...geometry,nativeToRayTranslationM:translation},
    replayRequirements:{wholeRunArchived:false,exactProductBytesRequired:true,sameBackendRequired:true,
      nativeStepInputLogRequired:true,sceneRevisionLogRequired:true,cameraTrackRequired:true,
      microframesRequireNativeReplay:true,sharedSceneAdmissionRequired:true,rayPoseBindingRequired:true}};
  return {...payload,sha256:await sha(encoder.encode(canonical(payload)))};
}

export async function verifyStudioCheckpoint(value) {
  const captured=structuredClone(value),{sha256,...payload}=captured;
  if(payload.schema!==schema||payload.scope!=="checkpoint-forward"||!digestPattern.test(sha256)||
      await sha(encoder.encode(canonical(payload)))!==sha256)throw new Error("Studio checkpoint digest mismatch");
  validateClock(payload.handoff,payload.manifest);
  const translation=validateStudioRigidBinding(payload);
  if(canonical(translation)!==canonical(payload.rayScene.nativeToRayTranslationM))throw new Error("Studio frame translation mismatch");
  await validateObjectReceipt(payload);
  const inspected=await inspectScene(unbase64(payload.rayScene.mrp1_base64),payload.savedObject,payload.objectDigest);
  if(Object.keys(inspected).some(key=>inspected[key]!==payload.rayScene[key]))throw new Error("Studio ray-scene digest mismatch");
  return captured;
}
