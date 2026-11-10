// One displayed subject -> one native body -> the same Phong/ray geometry.
// This is a coordinate/ownership adapter, not a simulator or a ray scheduler.
// It never changes the room, camera, materials, renderer, or active runtime.
import {rotationMatrix} from "./marble_roll.js";
import {studioCapturedPhysicsFrame,studioNativeStart,studioPoseFromHandoff} from "./marble_studio_physics.js";

const finiteVector=(value,n,name)=>{
  if(!(Array.isArray(value)||ArrayBuffer.isView(value))||value.length!==n||!Array.from(value).every(Number.isFinite))
    throw new TypeError(`${name} needs ${n} finite coordinates`);
  return Array.from(value);
};
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const matvec=(rows,v)=>rows.map(row=>dot(row,v));
const matrixFor=(position,q,scale)=>{
  const r=rotationMatrix([q[3],q[0],q[1],q[2]]);
  return [r[0][0]*scale,r[1][0]*scale,r[2][0]*scale,0,
    r[0][1]*scale,r[1][1]*scale,r[2][1]*scale,0,
    r[0][2]*scale,r[1][2]*scale,r[2][2]*scale,0,...position,1];
};
function immutable(value) {
  if(value&&typeof value==="object") {for(const child of Object.values(value))immutable(child);Object.freeze(value);}
  return value;
}
function visible(capture,frame) {
  const scene=capture?.state;
  if(!Number.isSafeInteger(capture?.revision)||capture.revision<0||scene?.schema!=="turing.studio-visible-scene.v1"||
      scene.channel!=="studio"||!scene.subject||scene.frame?.unitM!==frame?.unitM)
    throw new Error("A current Studio capture in the shared native frame is required");
  finiteVector(scene.frame.rayOriginM,3,"Ray origin");
  finiteVector(frame.originM,3,"Native origin");
  const expected=studioCapturedPhysicsFrame(capture);
  if(frame.radiusM!==expected.radiusM||frame.originM.some((v,i)=>v!==expected.originM[i])||
      ["low","high"].some(side=>!Array.isArray(frame.room?.[side])||frame.room[side].some((v,i)=>v!==expected.room[side][i])))
    throw new Error("Native frame differs from the captured Studio origin, room or physical radius");
  return scene;
}

// The existing renderer composes float32 affine matrices. Accept only its
// uniform-scale, proper rotation contract, and retain the measured float32
// reconstruction error. A shear, reflection or changed radius is not a pose.
export function studioDisplayedPose(subject) {
  const m=finiteVector(subject?.model,16,"Displayed model"),radius=subject?.radius;
  if(m[3]!==0||m[7]!==0||m[11]!==0||m[15]!==1||!Number.isFinite(radius)||radius<=0)
    throw new Error("Displayed subject needs a positive uniform radius and affine model");
  const columns=[m.slice(0,3),m.slice(4,7),m.slice(8,11)].map(c=>c.map(v=>v/radius));
  const tolerance=32*2**-23;
  if(columns.some(c=>Math.abs(dot(c,c)-1)>tolerance)||
      Math.abs(dot(columns[0],columns[1]))>tolerance||Math.abs(dot(columns[0],columns[2]))>tolerance||
      Math.abs(dot(columns[1],columns[2]))>tolerance||dot(cross(columns[0],columns[1]),columns[2])<=0)
    throw new Error("Displayed subject is not a rigid uniform-scale transform");
  const r=Array.from({length:3},(_,i)=>columns.map(c=>c[i]));
  const trace=r[0][0]+r[1][1]+r[2][2];let q;
  if(trace>0) {const s=2*Math.sqrt(trace+1);q=[(r[2][1]-r[1][2])/s,(r[0][2]-r[2][0])/s,(r[1][0]-r[0][1])/s,s/4];}
  else {
    const i=r[0][0]>r[1][1]&&r[0][0]>r[2][2]?0:r[1][1]>r[2][2]?1:2,j=(i+1)%3,k=(i+2)%3;
    const s=2*Math.sqrt(1+r[i][i]-r[j][j]-r[k][k]);q=[0,0,0,(r[k][j]-r[j][k])/s];
    q[i]=s/4;q[j]=(r[j][i]+r[i][j])/s;q[k]=(r[k][i]+r[i][k])/s;
  }
  const length=Math.hypot(...q);q=q.map(v=>v/length);
  const position=m.slice(12,15),reconstructed=matrixFor(position,q,radius);
  const error=Math.max(...m.map((v,i)=>Math.abs(v-reconstructed[i])));
  if(error>tolerance*radius)throw new Error("Displayed pose cannot be retained within its float32 matrix precision");
  return {position,quaternion:q,radius,reconstructionError:error};
}

export function studioDisplayedStart(capture,frame,pad=2) {
  const scene=visible(capture,frame),pose=studioDisplayedPose(scene.subject),motion=scene.subject.motion;
  if(Math.abs(pose.radius*frame.unitM-frame.radiusM)>32*2**-23*frame.radiusM)
    throw new Error("Displayed and native physical radii differ");
  if(!motion||motion.mode!=="off"||!motion.velocityM||!motion.angularVelocity)
    throw new Error("Displayed motion needs an explicit complete velocity/spin seed");
  const velocityM=finiteVector(motion.velocityM,3,"Displayed velocity"),angularVelocity=finiteVector(motion.angularVelocity,3,"Displayed spin");
  if(!Number.isFinite(motion.age)||motion.age<0)throw new Error("Displayed native age is missing");
  return {start:studioNativeStart(frame,{...pose,velocityM,angularVelocity,age:motion.age},pad),pose};
}

// Build a fresh immutable snapshot from an exact recorded native checkpoint.
// Both renderer adapters receive these very same transformed triangles. No
// array retained by a previously submitted ray packet is ever modified here.
// Runtime/scene admission and packet-boundary reader ownership remain caller
// responsibilities; this deliberately returns no ready/commit certificate.
export function studioDynamicGeometry(capture,frame,handoff,pad,{maxTriangles=100000}={}) {
  visible(capture,frame);
  if(handoff.start?.orientations_xyzw?.length!==1)
    throw new Error("This displayed-subject adapter requires exactly one recorded body");
  return geometryForPose(capture,frame,studioPoseFromHandoff(frame,handoff,pad),handoff,maxTriangles,"full-state-checkpoint");
}

// The renderer may consume an already displayed recording sample without a
// pause, native advance, key replay, or full-state GPU readback. Its quaternion
// must belong to that exact sample. This is rendering data, NOT a checkpoint:
// velocity/age that are absent from packed display points are never invented.
export function studioDisplayedDynamicGeometry(capture,frame,runtime,{maxTriangles=100000}={}) {
  visible(capture,frame);
  const manifest=runtime?.manifest,spec=runtime?.spec,rec=manifest?.sizes?.[spec?.size],f=runtime?.shown;
  const origin=manifest?.source_frame;
  if(!runtime?.ready||runtime.preparing||runtime.retiring||runtime.error||runtime.product?.failure||
      rec?.n!==1||spec?.studio_count!==1||manifest?.scene_schema!=="turing.studio-rigid-scene.v1"||
      origin?.unitM!==frame.unitM||!Array.isArray(origin.originM)||origin.originM.some((v,i)=>v!==frame.originM[i])||
      !Number.isSafeInteger(f)||f<0||!Number.isSafeInteger(runtime.snap)||runtime.snap<1||
      !Number.isFinite(manifest.law.dt)||manifest.law.dt<=0)
    throw new Error("A displayed sample from the admitted shared native product is required");
  const points=runtime.points(),rotation=runtime.rotationFor(0),step=f*runtime.snap;
  if(!Number.isSafeInteger(step)||!points||points.length<5||!(points[4]>=50)||rotation?.frame!==f)
    throw new Error("Displayed positions and orientation must belong to the same recorded sample");
  const quaternion=finiteVector(rotation.quaternion,4,"Displayed orientation");
  if(Math.abs(Math.hypot(...quaternion)-1)>1e-6)throw new Error("Displayed orientation must be a unit quaternion");
  const uniforms={...rec.uniforms,...runtime.uniforms};
  for(let i=0;i<4;i++)if(uniforms[`r${i}`]!==Math.fround(frame.radiusM)||uniforms[`swell${i}`]!==0)
    throw new Error("Display-only geometry requires the admitted constant physical radius");
  if(runtime.caseAngleAt?.(step)!==0||(runtime.caseDisplacementAt?.(step)?.displacement??[]).some(v=>v!==0))
    throw new Error("A moving container requires its matching scene transform");
  const pose={position:frame.toStudio(Array.from(points.subarray(0,3))),quaternion,
    radiusM:Math.fround(frame.radiusM),angularVelocity:finiteVector(rotation.omega,3,"Displayed spin"),frame:f,step};
  return geometryForPose(capture,frame,pose,{step,seconds:step*manifest.law.dt},maxTriangles,"display-recording-sample");
}

function geometryForPose(capture,frame,pose,handoff,maxTriangles,poseSource) {
  const scene=visible(capture,frame);
  if(!Number.isSafeInteger(handoff?.step)||handoff.step<0||!Number.isFinite(handoff.seconds)||handoff.seconds<0||
      !Number.isSafeInteger(maxTriangles)||maxTriangles<1)
    throw new Error("Dynamic geometry needs an exact recorded step and bounded capacity");
  if(!Array.isArray(scene.subject.faces)||scene.subject.faces.length+scene.ray.material_ids.length>maxTriangles)
    throw new Error("Complete dynamic scene exceeds its triangle capacity");
  const radius=pose.radiusM/frame.unitM;
  if(!(radius>0))throw new Error("Recorded body has no physical radius");
  const model=matrixFor(pose.position,pose.quaternion,radius),r=rotationMatrix([pose.quaternion[3],...pose.quaternion.slice(0,3)]);
  const origin=scene.frame.rayOriginM,vertices=[],normals=[],ids=[],faces=[];
  const materialCount=scene.ray.pbr.length/16;
  const toWorld=point=>matvec(r,point).map((v,i)=>v*radius+pose.position[i]);
  const toRay=point=>point.map((v,i)=>v*frame.unitM+origin[i]);
  for(const source of scene.subject.faces) {
    const points=source.slice(0,3).map(v=>toWorld(finiteVector(v,3,"Subject triangle")));
    if(source.length<5||!Array.isArray(source[4])||source[4].length!==3||
        !Number.isInteger(source[3])||source[3]<0||source[3]>=materialCount)
      throw new Error("Dynamic subject must retain every material and authored normal");
    const transformedNormals=source[4].map(n=>matvec(r,finiteVector(n,3,"Subject normal")));
    for(let i=0;i<3;i++){vertices.push(...toRay(points[i]));normals.push(...transformedNormals[i]);}
    ids.push(source[3]);faces.push([...points,source[3],transformedNormals,...structuredClone(source.slice(5))]);
  }
  return immutable({schema:"turing.studio-dynamic-geometry.v1",revision:capture.revision,step:handoff.step,seconds:handoff.seconds,poseSource,
    nativeFrame:{unitM:frame.unitM,originM:[...frame.originM]},rayOriginM:[...origin],
    subject:{id:scene.subject.id,pose:structuredClone(pose),model,faces,ray:{vertices,normals,material_ids:ids}},
    // Static room/fixtures and material rows retain source order byte-for-byte
    // (ordinary finite number arrays), with no camera/profile substitution.
    staticRay:structuredClone(scene.ray),triangleSources:structuredClone(scene.triangleSources),camera:structuredClone(scene.camera)});
}
