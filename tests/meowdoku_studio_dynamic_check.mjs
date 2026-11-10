import assert from 'node:assert/strict';
import {packStudioScene} from '../examples/meowdoku/web/marble_studio_scene.js';
import {studioPhysicsFrame,studioCapturedPhysicsFrame,studioNativeStart} from '../examples/meowdoku/web/marble_studio_physics.js';
import {studioDisplayedPose,studioDisplayedStart,studioDynamicGeometry,studioDisplayedDynamicGeometry} from '../examples/meowdoku/web/marble_studio_dynamic.js';
import {rotationMatrix} from '../examples/meowdoku/web/marble_roll.js';
import {decodeStateStart} from '../examples/meowdoku/web/compute_state.js';
const names=['x','y','z','vx','vy','vz','wx','wy','wz','age','rs'];
const near=(a,b,t=2e-6)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<=t,`${v} != ${b[i]}`));};
const frame=studioPhysicsFrame({scale:{version:1,mode:'fixed',unitMm:10},radiusMm:10,tableEnabled:true,
  room:{leftX:-10,rightX:10,floorY:-4,ceilingY:8,backZ:-10,frontZ:10}});
const model=(q,scale=1,position=[.2,.1,-.3])=>{
  const r=rotationMatrix([q[3],...q.slice(0,3)]);
  return Float32Array.from([r[0][0]*scale,r[1][0]*scale,r[2][0]*scale,0,
    r[0][1]*scale,r[1][1]*scale,r[2][1]*scale,0,r[0][2]*scale,r[1][2]*scale,r[2][2]*scale,0,...position,1]);
};
const face=[[-1,0,0],[1,0,0],[0,1,0],0,[[0,0,1],[0,0,1],[0,0,1]],{source:'kept'}];
const capture=(q=[0,0,0,1])=>({revision:42,state:packStudioScene({channel:'studio',unitM:.01,rayOriginM:[0,0,.18],
  parts:[{id:'room/floor',bodyId:'room',faces:[structuredClone(face)],closed:false,convex:true}],
  pbr:{stride:16,data:new Float32Array(16)},subject:{id:'subject',faces:[structuredClone(face)],model:model(q),radius:1,
    motion:{mode:'off',velocityM:[.1,.2,-.3],angularVelocity:[.4,-.5,.6],age:7}},
  camera:{view:[1,2,3],fovDegrees:45},authoring:{object:{design:{'base-shape':'marble'}}},
  room:{leftX:-10,rightX:10,floorY:-4,ceilingY:8,backZ:-10,frontZ:10}})});
for(const q of [[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[.5,.5,.5,.5],[.3,-.4,.5,Math.sqrt(.5)]]){
  for(const scale of [.01,1,100]){
    const source={model:model(q,scale),radius:scale},pose=studioDisplayedPose(source);
    near(model(pose.quaternion,scale,pose.position),source.model,2e-5);
    assert.ok(pose.reconstructionError<4e-6*scale);
  }
}
const initial=capture([.5,.5,.5,.5]),before=structuredClone(initial);
const capturedFrame=studioCapturedPhysicsFrame(initial);
assert.equal(capturedFrame.unitM,frame.unitM);assert.equal(capturedFrame.radiusM,frame.radiusM);
assert.deepEqual(capturedFrame.room,frame.room);assert.deepEqual(capturedFrame.originM,frame.originM);
assert.deepEqual(capturedFrame.subjectCenter,initial.state.subject.position);
const {start,pose}=studioDisplayedStart(initial,frame,2),columns=decodeStateStart(start,2,names);
near([columns.x[0],columns.y[0],columns.z[0]],frame.toNative(pose.position),1e-8);
near([columns.vx[0],columns.vy[0],columns.vz[0]],[.1,.2,-.3]);
near([columns.wx[0],columns.wy[0],columns.wz[0]],[.4,-.5,.6]);assert.equal(columns.age[0],7);
assert.deepEqual(initial,before);
const handoff={start,frame:1,step:6,seconds:1/60};
const first=studioDynamicGeometry(initial,frame,handoff,2);
assert.equal(first.revision,42);assert.equal(first.step,6);assert.deepEqual(first.staticRay,initial.state.ray);
assert.deepEqual(first.camera,initial.state.camera);assert.deepEqual(first.triangleSources,initial.state.triangleSources);
near(first.subject.ray.vertices,initial.state.subject.ray.vertices,1e-8);
for(let i=0;i<3;i++)near(first.subject.faces[0][i].map((x,k)=>x*frame.unitM+initial.state.frame.rayOriginM[k]),first.subject.ray.vertices.slice(i*3,i*3+3),1e-14);
assert.deepEqual(first.subject.faces[0][5],{source:'kept'});assert.deepEqual(first.subject.ray.material_ids,[0]);
assert.ok(Object.isFrozen(first.subject.ray.vertices));
assert.throws(()=>first.subject.ray.vertices[0]=99,TypeError);
const secondStart=studioNativeStart(frame,{position:[1,1,1],quaternion:[0,0,1,0]},2);
const second=studioDynamicGeometry(initial,frame,{start:secondStart,frame:2,step:12,seconds:1/30},2);
assert.notDeepEqual(second.subject.ray.vertices,first.subject.ray.vertices);
near(first.subject.ray.vertices,initial.state.subject.ray.vertices,1e-8);
initial.state.ray.vertices[0]=200;initial.state.subject.faces[0][0][0]=200;
assert.notEqual(first.staticRay.vertices[0],200);assert.notEqual(first.subject.faces[0][0][0],200);
for(const mutation of [s=>s.model[0]*=-1,s=>s.model[1]+=.1,s=>s.model[3]=1,s=>s.radius=2]){
  const s={model:Array.from(model([0,0,0,1])),radius:1};mutation(s);assert.throws(()=>studioDisplayedPose(s),/rigid|affine/);
}
const missing=capture();delete missing.state.subject.motion;assert.throws(()=>studioDisplayedStart(missing,frame),/complete/);
const exposure=capture();exposure.state.channel='exposure';assert.throws(()=>studioDisplayedStart(exposure,frame),/current Studio/);
assert.throws(()=>studioCapturedPhysicsFrame(exposure),/displayed Studio/);
const die=capture();die.state.authoring.object.design['base-shape']='d6';assert.throws(()=>studioCapturedPhysicsFrame(die),/authored marble/);
assert.throws(()=>studioDynamicGeometry(capture(),frame,handoff,2,{maxTriangles:1}),/capacity/);
assert.throws(()=>studioDynamicGeometry(capture(),frame,{...handoff,start:{...start,orientations_xyzw:[[0,0,0,1],[0,0,0,1]]}},2),/exactly one/);
const displayCapture=capture(),displayStart=studioDisplayedStart(displayCapture,frame,2).start;
const displayColumns=decodeStateStart(displayStart,2,names);
const points=Float32Array.of(displayColumns.x[0],displayColumns.y[0],displayColumns.z[0],0,210);
const displayRotation={frame:3,quaternion:displayStart.orientations_xyzw[0],omega:[0,0,0]};
const runtime={ready:true,shown:3,snap:6,steps:600,spec:{size:'studio',studio_count:1},
  manifest:{scene_schema:'turing.studio-rigid-scene.v1',source_frame:{unitM:frame.unitM,originM:frame.originM},
    law:{dt:1/360},sizes:{studio:{n:1,uniforms:Object.fromEntries(Array.from({length:4},(_,i)=>[[`r${i}`,Math.fround(frame.radiusM)],[`swell${i}`,0]]).flat())}}},
  points:()=>points,rotationFor:()=>displayRotation,caseAngleAt:()=>0,caseDisplacementAt:()=>({displacement:[0,0,0]}),
  pauseAtDisplayedState:()=>{throw Error('must not pause');},frame:()=>{throw Error('must not advance');}};
const displayed=studioDisplayedDynamicGeometry(displayCapture,frame,runtime);
assert.equal(displayed.step,18);assert.equal(displayed.seconds,18/360);assert.equal(displayed.poseSource,'display-recording-sample');
assert.equal('velocityM' in displayed.subject.pose,false);assert.equal('age' in displayed.subject.pose,false);
assert.equal(runtime.steps,600);points[0]=.123;
assert.notEqual(displayed.subject.pose.position[0],frame.toStudio([points[0],points[1],points[2]])[0]);
displayRotation.frame=2;assert.throws(()=>studioDisplayedDynamicGeometry(displayCapture,frame,runtime),/same recorded/);displayRotation.frame=3;
runtime.uniforms={swell0:.1};assert.throws(()=>studioDisplayedDynamicGeometry(displayCapture,frame,runtime),/constant physical radius/);
console.log('Studio dynamic geometry: pose/seed, immutable frames, exact Phong/ray/material/camera identity passed');
