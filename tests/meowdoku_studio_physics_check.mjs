import assert from 'node:assert/strict';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
import {studioScaleTransform} from '../examples/meowdoku/web/marble_studio_scale.js';
import {decodeStateStart} from '../examples/meowdoku/web/compute_state.js';
import {studioPhysicsFrame,studioRigidPacket,studioNativeStart,studioPoseFromHandoff,studioHandoffOptions,STUDIO_NATIVE_STATE} from '../examples/meowdoku/web/marble_studio_physics.js';

const near=(a,b,tolerance=1e-7)=>a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<=tolerance,`${v} != ${b[i]}`));
const room=studioRoomBounds(3,3);
const scale={version:1,mode:'fixed',unitMm:10};
for(const radiusMm of [.1,1,10,40]) {
  const frame=studioPhysicsFrame({scale,radiusMm,tableEnabled:true,room});
  const visual=studioScaleTransform(scale,radiusMm,true);
  assert.equal(frame.unitM,visual.unitM);
  near(frame.toStudio(frame.toNative(visual.subjectCenter)),visual.subjectCenter,1e-12);
  assert.ok(frame.room.low[1]>0);
  // The exact same table contact in preview, physical/photo metres and law.
  const base=frame.toNative([0,-1,0]);
  const centre=frame.toNative(visual.subjectCenter);
  assert.ok(Math.abs(centre[1]-frame.radiusM-base[1])<1e-15);
  const q=[0,Math.sin(.35),0,Math.cos(.35)],v=[.01,-.02,.03],w=[.4,.5,-.6];
  const start=studioNativeStart(frame,{quaternion:q,velocityM:v,angularVelocity:w,age:4},2);
  const state=decodeStateStart(start,2,STUDIO_NATIVE_STATE);
  assert.equal(state.x[1],0); assert.equal(state.age[1],0); assert.equal(state.rs[1],0);
  const pose=studioPoseFromHandoff(frame,{start,frame:12,step:72},2);
  near(pose.position,visual.subjectCenter,1e-6);
  assert.deepEqual(pose.quaternion,q); near(pose.velocityM,v); near(pose.angularVelocity,w);
  assert.equal(pose.age,4); assert.equal(pose.frame,12); assert.equal(pose.step,72);
  assert.equal(pose.radiusM,Math.fround(radiusMm/1000));
}
const small=studioPhysicsFrame({scale,radiusMm:1,tableEnabled:true,room});
const large=studioPhysicsFrame({scale,radiusMm:40,tableEnabled:true,room});
assert.deepEqual(small.originM,large.originM);
assert.deepEqual(small.room,large.room);
const part={id:'authored',bodyId:'authored',convex:true,closed:false,
  vertices:[[0,-1,0],[1,-1,0],[0,-1,1]],triangles:[[0,2,1]],materialIds:[9],source:'same-render-mesh'};
const before=JSON.stringify(part),packet=studioRigidPacket(small,[part]);
assert.equal(JSON.stringify(part),before);
assert.deepEqual(packet.parts[0].triangles,part.triangles);
assert.deepEqual(packet.parts[0].materialIds,[9]);
assert.equal(packet.parts[0].source,'same-render-mesh');
packet.parts[0].vertices.forEach((v,i)=>near(small.toStudio(v),part.vertices[i],1e-12));
assert.throws(()=>studioRigidPacket(small,[{...part,convex:false}]),/Unsupported/);
assert.throws(()=>studioNativeStart(small,{quaternion:[1,1,1,1]}),/unit quaternion/);
assert.throws(()=>studioNativeStart(small,{position:[1000,0,0]}),/does not fit/);
assert.equal(studioHandoffOptions({law:{dt:1/360}}).maxReplaySteps,360);
assert.equal(studioHandoffOptions({law:{dt:1/5760}}).maxReplaySteps,5760);
assert.equal(studioHandoffOptions({law:{dt:1/5760}},0).maxReplaySteps,0);
assert.throws(()=>studioHandoffOptions({law:{dt:1/360}},2),/at most one second/);
for(const tableEnabled of [false,true]) {
  const legacy=studioPhysicsFrame({scale:undefined,radiusMm:10,tableEnabled,room});
  assert.equal(legacy.unitM,studioScaleTransform(undefined,10,tableEnabled).unitM);
  near(legacy.toStudio(legacy.toNative([1,2,3])),[1,2,3],1e-12);
}
console.log('Studio shared frame, unchanged topology/materials, full pose/velocity handoff: passed');
