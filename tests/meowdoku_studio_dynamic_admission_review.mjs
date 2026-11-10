import assert from 'node:assert/strict';
import {packStudioScene} from '../examples/meowdoku/web/marble_studio_scene.js';
import {studioCapturedPhysicsFrame} from '../examples/meowdoku/web/marble_studio_physics.js';
import {studioDisplayedStart,studioDynamicGeometry} from '../examples/meowdoku/web/marble_studio_dynamic.js';
const face=[[0,0,0],[1,0,0],[0,1,0],0,[[0,0,1],[0,0,1],[0,0,1]]];
const capture={revision:1,state:packStudioScene({channel:'studio',unitM:.01,rayOriginM:[0,0,.18],
  parts:[{id:'wall',faces:[face]}],pbr:{stride:16,data:Array(16).fill(0)},
  subject:{radius:1,faces:[face],model:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    motion:{mode:'off',velocityM:[0,0,0],angularVelocity:[0,0,0],age:0}},
  room:{leftX:-10,rightX:10,floorY:-4,ceilingY:8,backZ:-10,frontZ:10},
  authoring:{object:{design:{'base-shape':'marble'}}},camera:{}})};
const frame=studioCapturedPhysicsFrame(capture);
const {start}=studioDisplayedStart(capture,frame);
const handoff={start,step:1,seconds:.1};
assert.ok(studioDynamicGeometry(capture,frame,handoff,2));
for(const mutate of [
  s=>s.room.floorY=-8,
  s=>s.room.rightX=20,
  s=>s.subject.radius=2,
  s=>s.frame.unitM=.02,
]) {
  const other=structuredClone(capture);mutate(other.state);
  const stale=studioCapturedPhysicsFrame(other);
  assert.throws(()=>studioDisplayedStart(capture,stale));
  assert.throws(()=>studioDynamicGeometry(capture,stale,handoff,2));
}
console.log('Dynamic admission rejects stale native origin, room, radius and unit');
