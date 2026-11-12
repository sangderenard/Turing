import assert from 'node:assert/strict';
import {recordStudioExposureFrames,packMarblePhoto,geodesicFaceSensorCameras} from '../examples/meowdoku/web/marble_photo.js';

let milliseconds=0;
let completedTicks=0;
const initial={vertices:[0,0,.2,1,0,.2,0,1,.2],material_ids:[0],pbr:Array(16).fill(0),
  emitter_profiles:{version:1,records:[]}};
const sample=()=>({...initial,pbr:initial.pbr.map((value,i)=>i===8?(milliseconds>=1000?0:milliseconds>=500?2:1):value)});
const frames=await recordStudioExposureFrames({sample,initialFixture:initial,duration:1.5,substeps:15,
  now:()=>milliseconds,nextFrame:callback=>{milliseconds+=500;completedTicks++;queueMicrotask(callback);}});
assert.equal(completedTicks,15,'Wall-clock delays must not skip contracted Studio ticks');
assert.deepEqual(frames.map(frame=>[frame.t0,frame.t1,frame.pbr[8]]),
  [[0,.1,1],[.1,.2,2],[.2,1.5,0]]);
const wire=packMarblePhoto({interior:[],glass:[]},'a'.repeat(64),
  {fixtures:initial,exposureStateFrames:frames});
const view=new DataView(wire);
const config=JSON.parse(new TextDecoder().decode(new Uint8Array(wire,16,view.getUint32(4,true))));
assert.deepEqual(config.exposure_state_frames,frames);
const surround=geodesicFaceSensorCameras({eye_m:[0,0,.4],target_m:[0,0,.18]});
assert.equal(surround.length,20);
const paired=packMarblePhoto({interior:[],glass:[]},'a'.repeat(64),
  {seconds:1.5,fixtures:initial,exposureStateFrames:frames,
   camera:surround[0],sensorLanes:[{},...surround.slice(1).map(({eye_m,target_m})=>({eye_m,target_m}))],
   cameraVisible:false});
const pairedView=new DataView(paired);
const pairedConfig=JSON.parse(new TextDecoder().decode(new Uint8Array(paired,16,pairedView.getUint32(4,true))));
assert.equal(pairedConfig.sensor_lanes.length,20);
assert.equal(pairedConfig.camera_visible,false);
const mixed=packMarblePhoto({interior:[],glass:[]},'a'.repeat(64),
  {seconds:1.5,fixtures:initial,exposureStateFrames:frames,
   sensorLanes:[{},{eye_m:[.1,0,.07],sensor_transport:'fixed:1'}]});
const mixedView=new DataView(mixed);
const mixedConfig=JSON.parse(new TextDecoder().decode(new Uint8Array(mixed,16,mixedView.getUint32(4,true))));
assert.equal(mixedConfig.sensor_lanes[1].sensor_transport,'fixed:1');
milliseconds=0;
const moving=await recordStudioExposureFrames({
  sample:()=>({...initial,vertices:milliseconds>=100?[0,0,.2,1,0,.2,0,1,.3]:initial.vertices}),
  initialFixture:initial,duration:.3,substeps:3,
  nextFrame:callback=>{milliseconds+=100;queueMicrotask(callback);}});
assert.equal(moving.length,2);
assert.deepEqual(moving[1].vertices,[0,0,.2,1,0,.2,0,1,.3]);
assert.equal(moving[0].vertices,undefined);
console.log('Timed Studio exposure frames retain ordered light and geometry slices');
