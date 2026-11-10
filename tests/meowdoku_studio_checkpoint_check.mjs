import assert from 'node:assert/strict';
import {webcrypto,createHash} from 'node:crypto';
import {encodeStateStart} from '../examples/meowdoku/web/compute_state.js';
import {geometrySha256} from '../examples/meowdoku/web/marble_object.js';
import {packMarblePhoto,OBJECT_CENTER} from '../examples/meowdoku/web/marble_photo.js';
import {createStudioCheckpoint,verifyStudioCheckpoint} from '../examples/meowdoku/web/marble_studio_checkpoint.js';
globalThis.crypto??=webcrypto;

const names=['x','y','z','vx','vy','vz','wx','wy','wz','age','rs'];
const normal=[0,1,0],geometry={interior:[],glass:[[[0,0,0],[0,0,1],[1,0,0],0,[normal,normal,normal]]],triangleCount:1};
const material=Array(16).fill(0);material[0]=.125;material[7]=1;
const savedObject={v:1,name:'Retained subject',generator:'fixture',design:{kind:'triangle-fixture'},physical:{'radius-mm':10},
  geometry_sha256:await geometrySha256(geometry),materials:{tables:{pbr:{stride:16,data:material}}}};
const canonical=value=>Array.isArray(value)?`[${value.map(canonical).join(',')}]`:value&&typeof value==='object'?
  `{${Object.keys(value).sort().map(key=>`${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`:JSON.stringify(value);
// Fixture numbers have identical JSON spelling in Python and JS; production
// callers supply marble_objects.source_receipt bytes, never JS reformatting.
const objectIdentityJson=process.env.STUDIO_CHECKPOINT_RECEIPT??canonical(Object.fromEntries(['v','generator','design','materials','physical','geometry_sha256'].map(k=>[k,savedObject[k]])));
const digest=createHash('sha256').update(objectIdentityJson).digest('hex');
const fixtures={vertices:[0,0,0,0,0,1,1,0,0],material_ids:[0],pbr:material.slice()};
const packet=()=>packMarblePhoto(geometry,digest,{fixtures,camera:{eye_m:[0,0,.07],target_m:OBJECT_CENTER}});
const columns=Object.fromEntries(names.map(name=>[name,[0,0]]));
columns.y[0]=.05;columns.rs[0]=.01;columns.age[0]=1/3;columns.vx[0]=.12;columns.wz[0]=.25;
const start=encodeStateStart(columns,2,names,[[0,0,0,1]]);
const manifest={law:{state:names,dt:1/720},sizes:{studio:{n:1,pad:2,source_digest:'compiled-native-product'}}};
const handoff={manifest:'studio.manifest.json',size:'studio',frame:20,step:240,seconds:240*(1/720),start,
  clock:{dt:1/720,snapEvery:12,epoch:3},product:{sourceDigest:'compiled-native-product',backend:'gpu',adapter:'test',count:1,pad:2},
  uniforms:{grav_y:-9.81},caseAngle:0,caseAxis:'y',caseMotion:{angle:0},caseDisplacement:{position:[0,0,0]}};
const rigidScene={schema:'turing.studio-rigid-scene.v1',frame:{unitM:.01,originM:[0,.0402,0]},
  room:{low:[-.1,.01,-.1],high:[.1,.2,.1]},parts:[]};
manifest.scene_schema=rigidScene.schema;manifest.source_frame=structuredClone(rigidScene.frame);manifest.source_topology=[];
for(const side of ['low','high'])for(const [axis,name] of [...'xyz'].entries())
  handoff.uniforms[`studio_room_${side}_${name}`]=Math.fround(rigidScene.room[side][axis]);
const input=()=>({handoff:structuredClone(handoff),manifest:structuredClone(manifest),photoPacket:packet(),
  savedObject:structuredClone(savedObject),objectDigest:digest,objectIdentityJson,rigidScene:structuredClone(rigidScene),rayOriginM:[...OBJECT_CENTER]});
const captured=input(),originalPacket=Buffer.from(captured.photoPacket).toString('base64');
const pending=createStudioCheckpoint(captured);
captured.handoff.step=999;captured.savedObject.materials.tables.pbr.data[0]=.9;
captured.rigidScene.frame.originM[1]=2;new Uint8Array(captured.photoPacket).fill(0);
const archive=await pending;
assert.equal(archive.scope,'checkpoint-forward');
assert.equal(archive.handoff.step,240);assert.equal(archive.handoff.seconds,240*(1/720));
assert.equal(archive.handoff.clock.epoch,3);assert.equal(archive.handoff.clock.snapEvery,12);
assert.equal(archive.savedObject.materials.tables.pbr.data[0],.125);
assert.equal(archive.rayScene.mrp1_base64,originalPacket);
assert.equal(archive.rayScene.subjectTriangles,1);assert.equal(archive.rayScene.sceneTriangles,1);
assert.deepEqual(archive.rayScene.nativeToRayTranslationM,[0,-.0402,.18]);
assert.equal(archive.replayRequirements.wholeRunArchived,false);
assert.equal(archive.replayRequirements.microframesRequireNativeReplay,true);
assert.deepEqual(await verifyStudioCheckpoint(archive),archive);
const changed=structuredClone(archive);changed.handoff.seconds+=.1;
await assert.rejects(verifyStudioCheckpoint(changed),/digest mismatch/);
const badClock=input();badClock.handoff.seconds=.5;
await assert.rejects(createStudioCheckpoint(badClock),/native step clock/);
const badProduct=input();badProduct.handoff.product.sourceDigest='different';
await assert.rejects(createStudioCheckpoint(badProduct),/identity changed/);
const oldToy=input();delete oldToy.manifest.scene_schema;
await assert.rejects(createStudioCheckpoint(oldToy),/shared native scene frame/);
const movedRoom=input();movedRoom.rigidScene.room.high[0]+=.1;
await assert.rejects(createStudioCheckpoint(movedRoom),/room differs/);
const noPose=input();delete noPose.handoff.start.orientations_xyzw;
await assert.rejects(createStudioCheckpoint(noPose),/every active body's/);
const badGeometry=input();new DataView(badGeometry.photoPacket).setFloat32(badGeometry.photoPacket.byteLength-76,.2,true);
await assert.rejects(createStudioCheckpoint(badGeometry),/geometry receipt/);
const noScene=input();noScene.photoPacket=packMarblePhoto(geometry,digest);
await assert.rejects(createStudioCheckpoint(noScene),/complete existing ray-scene/);
const badMaterial=input();const jsonLength=new DataView(badMaterial.photoPacket).getUint32(4,true);
new DataView(badMaterial.photoPacket).setUint32(16+jsonLength+72,1,true);
await assert.rejects(createStudioCheckpoint(badMaterial),/absent saved material/);
const nonfinite=input();nonfinite.handoff.uniforms.grav_y=Infinity;
await assert.rejects(createStudioCheckpoint(nonfinite),/nonfinite/);
const changedMaterial=input();changedMaterial.savedObject.materials.tables.pbr.data[0]=.9;
await assert.rejects(createStudioCheckpoint(changedMaterial),/identity receipt/);
const noReceipt=input();noReceipt.objectIdentityJson='{}';
await assert.rejects(createStudioCheckpoint(noReceipt),/identity receipt/);
const invalidMaterial=input();invalidMaterial.savedObject.materials.tables.pbr.data[0]='not-a-number';
await assert.rejects(createStudioCheckpoint(invalidMaterial),/identity receipt/);
// Even a self-consistent identity cannot make a nonnumeric table usable.
invalidMaterial.objectIdentityJson=canonical(Object.fromEntries(['v','generator','design','materials','physical','geometry_sha256']
  .map(k=>[k,invalidMaterial.savedObject[k]])));
invalidMaterial.objectDigest=createHash('sha256').update(invalidMaterial.objectIdentityJson).digest('hex');
invalidMaterial.photoPacket=packMarblePhoto(geometry,invalidMaterial.objectDigest,{fixtures});
await assert.rejects(createStudioCheckpoint(invalidMaterial),/material table/);
console.log('Studio checkpoint: exact native clock/state, complete MRP1 geometry/materials, immutable capture and digest checks passed');
