// Environment/clock exchange remains usable when the subject is not ammunition.
// No GL, browser, server, worker or compiler is started by this test.
import assert from 'node:assert/strict';
import {createMechanicalCreatureStudioReturnStore,createMechanicalCreatureStudioReturn} from '../examples/meowdoku/web/mechanical_creature_studio_return.js';
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],environment={buildingIdentity:'studio-room',
  environmentAssetIdentity:'studio-scene:dice',environmentRevision:3,environmentSequence:1,buildingPlacement:I,unitM:.3048};
const receipt=e=>({buildingIdentity:e.buildingIdentity,assetIdentity:e.environmentAssetIdentity,
  revision:e.environmentRevision,sequence:e.environmentSequence,geometryRevision:4,colliderPublication:'posted-to-original-worker'});
const packet={schema:'mechanical-creature-studio-state-v1',epoch:'empty-owner',dtAuthority:'original-viewport-slice',
  frameTimeMs:100,frameDtSeconds:.016,workerSequence:8,geometryRevision:4,workerReady:true,
  environments:[receipt(environment)],bodies:[],camera:{positionM:[0,1,2],facing:[0,0,-1]}};
const options={epoch:packet.epoch,environment,fixedQuaternionXYZW:[0,0,0,1]},
  store=createMechanicalCreatureStudioReturnStore({...options,appearanceAssetIdentities:[]});
assert.equal(store.receive(packet).accepted,true);assert.equal(store.read().bodies.length,0);
assert.equal(store.read().frameDtSeconds,.016);assert.deepEqual(store.read().camera,packet.camera);
assert.equal(store.receive(packet).reason,'stale-frame');const previous=store.read();
const bad=structuredClone(packet);bad.frameTimeMs=101;bad.bodies=[{identity:'unknown-body',appearanceAssetIdentity:'unregistered',ammunitionItemIdentity:'ammo'}];
assert.throws(()=>store.receive(bad),/unregistered appearance/);assert.equal(store.read(),previous);
const next={...environment,environmentAssetIdentity:'studio-scene:bunny',environmentRevision:4,environmentSequence:2};
store.setEnvironment(next);const updated={...packet,frameTimeMs:102,environments:[receipt(next)]};
assert.equal(store.receive(updated).accepted,true);assert.equal(store.readStatus().acceptedEnvironmentSequence,2);
assert.equal(store.readStatus().bodyCount,0);
for(const appearanceAssetIdentities of [null,[''],Array(9).fill('a'),['same','same']])
  assert.throws(()=>createMechanicalCreatureStudioReturnStore({...options,appearanceAssetIdentities}));
const renderer=await createMechanicalCreatureStudioReturn({...options,assets:[]});
assert.equal(renderer.receive(packet).accepted,true);
for(const phase of ['all','opaque','transparent'])assert.deepEqual(renderer.draw({phase}),{bodies:0,draws:0});
assert.throws(()=>renderer.receive(bad),/unregistered appearance/);
for(const assets of [null,[null],Array(9).fill(null)])await assert.rejects(()=>createMechanicalCreatureStudioReturn({...options,assets}));
renderer.dispose();assert.throws(()=>renderer.receive(packet),/closed/);assert.throws(()=>renderer.draw(),/closed/);
store.dispose();assert.equal(store.read(),null);
console.log(JSON.stringify({pass:true,emptyAppearanceSet:true,environmentClockAndCameraPreserved:true,
  unregisteredBodiesRejected:true,malformedAssetsRejected:true,zeroGpuAllocations:true,originalClockOnly:true}));
