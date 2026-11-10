// Actual original lifecycle writers + actual bridge publisher, with DOM/worker
// transport test doubles. No native physics or browser rendering claim.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {createHash} from 'node:crypto';
import {createMechanicalCreatureStudioReturnStore} from '../examples/meowdoku/web/mechanical_creature_studio_return.js';
import * as assetModule from '../examples/meowdoku/web/marble_game_asset.js';
import {studioRigToGame} from '../examples/meowdoku/web/mechanical_creature_studio_collection.js';
import {validateStudioLightTransform} from '../examples/meowdoku/web/marble_studio_subject_renderer.js';
import {rotationMatrix} from '../examples/meowdoku/web/marble_roll.js';

const root=new URL('../',import.meta.url),read=path=>fs.readFileSync(new URL(path,root),'utf8');
const originalBytes=fs.readFileSync(new URL('build/dot-playable-tires-20261110/index.html',root));
assert.equal(createHash('sha256').update(originalBytes).digest('hex'),'aa6cb6fb897d24e6bf769ca2dbfde3a0d3d0dbcdd64ddb87fd30173bc0f0d5e2');
const original=originalBytes.toString('utf8').replaceAll('\r\n','\n');
const model=JSON.parse(original.match(/<script type="application\/json" id="abstract-ui-model">([\s\S]*?)<\/script>/)[1]);
const definition=JSON.parse(read('build/dot-studio-game-transfer-20261110/studio-building-definition-v2.json'));
const marbleAsset=JSON.parse(read('build/dot-studio-game-transfer-20261110/studio-cat-eye.game-asset.json'));
const bridgeSource=read('examples/meowdoku/web/mechanical_creature_studio_bridge.js');
const fn=name=>{const start=original.indexOf(`function ${name}(`),end=original.indexOf('\n}',start);assert(start>=0&&end>start,name);return original.slice(start,end+2);};
const names=['initializePhysicsSnapshotSlots','reservePhysicsSnapshotSlot','releasePhysicsSnapshotSlot',
  'registerProjectilePhysicsMembership','firePhysicsBall','sleepProjectilePhysics','wakeProjectilePhysics',
  'expireProjectile','collectProjectilePickup','projectileAmmoItem'];
const noop=()=>{},node=()=>({dataset:{},style:{setProperty:noop},setAttribute:noop,append:noop});
const errors=[],sent=[];
const gameContext=vm.createContext({model,Map,Uint32Array,performance:{now:()=>1000},structuredClone,
  document:{querySelector:()=>null,getElementById:()=>node()},div:node,makeSelectable:node,
  normalized3:p=>{const n=Math.hypot(...p);return p.map(x=>x/n);},spectrumColorHex:()=>'#123456',
  ensureMusicRoomAudio:()=>Promise.resolve(),rebuildPortableSceneMesh:noop,refreshInventoryCounts:noop,setPlacementStatus:noop,
  reportRuntimeFault:(stage,error)=>errors.push({stage,error:String(error)}),sent});
vm.runInContext(`
const projectileState=new Map(),projectilePickupState=new Map(),entityState=new Map(),index=new Map();let projectileSequence=0;
const physicsRuntime={instance:true},hotbarState={inventory:{items:[{quantity:100,maximum_stack:200,properties:{operation:'projectile-ammunition'}}]}};
const stateLoopRuntime={ready:true,bodies:new Map(),slotRecords:[],latestSequence:50,worker:{postMessage:m=>sent.push(m)}};
const shaderViewer={cameraFacing:[0,0,-1],cameraPosition:[32,3,-17],geometry:[],colliders:[],revision:10};
const vehicleRuntime={active:null},viewportControls={position:null,policy:{actor:model.identity}};
const portalRuntime={lastPositions:new Map(),cooldowns:new Map(),transits:new Map()};
${names.map(fn).join('\n')}
initializePhysicsSnapshotSlots(128);
${bridgeSource}
globalThis.game={projectileState,projectilePickupState,stateLoopRuntime,shaderViewer,
  firePhysicsBall,sleepProjectilePhysics,wakeProjectilePhysics,expireProjectile,collectProjectilePickup,
  publish:publishMechanicalCreatureStudioSnapshot,setEnvironmentReceipt:r=>mechanicalCreatureStudioBridgeState.applied=r};
`,gameContext);
const game=gameContext.game;
assert(game.firePhysicsBall());
const state=[...game.projectileState.values()][0];
state.appearanceAssetIdentity=marbleAsset.identity;state.ammunitionItemIdentity='inventory:chosen';
const originalIdentity=state.identity;
const environment={buildingIdentity:definition.identity,environmentAssetIdentity:'studio-scene:initial',
  environmentRevision:10,environmentSequence:1,buildingPlacement:definition.placement,unitM:definition.unitM};
const receipt=env=>({buildingIdentity:env.buildingIdentity,assetIdentity:env.environmentAssetIdentity,
  previousAssetIdentity:'studio-scene:previous',revision:env.environmentRevision,sequence:env.environmentSequence,
  geometryRevision:10,colliderPublication:'posted-to-original-worker'});
game.setEnvironmentReceipt(receipt(environment));
const store=createMechanicalCreatureStudioReturnStore({epoch:'session-A',environment,
  appearanceAssetIdentities:[marbleAsset.identity],fixedQuaternionXYZW:[0,0,0,1]});
const receipts=[];
gameContext.mechanicalCreatureStudioBridge.setSink(packet=>receipts.push(store.receive(packet)),{epoch:'session-A'});
const publish=(time,dt=.016)=>{game.publish(time,dt);assert.equal(errors.length,0,JSON.stringify(errors));return store.read();};
const sourceStateBefore=JSON.stringify([...game.projectileState]),first=publish(100);
assert.equal(receipts.at(-1).accepted,true);assert.equal(receipts.at(-1).slot,0);
assert.equal(first.bodies.length,1);assert.equal(first.bodies[0].identity,originalIdentity);
assert.equal(first.bodies[0].radiusM,.11);assert.equal(first.bodies[0].physicsQuaternionXYZW,null);
assert.equal(first.frameDtSeconds,.016);assert.equal(first.workerSequence,50);
assert.equal(first.workerAppliedEnvironment,false);
assert.equal(JSON.stringify([...game.projectileState]),sourceStateBefore,'receiving cannot alter original state');
const body=first.bodies[0],near=(a,b,t=5e-13)=>assert(Math.abs(a-b)<=t,`${a} != ${b}`);
body.positionStudio.forEach((v,i)=>near(v,(state.position[i]-definition.translationM[i])/definition.unitM));
near(body.radiusStudio,.11/definition.unitM);
assert(Object.isFrozen(first)&&Object.isFrozen(body)&&Object.isFrozen(body.model));
assert.throws(()=>body.positionM[0]=999);
const rawFirst=gameContext.mechanicalCreatureStudioBridge.readSnapshot();
assert(Object.isFrozen(rawFirst)&&Object.isFrozen(rawFirst.bodies));

// Each new frame publishes into the other slot. Receivers keep immutable old
// views, rather than pointers into producer arrays that change next tick.
state.position[0]+=1;const second=publish(116);
assert.equal(receipts.at(-1).slot,1);assert.notEqual(second,first);
assert.equal(first.bodies[0].positionM[0],rawFirst.bodies[0].positionM[0]);
const copied=structuredClone(gameContext.mechanicalCreatureStudioBridge.readSnapshot());copied.frameTimeMs=117;
assert.equal(store.receive(copied).accepted,true);copied.bodies[0].positionM[0]=999;
assert.notEqual(store.read().bodies[0].positionM[0],999);
assert.equal(store.readStatus().activeSlot,0);
assert.equal(store.receive(copied).reason,'stale-frame');
const accepted=store.read();
const edit=(mutate,time=118)=>{const p=structuredClone(gameContext.mechanicalCreatureStudioBridge.readSnapshot());p.frameTimeMs=time;mutate(p);return p;};
assert.equal(store.receive(edit(p=>p.epoch='previous-session')).reason,'different-epoch');
assert.equal(store.receive(edit(p=>p.workerSequence=49)).reason,'stale-worker-sequence');
assert.equal(store.receive(edit(p=>p.geometryRevision=9)).reason,'stale-geometry-revision');
assert.equal(store.receive(edit(p=>p.environments=[])).reason,'environment-not-published');
assert.equal(store.receive(edit(p=>p.environments[0].colliderPublication='native-scene-ready-worker-pending')).reason,'environment-worker-pending');
assert.equal(store.read(),accepted,'rejected frames never clear or publish partial state');
for(const mutate of [p=>p.bodies[0].positionM[0]=NaN,p=>p.bodies[0].appearanceAssetIdentity='unknown',
  p=>p.bodies[0].ammunitionItemIdentity='different-item',p=>p.bodies[0].radiusM=.2,
  p=>p.bodies.push(structuredClone(p.bodies[0])),p=>p.bodies[0].membership=null,
  p=>p.bodies[0].quaternion=[0,0,0,1],p=>p.frameDtSeconds=-1,p=>p.dtAuthority='other-clock',
  p=>p.environments.push(structuredClone(p.environments[0])),p=>p.camera.facing=[0,0,0]])
  assert.throws(()=>store.receive(edit(mutate)));
assert.equal(store.read(),accepted);

// Original lifecycle functions remain authoritative. No synthetic integration
// advances position: the source only transitions actual membership and status.
assert(game.sleepProjectilePhysics(state,'return-test'));const sleeping=publish(132);
assert(sleeping.bodies[0].sleeping);assert.equal(sleeping.bodies[0].membership,null);
assert(game.wakeProjectilePhysics(state,'return-test',[.1,.2,.3]));const awake=publish(148);
assert(!awake.bodies[0].sleeping);assert.equal(awake.bodies[0].membership.generation,2);
game.expireProjectile(state,'return-test');const pickup=publish(164);
assert(pickup.bodies[0].pickup);assert.equal(pickup.bodies[0].status,'pickup');assert.equal(pickup.bodies[0].identity,originalIdentity);

// New same-authored-revision captures are legal, ordered by request sequence.
const nextEnvironment={...environment,environmentAssetIdentity:'studio-scene:changed',environmentSequence:2,
  buildingPlacement:environment.buildingPlacement.slice(),unitM:.25};
nextEnvironment.buildingPlacement[12]+=2;
store.setEnvironment(nextEnvironment);
assert.equal(store.read(),pickup,'staging a new environment never clears still-live bodies');
const oldWhilePending=publish(180);assert.equal(oldWhilePending.environment.environmentSequence,1);
game.setEnvironmentReceipt(receipt(nextEnvironment));const switched=publish(196);
assert.equal(switched.environment.environmentSequence,2);assert.equal(switched.bodies.length,1);
assert.equal(switched.bodies[0].identity,originalIdentity);
switched.bodies[0].positionStudio.forEach((v,i)=>near(v,(state.position[i]-nextEnvironment.buildingPlacement[12+i])/.25));
assert.equal(store.receive(edit(p=>p.environments=[receipt(environment)],212)).reason,'environment-receipt-not-registered');
assert.throws(()=>store.setEnvironment(environment),/regressed/);
assert.throws(()=>store.setEnvironment({...nextEnvironment,unitM:.5}),/two different/);
assert.throws(()=>store.setEnvironment({...nextEnvironment,buildingIdentity:'other',environmentSequence:3}),/cannot change/);
assert.equal(store.setEnvironment(nextEnvironment).environmentSequence,2);
assert(game.collectProjectilePickup(state));const empty=publish(228);
assert.equal(empty.bodies.length,0,'absence in complete accepted original frame means removal');

// Rotated registration maps the same world point and explicit fixed world
// orientation back to Studio, without inventing simulated angular state.
const rotatedEnvironment={...environment,buildingPlacement:[0,0,-1,0,0,1,0,0,1,0,0,0,5,6,7,1]};
const rotatedStore=createMechanicalCreatureStudioReturnStore({epoch:'session-A',environment:rotatedEnvironment,
  appearanceAssetIdentities:[marbleAsset.identity],fixedQuaternionXYZW:[0,0,0,1]});
const rotatedPacket=structuredClone(rawFirst);rotatedPacket.bodies[0].positionM=[5,6,6];
assert(rotatedStore.receive(rotatedPacket).accepted);
const rotatedBody=rotatedStore.read().bodies[0];near(rotatedBody.positionStudio[0],1/environment.unitM);
near(rotatedBody.positionStudio[1],0);near(rotatedBody.positionStudio[2],0);
near(rotatedBody.model[2],.11/environment.unitM);near(rotatedBody.model[8],-.11/environment.unitM);
assert.equal(rotatedBody.physicsQuaternionXYZW,null);

// Optional renderer contract: real hydration, pure store and rig math; only
// GPU allocation/draw is mocked. No claim that this test renders real pixels.
let drawContractTested=false;
if(vm.SourceTextModule){
  const draws=[],renderers=[],ctx=vm.createContext({structuredClone});
  const source=read('examples/meowdoku/web/mechanical_creature_studio_return.js');
  const exports={
    './marble_game_asset.js':assetModule,
    './marble_studio_subject_renderer.js':{validateStudioLightTransform,createStudioSubjectRenderer:args=>{
      const r={args,disposed:false};renderers.push(r);return {draw:p=>{assert(!r.disposed);draws.push(p);},dispose:()=>r.disposed=true};}},
    './mechanical_creature_studio_collection.js':{studioRigToGame},'./marble_roll.js':{rotationMatrix},
  };
  const m=new vm.SourceTextModule(source,{context:ctx});
  await m.link(name=>new vm.SyntheticModule(Object.keys(exports[name]),function(){for(const [key,value]of Object.entries(exports[name]))this.setExport(key,value);},{context:ctx}));
  await m.evaluate();
  const returned=await m.namespace.createMechanicalCreatureStudioReturn({epoch:'session-A',environment,
    assets:[marbleAsset],fixedQuaternionXYZW:[0,0,0,1]});
  returned.receive(rawFirst);const gl={},I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
  const draw=returned.draw({gl,view:I,projection:I});assert.equal(draw.draws,2);assert.equal(renderers.length,1);
  assert.deepEqual(draws.map(p=>p.phase),['opaque','transparent']);
  for(const call of draws)assert.deepEqual(Array.from(call.lightTransform),I,'captured Studio rig is not moved with each returning body');
  assert.equal(renderers[0].args.packet.scene.subject.faces.length,72960);
  returned.draw({gl,view:I,projection:I,phase:'transparent'});assert.equal(renderers.length,1);
  assert.throws(()=>returned.draw({gl:{},view:I,projection:I}),/context changed/);
  returned.dispose();assert(renderers[0].disposed);assert.equal(returned.read(),null);drawContractTested=true;
}
store.dispose();assert.equal(store.read(),null);assert.throws(()=>store.receive(rawFirst),/closed/);
console.log(JSON.stringify({passed:true,actualOriginalLifecycleFunctions:names.length,actualBridgePublisher:true,
  epochAndStaleFrameGates:true,doubleSlotPublication:true,environmentSequenceGates:true,
  sameRevisionCaptureAccepted:true,originalRadiusM:.11,identityLightTransform:true,
  workerAppliedEnvironmentClaim:false,drawContractTested,nativePhysicsTested:false,browserRenderingTested:false}));
