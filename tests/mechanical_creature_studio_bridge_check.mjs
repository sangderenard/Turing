import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import vm from "node:vm";
import {createMechanicalCreatureStudioReturnStore} from "../examples/meowdoku/web/mechanical_creature_studio_return.js";

const root = new URL("../", import.meta.url);
const base = process.argv[2] || "build/dot-living-map-runtime-20261110/matched/index.html";
const page = readFileSync(new URL(base, root), "utf8");
const model = JSON.parse(page.match(/<script type="application\/json" id="abstract-ui-model">([\s\S]*?)<\/script>/)[1]);
const initial = JSON.parse(readFileSync(new URL("build/dot-studio-game-transfer-20261110/studio-building-definition-v2.json", root), "utf8"));
const faults = [], posts = [];
const context = vm.createContext({structuredClone, Set, Map, Float32Array, console, DOMException,
  model, mechanicalCreatureStudioBuildingDefinition: initial,
  stateLoopRuntime: {ready: true, latestSequence: 11, bodies: new Map(), slotRecords: [],
    worker: {postMessage: value => posts.push(value)}},
  projectileState: new Map(), projectilePickupState: new Map(),
  reportRuntimeFault: (...values) => faults.push(values),
});
vm.runInContext(readFileSync(new URL("examples/meowdoku/web/mechanical_creature_studio_building.js", root), "utf8"), context);
context.appendMechanicalCreatureStudioBuilding(model, initial);
const exact = (name, kind = "function") => {
  const found = page.match(new RegExp(`^${kind} ${name}[^]*?^}`, "m"));
  assert.ok(found, `original ${name}`); return found[0];
};
vm.runInContext(exact("TuringWorldRegistry", "class") + "\nglobalThis.turingWorld=new TuringWorldRegistry(model.world);", context);
vm.runInContext(["colorVector", "boxYawDegrees", "buildExtrudedBoxMesh", "rebuildPortableSceneMesh"].map(name => exact(name)).join("\n"), context);
const boxes = model.document_geometry.boxes.slice(-initial.boxes.length);
context.shaderViewer = {geometry: boxes, revision: 1, cameraPosition: [1, 2, 3], cameraFacing: [0, 0, -1]};
Object.assign(context, {vehiclePresentationGeometry: () => [], buildNavigationRouteMesh: () => new Float32Array(),
  rebuildPortalGraph() {}, buildPortalSplatMesh: () => new Float32Array(), buildPortalTubeMesh: () => new Float32Array(),
  installSceneMesh(mesh) {context.shaderViewer.mesh = mesh; context.shaderViewer.vertexCount = mesh.length / 9;},
  installVehiclePresentationMesh() {}});
context.rebuildPortableSceneMesh(); posts.length = 0;
vm.runInContext(readFileSync(new URL("examples/meowdoku/web/mechanical_creature_studio_bridge.js", root), "utf8"), context);
const api = context.mechanicalCreatureStudioBridge;
const originalIds = JSON.stringify(model.world.identity_specialization), originalLabels = boxes.map(box => box.label);
const originalCount = model.world.objects.length, originalWorker = model.loop_deployment.workers[0].source;
const definition = structuredClone(initial); definition.assetIdentity = "studio-scene:revision-10"; definition.sourceRevision = 10;
for (const box of definition.boxes) box.appearance.studio_asset_identity = definition.assetIdentity;
definition.boxes[1].center[0] += .25; definition.boxes[1].half_extent[0] *= 1.2;
definition.boxes[1].height += .1;
const originalX = boxes[1].center[0]; let committed = false;
const pending = api.queueEnvironment({assetIdentity: definition.assetIdentity, revision: 10, sequence: 1, definition,
  commit(receipt) {assert.equal(receipt.sequence, 1); assert.equal(posts.length, 1); committed = true;}});
assert.equal(boxes[1].center[0], originalX, "enqueue never changes live game geometry");
definition.boxes[1].center[0] = 9999;
context.applyMechanicalCreatureStudioEnvironment();
const receipt = await pending;
assert.equal(committed, true); assert.equal(receipt.colliderPublication, "posted-to-original-worker");
assert.equal(boxes[1].center[0], originalX + .25, "owned complete snapshot prevents caller tearing");
assert.equal(posts.length, 1, "one original native collider rebuild");
assert.equal(posts[0].type, "colliders"); assert.equal(posts[0].colliders.length, 15);
assert.equal(context.shaderViewer.colliders.length, 15);
assert.equal(model.world.objects.length, originalCount); assert.equal(JSON.stringify(model.world.identity_specialization), originalIds);
assert.deepEqual(boxes.map(box => box.label), originalLabels); assert.equal(model.loop_deployment.workers[0].source, originalWorker);
assert.ok(boxes.every(box => box.appearance.studio_asset_identity === receipt.assetIdentity));
const worldBox = model.world.objects.find(object => object.identity === boxes[1].identity);
assert.equal(worldBox.transform.position[0], originalX + .25); assert.equal(worldBox.form.height, boxes[1].height);
const request = sequence => {const d = structuredClone(initial); d.assetIdentity = `studio-scene:${sequence}`;
  for (const box of d.boxes) box.appearance.studio_asset_identity = d.assetIdentity;
  return {assetIdentity: d.assetIdentity, revision: d.sourceRevision, sequence, definition: d};};
const canceled = api.queueEnvironment({...request(2), isCurrent: () => false});
const canceledCheck = assert.rejects(canceled, {name: "AbortError"});
context.applyMechanicalCreatureStudioEnvironment(); await canceledCheck; assert.equal(posts.length, 1);
const replaced = api.queueEnvironment(request(3)); const replacedCheck = assert.rejects(replaced, {name: "AbortError"});
const replacement = api.queueEnvironment(request(4)); await replacedCheck;
context.applyMechanicalCreatureStudioEnvironment(); await replacement; assert.equal(posts.length, 2);
assert.throws(() => api.queueEnvironment(request(4)), /Stale/);
const bad = request(5); bad.definition.boxes[1].identity = "not-the-registered-object";
assert.throws(() => api.queueEnvironment(bad), /identity|ownership/);
const badOpening = request(5); badOpening.definition.boxes[0].openings[0].identity = "new-opening";
assert.throws(() => api.queueEnvironment(badOpening), /openings/);
const beforeFailure = boxes[1].center[0];
const failing = api.queueEnvironment({...request(5), commit: () => false});
const failingCheck = assert.rejects(failing, /synchronously/);
context.applyMechanicalCreatureStudioEnvironment(); await failingCheck;
assert.equal(boxes[1].center[0], beforeFailure); assert.equal(faults.length, 1);
assert.equal(posts.length, 4, "failed commit republishes original native geometry without claiming acknowledgment");

const active = {identity: "marble:1", appearanceAssetIdentity: "marble:asset", ammunitionItemIdentity: "ammo:1",
  position: [2, 3, 4], velocity: [5, 6, 7], box: {identity: "marble:1", radius: .05},
  record: {identity: "marble:1", geometry: {radius: .05}, status: "active"}};
const body = {identity: active.identity, slot: 0, generation: 2};
context.projectileState.set(active.identity, active); context.stateLoopRuntime.bodies.set(active.identity, body);
context.stateLoopRuntime.slotRecords[0] = body;
const packets = []; api.setSink(packet => packets.push(packet), {epoch: "test-original-game"});
context.publishMechanicalCreatureStudioSnapshot(1000, .025);
const first = api.readSnapshot(); assert.equal(first, packets[0]); assert.equal(first.frameDtSeconds, .025);
assert.equal(first.workerSequence, 11); assert.equal(first.environments[0].sequence, 4);
assert.equal(first.bodies[0].membership.generation, 2); assert.ok(Object.isFrozen(first.bodies[0].positionM));
active.position[0] = 8; context.shaderViewer.cameraPosition[0] = 9;
context.publishMechanicalCreatureStudioSnapshot(1016, .016);
const second = api.readSnapshot(); assert.notEqual(first, second); assert.equal(first.bodies[0].positionM[0], 2);
assert.equal(first.camera.positionM[0], 1); assert.equal(second.camera.positionM[0], 9);
assert.equal(second.bodies[0].positionM[0], 8);
context.projectileState.delete(active.identity); context.projectilePickupState.set(active.identity, active);
context.stateLoopRuntime.bodies.delete(active.identity); context.stateLoopRuntime.slotRecords[0] = null;
active.record.status = "pickup";
context.publishMechanicalCreatureStudioSnapshot(1032, .016); assert.equal(api.readSnapshot().bodies[0].pickup, true);
context.projectilePickupState.clear(); context.publishMechanicalCreatureStudioSnapshot(1048, .016);
assert.equal(api.readSnapshot().bodies.length, 0, "full snapshot communicates removal");

context.stateLoopRuntime.ready = false;
const workerPending = api.queueEnvironment(request(6)); context.applyMechanicalCreatureStudioEnvironment();
assert.equal((await workerPending).colliderPublication, "native-scene-ready-worker-pending");
context.publishMechanicalCreatureStudioSnapshot(1064, .016);
assert.equal(api.readSnapshot().environments[0].colliderPublication, "native-scene-ready-worker-pending");
context.stateLoopRuntime.ready = true;
context.publishMechanicalCreatureStudioSnapshot(1080, .016);
assert.equal(api.readSnapshot().environments[0].colliderPublication, "native-scene-ready-worker-pending", "ready alone is not a publication receipt");
context.markMechanicalCreatureStudioColliderPublication();
context.publishMechanicalCreatureStudioSnapshot(1096, .016);
assert.equal(api.readSnapshot().environments[0].colliderPublication, "posted-to-original-worker");

// Combined original launcher -> actual bridge -> Studio return consumer.
// Reuse this original registry, native box realization and collider publisher.
// DOM/audio and worker transport remain explicit test doubles, not a solver.
const node=()=>({dataset:{},style:{setProperty(){}},setAttribute(){},append(){}});
Object.assign(context,{physicsRuntime:{instance:true},hotbarState:{inventory:{items:[{quantity:20,maximum_stack:50,
  properties:{operation:'projectile-ammunition'}}]}},vehicleRuntime:{active:null},viewportControls:{policy:{actor:model.identity}},
  entityState:new Map(),index:new Map(),projectileSequence:0,performance:{now:()=>1000},
  document:{querySelector:()=>null,getElementById:()=>node()},div:node,makeSelectable:node,
  ensureMusicRoomAudio:()=>Promise.resolve(),spectrumColorHex:()=>'#123456',refreshInventoryCounts(){},setPlacementStatus(){},
  portalRuntime:{lastPositions:new Map(),cooldowns:new Map(),transits:new Map()}});
vm.runInContext(['normalized3','initializePhysicsSnapshotSlots','reservePhysicsSnapshotSlot','releasePhysicsSnapshotSlot',
  'registerProjectilePhysicsMembership','projectileAmmoItem','firePhysicsBall','expireProjectile','collectProjectilePickup']
  .map(name=>exact(name)).join('\n'),context);
context.initializePhysicsSnapshotSlots(128);
assert(context.firePhysicsBall());
const fired=[...context.projectileState.values()][0];
fired.appearanceAssetIdentity='studio-marble:actual-fired-appearance';fired.ammunitionItemIdentity='inventory:actual-fired-item';
assert.equal(fired.box.radius,model.projectiles.archetype.geometry.radius);
const applied=api.readSnapshot().environments[0];
const returnEnvironment={buildingIdentity:initial.identity,environmentAssetIdentity:applied.assetIdentity,
  environmentRevision:applied.revision,environmentSequence:applied.sequence,buildingPlacement:initial.placement,unitM:initial.unitM};
const returned=createMechanicalCreatureStudioReturnStore({epoch:'combined-original-game',environment:returnEnvironment,
  appearanceAssetIdentities:[fired.appearanceAssetIdentity],fixedQuaternionXYZW:[0,0,0,1]});
const outcomes=[];
api.setSink(packet=>outcomes.push(returned.receive(packet)),{epoch:'combined-original-game'});
context.publishMechanicalCreatureStudioSnapshot(1128,.016);
assert(outcomes.at(-1).accepted);assert.equal(returned.read().bodies[0].identity,fired.identity);
assert.equal(returned.read().frameDtSeconds,.016);assert.equal(returned.read().workerSequence,context.stateLoopRuntime.latestSequence);
assert.equal(returned.read().bodies[0].radiusM,.11);
returned.read().bodies[0].positionStudio.forEach((value,axis)=>assert(Math.abs(value-
  (fired.position[axis]-initial.translationM[axis])/initial.unitM)<1e-12));
const editRequest=request(7);editRequest.definition.boxes[1].center[0]+=.375;
const beforeEditedCenter=boxes[1].center[0],nextReturnEnvironment={...returnEnvironment,
  environmentAssetIdentity:editRequest.assetIdentity,environmentRevision:editRequest.revision,environmentSequence:7};
returned.setEnvironment(nextReturnEnvironment);
context.publishMechanicalCreatureStudioSnapshot(1144,.016);
const beforeEditedPacket=api.readSnapshot();assert(outcomes.at(-1).accepted);
assert.equal(returned.read().environment.environmentSequence,6,'old frames remain usable while a new environment waits');
const editPending=api.queueEnvironment(editRequest);context.applyMechanicalCreatureStudioEnvironment();
const editedReceipt=await editPending;
assert.equal(editedReceipt.sequence,7);assert.equal(editedReceipt.colliderPublication,'posted-to-original-worker');
assert.equal(boxes[1].center[0],beforeEditedCenter+.375);
assert.equal(posts.at(-1).type,'colliders','original native rebuild publishes edited contacts');
context.publishMechanicalCreatureStudioSnapshot(1160,.016);
assert(outcomes.at(-1).accepted);assert.equal(returned.read().environment.environmentSequence,7);
assert.equal(returned.read().environmentReceipt.assetIdentity,editedReceipt.assetIdentity);
assert.equal(returned.read().bodies[0].identity,fired.identity,'environment edits preserve existing body identity');
const staleReceipt=structuredClone(beforeEditedPacket);staleReceipt.frameTimeMs=1176;
staleReceipt.geometryRevision=returned.read().geometryRevision;
assert.equal(returned.receive(staleReceipt).accepted,false,'superseded actual environment receipt is refused');
context.expireProjectile(fired,'combined-return-test');context.publishMechanicalCreatureStudioSnapshot(1192,.016);
assert(returned.read().bodies[0].pickup);assert.equal(returned.read().bodies[0].membership,null);
assert(context.collectProjectilePickup(fired));context.publishMechanicalCreatureStudioSnapshot(1208,.016);
assert.equal(returned.read().bodies.length,0);
assert.equal(faults.length,1,'only the deliberate earlier rollback fault was reported');
api.setSink(null); context.publishMechanicalCreatureStudioSnapshot(1224, .016); assert.equal(api.readSnapshot(), null);

// Execute the COMPLETE original worker and its exact original main-thread
// message handler. Controlled queues expose legal delivery order; no timer or
// physics tick is run and no worker protocol is substituted. Actual structured
// cloning/transfers detach the original buffers, as the browser transport does.
const workerInbox=[],mainInbox=[];
const originalWorkerContext=vm.createContext({onmessage:null,performance:{now:()=>0},console,
  postMessage:(message,transfer=[])=>mainInbox.push(structuredClone(message,{transfer})),
  setInterval:()=>1,clearInterval(){},close(){}});
vm.runInContext(originalWorker,originalWorkerContext);
vm.runInContext(`instance={};timer=1;sequence=40;snapshotCapacity=1;
  snapshotBuffers=Array.from({length:SNAPSHOT_POOL_SIZE},()=>new ArrayBuffer(SNAPSHOT_STRIDE*8));`,originalWorkerContext);
const workerObservers=new Set();
const originalTransport={postMessage:(message,transfer=[])=>workerInbox.push(structuredClone(message,{transfer})),
  addEventListener(type,listener){assert.equal(type,'message');workerObservers.add(listener);},
  removeEventListener(type,listener){assert.equal(type,'message');workerObservers.delete(listener);}};
context.stateLoopRuntime.worker=originalTransport;context.stateLoopRuntime.latestSequence=40;
context.stateLoopRuntime.slotRecords=[];
const originalHandler=page.slice(page.indexOf('      worker.onmessage = event => {',page.indexOf('function initializeWorldPhysicsWasm')),
  page.indexOf('      worker.onerror = event => {',page.indexOf('function initializeWorldPhysicsWasm')));
assert(originalHandler.startsWith('      worker.onmessage = event => {'));
vm.runInContext(`{const worker=stateLoopRuntime.worker;${originalHandler}}`,context);
const toWorker=async()=>{assert(workerInbox.length);const message=workerInbox.shift();
  await originalWorkerContext.onmessage({data:message});return message;};
const toMain=()=>{assert(mainInbox.length);const message=mainInbox.shift(),event={data:message};
  originalTransport.onmessage(event);for(const observer of workerObservers)observer(event);return message;};
const nativePublish=time=>originalWorkerContext.publishSnapshot(time);
nativePublish(2000);nativePublish(2001);
assert.equal(mainInbox.length,1,'the original three-buffer pool still permits only one snapshot in flight');
assert.equal(mainInbox[0].sequence,41);
const barrierRequest=request(8);barrierRequest.definition.boxes[1].center[0]+=.625;
const barrierUpdate=api.queueEnvironment(barrierRequest);context.applyMechanicalCreatureStudioEnvironment();await barrierUpdate;
assert.equal(workerInbox[0].type,'colliders');
api.setSink(()=>{},{epoch:'native-worker-barrier'});
context.publishMechanicalCreatureStudioSnapshot(2002,.016);
assert.deepEqual(structuredClone(api.readSnapshot().environments[0].workerColliderBarrier),{
  scope:'worker-collider-field',status:'awaiting-original-recycle',postedAfterSequence:40,requiredSequence:42,observedSequence:null});
const postedPacket=api.readSnapshot();
context.stateLoopRuntime.latestSequence=99;context.publishMechanicalCreatureStudioSnapshot(2002.5,.016);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier.observedSequence,null,
  'a global sequence change without this worker message is not a barrier');
context.stateLoopRuntime.latestSequence=40;
toMain();assert.equal(context.stateLoopRuntime.latestSequence,41);
assert.deepEqual(workerInbox.map(message=>message.type),['colliders','recycle']);
assert.equal(vm.runInContext('colliders.length',originalWorkerContext),0,
  'N+1 can reach main while the environment collider message is still unconsumed');
context.publishMechanicalCreatureStudioSnapshot(2003,.016);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier.observedSequence,null,'N+1 cannot acknowledge the environment');
await toWorker();
assert.equal(vm.runInContext('colliders.length',originalWorkerContext),15);
assert.equal(mainInbox.length,0,'full-dynamics collider handling emits no engine-state acknowledgment');
nativePublish(2004);assert.equal(mainInbox.length,0,'a consumed collider message cannot bypass the outstanding buffer');
await toWorker();nativePublish(2005);assert.equal(mainInbox[0].sequence,42);
toMain();context.publishMechanicalCreatureStudioSnapshot(2006,.016);
assert.equal(api.readSnapshot().environments[0].colliderPublication,'posted-to-original-worker');
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier.observedSequence,42);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier.status,'observed-after-original-recycle');
assert.equal(postedPacket.environments[0].workerColliderBarrier.observedSequence,null,'prior receipts remain immutable');
assert.equal(context.stateLoopRuntime.latestSequence,42);
assert.equal(vm.runInContext('JSON.stringify(colliders)',originalWorkerContext),JSON.stringify(context.shaderViewer.colliders),
  'original N+2 publication follows consumption of the exact original collider payload');
assert.equal(model.loop_deployment.workers[0].source,originalWorker,'worker code is untouched');

// A rejected render commit posts the candidate and then the restored field.
// Neither the prior proof nor N+1 may acknowledge that rollback publication.
const rollbackPending=api.queueEnvironment({...request(9),commit:()=>false});
const rollbackRejected=assert.rejects(rollbackPending,/synchronously/);
context.applyMechanicalCreatureStudioEnvironment();await rollbackRejected;
context.publishMechanicalCreatureStudioSnapshot(2007,.016);
assert.equal(api.readSnapshot().environments[0].sequence,8);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier.requiredSequence,44);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier.observedSequence,null);
assert.deepEqual(workerInbox.map(message=>message.type),['recycle','colliders','colliders']);
await toWorker();nativePublish(2008);toMain();
context.publishMechanicalCreatureStudioSnapshot(2009,.016);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier.observedSequence,null);
await toWorker();await toWorker();await toWorker();nativePublish(2010);toMain();
context.publishMechanicalCreatureStudioSnapshot(2011,.016);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier.observedSequence,44);
assert.equal(vm.runInContext('JSON.stringify(colliders)',originalWorkerContext),JSON.stringify(context.shaderViewer.colliders));
assert.equal(workerObservers.size,1,'reposting replaces the old passive observer');

const retiredEpochObserver=[...workerObservers][0];
api.setSink(null);assert.equal(workerObservers.size,0,'closed epochs unregister the passive observer');
assert.equal(api.readSnapshot(),null);
api.setSink(()=>{},{epoch:'reopened-native-worker-barrier'});
assert.equal(workerObservers.size,1);
context.publishMechanicalCreatureStudioSnapshot(2012,.016);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier.observedSequence,null);
retiredEpochObserver({data:{type:'snapshot-buffer',sequence:44}});
context.publishMechanicalCreatureStudioSnapshot(2013,.016);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier.observedSequence,null,
  'a retired epoch observer cannot satisfy the new epoch');
await toWorker();nativePublish(2014);toMain();context.publishMechanicalCreatureStudioSnapshot(2015,.016);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier.observedSequence,45);

const retiredWorkerObserver=[...workerObservers][0];
context.stateLoopRuntime.worker={postMessage(){}};
retiredWorkerObserver({data:{type:'snapshot-buffer',sequence:45}});
context.publishMechanicalCreatureStudioSnapshot(2016,.016);
assert.equal(workerObservers.size,0,'worker replacement unregisters the old source observer');
assert.equal(api.readSnapshot().environments[0].colliderPublication,'native-scene-ready-worker-pending');
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier,null,
  'replacement workers cannot inherit the retired worker proof');
context.stateLoopRuntime.worker=originalTransport;context.stateLoopRuntime.ready=false;
context.markMechanicalCreatureStudioColliderPublication();context.publishMechanicalCreatureStudioSnapshot(2017,.016);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier,null);
context.stateLoopRuntime.ready=true;
originalTransport.postMessage({type:'colliders',colliders:context.shaderViewer.colliders});
context.markMechanicalCreatureStudioColliderPublication();context.publishMechanicalCreatureStudioSnapshot(2018,.016);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier.postedAfterSequence,45);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier.requiredSequence,47);
assert.equal(workerObservers.size,1);
context.stateLoopRuntime.latestSequence=0;context.publishMechanicalCreatureStudioSnapshot(2019,.016);
assert.equal(api.readSnapshot().environments[0].workerColliderBarrier,null,'sequence reset invalidates proof even on the same object');
assert.equal(workerObservers.size,0);
assert.equal(faults.length,2,'only the two deliberate commit failures were reported');
api.setSink(null);

// Bounded topology: disable and restore only startup-registered table parts.
// The small authored-source fixture exercises prepareEnvironment as well as
// real queued native rebuilding; no fabricated registry or contact solver.
context.stateLoopRuntime.worker={postMessage:message=>posts.push(message)};
context.initializePhysicsSnapshotSlots(128);assert(context.firePhysicsBall());
const retainedProjectile=[...context.projectileState.values()][0];
const projectileBefore=JSON.stringify({position:retainedProjectile.position,velocity:retainedProjectile.velocity,
  radius:retainedProjectile.box.radius,membership:context.stateLoopRuntime.bodies.get(retainedProjectile.identity)});
const geometry=context.shaderViewer.geometry,registeredRefs=initial.proxyIdentities.map(identity=>
  model.document_geometry.boxes.find(box=>box.identity===identity));
const unrelated=model.document_geometry.boxes.find(box=>box.label==='Outer yard north flat apron');
assert(unrelated);geometry.unshift(unrelated);
const documentRefs=[...model.document_geometry.boxes],worldRefs=[...model.world.objects],
  runtimeRefs=[...context.turingWorld.runtimeObjects],registryBefore=JSON.stringify(model.world.identity_specialization),
  playerBefore=JSON.stringify(model.entity_mezzanine),hiddenCenter=[...registeredRefs[1].center];
const sourceRoom={},axes=[['leftX','rightX'],['floorY','ceilingY'],['backZ','frontZ']];
axes.forEach(([low,high],axis)=>{sourceRoom[low]=(initial.boundsM.minimum[axis]-initial.translationM[axis])/initial.unitM;
  sourceRoom[high]=(initial.boundsM.maximum[axis]-initial.translationM[axis])/initial.unitM;});
const tableOffSource={schema:'turing.studio-class-world-import.v1',classIdentity:initial.sourceClassIdentity,revision:10,
  scene:{frame:{unitM:initial.unitM},room:sourceRoom,parts:[]}};
const tableOff=api.prepareEnvironment(tableOffSource,{assetIdentity:'studio-scene:table-off'});
assert.equal(tableOff.definition.boxes.length,initial.boxes.length,'complete registered descriptors remain retained');
assert.deepEqual([...tableOff.definition.activeProxyIdentities],[initial.identity]);
assert.deepEqual([...tableOff.building.proxyIds],[initial.identity],'only active native proxies are sent to renderer admission');
assert.deepEqual([...tableOff.definition.boxes[1].center],hiddenCenter,'hidden table retains last admitted geometry');
assert.notEqual(hiddenCenter[0],initial.boxes[1].center[0],'retained geometry differs from the startup pose');
const disable=api.queueEnvironment({...tableOff,sequence:10});context.applyMechanicalCreatureStudioEnvironment();await disable;
const studioContacts=()=>context.shaderViewer.colliders.filter(row=>initial.proxyIdentities.includes(row.objectIdentity));
assert.equal(studioContacts().length,6);assert.equal(geometry,context.shaderViewer.geometry);
assert(geometry.includes(unrelated));assert(geometry.includes(retainedProjectile.box));
assert(registeredRefs.slice(1).every(box=>!geometry.includes(box)&&box.placement.custody==='placed'));
assert(!context.shaderViewer.identitySpans.some(span=>span.identity.includes('/part:table/')));
const disabledGeometry=geometry.slice();
const rollbackEnable=api.queueEnvironment({...request(11),commit:()=>false});
const rollbackEnableCheck=assert.rejects(rollbackEnable,/synchronously/);
context.applyMechanicalCreatureStudioEnvironment();await rollbackEnableCheck;
assert.equal(studioContacts().length,6);
disabledGeometry.forEach((box,index)=>assert.equal(geometry[index],box,'failed topology commit restores exact active geometry order'));
assert.equal(geometry.length,disabledGeometry.length);
const enable=api.queueEnvironment(request(12));context.applyMechanicalCreatureStudioEnvironment();await enable;
assert.equal(studioContacts().length,15);
registeredRefs.forEach(box=>assert.equal(geometry.find(value=>value.identity===box.identity),box));
const singleOff=request(13);singleOff.definition.activeProxyIdentities=initial.proxyIdentities.filter(id=>id!==registeredRefs[3].identity);
const partial=api.queueEnvironment(singleOff);context.applyMechanicalCreatureStudioEnvironment();await partial;
assert.equal(studioContacts().length,14);assert(!geometry.includes(registeredRefs[3]));
for(const span of context.shaderViewer.identitySpans)
  assert.equal(geometry[span.boxIndex].identity,span.identity,'unchanged native picking indices match each active original box');
assert.equal(geometry[0],unrelated);assert(geometry.includes(retainedProjectile.box));
assert.equal(JSON.stringify({position:retainedProjectile.position,velocity:retainedProjectile.velocity,
  radius:retainedProjectile.box.radius,membership:context.stateLoopRuntime.bodies.get(retainedProjectile.identity)}),projectileBefore);
assert.equal(JSON.stringify(model.entity_mezzanine),playerBefore);
assert.equal(JSON.stringify(model.world.identity_specialization),registryBefore);
documentRefs.forEach((box,index)=>assert.equal(model.document_geometry.boxes[index],box));
worldRefs.forEach((object,index)=>assert.equal(model.world.objects[index],object));
runtimeRefs.forEach((object,index)=>assert.equal(context.turingWorld.runtimeObjects[index],object));
const invalidActive=request(14);invalidActive.definition.activeProxyIdentities=[initial.identity,'unregistered-table'];
assert.throws(()=>api.queueEnvironment(invalidActive),/registered subset/);
const noRoom=request(14);noRoom.definition.activeProxyIdentities=initial.proxyIdentities.slice(1);
assert.throws(()=>api.queueEnvironment(noRoom),/containing its building/);
const custody=request(14);custody.definition.boxes[1].placement.custody='inventory';
assert.throws(()=>api.queueEnvironment(custody),/ownership/);
assert.equal(faults.length,3,'only deliberate commit failures were reported');

// A genuinely later-authored cuboid is admitted only by the canonical class
// extension. The unpatched original must refuse it, not invent another registry.
const cuboid=(id,x)=>({id,construction:'cuboid',closed:true,convex:true,faces:[
  [[x,0,0],[x+1,0,0],[x+1,1,1]],[[x,0,0],[x+1,1,1],[x,1,1]] ]});
const laterSource=structuredClone(tableOffSource);laterSource.revision=14;
laterSource.scene.parts=[cuboid('table/slab',0),cuboid('shelf/later',2)];
const lateIdentity=initial.identity+'/part:shelf/later';
if(typeof context.turingWorld.appendObjects!=='function') {
  assert.throws(()=>api.prepareEnvironment(laterSource,{assetIdentity:'studio-scene:later'}),/canonical append admission/);
  assert.equal(context.turingWorld.object(lateIdentity),null);
} else {
  const countBefore=model.world.objects.length,partCountBefore=model.world.identity_specialization.semantic_parts.length;
  const newPrepared=api.prepareEnvironment(laterSource,{assetIdentity:'studio-scene:later'});
  assert.equal(model.world.objects.length,countBefore,'preparation never admits an object');
  assert.equal(newPrepared.definition.boxes.length,11);
  assert.deepEqual([...newPrepared.building.proxyIds],[initial.identity,registeredRefs[1].identity,lateIdentity]);
  const beforeLateGeometry=geometry.slice(),beforeLateRevision=context.shaderViewer.revision;
  api.setSink(()=>{},{epoch:'late-admission'});context.publishMechanicalCreatureStudioSnapshot(3000,.016);
  const beforeLateReceipt=api.readSnapshot().environments[0];
  const failedLate=api.queueEnvironment({...newPrepared,sequence:14,commit:()=>false});
  assert.equal(model.world.objects.length,countBefore,'enqueue never admits an object');
  const failedLateCheck=assert.rejects(failedLate,/synchronously/);
  context.applyMechanicalCreatureStudioEnvironment();await failedLateCheck;
  assert.equal(model.world.objects.length,countBefore+1,'late failure retains the canonical admission');
  const lateObject=context.turingWorld.object(lateIdentity),lateBox=lateObject.extensions['abstract_ui.document_geometry'];
  const lateRuntimeId=context.turingWorld.objectRuntimeId(lateIdentity),latePartId=context.turingWorld.partRuntimeId(lateIdentity+'/surface:body');
  assert(lateRuntimeId>0&&latePartId>0);assert(model.document_geometry.boxes.includes(lateBox));
  assert(!geometry.includes(lateBox),'late render failure leaves admitted part inactive');
  assert.equal(model.world.identity_specialization.semantic_parts.length,partCountBefore+1);
  beforeLateGeometry.forEach((box,index)=>assert.equal(geometry[index],box));
  assert.equal(geometry.length,beforeLateGeometry.length);assert.equal(context.shaderViewer.revision,beforeLateRevision);
  context.publishMechanicalCreatureStudioSnapshot(3016,.016);
  assert.equal(api.readSnapshot().environments[0].assetIdentity,beforeLateReceipt.assetIdentity);
  assert.equal(api.readSnapshot().environments[0].sequence,13,'inactive admission is not an activated environment receipt');
  assert(!context.shaderViewer.colliders.some(row=>row.objectIdentity===lateIdentity));

  laterSource.revision=15;
  const retryPrepared=api.prepareEnvironment(laterSource,{assetIdentity:'studio-scene:later-retry'});
  const retry=api.queueEnvironment({...retryPrepared,sequence:15});context.applyMechanicalCreatureStudioEnvironment();await retry;
  assert.equal(model.world.objects.length,countBefore+1,'retry allocates no duplicate object');
  assert.equal(model.world.identity_specialization.semantic_parts.length,partCountBefore+1);
  assert.equal(context.turingWorld.object(lateIdentity),lateObject);assert.equal(context.turingWorld.objectRuntimeId(lateIdentity),lateRuntimeId);
  assert.equal(context.turingWorld.partRuntimeId(lateIdentity+'/surface:body'),latePartId);assert(geometry.includes(lateBox));
  assert.equal(geometry.find(box=>box.identity===registeredRefs[1].identity),registeredRefs[1],'mixed old/new update preserves old refs');
  assert.equal(context.shaderViewer.colliders.filter(row=>row.objectIdentity===lateIdentity).length,1);
  assert.equal(context.shaderViewer.colliders.find(row=>row.objectIdentity===lateIdentity).runtimePartId,latePartId);
  context.publishMechanicalCreatureStudioSnapshot(3032,.016);
  assert.equal(api.readSnapshot().environments[0].sequence,15);
  const lateOffSource=structuredClone(tableOffSource);lateOffSource.revision=16;
  const lateOff=api.prepareEnvironment(lateOffSource,{assetIdentity:'studio-scene:later-off'});
  const lateOffPending=api.queueEnvironment({...lateOff,sequence:16});context.applyMechanicalCreatureStudioEnvironment();await lateOffPending;
  assert(!geometry.includes(lateBox));assert.equal(studioContacts().length,6);
  laterSource.revision=17;
  const lateOn=api.prepareEnvironment(laterSource,{assetIdentity:'studio-scene:later-on'});
  const lateOnPending=api.queueEnvironment({...lateOn,sequence:17});context.applyMechanicalCreatureStudioEnvironment();await lateOnPending;
  assert.equal(context.turingWorld.object(lateIdentity),lateObject);assert(geometry.includes(lateBox));
  assert.equal(model.world.objects.length,countBefore+1);

  const rejectedSource=structuredClone(laterSource);rejectedSource.revision=18;
  rejectedSource.scene.parts.push(cuboid('shelf/new-first',4),cuboid('shelf/rejected',6));
  const rejected=api.prepareEnvironment(rejectedSource,{assetIdentity:'studio-scene:rejected'});
  const rejectionCount=model.world.objects.length,rejectionGeometry=geometry.slice();
  rejected.definition.boxes.at(-1).half_extent[0]=-1;
  assert.throws(()=>api.queueEnvironment({...rejected,sequence:18}),/dimensions/);
  assert.equal(model.world.objects.length,rejectionCount,'entire batch validates before any admission');
  const wrongParent=api.prepareEnvironment(rejectedSource,{assetIdentity:'studio-scene:wrong-parent'});
  wrongParent.definition.boxes.at(-1).parent_identity='unrelated-building';
  assert.throws(()=>api.queueEnvironment({...wrongParent,sequence:18}),/owned authored static cuboid/);
  const wrongRecipe=api.prepareEnvironment(rejectedSource,{assetIdentity:'studio-scene:wrong-recipe'});
  wrongRecipe.definition.boxes.at(-1).physics.body='dynamic';
  assert.throws(()=>api.queueEnvironment({...wrongRecipe,sequence:18}),/owned authored static cuboid/);
  const wrongClass=structuredClone(rejectedSource);wrongClass.classIdentity='python:unrelated.Source';
  assert.throws(()=>api.prepareEnvironment(wrongClass,{assetIdentity:'studio-scene:wrong-class'}),/owner/);
  assert.equal(model.world.objects.length,rejectionCount);
  rejectionGeometry.forEach((box,index)=>assert.equal(geometry[index],box));
  assert.equal(context.turingWorld.object(initial.identity+'/part:shelf/new-first'),null);
  for(const span of context.shaderViewer.identitySpans)assert.equal(geometry[span.boxIndex].identity,span.identity);
  assert.equal(JSON.stringify({position:retainedProjectile.position,velocity:retainedProjectile.velocity,
    radius:retainedProjectile.box.radius,membership:context.stateLoopRuntime.bodies.get(retainedProjectile.identity)}),projectileBefore);
  assert.equal(JSON.stringify(model.entity_mezzanine),playerBefore);
  worldRefs.forEach((object,index)=>assert.equal(model.world.objects[index],object));
  runtimeRefs.forEach((object,index)=>assert.equal(context.turingWorld.runtimeObjects[index],object));
  // Prevalidate while arrays have their normal bounded size. Then exercise
  // only the admission boundary with sparse maximum lengths: never iterate,
  // clone or stringify the expanded document array.
  const capacityPrepared=api.prepareEnvironment(rejectedSource,{assetIdentity:'studio-scene:capacity'});
  const capacityUpdates=context.validateMechanicalCreatureStudioEnvironment({...capacityPrepared,sequence:18}).updates;
  assert.equal(capacityUpdates.filter(update=>!update.object).length,2);
  const documentBoxes=model.document_geometry.boxes,normalLength=documentBoxes.length,
    documentKeys=Object.keys(documentBoxes);
  const counts=()=>({world:model.world.objects.length,objects:context.turingWorld.objects.size,
    parts:context.turingWorld.semanticParts.size,runtimeObjects:context.turingWorld.runtimeObjects.length,
    runtimeParts:context.turingWorld.runtimeParts.length,objectRows:model.world.identity_specialization.objects.length,
    partRows:model.world.identity_specialization.semantic_parts.length,order:model.world.object_order.length,
    structuralOrder:model.world.structural_object_order.length});
  const beforeCapacity=counts();
  for(const sparseLength of [0xfffffffe,0xffffffff]) {
    documentBoxes.length=sparseLength;
    let failure;
    try {context.admitMechanicalCreatureStudioParts(capacityPrepared.definition,capacityUpdates);}
    catch(error){failure=error;}
    finally{documentBoxes.length=normalLength;}
    assert.deepEqual(counts(),beforeCapacity,'document capacity refusal must precede canonical registry writes');
    assert.deepEqual(Object.keys(documentBoxes),documentKeys,'sparse capacity refusal must not add document keys');
    assert.match(failure?.message??'',/document geometry capacity/);
    assert.equal(context.turingWorld.object(initial.identity+'/part:shelf/new-first'),null);
    assert.equal(context.turingWorld.object(initial.identity+'/part:shelf/rejected'),null);
    assert(capacityUpdates.filter(update=>!update.object).length===2,'rejected updates remain unadmitted');
  }
  assert.equal(faults.length,4);api.setSink(null);
}
console.log(`PASS: ${base}; native contacts/original launcher, exact worker FIFO/recycle barrier, registered toggles, canonical new-part admission ${typeof context.turingWorld.appendObjects==='function'?'with inactive late-failure retention and no-growth retry':'explicitly unavailable'}; no physics ticks or GPU completion claim`);
