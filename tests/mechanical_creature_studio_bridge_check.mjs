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
console.log(`PASS: ${base}; native 15-contact updates, exact registry, cancellation, commit rollback, copied double-slot state, authoritative worker publication, actual original launcher and edited environment -> Studio return consumer`);
