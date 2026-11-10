// Bounded adapter contract test. Executes the ORIGINAL generated lifecycle
// functions; the worker and DOM are test doubles. This is not a physics or
// browser/rendering proof, and does not install another motion simulation.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createHash } from 'node:crypto';
import { adaptMechanicalCreatureProjectiles } from '../examples/meowdoku/web/mechanical_creature_instances.js';
import { readPoint, decodeMeta, designedMeshPlan } from '../examples/meowdoku/web/particle_styles.js';
import { catEyeMarble } from '../examples/meowdoku/web/marble_mesh.js';

const root = new URL('../', import.meta.url);
const playable = process.argv.includes('--playable');
const originalBytes = readFileSync(new URL(playable ? 'build/dot-playable-tires-20261110/index.html' :
  'build/dot-living-map-runtime-20261110/matched/behavior.js', root));
const original = originalBytes.toString('utf8').replaceAll('\r\n', '\n');
const canonical = readFileSync(new URL('src/compiler/abstract_ui_div_map.py', root), 'utf8').replaceAll('\r\n', '\n');
const expectedPlayableSha = 'aa6cb6fb897d24e6bf769ca2dbfde3a0d3d0dbcdd64ddb87fd30173bc0f0d5e2';
if (playable) assert.equal(createHash('sha256').update(originalBytes).digest('hex'), expectedPlayableSha);
const originalModel = JSON.parse(playable ? original.match(/<script type="application\/json" id="abstract-ui-model">([\s\S]*?)<\/script>/)[1] :
  readFileSync(new URL('build/dot-living-map-runtime-20261110/matched/model.json', root), 'utf8'));
const workerSource = readFileSync(new URL('build/dot-living-map-runtime-20261110/matched/physics-worker.js', root), 'utf8').replaceAll('\r\n', '\n');
function sourceFunction(text, name) {
  const start = text.indexOf(`function ${name}(`);
  assert(start >= 0, `${name} exists in the original`);
  const end = text.indexOf('\n}', start);
  assert(end > start, `${name} has a top-level closing brace`);
  return text.slice(start, end + 2);
}
const names = ['initializePhysicsSnapshotSlots', 'reservePhysicsSnapshotSlot', 'releasePhysicsSnapshotSlot',
  'registerProjectilePhysicsMembership', 'firePhysicsBall', 'updatePhysicsBalls', 'sleepProjectilePhysics',
  'wakeProjectilePhysics', 'expireProjectile', 'bounceProjectile', 'projectileContactNormal',
  'resolveProjectilePairs', 'resolveProjectilePlayerContacts', 'projectileAmmoItem', 'collectProjectilePickup'];
if (!playable) for (const name of names) assert.equal(sourceFunction(original, name), sourceFunction(canonical, name),
  `${name}: tested original generated function is still canonical`);
assert(!/quaternion|orientation/.test(sourceFunction(workerSource, 'publishSnapshot')),
  'the actual original snapshot writer has no projectile quaternion');

const sent = [], noOp = () => {}, node = () => ({ dataset: {}, style: { setProperty: noOp },
  setAttribute: noOp, append: noOp });
const context = vm.createContext({ model: structuredClone(originalModel),
  Map, Uint32Array, performance: { now: () => 1000 },
  document: { querySelector: () => null, getElementById: () => node() },
  div: node, makeSelectable: node, normalized3: a => { const n = Math.hypot(...a); return a.map(x => x / n); },
  spectrumColorHex: () => '#123456', ensureMusicRoomAudio: () => Promise.resolve(),
  rebuildPortableSceneMesh: noOp, refreshInventoryCounts: noOp, setPlacementStatus: noOp,
  sampleContactSurface: () => null, playToyImpact: noOp,
  runCompiledPhysicsState: () => { throw new Error('the worker snapshot path must be used'); },
  synchronizePortalBody: noOp, sent,
});
vm.runInContext(`
const projectileState = new Map(), projectilePickupState = new Map(), entityState = new Map(), index = new Map();
let projectileSequence = 0;
const physicsRuntime = {instance: true};
const hotbarState = {inventory:{items:[{quantity:100,maximum_stack:200,properties:{operation:'projectile-ammunition'}}]}};
const stateLoopRuntime = {ready: true, bodies: new Map(), slotRecords: [], latestSequence: 1, appliedSequence: 0,
  worker: {postMessage(message) {sent.push(message);}}};
const shaderViewer = {cameraFacing: [0,0,-1], cameraPosition: [2,5,7], geometry: [], colliders: [], revision: 0};
const vehicleRuntime = {active: null, state: null};
const viewportControls = {position: null, policy: {actor: model.identity}};
const portalRuntime = {lastPositions: new Map(), cooldowns: new Map(), transits: new Map()};
${names.map(name => sourceFunction(original, name)).join('\n')}
initializePhysicsSnapshotSlots(128);
globalThis.test = {projectileState,projectilePickupState,stateLoopRuntime,firePhysicsBall,updatePhysicsBalls,sleepProjectilePhysics,
  wakeProjectilePhysics,expireProjectile,collectProjectilePickup};
`, context);
const game = context.test;
assert.equal(context.model.projectiles.archetype.geometry.radius, .11);
assert.equal(game.firePhysicsBall(1, { ammunitionAuthority: 'vehicle-turret' }), true);
const [identity, projectile] = [...game.projectileState.entries()][0];
const workerBody = game.stateLoopRuntime.bodies.get(identity);
assert.equal(sent[0].type, 'upsert');
assert.equal(sent[0].body.radius, .11);
assert.equal(sent[0].body.identity, identity);
assert.equal(workerBody.slot, 0);
assert.equal(workerBody.generation, 1);
assert(!('quaternion' in projectile) && !('orientation' in projectile));
assert(!('quaternion' in workerBody) && !('orientation' in workerBody));
// Supply a worker-published position to the ORIGINAL update reader. The test
// supplies data, not a computed or alternative physical trajectory.
workerBody.position = [3.25, 7.5, -2.75];
workerBody.velocity = [1.25, -2.5, 3.75];
game.updatePhysicsBalls(.01);
assert.deepEqual(Array.from(projectile.position), workerBody.position);
assert.deepEqual(Array.from(projectile.velocity), workerBody.velocity);

// A real Studio generator, with its original integer face-material IDs.
// Tables are opaque fixtures: the adapter has no authority to reduce/rewrite
// their rows. No saved d4 is relabelled or admitted as a spherical source.
const geometry = catEyeMarble({glass:11,enamel:12,eye:13,accent:14,confetti:15,paintA:16,paintB:17},
  {rings:8,segments:12,strandCount:1});
const materials = {pbr: new Float32Array([.01,.02,.3,1.46]), phong: new Float32Array([.5,.8,17]),
  enamel: new Float32Array([700,2.6]), texstack: new Float32Array([1,2,3,4])};
const designedMesh = {center:[0,0,0], scale:1, count:geometry.triangleCount*3};
const source = {designedMesh,geometry,materials};
const base = {projectileState:game.projectileState,projectilePickupState:game.projectilePickupState,stateLoopRuntime:game.stateLoopRuntime,
  source,fixedQuaternionXYZW:[0,0,0,1]};
const before = JSON.stringify([...game.projectileState]);
const materialBefore = structuredClone(materials), geometryBefore = JSON.stringify(geometry);
const frame = adaptMechanicalCreatureProjectiles(base), instance = frame.instances[0];
assert.equal(frame.source, source);
assert.equal(frame.designedMesh, designedMesh);
assert.equal(frame.geometry, geometry);
assert.equal(frame.materials, materials);
assert.deepEqual(frame.identities, [identity]);
assert.equal(frame.snapshotSequence,1);
assert.deepEqual(instance.positionM, workerBody.position);
assert.deepEqual(instance.velocityMPerS, workerBody.velocity);
assert.equal(instance.radiusM, .11);
assert.deepEqual(instance.membership, {slot:0,generation:1});
assert.equal(instance.physicsQuaternionXYZW, null);
assert.equal(frame.quaternions, null);
assert.equal(frame.packed, null);
assert.equal(frame.colorMode, 'material');
assert.equal(instance.model[0], .11);
assert.equal(instance.model[5], .11);
assert.equal(instance.model[10], .11);
assert.deepEqual(Array.from(instance.model.slice(12,15)), workerBody.position);
assert.equal(designedMeshPlan({ready:true,designedMesh:frame.designedMesh,count:frame.n}).lod, 'full');
assert.match(frame.orientation.particleStyles, /slot-hash-not-authored-model/);
assert.equal(JSON.stringify([...game.projectileState]), before);
assert.equal(JSON.stringify(geometry), geometryBefore);
assert.deepEqual(materials, materialBefore);

const transformPoint = (m,p) => [0,1,2].map(r => m[r]*p[0]+m[4+r]*p[1]+m[8+r]*p[2]+m[12+r]);
const almost = (a,b,error=2e-14) => assert(Math.abs(a-b)<=error, `${a} != ${b}`);
// Caller-declared 2 metres per packed coordinate unit; no adapter default.
// The original game positions and authoritative radius remain unchanged.
const packing = {map:[.25,-.5,2],board:[16,24],planeHeight:-2};
const packedFrame = adaptMechanicalCreatureProjectiles({...base,packing});
const packed = packedFrame.packed, point = readPoint(packed.data,0), proof = packed.roundtrip[0];
assert.equal(packed.stride,20);
assert.equal(packed.worldMetresPerPackedUnit,2);
assert.equal(point.radius*2,.11); // exact radius roundtrip for the real radius
assert.equal(proof.radiusErrorM,0);
assert.equal(point.moving,true);
assert.equal(packed.data.length,5);
proof.positionM.forEach((v,i)=>almost(v,instance.positionM[i],2e-6));
proof.positionErrorM.forEach(v=>assert(Math.abs(v)<=2e-6));
assert.equal(decodeMeta(100+.11*1000).radius,.01); // actual direct-unit bug
// A different declared unit need not roundtrip exactly in floating point.
// The returned receipt exposes that error instead of claiming exactness.
const decimalScale = adaptMechanicalCreatureProjectiles({...base,
  packing:{...packing,map:[.25,-.5,10]}}).packed.roundtrip[0];
assert.equal(decimalScale.radiusM,.10999999999999999);
assert.equal(decimalScale.radiusErrorM,.10999999999999999-.11);
assert.throws(()=>adaptMechanicalCreatureProjectiles({...base,
  packing:{...packing,map:[0,0,1]}}),/cannot be represented/);
assert.throws(()=>adaptMechanicalCreatureProjectiles({...base,fixedQuaternionXYZW:undefined}),/fixedQuaternion/);
assert.throws(()=>adaptMechanicalCreatureProjectiles({...base,fixedQuaternionXYZW:[0,0,0,2]}),/unit length/);
assert.throws(()=>adaptMechanicalCreatureProjectiles({...base,
  packing:{...packing,map:[0,0,0]}}),/positive/);
projectile.quaternion = [0,0,0,1];
assert.throws(()=>adaptMechanicalCreatureProjectiles(base),/orientation contract changed/);
delete projectile.quaternion;

// Nontrivial authored orientation and normalization apply in original world
// xyz. They do not swap into Fountain's distinct law-coordinate basis.
const q = [0,Math.SQRT1_2,0,Math.SQRT1_2], center = [2,-1,3];
const rotated = adaptMechanicalCreatureProjectiles({...base,fixedQuaternionXYZW:q,
  source:{...source,designedMesh:{...designedMesh,center,scale:.5}}}).instances[0];
transformPoint(rotated.model,center).forEach((v,i)=>almost(v,instance.positionM[i]));
const pole = transformPoint(rotated.model,[4,-1,3]);
pole.forEach((v,i)=>almost(v,instance.positionM[i]+(i===2?-.11:0)));
assert.deepEqual(rotated.fixedQuaternionXYZW,q);
q[0]=99;
assert.equal(rotated.fixedQuaternionXYZW[0],0);

// The ORIGINAL sleep/wake and expiry functions own lifecycle and slot reuse.
assert.equal(game.sleepProjectilePhysics(projectile,'adapter-contract'),true);
const sleepingFrame = adaptMechanicalCreatureProjectiles({...base,packing});
assert.deepEqual(sleepingFrame.identities,[identity]);
assert.equal(sleepingFrame.instances[0].membership,null);
assert.equal(sleepingFrame.instances[0].status,'sleeping');
assert.equal(sleepingFrame.instances[0].radiusM,.11);
assert.equal(readPoint(sleepingFrame.packed.data,0).moving,false);
assert.equal(sleepingFrame.packed.roundtrip[0].radiusM,.11);
assert.equal(game.wakeProjectilePhysics(projectile,'adapter-contract',[.1,.2,.3]),true);
const awake = adaptMechanicalCreatureProjectiles(base);
assert.deepEqual(awake.identities,[identity]);
assert.deepEqual(awake.instances[0].membership,{slot:0,generation:2});
assert.deepEqual(awake.instances[0].velocityMPerS,[.1,.2,.3]);
const membership = game.stateLoopRuntime.bodies.get(identity);
game.stateLoopRuntime.slotRecords[membership.slot] = null;
assert.throws(()=>adaptMechanicalCreatureProjectiles(base),/membership is stale/);
game.stateLoopRuntime.slotRecords[membership.slot] = membership;
game.stateLoopRuntime.bodies.delete(identity);
assert.throws(()=>adaptMechanicalCreatureProjectiles(base),/no physics snapshot membership/);
game.stateLoopRuntime.bodies.set(identity,membership);
game.expireProjectile(projectile,'adapter-contract');
const pickup = adaptMechanicalCreatureProjectiles({...base,packing});
assert.equal(pickup.n,1);
assert.equal(pickup.identities[0],identity);
assert.equal(pickup.instances[0].stateSource,'projectilePickupState');
assert.equal(pickup.instances[0].status,'pickup');
assert.equal(pickup.instances[0].membership,null);
assert.equal(pickup.instances[0].pickup,true);
assert.deepEqual(pickup.instances[0].positionM,awake.instances[0].positionM);
assert.equal(readPoint(pickup.packed.data,0).moving,false);
game.projectileState.set(identity,projectile);
assert.throws(()=>adaptMechanicalCreatureProjectiles(base));
game.projectileState.delete(identity);
assert.equal(game.collectProjectilePickup(projectile),true);
assert.equal(adaptMechanicalCreatureProjectiles(base).n,0);
assert.equal(game.firePhysicsBall(1,{ammunitionAuthority:'vehicle-turret'}),true);
const next = adaptMechanicalCreatureProjectiles(base);
assert.notEqual(next.identities[0],identity);
assert.deepEqual(next.instances[0].membership,{slot:0,generation:3});
// Collection selection is an explicit visual tag, never a global replacement.
// The unchanged original fire function still creates every tested body.
assert.equal(adaptMechanicalCreatureProjectiles({...base,appearanceAssetIdentity:'studio:chosen'}).n,0);
const tagged = game.projectileState.get(next.identities[0]);
tagged.appearanceAssetIdentity = 'studio:chosen';
tagged.ammunitionItemIdentity = 'collection:item-7';
assert.equal(game.firePhysicsBall(1,{ammunitionAuthority:'vehicle-turret'}),true);
assert.equal(adaptMechanicalCreatureProjectiles(base).n,2);
const selected = adaptMechanicalCreatureProjectiles({...base,appearanceAssetIdentity:'studio:chosen'});
assert.equal(selected.n,1);
assert.equal(selected.identities[0],tagged.identity);
assert.equal(selected.instances[0].appearanceAssetIdentity,'studio:chosen');
assert.equal(selected.instances[0].ammunitionItemIdentity,'collection:item-7');
assert.equal(adaptMechanicalCreatureProjectiles({...base,appearanceAssetIdentity:'studio:other'}).n,0);
assert.throws(()=>adaptMechanicalCreatureProjectiles({...base,appearanceAssetIdentity:''}),/explicit nonempty/);
game.expireProjectile(tagged,'selected-appearance-retention');
const selectedPickup = adaptMechanicalCreatureProjectiles({...base,appearanceAssetIdentity:'studio:chosen'});
assert.equal(selectedPickup.n,1);
assert.equal(selectedPickup.instances[0].pickup,true);
assert.equal(selectedPickup.instances[0].ammunitionItemIdentity,'collection:item-7');
assert.equal(game.collectProjectilePickup(tagged),true);
assert.equal(adaptMechanicalCreatureProjectiles({...base,appearanceAssetIdentity:'studio:chosen'}).n,0);
assert.equal(adaptMechanicalCreatureProjectiles(base).n,1);
assert.equal(JSON.stringify(geometry),geometryBefore);
assert.deepEqual(materials,materialBefore);
let capturedStudioTriangleCount = null;
if (process.argv.includes('--captured-studio')) {
  const captured = JSON.parse(readFileSync(new URL('build/dot-studio-game-transfer-20261110/captured-studio-marble.json',root),'utf8'));
  assert.equal(captured.scene.authoring.object.design['base-shape'],'marble',
    'this captured source is authored spherical geometry, not the saved porcelain d4');
  const faces = captured.scene.subject.faces, capturedMaterials = captured.render.materials;
  const capturedSource = {geometry:faces,materials:capturedMaterials,
    designedMesh:{center:[0,0,0],scale:1,count:faces.length*3}};
  const capturedFrame = adaptMechanicalCreatureProjectiles({...base,source:capturedSource});
  assert.equal(capturedFrame.geometry,faces);
  assert.equal(capturedFrame.materials,capturedMaterials);
  assert.equal(capturedFrame.designedMesh,capturedSource.designedMesh);
  assert.equal(capturedFrame.instances[0].radiusM,.11);
  // The captured mesh coordinates are already the authored unit sphere.
  // Its Studio placement matrix and .3048m room unit must not be applied a
  // second time when attaching that mesh to the game's .11m rigid body.
  let maxSourceRadius = 0;
  const model = capturedFrame.instances[0].model, position = capturedFrame.instances[0].positionM;
  for (const face of faces) {
    assert(Number.isInteger(face[3]),'original integer material IDs retained');
    for (const vertex of face.slice(0,3)) {
      maxSourceRadius = Math.max(maxSourceRadius,Math.hypot(...vertex));
      transformPoint(model,vertex).forEach((v,axis)=>assert.equal(v,position[axis]+vertex[axis]*.11));
    }
  }
  almost(maxSourceRadius,1,4*Number.EPSILON);
  capturedStudioTriangleCount = faces.length;
  assert.equal(capturedStudioTriangleCount,72960);
}
console.log(JSON.stringify({passed:true,originalGeneratedLifecycleFunctions:names.length,
  source:playable?'published-playable':'matched-generated',sourceSha256:createHash('sha256').update(originalBytes).digest('hex'),
  projectileRadiusM:instance.radiusM,packedRadiusErrorM:proof.radiusErrorM,
  maximumPackedPositionErrorM:Math.max(...proof.positionErrorM.map(Math.abs)),
  originalPhysicsModified:false,physicsQuaternionPublished:false,
  sourceGeometryAndMaterialsSameReferences:true,capturedStudioTriangleCount,browserRenderingTested:false}));
