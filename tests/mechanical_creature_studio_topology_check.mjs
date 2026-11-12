import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import vm from "node:vm";

const root = new URL("../", import.meta.url);
const initial = JSON.parse(readFileSync(new URL("build/dot-studio-game-transfer-20261110/studio-building-definition-v2.json", root), "utf8"));
const page = readFileSync(new URL("build/meowdoku/web/mechanical_creature_play.html", root), "utf8");
const model = JSON.parse(page.match(/<script type="application\/json" id="abstract-ui-model">([\s\S]*?)<\/script>/)[1]);
const exact = (name, kind = "function") => {
  const found = page.match(new RegExp(`^${kind} ${name}[^]*?^}`, "m"));
  assert.ok(found, `original ${name}`); return found[0];
};
const posts = [];
const context = vm.createContext({structuredClone, Set, Map, Float32Array, model,
  stateLoopRuntime: {ready: true, worker: {postMessage: value => posts.push(value)}}});
vm.runInContext(readFileSync(new URL("examples/meowdoku/web/mechanical_creature_studio_building.js", root), "utf8"), context);
context.appendMechanicalCreatureStudioBuilding(model, initial);
vm.runInContext(exact("TuringWorldRegistry", "class") + "\nglobalThis.turingWorld=new TuringWorldRegistry(model.world);", context);
vm.runInContext(["colorVector", "boxYawDegrees", "buildExtrudedBoxMesh", "rebuildPortableSceneMesh"]
  .map(name => exact(name)).join("\n"), context);
const registered = model.document_geometry.boxes.slice(-initial.boxes.length);
const retained = new Map(registered.map(box => [box.identity, box]));
const selectedIdentity = initial.boxes[1].identity;
const selectedBefore = JSON.stringify({boxes: initial.boxes, objects: model.world.objects.slice(-initial.boxes.length)});
const selectedObjects = context.createMechanicalCreatureStudioWorldObjects(model, initial, [selectedIdentity]);
assert.equal(selectedObjects.length, 1);
assert.equal(JSON.stringify(selectedObjects[0]), JSON.stringify(model.world.objects.find(object => object.identity === selectedIdentity)),
  "runtime selection uses exactly the startup descriptor authoring path");
assert.equal(JSON.stringify({boxes: initial.boxes, objects: model.world.objects.slice(-initial.boxes.length)}), selectedBefore);
assert.notEqual(selectedObjects[0].extensions["abstract_ui.document_geometry"], retained.get(selectedIdentity));
assert.throws(() => context.createMechanicalCreatureStudioWorldObjects(model, initial, ["unknown"]), /known unique/);
assert.throws(() => context.createMechanicalCreatureStudioWorldObjects(model, initial, [selectedIdentity, selectedIdentity]), /known unique/);
const unrelated = model.document_geometry.boxes.find(box => box.label === "Outer yard north flat apron");
const projectile = {identity: "existing-projectile:test", kind: "physics-ball", geometry_mode: "sphere",
  center: [2, 3], center_y: 1, half_extent: [.11, .11], radius: .11, height: .22,
  palette_role: "building-face", placement: {custody: "placed"}};
const projectileState = new Map([[projectile.identity, {box: projectile, position: [2, 1, 3], velocity: [0, 0, 0]}]]);
const playerBefore = JSON.stringify(model.entity_mezzanine);
const projectileBefore = JSON.stringify([...projectileState]);
const registryBefore = JSON.stringify(model.world.identity_specialization);
const worldRefs = [...model.world.objects], runtimeObjects = [...context.turingWorld.runtimeObjects];
context.shaderViewer = {geometry: [unrelated, ...registered, projectile], revision: 1};
Object.assign(context, {vehiclePresentationGeometry: () => [], buildNavigationRouteMesh: () => new Float32Array(),
  rebuildPortalGraph() {}, buildPortalSplatMesh: () => new Float32Array(), buildPortalTubeMesh: () => new Float32Array(),
  installSceneMesh(mesh) {context.shaderViewer.mesh = mesh;}, installVehiclePresentationMesh() {}});
context.rebuildPortableSceneMesh();
const studioContacts = () => context.shaderViewer.colliders.filter(row => retained.has(row.objectIdentity));
assert.equal(studioContacts().length, 15);

const change = (active, revision) => {
  const next = structuredClone(initial); next.sourceRevision = revision; next.assetIdentity = `studio:topology-${revision}`;
  next.boxes = next.boxes.filter(box => active.includes(box.identity));
  next.proxyIdentities = next.boxes.map(box => box.identity);
  for (const box of next.boxes) box.appearance.studio_asset_identity = next.assetIdentity;
  return next;
};
const apply = plan => {
  const active = new Set(plan.activeProxyIds), geometry = context.shaderViewer.geometry;
  for (const descriptor of plan.definition.boxes) Object.assign(retained.get(descriptor.identity), structuredClone(descriptor));
  // Exactly the existing active-geometry array operations, not a registry swap.
  for (let index = geometry.length - 1; index >= 0; index--)
    if (retained.has(geometry[index].identity) && !active.has(geometry[index].identity)) geometry.splice(index, 1);
  for (const box of registered) if (active.has(box.identity) && !geometry.includes(box)) geometry.push(box);
  context.rebuildPortableSceneMesh();
};
const initialText = JSON.stringify(initial), off = change([initial.identity], 10), offText = JSON.stringify(off);
const previous = change(initial.proxyIdentities, 9);
previous.boxes[1].center[0] += .25;
const disabled = context.reconcileMechanicalCreatureStudioBuilding(initial, off, previous);
assert.equal(JSON.stringify(initial), initialText); assert.equal(JSON.stringify(off), offText);
assert.equal(disabled.definition.boxes.length, 10); assert.equal(disabled.activeProxyIds.length, 1);
assert.equal(disabled.inactiveProxyIds.length, 9);
assert.ok(disabled.definition.boxes.slice(1).every(box => box.placement.custody === "placed"));
assert.equal(disabled.definition.boxes[1].center[0], previous.boxes[1].center[0], "disabled part retains last admitted position");
apply(disabled);
assert.equal(studioContacts().length, 6, "native table contacts disappear");
assert.equal(context.shaderViewer.geometry.length, 3);
assert.ok(!context.shaderViewer.identitySpans.some(span => span.identity.includes("/part:table/")));
assert.equal(posts.at(-1).colliders.filter(row => retained.has(row.objectIdentity)).length, 6);
const on = change(initial.proxyIdentities, 11);
const enabled = context.reconcileMechanicalCreatureStudioBuilding(initial, on, disabled.definition);
apply(enabled);
assert.equal(studioContacts().length, 15, "native table contacts return");
assert.equal(enabled.activeProxyIds.length, 10); assert.equal(enabled.inactiveProxyIds.length, 0);
registered.forEach(box => assert.equal(context.shaderViewer.geometry.find(row => row.identity === box.identity), box));
for (const span of context.shaderViewer.identitySpans)
  assert.equal(context.shaderViewer.geometry[span.boxIndex].identity, span.identity, "native active geometry span indices stay aligned");
assert.equal(context.shaderViewer.geometry[0], unrelated); assert.ok(context.shaderViewer.geometry.includes(projectile));
assert.equal(JSON.stringify(model.entity_mezzanine), playerBefore);
assert.equal(JSON.stringify([...projectileState]), projectileBefore);
assert.equal(JSON.stringify(model.world.identity_specialization), registryBefore);
worldRefs.forEach((object, index) => assert.equal(model.world.objects[index], object));
runtimeObjects.forEach((object, index) => assert.equal(context.turingWorld.runtimeObjects[index], object));

const unknown = structuredClone(on); unknown.boxes[1].identity += ":new"; unknown.proxyIdentities[1] += ":new";
assert.throws(() => context.reconcileMechanicalCreatureStudioBuilding(initial, unknown), /not registered/);
const noBuilding = change(initial.proxyIdentities.slice(1), 12);
assert.throws(() => context.reconcileMechanicalCreatureStudioBuilding(initial, noBuilding), /building must remain/);
const wrongOwner = structuredClone(on); wrongOwner.boxes[1].parent_identity = "unrelated";
assert.throws(() => context.reconcileMechanicalCreatureStudioBuilding(initial, wrongOwner), /ownership/);
const duplicate = structuredClone(on); duplicate.proxyIdentities[1] = duplicate.proxyIdentities[0];
assert.throws(() => context.reconcileMechanicalCreatureStudioBuilding(initial, duplicate), /exactly once/);
const wrongPlacement = structuredClone(on); wrongPlacement.placement[12] += 1;
assert.throws(() => context.reconcileMechanicalCreatureStudioBuilding(initial, wrongPlacement), /placement/);
assert.throws(() => context.reconcileMechanicalCreatureStudioBuilding(initial, on, off), /complete registered/);
console.log("PASS: registered table disable/re-enable, native contacts/picking spans, preserved identities/object refs/player/projectiles; unknown topology rejected");
