import assert from "node:assert/strict";
import {readFileSync, writeFileSync, existsSync} from "node:fs";
import vm from "node:vm";

const root = new URL("../", import.meta.url);
const helper = readFileSync(new URL("examples/meowdoku/web/mechanical_creature_studio_building.js", root), "utf8");
const page = readFileSync(new URL("build/meowdoku/web/mechanical_creature_play.html", root), "utf8");
const captured = JSON.parse(readFileSync(new URL("build/dot-studio-game-transfer-20261110/captured-studio-marble.json", root), "utf8"));
const model = JSON.parse(page.match(/<script type="application\/json" id="abstract-ui-model">([\s\S]*?)<\/script>/)[1]);
const originalBoxes = [...model.document_geometry.boxes], originalObjects = [...model.world.objects];
const originalRegions = JSON.stringify(model.regions), originalSpecialization = structuredClone(model.world.identity_specialization);
const originalPlayer = JSON.stringify(model.entity_mezzanine), originalRuntime = JSON.stringify(model.world.wasm_modules);
const context = vm.createContext({structuredClone, model, Float32Array, Set, Map, console});
vm.runInContext(helper, context);
const beforeCapture = JSON.stringify(captured);
const definition = context.prepareMechanicalCreatureStudioBuilding(captured);
assert.equal(JSON.stringify(captured), beforeCapture, "capture remains untouched");
assert.equal(definition.boxes.length, 10);
assert.equal(definition.contacts.cuboids.length, 9);
assert.equal(definition.contacts.decorativeParts.length, 22);
assert.equal(definition.entrance.widthM, 2); assert.equal(definition.entrance.heightM, 2.4);
assert.equal(definition.boxes[0].openings[0].side, "north");
assert.ok(definition.boundsM.minimum[0] > 24);
const apron = originalBoxes.find(box => box.label === "Outer yard north flat apron");
for (const [axis, worldAxis] of [[0, 0], [1, 2]]) {
  assert.ok(definition.boundsM.minimum[worldAxis] > apron.center[axis] - apron.half_extent[axis]);
  assert.ok(definition.boundsM.maximum[worldAxis] < apron.center[axis] + apron.half_extent[axis]);
}
assert.ok(definition.boundsM.minimum[0] > 10.477673078 + 10, "clear of original player/car spawn");
context.appendMechanicalCreatureStudioBuilding(model, definition);
assert.equal(model.world.objects.length, originalObjects.length + 10);
assert.equal(model.document_geometry.boxes.length, originalBoxes.length + 10);
assert.equal(model.regions.length, JSON.parse(originalRegions).length + 1);
assert.equal(JSON.stringify(model.regions.slice(0, -1)), originalRegions);
assert.equal(JSON.stringify(model.entity_mezzanine), originalPlayer);
assert.equal(JSON.stringify(model.world.wasm_modules), originalRuntime);
originalBoxes.forEach((box, index) => assert.equal(model.document_geometry.boxes[index], box));
originalObjects.forEach((object, index) => assert.equal(model.world.objects[index], object));
assert.deepEqual(model.world.identity_specialization.objects.slice(0, originalSpecialization.objects.length), originalSpecialization.objects);
assert.deepEqual(model.world.identity_specialization.semantic_parts.slice(0, originalSpecialization.semantic_parts.length), originalSpecialization.semantic_parts);
context.appendMechanicalCreatureStudioBuilding(model, definition);
assert.equal(model.world.objects.length, originalObjects.length + 10, "idempotent startup");

function exact(name, kind = "function") {
  const found = page.match(new RegExp(`^${kind} ${name}[^]*?^}`, "m"));
  assert.ok(found, `original ${name} is present`); return found[0];
}
vm.runInContext(exact("TuringWorldRegistry", "class") + "\nglobalThis.turingWorld = new TuringWorldRegistry(model.world);", context);
vm.runInContext(["colorVector", "boxYawDegrees", "normalized3", "cross3", "buildExtrudedBoxMesh", "raySceneTriangle",
  "playerColliderVerticalOverlap", "resolvePlayerHorizontalMotion"].map(name => exact(name)).join("\n"), context);
const boxes = model.document_geometry.boxes.slice(-10);
context.shaderViewer = {revision: 0, geometry: boxes};
const realization = context.buildExtrudedBoxMesh(boxes, model.appearance.colors);
assert.equal(realization.spans.length, 10);
assert.equal(realization.colliders.length, 15, "existing room walls plus nine original cuboids");
assert.ok(realization.spans.every(span => span.runtimeObjectId > 0));
assert.ok(realization.semanticPartSpans.every(span => span.runtimePartId > 0));
assert.ok(realization.colliders.every(collider => collider.runtimePartId > 0));
assert.ok(boxes.every(box => box.appearance.studio_asset_identity === definition.assetIdentity));
Object.assign(context.shaderViewer, {baseMesh: realization.mesh, identitySpans: realization.spans,
  semanticPartSpans: realization.semanticPartSpans, colliders: realization.colliders});
const doorHit = context.raySceneTriangle([32, .4, -9], [0, 0, -1]);
assert.ok(doorHit.distance > 7, "native front opening does not pick an imaginary wall");
assert.ok(doorHit.objectIdentity.includes("table/"));
const wallHit = context.raySceneTriangle([34, .4, -9], [0, 0, -1]);
assert.ok(wallHit.distance < 2);
assert.equal(wallHit.objectIdentity, definition.identity);
context.viewportControls = {colliderSides: new Map()};
context.physicsRuntime = {parameters: new Map()};
const entryZ = definition.entrance.worldCenter[2];
const enter = context.resolvePlayerHorizontalMotion([32, .4, entryZ + .3], [32, .4, entryZ - .3]);
assert.equal(enter.blockedZ, false, "original player can cross entrance");
const wall = context.resolvePlayerHorizontalMotion([34, .4, entryZ + .3], [34, .4, entryZ - .3]);
assert.equal(wall.blockedZ, true, "original player is blocked by native wall");

const altered = structuredClone(captured);
altered.scene.parts.find(part => part.construction === "cuboid").faces[0][0][0] += .1;
assert.throws(() => context.prepareMechanicalCreatureStudioBuilding(altered), /axis-aligned/);
assert.throws(() => context.prepareMechanicalCreatureStudioBuilding(captured, {translationM: [1, NaN, 3]}), /finite/);
const collisionModel = structuredClone(model);
collisionModel.world.objects.find(object => object.identity === definition.identity).extensions = {};
assert.throws(() => context.appendMechanicalCreatureStudioBuilding(collisionModel, definition), /collision/);
if (process.argv[2]) {
  assert.equal(existsSync(process.argv[2]), false, "refusing to overwrite a definition");
  writeFileSync(process.argv[2], JSON.stringify(definition, null, 2) + "\n");
}
console.log(`PASS: 10 native Studio objects, ${realization.colliders.length} native contacts, runtime identities, original picking/player entry, map discovery, untouched capture/player/worker`);
