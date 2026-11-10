import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import vm from "node:vm";

// Read the actual authored literal without importing src.compiler: another
// worker can be updating its native DLL while this source-only gate runs.
const text = readFileSync(new URL("../src/compiler/javascript_runtime_utilities.py", import.meta.url), "utf8");
const source = text.split('WORLD_REGISTRY_SOURCE = r"""')[1]?.split('""".strip()')[0];
assert.ok(source?.includes("class TuringWorldRegistry"));
const context = vm.createContext({structuredClone});
vm.runInContext(source + "\nglobalThis.Registry=TuringWorldRegistry;", context);
const object = (identity, parent = "world", partIds = [`${identity}/surface`]) => ({identity, parent,
  semantic_parts: partIds.map(identity => ({identity, role: "body"})),
  transform: {position: [1, 2, 3]}, extensions: {"abstract_ui.document_geometry": {identity, center: [1, 3]}}});
const make = () => {
  const root = object("building"), oldPart = root.semantic_parts[0];
  const world = {identity: "world", root: "world", objects: [root], object_order: [root.identity],
    structural_object_order: [root.identity], observation_object_order: ["unrelated-observation"],
    identity_specialization: {authority: "authored-string-identity", missing_runtime_id: 0,
      objects: [{identity: root.identity, runtime_id: 4}],
      semantic_parts: [{identity: oldPart.identity, runtime_id: 7, object_runtime_id: 4}]}};
  return {world, registry: new context.Registry(world), root};
};
const snapshot = ({world, registry}) => JSON.stringify({world,
  objects: [...registry.objects], parts: [...registry.semanticParts], children: [...registry.children],
  objectIds: [...registry.objectRuntimeIds], partIds: [...registry.partRuntimeIds],
  runtimeObjects: registry.runtimeObjects, runtimeParts: registry.runtimeParts});
const subject = make(), {world, registry, root} = subject;
const arrays = [world.objects, world.object_order, world.structural_object_order,
  world.identity_specialization.objects, world.identity_specialization.semantic_parts,
  registry.runtimeObjects, registry.runtimeParts];
const maps = [registry.objects, registry.children, registry.semanticParts, registry.objectRuntimeIds, registry.partRuntimeIds];
const oldPart = registry.part("building/surface"), oldChildren = registry.children.get("world");
const rows = [object("table", "building"), object("leg", "table", ["leg/left", "leg/right"])];
const callerBefore = JSON.stringify(rows), receipt = registry.appendObjects(rows, {structural: true});
assert.equal(JSON.stringify(rows), callerBefore);
assert.equal(receipt.objects[0].runtime_id, 5); assert.equal(receipt.objects[1].runtime_id, 6);
assert.equal(receipt.semantic_parts[0].runtime_id, 8); assert.equal(receipt.semantic_parts[2].runtime_id, 10);
assert.equal(receipt.semantic_parts[2].object_runtime_id, 6);
assert.ok(Object.isFrozen(receipt)); assert.ok(Object.isFrozen(receipt.objects[0]));
assert.equal(registry.object("building"), root); assert.equal(registry.part("building/surface"), oldPart);
assert.equal(registry.children.get("world"), oldChildren);
[world.objects, world.object_order, world.structural_object_order,
  world.identity_specialization.objects, world.identity_specialization.semantic_parts,
  registry.runtimeObjects, registry.runtimeParts].forEach((array, index) => assert.equal(array, arrays[index]));
[registry.objects, registry.children, registry.semanticParts, registry.objectRuntimeIds, registry.partRuntimeIds]
  .forEach((map, index) => assert.equal(map, maps[index]));
assert.notEqual(registry.object("table"), rows[0], "admitted objects own detached records");
assert.equal(world.objects[1], registry.object("table"));
assert.equal(registry.objectFromRuntimeId(5), registry.object("table"));
assert.equal(registry.partFromRuntimeId(8), registry.part("table/surface"));
assert.deepEqual(Array.from(registry.ancestry("leg"), row => row.identity), ["leg", "table", "building"]);
assert.deepEqual(Array.from(registry.containedBy("table"), row => row.identity), ["leg"]);
rows[0].identity = "stale-caller-name"; rows[0].transform.position[0] = 999;
rows[0].semantic_parts[0].role = "stale-caller-role";
assert.equal(registry.object("table").transform.position[0], 1);
assert.equal(registry.part("table/surface").role, "body");
assert.equal(registry.object("stale-caller-name"), null);
assert.equal(world.observation_object_order[0], "unrelated-observation");
const later = registry.appendObjects([object("later", "building")], {structural: false});
assert.equal(later.objects[0].runtime_id, 7); assert.equal(later.semantic_parts[0].runtime_id, 11);
assert.ok(world.object_order.includes("later")); assert.ok(!world.structural_object_order.includes("later"));
const beforeEmpty = snapshot(subject); assert.equal(registry.appendObjects([]).objects.length, 0);
assert.equal(snapshot(subject), beforeEmpty);

let rejected = 0;
const rejects = (batch, expression, customize) => {
  const value = make(); customize?.(value); const before = snapshot(value);
  assert.throws(() => value.registry.appendObjects(batch), expression);
  assert.equal(snapshot(value), before, "rejection must preserve every registry/world entry"); rejected++;
};
rejects([object("valid"), object("building")], /duplicate/);
rejects([object("same"), object("same")], /duplicate/);
rejects([object("valid"), object("other", "world", ["building/surface"])], /duplicate/);
rejects([object("valid", "world", ["part"]), object("other", "world", ["part"])], /duplicate/);
rejects([object("valid"), object("orphan", "absent")], /unknown/);
rejects([object("one", "two"), object("two", "one")], /cycle/);
rejects([object("self", "self")], /cycle/);
const wrongOwner = object("wrong-owner"); wrongOwner.semantic_parts[0].object = "building";
rejects([object("valid"), wrongOwner], /owner/);
const missingParts = object("missing"); delete missingParts.semantic_parts;
rejects([missingParts], /semantic parts/);
rejects([object("")], /empty/);
rejects([object("valid")], /mutable/, ({world}) => Object.freeze(world.objects));
rejects([object("valid")], /writable/, ({world}) => Object.defineProperty(world, "object_order", {writable: false}));
rejects([object("valid")], /mutable/, ({registry}) => Object.freeze(registry.children.get("world")));
rejects([object("valid")], /duplicate/, ({world}) => world.objects.push(object("valid")));
rejects([object("valid", "world", ["live-dynamic/surface"])], /duplicate/,
  ({world}) => world.objects.push(object("live-dynamic")));
rejects([object("valid")], /specialization/, ({world}) => world.identity_specialization.authority = "not-authored");
const uncloneable = object("bad-clone"); uncloneable.callback = () => {};
rejects([object("valid"), uncloneable], /clone|cloned/i);
rejects([object("world", "world")], /duplicate/);
rejects([object("implicit-root", "implicit-root")], /duplicate/,
  ({world}) => {world.root = "implicit-root";});

// Maximum JavaScript array length is 2**32-1. These sparse arrays have only
// the existing few own properties: never serialize or iterate their lengths.
const boundedSnapshot = ({world, registry}) => JSON.stringify({world,
  objects: [...registry.objects], parts: [...registry.semanticParts], children: [...registry.children],
  objectIds: [...registry.objectRuntimeIds], partIds: [...registry.partRuntimeIds],
  runtimeObjectLength: registry.runtimeObjects.length, runtimePartLength: registry.runtimeParts.length,
  runtimeObjectKeys: Object.keys(registry.runtimeObjects), runtimePartKeys: Object.keys(registry.runtimeParts)});
const capacityCheck = (Registry, kind) => {
  const world = make().world, registry = new Registry(world), value = {world, registry};
  registry[kind].length = 0xffffffff;
  const before = boundedSnapshot(value); let failure;
  try { registry.appendObjects([object("at-capacity", "building")]); } catch (error) { failure = error; }
  return {guarded: /identity space is exhausted/.test(failure?.message || ""),
    unchanged: boundedSnapshot(value) === before, error: failure?.message};
};
for (const kind of ["runtimeObjects", "runtimeParts"]) {
  const result = capacityCheck(context.Registry, kind);
  assert.equal(result.guarded, true, `${kind}: exact 2**32 sum must be rejected before mutation`);
  assert.equal(result.unchanged, true, `${kind}: exhausted admission must preserve all state`);
  rejected++;
}

// Restore only the two reviewed old guards in memory. The permanent test must
// detect both regressions, without editing the canonical source or big dumps.
const oldGuards = source
  .replace(" || object.identity === world.identity || object.identity === world.root", "")
  .replace("firstObjectId + admitted.length >= 0x100000000", "firstObjectId + admitted.length > 0x100000000")
  .replace("firstPartId + newParts.size >= 0x100000000", "firstPartId + newParts.size > 0x100000000");
assert.notEqual(oldGuards, source, "reviewed guard negative control must differ");
const oldContext = vm.createContext({structuredClone});
vm.runInContext(oldGuards + "\nglobalThis.Registry=TuringWorldRegistry;", oldContext);
for (const identity of ["world", "implicit-root"]) {
  const oldWorld = make().world; oldWorld.root = "implicit-root";
  const oldRegistry = new oldContext.Registry(oldWorld);
  oldRegistry.appendObjects([object(identity, identity)]);
  assert.ok(oldRegistry.object(identity), "old root guard wrongly admits the implicit root as its own child");
}
for (const kind of ["runtimeObjects", "runtimeParts"]) {
  const result = capacityCheck(oldContext.Registry, kind);
  assert.equal(result.guarded, false, "old exact-capacity guard falls through");
  assert.equal(result.unchanged, false, "old exact-capacity push leaves partial admission state");
}
const forward = make(); forward.registry.appendObjects([object("child", "parent"), object("parent", "building")]);
assert.deepEqual(Array.from(forward.registry.ancestry("child"), row => row.identity), ["child", "parent", "building"]);
const minimalWorld = {identity: "minimal", objects: [object("existing", "minimal", [])]};
const minimal = new context.Registry(minimalWorld);
minimal.appendObjects([object("new", "existing")]);
assert.equal(minimalWorld.identity_specialization.objects[0].runtime_id, 1);
assert.equal(minimal.objectRuntimeId("new"), 2); assert.equal(minimal.partRuntimeId("new/surface"), 1);
assert.deepEqual(Array.from(minimalWorld.object_order), ["existing", "new"]);
console.log(`PASS: canonical registry append, stable refs/IDs, detached inputs, atomic ${rejected} rejections, forward parents and retained inactive IDs`);
