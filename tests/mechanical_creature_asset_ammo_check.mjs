import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import vm from "node:vm";

// Execute the actual patched launcher, tool selection and pickup/vacuum bodies.
// DOM/audio/worker transport are test doubles, not replacement game logic.
const path = process.argv[2];
if (!path) throw new Error("pass the generated candidate HTML path");
const html = readFileSync(path, "utf8").replaceAll("\r\n", "\n");
const modelMarker = html.indexOf('id="abstract-ui-model"');
const modelStart = html.indexOf(">", modelMarker) + 1;
const model = JSON.parse(html.slice(modelStart, html.indexOf("</script>", modelStart)));
const originalLaws = JSON.stringify(model.projectiles.archetype);
const fn = name => {
  const result = html.match(new RegExp(`^function ${name}\\([\\s\\S]*?^\\}`, "m"));
  assert.ok(result, name); return result[0];
};
class Element {
  constructor(tag = "div") {this.tagName = tag; this.children = []; this.dataset = {}; this.attributes = {};
    this.style = {setProperty() {}}; this.listeners = {}; this.classList = {toggle() {}};}
  append(...children) {this.children.push(...children);}
  setAttribute(name, value) {this.attributes[name] = value;}
  addEventListener(name, callback) {this.listeners[name] = callback;}
  querySelector() {return null;}
  querySelectorAll() {return [];}
  remove() {this.removed = true;}
  focus() {this.focused = true;}
}
const root = new Element(), layer = new Element(), calls = [];
const stateLoopRuntime = {ready: true, bodies: new Map(), worker: {postMessage(value) {calls.push(value);}}};
const context = vm.createContext({Set, model, performance: {now: () => 1000},
  hotbarState: {inventory: model.inventory, model: model.hotbar, activeSlot: model.hotbar.active_slot},
  shaderViewer: {element: root, cameraPosition: [0, 1.5, 0], cameraFacing: [0, 0, -1], geometry: [], revision: 1},
  viewportControls: {policy: {actor: model.inventory.owner}}, vehicleRuntime: {active: null},
  physicsRuntime: {instance: {}}, projectileState: new Map(), projectilePickupState: new Map(),
  entityState: new Map(), index: new Map(), projectileSequence: 0, stateLoopRuntime,
  portalRuntime: {lastPositions: new Map(), cooldowns: new Map(), transits: new Map()},
  primaryActionState: {down: false}, secondaryActionState: {down: false},
  placementState: {}, actionEdges: {register: () => "edge"}, abstractUISystemTimer: {issue(value) {calls.push(value);}},
  document: {createElement: tag => new Element(tag), querySelector: () => null,
    getElementById: id => id === "entity-layer" ? layer : null},
  div: (className, textContent) => Object.assign(new Element(), {className, textContent}),
  ensureMusicRoomAudio: () => Promise.resolve(), spectrumColorHex: () => "#ff7755",
  registerProjectilePhysicsMembership: state => calls.push({type: "upsert-test", identity: state.identity}),
  releasePhysicsSnapshotSlot: identity => calls.push({type: "release-test", identity}),
  rebuildPortableSceneMesh() {}, refreshInventoryCounts() {}, setPlacementStatus() {}, refreshToolModeControl() {},
});
vm.runInContext(readFileSync(new URL("../examples/meowdoku/web/mechanical_creature_asset_ammo.js", import.meta.url), "utf8"), context);
vm.runInContext(["projectileAmmoItem", "selectHotbarSlot", "normalized3", "firePhysicsBall", "expireProjectile",
  "collectProjectilePickup", "absorbProjectileIntoAmmo"].map(fn).join("\n"), context);
const api = context.mechanicalCreatureAmmo, original = api.selected(), originalCount = original.quantity;
const a = api.addAsset({assetIdentity: "studio/collection/a", name: "Marble A", iconUrl: "data:image/png;base64,AAAA", quantity: 2, maximumStack: 2});
const b = api.addAsset({assetIdentity: "studio/collection/b", name: "Marble B", iconUrl: "data:image/png;base64,BBBB", quantity: 1, maximumStack: 3});
assert.equal(api.selected(), original, "adding assets does not select them");
assert.equal(context.hotbarState.model.slots.length, 10, "no extra hotbar");
const fire = () => {assert.equal(context.firePhysicsBall(), true); return [...context.projectileState.values()].at(-1);};
const ordinaryShot = fire();
assert.equal(ordinaryShot.appearanceAssetIdentity, undefined, "default physics balls remain original");
assert.equal(original.quantity, originalCount - 1);
api.select(a.identity);
assert.equal(context.hotbarState.activeSlot, 8);
assert.equal(context.hotbarState.inventory.active_tool.entity, context.mechanicalCreatureBallLauncher().entity);
const shotA = fire();
assert.equal(a.quantity, 1); assert.equal(shotA.appearanceAssetIdentity, "studio/collection/a");
assert.equal(shotA.ammunitionItemIdentity, a.identity);
assert.equal(shotA.record.appearance_asset_identity, shotA.appearanceAssetIdentity);
assert.equal(shotA.box.appearance.asset_identity, shotA.appearanceAssetIdentity);
assert.equal(shotA.entity.traits.appearance_asset_identity, shotA.appearanceAssetIdentity);
assert.equal(shotA.entity.texture.reference, shotA.appearanceAssetIdentity);
assert.equal(shotA.box.radius, model.projectiles.archetype.geometry.radius);
assert.deepEqual([...shotA.velocity], [0, 0, -model.projectiles.archetype.launch_speed]);
api.addAsset({assetIdentity: "studio/collection/a", name: "Refreshed Marble A", iconUrl: "data:image/png;base64,CCCC", quantity: 2, maximumStack: 2});
assert.equal(a.quantity, 1, "refreshing the texture never refills spent ammo");
api.select(b.identity); const shotB = fire(); assert.equal(b.quantity, 0);
api.select(original.identity);
context.expireProjectile(shotA, "test-pickup");
assert.equal(context.collectProjectilePickup(shotA), true);
assert.equal(a.quantity, 2); assert.equal(original.quantity, originalCount - 1);
api.select(a.identity);
context.expireProjectile(ordinaryShot, "test-pickup");
assert.equal(context.collectProjectilePickup(ordinaryShot), true);
assert.equal(original.quantity, originalCount); assert.equal(a.quantity, 2);
api.select(original.identity);
assert.equal(context.absorbProjectileIntoAmmo(shotB), true, "full selected default stack cannot block different source stack");
assert.equal(b.quantity, 1); assert.equal(original.quantity, originalCount);
assert.equal(context.projectileState.size, 0); assert.equal(context.projectilePickupState.size, 0);
assert.equal(JSON.stringify(model.projectiles.archetype), originalLaws, "entire physics/launch archetype remains identical");
const absent = {ammunitionItemIdentity: "removed-item"};
assert.equal(context.projectileAmmoItem(absent), null, "missing source stack cannot silently become another ammo type");

const slot = new Element(); context.appendMechanicalCreatureAmmoPicker(slot, original);
const [toggle, choices] = slot.children;
assert.equal(choices.children.length, 3); assert.equal(choices.hidden, true);
const event = {preventDefault() {}, stopPropagation() {}};
toggle.listeners.click(event); assert.equal(choices.hidden, false);
assert.equal(choices.children[1].children[0].tagName, "img");
assert.equal(choices.children[1].children[0].src, a.properties.appearance_icon);
choices.children[1].listeners.click(event); assert.equal(api.selected(), a);
assert.equal(context.hotbarState.inventory.active_tool.entity, context.mechanicalCreatureBallLauncher().entity);
console.log("PASS: real launcher tags selected assets; pickup/vac returns original stack; actual icon picker routes original tool; physics unchanged");
