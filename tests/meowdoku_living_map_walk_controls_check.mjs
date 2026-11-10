import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {
  createLivingMapWalkController, createLivingMapControlState,
  LIVING_MAP_VIEWPORT_BINDINGS, LIVING_MAP_VIEWPORT_POLICY_DEFAULTS,
  LIVING_MAP_WALK_SOURCE_FUNCTIONS, LIVING_MAP_WALK_REQUIRED_CALLBACKS,
  LIVING_MAP_WALK_REQUIRED_STATE,
} from "../examples/meowdoku/web/living_map_walk_controls.js";

const source = readFileSync(new URL("../src/compiler/abstract_ui_div_map.py", import.meta.url), "utf8").replaceAll("\r\n", "\n");
const moduleSource = readFileSync(new URL("../examples/meowdoku/web/living_map_walk_controls.js", import.meta.url), "utf8");
const originalFunction = name => {
  const match = source.match(new RegExp(`^function ${name}\\([^\\n]*\\) \\{\\n[\\s\\S]*?^\\}`, "m"));
  assert.ok(match, `incumbent function ${name}`);
  return match[0];
};

class Target {
  listeners = new Map(); hidden = false; pointerLockElement = null;
  classList = {toggle() {}};
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(fn);
  }
  removeEventListener(type, fn) { this.listeners.get(type)?.delete(fn); }
  emit(type, fields = {}) {
    const event = {type, prevented: false, preventDefault() {this.prevented = true;}, ...fields};
    for (const fn of [...(this.listeners.get(type) || [])]) fn(event);
    return event;
  }
  count() { return [...this.listeners.values()].reduce((sum, set) => sum + set.size, 0); }
  focus() { this.focused = true; }
  blur() { this.focused = false; }
  exitPointerLock() { this.pointerLockElement = null; }
  getBoundingClientRect() { return {left: 0, top: 0, right: 200, bottom: 100}; }
}

// Test-only scene/body fixtures. Production construction must supply all the
// existing world and physics dependencies; the module supplies no such doubles.
function fixture() {
  const calls = [], host = new Target(), doc = new Target(), element = new Target(), canvas = new Target();
  canvas.requestPointerLock = () => { doc.pointerLockElement = canvas; };
  const model = {viewer: {camera: {eye_height: 1.15, collision_radius: .25},
    control_policy: {identity: "world/controls", actor: "player", ...structuredClone(LIVING_MAP_VIEWPORT_POLICY_DEFAULTS),
      bindings: structuredClone(LIVING_MAP_VIEWPORT_BINDINGS)}}, document_geometry: {}};
  const context = {model, viewportControls: createLivingMapControlState(model),
    shaderViewer: {element, canvas, mapElement: null, colliders: [], geometry: [],
      inhabitedCameraFacing: [1, 0, 0]},
    entityState: new Map([["player", {worldPosition: [0, 1.15, 0], velocity: [0, 0, 0], facing: [1, 0, 0]}]]),
    physicsRuntime: {verticalVelocity: 0, grounded: true, supportSuppressedUntil: 0,
      parameters: new Map([["radius", .25]]), instance: null, plugin: null},
    portalRuntime: {transits: new Map()}, mobileControlState: {move: [0, 0], look: [0, 0], touchLookSpeed: 2.4},
    vehicleRuntime: {active: null, parkedState: null}, controlFocus: {mode: "game", dialogue: null},
    stateLoopRuntime: {ready: false, worker: null, actorRegistered: false, bodies: new Map()},
    latestControlInput: {position: null}, primaryActionState: {down: false}, secondaryActionState: {down: false},
    navigator: {getGamepads: () => []},
    document: doc, performance: {now: () => 1000}};
  for (const name of LIVING_MAP_WALK_REQUIRED_CALLBACKS) context[name] = (...args) => calls.push([name, ...args]);
  context.sampleContactSurface = () => null;
  context.reservePhysicsSnapshotSlot = identity => {
    const body = {identity, slot: 0, generation: 1, position: [0, 1.15, 0], velocity: [0, 0, 0], controlGeneration: 0};
    context.stateLoopRuntime.bodies.set(identity, body); return body;
  };
  return {context, calls, host, doc, element, canvas};
}

function reference(context) {
  const keys = Object.keys(context);
  return new Function(...keys,
    `${LIVING_MAP_WALK_SOURCE_FUNCTIONS.map(originalFunction).join("\n\n")}\nreturn {${LIVING_MAP_WALK_SOURCE_FUNCTIONS.join(",")}};`
  )(...keys.map(key => context[key]));
}

const sample = fixture(), api = createLivingMapWalkController(sample.context);
for (const name of LIVING_MAP_WALK_SOURCE_FUNCTIONS) {
  assert.equal(api[name].toString(), originalFunction(name), `unchanged body: ${name}`);
}
const originalState = source.match(/^const viewportControls = \{\n[\s\S]*?^\};/m)[0];
assert.ok(moduleSource.includes(originalState), "state initializer is verbatim");
assert.ok(Object.isFrozen(LIVING_MAP_VIEWPORT_BINDINGS));
assert.ok(Object.isFrozen(LIVING_MAP_VIEWPORT_BINDINGS[0].inputs));
assert.ok(Object.isFrozen(LIVING_MAP_VIEWPORT_POLICY_DEFAULTS.release));
const independent = createLivingMapControlState(sample.context.model);
independent.keys.add("KeyW");
assert.equal(sample.context.viewportControls.keys.size, 0);

for (const name of [...LIVING_MAP_WALK_REQUIRED_STATE, ...LIVING_MAP_WALK_REQUIRED_CALLBACKS]) {
  const {context} = fixture(); delete context[name];
  assert.throws(() => createLivingMapWalkController(context), new RegExp(name));
}
const missingWorld = fixture(); delete missingWorld.context.shaderViewer.colliders;
assert.throws(() => createLivingMapWalkController(missingWorld.context), /authored camera\/policy\/world/);
assert.equal(missingWorld.context.viewportControls.position, null, "dependency failures cannot advance state");

// Run the original source and extracted functions through the same representative
// inputs. No expected output is calculated by a replacement movement formula.
function parity(label, scenario) {
  const a = fixture(), b = fixture();
  const actual = createLivingMapWalkController(a.context), original = reference(b.context);
  scenario(a, actual); scenario(b, original);
  for (const name of ["viewportControls", "entityState", "physicsRuntime", "stateLoopRuntime", "mobileControlState", "vehicleRuntime"]) {
    assert.deepEqual(a.context[name], b.context[name], `${label}: ${name}`);
  }
  assert.deepEqual(a.calls, b.calls, `${label}: external calls`);
}

parity("walk, diagonal, run, release and look", (f, c) => {
  c.requestViewportControls();
  const event = (type, code) => ({type, code, preventDefault() {}});
  c.viewportKeyboardInput(event("keydown", "KeyW")); c.updateViewportControls(.05);
  c.viewportKeyboardInput(event("keydown", "KeyD")); c.updateViewportControls(.05);
  c.viewportKeyboardInput(event("keydown", "ShiftLeft")); c.updateViewportControls(.05);
  c.viewportPointerLook({movementX: 12, movementY: -7}); c.updateViewportControls(.016);
  c.viewportKeyboardInput(event("keyup", "KeyW")); c.viewportKeyboardInput(event("keyup", "KeyD"));
  c.updateViewportControls(.1);
  c.setViewportControlHighlight(false); c.updateViewportControls(.1);
});

parity("real collider and sloped support dependency", (f, c) => {
  c.requestViewportControls();
  f.context.shaderViewer.colliders.push({identity: "wall", minimum: [.5, 0, -2], maximum: [.75, 3, 2]});
  f.context.viewportControls.keys.add("KeyW");
  for (let i = 0; i < 12; i++) c.updateViewportControls(.05);
  assert.equal(f.context.viewportControls.position[0], .25);
  assert.equal(f.context.viewportControls.horizontalVelocity[0], 0);
  assert.deepEqual(c.resolvePlayerVerticalSupport(1.15, 1.14, -.1), {y: 1.14, velocity: -.1, identity: "world-floor"});
});

parity("gamepad, touch and car's declared control record", (f, c) => {
  c.requestViewportControls();
  f.context.navigator.getGamepads = () => [{id: "pad", axes: [.3, -.6, .2, -.2], buttons: []}];
  f.context.mobileControlState.move = [.2, -.1]; f.context.mobileControlState.look = [.15, .25];
  c.updateViewportControls(.04);
  f.context.vehicleRuntime.active = {control_record: [
    {name: "throttle", control: "+move-forward -move-backward"},
    {name: "steering", control: "+strafe-right -strafe-left"}, {name: "brake", control: "jump"}]};
  f.context.viewportControls.keys.add("ShiftLeft"); c.updateViewportControls(.04);
  assert.ok(f.calls.some(call => call[0] === "updateActiveVehicle"));
});

parity("jump and portal suspension", (f, c) => {
  c.requestViewportControls(); assert.equal(c.requestViewportJump(), true);
  assert.equal(f.context.physicsRuntime.verticalVelocity, 3.6);
  assert.equal(c.requestViewportJump(), false);
  f.context.viewportControls.horizontalVelocity = [1, 2];
  f.context.portalRuntime.transits.set("player", {}); c.updateViewportControls(.1);
  assert.deepEqual(f.context.viewportControls.horizontalVelocity, [0, 0]);
});

// Explicit compiled-worker integration still sends the incumbent position
// control and jump message. It must not be represented as a wrench actuator.
const worker = fixture(), workerAPI = createLivingMapWalkController(worker.context), messages = [];
worker.context.stateLoopRuntime.ready = true;
worker.context.stateLoopRuntime.worker = {postMessage: value => messages.push(value)};
workerAPI.requestViewportControls(); workerAPI.registerPlayerPhysicsBody();
assert.equal(messages[0].type, "upsert"); assert.equal(messages[0].body.identity, "player");
worker.context.viewportControls.keys.add("KeyW"); workerAPI.updateViewportControls(.05); workerAPI.stepCompiledWorldPhysics(.05);
assert.equal(messages.at(-1).type, "control");
assert.equal(workerAPI.requestViewportJump(), true); assert.equal(messages.at(-1).type, "player-jump");
assert.equal(messages.some(message => message.type === "wrench"), false);
workerAPI.registerPlayerPhysicsBody(); assert.equal(messages.filter(message => message.type === "upsert").length, 1);

// The authoritative support callback is consumed directly, including its
// slope velocity constraint; neither meshes nor a new floor are fabricated.
const support = fixture();
support.context.sampleContactSurface = () => ({height: 2, gradient: [.2, .3], identity: "authored-ramp", normal: [0, 1, 0]});
const supportAPI = createLivingMapWalkController(support.context); supportAPI.requestViewportControls();
support.context.viewportControls.horizontalVelocity = [2, 3];
assert.deepEqual(supportAPI.resolvePlayerVerticalSupport(3.2, 3.1, -.2),
  {y: 3.15, velocity: 1.2999999999999998, identity: "authored-ramp", gradient: [.2, .3], normal: [0, 1, 0]});

const life = fixture(), lifecycle = createLivingMapWalkController(life.context);
const detach = lifecycle.attach({host: life.host});
assert.equal(lifecycle.attach({host: life.host}), detach, "repeated attach does not duplicate listeners");
lifecycle.requestViewportControls();
assert.equal(life.doc.emit("keydown", {code: "KeyW"}).prevented, true);
assert.equal(life.context.viewportControls.keys.has("KeyW"), true);
life.doc.emit("keyup", {code: "KeyW"}); assert.equal(life.context.viewportControls.keys.size, 0);
life.doc.emit("keydown", {code: "KeyW"}); life.host.emit("blur");
assert.equal(life.context.viewportControls.highlighted, false); assert.equal(life.context.viewportControls.keys.size, 0);
life.context.primaryActionState.down = true; life.context.secondaryActionState.down = true;
life.host.emit("blur");
assert.ok(life.calls.some(call => call[0] === "endViewportPrimary" && call[1] === "window-blur"));
assert.ok(life.calls.some(call => call[0] === "endViewportSecondary" && call[1] === "window-blur"));
life.context.primaryActionState.down = false; life.context.secondaryActionState.down = false;
lifecycle.requestViewportControls(); life.doc.emit("keydown", {code: "KeyW"});
life.doc.hidden = true; life.doc.emit("visibilitychange"); assert.equal(life.context.viewportControls.keys.size, 0);
life.doc.hidden = false; lifecycle.requestViewportControls(); life.doc.emit("keydown", {code: "KeyA"});
const positionBefore = [...life.context.viewportControls.position];
detach(); detach();
assert.equal(life.doc.count() + life.host.count() + life.element.count(), 0);
assert.equal(life.doc.pointerLockElement, null); assert.equal(life.context.viewportControls.keys.size, 0);
life.doc.emit("keydown", {code: "KeyW"}); lifecycle.updateViewportControls(.1);
assert.deepEqual(life.context.viewportControls.position, positionBefore);
const detachAgain = lifecycle.attach({host: life.host}); assert.notEqual(detachAgain, detach); detachAgain();
assert.equal(life.doc.count() + life.host.count() + life.element.count(), 0);

console.log(`PASS Living Map walking extraction: ${LIVING_MAP_WALK_SOURCE_FUNCTIONS.length} exact source functions, strict dependencies, source parity, controls, collision/support, worker protocol and listener lifecycle`);
