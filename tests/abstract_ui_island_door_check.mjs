// A door between ISLANDS (two worlds, each its own document), on the incumbent functions of the living map's
// browser template: the walker leaves through a sheet whose partner is in another island, and arrives in that
// island at the sheet carrying the same pair.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";

const source = readFileSync(new URL("../src/compiler/abstract_ui_div_map.py", import.meta.url), "utf8").replaceAll("\r\n", "\n");
const fn = name => {
  const match = source.match(new RegExp(`^function ${name}\\([^\\n]*\\) \\{\\n[\\s\\S]*?^\\}`, "m"));
  assert.ok(match, `incumbent function ${name}`);
  return match[0];
};
const NAMES = ["normalized3", "cross3", "quaternionNormalize", "quaternionMultiply", "quaternionBetween",
  "quaternionRotate", "portalCurvePoint", "portalCurveTangent", "portalCurveFrame", "portalApertureClass",
  "portalDeclaredPair", "portalDoorTransition", "rebuildPortalGraph", "boxYawDegrees", "portalOpeningTransform",
  "migrateLegacyPortalOpenings", "choosePortalGraphEdge", "synchronizePortalBody", "advancePortalTransit",
  "islandArrivalPair", "completeIslandArrival", "leaveThroughIslandDoor", "traversePortalBody"];

function island(hosts, {hash = "", cancel = false} = {}) {
  const clock = {now: 1000};
  const env = {
    portalRuntime: {splats: [], graph: {identity: "g", sigma: 4.0, traversalSpeed: 3.2, nodes: [], edges: []},
      transits: new Map(), lastPositions: new Map(), cooldowns: new Map(), cooldownMilliseconds: 240},
    shaderViewer: {geometry: hosts, cameraFacing: [1, 0, 0]},
    index: new Map(), livingEditPersistence: {dirty: new Set()}, model: {world: {objects: []}},
    placementState: {sequence: 0}, stateLoopRuntime: {bodies: new Map(), ready: false},
    viewportControls: {policy: {actor: "walker"}, yaw: 0, pitch: 0}, issued: [], events: [], assigned: [],
    actionEdges: {register: (a, kind, b) => `${a}|${kind}|${b}`},
    setPlacementStatus() {}, cancelEntityNavigation() {}, performance: {now: () => clock.now},
    CustomEvent: class { constructor(type, init) { this.type = type; this.detail = init.detail; } },
  };
  env.abstractUISystemTimer = {issue: event => env.issued.push(event)};
  env.window = {dispatchEvent: event => { env.events.push(event); return !cancel; }};
  env.location = {hash, assign: url => env.assigned.push(url)};
  const keys = Object.keys(env);
  const api = new Function(...keys, `${NAMES.map(fn).join("\n")}
    return {migrateLegacyPortalOpenings, traversePortalBody, islandArrivalPair, completeIslandArrival};`)(
    ...keys.map(key => env[key]));
  return {...env, ...api, clock};
}

const host = (identity, x, z, half, openings) => ({identity, center: [x, z], half_extent: [half, half],
  wall_thickness: 0.12, openings});
const door = "lot/door:island:owner/name/src/x.py", pair = `${door}/enter`;
const sheet = (name, role, extra = {}) => ({identity: `${door}/${name}:${role}`, kind: "portal", port_role: role,
  pair: `${door}/${name}`, transition: "door", side: "south", offset: 0, width: 0.72, height: 0.72, ...extra});
const lotHost = () => host("lot", 0, 0, 1, [
  {identity: "lot/opening:entry", kind: "door", side: "south", offset: 0, width: 0.72, height: 0.72},
  sheet("enter", "in", {island: "island:owner/name/src/x.py", href: "/islands/x.html"}), sheet("exit", "out")]);
const walkIn = (world, identity) => {
  const body = {position: [0, 0.36, -1.4], velocity: [0, 0, 2]};
  world.traversePortalBody(identity, body.position, body.velocity, 0.2);
  for (let i = 0; i < 12; i += 1) {
    world.clock.now += 16; body.position[2] += 0.08;
    world.traversePortalBody(identity, body.position, body.velocity, 0.2);
  }
  return body;
};

// OUTSIDE world: the lot's enter sheet has no partner here; its partner is in the island.
{
  const outside = island([lotHost()]);
  outside.migrateLegacyPortalOpenings();
  const enter = outside.portalRuntime.splats.find(s => s.identity === `${door}/enter:in`);
  assert.equal(enter.island, "island:owner/name/src/x.py");
  assert.equal(enter.href, "/islands/x.html");
  assert.equal(outside.portalRuntime.graph.edges.length, 0, "no link inside this world");

  walkIn(outside, "walker");
  assert.equal(outside.events.length, 1);
  assert.equal(outside.events[0].type, "abstract-ui-island-door");
  assert.deepEqual(outside.events[0].detail, {island: "island:owner/name/src/x.py", pair, source: `${door}/enter:in`,
    href: "/islands/x.html"});
  assert.deepEqual(outside.assigned, [`/islands/x.html#arrive=${encodeURIComponent(pair)}`]);
  assert.equal(outside.issued.at(-1).type, "leave-through-island-door");
  assert.equal(outside.issued.at(-1).destination, "island:owner/name/src/x.py");
  assert.equal(outside.portalRuntime.transits.size, 0);
}

// Only the walker leaves an island; a marble crossing the same sheet stays, and nothing is announced.
{
  const outside = island([lotHost()]);
  outside.migrateLegacyPortalOpenings();
  const marble = walkIn(outside, "marble-7");
  assert.equal(outside.events.length, 0);
  assert.equal(outside.assigned.length, 0);
  assert.ok(marble.position[2] > -1.4, "the marble kept rolling in this island");
}

// A host page may take over: cancelling the event stops the navigation.
{
  const outside = island([lotHost()], {cancel: true});
  outside.migrateLegacyPortalOpenings();
  walkIn(outside, "walker");
  assert.equal(outside.events.length, 1);
  assert.equal(outside.assigned.length, 0);
}

// THE ISLAND's own world: arriving through #arrive=<pair> puts the walker just inside its one passage.
{
  const inside = island([host("interior", 0, 0, 4, [
    {identity: "interior/opening:gate", kind: "gate", side: "south", offset: 0, width: 1.1, height: 0.82},
    sheet("enter", "out", {facing: "inward", width: 1.1, height: 0.82}),
    sheet("exit", "in", {facing: "inward", width: 1.1, height: 0.82, island: "world:outside", href: "/outside.html"})])],
    {hash: `#arrive=${encodeURIComponent(pair)}`});
  inside.migrateLegacyPortalOpenings();
  assert.equal(inside.islandArrivalPair(), pair);
  inside.portalRuntime.arrival = inside.islandArrivalPair();
  const position = [50, 0.36, 50], velocity = [3, 0, 3];
  assert.equal(inside.completeIslandArrival("walker", position, velocity, 0.2), true);
  assert.ok(Math.abs(position[0]) < 1e-9 && position[2] > -3.94 && position[2] < -3.5, `at ${position}`);
  assert.deepEqual(velocity, [0, 0, 0]);
  assert.ok(Math.abs(inside.viewportControls.yaw - Math.PI / 2) < 1e-9, "facing into the island");
  assert.equal(inside.portalRuntime.arrival, null);
  assert.equal(inside.completeIslandArrival("walker", position, velocity, 0.2), false, "arrival happens once");

  // And the way back: leaving by the passage names the world outside and the same door's exit pair.
  const body = {position: [0, 0.36, -3.0], velocity: [0, 0, -2]};
  inside.clock.now += 5000;
  inside.traversePortalBody("walker", body.position, body.velocity, 0.2);
  for (let i = 0; i < 20 && !inside.assigned.length; i += 1) {
    inside.clock.now += 16; body.position[2] -= 0.08;
    inside.traversePortalBody("walker", body.position, body.velocity, 0.2);
  }
  assert.deepEqual(inside.assigned, [`/outside.html#arrive=${encodeURIComponent(`${door}/exit`)}`]);
  assert.equal(inside.events.at(-1).detail.island, "world:outside");
}

// A page opened without an arrival, or with one this island does not have, leaves the walker where it is.
{
  const plain = island([lotHost()]);
  plain.migrateLegacyPortalOpenings();
  assert.equal(plain.islandArrivalPair(), null);
  const lost = island([lotHost()], {hash: "#arrive=some/other/door"});
  lost.migrateLegacyPortalOpenings();
  lost.portalRuntime.arrival = lost.islandArrivalPair();
  const position = [1, 2, 3];
  assert.equal(lost.completeIslandArrival("walker", position, null, 0.2), false);
  assert.deepEqual(position, [1, 2, 3]);
}
console.log("island door ok");

// A fresh visitor has no saved game. The declared doors must still become splats: every way out of the restore
// function converts them (found in a real browser: with no payload the outside page had no door at all).
{
  const restore = fn("restoreLivingEdits");
  assert.ok(restore.includes("if (!payload) { migrateLegacyPortalOpenings(); return; }"));
  assert.ok(restore.includes("{ migrateLegacyPortalOpenings(); return; }\n    Object.entries(saved.objects"));
  assert.ok(/console\.warn\("AbstractUI saved edits were ignored", error\);\n\s+migrateLegacyPortalOpenings\(\);/.test(restore));
  // Converting twice is harmless: the second pass finds no portal openings left on the hosts.
  const twice = island([lotHost()]);
  assert.equal(twice.migrateLegacyPortalOpenings(), 2);
  assert.equal(twice.migrateLegacyPortalOpenings(), 0);
  assert.equal(twice.portalRuntime.splats.length, 2);
}
console.log("fresh visitor gets doors ok");

// Found in a real page: the game's action timer refuses an issued action whose edge is not registered, and a
// walker seated in the vehicle was carried away again after arriving.
{
  const outside = island([lotHost()]);
  outside.migrateLegacyPortalOpenings();
  walkIn(outside, "walker");
  assert.equal(outside.issued.at(-1).edge,
    `${door}/enter:in|leave-through-island-door|island:owner/name/src/x.py`);
  const arrive = fn("completeIslandArrival");
  assert.ok(arrive.includes("vehicleRuntime.active) clearActiveVehicle();"));
  assert.ok(arrive.indexOf("clearActiveVehicle()") < arrive.indexOf("position.forEach"));
}
console.log("action edge and on-foot arrival ok");
