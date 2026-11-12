// A door (nonlocal portal) walked end to end on the incumbent functions of the living map's browser template:
// openings declared on two host boxes -> splats -> declared pairs -> a body crossing the sheet arrives on the
// other side at once. Local portals (no declared transition) keep the tube ride.
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
  "traversePortalBody"];

function engine(hosts) {
  const clock = {now: 1000};
  const env = {
    portalRuntime: {splats: [], graph: {identity: "g", sigma: 4.0, traversalSpeed: 3.2, nodes: [], edges: []},
      transits: new Map(), lastPositions: new Map(), cooldowns: new Map(), cooldownMilliseconds: 240},
    shaderViewer: {geometry: hosts, cameraFacing: [1, 0, 0]},
    index: new Map(), livingEditPersistence: {dirty: new Set()}, model: {world: {objects: []}},
    placementState: {sequence: 0}, stateLoopRuntime: {bodies: new Map(), ready: false},
    viewportControls: {policy: null}, issued: [],
    actionEdges: {register: (a, kind, b) => `${a}|${kind}|${b}`},
    setPlacementStatus() {}, cancelEntityNavigation() {}, performance: {now: () => clock.now},
  };
  env.abstractUISystemTimer = {issue: event => env.issued.push(event)};
  const keys = Object.keys(env);
  const api = new Function(...keys, `${NAMES.map(fn).join("\n")}
    return {migrateLegacyPortalOpenings, traversePortalBody};`)(...keys.map(key => env[key]));
  return {...env, ...api, clock};
}

const host = (identity, x, z, half, openings) => ({identity, center: [x, z], half_extent: [half, half],
  wall_thickness: 0.12, openings});
const door = "lot/door:interior";
const end = (pair, role, extra = {}) => ({identity: `${door}/${pair}:${role}`, kind: "portal", port_role: role,
  pair: `${door}/${pair}`, transition: "door", side: "south", offset: 0, width: 0.72, height: 0.72, ...extra});

// The lot stands at the origin; its interior is 900 units away, where a distance-weighted link would be zero.
const lot = host("lot", 0, 0, 1, [{identity: "lot/opening:entry", kind: "door", side: "south", offset: 0,
  width: 0.72, height: 0.72}, end("enter", "in"), end("exit", "out")]);
const interior = host("interior", 900, 0, 4, [{identity: "interior/opening:gate", kind: "gate", side: "south",
  offset: 0, width: 1.1, height: 0.82},
  end("enter", "out", {facing: "inward", width: 1.1, height: 0.82}),
  end("exit", "in", {facing: "inward", width: 1.1, height: 0.82})]);
const world = engine([lot, interior]);
assert.equal(world.migrateLegacyPortalOpenings(), 4);

// Openings became splats; the minted door and gate stay as openings.
assert.deepEqual(lot.openings.map(o => o.kind), ["door"]);
assert.deepEqual(interior.openings.map(o => o.kind), ["gate"]);
const splat = id => world.portalRuntime.splats.find(candidate => candidate.identity === `${door}/${id}`);
assert.deepEqual(splat("enter:in").normal, [0, 0, -1]);          // the lot's south wall faces out
assert.deepEqual(splat("enter:out").normal.map(v => v + 0), [0, 0, 1]);   // the interior's passage faces in
assert.ok(world.portalRuntime.splats.every(s => s.transition === "door" && s.pair));

// Exactly the two declared links, each certain, each a door; neither end leaks to the other pair.
const edges = world.portalRuntime.graph.edges;
assert.deepEqual(edges.map(e => `${e.source}>${e.target}`).sort(),
  [`${door}/enter:in>${door}/enter:out`, `${door}/exit:in>${door}/exit:out`]);
assert.ok(edges.every(e => e.weight === 1 && e.transition === "door" && e.path_model === "material-sheet"));

// Walk in the lot's door from outside (south of the wall, moving north).
const body = {position: [0, 0.36, -1.4], velocity: [0, 0, 2]};
const step = (dz, dt = 16) => { world.clock.now += dt; body.position[2] += dz;
  return world.traversePortalBody("walker", body.position, body.velocity, 0.2); };
assert.equal(step(0), false);                  // first sample only records where the body is
let moved = false;
for (let i = 0; i < 12 && !moved; i += 1) moved = step(0.08);
assert.ok(moved, "crossing the sheet starts and completes the transition in one call");
assert.equal(world.portalRuntime.transits.size, 0, "no tube ride is left in flight");
// Arrived just inside the interior's passage: at its wall (z = -4 + 0.06), pushed inward by the clearance.
assert.ok(Math.abs(body.position[0] - 900) < 0.5, `x ${body.position[0]}`);
assert.ok(body.position[2] > -3.94 && body.position[2] < -3.5, `z ${body.position[2]}`);
assert.ok(body.velocity[2] > 0, "still heading inward");
assert.equal(world.issued.at(-1).path_model, "material-sheet");
assert.equal(world.issued.at(-1).probability, 1);

// Walk on inward: nothing happens. Then turn round and leave by the one passage.
world.clock.now += 400; step(0.5); assert.ok(body.position[2] > -3.5);
body.velocity = [0, 0, -2];
moved = false;
for (let i = 0; i < 40 && !moved; i += 1) moved = step(-0.08);
assert.ok(moved, "leaving by the passage crosses the exit sheet");
// Back outside the lot's door: south of its wall, heading away from it.
assert.ok(Math.abs(body.position[0]) < 0.5, `x ${body.position[0]}`);
assert.ok(body.position[2] < -0.94 && body.position[2] > -1.5, `z ${body.position[2]}`);
assert.ok(body.velocity[2] < 0);

// A local portal pair (declared, no transition) is still a ride: crossing starts a transit that lasts.
{
  const a = host("a", 0, 0, 1, [{identity: "a/p:in", kind: "portal", port_role: "in", pair: "ride", side: "south",
    offset: 0, width: 0.72, height: 0.72}]);
  const b = host("b", 20, 0, 1, [{identity: "b/p:out", kind: "portal", port_role: "out", pair: "ride",
    side: "south", offset: 0, width: 0.72, height: 0.72}]);
  const local = engine([a, b]);
  local.migrateLegacyPortalOpenings();
  const edge = local.portalRuntime.graph.edges[0];
  assert.ok(!("transition" in edge) && edge.path_model === "relaxed-quaternion-cubic");
  const rider = {position: [0, 0.36, -1.4], velocity: [0, 0, 2]};
  local.traversePortalBody("rider", rider.position, rider.velocity, 0.2);
  let riding = false;
  for (let i = 0; i < 12 && !riding; i += 1) {
    local.clock.now += 16; rider.position[2] += 0.08;
    riding = local.traversePortalBody("rider", rider.position, rider.velocity, 0.2);
  }
  assert.ok(riding);
  assert.equal(local.portalRuntime.transits.size, 1, "the ride is in flight");
  assert.ok(local.portalRuntime.transits.get("rider").duration >= 280);
}
console.log("door transition ok");
