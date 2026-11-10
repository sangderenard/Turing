import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../examples/meowdoku/web/mechanical_creature_appearance_hook.js", import.meta.url), "utf8");
const calls = [], faults = [];
const gl = {TRIANGLES: 4, bindVertexArray: value => calls.push(["vao", value]),
  drawArrays: (...args) => calls.push(["draw", ...args]), useProgram: value => calls.push(["program", value])};
const spans = [
  {identity: "world", firstVertex: 0, vertexCount: 3},
  {identity: "ball-a", firstVertex: 3, vertexCount: 6},
  {identity: "ball-b", firstVertex: 12, vertexCount: 6},
  {identity: "pickup", firstVertex: 24, vertexCount: 6},
  {identity: "vehicle", firstVertex: 30, vertexCount: 6},
  {identity: "bad", firstVertex: 6.1, vertexCount: 3},
];
const projectileState = new Map([["ball-a", {position: [1, 2, 3], appearanceAssetIdentity: "asset-a"}],
  ["ball-b", {sleeping: true, appearanceAssetIdentity: "asset-a"}], ["bad", {appearanceAssetIdentity: "asset-a"}], ["world", {}]]);
const projectilePickupState = new Map([["pickup", {position: [4, 5, 6], appearanceAssetIdentity: "asset-a"}]]);
const stateLoopRuntime = {bodies: new Map([["ball-a", {slot: 2, generation: 3}]]), ready: true,
  slotRecords: [], latestSequence: 7};
const context = vm.createContext({Set, gl, shaderViewer: {vao: "base", vertexCount: 30, identitySpans: spans, program: "original"},
  projectileState, projectilePickupState, stateLoopRuntime, reportRuntimeFault: (...args) => faults.push(args)});
vm.runInContext(source, context);
const prepare = () => context.prepareMechanicalCreatureImportedAppearance(gl, [1, 2, 3], [0, 0, -1], 400, 800);
const drawCalls = () => calls.filter(call => call[0] === "draw");
assert.equal(prepare(), null, "adapter absent leaves original game alone");
context.drawMechanicalCreatureBaseAppearance(gl);
assert.deepEqual(drawCalls(), [["draw", 4, 0, 30]], "unfiltered depth/base draw remains whole VAO");
calls.length = 0;

let selectedContext, phases = [];
context.mechanicalCreatureImportedAppearance = {
  identities(value) {selectedContext = value; return new Set(["ball-a", "ball-b", "pickup", "world", "vehicle", "absent", "bad"]);},
  draw(value) {phases.push(value.phase); assert.equal(value.stateLoopRuntime, stateLoopRuntime);
    assert.equal(value.snapshotBodies, stateLoopRuntime.bodies);},
};
const frame = prepare();
assert.deepEqual([...frame.identities], ["ball-a", "ball-b", "pickup"]);
assert.equal(selectedContext.projectileState, projectileState);
assert.equal(selectedContext.projectilePickupState, projectilePickupState);
assert.equal(selectedContext.stateLoopRuntime, stateLoopRuntime);
context.drawMechanicalCreatureBaseAppearance(gl, frame.identities);
assert.deepEqual(drawCalls(), [["draw", 4, 0, 3], ["draw", 4, 9, 3], ["draw", 4, 18, 6]],
  "only chosen base color ranges are skipped, including sleeping/pickup bodies");
context.drawMechanicalCreatureImportedAppearance(frame);
assert.deepEqual(phases, ["opaque", "transparent"]);
assert.equal(faults.length, 0);
assert.equal(spans.length, 6); assert.equal(projectileState.size, 4);
assert.equal(projectilePickupState.size, 1); assert.equal(stateLoopRuntime.latestSequence, 7);

calls.length = 0;
context.drawMechanicalCreatureBaseAppearance(gl);
assert.deepEqual(drawCalls(), [["draw", 4, 0, 30]], "depth remains unfiltered even with active adapter");
calls.length = 0;
context.mechanicalCreatureImportedAppearance.identities = () => new Set(["vehicle", "world", "absent", "bad"]);
assert.equal(prepare(), null, "unadmitted identities cannot hide world geometry");
context.mechanicalCreatureImportedAppearance.identities = () => ["ball-a"];
assert.equal(prepare(), null, "malformed adapter keeps proxy visible");
assert.equal(faults.length, 1);

context.mechanicalCreatureImportedAppearance.identities = () => new Set(["ball-a"]);
context.mechanicalCreatureImportedAppearance.draw = () => {throw new Error("test renderer failure");};
context.drawMechanicalCreatureImportedAppearance(prepare());
assert.deepEqual(drawCalls(), [["draw", 4, 3, 6]], "renderer failure draws original selected proxy");
assert.ok(calls.some(call => call[0] === "program" && call[1] === "original"));
assert.equal(faults.length, 2);
calls.length = 0;
context.mechanicalCreatureImportedAppearance.draw = () => false;
context.drawMechanicalCreatureImportedAppearance(prepare());
assert.deepEqual(drawCalls(), [["draw", 4, 3, 6]], "declined draw restores proxy");
calls.length = 0;
context.mechanicalCreatureImportedAppearance = null;
assert.equal(prepare(), null);
context.drawMechanicalCreatureBaseAppearance(gl);
assert.deepEqual(drawCalls(), [["draw", 4, 0, 30]], "uninstall restores original draw immediately");
context.shaderViewer.geometry=[{identity:'world',appearance:{studio_asset_identity:'studio-room'}}];
context.mechanicalCreatureImportedAppearance={drawWhenEmpty:true,identities:()=>new Set(),draw:()=>true};
assert.equal(prepare().identities.size,0,'optional static renderer can run before any projectile exists');
context.mechanicalCreatureImportedAppearance.identities=()=>new Set(['world','vehicle']);
assert.deepEqual([...prepare().identities],['world'],'only explicitly tagged original static assets can suppress their color proxy');
delete context.shaderViewer.geometry[0].appearance.studio_asset_identity;
assert.equal(prepare().identities.size,0,'unmarked original world cannot be suppressed');
assert.doesNotMatch(source, /requestAnimationFrame|setInterval|setTimeout|postMessage|\.position\s*=/,
  "appearance hook owns no clock, movement or worker action");
console.log("PASS: original-game appearance identity, range, phase, fallback and state-preservation checks");
