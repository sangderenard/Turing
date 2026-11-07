import assert from "node:assert/strict";

// The access module's network probe has no browser here; the panel's pure
// timing helper can still be exercised without a WebGL context or server.
globalThis.window = {};
const { advanceStage, MIN_STAGE_MS } = await import("../examples/meowdoku/web/loading_panel.js");
const download = { phase: "downloading" }, init = { phase: "initializing" }, compile = { phase: "compiling" };
let current = advanceStage(download, [init, compile], null, 100);
assert.equal(current.stage, download);
current = advanceStage(current.stage, current.stages, current.stageAt, 100 + MIN_STAGE_MS - 1);
assert.equal(current.stage, download);
current = advanceStage(current.stage, current.stages, current.stageAt, 100 + MIN_STAGE_MS);
assert.equal(current.stage, init);
current = advanceStage(current.stage, current.stages, current.stageAt, 100 + 2 * MIN_STAGE_MS - 1);
assert.equal(current.stage, init);
current = advanceStage(current.stage, current.stages, current.stageAt, 100 + 2 * MIN_STAGE_MS);
assert.equal(current.stage, compile);
assert.equal(current.stages.length, 0);
console.log("loading stages each remain visible for the minimum interval");
