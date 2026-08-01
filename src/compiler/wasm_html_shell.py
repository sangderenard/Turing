"""A launchable HTML shell for a compiled WebAssembly program.

The other shells in this repository launch a compiled artifact in its own
environment: ``profiled_c_shell`` runs a C artifact and reports timings,
``control_source.compile_python_shell`` loads a Python one. This is the same
idea for the WebAssembly target -- the browser is that target's launch
environment, so its shell is a page.

It exists for hands-on diagnosis. A compiled program is otherwise opaque:
you can read the WAT and you can read the API descriptor, but you cannot
*poke* it -- change one input, run it again, and see which output moved.
That is the loop this page gives.

Nothing here is specific to any one program. The page is generated from the
API descriptor (``compiled_program_api.py``), so the controls are whatever
that program's parameters are: one input row per feed, one result row per
output, the entry point's name on the button. Compile something else and the
page reshapes itself.

Deliberately not a layout manager. Layout belongs to a different subrepo, so
this is plain ``div``s and a little CSS -- a stack of labelled rows, one
column, no grid engine, no component model, no dependencies. If it ever
starts growing a layout system, that is the signal it should be handed to
the subrepo that already owns one instead.

WebAssembly binaries, not text: a browser cannot assemble WAT. When an
assembled ``.wasm`` is available the page carries it inline as base64 and is
a single self-contained file. When it is not -- ``wat2wasm`` is not
installed here -- the page still generates, shows the WAT for reading, and
offers a file picker so a caller who assembles it elsewhere can drop the
binary in and run. It says which of those two states it is in rather than
looking broken.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_CSS = """
:root {
  color-scheme: light dark;
  --line: color-mix(in srgb, currentColor 18%, transparent);
  --soft: color-mix(in srgb, currentColor 6%, transparent);
  --accent: #3b82f6;
  --bad: #dc2626;
  --good: #16a34a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 1.5rem;
  font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  max-width: 60rem;
}
.title { font-size: 1.25rem; font-weight: 600; }
.sub { opacity: .7; font-size: .85rem; margin-top: .15rem; }
.panel {
  border: 1px solid var(--line);
  border-radius: .5rem;
  padding: .9rem 1rem;
  margin-top: 1rem;
}
.panel-title {
  font-weight: 600;
  font-size: .8rem;
  letter-spacing: .04em;
  text-transform: uppercase;
  opacity: .65;
  margin-bottom: .6rem;
}
.row { display: flex; align-items: baseline; gap: .75rem; padding: .3rem 0; }
.row + .row { border-top: 1px solid var(--soft); }
.name { font-family: ui-monospace, monospace; min-width: 8rem; font-weight: 600; }
.meta { opacity: .6; font-size: .8rem; min-width: 12rem; }
.grow { flex: 1; }
input[type=text], input[type=number], textarea {
  min-height: 8rem;
  font: inherit;
  font-family: ui-monospace, monospace;
  width: 100%;
  padding: .35rem .5rem;
  border: 1px solid var(--line);
  border-radius: .35rem;
  background: transparent;
  color: inherit;
}
button {
  font: inherit;
  font-weight: 600;
  padding: .5rem 1.1rem;
  border: 0;
  border-radius: .35rem;
  background: var(--accent);
  color: #fff;
  cursor: pointer;
}
button:disabled { opacity: .45; cursor: not-allowed; }
pre {
  margin: 0;
  padding: .75rem;
  overflow: auto;
  max-height: 24rem;
  background: var(--soft);
  border-radius: .35rem;
  font-family: ui-monospace, monospace;
  font-size: .8rem;
}
.out { font-family: ui-monospace, monospace; white-space: pre-wrap; word-break: break-word; }
.note { font-size: .85rem; padding: .6rem .75rem; border-radius: .35rem; background: var(--soft); }
.bad { color: var(--bad); }
.tabs { display: flex; gap: .25rem; margin-bottom: .75rem; }
.tab {
  padding: .3rem .8rem; border-radius: .3rem; cursor: pointer;
  font-size: .8rem; font-weight: 600; background: var(--soft);
  border: 1px solid transparent;
}
.tab[aria-selected="true"] { border-color: var(--accent); color: var(--accent); }
.tabview[hidden] { display: none; }
canvas { max-width: 100%; image-rendering: pixelated; border-radius: .35rem;
  border: 1px solid var(--line); }
.imgctl { display: flex; gap: .75rem; align-items: baseline; flex-wrap: wrap;
  margin-bottom: .6rem; }
.imgctl input[type=number] { width: 6rem; }
.imgctl label { font-size: .8rem; opacity: .7; }
#fatal { background: color-mix(in srgb, var(--bad) 15%, transparent);
  border: 1px solid var(--bad); color: var(--bad); }
#log { max-height: 14rem; overflow: auto; font-family: ui-monospace, monospace;
  font-size: .75rem; }
.logline { padding: .1rem 0; border-bottom: 1px solid var(--soft); white-space: pre-wrap; }
.logline.error { color: var(--bad); }
.logline.ok, .logline.call { color: var(--good); }
.logline.warn { opacity: .8; }
.logline.profile { color: #a78bfa; }
.logline.progress { color: var(--accent); }
.bar { height: .5rem; border-radius: .25rem; background: var(--soft); overflow: hidden; }
.bar > i { display: block; height: 100%; width: 0; background: var(--accent);
  transition: width .12s linear; }
.barlabel { display: flex; justify-content: space-between; font-size: .75rem;
  opacity: .7; margin-bottom: .25rem; }
.kv { display: flex; gap: .5rem; font-size: .8rem; padding: .15rem 0; }
.kv b { font-family: ui-monospace, monospace; font-weight: 600; min-width: 9rem; }
.chip { display: inline-block; font-size: .72rem; font-family: ui-monospace, monospace;
  padding: .1rem .4rem; border-radius: .25rem; background: var(--soft); margin: .1rem; }
.execution-modes { display: flex; flex-wrap: wrap; gap: .45rem; margin: .5rem 0; }
.execution-mode { background: var(--soft); color: inherit; border: 1px solid var(--line); }
.execution-mode[aria-pressed="true"] { background: var(--accent); color: white; }
.schedule-level { display: flex; align-items: stretch; gap: .45rem; padding: .3rem 0; }
.schedule-level-label { width: 4rem; flex: 0 0 4rem; font: .72rem ui-monospace, monospace;
  opacity: .6; padding-top: .35rem; }
.deployment-node { flex: 1; min-width: 10rem; border: 1px solid var(--line);
  border-radius: .35rem; padding: .4rem .55rem; transition: .12s ease;
  cursor: pointer; background: var(--soft); }
.deployment-node b { display: block; font: 600 .74rem ui-monospace, monospace; }
.deployment-node .node-state { font: .68rem ui-monospace, monospace; opacity: .75; }
.deployment-node[data-state="downloading"] { border-color: #f59e0b; }
.deployment-node[data-state="running"] { background: var(--accent); color: white;
  transform: translateY(-1px); box-shadow: 0 .2rem .7rem color-mix(in srgb, var(--accent) 35%, transparent); }
.deployment-node[data-state="done"] { border-color: var(--good); }
.deployment-node[data-state="error"] { border-color: var(--bad); color: var(--bad); }
.node-detail { margin-top: .5rem; }
.graph-toolbar { display: flex; gap: .4rem; flex-wrap: wrap; align-items: center;
  margin: .5rem 0; }
.graph-view-button { background: var(--soft); color: inherit; border: 1px solid var(--line); }
.graph-view-button[aria-pressed="true"] { background: var(--accent); color: white; }
.graph-scroll { max-height: 34rem; overflow: auto; border: 1px solid var(--line);
  border-radius: .35rem; background: #05070c; }
#process-graph-canvas { display: block; max-width: none; border: 0; border-radius: 0;
  image-rendering: auto; }
.filters { display: flex; gap: .4rem; margin-bottom: .5rem; font-size: .75rem; }
.filters label { cursor: pointer; opacity: .75; }
.srctabs { display: flex; flex-wrap: wrap; gap: .25rem; margin: .5rem 0; }
.srctab { padding: .25rem .7rem; border-radius: .3rem; cursor: pointer;
  font-size: .75rem; font-weight: 600; background: var(--soft);
  border: 1px solid transparent; font-family: ui-monospace, monospace; }
.srctab[aria-selected="true"] { border-color: var(--accent); color: var(--accent); }
.srcview[hidden] { display: none; }
.stat { display: flex; gap: 1.2rem; flex-wrap: wrap; font-size: .8rem;
  font-family: ui-monospace, monospace; margin-top: .4rem; }
.good { color: var(--good); }
details summary { cursor: pointer; font-weight: 600; font-size: .8rem; opacity: .65;
  text-transform: uppercase; letter-spacing: .04em; }
"""

# The runtime. Written against the API descriptor rather than any particular
# program: it lays the arrays out in the module's memory, calls the entry
# point, and reads the outputs back.
# Installed in its own <script> tag, ahead of the program script. A
# handler defined inside a script cannot catch that script's own parse
# error -- nothing in it has run yet -- so the shell would fail silently
# and look merely inert. This one survives that and says so.
_BOOT_JS = r"""
// Diagnostics first, before anything that can fail. A shell whose script
// dies at load looks identical to a shell that simply does nothing -- the
// controls render either way, because they are static HTML -- so the failure
// has to announce itself here rather than only in a console nobody opened.
const BUILD_TELEMETRY = __TELEMETRY__;
// Build-time and run-time records share one schema and one list, so the
// compilation and the execution read as a single timeline rather than two
// logs a person has to interleave by eye.
const LOG = (BUILD_TELEMETRY.records || []).map(r => ({
  at: (r.at_ns / 1e6).toFixed(1) + "ms",
  kind: r.kind,
  message: r.message,
  detail: Object.keys(r.detail || {}).length ? r.detail : null,
  path: r.path || "",
  phase: "build"
}));
function log(kind, message, detail) {
  const entry = {
    at: new Date().toISOString().slice(11, 23),
    kind: kind,
    message: String(message),
    detail: detail === undefined ? null : detail
  };
  entry.phase = "run";
  LOG.push(entry);
  if (kind === "progress" && detail) setProgress(detail.done, detail.total, message);
  const pane = document.getElementById("log");
  if (pane) {
    const line = document.createElement("div");
    line.className = "logline " + kind;
    line.textContent = entry.at + "  " + kind.toUpperCase() + "  " +
      (entry.path ? "[" + entry.path + "] " : "") + entry.message +
      (entry.detail === null ? "" : "  " + JSON.stringify(entry.detail));
    pane.appendChild(line);
    pane.scrollTop = pane.scrollHeight;
  }
  return entry;
}

function setProgress(done, total, label) {
  const bar = document.getElementById("bar-wrap");
  const fill = document.getElementById("barfill");
  const text = document.getElementById("bartext");
  const pct = document.getElementById("barpct");
  if (!bar) return;
  bar.hidden = false;
  const fraction = total ? Math.max(0, Math.min(1, done / total)) : 0;
  fill.style.width = (fraction * 100).toFixed(1) + "%";
  if (text) text.textContent = label || "";
  if (pct) pct.textContent = done + " / " + total;
}

window.addEventListener("error", (event) => {
  log("error", event.message, {
    line: event.lineno, column: event.colno,
    source: (event.filename || "").split("/").pop()
  });
  const banner = document.getElementById("fatal");
  if (banner) {
    banner.hidden = false;
    banner.textContent = "This page's script failed to load: " + event.message +
      " (line " + event.lineno + "). The controls below are inert.";
  }
});
window.addEventListener("unhandledrejection", (event) => {
  log("error", "unhandled rejection: " + (event.reason && event.reason.message
    ? event.reason.message : event.reason));
});

"""

# The program script proper.
_JS = r"""const API = __API__;
const WASM_BASE64 = __WASM__;

const $ = (id) => document.getElementById(id);
const GRAPH = __GRAPH__;
const GRAPH_VIEWS = __GRAPH_VIEWS__;
const NETWORK = __NETWORK__;
const CLASS_GRAPH = __CLASS_GRAPH__;
const SOURCE_DOWNLOADS = __SOURCE_DOWNLOADS__;
const entry = API.entry_points.find(e => e.name === API.entry) || API.entry_points[0];
const params = entry.parameters;
const inputs = params.filter(p => p.role === "input");
const outputs = params.filter(p => p.role === "output");
const bytes = API.metadata.element_bytes || 8;
const isF32 = (API.metadata.value_type || "f64") === "f32";

let moduleBytes = null;
if (WASM_BASE64) {
  const raw = atob(WASM_BASE64);
  moduleBytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) moduleBytes[i] = raw.charCodeAt(i);
}

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = "out " + (kind || "");
}

function parseNumbers(text) {
  return text.split(/[\s,]+/).filter(s => s.length).map(Number);
}

// Box-Muller, with the spare kept: two normals come out of one pair of
// uniforms, and throwing one away doubles the cost for no reason.
let spareNormal = null;
function gaussian() {
  if (spareNormal !== null) {
    const value = spareNormal;
    spareNormal = null;
    return value;
  }
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const radius = Math.sqrt(-2 * Math.log(u));
  const angle = 2 * Math.PI * v;
  spareNormal = radius * Math.sin(angle);
  return radius * Math.cos(angle);
}

const RUN_LABEL = "Run __ENTRY__";

function reportTimings(timings, count) {
  if (timings.length < 2) { $("stats").innerHTML = ""; return; }
  const ordered = timings.slice().sort((a, b) => a - b);
  const total = timings.reduce((a, b) => a + b, 0);
  const median = ordered[ordered.length >> 1];
  $("stats").innerHTML =
    "<span>runs " + timings.length + "</span>" +
    "<span>median " + median.toFixed(3) + " ms</span>" +
    "<span>min " + ordered[0].toFixed(3) + "</span>" +
    "<span>max " + ordered[ordered.length - 1].toFixed(3) + "</span>" +
    "<span>" + (count * timings.length / total / 1000).toFixed(1) + " Melem/s</span>" +
    "<span>" + (timings.length / (total / 1000)).toFixed(1) + " fps</span>";
}


// Resume after the browser has actually painted.
//
// A requestAnimationFrame callback runs *before* the repaint it is
// scheduling, so resuming there and immediately calling the kernel again
// saturates the main thread and blocks the very paint being waited for. On
// a desktop that mostly squeaks through; on a phone it does not, and the
// document sits there looking frozen while the loop happily keeps running.
// Two nested frames means the first repaint has completed before the second
// callback fires.
//
// rAF is also suspended entirely while the document is hidden, so a bare
// await here would hang the loop forever rather than pause it. The timeout
// is the floor that keeps the loop answerable in that case.
function presented() {
  // The draw has already happened, synchronously, in the loop. This waits
  // for the frame carrying it to be presented before the loop computes
  // again, so a completed picture is never overwritten by the next one
  // before it has been seen.
  //
  // requestAnimationFrame is suspended while the document is hidden, so the
  // timeout is the floor that keeps the loop answerable rather than hung.
  return new Promise(resolve => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    requestAnimationFrame(finish);
    setTimeout(finish, 250);
  });
}

function domain() {
  const w = Math.max(1, Number($("dom_w").value) | 0);
  const h = Math.max(1, Number($("dom_h").value) | 0);
  return { w: w, h: h, n: w * h };
}

// Compiled once per run rather than per element: a 256,000-element feed
// evaluated through a fresh Function each time is the difference between a
// responsive page and a hung one.
let frameIndex = 0;

function feedValues(param, n, d, t) {
  const mode = $("mode_" + param.name).value;
  if (mode === "values") {
    return parseNumbers($("in_" + param.name).value);
  }
  if (mode === "gaussian") {
    const mean = Number($("mean_" + param.name).value) || 0;
    const sigma = Number($("sigma_" + param.name).value);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = mean + (sigma || 0) * gaussian();
    return out;
  }
  const body = $("expr_" + param.name).value;
  let fn;
  try {
    fn = new Function("i", "x", "y", "w", "h", "t", "return (" + body + ");");
  } catch (err) {
    throw new Error("feed " + param.name + ": " + err.message);
  }
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = i % d.w, y = (i / d.w) | 0;
    const value = fn(i, x, y, d.w, d.h, t);
    if (!Number.isFinite(value)) {
      throw new Error("feed " + param.name + " gave " + value + " at i=" + i);
    }
    out[i] = value;
  }
  return out;
}

let feedbackRuntime = null;
let feedbackState = { travel: 0, speed: 1, scores: [] };

async function ensureFeedbackRuntime() {
  const descriptor = NETWORK.module;
  if (!descriptor || !descriptor.wasm_base64) return null;
  if (feedbackRuntime) return feedbackRuntime;
  const raw = atob(descriptor.wasm_base64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const api = descriptor.api;
  const entryPoint = api.entry_points.find(e => e.name === api.entry) || api.entry_points[0];
  feedbackRuntime = { instance, api, entry: entryPoint };
  return feedbackRuntime;
}

async function advanceFeedback(ticks = 1) {
  for (let tick = 0; tick < ticks; tick++) {
  const runtime = await ensureFeedbackRuntime();
  const contract = NETWORK.feedback;
  if (!runtime || !contract) return;
  const offsets = contract.candidate_offsets || [0, 0.45, 0.9];
  const count = offsets.length;
  const memory = runtime.instance.exports[runtime.api.metadata.memory_export || "memory"];
  const fn = runtime.instance.exports[runtime.entry.symbol];
  const inputs = runtime.entry.parameters.filter(p => p.role === "input");
  const outputs = runtime.entry.parameters.filter(p => p.role === "output");
  const elementBytes = runtime.api.metadata.element_bytes || 8;
  // A module that needed a baked table carries it at offset 0 and says how
  // far it reaches. Laying arrays out from 0 anyway overwrites the table --
  // and this module's activation IS that table, so the first tick destroys
  // the network it is about to ask. The scores then drift to garbage, the
  // trajectory stops advancing, and every subsequent frame is identical:
  // the loop keeps running at full speed while the picture sits still.
  const reserved = runtime.api.metadata.reserved_bytes || 0;
  const required = reserved + (inputs.length + outputs.length) * count * elementBytes;
  if (required > memory.buffer.byteLength) memory.grow(Math.ceil((required - memory.buffer.byteLength) / 65536));
  const View = runtime.api.metadata.value_type === "f32" ? Float32Array : Float64Array;
  const offsetsBytes = Array.from({length: inputs.length + outputs.length}, (_, i) => reserved + i * count * elementBytes);
  new View(memory.buffer, offsetsBytes[0], count).fill(feedbackState.travel);
  new View(memory.buffer, offsetsBytes[1], count).set(offsets.map(value => feedbackState.travel + value));
  fn(count, ...offsetsBytes);
  const scores = Array.from(new View(memory.buffer, offsetsBytes[inputs.length], count));
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
  feedbackState.scores = scores;
  feedbackState.speed = 0.45 + 1.8 * (1 - scores[0]) + 0.35 * best;
  const step = feedbackState.speed / Math.max(1, contract.fps || 120);
  if (!Number.isFinite(step)) {
    // Say so instead of advancing by NaN. A NaN trajectory fills the feed
    // with NaN, every escape count comes out equal, and the result looks
    // exactly like a page that has stopped redrawing.
    log("error", "feedback trajectory is not finite; holding position", {
      scores: scores, speed: feedbackState.speed
    });
    return;
  }
  feedbackState.travel += step;
}
}

function applyFeedbackFeed(feeds, count) {
  const contract = NETWORK.feedback;
  if (!contract) return;
  const index = inputs.findIndex(p => p.name === contract.travel_feed);
  if (index >= 0) feeds[index] = new Float64Array(count).fill(feedbackState.travel);
}
let running = false;

// --- shared-memory class-graph execution --------------------------------
// Every punch card imports the same WebAssembly.Memory. JavaScript composes
// byte offsets from the compiler-derived graph, calls the cards in schedule
// order, and reads only the public outputs. No live tensor is copied through
// JavaScript at a seam.
class ClassGraphRunner {
  constructor(manifest) {
    if (!manifest.shared_memory) throw new Error(
      "segmented manifest does not declare the shared-memory ABI"
    );
    this.manifest = manifest;
    this.modulesByName = new Map(manifest.modules.map(m => [m.name, m]));
    this.instances = new Map();
    this.offsetForInput = new Map();
    this.offsetForOutput = new Map();
    this.layoutCount = 0;
    const staticBytes = Number(manifest.shared_static_bytes || 0);
    this.memory = new WebAssembly.Memory({
      initial: Math.max(1, Math.ceil(staticBytes / 65536))
    });
    this.sourceOf = new Map();
    for (const edge of manifest.edges || []) {
      this.sourceOf.set(
        edge.to.module + "::" + edge.to.input,
        "out::" + edge.from.module + "::" + edge.from.output
      );
    }
    for (const [logicalName, targets] of Object.entries(manifest.logical_inputs || {})) {
      for (const [moduleName, inputName] of targets) {
        this.sourceOf.set(moduleName + "::" + inputName, "in::" + logicalName);
      }
    }
  }

  async instance(name) {
    if (this.instances.has(name)) return this.instances.get(name);
    const spec = this.modulesByName.get(name);
    markDeploymentNode(name, "downloading");
    let moduleBinary;
    if (spec.url) {
      const response = await fetch(spec.url);
      if (!response.ok) throw new Error(
        "failed to load private WASM region " + name + ": HTTP " + response.status
      );
      moduleBinary = await response.arrayBuffer();
    } else if (spec.wasm_base64) {
      const raw = atob(spec.wasm_base64);
      moduleBinary = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) moduleBinary[i] = raw.charCodeAt(i);
    } else {
      throw new Error("private WASM region " + name + " has no URL or bytes");
    }
    const memoryImport = spec.shared_memory_import || {module: "env", field: "memory"};
    const imports = {};
    imports[memoryImport.module] = {[memoryImport.field]: this.memory};
    const { instance } = await WebAssembly.instantiate(moduleBinary, imports);
    this.instances.set(name, instance);
    markDeploymentNode(name, "ready");
    return instance;
  }

  layout(count) {
    if (this.layoutCount === count) return;
    const elementBytes = Number(this.manifest.modules[0].element_bytes || 8);
    let cursor = Math.ceil(Number(this.manifest.shared_static_bytes || 0) / elementBytes) * elementBytes;
    this.offsetForInput.clear();
    this.offsetForOutput.clear();
    for (const logicalName of Object.keys(this.manifest.logical_inputs || {})) {
      this.offsetForInput.set(logicalName, cursor);
      cursor += count * elementBytes;
    }
    for (const spec of this.manifest.modules) {
      for (const outputName of spec.outputs) {
        this.offsetForOutput.set(spec.name + "::" + outputName, cursor);
        cursor += count * elementBytes;
      }
    }
    if (cursor > this.memory.buffer.byteLength) {
      this.memory.grow(Math.ceil((cursor - this.memory.buffer.byteLength) / 65536));
    }
    this.layoutCount = count;
  }

  offsetFor(source) {
    if (source.startsWith("in::")) return this.offsetForInput.get(source.slice(4));
    if (source.startsWith("out::")) return this.offsetForOutput.get(source.slice(5));
    throw new Error("unknown shared-memory slot " + source);
  }

  async call(moduleName, count) {
    const spec = this.modulesByName.get(moduleName);
    const instance = await this.instance(moduleName);
    const inputOffsets = spec.inputs.map(name => {
      const source = this.sourceOf.get(moduleName + "::" + name);
      if (!source) throw new Error(moduleName + " input " + name + " has no graph binding");
      return this.offsetFor(source);
    });
    const outputOffsets = spec.outputs.map(name =>
      this.offsetForOutput.get(moduleName + "::" + name)
    );
    const args = [count, ...inputOffsets, ...outputOffsets];
    markDeploymentNode(moduleName, "running");
    const started = performance.now();
    instance.exports[spec.entry](...args);
    markDeploymentNode(moduleName, "done", performance.now() - started);
  }

  async run(logicalInputs, count) {
    this.layout(count);
    const View = this.manifest.modules[0].value_type === "f32" ? Float32Array : Float64Array;
    for (const [logicalName, source] of Object.entries(logicalInputs)) {
      const offset = this.offsetForInput.get(logicalName);
      if (offset === undefined) continue;
      const target = new View(this.memory.buffer, offset, count);
      if (ArrayBuffer.isView(source) || Array.isArray(source)) {
        if (source.length === 1) target.fill(Number(source[0]));
        else if (source.length >= count) target.set(source.subarray ? source.subarray(0, count) : source.slice(0, count));
        else throw new Error(logicalName + " has " + source.length + " values for extent " + count);
      } else {
        target.fill(Number(source));
      }
    }
    const order = (this.manifest.schedule && this.manifest.schedule.nodes)
      ? this.manifest.schedule.nodes.slice().sort((a, b) => a.level - b.level || a.id.localeCompare(b.id)).map(n => n.id)
      : this.manifest.modules.map(module => module.name);
    for (const moduleName of order) await this.call(moduleName, count);
    return outputs.map(parameter => {
      const binding = this.manifest.logical_outputs[parameter.name];
      if (!binding) throw new Error("logical output " + parameter.name + " has no deployment binding");
      const offset = this.offsetForOutput.get(binding[0] + "::" + binding[1]);
      return new View(this.memory.buffer, offset, count).slice();
    });
  }
}

class ContiguousRunner {
  constructor(spec) { this.spec = spec; this.runtime = null; }
  async instance() {
    if (this.runtime) return this.runtime;
    markContiguousState("downloading");
    const response = await fetch(this.spec.url);
    if (!response.ok) throw new Error("failed to load contiguous WASM: HTTP " + response.status);
    const {instance} = await WebAssembly.instantiate(await response.arrayBuffer(), {});
    this.runtime = instance;
    markContiguousState("ready");
    return instance;
  }
  async run(logicalInputs, count) {
    const instance = await this.instance();
    const memory = instance.exports[this.spec.memory_export || "memory"];
    const elementBytes = Number(this.spec.element_bytes || 8);
    const View = this.spec.value_type === "f32" ? Float32Array : Float64Array;
    let cursor = Math.ceil(Number(this.spec.reserved_bytes || 0) / elementBytes) * elementBytes;
    const offsets = {};
    for (const name of [...this.spec.inputs, ...this.spec.outputs]) {
      offsets[name] = cursor; cursor += count * elementBytes;
    }
    if (cursor > memory.buffer.byteLength) memory.grow(Math.ceil((cursor - memory.buffer.byteLength) / 65536));
    for (const name of this.spec.inputs) {
      const source = logicalInputs[name];
      const target = new View(memory.buffer, offsets[name], count);
      if (source.length === 1) target.fill(Number(source[0]));
      else target.set(source.subarray ? source.subarray(0, count) : source.slice(0, count));
    }
    markContiguousState("running");
    const started = performance.now();
    instance.exports[this.spec.entry](count, ...this.spec.inputs.map(n => offsets[n]), ...this.spec.outputs.map(n => offsets[n]));
    markContiguousState("done", performance.now() - started);
    return this.spec.outputs.map(name => new View(memory.buffer, offsets[name], count).slice());
  }
}

// One pass through every private module in the graph, using the full arrays
// supplied through the logical program's public input contract.
const classGraphRunner = CLASS_GRAPH ? new ClassGraphRunner(CLASS_GRAPH) : null;
const contiguousRunner = CLASS_GRAPH && CLASS_GRAPH.contiguous
  ? new ContiguousRunner(CLASS_GRAPH.contiguous) : null;
let activeExecutionMode = "staged";

async function computeViaSelectedRunner(feeds, count) {
  const logicalInputs = {};
  for (const logicalName of Object.keys(CLASS_GRAPH.logical_inputs || {})) {
    const paramIndex = inputs.findIndex(p => p.name === logicalName);
    if (paramIndex < 0) throw new Error("logical input " + logicalName + " is not in the API");
    logicalInputs[logicalName] = feeds[paramIndex];
  }
  if (activeExecutionMode === "contiguous") {
    if (!contiguousRunner) throw new Error("no contiguous compile is published");
    return contiguousRunner.run(logicalInputs, count);
  }
  return classGraphRunner.run(logicalInputs, count);
}

// The segmented deployment follows the same full-domain run loop as a
// monolithic module.  Animation, feedback, rendering and timing all remain
// properties of the one logical program exposed by the page.
async function runClassGraphMode() {
  if (running) {
    running = false;
    return;
  }
  try {
    const d = domain();
    const anyExpression = inputs.some(p => $("mode_" + p.name).value === "expression");
    const anyGaussian = inputs.some(p => $("mode_" + p.name).value === "gaussian");
    const anyNetwork = inputs.some(p => $("mode_" + p.name).value === "network");
    const renderFps = Math.max(1, Number((NETWORK.feedback || {}).render_fps) || 24);
    const feedbackTicks = Math.max(
      1,
      Math.round((Number((NETWORK.feedback || {}).fps) || 120) / renderFps)
    );
    await advanceFeedback(feedbackTicks);
    let activeFeeds = inputs.map(p => feedValues(p, d.n, d, frameIndex));
    applyFeedbackFeed(activeFeeds, d.n);
    const count = anyExpression
      ? d.n
      : (activeFeeds.length
          ? Math.min(...activeFeeds.map(feed => feed.length))
          : d.n);
    if (!count) throw new Error("no elements to run");
    const repeats = Math.max(0, Number($("repeats").value) | 0);
    const continuous = repeats === 0;
    const animated = (continuous || repeats > 1) &&
      (anyExpression || anyGaussian || anyNetwork);
    const timings = [];
    if (animated) {
      document.querySelectorAll(".tab").forEach(tab =>
        tab.setAttribute("aria-selected", String(tab.dataset.view === "image"))
      );
      renderActiveTab();
    }
    running = true;
    $("run").textContent = "Stop";
    log("info", activeExecutionMode + " deployment", {
      modules: CLASS_GRAPH.modules.length,
      elements: count,
    });
    for (let r = 0; running && (continuous || r < repeats); r++) {
      if (r > 0 && animated) {
        frameIndex = r;
        await advanceFeedback(feedbackTicks);
        activeFeeds = inputs.map(p => feedValues(p, count, d, frameIndex));
        applyFeedbackFeed(activeFeeds, count);
      }
      const frameStarted = performance.now();
      const t0 = performance.now();
      const result = await computeViaSelectedRunner(activeFeeds, count);
      timings.push(performance.now() - t0);
      lastOutputs = outputs.map((p, index) => ({
        name: p.name,
        values: result[index],
      }));
      if (animated) {
        if ((r % 15) === 0) reportTimings(timings, count);
        renderActiveTab();
        renderNetworkStats(activeFeeds);
        await presented();
        const remaining = 1000 / renderFps - (performance.now() - frameStarted);
        await new Promise(resolve => setTimeout(resolve, Math.max(0, remaining)));
      } else if (!continuous && repeats > 1 && (r % 200) === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    running = false;
    $("run").textContent = RUN_LABEL;
    frameIndex = 0;
    const ordered = timings.slice().sort((a, b) => a - b);
    const elapsed = ordered[ordered.length >> 1];
    reportTimings(timings, count);
    renderActiveTab();
    renderNetworkStats(activeFeeds);
    setStatus(
      "ran " + count + " elements in " + elapsed.toFixed(3) +
      " ms (" + (activeExecutionMode === "staged" ? "shared-memory staged" : "contiguous") + " WASM)",
      "good"
    );
    log("ok", "segmented run complete", {
      median_ms: Number(elapsed.toFixed(4)),
      elements: count,
      modules: CLASS_GRAPH.modules.length,
    });
  } catch (err) {
    running = false;
    $("run").textContent = RUN_LABEL;
    setStatus(String(err), "bad");
    log("error", err && err.message ? err.message : err,
        { stack: err && err.stack ? err.stack.split("\n")[0] : null });
  }
}

async function run() {
  if (CLASS_GRAPH) { await runClassGraphMode(); return; }
  if (running) {            // the button is a toggle while a run is live
    running = false;
    return;
  }
  if (!moduleBytes) { setStatus("No .wasm loaded yet.", "bad"); log("warn", "run with no module"); return; }
  try {
    log("progress", "instantiating", { done: 1, total: 4 });
    log("info", "instantiating", { bytes: moduleBytes.length });
    const { instance } = await WebAssembly.instantiate(moduleBytes, {});
    const memory = instance.exports[API.metadata.memory_export || "memory"];
    const fn = instance.exports[entry.symbol];
    if (!fn) throw new Error("export '" + entry.symbol + "' not found");

    const d = domain();
    const anyExpression = inputs.some(p => $("mode_" + p.name).value === "expression");
    const anyGaussian = inputs.some(p => $("mode_" + p.name).value === "gaussian");
    const anyNetwork = inputs.some(p => $("mode_" + p.name).value === "network");
    const renderFps = Math.max(1, Number((NETWORK.feedback || {}).render_fps) || 24);
    const feedbackTicks = Math.max(1, Math.round((Number((NETWORK.feedback || {}).fps) || 120) / renderFps));
    await advanceFeedback(feedbackTicks);
    const feeds = inputs.map(p => feedValues(p, d.n, d, frameIndex));
    applyFeedbackFeed(feeds, d.n);
    let activeFeeds = feeds;
    // An expression covers the whole grid; literal values only go as far as
    // the shortest list supplied.
    const count = anyExpression
      ? d.n
      : (feeds.length ? Math.min(...feeds.map(f => f.length)) : d.n);
    if (!count) throw new Error("no elements to run");
    log("info", "domain", { width: d.w, height: d.h, elements: count,
                            generated: anyExpression, frame: frameIndex });

    // The caller owns memory, so the layout is decided here: every array
    // gets its own contiguous block, feeds first and then outputs.
    // Same rule as the feedback module: start past anything the program
    // baked into its own memory. Zero for a program that needed no table.
    const reservedBytes = API.metadata.reserved_bytes || 0;
    const need = reservedBytes + (inputs.length + outputs.length) * count * bytes;
    const have = memory.buffer.byteLength;
    if (need > have) {
      memory.grow(Math.ceil((need - have) / 65536));
    }
    const View = isF32 ? Float32Array : Float64Array;
    const offsets = [];
    let cursor = reservedBytes;
    for (let i = 0; i < inputs.length + outputs.length; i++) {
      offsets.push(cursor);
      cursor += count * bytes;
    }
    inputs.forEach((p, i) => {
      new View(memory.buffer, offsets[i], count).set(feeds[i].slice(0, count));
    });

    log("progress", "writing feeds", { done: 2, total: 4 });
    const args = [count, ...offsets];
    // The exact call, recorded: argument order and the memory offsets it
    // computed are the two things most likely to be wrong, and the two least
    // visible from a wrong answer alone.
    log("call", entry.symbol + "(" + args.join(", ") + ")", {
      count: count, offsets: offsets, elementBytes: bytes,
      memoryPages: memory.buffer.byteLength / 65536
    });
    log("progress", "executing", { done: 3, total: 4 });
    // Steady state, not one sample: a single call measures instantiation,
    // first touch of the memory, and whatever the JIT had not done yet.
    // Each repeat is also a frame -- feeds are regenerated for it, so a
    // gaussian feed redraws and an expression sees a new `t`. Only the
    // kernel call is timed, so the numbers still describe the kernel rather
    // than the feed generation around it.
    // 0 (or blank) means keep going until stopped.
    const repeats = Math.max(0, Number($("repeats").value) | 0);
    const continuous = repeats === 0;
    const timings = [];
    // A routed network is a time-varying source too.  Without this term a page
    // set entirely to network feeds runs its first frame but never reaches the
    // redraw/yield path below.
    const animated = (continuous || repeats > 1) && (anyExpression || anyGaussian || anyNetwork);
    if (animated) {
      document.querySelectorAll(".tab").forEach(tab => tab.setAttribute("aria-selected", String(tab.dataset.view === "image")));
      renderActiveTab();
    }
    running = true;
    $("run").textContent = "Stop";
    for (let r = 0; running && (continuous || r < repeats); r++) {
      if (r > 0 && animated) {
        frameIndex = r;
        await advanceFeedback(feedbackTicks);
        const refreshed = inputs.map(p => feedValues(p, count, d, frameIndex));
        applyFeedbackFeed(refreshed, count);
        activeFeeds = refreshed;
        inputs.forEach((p, i) => {
          new View(memory.buffer, offsets[i], count).set(
            refreshed[i].slice(0, count));
        });
      }
      const frameStarted = performance.now();
      const t0 = frameStarted;
      fn(...args);
      timings.push(performance.now() - t0);
      if (animated) {
        // Read this frame out and paint it before the next one starts,
        // otherwise the loop finishes and only the last frame is ever seen.
        lastOutputs = outputs.map((p, i) => ({
          name: p.name,
          values: new View(memory.buffer, offsets[inputs.length + i], count)
        }));
        if ((r % 15) === 0) reportTimings(timings, count);
        // The redraw is synchronous and inside the computation loop: the
        // loop calls it directly and does not continue until it has
        // finished. Handing it to a frame callback deferred it out of the
        // loop, which is what left the compute running with nothing being
        // drawn.
        renderActiveTab();
        renderNetworkStats(activeFeeds);
        await presented();
        const remaining = 1000 / renderFps - (performance.now() - frameStarted);
        // Always yield, even when the frame already overran its budget --
        // which on a phone is every frame. Without this the only yield was
        // the frame above, and a negative `remaining` meant none at all.
        await new Promise(resolve => setTimeout(resolve, Math.max(0, remaining)));
      } else if (!continuous && repeats > 1 && (r % 200) === 0) {
        // Even a non-animated sweep should not freeze the page.
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    running = false;
    $("run").textContent = RUN_LABEL;
    frameIndex = 0;
    const ordered = timings.slice().sort((a, b) => a - b);
    const elapsed = ordered[ordered.length >> 1];
    reportTimings(timings, count);
    if (timings.length > 1) {
      log("profile", "steady state over " + timings.length + " runs", {
        median_ms: Number(ordered[ordered.length >> 1].toFixed(4)),
        min_ms: Number(ordered[0].toFixed(4)),
        max_ms: Number(ordered[ordered.length - 1].toFixed(4)),
        elements: count
      });
    }
    log("ok", "returned in " + elapsed.toFixed(3) + " ms");

    lastOutputs = outputs.map((p, i) => ({
      name: p.name,
      values: new View(memory.buffer, offsets[inputs.length + i], count)
    }));
    log("progress", "reading outputs", { done: 4, total: 4 });
    renderActiveTab();
    renderNetworkStats(feeds);
    setStatus("ran " + count + " elements in " + elapsed.toFixed(3) + " ms", "good");
  } catch (err) {
    running = false;
    $("run").textContent = RUN_LABEL;
    $("raw").textContent = "";
    setStatus(String(err), "bad");
    log("error", err && err.message ? err.message : err,
        { stack: err && err.stack ? err.stack.split("\n")[0] : null });
  }
}

// --- output views -------------------------------------------------------
// The numbers a program returns are just numbers; how to *look* at them is
// the caller's question, so it is a tab rather than a property of the
// program. "raw" is the numbers; "image" reads the same buffer as a picture.

let lastOutputs = null;

function renderWebGLPalette(canvas, values, w, h, lo, span, invert) {
  // Resizing a canvas clears its WebGL drawing buffer and state.  Do it before
  // retrieving/caching the context so an animated frame can reuse its GPU objects.
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h; canvas._turingWebGL = null;
  }
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, preserveDrawingBuffer: false });
  if (!gl) return false;
  let state = canvas._turingWebGL;
  if (!state) {
    const vertex = "#version 300 es\nin vec2 p; out vec2 uv; void main(){ uv=(p+1.0)*0.5; gl_Position=vec4(p,0,1); }";
    const fragment = "#version 300 es\nprecision highp float; uniform sampler2D scalar; uniform bool invert; in vec2 uv; out vec4 outColor; void main(){ float t=texture(scalar,uv).r; if(invert)t=1.0-t; vec3 c=vec3(min(1.0,t*1.6),min(1.0,t*t*1.4),min(1.0,max(0.0,0.35+t*0.65-t*t))); outColor=vec4(c,1.0); }";
    const compile = (kind, source) => { const shader = gl.createShader(kind); gl.shaderSource(shader, source); gl.compileShader(shader); return shader; };
    const program = gl.createProgram(); gl.attachShader(program, compile(gl.VERTEX_SHADER, vertex)); gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragment)); gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return false;
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const buffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
    const location = gl.getAttribLocation(program, "p"); gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
    const texture = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, texture); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    state = canvas._turingWebGL = { program, vao, texture, invert: gl.getUniformLocation(program, "invert") };
  }
  const scalar = new Uint8Array(w * h);
  for (let i = 0; i < scalar.length; i++) scalar[i] = Math.max(0, Math.min(255, Math.round(255 * ((values[i] - lo) / span))));
  gl.viewport(0, 0, w, h); gl.useProgram(state.program); gl.bindVertexArray(state.vao); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, state.texture); gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1); gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, scalar); gl.uniform1i(state.invert, invert ? 1 : 0); gl.drawArrays(gl.TRIANGLES, 0, 3);
  // Wait for the draw to actually complete. drawArrays only queues commands;
  // returning here lets the loop start the next compute while this frame is
  // still sitting in the queue, so nothing reaches the screen. finish()
  // blocks until the GPU has drained it.
  gl.finish();
  return true;
}
function renderRaw() {
  $("raw").textContent = (lastOutputs || [])
    .map(o => o.name + ": [" + o.values.join(", ") + "]")
    .join("\n");
}

function renderChannels(canvas, note) {
  // A program returning three outputs has already said what each pixel is.
  // Copy them; do not interpret them. The ramp below exists for programs
  // that return one number and leave the looking-at to the caller -- a
  // program that produced its own channels has answered that question, and
  // a second opinion from the shell would only corrupt it.
  const [red, green, blue] = lastOutputs;
  const count = red.values.length;
  if (green.values.length !== count || blue.values.length !== count) return false;
  const d = domain();
  let w = d.w, h = d.h;
  if (w * h !== count) { w = Math.round(Math.sqrt(count)) || 1; h = Math.ceil(count / w); }
  if (w * h !== count) return false;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return false;
  const image = ctx.createImageData(w, h);
  const invert = $("img_invert").checked;
  for (let i = 0; i < count; i++) {
    const clamp = (v) => Math.max(0, Math.min(255, Math.round(invert ? 255 - v : v)));
    image.data[i*4+0] = clamp(red.values[i]);
    image.data[i*4+1] = clamp(green.values[i]);
    image.data[i*4+2] = clamp(blue.values[i]);
    image.data[i*4+3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  note.textContent = w + "x" + h + ", " + lastOutputs.length +
    " outputs read as " + red.name + "/" + green.name + "/" + blue.name +
    " channels, exactly as the program produced them";
  return true;
}

function renderImage() {
  const canvas = $("canvas");
  const note = $("imgnote");
  if (!lastOutputs || !lastOutputs.length) { note.textContent = "Run first."; return; }
  if (lastOutputs.length >= 3 && renderChannels(canvas, note)) return;
  const values = lastOutputs[0].values;
  const d = domain();
  let w = d.w, h = d.h;
  if (w * h !== values.length) {
    // The run did not cover the grid (literal feeds, or a short list), so
    // the stated geometry would lie about the picture.
    w = 0; h = 0;
  }
  if (!w || !h) {
    // Default to the squarest rectangle that fits, so a run with no stated
    // geometry still shows something honest rather than nothing.
    w = Math.round(Math.sqrt(values.length)) || 1;
    h = Math.ceil(values.length / w);
  }
  const invert = $("img_invert").checked;
  let lo = Infinity, hi = -Infinity;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const span = (hi - lo) || 1;

  if (renderWebGLPalette(canvas, values, w, h, lo, span, invert)) {
    note.textContent = w + "x" + h + ", raw scalar field rendered by WebGL RGB palette, range " + lo.toPrecision(4) + " to " + hi.toPrecision(4);
    return;
  }
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const raw = i < values.length ? values[i] : lo;
    let t = (raw - lo) / span;
    if (invert) t = 1 - t;
    // A simple perceptual ramp: dark blue -> orange -> white. Enough to see
    // structure without pretending to be a colour science module.
    const r = Math.round(255 * Math.min(1, Math.max(0, t * 1.6)));
    const g = Math.round(255 * Math.min(1, Math.max(0, t * t * 1.4)));
    const b = Math.round(255 * Math.min(1, Math.max(0, 0.35 + t * 0.65 - t * t)));
    image.data[i*4+0] = r; image.data[i*4+1] = g; image.data[i*4+2] = b;
    image.data[i*4+3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  note.textContent = w + "x" + h + ", raw scalar field rendered into RGB canvas pixels, range " +
    lo.toPrecision(4) + " to " + hi.toPrecision(4);
}

function renderActiveTab() {
  const active = document.querySelector('.tab[aria-selected="true"]').dataset.view;
  document.querySelectorAll(".tabview").forEach(v => {
    v.hidden = v.dataset.view !== active;
  });
  if (active === "raw") renderRaw(); else renderImage();
}

function wireFilters() {
  document.querySelectorAll(".filters input").forEach(box => {
    box.addEventListener("change", () => {
      const on = new Set(
        Array.from(document.querySelectorAll(".filters input"))
          .filter(b => b.checked).map(b => b.dataset.kind));
      document.querySelectorAll(".logline").forEach(line => {
        const kind = line.className.split(" ")[1];
        line.hidden = !on.has(kind === "info" ? "log" : kind);
      });
    });
  });
}

let activeGraphView = "original";
let graphLayout = null;
let graphFrame = null;
const graphPulses = new Map();

function hueRgb(hue) {
  const h = ((hue % 360) + 360) % 360 / 60;
  const x = 1 - Math.abs(h % 2 - 1);
  const table = [[1,x,0],[x,1,0],[0,1,x],[0,x,1],[x,0,1],[1,0,x]];
  const rgb = table[Math.floor(h) % 6];
  return rgb.map(value => Math.round((0.18 + value * 0.82) * 255));
}

function mixedIdentityRgb(contributors) {
  const colours = (contributors || []).map(identity =>
    hueRgb((GRAPH_VIEWS.identities[identity] || {hue: 210}).hue)
  );
  if (!colours.length) return [90, 110, 145];
  return [0, 1, 2].map(channel => Math.round(
    colours.reduce((sum, colour) => sum + colour[channel], 0) / colours.length
  ));
}

// Rolling phosphor integrator. Calls faster than the display refresh are not
// dropped: every completed region deposits energy, and the exponentially
// decaying window determines the node's visible colour on the next redraw.
function phosphorColor(node, now) {
  const decay = Math.max(80, Number($("graph-decay") && $("graph-decay").value) || 1200);
  const pulses = graphPulses.get(String(node.id)) || [];
  let energy = 0;
  const live = [];
  for (const pulse of pulses) {
    const age = Math.max(0, now - pulse.at);
    if (age > decay * 7) continue;
    energy += pulse.energy * Math.exp(-age / decay);
    live.push(pulse);
  }
  if (live.length !== pulses.length) graphPulses.set(String(node.id), live);
  const base = mixedIdentityRgb(node.contributors);
  const glow = 1 - Math.exp(-energy);
  return {
    rgb: base.map(value => Math.min(255, Math.round(value * (0.28 + glow * 0.9)))),
    alpha: 0.28 + glow * 0.72,
    glow: glow,
    active: live.length > 0,
  };
}

function pulseGraphNodes(nodeIds, elapsedMs) {
  const at = performance.now();
  const energy = Math.max(0.14, Math.log1p(Math.max(0, elapsedMs || 0)) * 0.5);
  for (const nodeId of nodeIds || []) {
    const key = String(nodeId);
    const pulses = graphPulses.get(key) || [];
    pulses.push({at: at, energy: energy});
    if (pulses.length > 96) pulses.splice(0, pulses.length - 96);
    graphPulses.set(key, pulses);
  }
  if (!graphFrame) graphFrame = requestAnimationFrame(drawProcessGraph);
}

function prepareProcessGraph() {
  const canvas = $("process-graph-canvas");
  const view = GRAPH_VIEWS.views && GRAPH_VIEWS.views[activeGraphView];
  if (!canvas || !view) return;
  const groups = Math.max(1, view.groups || 1);
  const levels = Math.max(1, (view.level_max || 0) - (view.level_min || 0) + 1);
  const width = Math.max(760, groups * 180);
  const height = Math.max(420, Math.min(30000, levels * 8 + 36));
  canvas.width = width; canvas.height = height;
  const buckets = new Map();
  const positions = new Map();
  for (const node of view.nodes) {
    const key = node.level + "::" + node.group;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(node);
  }
  for (const [key, nodes] of buckets) {
    const [level, group] = key.split("::").map(Number);
    nodes.forEach((node, index) => {
      const span = 150;
      const x = 18 + group * 180 + (index + 1) * span / (nodes.length + 1);
      const y = 25 + (level - (view.level_min || 0)) * ((height - 40) / levels);
      positions.set(String(node.id), {x, y, node});
    });
  }
  graphLayout = {view, positions};
  drawProcessGraph(performance.now());
}

function drawProcessGraph(now) {
  graphFrame = null;
  const canvas = $("process-graph-canvas");
  if (!canvas || !graphLayout) return;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#05070c"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(150,170,210,.10)"; ctx.lineWidth = 1;
  ctx.beginPath();
  for (const [left, right] of graphLayout.view.edges) {
    const a = graphLayout.positions.get(String(left));
    const b = graphLayout.positions.get(String(right));
    if (!a || !b) continue;
    ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
  }
  ctx.stroke();
  let stillGlowing = false;
  for (const {x, y, node} of graphLayout.positions.values()) {
    const colour = phosphorColor(node, now);
    stillGlowing = stillGlowing || colour.active;
    ctx.fillStyle = "rgba(" + colour.rgb.join(",") + "," + colour.alpha.toFixed(3) + ")";
    if (colour.glow > .08) {
      ctx.shadowColor = "rgb(" + colour.rgb.join(",") + ")";
      ctx.shadowBlur = 2 + colour.glow * 10;
    } else ctx.shadowBlur = 0;
    ctx.beginPath(); ctx.arc(x, y, 2.2 + colour.glow * 2.8, 0, Math.PI * 2); ctx.fill();
  }
  ctx.shadowBlur = 0;
  if (stillGlowing) graphFrame = requestAnimationFrame(drawProcessGraph);
}

function wireProcessGraphCanvas() {
  document.querySelectorAll(".graph-view-button").forEach(button => {
    button.addEventListener("click", () => {
      activeGraphView = button.dataset.graphView;
      document.querySelectorAll(".graph-view-button").forEach(candidate =>
        candidate.setAttribute("aria-pressed", String(candidate === button))
      );
      prepareProcessGraph();
    });
  });
  const decay = $("graph-decay");
  if (decay) decay.addEventListener("input", () => drawProcessGraph(performance.now()));
  const canvas = $("process-graph-canvas");
  if (canvas) canvas.addEventListener("click", event => {
    if (!graphLayout) return;
    let best = null, distance = Infinity;
    for (const position of graphLayout.positions.values()) {
      const d = Math.hypot(position.x - event.offsetX, position.y - event.offsetY);
      if (d < distance) { distance = d; best = position.node; }
    }
    if (best && distance < 12) {
      const labels = (best.contributors || []).map(identity =>
        (GRAPH_VIEWS.identities[identity] || {label: identity}).label
      );
      $("graph-node-inspector").textContent = best.id + " · " + best.type + " · level " +
        best.level + " · group " + best.group + " · " + best.label +
        " · contributors [" + labels.join(", ") + "]";
    }
  });
  prepareProcessGraph();
}

function renderGraph() {
  const target = document.getElementById("graph");
  if (!target) return;
  let html = "";
  if (GRAPH_VIEWS && GRAPH_VIEWS.views) {
    const original = GRAPH_VIEWS.views.original || {nodes: []};
    const reduced = GRAPH_VIEWS.views.reduced || {nodes: []};
    html += '<div class="graph-toolbar">' +
      '<button class="graph-view-button" data-graph-view="original" aria-pressed="true">Original ProcessGraph · ' + original.nodes.length + '</button>' +
      '<button class="graph-view-button" data-graph-view="reduced" aria-pressed="false">Reduced ProcessGraph · ' + reduced.nodes.length + '</button>' +
      '<label class="meta">phosphor decay <input id="graph-decay" type="range" min="80" max="5000" value="1200"> rolling window</label>' +
      '</div><div class="meta">Rows preserve schedule levels; columns preserve groups. Region and feed identities mix at every node. Runtime profiles deposit coloured energy into an exponential rolling window, so sub-frame calls remain visible.</div>' +
      '<div class="graph-scroll"><canvas id="process-graph-canvas"></canvas></div>' +
      '<div id="graph-node-inspector" class="note node-detail">Click a node to inspect its contributors.</div>';
  } else if (GRAPH && GRAPH.nodes) {
    const hist = Object.entries(GRAPH.histogram || {})
      .map(([k, v]) => '<span class="chip">' + k + " x" + v + "</span>").join("");
    const rows = (GRAPH.table || []).map(n =>
      '<div class="kv"><b>' + n.id + "</b><span>" + n.type + "</span><span>" +
      (n.label || "") + "</span><span class=meta>" +
      (n.parents.length ? "&larr; " + n.parents.join(", ") : "") + "</span></div>"
    ).join("");
    html += '<div class="meta">Logical ProcessGraph</div>' +
      '<div class="kv"><b>nodes</b><span>' + GRAPH.nodes + "</span></div>" +
      '<div class="kv"><b>edges</b><span>' + GRAPH.edges + "</span></div>" +
      "<div>" + hist + "</div>" +
      (GRAPH.truncated ? '<div class="meta">table truncated</div>' : "") +
      rows;
  }
  if (CLASS_GRAPH && CLASS_GRAPH.schedule) {
    const moduleByName = new Map(CLASS_GRAPH.modules.map(module => [module.name, module]));
    const levels = new Map();
    for (const node of CLASS_GRAPH.schedule.nodes) {
      if (!levels.has(node.level)) levels.set(node.level, []);
      levels.get(node.level).push(node);
    }
    const deploymentRows = Array.from(levels).sort((a, b) => a[0] - b[0]).map(([level, nodes]) => {
      const cards = nodes.sort((a, b) => a.id.localeCompare(b.id)).map(node => {
        const module = moduleByName.get(node.id) || {};
        return '<div class="deployment-node" data-module="' + node.id + '" data-state="idle" tabindex="0">' +
          '<b>' + node.id + '</b><span class="node-state">idle · ' +
          (module.operation_count || 0) + ' ops</span></div>';
      }).join("");
      return '<div class="schedule-level"><div class="schedule-level-label">level ' + level +
        '</div>' + cards + '</div>';
    }).join("");
    html += '<div class="meta" style="margin-top:.7rem">Live deployment schedule: ' +
      CLASS_GRAPH.modules.length + ' punch cards sharing one global WASM memory. Click a node for its ABI.</div>' +
      deploymentRows + '<div id="node-detail" class="note node-detail">Select a punch card.</div>';
  }
  target.innerHTML = html;
  wireProcessGraphCanvas();
  document.querySelectorAll(".deployment-node").forEach(node => {
    const show = () => {
      const spec = CLASS_GRAPH.modules.find(module => module.name === node.dataset.module) || {};
      $("node-detail").textContent = spec.name + " · " + (spec.operation_count || 0) +
        " operations · inputs [" + (spec.inputs || []).join(", ") + "] · outputs [" +
        (spec.outputs || []).join(", ") + "] · ProcessGraph nodes [" +
        (spec.node_ids || []).join(", ") + "]";
    };
    node.addEventListener("click", show);
    node.addEventListener("keydown", event => { if (event.key === "Enter") show(); });
  });
}

function markDeploymentNode(moduleName, state, elapsedMs) {
  const node = document.querySelector('.deployment-node[data-module="' + CSS.escape(moduleName) + '"]');
  if (!node) return;
  node.dataset.state = state;
  const label = node.querySelector(".node-state");
  const calls = Number(node.dataset.calls || 0) + (state === "done" ? 1 : 0);
  if (state === "done") node.dataset.calls = String(calls);
  const timing = elapsedMs === undefined ? "" : " · " + elapsedMs.toFixed(3) + " ms";
  if (label) label.textContent = state + timing + (calls ? " · " + calls + " calls" : "");
  if (state === "done") {
    const spec = CLASS_GRAPH.modules.find(module => module.name === moduleName);
    if (spec) pulseGraphNodes(spec.node_ids, elapsedMs);
  }
}

function markContiguousState(state, elapsedMs) {
  const label = $("contiguous-state");
  if (!label) return;
  label.textContent = state + (elapsedMs === undefined ? "" : " · " + elapsedMs.toFixed(3) + " ms");
}

function wireSourceTabs() {
  document.querySelectorAll(".srctab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".srctab").forEach(t =>
        t.setAttribute("aria-selected", String(t === tab)));
      document.querySelectorAll(".srcview").forEach(v => {
        v.hidden = v.dataset.lang !== tab.dataset.lang;
      });
    });
  });
  document.querySelectorAll(".download-source").forEach(button => {
    button.addEventListener("click", async () => {
      const descriptor = SOURCE_DOWNLOADS.find(source => source.language === button.dataset.lang);
      if (!descriptor || !descriptor.url) return;
      button.disabled = true;
      const old = button.textContent;
      button.textContent = "Downloading…";
      try {
        const response = await fetch(descriptor.url);
        if (!response.ok) throw new Error("HTTP " + response.status);
        const blob = await response.blob();
        const href = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = href; link.download = descriptor.filename || descriptor.url.split("/").pop();
        document.body.appendChild(link); link.click(); link.remove();
        setTimeout(() => URL.revokeObjectURL(href), 1000);
        log("ok", "source downloaded", {language: descriptor.language, bytes: blob.size});
      } catch (error) {
        log("error", "source download failed", {language: descriptor.language, message: error.message});
      } finally {
        button.disabled = false; button.textContent = old;
      }
    });
  });
}

function wireExecutionModes() {
  document.querySelectorAll(".execution-mode").forEach(button => {
    button.addEventListener("click", () => {
      if (running) return;
      activeExecutionMode = button.dataset.mode;
      document.querySelectorAll(".execution-mode").forEach(candidate =>
        candidate.setAttribute("aria-pressed", String(candidate === button))
      );
      setStatus(activeExecutionMode === "staged"
        ? "staged punch cards selected; regions download on first use"
        : "contiguous compile selected; full module downloads on first use", "good");
    });
  });
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(t =>
        t.setAttribute("aria-selected", String(t === tab)));
      renderActiveTab();
    });
  });
  ["dom_w", "dom_h", "img_invert"].forEach(id =>
    $(id).addEventListener("change", () => renderActiveTab()));
  document.querySelectorAll("select[id^='mode_']").forEach(select => {
    select.addEventListener("change", () => {
      const name = select.id.slice(5);
      $("row_values_" + name).hidden = select.value !== "values";
      $("row_expr_" + name).hidden = select.value !== "expression" && select.value !== "network";
      $("row_gauss_" + name).hidden = select.value !== "gaussian";
    });
    select.dispatchEvent(new Event("change"));
  });
}

function renderNetworkStats(feeds) {
  const pane = $("networkstats");
  if (!pane) return;
  const route = NETWORK.routes && NETWORK.routes[0];
  if (!route || !feeds || !feeds.length) {
    pane.textContent = "No feedback route is attached to this module.";
    return;
  }
  const index = inputs.findIndex(p => p.name === route.feed);
  const values = index < 0 ? null : feeds[index];
  if (!values || !values.length) {
    pane.textContent = "Route " + route.feed + " is awaiting input.";
    return;
  }
  let low = Infinity, high = -Infinity, total = 0;
  for (const value of values) { low = Math.min(low, value); high = Math.max(high, value); total += value; }
  const output = lastOutputs[0] && lastOutputs[0].values || [];
  pane.textContent = NETWORK.name + " · scores [" + feedbackState.scores.map(v => v.toFixed(3)).join(", ") + "] · speed " + feedbackState.speed.toFixed(3) + " · travel " + feedbackState.travel.toFixed(3) + " · " + route.feed + " → " + route.effect +
    " · " + values.length + " samples · mean " + (total / values.length).toFixed(4) +
    " · range [" + low.toFixed(4) + ", " + high.toFixed(4) + "]" +
    " · returned " + output.length + " image values";
}
function wireFilePicker() {
  const picker = $("picker");
  if (!picker) return;
  picker.addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    moduleBytes = new Uint8Array(await file.arrayBuffer());
    $("run").disabled = false;
    setStatus("loaded " + file.name + " (" + moduleBytes.length + " bytes)", "good");
  });
}

wireFilePicker();
wireTabs();
$("run").addEventListener("click", run);
$("copyapi").addEventListener("click", () => {
  navigator.clipboard.writeText($("apiyaml").value);
  log("info", "API descriptor copied");
});
$("apiyaml").addEventListener("input", () => {
  // The page binds against the JSON it was generated with, so an edit here
  // is not live yet. Say that at the moment it starts to matter, rather
  // than letting someone type into a field that quietly does nothing.
  $("apistatus").textContent = "edited -- not applied (submission not wired up)";
  $("apistatus").className = "out warn";
});

$("copylog").addEventListener("click", () => {
  navigator.clipboard.writeText(JSON.stringify(LOG, null, 2));
  log("info", "log copied to clipboard");
});
// Show what happened before the page existed, in order, then continue in
// the same pane.
(BUILD_TELEMETRY.records || []).forEach(r => {
  const pane = document.getElementById("log");
  if (!pane) return;
  const line = document.createElement("div");
  line.className = "logline " + r.kind;
  line.textContent = (r.at_ns / 1e6).toFixed(1) + "ms  BUILD " + r.kind.toUpperCase() +
    "  " + (r.path ? "[" + r.path + "] " : "") + r.message +
    (Object.keys(r.detail || {}).length ? "  " + JSON.stringify(r.detail) : "");
  pane.appendChild(line);
});
wireFilters();
wireSourceTabs();
wireExecutionModes();
renderGraph();

log("info", "shell ready", {
  entry: entry.symbol,
  parameters: params.length,
  valueType: API.metadata.value_type,
  embedded: Boolean(WASM_BASE64)
});
if (moduleBytes) setStatus("module embedded, ready", "good");
else if (CLASS_GRAPH) setStatus("manifest ready; staged artifacts are still unloaded", "good");
"""


@dataclass(frozen=True)
class HtmlShell:
    """A generated page, and where it came from."""

    name: str
    html: str
    embedded: bool

    def write(self, directory: str | Path) -> Path:
        path = Path(directory) / f"{self.name}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.html, encoding="utf-8")
        return path


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _signature_rows(parameters: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for parameter in parameters:
        detail = f"{parameter['role']} &middot; {parameter['dtype']} &middot; {parameter['passing']}"
        if parameter.get("extent"):
            detail += f" &middot; extent {parameter['extent']}"
        rows.append(
            '<div class="row">'
            f'<div class="name">{_escape(str(parameter["name"]))}</div>'
            f'<div class="meta grow">{detail}</div>'
            "</div>"
        )
    return "\n".join(rows)


def _input_rows(
    parameters: Sequence[Mapping[str, Any]],
    feed_expressions: Mapping[str, str] | None = None,
    network_routes: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """One row per feed, each able to be literal values or an expression.

    A kernel's feeds are usually a function of position, so an expression is
    offered alongside literal values: it is evaluated per element with the
    grid coordinates in scope (``i``, ``x``, ``y``, ``w``, ``h``). Typing a
    quarter of a million numbers into a text field is not a control surface.
    """

    expressions = dict(feed_expressions or {})
    routes = dict(network_routes or {})
    feeds = [p for p in parameters if p["role"] == "input"]
    rows = []
    for parameter in feeds:
        name = str(parameter["name"])
        expression = expressions.get(name, "i")
        route = routes.get(name, {})
        default_mode = "network" if route else ("expression" if name in expressions else "values")
        rows.append(
            '<div class="row">'
            f'<div class="name">{_escape(name)}</div>'
            f'<select id="mode_{_escape(name)}">'
            f'<option value="values"{"" if default_mode == "values" else ""}>values</option>'
            f'<option value="expression"{" selected" if default_mode == "expression" else ""}>expression</option>'
            f'<option value="network"{" selected" if default_mode == "network" else ""}>network trajectory</option>'
            '<option value="gaussian">gaussian</option>'
            "</select>"
            '<div class="grow">'
            f'<div id="row_values_{_escape(name)}">'
            f'<input type="text" id="in_{_escape(name)}" value="1, 2, 3, 4" '
            'placeholder="comma or space separated numbers"></div>'
            f'<div id="row_expr_{_escape(name)}" hidden>'
            f'<input type="text" id="expr_{_escape(name)}" '
            f'value="{_escape(expression)}" '
            'placeholder="expression over i, x, y, w, h, t"></div>'
            f'<div id="row_gauss_{_escape(name)}" hidden class="imgctl">'
            f'<label>mean <input type="number" step="any" '
            f'id="mean_{_escape(name)}" value="0"></label>'
            f'<label>sigma <input type="number" step="any" '
            f'id="sigma_{_escape(name)}" value="1"></label></div>'
            "</div></div>"
        )
    if not feeds:
        rows.append(
            '<div class="meta">This program takes no array feeds; the domain '
            "width and height decide how many elements one run covers.</div>"
        )
    return "\n".join(rows)


def _source_tabs(sources: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Tabs and panes for every backend this program was emitted through.

    A backend that could not serve the program keeps its tab and says why.
    Which languages a program can reach is a real property of the program,
    and a tab that quietly vanished would hide it.
    """

    if not sources:
        return "", '<div class="meta">No backend sources were collected.</div>'
    tabs, views = [], []
    first_available = next(
        (s["language"] for s in sources if s.get("available")),
        sources[0]["language"],
    )
    for entry in sources:
        language = str(entry["language"])
        selected = "true" if language == first_available else "false"
        mark = "" if entry.get("available") else " &middot; n/a"
        lines = entry.get("lines") or 0
        tabs.append(
            f'<div class="srctab" data-lang="{_escape(language)}" '
            f'aria-selected="{selected}">{_escape(str(entry["title"]))}{mark}</div>'
        )
        if entry.get("available"):
            if entry.get("url"):
                body = (
                    f'<div class="meta">{lines} lines &middot; not loaded</div>'
                    f'<button class="download-source" data-lang="{_escape(language)}">'
                    f'Download {_escape(str(entry["title"]))} source</button>'
                    '<div class="meta">The file is fetched only after this button is clicked.</div>'
                )
            else:
                body = (
                    f'<div class="meta">{lines} lines</div>'
                    f"<pre>{_escape(str(entry.get('source') or ''))}</pre>"
                )
        else:
            body = (
                '<div class="note">This backend did not serve the program: '
                f"{_escape(str(entry.get('reason') or 'no reason given'))}</div>"
            )
        views.append(
            f'<div class="srcview" data-lang="{_escape(language)}"'
            f'{"" if language == first_available else " hidden"}>{body}</div>'
        )
    return "\n".join(tabs), "\n".join(views)


def _build_rows(parameters: Mapping[str, Any]) -> str:
    """Parameters that were fixed when the program was compiled.

    Shown read-only because they are not inputs at all: an unrolled loop
    count is part of the emitted instructions, so changing it here would be
    a lie -- it needs a recompile.
    """

    if not parameters:
        return ""
    rows = "".join(
        f'<div class="kv"><b>{_escape(str(key))}</b><span>{_escape(str(value))}</span></div>'
        for key, value in parameters.items()
    )
    return (
        '<div class="meta" style="margin-top:.6rem">Compiled in &mdash; changing '
        "these needs a recompile, not a re-run.</div>" + rows
    )


def emit_html_shell(
    api: Any,
    *,
    source: str = "",
    wasm_bytes: bytes | None = None,
    name: str | None = None,
    telemetry: Any = None,
    process_graph: Any = None,
    origin_source: str = "",
    feed_expressions: Mapping[str, str] | None = None,
    build_parameters: Mapping[str, Any] | None = None,
    default_width: int = 64,
    default_height: int = 40,
    backend_sources: Any = None,
    network_manifest: Mapping[str, Any] | None = None,
    class_graph: Mapping[str, Any] | None = None,
    graph_views: Mapping[str, Any] | None = None,
) -> HtmlShell:
    """Generate a launchable page for one compiled program.

    ``api`` is a ``CompiledProgramAPI`` (or its mapping). ``source`` is the
    WAT, shown for reading. ``wasm_bytes`` is the assembled binary when one
    exists; without it the page offers a file picker instead of pretending
    it can assemble text itself.

    ``telemetry`` is a ``shell_telemetry.TelemetryChannel`` (or its mapping)
    carrying what happened while the program was compiled. Its records share
    a schema with the ones the page produces at run time, so the page shows
    one timeline rather than a build log and a run log side by side --
    including progress, which drives the bar from the same records that
    appear in the pane, so the two cannot disagree.

    ``process_graph`` is a ``ProcessGraph`` (or an already-summarized
    mapping) and ``origin_source`` the Python the program was compiled from.
    Both are shown because "what did this come from" is the first question
    asked of a compiled artifact and the hardest to answer from the artifact
    alone.

    ``class_graph`` -- ``wasm_class_modules.build_embedded_class_graph``'s
    output -- switches the page's *execution* from one embedded module to
    the segmented class-graph runner while leaving every existing control
    (the input/output rows, the API panel, the source tabs) exactly as
    ``api`` already describes them; see ``describe_process_graph_api`` for
    building an ``api`` that matches a class graph. ``None`` (the default)
    reproduces the exact single-module page this function has always
    produced -- every existing caller, ``build_homepage.py`` included, is
    unaffected.
    """

    mapping = api.to_mapping() if hasattr(api, "to_mapping") else dict(api)
    entry_name = mapping.get("entry")
    entry = next(
        (e for e in mapping["entry_points"] if e["name"] == entry_name),
        mapping["entry_points"][0],
    )
    parameters = entry["parameters"]
    shell_name = name or f"{mapping['module']}_shell"

    encoded = (
        json.dumps(base64.b64encode(wasm_bytes).decode("ascii"))
        if wasm_bytes
        else "null"
    )
    if telemetry is None:
        telemetry_mapping: dict[str, Any] = {"records": []}
    elif hasattr(telemetry, "to_mapping"):
        telemetry_mapping = telemetry.to_mapping()
    else:
        telemetry_mapping = dict(telemetry)
    if process_graph is None:
        graph_mapping: dict[str, Any] = {}
    elif hasattr(process_graph, "G") or hasattr(process_graph, "nodes"):
        from .shell_telemetry import summarize_process_graph

        graph_mapping = summarize_process_graph(process_graph)
    else:
        graph_mapping = dict(process_graph)
    network_mapping = dict(network_manifest or {})
    network_mapping.setdefault("name", "No feedback network attached")
    network_mapping.setdefault("routes", [])
    network_routes = {str(route["feed"]): route for route in network_mapping["routes"] if isinstance(route, Mapping) and route.get("feed")}
    script = (
        _JS.replace("__API__", json.dumps(mapping))
        .replace("__WASM__", encoded)
        .replace("__GRAPH__", json.dumps(graph_mapping, default=str))
        .replace("__GRAPH_VIEWS__", json.dumps(dict(graph_views or {}), default=str))
        .replace("__NETWORK__", json.dumps(network_mapping, default=str))
        .replace("__SOURCE_DOWNLOADS__", json.dumps([
            {
                "language": str(entry.get("language", "")),
                "url": str(entry.get("url", "")),
                "filename": str(entry.get("filename", "")),
            }
            for entry in (
                backend_sources.to_mapping()["sources"]
                if hasattr(backend_sources, "to_mapping")
                else list(backend_sources or [])
            )
            if entry.get("url")
        ], default=str))
        .replace(
            "__CLASS_GRAPH__",
            json.dumps(dict(class_graph), default=str) if class_graph else "null",
        )
        .replace("__ENTRY__", str(entry["name"]))
    )
    boot_script = _BOOT_JS.replace(
        "__TELEMETRY__", json.dumps(telemetry_mapping, default=str)
    )

    external_class_graph = bool(
        class_graph and any(module.get("url") for module in class_graph.get("modules", ()))
    )
    if external_class_graph:
        banner = (
            '<div class="note good">Versioned deployment manifest loaded. '
            'WebAssembly regions, the contiguous compile, and language source '
            'files remain unloaded until their corresponding run or download action.</div>'
        )
        picker = ""
        disabled = ""
    elif wasm_bytes or class_graph:
        banner = (
            '<div class="note good">Binary embedded &mdash; this file is '
            "self-contained and runs offline.</div>"
        )
        picker = ""
        disabled = ""
    else:
        banner = (
            '<div class="note">No assembled <code>.wasm</code> was available '
            "when this page was generated (a browser cannot assemble WAT "
            "itself). Assemble the <code>.wat</code> below with "
            "<code>wat2wasm</code> and load the binary here.</div>"
        )
        picker = (
            '<div class="row"><div class="name">.wasm</div>'
            '<div class="grow"><input type="file" id="picker" accept=".wasm"></div>'
            "</div>"
        )
        disabled = " disabled"

    try:
        api_yaml = api.to_yaml() if hasattr(api, "to_yaml") else ""
    except Exception:
        # A descriptor that cannot render is not a reason to lose the page.
        api_yaml = ""

    build_rows = _build_rows(build_parameters or {})

    if backend_sources is None:
        source_entries: list[Mapping[str, Any]] = []
    elif hasattr(backend_sources, "to_mapping"):
        source_entries = list(backend_sources.to_mapping()["sources"])
    else:
        source_entries = list(backend_sources)
    source_tabs, source_views = _source_tabs(source_entries)
    lazy_sources = any(entry.get("url") for entry in source_entries)
    original_source_body = (
        '<div class="meta">Use the Python source download above; it has not been loaded.</div>'
        if lazy_sources else f'<pre>{_escape(origin_source)}</pre>'
    )
    emitted_source_body = (
        '<div class="meta">Use the WebAssembly source download above; it has not been loaded.</div>'
        if lazy_sources else f'<pre>{_escape(source)}</pre>'
    )

    note = entry.get("note")
    note_html = f'<div class="note">{_escape(str(note))}</div>' if note else ""
    execution_modes_html = ""
    if class_graph:
        contiguous = dict(class_graph).get("contiguous")
        contiguous_button = (
            '<button class="execution-mode" data-mode="contiguous" aria-pressed="false">'
            'Lazy contiguous compile</button>'
            if contiguous else ""
        )
        execution_modes_html = (
            '<div class="panel"><div class="panel-title">Execution shape</div>'
            '<div class="meta">Switch the same object and API between shared-memory '
            'punch cards and a full contiguous compile. Neither artifact downloads until '
            'its first run.</div><div class="execution-modes">'
            '<button class="execution-mode" data-mode="staged" aria-pressed="true">'
            'Staged punch cards</button>' + contiguous_button + '</div>'
            '<div class="stat"><span>shared globals · offset ABI · zero tensor seam copies</span>'
            '<span id="contiguous-state">contiguous not loaded</span></div></div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(shell_name)}</title>
<style>{_CSS}</style>
</head>
<body>
  <div class="title">{_escape(str(mapping["module"]))}</div>
  <div class="sub">{_escape(str(mapping["language"]))} &middot; entry
    <code>{_escape(str(entry["name"]))}</code> &middot;
    {len(parameters)} parameters</div>

  <div id="fatal" class="note" hidden></div>
  {banner}
  {note_html}

  <div class="panel">
    <div class="panel-title">Signature</div>
    {_signature_rows(parameters)}
  </div>

  {execution_modes_html}

  <div class="panel">
    <div class="panel-title">Domain</div>
    <div class="meta">A kernel's feeds are computed over a grid. Width and
      height set how many elements one run covers, and the image view uses
      the same numbers. Feed expressions see <code>i</code>, <code>x</code>,
      <code>y</code>, <code>w</code>, <code>h</code> and <code>t</code> &mdash;
      <code>t</code> is the frame number, so anything an expression does with
      it is what Animate shows.</div>
    <div class="imgctl">
      <label>width <input type="number" id="dom_w" min="1" value="{default_width}"></label>
      <label>height <input type="number" id="dom_h" min="1" value="{default_height}"></label>
    </div>
    {build_rows}
  </div>

  <div class="panel">
    <div class="panel-title">Inputs</div>
    {picker}
    {_input_rows(parameters, feed_expressions, network_routes)}
    <div id="stats" class="stat"></div>
    <div class="row">
      <button id="run"{disabled}>Run {_escape(str(entry["name"]))}</button>
      <label class="meta">repeat&nbsp;<input type="number" id="repeats" min="0"
        value="0" style="width:4.5rem" title="0 keeps going until you press
        Stop. Each repeat is a frame: feeds are regenerated, so a gaussian
        redraws and an expression sees a new t"></label>
      <span class="meta">0 = continuous</span>

      <div class="grow"><div id="status" class="out"></div></div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-title">Feedback network</div>
    <div class="meta">This is the standard routing surface for an optional compiled inference network. Its manifest records which feed it observes and what compiled output it can influence; a service can supply the same contract when it creates a new page.</div>
    <div id="networkstats" class="stat">Awaiting a run.</div>
  </div>
  <div class="panel">
    <div class="panel-title">Results</div>
    <div class="tabs" role="tablist">
      <div class="tab" data-view="raw" role="tab" aria-selected="true">Raw</div>
      <div class="tab" data-view="image" role="tab" aria-selected="false">RGB pixels</div>
    </div>
    <div class="tabview" data-view="raw">
      <div id="raw" class="out"></div>
    </div>
    <div class="tabview" data-view="image" hidden>
      <div class="imgctl">
        <label><input type="checkbox" id="img_invert"> invert</label>
      </div>
      <div id="imgnote" class="meta"></div>
      <canvas id="canvas" width="1" height="1"></canvas>
    </div>
  </div>

  <div class="panel">
    <div class="panel-title">Diagnostics</div>
    <div id="bar-wrap" hidden>
      <div class="barlabel"><span id="bartext"></span><span id="barpct"></span></div>
      <div class="bar"><i id="barfill"></i></div>
    </div>
    <div class="filters">
      <label><input type="checkbox" data-kind="log" checked> log</label>
      <label><input type="checkbox" data-kind="error" checked> error</label>
      <label><input type="checkbox" data-kind="profile" checked> profile</label>
      <label><input type="checkbox" data-kind="progress" checked> progress</label>
      <label><input type="checkbox" data-kind="call" checked> call</label>
      <label><input type="checkbox" data-kind="ok" checked> ok</label>
    </div>
    <div id="log"></div>
    <div class="row"><button id="copylog">Copy log</button></div>
  </div>

  <div class="panel">
    <details open>
      <summary>What made this</summary>
      <div class="meta">One ordinary Python function, ingested as an AST,
        planned once as a ProcessGraph, and lowered through the AOT compiler.
        Every tab below came out of <em>that one compilation</em> &mdash; not
        from separate re-implementations kept in step by hand.</div>
      <div class="srctabs">{source_tabs}</div>
      {source_views}
    </details>
  </div>

  <div class="panel">
    <details>
      <summary>API descriptor</summary>
      <div class="meta">The contract this page binds against. Editing it here
        does not yet change anything -- applying an edited descriptor is not
        wired up, and pretending otherwise would be worse than saying so.</div>
      <textarea id="apiyaml" rows="16" spellcheck="false">{_escape(api_yaml)}</textarea>
      <div class="row">
        <button id="applyapi" disabled title="not wired up yet">Apply</button>
        <button id="copyapi">Copy</button>
        <div class="grow"><div id="apistatus" class="out"></div></div>
      </div>
    </details>
  </div>

  <div class="panel">
    <details>
      <summary>Process graph</summary>
      <div id="graph"></div>
    </details>
  </div>

  <div class="panel">
    <details>
      <summary>Original source</summary>
      {original_source_body}
    </details>
  </div>

  <div class="panel">
    <details>
      <summary>Emitted source</summary>
      {emitted_source_body}
    </details>
  </div>

<script>{boot_script}</script>
<script>{script}</script>
</body>
</html>
"""
    return HtmlShell(name=shell_name, html=html, embedded=wasm_bytes is not None)


def shell_for_artifact(
    artifact: Any,
    *,
    wasm_bytes: bytes | None = None,
    telemetry: Any = None,
    process_graph: Any = None,
    origin_source: str = "",
    feed_expressions: Mapping[str, str] | None = None,
    build_parameters: Mapping[str, Any] | None = None,
    default_width: int = 64,
    default_height: int = 40,
    backend_sources: Any = None,
    network_manifest: Mapping[str, Any] | None = None,
) -> HtmlShell:
    """Generate the page straight from a ``machine_targets.TargetArtifact``."""

    if artifact.api is None:
        raise ValueError(
            f"{artifact.target} artifact carries no API descriptor, so a "
            "shell cannot be generated for it"
        )
    return emit_html_shell(
        artifact.api,
        source=artifact.source,
        wasm_bytes=wasm_bytes,
        name=f"{artifact.name}_shell",
        telemetry=telemetry,
        process_graph=process_graph,
        origin_source=origin_source,
        feed_expressions=feed_expressions,
        build_parameters=build_parameters,
        default_width=default_width,
        default_height=default_height,
        backend_sources=backend_sources,
    )


__all__ = ["HtmlShell", "emit_html_shell", "shell_for_artifact"]
