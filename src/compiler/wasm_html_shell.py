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
page reshapes itself. Structural source pages extend the same contract to all
callables: outer tabs group classes, inner tabs select methods, and module
functions follow the class groups without stretching the document vertically.

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
import re
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
.bar > i { display: block; height: 100%; width: 0; background: var(--accent); }
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
  border-radius: .35rem; padding: .4rem .55rem;
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
.graph-scroll { max-height: 26rem; overflow: auto; border: 1px solid var(--line);
  border-radius: .45rem; background: #05070c; }
.process-graph-grid { display: grid; min-width: 42rem; min-height: 22rem;
  grid-template-columns: repeat(var(--graph-columns), minmax(2px, 1fr));
  grid-template-rows: repeat(var(--graph-rows), 4px); padding: .7rem; }
.graph-indicator { width: 3px; height: 3px; border-radius: 50%; align-self: center;
  justify-self: center; cursor: pointer; background: rgb(var(--node-r) var(--node-g) var(--node-b));
  opacity: var(--node-opacity, .26); transform: scale(var(--node-scale, 1));
  box-shadow: 0 0 var(--node-blur, 0) rgb(var(--node-r) var(--node-g) var(--node-b)); }
.graph-indicator[data-selected="true"] { outline: 1px solid #fff; outline-offset: 2px; }
.graph-profile-stats { padding: .35rem .5rem; border-radius: .3rem;
  background: color-mix(in srgb, var(--accent) var(--profile-glow, 0%), transparent); }
.graph-option { display: inline-flex; align-items: center; gap: .25rem; cursor: pointer; }
.filters { display: flex; gap: .4rem; margin-bottom: .5rem; font-size: .75rem; }
.filters label { cursor: pointer; opacity: .75; }
.server-controls { display: grid; grid-template-columns: minmax(9rem, 1fr) minmax(9rem, 1fr);
  gap: .55rem .8rem; margin-top: .65rem; }
.server-controls label { font-size: .75rem; opacity: .78; }
.server-controls input { min-height: 2.1rem; }
.server-controls .wide { grid-column: 1 / -1; }
.gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
  gap: .6rem; margin-top: .7rem; }
.gallery-card { border: 1px solid var(--line); border-radius: .4rem;
  padding: .6rem .7rem; background: var(--soft); }
.gallery-card a { font-weight: 600; }
.gallery-card .meta { display: block; min-width: 0; margin-top: .2rem; }
.callable-group { margin-top: 1rem; }
.callable-group-title { color: var(--accent); font-weight: 700; margin-bottom: .45rem; }
.callable-tabs { display: flex; flex-wrap: wrap; gap: .3rem; margin-bottom: .55rem; }
.callable-owner-tabs { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .75rem; }
.callable-owner-tab { border: 1px solid var(--line); border-radius: .4rem; padding: .4rem .65rem;
  cursor: pointer; font-weight: 650; user-select: none; }
.callable-owner-tab[aria-selected="true"] { border-color: var(--accent); color: var(--accent);
  background: color-mix(in srgb, var(--accent) 10%, var(--panel)); }
.callable-owner-tab:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.callable-owner-view { margin-top: .6rem; }
.class-map-tabs { display: flex; gap: .35rem; margin: .65rem 0; }
.class-map-tab { border: 1px solid var(--line); border-radius: .35rem; padding: .35rem .6rem;
  cursor: pointer; user-select: none; }
.class-map-tab[aria-selected="true"] { border-color: var(--accent); color: var(--accent); }
.callable-tab { border: 1px solid var(--line); border-radius: .35rem; padding: .3rem .55rem;
  color: var(--muted); cursor: pointer; font-size: .82rem; user-select: none; }
.callable-tab[aria-selected="true"] { color: var(--fg); border-color: var(--accent);
  background: color-mix(in srgb, var(--accent) 12%, var(--panel2)); }
.callable-tab:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.callable-run-system { border: 1px solid var(--line); border-radius: .4rem;
  padding: .75rem; margin: .55rem 0; background: var(--panel2); }
.callable-run-system .method-title { font-weight: 650; overflow-wrap: anywhere; }
.callable-inputs { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
  gap: .45rem; margin: .6rem 0; }
.callable-inputs label { color: var(--muted); font-size: .82rem; }
.callable-inputs input { display: block; width: 100%; margin-top: .2rem; }
.srctabs { display: flex; flex-wrap: wrap; gap: .25rem; margin: .5rem 0; }
.srctab { padding: .25rem .7rem; border-radius: .3rem; cursor: pointer;
  font-size: .75rem; font-weight: 600; background: var(--soft);
  border: 1px solid transparent; font-family: ui-monospace, monospace; }
.srctab[aria-selected="true"] { border-color: var(--accent); color: var(--accent); }
.srcview[hidden] { display: none; }
.chalkboard { overflow: auto; margin: .65rem 0; padding: .85rem 1rem;
  border: 1px solid var(--line); border-radius: .35rem; background: var(--soft); }
.chalkboard math { min-width: max-content; font-size: 1.08rem; }
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

# Managed-time audio is installed before feed expressions are evaluated.  It
# observes compiled outputs but never writes ``dt``: the state machine remains
# the sole time authority, while speaker playbackRate follows managed-time
# advancement divided by wall-time advancement.
_AUDIO_RUNTIME_JS = r"""(() => {
  const CONFIG = __AUDIO_RUNTIME__;
  if (!CONFIG) return;
  let context = null;
  let source = null;
  let panner = null;
  let decoded = null;
  let featureDocument = null;
  let managedTime = 0;
  let previousManagedTime = null;
  let previousWallTime = null;
  let observedRevision = -1;

  const ready = fetch(
    new URL(CONFIG.features_url, document.baseURI), {cache: "no-store"}
  ).then(response => {
    if (!response.ok) throw new Error("audio features failed: HTTP " + response.status);
    return response.json();
  }).then(features => { featureDocument = features; });

  function outputValue(frame, name) {
    const output = frame && frame.outputs.find(item => item.name === name);
    return output && output.values && output.values.length ? Number(output.values[0]) : null;
  }

  async function ensureDecoded() {
    if (decoded) return decoded;
    const response = await fetch(new URL(CONFIG.audio_url, document.baseURI), {cache: "no-store"});
    if (!response.ok) throw new Error("audio load failed: HTTP " + response.status);
    const bytes = await response.arrayBuffer();
    context = context || new AudioContext({sampleRate: CONFIG.sample_rate});
    decoded = await context.decodeAudioData(bytes.slice(0));
    return decoded;
  }

  async function start() {
    const buffer = await ensureDecoded();
    if (context.state === "suspended") await context.resume();
    if (source) return;
    source = context.createBufferSource();
    source.buffer = buffer;
    source.loop = true;
    panner = context.createStereoPanner();
    source.connect(panner).connect(context.destination);
    source.start(0, ((managedTime % buffer.duration) + buffer.duration) % buffer.duration);
  }

  function observe() {
    const runtime = window.TuringWasmRuntime;
    const frame = runtime && runtime.outputFrame ? runtime.outputFrame() : null;
    if (frame && frame.revision !== observedRevision) {
      observedRevision = frame.revision;
      const nextManagedTime = outputValue(frame, CONFIG.managed_time_output);
      const wallTime = performance.now() / 1000;
      if (nextManagedTime !== null) {
        if (source && previousManagedTime !== null && previousWallTime !== null) {
          const wallDelta = wallTime - previousWallTime;
          const managedDelta = nextManagedTime - previousManagedTime;
          if (wallDelta > 0 && managedDelta >= 0) {
            source.playbackRate.setTargetAtTime(
              Math.max(0.0001, managedDelta / wallDelta), context.currentTime, 0.015
            );
          }
        }
        managedTime = nextManagedTime;
        previousManagedTime = nextManagedTime;
        previousWallTime = wallTime;
      }
      if (panner && CONFIG.pan_output) {
        const position = outputValue(frame, CONFIG.pan_output);
        const range = CONFIG.pan_range || [-1, 1];
        if (position !== null && range[1] !== range[0]) {
          const pan = Math.max(-1, Math.min(1,
            2 * (position - range[0]) / (range[1] - range[0]) - 1
          ));
          panner.pan.setTargetAtTime(pan, context.currentTime, 0.02);
        }
      }
    }
    requestAnimationFrame(observe);
  }

  window.TuringAudioRuntime = Object.freeze({
    ready,
    start,
    suspend: () => context ? context.suspend() : Promise.resolve(),
    feature(name) {
      if (!featureDocument) return 0;
      const values = featureDocument.feeds[name];
      if (!values || !values.length) return 0;
      const position = ((managedTime % featureDocument.duration)
        + featureDocument.duration) % featureDocument.duration;
      const frame = position * featureDocument.feature_fps;
      const lower = Math.floor(frame) % values.length;
      const upper = (lower + 1) % values.length;
      const blend = frame - Math.floor(frame);
      return values[lower] * (1 - blend) + values[upper] * blend;
    },
    async outputDevices() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return [];
      return (await navigator.mediaDevices.enumerateDevices())
        .filter(device => device.kind === "audiooutput");
    },
    async selectOutput(deviceId) {
      context = context || new AudioContext({sampleRate: CONFIG.sample_rate});
      if (typeof context.setSinkId !== "function") {
        throw new Error("this browser does not support selecting an audio output");
      }
      await context.setSinkId(deviceId);
    },
    get managedTime() { return managedTime; },
  });
  window.addEventListener("pointerdown", () => start().catch(console.error), {once: true});
  ready.catch(console.error);
  requestAnimationFrame(observe);
})();"""

# The program script proper.
_JS = r"""const API = __API__;
const WASM_BASE64 = __WASM__;

const $ = (id) => document.getElementById(id);
const GRAPH = __GRAPH__;
let GRAPH_VIEWS = __GRAPH_VIEWS__;
const NETWORK = __NETWORK__;
const CLASS_GRAPH = __CLASS_GRAPH__;
const MAP_IR = __MAP_IR__;
const SOURCE_DOWNLOADS = __SOURCE_DOWNLOADS__;
const MATHEMATICS = __MATHEMATICS__;
const RESOURCE_ROUTE = __RESOURCE_ROUTE__;
const STATIC_GALLERY = __STATIC_GALLERY__;
const DEFAULT_SERVER_ADDRESS = __DEFAULT_SERVER_ADDRESS__;
const entry = API.entry_points.find(e => e.name === API.entry) || API.entry_points[0];
const params = entry.parameters;
const inputs = params.filter(p => p.role === "input");
const outputs = params.filter(p => p.role === "output");
const bytes = API.metadata.element_bytes || 8;
const isF32 = (API.metadata.value_type || "f64") === "f32";
// A stateful Python program names which returned arrays become inputs to its
// next admitted tick.  The shell only transports those arrays; transition
// math and managed-time evolution remain in the compiled Python program.
const STATE_FEEDBACK = (API.metadata || {}).state_feedback || {};
const STATE_FEEDBACK_PAIRS = Object.entries(STATE_FEEDBACK);
const HAS_STATE_FEEDBACK = STATE_FEEDBACK_PAIRS.length > 0;

function refreshNonStateFeeds(activeFeeds, count, d, frameIndex) {
  return inputs.map((p, index) =>
    Object.prototype.hasOwnProperty.call(STATE_FEEDBACK, p.name)
      ? activeFeeds[index]
      : feedValues(p, count, d, frameIndex)
  );
}

function acceptCompiledState(activeFeeds, result) {
  for (const [inputName, outputName] of STATE_FEEDBACK_PAIRS) {
    const inputIndex = inputs.findIndex(item => item.name === inputName);
    const outputIndex = outputs.findIndex(item => item.name === outputName);
    if (inputIndex < 0 || outputIndex < 0) {
      throw new Error(
        "state feedback ABI mismatch: " + inputName + " <- " + outputName
      );
    }
    const values = residentValues(result[outputIndex]);
    activeFeeds[inputIndex] = values.slice
      ? values.slice()
      : new (isF32 ? Float32Array : Float64Array)(values);
  }
}

function serverAddress() {
  const input = document.getElementById("server-address");
  const value = (input ? input.value : DEFAULT_SERVER_ADDRESS).trim();
  if (!value) return null;
  const url = new URL(value);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("server address must use http or https");
  }
  return url;
}

function serverURL(path) {
  const server = serverAddress();
  if (!server) return new URL(path, document.baseURI).href;
  return new URL(path, server).href;
}

function resourceURLs(path) {
  if (!path) throw new Error("resource path is empty");
  if (/^(?:[a-z]+:)?\/\//i.test(path) || path.startsWith("data:") || path.startsWith("blob:")) {
    return [path];
  }
  const route = RESOURCE_ROUTE.endsWith("/") ? RESOURCE_ROUTE : RESOURCE_ROUTE + "/";
  const relative = String(path).replace(/^\.\//, "").replace(/^\//, "");
  const candidates = [];
  const add = value => {
    const url = new URL(value, document.baseURI).href;
    if (!candidates.includes(url)) candidates.push(url);
  };
  // The generated page and its artifacts are published together.  This is
  // the correct first location for a file opened from the Go server, a
  // repository-prefixed GitHub Pages URL, and a versioned bundle page.
  // A file:// fetch may be rejected by the browser, in which case the
  // configured loopback-server candidates below remain available.
  add(relative);
  if (relative.startsWith("site/")) {
    const siteIndex = window.location.pathname.indexOf("/site/");
    if (siteIndex >= 0) {
      const repositoryRoot = new URL(window.location.href);
      repositoryRoot.pathname = window.location.pathname.slice(0, siteIndex + 1);
      repositoryRoot.search = "";
      repositoryRoot.hash = "";
      add(new URL(relative, repositoryRoot));
    }
  }
  if (route.startsWith("/") && /^https?:$/.test(window.location.protocol)) {
    const routeIndex = window.location.pathname.indexOf(route);
    const pagesPrefix = routeIndex >= 0 ? window.location.pathname.slice(0, routeIndex) : "";
    add(new URL(pagesPrefix + route + relative, window.location.origin));
  } else {
    add(new URL(relative, new URL(route, document.baseURI)));
  }
  try {
    const siteIndex = window.location.pathname.indexOf("/site/");
    if (siteIndex >= 0 && !relative.startsWith("site/")) {
      const pageDirectory = window.location.pathname.slice(
        siteIndex, window.location.pathname.lastIndexOf("/") + 1
      );
      add(serverURL(pageDirectory + relative));
    }
    add(serverURL("/" + relative));
  } catch (_) {
    // A malformed optional server address must not break static navigation.
  }
  return candidates;
}

function resourceURL(path) {
  return resourceURLs(path)[0];
}

async function fetchResource(path, options) {
  const failures = [];
  for (const url of resourceURLs(path)) {
    try {
      const response = await fetch(url, options);
      if (response.ok) return response;
      failures.push(url + " (HTTP " + response.status + ")");
    } catch (error) {
      failures.push(url + " (" + error.message + ")");
    }
  }
  throw new Error("resource not found in any configured location: " + failures.join(", "));
}

function evaluateClassPermission(identity, required, evaluator) {
  if (typeof evaluator !== "function") {
    throw new TypeError("class navigation requires a permission evaluator");
  }
  if (!evaluator(identity, required || [])) {
    throw new Error("access denied to " + identity);
  }
}

function resolveClass(classIdentity, evaluator) {
  const classes = ((MAP_IR.class_navigation || {}).classes || []);
  const matches = classes.filter(item => item.identity === classIdentity);
  if (matches.length !== 1) throw new Error(
    "unknown or ambiguous class identity " + classIdentity
  );
  evaluateClassPermission(matches[0].identity, matches[0].permissions, evaluator);
  return matches[0];
}

function resolveClassMember(classIdentity, memberName, evaluator) {
  const record = resolveClass(classIdentity, evaluator);
  const matches = (record.members || []).filter(item => item.name === memberName);
  if (matches.length !== 1) throw new Error(
    "unknown or ambiguous member " + classIdentity + "." + memberName
  );
  evaluateClassPermission(matches[0].identity, matches[0].permissions, evaluator);
  return matches[0];
}

function resolveClassInstantiation(classIdentity, evaluator) {
  const record = resolveClass(classIdentity, evaluator);
  const constructors = new Set(record.instantiation_functions || []);
  for (const member of record.members || []) {
    if (constructors.has(member.function_reference)) {
      evaluateClassPermission(member.identity, member.permissions, evaluator);
    }
  }
  return Array.from(constructors);
}

window.TuringClassNavigation = Object.freeze({
  map: MAP_IR,
  resolveClass,
  resolveDot: resolveClassMember,
  instantiate: resolveClassInstantiation,
  evaluatePermission: evaluateClassPermission
});

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
// Every method card imports the same WebAssembly.Memory. JavaScript creates
// the class instance (memory + field-slot table + method inventory), then
// calls one translated WASM coordinator. Card-to-card calls stay in WASM.
// No live tensor is copied through JavaScript at a seam.
function wasmTileWorkerSource() {
  return `self.onmessage = async event => {
    try {
      const {manifest, inventory, methodIds, count, fields} = event.data;
      const specs = new Map(manifest.modules.map(spec => [spec.name, spec]));
      const cards = new Map((inventory.methods || []).map(card => [card.index, card]));
      const elementBytes = Number(manifest.modules[0].element_bytes || 8);
      const View = manifest.modules[0].value_type === "f32" ? Float32Array : Float64Array;
      const fieldCount = (inventory.field_slots || []).length;
      let cursor = Math.ceil(Number(manifest.shared_static_bytes || 0) / 4) * 4;
      const inventoryOffset = cursor;
      cursor += fieldCount * 4;
      cursor = Math.ceil(cursor / elementBytes) * elementBytes;
      const offsets = Array.from({length: fieldCount}, () => {
        const offset = cursor; cursor += count * elementBytes; return offset;
      });
      const memory = new WebAssembly.Memory({initial: Math.max(1, Math.ceil(cursor / 65536))});
      new Int32Array(memory.buffer, inventoryOffset, fieldCount).set(offsets);
      for (const [indexText, values] of Object.entries(fields)) {
        const index = Number(indexText);
        new View(memory.buffer, offsets[index], count).set(values);
      }
      const outputSlots = new Set();
      for (const methodId of methodIds) {
        const card = cards.get(methodId);
        if (!card) throw new Error("unknown deployment method " + methodId);
        const spec = specs.get(card.module);
        if (!spec) throw new Error("missing deployment module " + card.module);
        let bytes;
        if (spec.url) {
          const response = await fetch(spec.absolute_url || spec.url);
          if (!response.ok) throw new Error("worker fetch failed: HTTP " + response.status);
          bytes = await response.arrayBuffer();
        } else if (spec.wasm_base64) {
          const raw = atob(spec.wasm_base64);
          const decoded = new Uint8Array(raw.length);
          for (let i = 0; i < raw.length; i++) decoded[i] = raw.charCodeAt(i);
          bytes = decoded;
        } else throw new Error("deployment module has no bytes");
        const memoryImport = spec.shared_memory_import || {module: "env", field: "memory"};
        const imports = {};
        imports[memoryImport.module] = {[memoryImport.field]: memory};
        const {instance} = await WebAssembly.instantiate(bytes, imports);
        const args = [...card.input_slots, ...card.output_slots].map(slot => offsets[slot]);
        instance.exports[card.entry](count, ...args);
        card.output_slots.forEach(slot => outputSlots.add(slot));
      }
      const outputs = {};
      const transfer = [];
      for (const slot of outputSlots) {
        const values = new View(memory.buffer, offsets[slot], count).slice();
        outputs[slot] = values;
        transfer.push(values.buffer);
      }
      self.postMessage({outputs}, transfer);
    } catch (error) {
      self.postMessage({error: String(error && (error.stack || error.message || error))});
    }
  };`;
}

class ClassGraphRunner {
  constructor(manifest) {
    if (!manifest.shared_memory) throw new Error(
      "segmented manifest does not declare the shared-memory ABI"
    );
    this.manifest = manifest;
    this.modulesByName = new Map(manifest.modules.map(m => [m.name, m]));
    this.instances = new Map();
    this.runtime = null;
    this.lastExecutionMs = 0;
    this.fieldOffsets = [];
    this.fieldIndex = new Map(
      (manifest.class_inventory.field_slots || []).map(field => [field.key, field.index])
    );
    for (const redirect of manifest.class_inventory.storage_redirects || []) {
      const storageIndex = this.fieldIndex.get(redirect.storage);
      if (storageIndex === undefined) throw new Error(
        "storage redirect target is absent from the class inventory: " + redirect.storage
      );
      this.fieldIndex.set(redirect.identity, storageIndex);
    }
    this.layoutCount = 0;
    this.inventoryOffset = 0;
    this.tileWorkerURL = null;
    const staticBytes = Number(manifest.shared_static_bytes || 0);
    this.memory = new WebAssembly.Memory({
      initial: Math.max(1, Math.ceil(staticBytes / 65536))
    });
  }

  callsInDeploymentNode(node) {
    if (node.kind === "call") return [Number(node.method)];
    if (node.kind === "sequence") return node.children.flatMap(
      child => this.callsInDeploymentNode(child)
    );
    throw new Error("worker lane contains unsupported " + node.kind + " node");
  }

  tileRanges(count, workerCount) {
    const contract = this.manifest.thread_deployment || {};
    const alignment = Math.max(1, Number(contract.tile_alignment || 8));
    const desired = Math.max(1, workerCount * Number(contract.tiles_per_worker || 2));
    const tile = Math.max(
      alignment,
      Math.ceil(Math.ceil(count / desired) / alignment) * alignment
    );
    const ranges = [];
    for (let start = 0; start < count; start += tile) {
      ranges.push([start, Math.min(count, start + tile)]);
    }
    return ranges;
  }

  workerCount(taskCount) {
    const hardware = Math.max(1, Number(navigator.hardwareConcurrency || 2));
    return Math.max(1, Math.min(taskCount, 8, Math.max(1, hardware - 1)));
  }

  async runTileTask(methodIds, start, end, View) {
    if (!this.tileWorkerURL) {
      this.tileWorkerURL = URL.createObjectURL(new Blob(
        [wasmTileWorkerSource()], {type: "text/javascript"}
      ));
    }
    const count = end - start;
    const elementBytes = Number(this.manifest.modules[0].element_bytes || 8);
    const cards = new Map((this.manifest.class_inventory.methods || []).map(
      card => [Number(card.index), card]
    ));
    const touchedSlots = new Set();
    for (const methodId of methodIds) {
      const card = cards.get(Number(methodId));
      [...card.input_slots, ...card.output_slots].forEach(
        slot => touchedSlots.add(Number(slot))
      );
    }
    const fields = {};
    for (const slot of touchedSlots) {
      fields[slot] = new View(
        this.memory.buffer,
        this.fieldOffsets[slot] + start * elementBytes,
        count,
      ).slice();
    }
    const manifest = {
      ...this.manifest,
      modules: this.manifest.modules.map(spec => ({
        ...spec,
        absolute_url: spec.url ? new URL(spec.url, document.baseURI).href : null,
      })),
    };
    return new Promise((resolve, reject) => {
      const worker = new Worker(this.tileWorkerURL);
      worker.onmessage = event => {
        worker.terminate();
        if (event.data.error) reject(new Error(event.data.error));
        else resolve({start, end, outputs: event.data.outputs});
      };
      worker.onerror = event => {
        worker.terminate();
        reject(new Error(event.message || "WebAssembly tile worker failed"));
      };
      worker.postMessage({
        manifest,
        inventory: this.manifest.class_inventory,
        methodIds,
        count,
        fields,
      });
    });
  }

  async executeDeploymentNodeSerial(node, count) {
    if (node.kind === "call") {
      const method = (this.manifest.class_inventory.methods || []).find(
        candidate => Number(candidate.index) === Number(node.method)
      );
      const spec = this.modulesByName.get(method.module);
      const instance = this.instances.get(method.module);
      const args = [...method.input_slots, ...method.output_slots].map(
        slot => this.fieldOffsets[slot]
      );
      instance.exports[method.entry](count, ...args);
      return;
    }
    const children = node.kind === "deploy" ? node.lanes : node.children;
    for (const child of children) await this.executeDeploymentNodeSerial(child, count);
  }

  async executeDeploy(node, count, View) {
    const laneCalls = node.lanes.map(lane => this.callsInDeploymentNode(lane));
    const methods = new Map((this.manifest.class_inventory.methods || []).map(
      method => [Number(method.index), method]
    ));
    const written = new Set();
    for (const calls of laneCalls) for (const methodId of calls) {
      for (const slot of methods.get(methodId).output_slots) {
        if (written.has(slot)) {
          log("warn", "parallel lanes share an output slot; using serial fallback", {slot});
          return this.executeDeploymentNodeSerial(node, count);
        }
        written.add(slot);
      }
    }
    if (typeof Worker === "undefined" || count < 2) {
      return this.executeDeploymentNodeSerial(node, count);
    }
    const provisionalWorkers = this.workerCount(Math.max(1, laneCalls.length * 2));
    const ranges = this.tileRanges(count, provisionalWorkers);
    const tasks = laneCalls.flatMap(methodIds => ranges.map(
      ([start, end]) => ({methodIds, start, end})
    ));
    const limit = this.workerCount(tasks.length);
    log("progress", "Deploy: dispatching WebAssembly tiles", {
      done: 0, total: tasks.length, workers: limit, tiles: ranges.length,
      lanes: laneCalls.length, join: node.join.mode
    });
    const completed = [];
    try {
      for (let cursor = 0; cursor < tasks.length; cursor += limit) {
        const batch = tasks.slice(cursor, cursor + limit);
        completed.push(...await Promise.all(batch.map(task =>
          this.runTileTask(task.methodIds, task.start, task.end, View)
        )));
        setProgress(completed.length, tasks.length, "Join: awaiting WebAssembly tiles");
      }
    } catch (error) {
      log("warn", "thread deployment failed; replaying serial Wasm schedule", {
        error: String(error)
      });
      return this.executeDeploymentNodeSerial(node, count);
    }
    const elementBytes = Number(this.manifest.modules[0].element_bytes || 8);
    for (const result of completed) {
      for (const [slotText, values] of Object.entries(result.outputs)) {
        const slot = Number(slotText);
        new View(
          this.memory.buffer,
          this.fieldOffsets[slot] + result.start * elementBytes,
          result.end - result.start,
        ).set(values);
      }
    }
    log("ok", "Join: all WebAssembly tiles committed", {
      tiles: completed.length, workers: limit
    });
  }

  async executeDeploymentNode(node, count, View) {
    if (node.kind === "deploy") return this.executeDeploy(node, count, View);
    if (node.kind === "sequence") {
      for (const child of node.children) {
        await this.executeDeploymentNode(child, count, View);
      }
      return;
    }
    return this.executeDeploymentNodeSerial(node, count);
  }

  async binary(url, label) {
    const response = await fetchResource(url);
    if (!response.ok) throw new Error(
      "failed to load " + label + ": HTTP " + response.status
    );
    return response.arrayBuffer();
  }

  async instantiateCard(spec) {
    if (this.instances.has(spec.name)) return this.instances.get(spec.name);
    markDeploymentNode(spec.name, "downloading");
    let moduleBinary;
    if (spec.url) {
      const response = await fetchResource(spec.url);
      if (!response.ok) throw new Error(
        "failed to load method card " + spec.name + ": HTTP " + response.status
      );
      moduleBinary = await response.arrayBuffer();
    } else if (spec.wasm_base64) {
      const raw = atob(spec.wasm_base64);
      moduleBinary = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) moduleBinary[i] = raw.charCodeAt(i);
    } else if (this.manifest.thread_deployment) {
      const started = performance.now();
      await this.executeDeploymentNode(
        this.manifest.thread_deployment.root, count, View
      );
      this.lastExecutionMs = performance.now() - started;
    } else {
      throw new Error("method card " + spec.name + " has no URL or bytes");
    }
    const memoryImport = spec.shared_memory_import || {module: "env", field: "memory"};
    const imports = {};
    imports[memoryImport.module] = {[memoryImport.field]: this.memory};
    const { instance } = await WebAssembly.instantiate(moduleBinary, imports);
    this.instances.set(spec.name, instance);
    markDeploymentNode(spec.name, "ready");
    return instance;
  }

  async ensureRuntime() {
    if (this.runtime) return this.runtime;
    if (!this.manifest.coordinator) throw new Error(
      "class manifest has no translated coordinator"
    );
    const pairs = await Promise.all(this.manifest.modules.map(async spec => [
      spec, await this.instantiateCard(spec)
    ]));
    const imports = {env: {memory: this.memory}};
    for (const method of this.manifest.class_inventory.methods || []) {
      const pair = pairs.find(([spec]) => spec.name === method.module);
      if (!pair) throw new Error("inventory method module not loaded: " + method.module);
      if (!imports[method.module]) imports[method.module] = {};
      imports[method.module][method.entry] = pair[1].exports[method.entry];
    }
    const bytes = await this.binary(
      this.manifest.coordinator.url, "class coordinator"
    );
    const {instance} = await WebAssembly.instantiate(bytes, imports);
    this.runtime = instance;
    return instance;
  }

  layout(count) {
    if (this.layoutCount === count) return;
    const elementBytes = Number(this.manifest.modules[0].element_bytes || 8);
    const fields = this.manifest.class_inventory.field_slots || [];
    this.inventoryOffset = Math.ceil(Number(this.manifest.shared_static_bytes || 0) / 4) * 4;
    let cursor = this.inventoryOffset + fields.length * 4;
    cursor = Math.ceil(cursor / elementBytes) * elementBytes;
    this.fieldOffsets = fields.map(() => {
      const offset = cursor; cursor += count * elementBytes; return offset;
    });
    if (cursor > this.memory.buffer.byteLength) {
      this.memory.grow(Math.ceil((cursor - this.memory.buffer.byteLength) / 65536));
    }
    new Int32Array(this.memory.buffer, this.inventoryOffset, fields.length).set(
      this.fieldOffsets
    );
    this.layoutCount = count;
  }

  offsetForKey(key) {
    const index = this.fieldIndex.get(key);
    if (index === undefined) throw new Error("unknown shared-memory slot / class field " + key);
    return this.fieldOffsets[index];
  }

  redirectStorageOffset(identity, offset) {
    const index = this.fieldIndex.get(identity);
    if (index === undefined) throw new Error("unknown shared-memory identity " + identity);
    if (!Number.isInteger(offset) || offset < 0) throw new Error(
      "shared-memory storage offset must be a non-negative integer"
    );
    this.fieldOffsets[index] = offset;
    if (this.layoutCount > 0) {
      new Int32Array(this.memory.buffer, this.inventoryOffset + index * 4, 1)[0] = offset;
    }
  }

  redirectStorage(identity, storage) {
    this.redirectStorageOffset(identity, this.offsetForKey(storage));
  }

  storageReference(identity, count) {
    this.layout(count);
    return Object.freeze({
      turingStorageReference: true,
      memory: this.memory,
      identity,
      offset: this.offsetForKey(identity),
      count,
      valueType: this.manifest.modules[0].value_type,
    });
  }

  logicalOutputReference(logicalName, count) {
    const binding = this.manifest.logical_outputs[logicalName];
    if (!binding) throw new Error("unknown logical output " + logicalName);
    return this.storageReference("out::" + binding[0] + "::" + binding[1], count);
  }

  async run(
    logicalInputs, count, start = 0, end = null, latch = false,
    residentOutputs = false
  ) {
    this.layout(count);
    const View = this.manifest.modules[0].value_type === "f32" ? Float32Array : Float64Array;
    for (const [logicalName, source] of Object.entries(logicalInputs)) {
      const identity = "in::" + logicalName;
      if (source && source.turingStorageReference === true) {
        if (source.memory !== this.memory) throw new Error(
          "zero-copy storage references must belong to this shared WebAssembly.Memory"
        );
        if (Number(source.count) < count) throw new Error(
          logicalName + " resident storage is smaller than extent " + count
        );
        this.redirectStorageOffset(identity, Number(source.offset));
        continue;
      }
      const offset = this.offsetForKey(identity);
      const target = new View(this.memory.buffer, offset, count);
      if (ArrayBuffer.isView(source) || Array.isArray(source)) {
        if (source.length === 1) target.fill(Number(source[0]));
        else if (source.length >= count) target.set(source.subarray ? source.subarray(0, count) : source.slice(0, count));
        else throw new Error(logicalName + " has " + source.length + " values for extent " + count);
      } else {
        target.fill(Number(source));
      }
    }
    const runtime = await this.ensureRuntime();
    const methodCount = Number(this.manifest.coordinator.method_count);
    const supportsRanges = this.manifest.coordinator.supports_ranges !== false;
    const rangeStart = supportsRanges ? start : 0;
    const rangeEnd = supportsRanges && end !== null
      ? Math.min(methodCount, end) : methodCount;
    const activeMethods = (this.manifest.class_inventory.methods || []).filter(
      method => method.index >= rangeStart && method.index < rangeEnd
    );
    const coordinate = runtime.exports[this.manifest.coordinator.entry || "run_range"];
    if (latch && supportsRanges) {
      for (let index = 0; index < activeMethods.length && running; index++) {
        const method = activeMethods[index];
        markDeploymentNode(method.module, "running");
        const started = performance.now();
        coordinate(count, this.inventoryOffset, method.index, method.index + 1);
        markDeploymentNode(method.module, "done", performance.now() - started, "latched method");
        if (index + 1 < activeMethods.length && running) {
          await waitForCardLatch(index + 1, activeMethods.length);
        }
      }
    } else {
      const started = performance.now();
      coordinate(count, this.inventoryOffset, rangeStart, rangeEnd);
      const elapsed = performance.now() - started;
      this.lastExecutionMs = elapsed;
      activeMethods.forEach(method => queueDeploymentProfile(
        method.module, elapsed / Math.max(1, activeMethods.length),
        "coordinator amortized"
      ));
    }
    return outputs.map(parameter => {
      const binding = this.manifest.logical_outputs[parameter.name];
      if (!binding) throw new Error("logical output " + parameter.name + " has no deployment binding");
      const identity = "out::" + binding[0] + "::" + binding[1];
      if (residentOutputs) return this.storageReference(identity, count);
      const offset = this.offsetForKey(identity);
      return new View(this.memory.buffer, offset, count).slice();
    });
  }
}

class ContiguousRunner {
  constructor(spec) { this.spec = spec; this.runtime = null; }
  async instance() {
    if (this.runtime) return this.runtime;
    markContiguousState("downloading");
    const response = await fetchResource(this.spec.url);
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
const classGraphRunners = new Map();
const contiguousRunner = CLASS_GRAPH && CLASS_GRAPH.contiguous
  ? new ContiguousRunner(CLASS_GRAPH.contiguous) : null;
let activeExecutionMode = null;
let activeRegionSize = null;
let releaseCardLatch = null;

async function waitForCardLatch(completed, total) {
  const button = $("card-continue");
  if (!button) return;
  button.disabled = false;
  setStatus("breakpoint after method card " + completed + " / " + total, "good");
  await new Promise(resolve => { releaseCardLatch = resolve; });
  releaseCardLatch = null;
  button.disabled = true;
}

function activeStagedManifest() {
  if (!CLASS_GRAPH || activeRegionSize === null) return null;
  return (CLASS_GRAPH.variants || {})[String(activeRegionSize)] || null;
}

function activeClassGraphRunner() {
  const manifest = activeStagedManifest();
  if (!manifest) return null;
  const key = String(activeRegionSize);
  if (!classGraphRunners.has(key)) {
    classGraphRunners.set(key, new ClassGraphRunner(manifest));
  }
  return classGraphRunners.get(key);
}

window.TuringSharedClassMemory = Object.freeze({
  redirect(identity, storage) {
    const runner = activeClassGraphRunner();
    if (!runner) throw new Error("no active divided-program runner");
    runner.redirectStorage(identity, storage);
  },
  output(logicalName, count) {
    const runner = activeClassGraphRunner();
    if (!runner) throw new Error("no active divided-program runner");
    return runner.logicalOutputReference(logicalName, count);
  },
  runResident(logicalInputs, count, start = 0, end = null, latch = false) {
    const runner = activeClassGraphRunner();
    if (!runner) throw new Error("no active divided-program runner");
    return runner.run(logicalInputs, count, start, end, latch, true);
  },
});

// Stable bridge for presentation layers.  A shader surface may ask the page
// runtime to execute its bundled companion Wasm without reaching into the
// inspector controls or owning WebAssembly instantiation itself.
window.TuringWasmRuntime = Object.freeze({
  api: API,
  io: (API.metadata || {}).shell_io || null,
  sharedMemory: window.TuringSharedClassMemory,
  get running() { return running; },
  outputFrame() {
    const dimensions = domain();
    return {
      revision: outputRevision,
      width: dimensions.w,
      height: dimensions.h,
      outputs: (lastOutputs || []).map(item => ({
        name: item.name,
        values: item.values,
      })),
    };
  },
  start({continuous = true, preferContiguous = true} = {}) {
    if (running) return Promise.resolve();
    if (continuous && $("repeats")) $("repeats").value = "0";
    if (CLASS_GRAPH && !activeExecutionMode) {
      const preferred = preferContiguous && contiguousRunner
        ? document.querySelector('.execution-mode[data-mode="contiguous"]')
        : null;
      const choice = preferred || document.querySelector(".execution-mode");
      if (!choice) return Promise.reject(new Error("no WebAssembly deployment is published"));
      choice.click();
    }
    return run();
  },
  async run(logicalInputs, count) {
    if (contiguousRunner) return contiguousRunner.run(logicalInputs, count);
    const variants = CLASS_GRAPH && CLASS_GRAPH.variants
      ? Object.keys(CLASS_GRAPH.variants) : [];
    if (variants.length) {
      activeExecutionMode = "staged";
      activeRegionSize = Number(variants[0]);
      return activeClassGraphRunner().run(logicalInputs, count, 0, null, false, true);
    }
    throw new Error("this page has no liaison-compatible WebAssembly runner");
  },
});

function residentValues(value) {
  if (!value || value.turingStorageReference !== true) return value;
  const View = value.valueType === "f32" ? Float32Array : Float64Array;
  return new View(value.memory.buffer, value.offset, value.count);
}

async function computeViaSelectedRunner(feeds, count) {
  const logicalInputs = {};
  const manifest = activeExecutionMode === "staged" ? activeStagedManifest() : CLASS_GRAPH;
  if (!manifest) throw new Error("choose an execution shape before running");
  for (const logicalName of Object.keys(manifest.logical_inputs || {})) {
    const paramIndex = inputs.findIndex(p => p.name === logicalName);
    if (paramIndex < 0) throw new Error("logical input " + logicalName + " is not in the API");
    logicalInputs[logicalName] = feeds[paramIndex];
  }
  if (activeExecutionMode === "contiguous") {
    if (!contiguousRunner) throw new Error("no contiguous compile is published");
    return contiguousRunner.run(logicalInputs, count);
  }
  const runner = activeClassGraphRunner();
  if (!runner) throw new Error("choose a punch-card size before running");
  return runner.run(
    logicalInputs, count, 0, null,
    Boolean($("card-latch") && $("card-latch").checked),
    true
  );
}

// The segmented deployment follows the same full-domain run loop as a
// monolithic module.  Animation, feedback, rendering and timing all remain
// properties of the one logical program exposed by the page.
async function runClassGraphMode() {
  if (running) {
    running = false;
    if (releaseCardLatch) releaseCardLatch();
    return;
  }
  try {
    if (!activeExecutionMode) throw new Error(
      "choose Mono or a punch-card size before running"
    );
    const deployment = activeExecutionMode === "staged"
      ? activeStagedManifest() : CLASS_GRAPH;
    const d = domain();
    const anyExpression = inputs.some(p => $("mode_" + p.name).value === "expression");
    const anyGaussian = inputs.some(p => $("mode_" + p.name).value === "gaussian");
    const anyNetwork = inputs.some(p => $("mode_" + p.name).value === "network");
    const renderFps = Math.max(
      1,
      Number((API.metadata || {}).render_fps)
        || Number((NETWORK.feedback || {}).render_fps)
        || 24
    );
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
      (anyExpression || anyGaussian || anyNetwork || HAS_STATE_FEEDBACK);
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
      modules: activeExecutionMode === "staged" ? deployment.modules.length : 1,
      region_steps: activeRegionSize,
      elements: count,
    });
    for (let r = 0; running && (continuous || r < repeats); r++) {
      if (r > 0 && animated) {
        frameIndex = r;
        await advanceFeedback(feedbackTicks);
        activeFeeds = refreshNonStateFeeds(activeFeeds, count, d, frameIndex);
        applyFeedbackFeed(activeFeeds, count);
      }
      const frameStarted = performance.now();
      const t0 = performance.now();
      const result = await computeViaSelectedRunner(activeFeeds, count);
      const wallElapsed = performance.now() - t0;
      const stagedRunner = activeExecutionMode === "staged"
        ? activeClassGraphRunner() : null;
      timings.push(stagedRunner ? stagedRunner.lastExecutionMs : wallElapsed);
      publishOutputs(outputs.map((p, index) => ({
        name: p.name,
        values: residentValues(result[index]),
      })));
      acceptCompiledState(activeFeeds, result);
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
      " ms (" + (activeExecutionMode === "staged" ? "WASM-coordinated class" : "contiguous") + " WASM)",
      "good"
    );
    log("ok", "segmented run complete", {
      median_ms: Number(elapsed.toFixed(4)),
      elements: count,
      modules: activeExecutionMode === "staged" ? deployment.modules.length : 1,
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
    const renderFps = Math.max(
      1,
      Number((API.metadata || {}).render_fps)
        || Number((NETWORK.feedback || {}).render_fps)
        || 24
    );
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
    const animated = (continuous || repeats > 1) &&
      (anyExpression || anyGaussian || anyNetwork || HAS_STATE_FEEDBACK);
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
        const refreshed = refreshNonStateFeeds(
          activeFeeds, count, d, frameIndex
        );
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
        publishOutputs(outputs.map((p, i) => ({
          name: p.name,
          values: new View(memory.buffer, offsets[inputs.length + i], count)
        })));
        acceptCompiledState(
          activeFeeds,
          lastOutputs.map(item => item.values)
        );
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

    publishOutputs(outputs.map((p, i) => ({
      name: p.name,
      values: new View(memory.buffer, offsets[inputs.length + i], count)
    })));
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
let outputRevision = 0;

function publishOutputs(nextOutputs) {
  lastOutputs = nextOutputs;
  outputRevision += 1;
}

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
let graphSelectedNode = null;
let graphIndicatorIdsByIdentity = new Map();
let graphNodesById = new Map();
const graphProfileSamples = [];

function hueRgb(hue) {
  const h = ((hue % 360) + 360) % 360 / 60;
  const x = 1 - Math.abs(h % 2 - 1);
  const table = [[1,x,0],[x,1,0],[0,1,x],[0,x,1],[x,0,1],[1,0,x]];
  const rgb = table[Math.floor(h) % 6];
  return rgb.map(value => Math.round((0.18 + value * 0.82) * 255));
}

function mixedIdentityRgb(contributors) {
  const hues = (contributors || []).map(identity =>
    Number((GRAPH_VIEWS.identities[identity] || {hue: 210}).hue)
  );
  if (!hues.length) return [90, 110, 145];
  // Average on the hue circle. Direct RGB averaging turns a rich set of
  // contributors grey; this keeps the provenance blend legible and bright.
  const vector = hues.reduce((sum, hue) => {
    const radians = hue * Math.PI / 180;
    return [sum[0] + Math.cos(radians), sum[1] + Math.sin(radians)];
  }, [0, 0]);
  const mixed = Math.hypot(vector[0], vector[1]) < 1e-6
    ? hues[0]
    : Math.atan2(vector[1], vector[0]) * 180 / Math.PI;
  return hueRgb(mixed);
}

function graphDecayMs() {
  return Math.max(80, Number($("graph-decay") && $("graph-decay").value) || 1200);
}

function graphIndicatorId(viewName, nodeId) {
  return "graph-node-" + viewName + "-" + String(nodeId).replace(/[^a-zA-Z0-9_-]/g, "_");
}

function percentile(sorted, fraction) {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))];
}

function updateGraphProfileStats(now = performance.now()) {
  const pane = $("graph-profile-stats");
  const decay = graphDecayMs();
  const live = graphProfileSamples.filter(sample => now - sample.at <= decay * 4);
  const root = document.documentElement;
  root.style.setProperty("--profile-decay-ms", decay.toFixed(0));
  if (!live.length) {
    root.style.setProperty("--profile-glow", "0%");
    root.style.setProperty("--profile-normalizer-us", "0");
    if (pane) pane.innerHTML = "<span>awaiting execution samples</span><span>τ " +
      decay.toFixed(0) + " ms · half-life " + (decay * Math.LN2).toFixed(0) + " ms</span>";
    return;
  }
  const elapsed = live.map(sample => sample.elapsedMs).sort((a, b) => a - b);
  const perNode = live.map(sample => sample.perNodeUs).sort((a, b) => a - b);
  const median = percentile(elapsed, .5);
  const p95 = percentile(elapsed, .95);
  const normalizerUs = Math.max(.001, percentile(perNode, .95));
  const load = Math.min(100, Math.log1p(p95) * 24);
  root.style.setProperty("--profile-median-ms", median.toFixed(4));
  root.style.setProperty("--profile-p95-ms", p95.toFixed(4));
  root.style.setProperty("--profile-normalizer-us", normalizerUs.toFixed(3));
  root.style.setProperty("--profile-glow", load.toFixed(1) + "%");
  if (pane) pane.innerHTML =
    "<span>window samples " + live.length + "</span>" +
    "<span>median " + median.toFixed(3) + " ms/card</span>" +
    "<span>p95 " + p95.toFixed(3) + " ms/card</span>" +
    "<span>median " + percentile(perNode, .5).toFixed(2) + " μs/node</span>" +
    "<span>phosphor scale p95 " + normalizerUs.toFixed(2) + " μs/node</span>" +
    "<span>latest intensity " + (live[live.length - 1].intensity * 100).toFixed(0) + "%</span>" +
    "<span>latest " + live[live.length - 1].module + "</span>" +
    "<span>" + live[live.length - 1].scope + "</span>" +
    "<span>τ " + decay.toFixed(0) + " ms</span>";
}

function pulseIndicator(element, depositedEnergy, at, decay) {
  if (!element) return;
  const previousAt = Number(element.dataset.profileAt || at);
  const previousEnergy = Number(element.dataset.profileEnergy || 0);
  const energy = previousEnergy * Math.exp(-Math.max(0, at - previousAt) / decay) + depositedEnergy;
  element.dataset.profileAt = String(at);
  element.dataset.profileEnergy = String(energy);
  element.style.setProperty("--node-opacity", Math.min(1, .26 + energy * .48).toFixed(3));
  element.style.setProperty("--node-scale", (1 + Math.min(2.5, energy * 1.3)).toFixed(3));
  element.style.setProperty("--node-blur", Math.min(15, energy * 7).toFixed(2) + "px");
}

function pulseGraphNodes(nodeIds, elapsedMs, moduleName = "program", scope = "method") {
  const at = performance.now();
  const ids = nodeIds || [];
  const measuredMs = Math.max(0, Number(elapsedMs) || 0);
  const perNodeUs = measuredMs * 1000 / Math.max(1, ids.length);
  const decay = graphDecayMs();
  const normalizationWindow = graphProfileSamples
    .filter(sample => at - sample.at <= decay * 4)
    .map(sample => sample.perNodeUs);
  normalizationWindow.push(perNodeUs);
  normalizationWindow.sort((a, b) => a - b);
  const normalizerUs = Math.max(.001, percentile(normalizationWindow, .95));
  const intensity = Math.min(1.5, perNodeUs / normalizerUs);
  const energy = .08 + intensity * .92;
  const targets = new Set();
  const reduced = ((((GRAPH_VIEWS && GRAPH_VIEWS.views) || {}).reduced) || {nodes: []}).nodes || [];
  const reducedById = new Map(reduced.map(node => [String(node.id), node]));
  for (const nodeId of ids) {
    const direct = document.getElementById(graphIndicatorId(activeGraphView, nodeId));
    if (direct) targets.add(direct);
    const reducedNode = reducedById.get(String(nodeId));
    for (const identity of (reducedNode && reducedNode.contributors) || []) {
      for (const elementId of graphIndicatorIdsByIdentity.get(String(identity)) || []) {
        const element = document.getElementById(elementId);
        if (element) targets.add(element);
      }
    }
  }
  targets.forEach(element => pulseIndicator(element, energy, at, decay));
  graphProfileSamples.push({
    at, elapsedMs: measuredMs, perNodeUs, nodeCount: ids.length,
    normalizerUs, intensity, module: moduleName, scope,
  });
  if (graphProfileSamples.length > 768) graphProfileSamples.splice(0, graphProfileSamples.length - 768);
  updateGraphProfileStats(at);
}

function wireProcessGraphIndicators() {
  document.querySelectorAll(".graph-view-button").forEach(button => {
    button.addEventListener("click", () => {
      activeGraphView = button.dataset.graphView;
      graphSelectedNode = null;
      renderGraph();
    });
  });
  const decay = $("graph-decay");
  if (decay) decay.addEventListener("input", () => updateGraphProfileStats(performance.now()));
  const grid = $("process-graph-grid");
  if (grid) grid.addEventListener("click", event => {
    const indicator = event.target.closest(".graph-indicator");
    if (!indicator) return;
    const previous = grid.querySelector('[data-selected="true"]');
    if (previous) previous.removeAttribute("data-selected");
    indicator.setAttribute("data-selected", "true");
    graphSelectedNode = indicator.dataset.nodeId;
    const node = graphNodesById.get(graphSelectedNode) || {};
    const labels = (node.contributors || []).map(identity =>
      (GRAPH_VIEWS.identities[identity] || {label: identity}).label
    );
    $("graph-node-inspector").textContent = graphSelectedNode + " · " + node.type +
      " · level " + node.level + " · group " + node.group + " · " + node.label +
      " · contributors [" + labels.join(", ") + "]";
  });
  updateGraphProfileStats(performance.now());
}

function renderGraph() {
  const target = document.getElementById("graph");
  if (!target) return;
  let html = "";
  if (GRAPH_VIEWS && GRAPH_VIEWS.views) {
    const original = GRAPH_VIEWS.views.original || {nodes: []};
    const reduced = GRAPH_VIEWS.views.reduced || {nodes: []};
    const view = GRAPH_VIEWS.views[activeGraphView] || original;
    const bucketSizes = new Map();
    for (const node of view.nodes) {
      const key = node.level + "::" + node.group;
      bucketSizes.set(key, (bucketSizes.get(key) || 0) + 1);
    }
    const maxBucket = Math.max(1, ...bucketSizes.values());
    const bucketSlots = new Map();
    const levelMin = view.level_min || 0;
    graphIndicatorIdsByIdentity = new Map();
    graphNodesById = new Map();
    const indicators = view.nodes.map(node => {
      const key = node.level + "::" + node.group;
      const slot = bucketSlots.get(key) || 0;
      bucketSlots.set(key, slot + 1);
      const column = node.group * maxBucket + slot + 1;
      const row = node.level - levelMin + 1;
      const id = graphIndicatorId(activeGraphView, node.id);
      const rgb = mixedIdentityRgb(node.contributors);
      graphNodesById.set(String(node.id), node);
      for (const identity of node.contributors || []) {
        const identityKey = String(identity);
        if (!graphIndicatorIdsByIdentity.has(identityKey)) graphIndicatorIdsByIdentity.set(identityKey, []);
        graphIndicatorIdsByIdentity.get(identityKey).push(id);
      }
      return `<div id="${id}" class="graph-indicator" data-node-id="${node.id}" style="grid-column:${column};grid-row:${row};--node-r:${rgb[0]};--node-g:${rgb[1]};--node-b:${rgb[2]}"></div>`;
    }).join("");
    const originalSelected = activeGraphView === "original";
    const graphColumns = Math.max(1, (view.groups || 1) * maxBucket);
    const graphRows = Math.max(1, (view.level_max || 0) - levelMin + 1);
    html += `<div class="graph-toolbar">` +
      `<button class="graph-view-button" data-graph-view="original" aria-pressed="${originalSelected}">Original · ${original.nodes.length}</button>` +
      `<button class="graph-view-button" data-graph-view="reduced" aria-pressed="${!originalSelected}">Reduced · ${reduced.nodes.length}</button>` +
      `<label class="meta">profiling decay τ <input id="graph-decay" type="range" min="80" max="5000" value="1200"> ms</label>` +
      `</div><div class="meta">Unique-ID document indicators: schedule levels flow down, groups run across, and runtime profiling writes explicit CSS values. There is no graph canvas, shader, edge scan, or animation.</div>` +
      `<div id="graph-profile-stats" class="stat graph-profile-stats"><span>awaiting execution samples</span></div>` +
      `<div class="graph-scroll"><div id="process-graph-grid" class="process-graph-grid" style="--graph-columns:${graphColumns};--graph-rows:${graphRows}">` +
      `${indicators}</div></div>` +
      `<div id="graph-node-inspector" class="note node-detail">Click a node to inspect its contributors.</div>`;
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
  const deployment = activeStagedManifest();
  if (deployment && deployment.schedule) {
    const moduleByName = new Map(deployment.modules.map(module => [module.name, module]));
    const levels = new Map();
    for (const node of deployment.schedule.nodes) {
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
      deployment.modules.length + ' method cards coordinated inside WASM over one class memory. Click a node for its ABI.</div>' +
      deploymentRows + '<div id="node-detail" class="note node-detail">Select a punch card.</div>';
  }
  target.innerHTML = html;
  wireProcessGraphIndicators();
  document.querySelectorAll(".deployment-node").forEach(node => {
    const show = () => {
      const spec = deployment.modules.find(module => module.name === node.dataset.module) || {};
      $("node-detail").textContent = spec.name + " · " + (spec.operation_count || 0) +
        " operations · inputs [" + (spec.inputs || []).join(", ") + "] · outputs [" +
        (spec.outputs || []).join(", ") + "] · ProcessGraph nodes [" +
        (spec.node_ids || []).join(", ") + "]";
    };
    node.addEventListener("click", show);
    node.addEventListener("keydown", event => { if (event.key === "Enter") show(); });
  });
}

const pendingDeploymentProfiles = new Map();
let deploymentProfileFrame = 0;

function queueDeploymentProfile(moduleName, elapsedMs, profileScope) {
  const previous = pendingDeploymentProfiles.get(moduleName) || {
    calls: 0, totalMs: 0, scope: profileScope,
  };
  previous.calls += 1;
  previous.totalMs += elapsedMs;
  pendingDeploymentProfiles.set(moduleName, previous);
  if (!deploymentProfileFrame) {
    deploymentProfileFrame = requestAnimationFrame(() => {
      deploymentProfileFrame = 0;
      for (const [name, sample] of pendingDeploymentProfiles) {
        markDeploymentNode(
          name, "done", sample.totalMs / sample.calls, sample.scope,
          sample.calls, sample.totalMs
        );
      }
      pendingDeploymentProfiles.clear();
    });
  }
}

function markDeploymentNode(
  moduleName, state, elapsedMs, profileScope = "method",
  callIncrement = 1, depositedElapsedMs = elapsedMs
) {
  const node = document.querySelector('.deployment-node[data-module="' + CSS.escape(moduleName) + '"]');
  if (!node) return;
  node.dataset.state = state;
  const label = node.querySelector(".node-state");
  const calls = Number(node.dataset.calls || 0) + (
    state === "done" ? callIncrement : 0
  );
  if (state === "done") node.dataset.calls = String(calls);
  const timing = elapsedMs === undefined ? "" : " · " + elapsedMs.toFixed(3) + " ms";
  if (label) label.textContent = state + timing + (calls ? " · " + calls + " calls" : "");
  if (state === "done") {
    const deployment = activeStagedManifest();
    const spec = deployment && deployment.modules.find(module => module.name === moduleName);
    if (spec) pulseGraphNodes(
      spec.node_ids, depositedElapsedMs, moduleName, profileScope
    );
  }
}

function markContiguousState(state, elapsedMs) {
  const label = $("contiguous-state");
  if (!label) return;
  label.textContent = state + (elapsedMs === undefined ? "" : " · " + elapsedMs.toFixed(3) + " ms");
  if (state === "done") {
    const reduced = ((((GRAPH_VIEWS && GRAPH_VIEWS.views) || {}).reduced) || {nodes: []}).nodes || [];
    pulseGraphNodes(
      reduced.map(node => node.id), elapsedMs, "contiguous program", "whole-program amortized"
    );
  }
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
        const response = await fetchResource(descriptor.url);
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

function wireMathematics() {
  const button = $("download-mathematics");
  if (!button || !MATHEMATICS || !MATHEMATICS.url) return;
  button.addEventListener("click", async () => {
    button.disabled = true;
    const old = button.textContent;
    button.textContent = "Downloading…";
    try {
      const response = await fetchResource(MATHEMATICS.url);
      if (!response.ok) throw new Error("HTTP " + response.status);
      const blob = await response.blob();
      const href = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = href;
      link.download = MATHEMATICS.filename || MATHEMATICS.url.split("/").pop();
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(href), 1000);
      log("ok", "exact SymPy model downloaded", {bytes: blob.size});
    } catch (error) {
      log("error", "SymPy model download failed", {message: error.message});
    } finally {
      button.disabled = false;
      button.textContent = old;
    }
  });
}

function wireExecutionModes() {
  const continueButton = $("card-continue");
  if (continueButton) continueButton.addEventListener("click", () => {
    if (releaseCardLatch) releaseCardLatch();
  });
  document.querySelectorAll(".execution-mode").forEach(button => {
    button.addEventListener("click", () => {
      if (running) return;
      activeExecutionMode = button.dataset.mode;
      activeRegionSize = button.dataset.size ? Number(button.dataset.size) : null;
      document.querySelectorAll(".execution-mode").forEach(candidate =>
        candidate.setAttribute("aria-pressed", String(candidate === button))
      );
      if (activeExecutionMode === "staged") {
        const deployment = activeStagedManifest();
        if (!deployment) throw new Error("no deployment for size " + activeRegionSize);
        GRAPH_VIEWS = deployment.graph_views || GRAPH_VIEWS;
        setStatus(activeRegionSize + " operations/card selected; method cards and coordinator download on first run", "good");
      } else {
        setStatus("Mono selected; contiguous compile downloads on first run", "good");
      }
      $("run").disabled = false;
      renderGraph();
    });
  });
}

function renderGallery(items, fromServer = false) {
  const gallery = $("gallery");
  if (!gallery) return;
  gallery.replaceChildren();
  if (!items.length) {
    gallery.textContent = "No versioned program bundles have been prepared yet.";
    gallery.className = "meta";
    return;
  }
  gallery.className = "gallery";
  const programs = new Map();
  for (const item of items) {
    if (!programs.has(item.slug)) programs.set(item.slug, []);
    programs.get(item.slug).push(item);
  }
  programs.forEach(versions => {
    const newest = versions.find(item => item.latest) || versions[0];
    const card = document.createElement("div");
    card.className = "gallery-card";
    const link = document.createElement("a");
    const itemURL = item => fromServer ? serverURL(item.url) : resourceURL(item.url);
    link.href = itemURL(newest);
    link.textContent = newest.title || newest.slug;
    const version = document.createElement("label");
    version.className = "meta";
    version.append((newest.entrypoint || "program") + " · version ");
    const selector = document.createElement("select");
    selector.className = "gallery-version";
    for (const item of versions) {
      const option = document.createElement("option");
      option.value = item.url;
      option.textContent = item.version;
      option.selected = item === newest;
      selector.appendChild(option);
    }
    version.appendChild(selector);
    const detail = document.createElement("span");
    detail.className = "meta";
    const show = item => {
      link.href = itemURL(item);
      detail.textContent = item.artifacts + " artifacts · " + item.bytes + " bytes";
    };
    selector.addEventListener("change", () => {
      const selected = versions.find(item => item.url === selector.value) || newest;
      show(selected);
    });
    show(newest);
    card.append(link, version, detail);
    gallery.appendChild(card);
  });
  return programs.size;
}

async function refreshGallery() {
  const status = $("publisher-status");
  try {
    const response = await fetch(serverURL("/api/gallery"), {cache: "no-store"});
    if (!response.ok) throw new Error("HTTP " + response.status);
    const payload = await response.json();
    const items = payload.items || [];
    const programCount = renderGallery(items, true) || 0;
    if (status) status.textContent = programCount + " program(s) · " +
      items.length + " prepared version(s)";
  } catch (error) {
    const programCount = renderGallery(STATIC_GALLERY) || 0;
    if (status) status.textContent = programCount + " static program(s) · " +
      STATIC_GALLERY.length + " prepared version(s); local server unavailable";
  }
}

function wireLocalPublisher() {
  const address = $("server-address");
  if (!address) return;
  const remembered = localStorage.getItem("turing-server-address");
  if (remembered) address.value = remembered;
  address.addEventListener("change", () => {
    localStorage.setItem("turing-server-address", address.value.trim());
    refreshGallery();
  });
  $("refresh-gallery").addEventListener("click", refreshGallery);
  $("generate-page").addEventListener("click", async () => {
    const file = $("python-source").files[0];
    const status = $("publisher-status");
    if (!file) {
      status.textContent = "Choose a Python file first.";
      return;
    }
    const data = new FormData();
    data.append("source", file, file.name);
    for (const name of ["entrypoint", "title", "slug", "probes"]) {
      const value = $("publish-" + name).value.trim();
      if (value) data.append(name, value);
    }
    const button = $("generate-page");
    button.disabled = true;
    status.textContent = "Compiling and packaging " + file.name + "…";
    try {
      const response = await fetch(serverURL("/api/generate"), {
        method: "POST", body: data
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) throw new Error(payload.error || "HTTP " + response.status);
      status.replaceChildren();
      const link = document.createElement("a");
      link.href = serverURL(payload.url);
      link.textContent = "Open generated page";
      status.append("Prepared ", link, ".");
      await refreshGallery();
    } catch (error) {
      status.textContent = "Generation failed: " + error.message;
    } finally {
      button.disabled = false;
    }
  });
  refreshGallery();
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

function decodePythonBytes(value) {
  if (!value || value.kind !== "bytes") return null;
  const binary = atob(value.base64 || "");
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function runPythonCallable(button) {
  const section = button.closest(".callable-run-system");
  const status = section.querySelector(".python-callable-status");
  const canvas = section.querySelector(".python-callable-canvas");
  const arguments = {};
  try {
    section.querySelectorAll(".python-callable-input").forEach(input => {
      const value = input.value.trim();
      if (value) arguments[input.dataset.parameter] = JSON.parse(value);
    });
  } catch (error) {
    status.textContent = "Invalid JSON input: " + error.message;
    status.className = "python-callable-status out bad";
    return;
  }
  button.disabled = true;
  status.textContent = "Running " + button.dataset.identity + "…";
  status.className = "python-callable-status out";
  try {
    const response = await fetch(serverURL("/api/run"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        source: button.dataset.source,
        callable: button.dataset.identity,
        arguments: arguments,
      }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error || "HTTP " + response.status);
    const result = payload.result || {};
    const items = result.kind === "tuple" || result.kind === "list" ? result.items || [] : [];
    const planes = items.slice(0, 3).map(decodePythonBytes);
    const width = Number(arguments.width || 512);
    const height = Number(arguments.height || 512);
    if (planes.length === 3 && planes.every(plane => plane && plane.length === width * height)) {
      canvas.hidden = false;
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d");
      const image = context.createImageData(width, height);
      for (let i = 0; i < width * height; i++) {
        image.data[i * 4] = planes[0][i];
        image.data[i * 4 + 1] = planes[1][i];
        image.data[i * 4 + 2] = planes[2][i];
        image.data[i * 4 + 3] = 255;
      }
      context.putImageData(image, 0, 0);
      status.textContent = width + "x" + height + " RGB frame returned by " + button.dataset.identity;
    } else {
      canvas.hidden = true;
      status.textContent = JSON.stringify(result).slice(0, 4000);
    }
    status.className = "python-callable-status out good";
  } catch (error) {
    status.textContent = error.message;
    status.className = "python-callable-status out bad";
  } finally {
    button.disabled = false;
  }
}

function wirePythonCallables() {
  document.querySelectorAll(".python-callable-run").forEach(button => {
    button.addEventListener("click", () => runPythonCallable(button));
  });
}

function wireCallableTabs() {
  document.querySelectorAll(".panel").forEach(panel => {
    const ownerTabs = Array.from(panel.querySelectorAll(".callable-owner-tab"));
    const ownerViews = Array.from(panel.querySelectorAll(".callable-owner-view"));
    const activateOwner = tab => {
      const key = tab.dataset.callableOwnerTab;
      ownerTabs.forEach(item => item.setAttribute(
        "aria-selected", String(item === tab)
      ));
      ownerViews.forEach(view => {
        view.hidden = view.dataset.callableOwnerView !== key;
      });
    };
    ownerTabs.forEach(tab => {
      tab.addEventListener("click", () => activateOwner(tab));
      tab.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activateOwner(tab);
        }
      });
    });
  });
  document.querySelectorAll(".callable-group").forEach(group => {
    const tabs = Array.from(group.querySelectorAll(".callable-tab"));
    const views = Array.from(group.querySelectorAll(".callable-tabview"));
    const activate = tab => {
      const key = tab.dataset.callableTab;
      tabs.forEach(item => item.setAttribute(
        "aria-selected", String(item === tab)
      ));
      views.forEach(view => {
        view.hidden = view.dataset.callableView !== key;
      });
    };
    tabs.forEach(tab => {
      tab.addEventListener("click", () => activate(tab));
      tab.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate(tab);
        }
      });
    });
  });
}

function wireClassMapTabs() {
  document.querySelectorAll(".class-map-tabs").forEach(tablist => {
    const panel = tablist.closest(".panel");
    const tabs = Array.from(tablist.querySelectorAll(".class-map-tab"));
    const views = Array.from(panel.querySelectorAll(".class-map-tabview"));
    const activate = tab => {
      const key = tab.dataset.classMapTab;
      tabs.forEach(item => item.setAttribute("aria-selected", String(item === tab)));
      views.forEach(view => { view.hidden = view.dataset.classMapView !== key; });
    };
    tabs.forEach(tab => {
      tab.addEventListener("click", () => activate(tab));
      tab.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate(tab);
        }
      });
    });
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
wireCallableTabs();
wirePythonCallables();
wireClassMapTabs();
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
wireMathematics();
wireExecutionModes();
wireLocalPublisher();
renderGraph();

log("info", "shell ready", {
  entry: entry.symbol,
  parameters: params.length,
  valueType: API.metadata.value_type,
  embedded: Boolean(WASM_BASE64)
});
if (moduleBytes) setStatus("module embedded, ready", "good");
else if (CLASS_GRAPH) setStatus("Choose Mono or a punch-card size; no runtime artifact is loaded.", "good");
if ((API.metadata || {}).autostart && (moduleBytes || CLASS_GRAPH)) {
  requestAnimationFrame(() => {
    window.TuringWasmRuntime.start({continuous: true, preferContiguous: true})
      .catch(error => setStatus(String(error), "bad"));
  });
}
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


_SHADER_EXECUTION_CSS = """
html, body.shader-execution {
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  max-width: none;
  overflow: hidden;
  background: #000;
}
body.shader-execution #shader-surface {
  display: block;
  position: fixed;
  inset: 0;
  z-index: 2147483647;
  width: 100%;
  height: 100%;
  border: 0;
  outline: 0;
  touch-action: none;
}
body.shader-execution #shader-layout-document {
  position: fixed;
  inset: 0;
  z-index: 0;
  width: 100%;
  height: 100%;
  border: 0;
}
"""


_SHADER_EXECUTION_JS = r"""
const SHADER = __SHADER_EXECUTION__;
const canvas = document.getElementById("shader-surface");
const gl = canvas.getContext("webgl2", {
  alpha: false,
  antialias: false,
  depth: false,
  stencil: false,
  preserveDrawingBuffer: false,
});
const input = {
  pointer: [0, 0],
  buttons: 0,
  wheel: [0, 0],
  keys: new Set(),
};

function elementIdentity(element, index) {
  if (element.id) return "#" + element.id;
  const explicit = element.getAttribute("data-turing-identity");
  if (explicit) return explicit;
  const parts = [];
  for (let node = element; node && node.nodeType === 1; node = node.parentElement) {
    const siblings = node.parentElement
      ? Array.from(node.parentElement.children).filter(item => item.tagName === node.tagName)
      : [];
    const ordinal = siblings.length > 1 ? ":nth-of-type(" + (siblings.indexOf(node) + 1) + ")" : "";
    parts.unshift(node.tagName.toLowerCase() + ordinal);
  }
  return parts.join(">") || "element-" + index;
}

function layoutSnapshot(targetDocument = document) {
  const view = targetDocument.defaultView || window;
  const excluded = new Set(["SCRIPT", "STYLE", "LINK", "META", "TITLE", "BASE"]);
  const nodes = Array.from(targetDocument.body
    ? targetDocument.body.querySelectorAll("*") : []
  ).filter(element => !excluded.has(element.tagName) && element.id !== "shader-surface");
  const indexOf = new Map(nodes.map((element, index) => [element, index]));
  const measured = nodes.map((element, index) => {
    const rect = element.getBoundingClientRect();
    const style = view.getComputedStyle(element);
    return {
      index,
      identity: elementIdentity(element, index),
      parent: indexOf.has(element.parentElement) ? indexOf.get(element.parentElement) : -1,
      tag: element.tagName.toLowerCase(),
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
      z_index: Number.isFinite(Number(style.zIndex)) ? Number(style.zIndex) : 0,
      opacity: Number(style.opacity) || 0,
      border_radius: parseFloat(style.borderTopLeftRadius) || 0,
      background: style.backgroundColor,
      color: style.color,
      interactive: element.matches("a,button,input,select,textarea,[tabindex]"),
    };
  }).filter(element => element.width > 0 && element.height > 0);
  const retained = new Map(measured.map((element, index) => [element.index, index]));
  const elements = measured.map((element, index) => ({
    ...element,
    index,
    parent: retained.has(element.parent) ? retained.get(element.parent) : -1,
  }));
  const packed = new Float32Array(elements.length * 8);
  elements.forEach((element, index) => packed.set([
    element.x, element.y, element.width, element.height,
    element.z_index, element.opacity, element.border_radius,
    element.interactive ? 1 : 0,
  ], index * 8));
  return {
    schema: "turing-dom-layout",
    version: 1,
    viewport: {
      width: targetDocument.documentElement.clientWidth,
      height: targetDocument.documentElement.clientHeight,
      device_pixel_ratio: view.devicePixelRatio || 1,
    },
    stride: 8,
    fields: ["x", "y", "width", "height", "z_index", "opacity", "border_radius", "interactive"],
    elements,
    packed,
  };
}

async function settledLayout(targetDocument) {
  if (targetDocument.fonts && targetDocument.fonts.ready) {
    await targetDocument.fonts.ready.catch(() => {});
  }
  const images = Array.from(targetDocument.images || []);
  await Promise.all(images.map(item => item.decode ? item.decode().catch(() => {}) : Promise.resolve()));
  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  return layoutSnapshot(targetDocument);
}

function htmlWithBase(source, baseURL) {
  const parsed = new DOMParser().parseFromString(String(source), "text/html");
  let base = parsed.head.querySelector("base");
  if (!base) {
    base = parsed.createElement("base");
    parsed.head.prepend(base);
  }
  base.href = baseURL;
  return "<!doctype html>\n" + parsed.documentElement.outerHTML;
}

const domLayout = {
  frame: null,
  latest: null,
  listeners: new Set(),
  snapshot(targetDocument = document) {
    this.latest = layoutSnapshot(targetDocument);
    this.listeners.forEach(listener => listener(this.latest));
    return this.latest;
  },
  subscribe(listener) {
    this.listeners.add(listener);
    if (this.latest) listener(this.latest);
    return () => this.listeners.delete(listener);
  },
  async loadHTML(source, {baseURL = document.baseURI} = {}) {
    if (!this.frame) {
      this.frame = document.createElement("iframe");
      this.frame.id = "shader-layout-document";
      this.frame.setAttribute("aria-hidden", "true");
      this.frame.setAttribute("sandbox", "allow-same-origin");
      document.body.insertBefore(this.frame, canvas);
    }
    const loaded = new Promise(resolve => this.frame.addEventListener("load", resolve, {once: true}));
    this.frame.srcdoc = htmlWithBase(source, baseURL);
    await loaded;
    const snapshot = await settledLayout(this.frame.contentDocument);
    this.latest = snapshot;
    this.listeners.forEach(listener => listener(snapshot));
    return snapshot;
  },
  async loadFile(file) {
    if (!file || !/html?/i.test(file.type || file.name || "")) {
      throw new Error("shader layout input must be an HTML file");
    }
    return this.loadHTML(await file.text(), {baseURL: document.baseURI});
  },
};
const liaison = {
  role: SHADER.role,
  canvas,
  gl,
  input,
  io: SHADER.io || (window.TuringWasmRuntime && window.TuringWasmRuntime.io) || null,
  wasm: window.TuringWasmRuntime || null,
  dom: domLayout,
  ready: null,
};

function fail(error) {
  const message = error instanceof Error ? error.message : String(error);
  canvas.dataset.error = message;
  canvas.title = message;
  console.error("shader execution failed", error);
}

function capturePointer(event) {
  canvas.focus({preventScroll: true});
  input.pointer[0] = event.offsetX;
  input.pointer[1] = canvas.clientHeight - event.offsetY;
  input.buttons = event.buttons;
}

canvas.addEventListener("pointerdown", event => {
  capturePointer(event);
  canvas.setPointerCapture(event.pointerId);
  event.preventDefault();
});
canvas.addEventListener("pointermove", capturePointer);
canvas.addEventListener("pointerup", event => {
  capturePointer(event);
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  event.preventDefault();
});
canvas.addEventListener("pointercancel", event => {
  input.buttons = 0;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
});
canvas.addEventListener("wheel", event => {
  input.wheel[0] += event.deltaX;
  input.wheel[1] += event.deltaY;
  event.preventDefault();
}, {passive: false});
canvas.addEventListener("keydown", event => {
  input.keys.add(event.code);
  event.preventDefault();
});
canvas.addEventListener("keyup", event => {
  input.keys.delete(event.code);
  event.preventDefault();
});
canvas.addEventListener("contextmenu", event => event.preventDefault());
canvas.addEventListener("dragover", event => event.preventDefault());
canvas.addEventListener("drop", event => {
  event.preventDefault();
  const file = event.dataTransfer && event.dataTransfer.files[0];
  if (file) domLayout.loadFile(file).catch(fail);
});
canvas.focus({preventScroll: true});

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const report = gl.getShaderInfoLog(shader) || "unknown shader compiler error";
    gl.deleteShader(shader);
    throw new Error(report);
  }
  return shader;
}

const vertexSource = `#version 300 es
precision highp float;
const vec2 TURING_TRIANGLE[3] = vec2[3](
  vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0)
);
void main() {
  gl_Position = vec4(TURING_TRIANGLE[gl_VertexID], 0.0, 1.0);
}`;

if (!gl) {
  fail(new Error("WebGL 2 is required by this execution page"));
} else {
  const ready = (async () => {
    const response = await fetch(new URL(SHADER.url, document.baseURI), {cache: "no-store"});
    if (!response.ok) throw new Error("shader load failed: HTTP " + response.status);
    const fragmentSource = await response.text();
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || "shader link failed");
    }
    gl.useProgram(program);
    const configuration = SHADER.configuration || {};
    if (configuration.document_url) {
      const documentResponse = await fetch(
        new URL(configuration.document_url, document.baseURI),
        {cache: "no-store"},
      );
      if (!documentResponse.ok) {
        throw new Error("shader document load failed: HTTP " + documentResponse.status);
      }
      await domLayout.loadHTML(await documentResponse.text(), {
        baseURL: new URL(configuration.document_url, document.baseURI).href,
      });
    } else {
      domLayout.latest = await settledLayout(document);
    }

    const feedNames = [...fragmentSource.matchAll(/uniform\s+sampler2D\s+(turing_feed_\d+)\s*;/g)]
      .map(match => match[1]);
    const outputFeedBindings = configuration.output_feed_bindings || {};
    const feeds = feedNames.map((name, index) => {
      const texture = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + index);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      const outputName = outputFeedBindings[name] || null;
      // R32F sampling is core in WebGL 2, but LINEAR filtering of float
      // textures requires OES_texture_float_linear.  Generated presentation
      // feeds must work without that optional extension, so retain the core
      // NEAREST contract.  The full-screen surface still stretches them.
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.R32F, 1, 1, 0,
        gl.RED, gl.FLOAT, new Float32Array([0])
      );
      gl.uniform1i(gl.getUniformLocation(program, name), index);
      return {texture, index, width: 0, height: 0, outputName, revision: -1};
    });

    const domSurfaceEnabled = configuration.dom_surface === true;
    const domTextureLocation = gl.getUniformLocation(program, "turing_dom_state");
    const domCountLocation = gl.getUniformLocation(program, "turing_dom_count");
    const resolutionLocation = gl.getUniformLocation(program, "turing_resolution");
    const pointerLocation = gl.getUniformLocation(program, "turing_pointer");
    const timeLocation = gl.getUniformLocation(program, "turing_time");
    const domTexture = domTextureLocation === null ? null : gl.createTexture();
    const domTextureUnit = feeds.length;
    if (domTexture) {
      gl.activeTexture(gl.TEXTURE0 + domTextureUnit);
      gl.bindTexture(gl.TEXTURE_2D, domTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.uniform1i(domTextureLocation, domTextureUnit);
    }

    // A presentation shader may consume the latest logical RGB frame emitted
    // by its companion Wasm program.  This is deliberately a display liaison:
    // computation remains in Wasm and the shader only samples the published
    // output texture across its full-screen triangle.
    const outputTextureConfiguration = configuration.output_texture || null;
    const outputTextureLocation = gl.getUniformLocation(
      program, "turing_output_texture"
    );
    const outputTextureUnit = feeds.length + (domTexture ? 1 : 0);
    const outputTexture = outputTextureConfiguration && outputTextureLocation !== null
      ? gl.createTexture() : null;
    let outputTextureRevision = -1;
    if (outputTexture) {
      gl.activeTexture(gl.TEXTURE0 + outputTextureUnit);
      gl.bindTexture(gl.TEXTURE_2D, outputTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA,
        gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255])
      );
      gl.uniform1i(outputTextureLocation, outputTextureUnit);
    }

    function uploadOutputTexture() {
      if (!outputTexture || !liaison.wasm || !liaison.wasm.outputFrame) return;
      const frame = liaison.wasm.outputFrame();
      if (!frame || frame.revision === outputTextureRevision || !frame.outputs.length) return;
      const channelNames = Array.isArray(outputTextureConfiguration.channels)
        ? outputTextureConfiguration.channels : ["red", "green", "blue"];
      const byName = new Map(frame.outputs.map(item => [item.name, item.values]));
      const channels = channelNames.slice(0, 3).map(name => byName.get(name));
      if (channels.length !== 3 || channels.some(values => !values)) return;
      const count = frame.width * frame.height;
      if (channels.some(values => values.length < count)) return;
      const pixels = new Uint8Array(count * 4);
      for (let index = 0; index < count; index += 1) {
        const base = index * 4;
        pixels[base] = Math.max(0, Math.min(255, Math.round(channels[0][index])));
        pixels[base + 1] = Math.max(0, Math.min(255, Math.round(channels[1][index])));
        pixels[base + 2] = Math.max(0, Math.min(255, Math.round(channels[2][index])));
        pixels[base + 3] = 255;
      }
      gl.activeTexture(gl.TEXTURE0 + outputTextureUnit);
      gl.bindTexture(gl.TEXTURE_2D, outputTexture);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA, frame.width, frame.height, 0,
        gl.RGBA, gl.UNSIGNED_BYTE, pixels
      );
      outputTextureRevision = frame.revision;
    }

    function cssColor(value) {
      const components = String(value || "").match(/[\d.]+/g) || [];
      return [0.16, 0.34, 0.64, 1.0].map((fallback, index) =>
        components[index] === undefined
          ? fallback
          : Number(components[index]) / (index < 3 ? 255 : 1)
      );
    }

    let surfaceState = null;
    function initializeSurfaceState(snapshot) {
      const elements = snapshot.elements.slice(0, 256);
      const values = key => new Float64Array(elements.map(element => element[key]));
      const extentX = values("width");
      const extentY = values("height");
      const anchorX = new Float64Array(elements.map(element => element.x + element.width * 0.5));
      // DOM rectangles are top-origin; WebGL fragment coordinates are
      // bottom-origin.  Normalize once at the liaison boundary so physics,
      // pointer hit testing, and shader geometry all share one coordinate
      // system.
      const anchorY = new Float64Array(elements.map(element =>
        snapshot.viewport.height - (element.y + element.height * 0.5)
      ));
      surfaceState = {
        snapshot,
        elements,
        anchorX,
        anchorY,
        extentX,
        extentY,
        positionX: anchorX.slice(),
        positionY: anchorY.slice(),
        velocityX: new Float64Array(elements.length),
        velocityY: new Float64Array(elements.length),
        activity: new Float64Array(elements.length),
        lastMilliseconds: null,
      };
      return surfaceState;
    }

    function uploadDomState(state) {
      if (!domTexture) return;
      const count = state.elements.length;
      const packed = new Float32Array(Math.max(1, count) * 12);
      state.elements.forEach((element, index) => {
        const color = cssColor(element.background);
        const base = index * 12;
        packed.set([
          state.positionX[index], state.positionY[index],
          state.extentX[index] * 0.5, state.extentY[index] * 0.5,
          element.z_index + index * 0.025, state.activity[index],
          element.border_radius, element.interactive ? 1 : 0,
          color[0], color[1], color[2], color[3],
        ], base);
      });
      gl.activeTexture(gl.TEXTURE0 + domTextureUnit);
      gl.bindTexture(gl.TEXTURE_2D, domTexture);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA32F, 3, Math.max(1, count), 0,
        gl.RGBA, gl.FLOAT, packed
      );
      if (domCountLocation !== null) gl.uniform1i(domCountLocation, count);
    }

    async function advanceDomSurface(milliseconds) {
      const snapshot = domLayout.latest;
      if (!snapshot) return;
      if (!surfaceState || surfaceState.snapshot !== snapshot) {
        initializeSurfaceState(snapshot);
      }
      const state = surfaceState;
      const count = state.elements.length;
      if (!count || !liaison.wasm) {
        uploadDomState(state);
        return;
      }
      const elapsed = state.lastMilliseconds === null
        ? 1 / 60
        : Math.min(1 / 20, Math.max(1 / 240, (milliseconds - state.lastMilliseconds) / 1000));
      state.lastMilliseconds = milliseconds;
      const fill = value => new Float64Array(count).fill(value);
      const result = await liaison.wasm.run({
        position_x: state.positionX,
        position_y: state.positionY,
        velocity_x: state.velocityX,
        velocity_y: state.velocityY,
        anchor_x: state.anchorX,
        anchor_y: state.anchorY,
        extent_x: state.extentX,
        extent_y: state.extentY,
        pointer_x: fill(input.pointer[0]),
        pointer_y: fill(canvas.clientHeight - input.pointer[1]),
        pointer_buttons: fill(input.buttons),
        dt: fill(elapsed),
      }, count);
      [state.positionX, state.positionY, state.velocityX,
        state.velocityY, state.activity] = result;
      uploadDomState(state);
    }

    function uploadFeeds(time) {
      const width = canvas.width;
      const height = canvas.height;
      const outputFrame = liaison.wasm && liaison.wasm.outputFrame
        ? liaison.wasm.outputFrame() : null;
      const outputsByName = outputFrame
        ? new Map(outputFrame.outputs.map(item => [item.name, item.values]))
        : new Map();
      for (const feed of feeds) {
        if (feed.outputName) {
          const source = outputsByName.get(feed.outputName);
          if (!source || !outputFrame || feed.revision === outputFrame.revision) continue;
          const count = outputFrame.width * outputFrame.height;
          if (source.length < count) continue;
          const values = Float32Array.from(source.slice(0, count));
          gl.activeTexture(gl.TEXTURE0 + feed.index);
          gl.bindTexture(gl.TEXTURE_2D, feed.texture);
          gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
          gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.R32F, outputFrame.width, outputFrame.height,
            0, gl.RED, gl.FLOAT, values
          );
          feed.revision = outputFrame.revision;
          continue;
        }
        const values = new Float32Array(width * height);
        for (let y = 0; y < height; y += 1) {
          for (let x = 0; x < width; x += 1) {
            const channel = feed.index % 8;
            values[y * width + x] = channel === 0 ? x / Math.max(1, width - 1)
              : channel === 1 ? y / Math.max(1, height - 1)
              : channel === 2 ? time
              : channel === 3 ? input.pointer[0] / Math.max(1, canvas.clientWidth)
              : channel === 4 ? input.pointer[1] / Math.max(1, canvas.clientHeight)
              : channel === 5 ? input.buttons
              : channel === 6 ? input.wheel[1]
              : input.keys.size;
          }
        }
        gl.activeTexture(gl.TEXTURE0 + feed.index);
        gl.bindTexture(gl.TEXTURE_2D, feed.texture);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, values);
      }
    }

    let lastInputSignature = "";
    async function frame(milliseconds) {
      const ratio = Math.max(1, window.devicePixelRatio || 1);
      const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
      const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
      const resized = canvas.width !== width || canvas.height !== height;
      if (resized) {
        canvas.width = width;
        canvas.height = height;
        gl.viewport(0, 0, width, height);
      }
      const time = milliseconds / 1000;
      if (resolutionLocation !== null) gl.uniform2f(resolutionLocation, width, height);
      if (pointerLocation !== null) gl.uniform2f(
        pointerLocation,
        input.pointer[0] * ratio,
        (canvas.clientHeight - input.pointer[1]) * ratio,
      );
      if (timeLocation !== null) gl.uniform1f(timeLocation, time);
      if (domSurfaceEnabled) await advanceDomSurface(milliseconds);
      uploadOutputTexture();
      const signature = [width, height, ...input.pointer, input.buttons,
        ...input.wheel, input.keys.size, Math.floor(time * 60)].join(":");
      if (resized || signature !== lastInputSignature) {
        uploadFeeds(time);
        lastInputSignature = signature;
      }
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      requestAnimationFrame(value => frame(value).catch(fail));
    }
    requestAnimationFrame(value => frame(value).catch(fail));
    if (!domSurfaceEnabled && SHADER.autostart !== false && liaison.wasm) {
      const execution = SHADER.execution || {};
      liaison.wasm.start({
        continuous: execution.continuous !== false,
        preferContiguous: execution.prefer_contiguous !== false,
      }).catch(fail);
    }
    return {canvas, gl, program, input, fragmentSource};
  })();
  liaison.ready = ready;
  window.TuringShaderLiaison = liaison;
  // Compatibility name for callers of the first shader-surface probe.
  window.TuringShaderSurface = liaison;
  ready.catch(fail);
}
"""


def _map_ir_mapping(map_ir: Any) -> dict[str, Any]:
    if map_ir is None:
        return {}
    mapping = dict(map_ir)
    navigation = mapping.get("class_navigation")
    if hasattr(navigation, "to_mapping"):
        mapping["class_navigation"] = navigation.to_mapping()
    return mapping


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


def _callable_systems_panel(map_mapping: Mapping[str, Any]) -> str:
    """Render one complete run section per method, then module function."""

    systems = dict(map_mapping.get("callable_systems") or {})
    classes = list(systems.get("classes", ()))
    file_scope = dict(systems.get("file_scope") or {})
    functions = list(file_scope.get("functions", systems.get("functions", ())))
    symbols = list(file_scope.get("symbols", ()))
    if not classes and not functions and not symbols:
        return ""

    def run_system(item: Mapping[str, Any]) -> str:
        identity = str(item.get("identity", item.get("name", "callable")))
        parameters = list(item.get("parameters", ()))
        prefix = re.sub(r"[^a-zA-Z0-9_-]+", "-", identity).strip("-") or "callable"
        inputs = "".join(
            '<label>' + _escape(str(parameter.get("name", "input")))
            + '<input class="python-callable-input" data-parameter="'
            + _escape(str(parameter.get("name", "input")))
            + '" type="text" id="callable-' + _escape(prefix) + "-"
            + _escape(str(parameter.get("name", "input")))
            + '" value="' + _escape(str(parameter.get("default", "")))
            + '" placeholder="JSON value or numeric sequence"></label>'
            for parameter in parameters
        ) or '<div class="meta">This callable declares no inputs.</div>'
        reference = item.get("function_reference")
        graph_state = (
            f"ProcessGraph function reference {reference} retained"
            if reference is not None
            else "structural signature retained; no function-table body"
        )
        page_url = str(item.get("page_url") or "")
        page_link = (
            f'<a class="button" href="{_escape(page_url)}">Open generated callable page</a>'
            if page_url else '<span class="meta">No separate callable page was requested.</span>'
        )
        python_source_url = str(item.get("python_source_url") or "")
        if python_source_url:
            run_button = (
                '<button type="button" class="python-callable-run" data-source="'
                + _escape(python_source_url) + '" data-identity="' + _escape(identity)
                + '">Run ' + _escape(identity) + '</button>'
            )
            runtime_state = "trusted local Python runtime ready"
        else:
            run_button = (
                '<button type="button" disabled title="No executable artifact has been '
                'emitted for this callable yet">Run ' + _escape(identity) + '</button>'
            )
            runtime_state = "executable backend pending"
        return (
            '<section class="callable-run-system" data-callable="' + _escape(identity) + '">'
            '<div class="method-title"><code>' + _escape(str(item.get("signature") or identity))
            + '</code></div><div class="callable-inputs">' + inputs + '</div>'
            '<div class="row">' + run_button
            + page_link + '<div class="grow meta">' + _escape(graph_state)
            + '; ' + _escape(runtime_state) + '</div></div>'
            '<div class="python-callable-status out"></div>'
            '<canvas class="python-callable-canvas" width="1" height="1" hidden></canvas>'
            '</section>'
        )

    def tabbed_group(
        label: str,
        items: Sequence[Mapping[str, Any]],
        scope_symbols: Sequence[Mapping[str, Any]] = (),
    ) -> str:
        if not items and not scope_symbols:
            return (
                '<div class="callable-group"><div class="callable-group-title">'
                + _escape(label) + '</div><div class="meta">No explicit methods declared.</div></div>'
            )
        tabs = []
        views = []
        for index, item in enumerate(items):
            identity = str(item.get("identity", item.get("name", index)))
            key = re.sub(r"[^a-zA-Z0-9_-]+", "-", identity).strip("-") or str(index)
            selected = "true" if index == 0 else "false"
            hidden = "" if index == 0 else " hidden"
            tabs.append(
                '<div class="callable-tab" role="tab" tabindex="0" '
                f'data-callable-tab="{_escape(key)}" aria-selected="{selected}">'
                + _escape(str(item.get("name", identity))) + "</div>"
            )
            views.append(
                f'<div class="callable-tabview" data-callable-view="{_escape(key)}"{hidden}>'
                + run_system(item) + "</div>"
            )
        if scope_symbols:
            selected = "true" if not items else "false"
            hidden = "" if not items else " hidden"
            tabs.append(
                '<div class="callable-tab" role="tab" tabindex="0" '
                f'data-callable-tab="file-symbols" aria-selected="{selected}">symbols</div>'
            )
            symbol_rows = "".join(
                '<div class="kv"><b>' + _escape(str(symbol.get("name", "symbol")))
                + '</b><span>' + _escape(str(symbol.get("kind", "binding")))
                + '</span><span class="meta">' + _escape(str(symbol.get("expression", "")))
                + '</span></div>'
                for symbol in scope_symbols
            )
            views.append(
                f'<div class="callable-tabview" data-callable-view="file-symbols"{hidden}>'
                '<section class="callable-run-system" data-callable="file-symbols">'
                '<div class="method-title"><code>file symbols</code></div>'
                '<div class="meta">Global constants, aliases, imports, and symbolic bindings '
                'retained at module scope.</div>' + symbol_rows + '</section></div>'
            )
        return (
            '<div class="callable-group"><div class="callable-group-title">'
            + _escape(label) + '</div><div class="callable-tabs" role="tablist">'
            + "".join(tabs) + "</div>" + "".join(views) + "</div>"
        )

    owner_groups: list[
        tuple[str, str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
    ] = []
    if functions or symbols:
        owner_groups.append(("file scope", "file-scope", functions, symbols))
    for record in classes:
        methods = list(record.get("methods", ()))
        label = "class " + str(record.get("identity", "class"))
        key = re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-") or "class"
        owner_groups.append((label, key, methods, ()))

    owner_tabs = []
    owner_views = []
    for index, (label, key, items, scope_symbols) in enumerate(owner_groups):
        selected = "true" if index == 0 else "false"
        hidden = "" if index == 0 else " hidden"
        owner_tabs.append(
            '<div class="callable-owner-tab" role="tab" tabindex="0" '
            f'data-callable-owner-tab="{_escape(key)}" aria-selected="{selected}">'
            + _escape(label) + "</div>"
        )
        owner_views.append(
            f'<div class="callable-owner-view" data-callable-owner-view="{_escape(key)}"{hidden}>'
            + tabbed_group(label, items, scope_symbols) + "</div>"
        )
    return (
        '<div class="panel"><div class="panel-title">Callable run systems</div>'
        '<div class="meta">File-level functions and symbols are grouped under file scope; '
        'classes follow in source order. Each callable keeps independent inputs and a '
        'separately generated inspection page.</div><div class="callable-owner-tabs" '
        'role="tablist">' + "".join(owner_tabs) + "</div>"
        + "".join(owner_views) + "</div>"
    )


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


def _mathematics_panel(mathematics: Mapping[str, Any] | None) -> str:
    """Show semantic chalkboard notation; keep exact SSA relations click-lazy."""

    if not mathematics:
        return ""
    depiction = dict(mathematics.get("depiction") or {})
    kind = str(depiction.get("kind") or "relation")
    name = _escape(str(depiction.get("name") or "program"))
    inputs = [_escape(str(value)) for value in depiction.get("inputs", ())]
    outputs = [_escape(str(value)) for value in depiction.get("outputs", ())]
    input_count = len(inputs)
    output_count = len(outputs)
    input_tuple = "<mo>,</mo>".join(f"<mi>{value}</mi>" for value in inputs)
    output_tuple = "<mo>,</mo>".join(f"<mi>{value}</mi>" for value in outputs)
    domain = f"<msup><mi>ℝ</mi><mn>{input_count}</mn></msup>"
    codomain = f"<msup><mi>ℝ</mi><mn>{output_count}</mn></msup>"

    if kind == "function":
        title = "Deterministic numeric map"
        statement = (
            f"<mi>{name}</mi><mo>:</mo>{domain}<mo>→</mo>{codomain}"
            f"<mspace width='1.2em'/><mi>{name}</mi><mo>(</mo>{input_tuple}<mo>)</mo>"
            f"<mo>=</mo><mo>[</mo>{output_tuple}<mo>]</mo>"
        )
        interpretation = (
            "The compiled program is one map from its public inputs to its named outputs. "
            "Its inverse problem is the preimage of a desired output, not a page of SSA assignments."
        )
        secondary = (
            f"<mrow><mo>{{</mo><mi>x</mi><mo>∈</mo>{domain}<mo>|</mo>"
            f"<mi>{name}</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo>"
            "<msup><mi>y</mi><mo>*</mo></msup><mo>}}</mo></mrow>"
        )
    elif kind == "predicate":
        title = "Boolean predicate"
        statement = f"<mi>{name}</mi><mo>:</mo>{domain}<mo>→</mo><mi>𝔹</mi>"
        interpretation = "The program defines the set of public inputs for which its predicate is true."
        secondary = (
            f"<mi>S</mi><mo>=</mo><mo>{{</mo><mi>x</mi><mo>∈</mo>{domain}<mo>|</mo>"
            f"<mi>{name}</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>}}</mo>"
        )
    elif kind == "transition":
        title = "State transition"
        statement = (
            f"<mo>(</mo><msub><mi>s</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub>"
            f"<mo>,</mo><msub><mi>y</mi><mi>k</mi></msub><mo>)</mo><mo>=</mo>"
            f"<mi>{name}</mi><mo>(</mo><msub><mi>s</mi><mi>k</mi></msub><mo>,</mo>"
            f"<msub><mi>x</mi><mi>k</mi></msub><mo>)</mo>"
        )
        interpretation = "The program advances state and emits observations one schedule step at a time."
        secondary = ""
    else:
        title = "Implicit relation"
        statement = f"<msub><mi>R</mi><mi>{name}</mi></msub><mo>⊆</mo>{domain}<mo>×</mo>{codomain}"
        interpretation = "The program is presented as a relation because no narrower semantic shape applies."
        secondary = ""

    second_line = (
        f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow>{secondary}</mrow></math>'
        if secondary else ""
    )
    operation_count = len(mathematics.get("uninterpreted", ()))
    return f"""
  <div class="panel">
    <div class="panel-title">Mathematics</div>
    <div class="note"><strong>Math is programming is math.</strong> This is
      the reduced program selected through the existing SymPy target, not a
      separately handwritten formula.</div>
    <div class="stat">
      <span>{_escape(str(mathematics.get("node_count", 0)))} symbolic nodes</span>
      <span>{_escape(str(mathematics.get("equation_count", 0)))} equations</span>
      <span>{_escape(str(mathematics.get("constraint_count", 0)))} constraints</span>
      <span>{operation_count} explicit uninterpreted operations</span>
      <span>SymPy &rarr; native MathML</span>
    </div>
    <div class="chalkboard">
      <div class="meta">{title}</div>
      <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
        <mrow>{statement}</mrow>
      </math>
      {second_line}
      <div class="note">{interpretation}</div>
    </div>
    <div class="math-browser">
      <button id="download-mathematics">Download exact SymPy model</button>
      <div class="meta">The exact reduced ProcessGraph, including its SSA relations,
        is not loaded by this page. It is fetched only after you request the file.</div>
    </div>
  </div>"""


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
    mathematics: Mapping[str, Any] | None = None,
    network_manifest: Mapping[str, Any] | None = None,
    map_ir: Mapping[str, Any] | None = None,
    class_graph: Mapping[str, Any] | None = None,
    graph_views: Mapping[str, Any] | None = None,
    resource_route: str = "/",
    static_gallery: Sequence[Mapping[str, Any]] | None = None,
    shader_execution: Mapping[str, Any] | None = None,
    audio_runtime: Mapping[str, Any] | None = None,
    default_server_address: str = "http://localhost:8787",
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

    shader_css = ""
    shader_canvas = ""
    shader_script = ""
    audio_script = ""
    body_class = ""
    if shader_execution is not None:
        shader = dict(shader_execution)
        if not shader.get("url"):
            raise ValueError("shader execution requires a published shader URL")
        if shader.get("role") != "shader-surface":
            raise ValueError("shader execution requires the shader-surface role")
        # Keep the complete inspection document available for tooling and
        # diagnostics, but graduate its presentation to the WebGL surface.
        # CSS owns the visibility switch exactly so removing this class is a
        # sufficient way to inspect the same generated page again.
        body_class = ' class="shader-execution"'
        shader_css = _SHADER_EXECUTION_CSS
        shader_canvas = (
            '<canvas id="shader-surface" tabindex="0" '
            'aria-label="WebGL shader execution surface"></canvas>'
        )
        shader_script = "<script>" + _SHADER_EXECUTION_JS.replace(
            "__SHADER_EXECUTION__", json.dumps(shader, default=str)
        ) + "</script>"
    if audio_runtime is not None:
        audio_script = "<script>" + _AUDIO_RUNTIME_JS.replace(
            "__AUDIO_RUNTIME__", json.dumps(dict(audio_runtime), default=str)
        ) + "</script>"

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
    map_mapping = _map_ir_mapping(map_ir)
    script = (
        _JS.replace("__API__", json.dumps(mapping))
        .replace("__WASM__", encoded)
        .replace("__GRAPH__", json.dumps(graph_mapping, default=str))
        .replace("__GRAPH_VIEWS__", json.dumps(dict(graph_views or {}), default=str))
        .replace("__NETWORK__", json.dumps(network_mapping, default=str))
        .replace("__MAP_IR__", json.dumps(map_mapping, default=str))
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
        .replace("__MATHEMATICS__", json.dumps(dict(mathematics or {}), default=str))
        .replace("__RESOURCE_ROUTE__", json.dumps(str(resource_route)))
        .replace("__STATIC_GALLERY__", json.dumps(list(static_gallery or []), default=str))
        .replace("__DEFAULT_SERVER_ADDRESS__", json.dumps(str(default_server_address)))
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
        class_graph
        and (
            any(module.get("url") for module in class_graph.get("modules", ()))
            or dict(class_graph).get("contiguous", {}).get("url")
        )
    )
    if external_class_graph:
        banner = (
            '<div class="note good">Versioned deployment manifest loaded. '
            'WebAssembly regions, the contiguous compile, and language source '
            'files remain unloaded until their corresponding run or download action.</div>'
        )
        picker = ""
        disabled = " disabled"
    elif wasm_bytes or class_graph:
        banner = (
            '<div class="note good">Binary embedded &mdash; this file is '
            "self-contained and runs offline.</div>"
        )
        picker = ""
        disabled = " disabled" if class_graph else ""
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
    mathematics_html = _mathematics_panel(mathematics)
    callable_systems_html = _callable_systems_panel(map_mapping)
    class_records = list(
        dict(map_mapping.get("class_navigation") or {}).get("classes", ())
    )
    mapped_graphs = list(map_mapping.get("graphs", ()))
    class_map_html = (
        '<div class="panel"><div class="panel-title">Class map and navigation LUT</div>'
        '<div class="meta">Retained class space, graph/function identities, permissions, '
        'dependency regions, and SSA class-navigation methods.</div>'
        '<div class="class-map-tabs" role="tablist">'
        '<div class="class-map-tab" role="tab" tabindex="0" data-class-map-tab="summary" '
        'aria-selected="true">Summary</div>'
        '<div class="class-map-tab" role="tab" tabindex="0" data-class-map-tab="raw" '
        'aria-selected="false">Raw map IR</div></div>'
        '<div class="class-map-tabview" data-class-map-view="summary">'
        f'<div class="stat"><span>{len(class_records)} classes</span>'
        f'<span>{len(mapped_graphs)} callable graphs</span></div></div>'
        '<div class="class-map-tabview" data-class-map-view="raw" hidden>'
        f'<pre id="class-map">{_escape(json.dumps(map_mapping, indent=2, default=str))}</pre></div>'
        '</div>'
        if map_mapping
        else '<div class="panel"><div class="panel-title">Class map and navigation LUT</div>'
        '<div class="meta">This compilation carries no class-map entries.</div></div>'
    )

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
            'Mono · contiguous</button>'
            if contiguous else ""
        )
        variants = dict(class_graph).get("variants", {})
        staged_buttons = "".join(
            '<button class="execution-mode" data-mode="staged" '
            f'data-size="{_escape(str(size))}" aria-pressed="false">'
            f'{_escape(str(size))} operations/card</button>'
            for size in sorted(variants, key=lambda value: int(value))
        )
        execution_modes_html = (
            '<div class="panel"><div class="panel-title">Execution shape</div>'
            '<div class="meta">Choose how the same object and API is deployed. Nothing '
            'is selected by default, and no runtime artifact downloads before a choice '
            'and first run. Size controls the maximum lowered operations in each method card.</div>'
            '<div class="execution-modes">' + contiguous_button + staged_buttons + '</div>'
            '<div class="stat"><span>class memory · field-slot table · WASM method inventory</span>'
            '<span id="contiguous-state">contiguous not loaded</span></div>'
            '<div class="execution-modes"><label class="meta"><input id="card-latch" '
            'type="checkbox"> break at every method-card boundary</label>'
            '<button id="card-continue" disabled>Release latch</button></div></div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(shell_name)}</title>
<style>{_CSS}{shader_css}</style>
</head>
<body{body_class}>
  {shader_canvas}
  <div class="title">{_escape(str(mapping["module"]))}</div>
  <div class="sub">{_escape(str(mapping["language"]))} &middot; entry
    <code>{_escape(str(entry["name"]))}</code> &middot;
    {len(parameters)} parameters</div>

  <div id="fatal" class="note" hidden></div>
  {banner}
  {note_html}

  <div class="panel">
    <div class="panel-title">Local publisher and gallery</div>
    <div class="meta">The loopback Go server can compile a trusted Python file,
      package every generated artifact together, and make the versioned page
      discoverable from the site tree. Uploaded Python is compiler input and
      must be treated as trusted local code.</div>
    <div class="server-controls">
      <label class="wide">server address
        <input type="text" id="server-address" value="{_escape(default_server_address)}"
          spellcheck="false"></label>
      <label class="wide">Python source
        <input type="file" id="python-source" accept=".py,text/x-python"></label>
      <label>entrypoint (optional)
        <input type="text" id="publish-entrypoint" placeholder="render or kernel"></label>
      <label>title (optional)
        <input type="text" id="publish-title" placeholder="Gallery title"></label>
      <label>slug (optional)
        <input type="text" id="publish-slug" placeholder="url-name"></label>
      <label>probe values JSON (optional)
        <input type="text" id="publish-probes" placeholder='{{"gain": [1.0]}}'></label>
    </div>
    <div class="row">
      <button id="generate-page">Generate page</button>
      <button id="refresh-gallery">Refresh gallery</button>
      <div class="grow"><div id="publisher-status" class="out"></div></div>
    </div>
    <div id="gallery" class="meta">Looking for prepared pages…</div>
  </div>

  {callable_systems_html}

  <div class="panel">
    <div class="panel-title">Signature</div>
    {_signature_rows(parameters)}
  </div>

  {execution_modes_html}

  {class_map_html}

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

{mathematics_html}

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

{audio_script}
<script>{boot_script}</script>
<script>{script}</script>
{shader_script}
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
    mathematics: Mapping[str, Any] | None = None,
    network_manifest: Mapping[str, Any] | None = None,
    map_ir: Mapping[str, Any] | None = None,
    resource_route: str = "/",
    shader_execution: Mapping[str, Any] | None = None,
    audio_runtime: Mapping[str, Any] | None = None,
    default_server_address: str = "http://localhost:8787",
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
        mathematics=mathematics,
        map_ir=map_ir,
        resource_route=resource_route,
        shader_execution=shader_execution,
        audio_runtime=audio_runtime,
        default_server_address=default_server_address,
    )


__all__ = ["HtmlShell", "emit_html_shell", "shell_for_artifact"]
