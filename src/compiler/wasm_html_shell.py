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

function domain() {
  const w = Math.max(1, Number($("dom_w").value) | 0);
  const h = Math.max(1, Number($("dom_h").value) | 0);
  return { w: w, h: h, n: w * h };
}

// Compiled once per run rather than per element: a 256,000-element feed
// evaluated through a fresh Function each time is the difference between a
// responsive page and a hung one.
function feedValues(param, n, d) {
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
    fn = new Function("i", "x", "y", "w", "h", "return (" + body + ");");
  } catch (err) {
    throw new Error("feed " + param.name + ": " + err.message);
  }
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = i % d.w, y = (i / d.w) | 0;
    const value = fn(i, x, y, d.w, d.h);
    if (!Number.isFinite(value)) {
      throw new Error("feed " + param.name + " gave " + value + " at i=" + i);
    }
    out[i] = value;
  }
  return out;
}

async function run() {
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
    const feeds = inputs.map(p => feedValues(p, d.n, d));
    // An expression covers the whole grid; literal values only go as far as
    // the shortest list supplied.
    const count = anyExpression
      ? d.n
      : (feeds.length ? Math.min(...feeds.map(f => f.length)) : d.n);
    if (!count) throw new Error("no elements to run");
    log("info", "domain", { width: d.w, height: d.h, elements: count,
                            generated: anyExpression });

    // The caller owns memory, so the layout is decided here: every array
    // gets its own contiguous block, feeds first and then outputs.
    const need = (inputs.length + outputs.length) * count * bytes;
    const have = memory.buffer.byteLength;
    if (need > have) {
      memory.grow(Math.ceil((need - have) / 65536));
    }
    const View = isF32 ? Float32Array : Float64Array;
    const offsets = [];
    let cursor = 0;
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
    // Steady state, not one sample. A single call measures instantiation
    // effects, first-touch of the memory, and whatever the JIT had not done
    // yet; repeating and reporting the spread is what says how fast the
    // kernel actually is.
    const repeats = Math.max(1, Number($("repeats").value) | 0);
    const timings = [];
    for (let r = 0; r < repeats; r++) {
      const t0 = performance.now();
      fn(...args);
      timings.push(performance.now() - t0);
    }
    timings.sort((a, b) => a - b);
    const elapsed = timings[timings.length >> 1];
    const total = timings.reduce((a, b) => a + b, 0);
    if (repeats > 1) {
      $("stats").innerHTML =
        "<span>runs " + repeats + "</span>" +
        "<span>median " + elapsed.toFixed(3) + " ms</span>" +
        "<span>min " + timings[0].toFixed(3) + "</span>" +
        "<span>max " + timings[timings.length - 1].toFixed(3) + "</span>" +
        "<span>total " + total.toFixed(1) + " ms</span>" +
        "<span>" + (count * repeats / total / 1000).toFixed(1) + " Melem/s</span>";
      log("profile", "steady state over " + repeats + " runs", {
        median_ms: Number(elapsed.toFixed(4)),
        min_ms: Number(timings[0].toFixed(4)),
        max_ms: Number(timings[timings.length - 1].toFixed(4)),
        elements: count
      });
    } else {
      $("stats").innerHTML = "";
    }
    log("ok", "returned in " + elapsed.toFixed(3) + " ms");

    lastOutputs = outputs.map((p, i) => ({
      name: p.name,
      values: Array.from(new View(memory.buffer, offsets[inputs.length + i], count))
    }));
    log("progress", "reading outputs", { done: 4, total: 4 });
    renderActiveTab();
    setStatus("ran " + count + " elements in " + elapsed.toFixed(3) + " ms", "good");
  } catch (err) {
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

function renderRaw() {
  $("raw").textContent = (lastOutputs || [])
    .map(o => o.name + ": [" + o.values.join(", ") + "]")
    .join("\n");
}

function renderImage() {
  const canvas = $("canvas");
  const note = $("imgnote");
  if (!lastOutputs || !lastOutputs.length) { note.textContent = "Run first."; return; }
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
  const jpeg = canvas.toDataURL("image/jpeg", 0.92);
  const link = $("download");
  link.href = jpeg;
  link.style.display = "inline";
  note.textContent = w + "x" + h + ", range " + lo.toPrecision(4) + " to " +
    hi.toPrecision(4) + ", jpeg " + Math.round((jpeg.length * 3) / 4 / 1024) + " KB";
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

function renderGraph() {
  const target = document.getElementById("graph");
  if (!target || !GRAPH || !GRAPH.nodes) return;
  const hist = Object.entries(GRAPH.histogram || {})
    .map(([k, v]) => '<span class="chip">' + k + " x" + v + "</span>").join("");
  const rows = (GRAPH.table || []).map(n =>
    '<div class="kv"><b>' + n.id + "</b><span>" + n.type + "</span><span>" +
    (n.label || "") + "</span><span class=meta>" +
    (n.parents.length ? "&larr; " + n.parents.join(", ") : "") + "</span></div>"
  ).join("");
  target.innerHTML =
    '<div class="kv"><b>nodes</b><span>' + GRAPH.nodes + "</span></div>" +
    '<div class="kv"><b>edges</b><span>' + GRAPH.edges + "</span></div>" +
    "<div>" + hist + "</div>" +
    (GRAPH.truncated ? '<div class="meta">table truncated</div>' : "") +
    rows;
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
      $("row_expr_" + name).hidden = select.value !== "expression";
      $("row_gauss_" + name).hidden = select.value !== "gaussian";
    });
    select.dispatchEvent(new Event("change"));
  });
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
renderGraph();

log("info", "shell ready", {
  entry: entry.symbol,
  parameters: params.length,
  valueType: API.metadata.value_type,
  embedded: Boolean(WASM_BASE64)
});
if (moduleBytes) setStatus("module embedded, ready", "good");
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
) -> str:
    """One row per feed, each able to be literal values or an expression.

    A kernel's feeds are usually a function of position, so an expression is
    offered alongside literal values: it is evaluated per element with the
    grid coordinates in scope (``i``, ``x``, ``y``, ``w``, ``h``). Typing a
    quarter of a million numbers into a text field is not a control surface.
    """

    expressions = dict(feed_expressions or {})
    feeds = [p for p in parameters if p["role"] == "input"]
    rows = []
    for parameter in feeds:
        name = str(parameter["name"])
        expression = expressions.get(name, "i")
        default_mode = "expression" if name in expressions else "values"
        rows.append(
            '<div class="row">'
            f'<div class="name">{_escape(name)}</div>'
            f'<select id="mode_{_escape(name)}">'
            f'<option value="values"{"" if default_mode == "values" else ""}>values</option>'
            f'<option value="expression"{" selected" if default_mode == "expression" else ""}>expression</option>'
            '<option value="gaussian">gaussian</option>'
            "</select>"
            '<div class="grow">'
            f'<div id="row_values_{_escape(name)}">'
            f'<input type="text" id="in_{_escape(name)}" value="1, 2, 3, 4" '
            'placeholder="comma or space separated numbers"></div>'
            f'<div id="row_expr_{_escape(name)}" hidden>'
            f'<input type="text" id="expr_{_escape(name)}" '
            f'value="{_escape(expression)}" '
            'placeholder="expression over i, x, y, w, h"></div>'
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
    script = (
        _JS.replace("__API__", json.dumps(mapping))
        .replace("__WASM__", encoded)
        .replace("__GRAPH__", json.dumps(graph_mapping, default=str))
    )
    boot_script = _BOOT_JS.replace(
        "__TELEMETRY__", json.dumps(telemetry_mapping, default=str)
    )

    if wasm_bytes:
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

    note = entry.get("note")
    note_html = f'<div class="note">{_escape(str(note))}</div>' if note else ""

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

  <div class="panel">
    <div class="panel-title">Domain</div>
    <div class="meta">A kernel's feeds are computed over a grid. Width and
      height set how many elements one run covers, and the image view uses
      the same numbers.</div>
    <div class="imgctl">
      <label>width <input type="number" id="dom_w" min="1" value="{default_width}"></label>
      <label>height <input type="number" id="dom_h" min="1" value="{default_height}"></label>
    </div>
    {build_rows}
  </div>

  <div class="panel">
    <div class="panel-title">Inputs</div>
    {picker}
    {_input_rows(parameters, feed_expressions)}
    <div id="stats" class="stat"></div>
    <div class="row">
      <button id="run"{disabled}>Run {_escape(str(entry["name"]))}</button>
      <label class="meta">repeat <input type="number" id="repeats" min="1"
        value="1" style="width:5rem"></label>
      <div class="grow"><div id="status" class="out"></div></div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-title">Results</div>
    <div class="tabs" role="tablist">
      <div class="tab" data-view="raw" role="tab" aria-selected="true">Raw</div>
      <div class="tab" data-view="image" role="tab" aria-selected="false">JPEG</div>
    </div>
    <div class="tabview" data-view="raw">
      <div id="raw" class="out"></div>
    </div>
    <div class="tabview" data-view="image" hidden>
      <div class="imgctl">
        <label><input type="checkbox" id="img_invert"> invert</label>
        <a id="download" download="output.jpg" style="display:none">download .jpg</a>
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
      <pre>{_escape(origin_source)}</pre>
    </details>
  </div>

  <div class="panel">
    <details>
      <summary>Emitted source</summary>
      <pre>{_escape(source)}</pre>
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
