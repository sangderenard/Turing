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
.good { color: var(--good); }
details summary { cursor: pointer; font-weight: 600; font-size: .8rem; opacity: .65;
  text-transform: uppercase; letter-spacing: .04em; }
"""

# The runtime. Written against the API descriptor rather than any particular
# program: it lays the arrays out in the module's memory, calls the entry
# point, and reads the outputs back.
_JS = """
const API = __API__;
const WASM_BASE64 = __WASM__;

const $ = (id) => document.getElementById(id);
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
  return text.split(/[\\s,]+/).filter(s => s.length).map(Number);
}

async function run() {
  if (!moduleBytes) { setStatus("No .wasm loaded yet.", "bad"); return; }
  try {
    const { instance } = await WebAssembly.instantiate(moduleBytes, {});
    const memory = instance.exports[API.metadata.memory_export || "memory"];
    const fn = instance.exports[entry.symbol];
    if (!fn) throw new Error("export '" + entry.symbol + "' not found");

    const feeds = inputs.map(p => parseNumbers($("in_" + p.name).value));
    const count = feeds.length ? Math.min(...feeds.map(f => f.length))
                               : Number($("in_count").value);
    if (!count) throw new Error("no elements to run");

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

    const args = [count, ...offsets];
    const started = performance.now();
    fn(...args);
    const elapsed = performance.now() - started;

    const lines = outputs.map((p, i) => {
      const view = new View(memory.buffer, offsets[inputs.length + i], count);
      return p.name + ": [" + Array.from(view).join(", ") + "]";
    });
    $("results").textContent = lines.join("\\n");
    setStatus("ran " + count + " elements in " + elapsed.toFixed(3) + " ms", "good");
  } catch (err) {
    $("results").textContent = "";
    setStatus(String(err), "bad");
  }
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
$("run").addEventListener("click", run);
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


def _input_rows(parameters: Sequence[Mapping[str, Any]]) -> str:
    feeds = [p for p in parameters if p["role"] == "input"]
    rows = []
    if not feeds:
        # A program with no array feeds still needs an element count.
        rows.append(
            '<div class="row">'
            '<div class="name">count</div>'
            '<div class="grow"><input type="number" id="in_count" value="8" min="1"></div>'
            "</div>"
        )
    for parameter in feeds:
        name = str(parameter["name"])
        rows.append(
            '<div class="row">'
            f'<div class="name">{_escape(name)}</div>'
            f'<div class="grow"><input type="text" id="in_{_escape(name)}" '
            'value="1, 2, 3, 4" '
            'placeholder="comma or space separated numbers"></div>'
            "</div>"
        )
    return "\n".join(rows)


def emit_html_shell(
    api: Any,
    *,
    source: str = "",
    wasm_bytes: bytes | None = None,
    name: str | None = None,
) -> HtmlShell:
    """Generate a launchable page for one compiled program.

    ``api`` is a ``CompiledProgramAPI`` (or its mapping). ``source`` is the
    WAT, shown for reading. ``wasm_bytes`` is the assembled binary when one
    exists; without it the page offers a file picker instead of pretending
    it can assemble text itself.
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
    script = (
        _JS.replace("__API__", json.dumps(mapping)).replace("__WASM__", encoded)
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

  {banner}
  {note_html}

  <div class="panel">
    <div class="panel-title">Signature</div>
    {_signature_rows(parameters)}
  </div>

  <div class="panel">
    <div class="panel-title">Inputs</div>
    {picker}
    {_input_rows(parameters)}
    <div class="row">
      <button id="run"{disabled}>Run {_escape(str(entry["name"]))}</button>
      <div class="grow"><div id="status" class="out"></div></div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-title">Results</div>
    <div id="results" class="out"></div>
  </div>

  <div class="panel">
    <details>
      <summary>Emitted source</summary>
      <pre>{_escape(source)}</pre>
    </details>
  </div>

<script>{script}</script>
</body>
</html>
"""
    return HtmlShell(name=shell_name, html=html, embedded=wasm_bytes is not None)


def shell_for_artifact(artifact: Any, *, wasm_bytes: bytes | None = None) -> HtmlShell:
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
    )


__all__ = ["HtmlShell", "emit_html_shell", "shell_for_artifact"]
