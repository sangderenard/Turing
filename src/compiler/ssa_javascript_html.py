"""Generic browser host for repository-SSA JavaScript programs.

The host knows only the emitted argument manifest and a declarative page
configuration.  It does not implement a numerical step, controller, or
program-specific equation; those remain in ``program.mjs``.
"""

from __future__ import annotations

import json
from html import escape
from typing import Any, Mapping


def emit_repository_ssa_javascript_page(
    configuration: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Return ``(html, host_module, configuration_json)`` for one program."""

    title = str(configuration.get("title") or "Compiled program")
    escaped_title = escape(title)
    config_json = json.dumps(configuration, indent=2, sort_keys=True)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{escaped_title}</title>
<style>
:root{{color-scheme:dark;background:#071018;color:#dceaf0;font:15px/1.35 system-ui,sans-serif}}
*{{box-sizing:border-box}} body{{margin:0;min-height:100vh;overflow-x:hidden;display:grid;grid-template-rows:auto 1fr;background:radial-gradient(circle at 50% 20%,#183040,#071018 62%)}}
header{{display:flex;gap:.7rem;align-items:center;flex-wrap:wrap;padding:.75rem 1rem;background:#08131ddd;border-bottom:1px solid #294250}}
h1{{font-size:1rem;margin:0 auto 0 0;font-weight:650}} button,select{{color:inherit;background:#132733;border:1px solid #3b5968;border-radius:.45rem;padding:.45rem .7rem}}
button:active{{transform:translateY(1px)}} main{{min-height:0;display:grid;place-items:center;padding:1rem}}
.stage{{width:min(92vw,82vh);aspect-ratio:1;position:relative;border:1px solid #365764;border-radius:.8rem;overflow:hidden;box-shadow:0 1.2rem 4rem #0008;background:#020609}}
canvas{{width:100%;height:100%;display:block;image-rendering:auto;touch-action:none}}
.readout{{position:absolute;left:.7rem;bottom:.65rem;padding:.4rem .55rem;border-radius:.4rem;background:#071018bb;font:12px/1.25 ui-monospace,monospace;pointer-events:none}}
.hint{{opacity:.72;font-size:.8rem}} @media(max-width:600px){{header{{padding:.6rem;display:grid;grid-template-columns:repeat(2,minmax(0,1fr))}}header h1{{grid-column:1/-1;margin:0}}header button,header select{{width:100%;min-width:0}}.hint{{display:none}}main{{padding:.4rem;overflow:hidden}}.stage{{width:min(94vw,70vh)}}}}
</style></head><body>
<header><h1>{escaped_title}</h1><button id="toggle">Pause</button><button id="step">Step</button><button id="reset">Reset</button><select id="field"></select><span class="hint">drag: raise · press/secondary: lower</span></header>
<main><div class="stage"><canvas id="view"></canvas><div class="readout" id="readout">loading compiler output…</div></div></main>
<script type="module" src="./runtime/host.js"></script></body></html>
"""
    return html, _HOST_MODULE, config_json


_HOST_MODULE = r'''import program from "../program/program.js";

const [manifest, config] = await Promise.all([
  fetch(new URL("../program/program.json", import.meta.url)).then((response) => response.json()),
  fetch(new URL("../program/page-config.json", import.meta.url)).then((response) => response.json()),
]);
const canvas = document.querySelector("#view");
const context = canvas.getContext("2d", {alpha: false});
const readout = document.querySelector("#readout");
const fieldSelect = document.querySelector("#field");
const toggle = document.querySelector("#toggle");
const canonicalOwner = (owner) => config.ownerAliases?.[owner] ?? owner;
const keyFor = (argument) => `${canonicalOwner(argument.programParameter)}.${argument.programField}`;
const grid = (spec) => Array.from({length: config.grid.height}, (_, row) =>
  Array.from({length: config.grid.width}, (_, column) => {
    let value = Number(spec.fill ?? 0);
    for (const bump of spec.bumps ?? []) {
      const dx = column - Number(bump.x) * (config.grid.width - 1);
      const dy = row - Number(bump.y) * (config.grid.height - 1);
      value += Number(bump.amplitude) * Math.exp(-(dx * dx + dy * dy) / (2 * Number(bump.radius) ** 2));
    }
    return value;
  })
);
let buffers, fields, dtPosition, running = true, busy = false, frames = 0, lastResult;
function buildBuffers() {
  const shared = new Map();
  fields = new Map();
  const result = manifest.arguments.map((argument) => {
    const key = keyFor(argument);
    const authored = config.fields[key];
    if (authored !== undefined) {
      if (shared.has(key)) return shared.get(key);
      let value = authored?.kind === "grid" ? grid(authored) : authored;
      if (argument.pointer && !Array.isArray(value) && !ArrayBuffer.isView(value)) value = [value];
      shared.set(key, value); fields.set(key, value); return value;
    }
    if (String(argument.valueId) in config.inputs) return config.inputs[String(argument.valueId)];
    if (!argument.pointer) return argument.dtype === "unknown" ? null : 0;
    const scratchKey = JSON.stringify([key, argument.accounting?.compiler_frame_sequence_id, argument.accounting?.split_from_unproven_alias]);
    if (!shared.has(scratchKey)) shared.set(scratchKey, Array.from({length: config.scratchCapacity}, () => 0));
    return shared.get(scratchKey);
  });
  dtPosition = manifest.arguments.find((argument) => argument.valueId === config.feedback.inputValueId)?.position;
  return result;
}
function selectedGrid() {
  const index = Number(fieldSelect.value);
  const value = lastResult?.[index] ?? fields.get(config.display[index]?.field);
  return value;
}
function render() {
  const values = selectedGrid();
  if (!Array.isArray(values) || !Array.isArray(values[0])) return;
  const height = values.length, width = values[0].length;
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const flat = values.flat().map(Number); const lo = Math.min(...flat), hi = Math.max(...flat); const span = Math.max(hi - lo, 1e-12);
  const image = context.createImageData(width, height);
  for (let index = 0; index < flat.length; index += 1) {
    const t = Math.max(0, Math.min(1, (flat[index] - lo) / span));
    image.data[index * 4] = 12 + 80 * t;
    image.data[index * 4 + 1] = 32 + 190 * Math.sqrt(t);
    image.data[index * 4 + 2] = 62 + 185 * (1 - Math.abs(2 * t - 1));
    image.data[index * 4 + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  const dt = lastResult?.[config.feedback.outputIndex];
  const diagnostic = lastResult?.[config.diagnosticOutputIndex];
  readout.textContent = `frame ${frames}  dt ${Number(dt ?? config.inputs[String(config.feedback.inputValueId)]).toExponential(3)}  diagnostic ${Number(diagnostic ?? 0).toFixed(5)}  range ${lo.toFixed(5)}…${hi.toFixed(5)}`;
}
async function advance() {
  if (busy) return; busy = true;
  try {
    lastResult = await program.run(buffers); frames += 1;
    const next = Number(lastResult[config.feedback.outputIndex]);
    if (!Number.isFinite(next) || next <= 0) throw new Error(`compiled program returned invalid feedback ${next}`);
    if (dtPosition !== undefined) buffers[dtPosition] = next;
    render();
  } catch (error) { running = false; toggle.textContent = "Run"; readout.textContent = error.stack ?? String(error); console.error(error); }
  finally { busy = false; }
}
function reset() { buffers = buildBuffers(); lastResult = undefined; frames = 0; render(); }
for (const [index, item] of config.display.entries()) fieldSelect.add(new Option(item.label, index));
fieldSelect.addEventListener("change", render); toggle.addEventListener("click", () => { running = !running; toggle.textContent = running ? "Pause" : "Run"; });
document.querySelector("#step").addEventListener("click", advance); document.querySelector("#reset").addEventListener("click", reset);
function perturb(event) {
  const target = fields.get(config.interaction.field); if (!Array.isArray(target?.[0])) return;
  const bounds = canvas.getBoundingClientRect(); const column = Math.floor((event.clientX - bounds.left) / bounds.width * target[0].length); const row = Math.floor((event.clientY - bounds.top) / bounds.height * target.length);
  const sign = (event.buttons === 2 || event.shiftKey || event.pressure > .55) ? -1 : 1;
  for (let y = 0; y < target.length; y += 1) for (let x = 0; x < target[y].length; x += 1) {
    const distance2 = (x - column) ** 2 + (y - row) ** 2;
    target[y][x] += sign * config.interaction.amplitude * Math.exp(-distance2 / (2 * config.interaction.radius ** 2));
  }
  render();
}
canvas.addEventListener("contextmenu", (event) => event.preventDefault()); canvas.addEventListener("pointerdown", (event) => { canvas.setPointerCapture(event.pointerId); perturb(event); }); canvas.addEventListener("pointermove", (event) => { if (event.buttons) perturb(event); });
reset(); await advance();
let previous = 0; async function loop(now) { if (running && now - previous >= 1000 / config.renderFps) { previous = now; await advance(); } requestAnimationFrame(loop); } requestAnimationFrame(loop);
'''


__all__ = ["emit_repository_ssa_javascript_page"]
