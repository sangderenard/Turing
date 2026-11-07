"""Parity of the fountain law across its lanes, and the emitted WebGPU parity page.

One law (``fountain.make_law(n)``), several lanes:

* ``numpy``  the AbstractTensor law run eagerly on the NumPy backend (float64) -- the reference;
* ``native`` the law compiled by ``LawPiece._compile_native`` (the repository's LLVM lane);
* ``wasm``   the page's ``fountain_<tag>.wasm`` run in node through ``TuringGraphProgram``;
* ``webgpu`` the compiler-generated dispatch program (f32), run by the emitted page
             ``fountain_parity.html`` -- WebGPU is a browser feature, so that comparison
             is made in a browser and not here.

A granular pile is chaotic: f32 against f64 agrees tightly over a short horizon and then
diverges, so the comparison is a SHORT horizon tight (max abs difference per column) and the
SETTLED pile statistical (pile height profile, packing fraction).

    python examples/meowdoku/fountain_parity.py headless [--tag small] [--steps 120]
    python examples/meowdoku/fountain_parity.py page --out DIR [--tag full]
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _path in (str(ROOT), str(ROOT / "examples")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from meowdoku.s2_titles.fountain import fountain as F  # noqa: E402
from meowdoku.s4_generate import fountain_build as B  # noqa: E402

SHELL = ROOT / "examples" / "mobile_reference_app" / "web" / "turing-wasm-shell.js"
WORK = ROOT / "build" / "meowdoku" / "fountain_parity"
FRAMES = (50, 150, 700)       # frames the page compares

NODE = r"""
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const [dir, shell, job] = process.argv.slice(2);
const {TuringGraphProgram} = await import(pathToFileURL(shell).href);
const spec = JSON.parse(fs.readFileSync(`${dir}/${job}.job.json`, 'utf8'));
const abi = JSON.parse(fs.readFileSync(`${spec.web}/${spec.stem}.abi.json`, 'utf8'));
const program = await TuringGraphProgram.fromBytes(fs.readFileSync(`${spec.web}/${spec.stem}.wasm`), abi);
const raw = fs.readFileSync(`${dir}/${job}.in.bin`);
const all = new Float64Array(raw.buffer, raw.byteOffset, raw.length / 8);
const cols = {}; let at = 0;
for (const [name, size] of spec.inputs) { cols[name] = all.slice(at, at + size); at += size; }
const state = spec.state;
const t0 = performance.now();
for (let step = 0; step < spec.steps; step++) {
  for (const name in cols) program.setInput(name, cols[name]);
  program.run();
  for (const name of state) cols[name] = Float64Array.from(program.output(name + '_next'));
}
const wall = performance.now() - t0;
const out = {};
for (const name of state) out[name] = Array.from(cols[name]);
for (const name of ['cr', 'cg', 'cb']) out[name] = Array.from(program.output(name));
fs.writeFileSync(`${dir}/${job}.out.json`, JSON.stringify({out, ms_per_step: wall / spec.steps}));
"""


def wasm_frames(columns: dict, tag: str, web_dir: Path, steps: int, job: str = "wasm") -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "node_fountain.mjs").write_text(NODE, encoding="utf-8")
    inputs = [(n, np.asarray(columns[n], dtype=np.float64)) for n in F.ARGUMENTS]
    np.concatenate([a.reshape(-1) for _n, a in inputs]).tofile(WORK / f"{job}.in.bin")
    (WORK / f"{job}.job.json").write_text(json.dumps({
        "stem": f"fountain_{tag}", "web": str(web_dir), "steps": steps, "state": list(F.STATE),
        "inputs": [[n, int(a.size)] for n, a in inputs]}))
    done = subprocess.run(["node", str(WORK / "node_fountain.mjs"), str(WORK), str(SHELL), job],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-2000:])
    result = json.loads((WORK / f"{job}.out.json").read_text())
    return {**{k: np.asarray(v) for k, v in result["out"].items()}, "ms_per_step": result["ms_per_step"]}


def compare(a: dict, b: dict) -> dict:
    names = F.STATE + ("cr", "cg", "cb")
    return {"max_abs": {n: float(np.max(np.abs(np.asarray(a[n]) - np.asarray(b[n])))) for n in names}}


def pile_statistics(final: dict, columns: dict, n: int) -> dict:
    """The settled pile as numbers that survive chaos: height profile (mean top of each of 8
    column bands), 99th-percentile height, and packing fraction."""
    geo = F.basin(n)
    x, y, r = np.asarray(final["x"]), np.asarray(final["y"]), np.asarray(final["rad"])
    edges = np.linspace(-geo["half"], geo["half"], 9)
    profile = [float(np.mean(np.sort(y[(x >= lo) & (x < hi)])[-max(1, int((x >= lo).sum() // 8)):]))
               if ((x >= lo) & (x < hi)).any() else 0.0 for lo, hi in zip(edges[:-1], edges[1:])]
    top = float(np.percentile(y, 99))
    return {"profile": profile, "top99": top, "packing": float((np.pi * r ** 2).sum() / (geo["width"] * top))}


def native_reference(n: int, columns: dict, steps: int, record=()) -> tuple[dict, dict]:
    piece = F.make_piece(n)
    piece.compile(columns)
    state = {k: np.array(columns[k], dtype=np.float64) for k in F.STATE}
    saved, colours = {}, {}
    for step in range(1, steps + 1):
        produced = dict(zip(F.OUTPUTS, piece(*[state[k] if k in state else columns[k] for k in F.ARGUMENTS])))
        for k in F.STATE:
            state[k] = produced[k + "_next"]
        colours = {k: produced[k] for k in ("cr", "cg", "cb")}
        if step in record:
            saved[step] = {**{k: state[k].copy() for k in F.STATE}, **{k: colours[k].copy() for k in colours}}
    return {**state, **colours}, saved


def numpy_reference(n: int, columns: dict, steps: int) -> dict:
    """The eager law on the NumPy backend (slow: for small n and short horizons)."""
    piece = F.make_piece(n)
    return F.run_frames(piece, columns, steps)      # no compile(): the AbstractTensor law itself


def headless(tag: str, steps: int, web_dir: Path) -> dict:
    active, n = B.SIZES[tag]
    B.build(web_dir, (tag,))
    columns = F.initial_columns(n)
    report = {"tag": tag, "n": n, "steps": steps}
    started = time.perf_counter()
    ref = numpy_reference(n, columns, min(steps, 40))
    report["numpy_s_per_40_steps"] = round(time.perf_counter() - started, 1)
    got40 = wasm_frames(columns, tag, web_dir, min(steps, 40), job="wasm40")
    report["wasm_vs_numpy_40_steps"] = compare(ref, got40)
    final, _ = native_reference(n, columns, steps)
    got = wasm_frames(columns, tag, web_dir, steps)
    report["wasm_vs_native"] = compare(final, got)
    report["wasm_ms_per_step_node"] = got["ms_per_step"]
    return report


def page(tag: str, web_dir: Path, out: Path) -> Path:
    """Emit ``fountain_parity.html`` for ``tag``: the generated dispatch program, the round-zero
    columns, and the native (f64) reference at ``FRAMES``."""
    B.build(web_dir, (tag,))
    active, n = B.SIZES[tag]
    columns = F.initial_columns(n)
    _final, saved = native_reference(n, columns, max(FRAMES), record=FRAMES)
    record = json.loads((web_dir / "fountain.manifest.json").read_text(encoding="utf-8"))["sizes"][tag]
    wgsl = (web_dir / record["wgsl"]).read_text(encoding="utf-8")

    def pack(arrays):
        return base64.b64encode(np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays]).tobytes()).decode()

    names = F.STATE + ("cr", "cg", "cb")
    payload = {
        "n": n, "abi": record["wgsl_abi"], "wgsl": wgsl, "verdict": record["verdict"], "frames": list(FRAMES),
        "state": list(F.STATE), "names": list(names),
        "reference": {str(f): pack([saved[f][k] for k in names]) for f in FRAMES},
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fountain_parity.html"
    path.write_text(PAGE.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8", newline="\n")
    return path


PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>Fountain law: WebGPU dispatch program vs native reference</title>
<body style="font:14px/1.4 monospace;padding:12px;background:#fff;color:#111">
<h3>Fountain: compiler-generated WGSL dispatch program (f32) vs the native law (f64)</h3>
<p>A granular pile is chaotic, so f32 against f64 is tight over a short horizon and diverges after.
Every step is the program's dispatches plus a state commit; no readback until a compared frame.</p>
<pre id="out">running...</pre>
<script>
const P = __PAYLOAD__;
const out = [];
const log = (s) => { out.push(s); document.getElementById('out').textContent = out.join('\n'); };
const f32 = (b64) => { const bin = atob(b64); const u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Float32Array(u8.buffer); };
async function main() {
  if (!navigator.gpu) { log('NO WEBGPU'); window.__result = {error: 'no webgpu'}; return; }
  const adapter = await navigator.gpu.requestAdapter(); const device = await adapter.requestDevice();
  const info = adapter.info || {}; log('adapter: ' + [info.vendor, info.architecture, info.device].filter(Boolean).join(' / '));
  const n = P.n, A = P.abi, S = A.feed_slots, O = A.output_slots, at = (slot) => A.lane_base + slot * n;
  log(`n=${n}  ${A.dispatches} dispatches  verdict pays=${P.verdict.pays} predicted ${P.verdict.speedup.toFixed(2)}x`);
  const module = device.createShaderModule({code: P.wgsl});
  const ci = await module.getCompilationInfo();
  const errs = ci.messages.filter(m => m.type === 'error').map(m => m.lineNum + ': ' + m.message);
  if (errs.length) { log('WGSL ERRORS: ' + errs.join(' | ')); window.__result = {error: errs}; return; }
  const layout = device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: 'storage'}}]});
  const pl = device.createPipelineLayout({bindGroupLayouts: [layout]});
  const pipes = await Promise.all(A.entries.map(entryPoint => device.createComputePipelineAsync({layout: pl, compute: {module, entryPoint}})));
  const commitSrc = `@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@compute @workgroup_size(64) fn commit(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) { let i = g.x + g.y * grid.x * 64u; if (i >= ${n}u) { return; }
${P.state.map(s => `  arena[${at(S[s])}u + i] = arena[${at(O[s + '_next'])}u + i];`).join('\n')}
}`;
  pipes.push(await device.createComputePipelineAsync({layout: pl, compute: {module: device.createShaderModule({code: commitSrc}), entryPoint: 'commit'}}));
  const U = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const bytes = A.arena_elements * 4;
  const arena = device.createBuffer({size: bytes, usage: U});
  const read = device.createBuffer({size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  const group = device.createBindGroup({layout, entries: [{binding: 0, resource: {buffer: arena}}]});
  const host = new Float32Array(A.arena_elements);      // zero state and colours; uniforms in the head; no per-id data
  host[A.uniform_index.mode] = 0; host[A.uniform_index.use_image] = 0; host[A.uniform_index.mix] = 0.5;
  const run = (steps) => { const enc = device.createCommandEncoder();
    for (let k = 0; k < steps; k++) { const pass = enc.beginComputePass(); pass.setBindGroup(0, group); for (const p of pipes) { pass.setPipeline(p); pass.dispatchWorkgroups(A.groups[0], A.groups[1]); } pass.end(); }
    device.queue.submit([enc.finish()]); };
  const results = []; let frame = 0;
  device.queue.writeBuffer(arena, 0, host);
  for (const target of P.frames) {
    run(target - frame); frame = target;
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(arena, 0, read, 0, bytes); device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ); const all = new Float32Array(read.getMappedRange().slice(0)); read.unmap();
    const ref = f32(P.reference[target]); const row = {frame: target, worst: {}, over1e3: {}};
    P.names.forEach((name, k) => {
      const base = at((name in O) ? O[name] : S[name]); let worst = 0, bad = 0;
      for (let i = 0; i < n; i++) { const d = Math.abs(all[base + i] - ref[k * n + i]); worst = Math.max(worst, d); if (d > 1e-3) bad++; }
      row.worst[name] = worst; row.over1e3[name] = bad; });
    results.push(row);
    log(`frame ${target}: max abs diff ` + ['x', 'y', 'vx', 'vy'].map(k => `${k} ${row.worst[k].toExponential(2)} (${row.over1e3[k]} over 1e-3)`).join('  '));
  }
  // determinism: the same program twice from the same initial bytes
  const again = async () => { device.queue.writeBuffer(arena, 0, host); run(200); const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(arena, 0, read, 0, bytes); device.queue.submit([enc.finish()]); await read.mapAsync(GPUMapMode.READ); const a = new Float32Array(read.getMappedRange().slice(0)); read.unmap(); return a; };
  const r1 = await again(), r2 = await again(); let diff = 0; for (let i = 0; i < r1.length; i++) if (r1[i] !== r2[i]) diff++;
  log(`determinism: two runs of 200 steps differ in ${diff} of ${r1.length} arena values`);
  device.queue.writeBuffer(arena, 0, host); await device.queue.onSubmittedWorkDone();
  const reps = 120, t0 = performance.now(); run(reps); await device.queue.onSubmittedWorkDone();
  const per = (performance.now() - t0) / reps;
  log(`resident loop: ${per.toFixed(3)} ms per step (${A.dispatches + 1} dispatches), ${n} particles, no readback`);
  window.__result = {adapter: [info.vendor, info.architecture, info.device].filter(Boolean).join('/'), n, results, determinism_diff: diff, ms_per_step: per};
  log('DONE');
}
main().catch(e => { log('FAILED: ' + e); window.__result = {error: String(e)}; });
</script>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("headless", "page"))
    parser.add_argument("--tag", default="small")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--web-dir", default=str(WORK / "web"))
    parser.add_argument("--out", default=str(ROOT / "build" / "meowdoku"))
    args = parser.parse_args(argv)
    web_dir = Path(args.web_dir)
    if args.mode == "headless":
        print(json.dumps(headless(args.tag, args.steps, web_dir), indent=1))
    else:
        print("wrote", page(args.tag, web_dir, Path(args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
