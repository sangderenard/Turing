"""Headless parity of the fountain law across its lanes, and the emitted WebGPU parity page.

One law (``fountain.fountain_step``), several lanes:

* ``numpy``  the AbstractTensor law run eagerly on the NumPy backend (float64) -- the reference;
* ``native`` the law compiled by ``LawPiece._compile_native`` (the repository's LLVM lane);
* ``wasm``   the page's ``fountain_<tag>.wasm`` run in node through ``TuringGraphProgram``;
* ``webgpu`` the compiler-generated compute shader (f32), run by the emitted page
             ``fountain_parity.html`` -- WebGPU is a browser feature, so that comparison
             is made in a browser and not here.

Every lane is stepped from the same round-zero columns with the same fixed ``dt``; the
state of step ``k`` feeds step ``k + 1``.  The page file embeds the round-zero columns and
the reference's state at a few frames so it needs nothing else.

    python examples/meowdoku/fountain_parity.py headless [--steps 360] [--tag full]
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
ROOT = HERE.parents[1]
for _path in (str(ROOT), str(ROOT / "examples")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from meowdoku import fountain as F  # noqa: E402
from meowdoku import fountain_build as B  # noqa: E402

SHELL = ROOT / "examples" / "mobile_reference_app" / "web" / "turing-wasm-shell.js"
WORK = ROOT / "build" / "meowdoku" / "fountain_parity"
FRAMES = (90, 300, 540)       # frames the page compares (0.75 s, 2.5 s, 4.5 s of play)

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
    worst = {n: float(np.max(np.abs(np.asarray(a[n]) - np.asarray(b[n])))) for n in names}
    flips = int(np.sum((np.asarray(a["landed"]) > 0.5) != (np.asarray(b["landed"]) > 0.5)))
    return {"max_abs": worst, "landed_flips": flips}


def numpy_reference(columns: dict, steps: int, record=()) -> tuple[dict, dict]:
    """The eager law (NumPy backend, float64); returns the final state and the state at ``record`` frames."""
    piece = F.FountainPiece(len(columns["x"]))
    state = {n: np.array(columns[n], dtype=np.float64) for n in F.STATE}
    saved = {}
    colours = {}
    for step in range(1, steps + 1):
        produced = dict(zip(F.OUTPUTS, piece(*[state[n] if n in state else columns[n] for n in F.ARGUMENTS])))
        for n in F.STATE:
            state[n] = produced[n + "_next"]
        colours = {n: produced[n] for n in ("cr", "cg", "cb")}
        if step in record:
            saved[step] = {**{n: state[n].copy() for n in F.STATE}, **{n: colours[n].copy() for n in colours}}
    return {**state, **colours}, saved


def headless(tag: str, steps: int, web_dir: Path) -> dict:
    cols, rows = B.SIZES[tag]
    B.build(web_dir, (tag,))
    columns = F.initial_columns(cols, rows)
    n = cols * rows
    started = time.perf_counter()
    ref, _ = numpy_reference(columns, steps)
    report = {"tag": tag, "n": n, "steps": steps, "numpy_ms_per_step": (time.perf_counter() - started) * 1e3 / steps,
              "landed": int(np.sum(ref["landed"] > 0.5))}
    got = wasm_frames(columns, tag, web_dir, steps)
    report["wasm"] = compare(ref, got)
    report["wasm_ms_per_step_node"] = got["ms_per_step"]
    from meowdoku.entities import LawPiece  # noqa: F401
    piece = F.FountainPiece(n)
    started = time.perf_counter()
    piece.compile(columns)
    report["native_lane"] = piece.lane
    report["native_compile_s"] = round(time.perf_counter() - started, 1)
    if piece.lane != "eager":
        started = time.perf_counter()
        native = F.run_frames(piece, columns, steps)
        report["native_ms_per_step"] = (time.perf_counter() - started) * 1e3 / steps
        report["native"] = compare(ref, native)
        report["native_vs_wasm"] = compare(native, got)
    return report


def page(tag: str, web_dir: Path, out: Path) -> Path:
    """Emit ``fountain_parity.html`` for ``tag``: the generated shader, the round-zero columns and
    the NumPy reference at ``FRAMES``."""
    B.build(web_dir, (tag,))
    cols, rows = B.SIZES[tag]
    columns = F.initial_columns(cols, rows)
    _final, saved = numpy_reference(columns, max(FRAMES), record=FRAMES)
    manifest = json.loads((web_dir / "fountain.manifest.json").read_text(encoding="utf-8"))
    record = manifest["sizes"][tag]
    wgsl = (web_dir / record["wgsl"]).read_text(encoding="utf-8")

    def pack(arrays):
        return base64.b64encode(np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays]).tobytes()).decode()

    payload = {
        "n": record["n"], "abi": record["wgsl_abi"], "wgsl": wgsl, "verdict": record["verdict"],
        "frames": list(FRAMES), "arguments": list(F.ARGUMENTS), "state": list(F.STATE),
        "outputs": list(F.OUTPUTS),
        "initial": pack([np.broadcast_to(np.asarray(columns[n]), (record["n"],)) for n in F.ARGUMENTS]),
        "scalar_names": list(F.UNIFORMS),
        "reference": {str(f): pack([saved[f][n] for n in F.STATE + ("cr", "cg", "cb")]) for f in FRAMES},
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fountain_parity.html"
    path.write_text(PAGE.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8", newline="\n")
    return path


PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>Fountain law: WebGPU dispatch vs reference</title>
<body style="font:14px/1.4 monospace;padding:12px;background:#fff;color:#111">
<h3>Fountain law: compiler-generated WGSL (f32) vs the NumPy reference (f64)</h3>
<p>The shader is the compiler's output for <code>fountain_step</code>; the reference is the same law run eagerly on the
NumPy backend. Every frame is a fixed <code>dt</code> step; frame k feeds k+1 (state copied GPU-side).</p>
<pre id="out">running...</pre>
<script>
const P = __PAYLOAD__;
const out = [];
const log = (s) => { out.push(s); document.getElementById('out').textContent = out.join('\n'); };
const f32 = (b64) => { const bin = atob(b64); const u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Float32Array(u8.buffer); };
async function main() {
  if (!navigator.gpu) { log('NO WEBGPU: navigator.gpu is undefined'); window.__result = {error: 'no webgpu'}; return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { log('NO ADAPTER'); window.__result = {error: 'no adapter'}; return; }
  const device = await adapter.requestDevice();
  const info = adapter.info || {};
  log('adapter: ' + [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' / '));
  const n = P.n, A = P.arguments, O = P.outputs;
  log(`n=${n}  compiler verdict: pays=${P.verdict.pays} predicted ${P.verdict.speedup.toFixed(2)}x`);
  const module = device.createShaderModule({code: P.wgsl});
  const ci = await module.getCompilationInfo();
  const errs = ci.messages.filter(m => m.type === 'error').map(m => m.lineNum + ': ' + m.message);
  if (errs.length) { log('WGSL ERRORS: ' + errs.join(' | ')); window.__result = {error: errs}; return; }
  const pipeline = device.createComputePipeline({layout: 'auto', compute: {module, entryPoint: 'main'}});
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const feeds = device.createBuffer({size: A.length * n * 4, usage: S});
  const outs = device.createBuffer({size: O.length * n * 4, usage: S});
  const read = device.createBuffer({size: O.length * n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [{binding: 0, resource: {buffer: feeds}}, {binding: 1, resource: {buffer: outs}}]});
  const initial = f32(P.initial);   // A.length slots; a scalar's value is in its slot's first element
  device.queue.writeBuffer(feeds, 0, initial.slice(0, A.length * n));
  const slot = (name) => A.indexOf(name), oslot = (name) => O.indexOf(name);
  const results = [];
  let frame = 0; const t0 = performance.now();
  const last = Math.max(...P.frames);
  const refs = {}; for (const f of P.frames) refs[f] = f32(P.reference[f]);
  const names = P.state.concat(['cr', 'cg', 'cb']);
  while (frame < last) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(P.abi.groups[0], P.abi.groups[1], P.abi.groups[2]); pass.end();
    for (const s of P.state) enc.copyBufferToBuffer(outs, oslot(s + '_next') * n * 4, feeds, slot(s) * n * 4, n * 4);
    frame++;
    const check = P.frames.includes(frame);
    if (check) enc.copyBufferToBuffer(outs, 0, read, 0, O.length * n * 4);
    device.queue.submit([enc.finish()]);
    if (check) {
      await read.mapAsync(GPUMapMode.READ);
      const got = new Float32Array(read.getMappedRange().slice(0)); read.unmap();
      const ref = refs[frame]; const row = {frame, worst: {}};
      let flips = 0;
      names.forEach((name, k) => {
        const g0 = (name === 'cr' || name === 'cg' || name === 'cb') ? oslot(name) : oslot(name + '_next');
        let worst = 0;
        for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(got[g0 * n + i] - ref[k * n + i]));
        row.worst[name] = worst;
      });
      const lk = names.indexOf('landed'), lg = oslot('landed_next');
      for (let i = 0; i < n; i++) if ((got[lg * n + i] > 0.5) !== (ref[lk * n + i] > 0.5)) flips++;
      row.landed_flips = flips; results.push(row);
      log(`frame ${frame}: landed flips ${flips}/${n}; max abs diff ` + names.map(k => `${k} ${row.worst[k].toExponential(2)}`).join('  '));
    }
  }
  await device.queue.onSubmittedWorkDone();
  // timing: steps/sec of the resident loop with no readback
  const reps = 240, t1 = performance.now();
  for (let r = 0; r < reps; r++) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(P.abi.groups[0], P.abi.groups[1], P.abi.groups[2]); pass.end();
    for (const s of P.state) enc.copyBufferToBuffer(outs, oslot(s + '_next') * n * 4, feeds, slot(s) * n * 4, n * 4);
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const per = (performance.now() - t1) / reps;
  log(`resident loop: ${per.toFixed(3)} ms per step (dispatch + state copy), ${n} particles, no readback`);
  window.__result = {adapter: [info.vendor, info.architecture, info.device].filter(Boolean).join('/'), n, verdict: P.verdict, results, ms_per_step: per, precision: 'f32 shader vs f64 reference'};
  log('DONE');
}
main().catch(e => { log('FAILED: ' + e); window.__result = {error: String(e)}; });
</script>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("headless", "page"))
    parser.add_argument("--tag", default="full")
    parser.add_argument("--steps", type=int, default=360)
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
