"""Choose WebGPU compute for a pairwise rows x cols block, emit it, publish a parity page.

The block is the shape of ``examples/meowdoku/scroller.py``'s
``near_count_of_block`` / ``nearest_of_block``: a great-circle gap for every
(row lane, column lane) pair, reduced along the columns (``sum`` of a
near-mask, ``min`` of the live gap).

1. The block is lowered through the public entry,
   ``lower_ast_source_to_ssa``, once per size.  The deployment layer reads
   the planned regions of that module (``webgpu_compute_payoff``) and
   decides, from extent, intensity and boundary bytes, whether one compute
   dispatch pays.
2. The per-pair arithmetic is emitted by the WebGPU SSA backend
   (``emit_pairwise_reduce_module``) as a WGSL helper.
3. A self-contained HTML page runs the shader beside a float64 host
   reference and reports max abs difference and timing, per size.

Usage: python tools/build_pairwise_webgpu_demo.py [--out build/pairwise_webgpu.html]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.tensors.abstraction import AbstractTensor  # noqa: E402
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (  # noqa: E402
    c_backend_repository_ssa_reference,
)
from src.compiler import ssa_webgpu_backend as webgpu  # noqa: E402
from src.compiler.deployment_classification import (  # noqa: E402
    WebGPUComputeCostModel,
    webgpu_compute_payoff,
)
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: The block as authored.  Row and column lane sets are separate spans, as in
#: ``near_count_of_block`` (``u[i][r0:r1]`` against ``u[i][c0:c1]``).
BLOCK_SOURCE = '''
def near_count_block(rux, ruy, ruz, cux, cuy, cuz, live, reach, radius):
    a = int(rux.shape[0])
    b = int(cux.shape[0])
    rx = rux.reshape((a, 1))
    ry = ruy.reshape((a, 1))
    rz = ruz.reshape((a, 1))
    cx = cux.reshape((1, b))
    cy = cuy.reshape((1, b))
    cz = cuz.reshape((1, b))
    crx = ry * cz - rz * cy
    cry = rz * cx - rx * cz
    crz = rx * cy - ry * cx
    gap = radius * (crx * crx + cry * cry + crz * crz).sqrt().atan2(rx * cx + ry * cy + rz * cz)
    near = (gap < reach) * live.reshape((1, b))
    return near.sum(dim=1)
'''
ROW_FEEDS = ("rux", "ruy", "ruz")
COL_FEEDS = ("cux", "cuy", "cuz", "live")


def lower_block(rows: int, cols: int):
    spans = ROW_FEEDS + COL_FEEDS
    contract = ExtractionContract(
        ROOT / "extraction_contracts" / "program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [
            {"function": "near_count_block", "parameter": name,
             "storage": "span" if name in spans else "scalar",
             "dtype": "float32", "rank": 1 if name in spans else 0,
             **({"shape": [rows if name in ROW_FEEDS else cols]} if name in spans else {}),
             "python_type": (
                 "src.common.tensors.abstraction.AbstractTensor"
                 if name in spans else "builtins.float"),
             "mutable": False}
            for name in spans + ("reach", "radius")
        ],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        BLOCK_SOURCE, "near_count_block", name="pairwise",
        progress_bars=False, progress=lambda _: None,
        extraction_contract=contract, runtime_closure_only=True,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    return module, "pairwise__near_count_block"


def pair_gap_function():
    """The per-pair great-circle gap as repository SSA (f32 scalars).

    ``radius * atan2(|r x c|, r . c)``.  Built as SSA directly because the
    scalar public-entry lowering has no ``math.atan2`` (see the report); the
    deployment decision above is still made on the module the public entry
    produced for the authored block.
    """

    ids = iter(range(1, 100))
    value = lambda: SSAValue(next(ids), "float32")  # noqa: E731
    rx, ry, rz, cx, cy, cz, radius = (value() for _ in range(7))
    body: list[Instr] = []

    def put(op, *args):
        result = value()
        body.append(Instr(op, list(args), result))
        return result

    crx = put("Sub", put("Mul", ry, cz), put("Mul", rz, cy))
    cry = put("Sub", put("Mul", rz, cx), put("Mul", rx, cz))
    crz = put("Sub", put("Mul", rx, cy), put("Mul", ry, cx))
    norm = put("Sqrt", put("Add", put("Add", put("Mul", crx, crx), put("Mul", cry, cry)), put("Mul", crz, crz)))
    dot = put("Add", put("Add", put("Mul", rx, cx), put("Mul", ry, cy)), put("Mul", rz, cz))
    gap = put("Mul", radius, put("atan2", norm, dot))
    body.append(Instr("Ret", [gap], None))
    function = Function(
        "pair_gap", [rx, ry, rz, cx, cy, cz, radius],
        {"entry": BasicBlock("entry", body)},
    )
    roles = {
        rx.id: ("row", 0), ry.id: ("row", 1), rz.id: ("row", 2),
        cx.id: ("col", 0), cy.id: ("col", 1), cz.id: ("col", 2),
        radius.id: ("uniform", 0),
    }
    return function, gap, roles


def emit_shader(rows: int, cols: int):
    function, gap, roles = pair_gap_function()
    return webgpu.emit_pairwise_reduce_module(
        function, gap, rows=rows, cols=cols, roles=roles,
        row_count=3, col_count=4, uniform_count=2,
        reductions=(
            ("best", "min", "select(1.0e9f, {gap}, {col3} > 0.0f)"),
            ("count", "sum", "select(0.0f, 1.0f, ({gap} < {u1}) && ({col3} > 0.0f))"),
        ),
        name="pairwise_block",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "build" / "pairwise_webgpu.html"))
    parser.add_argument("--sizes", default="4x4,24x256,256x256,1024x1024")
    args = parser.parse_args(argv)
    model = WebGPUComputeCostModel()
    cases = []
    for tag in args.sizes.split(","):
        rows, cols = (int(part) for part in tag.split("x"))
        module, entry = lower_block(rows, cols)
        payoff = webgpu_compute_payoff(module, entry, model=model)
        shader = emit_shader(rows, cols)
        if not shader.complete:
            raise SystemExit(f"shader shortfalls: {shader.shortfalls}")
        print(f"{tag}: pays={payoff.pays} " + " | ".join(payoff.reasons), flush=True)
        cases.append({
            "rows": rows, "cols": cols, "payoff": payoff.as_record(),
            "wgsl": shader.source, "workgroups": shader.launch_plan.groups[0],
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        PAGE.replace("__CASES__", json.dumps(cases)).replace(
            "__MODEL__", json.dumps(model.__dict__)),
        encoding="utf-8", newline="\n",
    )
    print("wrote", out)
    return 0


PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>Pairwise block: WebGPU compute vs host</title>
<body style="font:14px/1.4 monospace;padding:12px">
<h3>Pairwise rows x cols block (great-circle min / near-count)</h3>
<p>Shader precision: <b>f32</b> (WGSL has no f64). Reference: float64 JS loops on the host.
Deployment verdict is the compiler's (webgpu_compute_payoff over the public-entry module).</p>
<pre id="out">running...</pre>
<script>
const CASES = __CASES__;
const MODEL = __MODEL__;
const out = [];
const log = (s) => { out.push(s); document.getElementById('out').textContent = out.join('\n'); };
let seed = 12345;
const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
function sphere(n) {
  const x = new Float64Array(n), y = new Float64Array(n), z = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const u = rnd() * 2 - 1, t = rnd() * 2 * Math.PI, r = Math.sqrt(1 - u * u);
    x[i] = r * Math.cos(t); y[i] = r * Math.sin(t); z[i] = u;
  }
  return [x, y, z];
}
function hostBlock(R, C, live, reach, radius) {
  const best = new Float64Array(R.x.length), count = new Float64Array(R.x.length);
  for (let i = 0; i < R.x.length; i++) {
    let b = 1e9, n = 0;
    for (let j = 0; j < C.x.length; j++) {
      const cx = R.y[i]*C.z[j]-R.z[i]*C.y[j], cy = R.z[i]*C.x[j]-R.x[i]*C.z[j], cz = R.x[i]*C.y[j]-R.y[i]*C.x[j];
      const gap = radius * Math.atan2(Math.sqrt(cx*cx+cy*cy+cz*cz), R.x[i]*C.x[j]+R.y[i]*C.y[j]+R.z[i]*C.z[j]);
      if (live[j] > 0) { if (gap < b) b = gap; if (gap < reach) n++; }
    }
    best[i] = b; count[i] = n;
  }
  return [best, count];
}
const median = (a) => { const s = [...a].sort((p, q) => p - q); return s[s.length >> 1]; };
async function main() {
  if (!navigator.gpu) { log('NO WEBGPU: navigator.gpu is undefined'); window.__result = {error: 'no webgpu'}; return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { log('NO ADAPTER'); window.__result = {error: 'no adapter'}; return; }
  const device = await adapter.requestDevice();
  const info = adapter.info || {};
  log('adapter: ' + [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' / '));
  const results = [];
  for (const c of CASES) {
    const {rows, cols} = c;
    const R = (() => { const [x, y, z] = sphere(rows); return {x, y, z}; })();
    const Cc = (() => { const [x, y, z] = sphere(cols); return {x, y, z}; })();
    const live = new Float64Array(cols).map(() => rnd() < 0.7 ? 1 : 0);
    const radius = 1.0, reach = 0.35;
    const rowsIn = new Float32Array(3 * rows), colsIn = new Float32Array(4 * cols);
    rowsIn.set(R.x, 0); rowsIn.set(R.y, rows); rowsIn.set(R.z, 2 * rows);
    colsIn.set(Cc.x, 0); colsIn.set(Cc.y, cols); colsIn.set(Cc.z, 2 * cols); colsIn.set(live, 3 * cols);
    const params = new Float32Array([radius, reach]);
    const mk = (data, usage) => { const b = device.createBuffer({size: Math.max(16, data.byteLength), usage}); if (usage & GPUBufferUsage.COPY_DST) device.queue.writeBuffer(b, 0, data); return b; };
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const bR = mk(rowsIn, S), bC = mk(colsIn, S), bP = mk(params, S);
    const outBytes = 2 * rows * 4;
    const bO = device.createBuffer({size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
    const bRead = device.createBuffer({size: outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
    const module = device.createShaderModule({code: c.wgsl});
    const info2 = await module.getCompilationInfo();
    const errs = info2.messages.filter(m => m.type === 'error').map(m => m.lineNum + ': ' + m.message);
    if (errs.length) { log(rows + 'x' + cols + ' WGSL ERRORS: ' + errs.join(' | ')); results.push({rows, cols, error: errs}); continue; }
    const pipeline = device.createComputePipeline({layout: 'auto', compute: {module, entryPoint: 'main'}});
    const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [bR, bC, bP, bO].map((b, i) => ({binding: i, resource: {buffer: b}}))});
    const run = async () => {
      const t0 = performance.now();
      device.queue.writeBuffer(bR, 0, rowsIn); device.queue.writeBuffer(bC, 0, colsIn);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(c.workgroups); pass.end();
      enc.copyBufferToBuffer(bO, 0, bRead, 0, outBytes);
      device.queue.submit([enc.finish()]);
      await bRead.mapAsync(GPUMapMode.READ);
      const got = new Float32Array(bRead.getMappedRange().slice(0));
      bRead.unmap();
      return [performance.now() - t0, got];
    };
    let got; for (let i = 0; i < 3; i++) got = (await run())[1];
    const gt = [], ht = [];
    const reps = rows * cols > 1e6 ? 5 : 21;
    for (let i = 0; i < reps; i++) { const r = await run(); gt.push(r[0]); got = r[1]; }
    let ref; for (let i = 0; i < reps; i++) { const t = performance.now(); ref = hostBlock(R, Cc, live, reach, radius); ht.push(performance.now() - t); }
    let maxBest = 0, maxRel = 0, countMismatch = 0, maxCountDiff = 0;
    for (let i = 0; i < rows; i++) {
      const d = Math.abs(got[i] - ref[0][i]); maxBest = Math.max(maxBest, d);
      maxRel = Math.max(maxRel, d / Math.max(1e-9, Math.abs(ref[0][i])));
      const cd = Math.abs(got[rows + i] - ref[1][i]); if (cd !== 0) countMismatch++; maxCountDiff = Math.max(maxCountDiff, cd);
    }
    const gms = median(gt), hms = median(ht);
    const chosen = c.payoff.pays ? 'webgpu-compute' : 'host';
    log(`${rows}x${cols}: compiler chose ${chosen} (predicted ${c.payoff.speedup.toFixed(2)}x)  measured: gpu ${gms.toFixed(3)} ms (upload+dispatch+readback), host ${hms.toFixed(3)} ms, measured speedup ${(hms / gms).toFixed(2)}x`);
    log(`    max abs diff best-gap = ${maxBest.toExponential(3)} (max rel ${maxRel.toExponential(3)}); count mismatches ${countMismatch}/${rows}, max count diff ${maxCountDiff}`);
    results.push({rows, cols, chosen, predicted: c.payoff.speedup, gpu_ms: gms, host_ms: hms, measured_speedup: hms / gms, max_abs_best: maxBest, max_rel_best: maxRel, count_mismatches: countMismatch, max_count_diff: maxCountDiff});
  }
  window.__result = {adapter: [info.vendor, info.architecture, info.device].filter(Boolean).join('/'), precision: 'f32 shader vs f64 host', results};
  log('DONE');
}
main().catch(e => { log('FAILED: ' + e); window.__result = {error: String(e)}; });
</script>
"""

if __name__ == "__main__":
    raise SystemExit(main())
