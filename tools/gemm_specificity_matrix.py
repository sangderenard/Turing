"""Build a browser-ready GEMM compilation-specificity benchmark matrix.

Each workload is measured against every subset of the ``m/n/k`` parameters:
fully parametric, one baked axis, two baked axes, and fully baked.  Every cell
is a real admitted KernelBank variant and records correctness, native compute,
end-to-end launch/copy time, GF/s, NumPy time, and the loop identities applied.

Run:

    python tools/gemm_specificity_matrix.py
    python tools/gemm_specificity_matrix.py --sets 32x32x32,64x128x32
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.compiler.kernel_bank import open_blas_bank


AXES = ("m", "n", "k")
SPECIFICITIES = (
    (), ("m",), ("n",), ("k",),
    ("m", "n"), ("m", "k"), ("n", "k"),
    ("m", "n", "k"),
)


def _parse_sets(raw: str) -> tuple[dict[str, int], ...]:
    result = []
    for item in str(raw).split(","):
        values = tuple(int(value) for value in item.lower().split("x"))
        if len(values) != 3 or any(value < 1 for value in values):
            raise ValueError(
                f"workload {item!r} must be positive MxNxK"
            )
        result.append(dict(zip(AXES, values)))
    return tuple(result)


def _median(
    operation, repetitions: int, *, batch: int = 1, warmup: int = 3,
) -> float:
    for _ in range(warmup):
        for _item in range(batch):
            operation()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        for _item in range(batch):
            operation()
        samples.append((time.perf_counter() - started) / batch)
    return float(statistics.median(samples))


def _repetitions(sizes: dict[str, int]) -> int:
    largest = max(sizes.values())
    return 50 if largest <= 32 else 30 if largest <= 64 else 15 if largest <= 128 else 9


def _batch_count(sizes: dict[str, int]) -> int:
    flops = 2 * sizes["m"] * sizes["n"] * sizes["k"]
    return max(1, min(200, 50_000_000 // flops))


def _variant_label(axes: tuple[str, ...]) -> str:
    return "parametric" if not axes else "+".join(axes) + " baked"


def _measure_variant(bank, sizes, axes, arrays, numpy_seconds, contract):
    specialized = {axis: sizes[axis] for axis in axes}
    try:
        variant = bank.get(
            "gemm", contract=contract, specialized=specialized,
        )
    except Exception as error:  # admission/refusal belongs in the matrix
        return {
            "status": "refused",
            "variant": _variant_label(axes),
            "specificity": list(axes),
            "specialized": specialized,
            "error": f"{type(error).__name__}: {error}",
        }

    arguments = {
        "A": arrays["A"], "B": arrays["B"], "C": arrays["C"],
        "alpha": arrays["alpha"], "beta": arrays["beta"], **sizes,
    }
    execution = variant._execute(arguments)
    c_id = variant.id_by_name["C"]
    repetitions = _repetitions(sizes)
    batch = _batch_count(sizes)

    def native_compute():
        np.asarray(execution.buffers[c_id]).reshape(-1)[...] = arrays["C"]
        execution.run()

    native_seconds = _median(native_compute, repetitions, batch=batch)
    native_compute()
    produced = np.asarray(execution.buffers[c_id]).reshape(-1).copy()
    worst_error = float(np.max(np.abs(produced - arrays["expected"])))
    end_to_end_seconds = _median(
        lambda: variant.run(arguments), max(3, repetitions // 2),
        batch=batch, warmup=1,
    )
    flops = 2.0 * sizes["m"] * sizes["n"] * sizes["k"]
    decisions = variant.module.metadata.get("loop_interchange", {}).get(
        "decisions", ()
    )
    return {
        "status": "admitted" if worst_error < 1.0e-9 else "wrong",
        "variant": _variant_label(axes),
        "specificity": list(axes),
        "specialized": specialized,
        "key": variant.key,
        "native_seconds": native_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "numpy_seconds": numpy_seconds,
        "native_gflops": flops / native_seconds / 1.0e9,
        "numpy_gflops": flops / numpy_seconds / 1.0e9,
        "speedup_vs_numpy": numpy_seconds / native_seconds,
        "end_to_end_vs_numpy": numpy_seconds / end_to_end_seconds,
        "worst_abs_error": worst_error,
        "repetitions": repetitions,
        "batch": batch,
        "identities": [
            {
                "identity": decision.get("identity"),
                "applied": bool(decision.get("interchanged")),
                "reasons": list(decision.get("reasons") or ()),
            }
            for decision in decisions
        ],
    }


def build_matrix(root: Path, sets, *, contract: str) -> dict:
    bank = open_blas_bank(root / "bank")
    rows = []
    for row_index, sizes in enumerate(sets):
        rng = np.random.default_rng(4100 + row_index)
        a = rng.standard_normal((sizes["m"], sizes["k"]))
        b = rng.standard_normal((sizes["k"], sizes["n"]))
        c = rng.standard_normal((sizes["m"], sizes["n"]))
        alpha, beta = 1.25, 0.5
        expected = alpha * (a @ b) + beta * c
        repetitions = _repetitions(sizes)
        batch = _batch_count(sizes)
        numpy_seconds = _median(
            lambda: alpha * (a @ b) + beta * c, repetitions, batch=batch,
        )
        arrays = {
            "A": a.reshape(-1), "B": b.reshape(-1), "C": c.reshape(-1),
            "alpha": alpha, "beta": beta, "expected": expected.reshape(-1),
        }
        cells = []
        for axes in SPECIFICITIES:
            print(
                f"[{row_index + 1}/{len(sets)}] "
                f"{sizes['m']}x{sizes['n']}x{sizes['k']} "
                f"{_variant_label(axes)}",
                flush=True,
            )
            cells.append(_measure_variant(
                bank, sizes, axes, arrays, numpy_seconds, contract,
            ))
        admitted = [cell for cell in cells if cell["status"] == "admitted"]
        winner = min(admitted, key=lambda cell: cell["native_seconds"])
        rows.append({
            "sizes": sizes,
            "label": f"{sizes['m']}×{sizes['n']}×{sizes['k']}",
            "numpy_seconds": numpy_seconds,
            "winner": winner["variant"],
            "cells": cells,
        })
    return {
        "schema": "turing.gemm-specificity-matrix.v1",
        "generated_unix": time.time(),
        "contract": contract,
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
        },
        "specificities": [
            {"axes": list(axes), "label": _variant_label(axes)}
            for axes in SPECIFICITIES
        ],
        "rows": rows,
    }


def render_html(report: dict) -> str:
    payload = json.dumps(report, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Turing GEMM compilation specificity matrix</title>
<style>
:root{{--ink:#eaf1ff;--muted:#9aabc4;--panel:#111a2d;--line:#273755;--hot:#70e1a1;--cold:#ff8b82}}
*{{box-sizing:border-box}} body{{margin:0;background:#08101f;color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:30px}} h1{{font-size:clamp(25px,4vw,48px);margin:.1em 0}}
.lede{{color:var(--muted);max-width:900px;font-size:16px}} .cards{{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:13px 16px;min-width:180px}}
.card b{{display:block;font-size:20px}} .matrix{{overflow:auto;border:1px solid var(--line);border-radius:14px}}
table{{border-collapse:separate;border-spacing:0;width:100%;min-width:1250px;background:var(--panel)}}
th,td{{padding:11px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);vertical-align:top}}
th{{position:sticky;top:0;background:#15223a;text-align:left;z-index:2}} th:first-child{{left:0;z-index:3}}
td:first-child{{position:sticky;left:0;background:#101a2e;font-weight:700;z-index:1}}
.cell{{min-width:142px;border-left:4px solid var(--line)}} .cell.win{{border-left-color:var(--hot)}}
.ratio{{font-size:19px;font-weight:800}} .faster{{color:var(--hot)}} .slower{{color:var(--cold)}}
.badge{{display:inline-block;margin-top:5px;padding:2px 6px;border-radius:10px;background:#254b65;color:#bdeaff;font-size:10px;font-weight:800}}
.detail{{color:var(--muted);font-size:12px}} button{{all:unset;cursor:pointer;display:block;width:100%}}
dialog{{max-width:800px;background:#101a2e;color:var(--ink);border:1px solid var(--line);border-radius:14px;padding:22px}}
pre{{white-space:pre-wrap;color:#bed0eb}} .legend{{margin:15px 0;color:var(--muted)}}
</style></head><body><main>
<h1>Compiled GEMM specificity matrix</h1>
<p class="lede">Every cell is a separately admitted native compiler product. Rows vary M×N×K; columns progressively bake parameter subsets. Ratios compare steady native compute with warmed NumPy on this machine. Click a cell for artifact identity, correctness, launch cost, and applied loop identities.</p>
<div class="cards" id="cards"></div><div class="legend">Green ratio ≥ 1 means compiled native compute beat NumPy. The highlighted border marks the fastest admitted compile for that workload.</div>
<div class="matrix"><table id="matrix"></table></div>
<dialog id="details"><button onclick="details.close()" style="float:right">Close ×</button><h2 id="detailTitle"></h2><pre id="detailBody"></pre></dialog>
</main><script id="report" type="application/json">{payload}</script><script>
const data=JSON.parse(document.querySelector('#report').textContent), table=document.querySelector('#matrix'), details=document.querySelector('#details');
const admitted=data.rows.flatMap(r=>r.cells).filter(c=>c.status==='admitted');
const best=admitted.reduce((a,c)=>!a||c.speedup_vs_numpy>a.speedup_vs_numpy?c:a,null);
document.querySelector('#cards').innerHTML=`<div class="card"><span>Compiled variants</span><b>${{admitted.length}}</b></div><div class="card"><span>Workloads</span><b>${{data.rows.length}}</b></div><div class="card"><span>Best vs NumPy</span><b>${{best.speedup_vs_numpy.toFixed(2)}}×</b></div><div class="card"><span>Contract</span><b>${{data.contract}}</b></div>`;
table.innerHTML='<thead><tr><th>M×N×K</th>'+data.specificities.map(s=>`<th>${{s.label}}</th>`).join('')+'</tr></thead><tbody>'+data.rows.map(row=>'<tr><td>'+row.label+`<div class="detail">NumPy ${{(row.numpy_seconds*1e3).toFixed(3)}} ms</div></td>`+row.cells.map(cell=>{{if(cell.status!=='admitted')return `<td><div class="cell slower">${{cell.status}}<div class="detail">${{cell.error||''}}</div></div></td>`;const ratio=cell.speedup_vs_numpy, cls=ratio>=1?'faster':'slower', win=cell.variant===row.winner?' win':'', rb=cell.identities.some(i=>i.identity==='unit_stride_reduction_register_block'&&i.applied);return `<td><button class="cell${{win}}" data-key="${{cell.key}}"><div class="ratio ${{cls}}">${{ratio.toFixed(2)}}×</div><div>${{cell.native_gflops.toFixed(2)}} GF/s</div>${{rb?'<span class="badge">REGISTER BLOCK</span>':''}}<div class="detail">${{(cell.native_seconds*1e3).toFixed(3)}} ms native<br>${{(cell.end_to_end_seconds*1e3).toFixed(3)}} ms end-to-end<br>error ${{cell.worst_abs_error.toExponential(1)}}</div></button></td>`;}}).join('')+'</tr>').join('')+'</tbody>';
table.addEventListener('click',event=>{{const button=event.target.closest('[data-key]');if(!button)return;const cell=admitted.find(c=>c.key===button.dataset.key);document.querySelector('#detailTitle').textContent=cell.variant+' · '+cell.key;document.querySelector('#detailBody').textContent=JSON.stringify(cell,null,2);details.showModal();}});
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sets",
        default=(
            "32x32x32,64x64x64,128x128x128,256x256x256,"
            "64x128x32,128x64x256"
        ),
        help="comma-separated MxNxK workloads",
    )
    parser.add_argument("--contract", default="fast")
    parser.add_argument(
        "--out", type=Path, default=Path("build/gemm-specificity-matrix"),
    )
    args = parser.parse_args()
    sets = _parse_sets(args.sets)
    args.out.mkdir(parents=True, exist_ok=True)
    report = build_matrix(args.out, sets, contract=args.contract)
    (args.out / "results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    (args.out / "index.html").write_text(
        render_html(report), encoding="utf-8",
    )
    print(f"wrote {args.out / 'results.json'}")
    print(f"wrote {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
