"""Build the fountain's page products from the one authored law.

``fountain.fountain_step`` is traced ONCE per particle count through the SSA
tensor backend (the same trace ``LawPiece._compile_native`` and
``graph_wasm._compile`` use), and from that one trace the compiler produces:

* the DEPLOYMENT VERDICT -- ``webgpu_compute_payoff`` over the traced module:
  does one compute dispatch beat the host at this particle count?
* the WEBGPU COMPUTE SHADER -- ``ssa_webgpu_backend.emit_tensor_program_module``
  rewrites the traced elementwise tensor kernels to one invocation per
  particle and emits the WGSL.  Nothing of the law is hand-written in WGSL.
* the WASM PRODUCT -- ``compile_graph_forward_to_wasm`` (``graph_wasm._compile``),
  the route when the verdict says the dispatch does not pay or the browser has
  no WebGPU.

Files written to the page's directory (rebuilt only when their source digest
changes):

    fountain.manifest.json        sizes, verdicts, ABIs, the declared cost model
    fountain_<tag>.wgsl           the compute shader for that size
    fountain_<tag>.abi.json       (wasm) buffer table, the graph_wasm ABI

    python examples/meowdoku/fountain_build.py [--web-dir DIR] [--sizes full,small]
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
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

SCHEMA = "turing.fountain-products.v1"
#: tag -> (cols, rows).  ``full`` is the game; ``small`` is a handful of
#: particles on which the compiler's verdict is NO and the page runs wasm.
SIZES = {"full": (F.COLS, F.ROWS), "small": (12, 8)}
#: extra counts the verdict is published for (no products are built for them)
SWEEP = (96, 512, 2048, 8192, F.N_FULL)
WGSL_CONTRACT = "fast"           # float64 storage -> f32 through a declared backend identity
WASM_CONTRACT = "develop"


def _sources() -> str:
    from src.compiler import deployment_classification, llvm_training_runtime, ssa_webgpu_backend
    return "\0".join(inspect.getsource(m) for m in (F, sys.modules[__name__], ssa_webgpu_backend,
                                                     deployment_classification, llvm_training_runtime))


def digest(tag: str, cols: int, rows: int) -> str:
    return hashlib.sha256(f"{tag}:{cols}x{rows}\0{_sources()}".encode()).hexdigest()[:16]


def shapes_for(n: int) -> dict:
    return {name: ((n,) if name in F.STATE + F.CONSTANTS else ()) for name in F.ARGUMENTS}


def trace(n: int, contract: str):
    """The law traced at ``n`` particles: (program, output values, bindings)."""
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    with scoped_work_contract(PRESETS[contract]):
        program = SSATensorProgram("fountain_step")
        shapes = shapes_for(n)
        bindings = {name: SSATensorOperations.input(program, shapes[name]) for name in F.ARGUMENTS}
        with program.activate():
            produced = F.fountain_step(*(bindings[name] for name in F.ARGUMENTS))
        outputs = [item.data.value for item in produced]
    return program, outputs, bindings


def verdict_for(n: int, model=None):
    """The compiler's WebGPU-compute verdict for the law at ``n`` particles."""
    from src.compiler.deployment_classification import WebGPUComputeCostModel, webgpu_compute_payoff
    program, outputs, _ = trace(n, WGSL_CONTRACT)
    return webgpu_compute_payoff(program.module, "fountain_step", outputs=outputs,
                                 model=model or WebGPUComputeCostModel())


def shader_for(n: int):
    """The compiler-generated compute shader at ``n`` particles."""
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    program, outputs, bindings = trace(n, WGSL_CONTRACT)
    with scoped_work_contract(PRESETS[WGSL_CONTRACT]):
        artifact = webgpu.emit_tensor_program_module(
            program.module, "fountain_step", outputs, count=n, name=f"fountain_{n}")
    ids = {name: int(getattr(binding.data, "value").id) for name, binding in bindings.items()}
    return artifact, ids


def build_size(tag: str, cols: int, rows: int, web_dir: Path) -> dict:
    from meowdoku import graph_wasm
    from src.compiler.deployment_classification import WebGPUComputeCostModel
    n = cols * rows
    started = time.perf_counter()
    payoff = verdict_for(n)
    artifact, ids = shader_for(n)
    if not artifact.complete:
        raise RuntimeError("fountain shader is incomplete: " + "; ".join(s.format() for s in artifact.shortfalls))
    meta = artifact.api.to_mapping()["metadata"]
    feed_order = [name for name in F.ARGUMENTS]
    if meta["feed_span"] != [ids[name] for name in feed_order]:
        raise RuntimeError("shader feed order differs from the law's argument order")
    (web_dir / f"fountain_{tag}.wgsl").write_text(artifact.source, encoding="utf-8", newline="\n")
    wasm_abi = graph_wasm._compile(
        f"fountain_{tag}", shapes_for(n),
        lambda b: F.fountain_step(*(b[name] for name in F.ARGUMENTS)),
        F.OUTPUTS, WASM_CONTRACT, web_dir,
        {"argument_names": list(F.ARGUMENTS), "count": n})
    record = {
        "tag": tag, "n": n, "cols": cols, "rows": rows,
        "route": "webgpu-compute" if payoff.pays else "wasm",
        "verdict": payoff.as_record(),
        "wgsl": f"fountain_{tag}.wgsl",
        "wgsl_abi": {
            "count": n, "workgroup_size": list(meta["workgroup_size"]),
            "groups": list(meta["dispatch_workgroups"]),
            "arguments": feed_order, "uniforms": [name for name in feed_order if name in F.UNIFORMS],
            "outputs": list(F.OUTPUTS), "feed_span": meta["feed_span"], "output_span": meta["output_span"],
            "deployment": meta["deployment"], "backend_identities": meta["backend_identities"],
        },
        "uniforms": {k: float(v) for k, v in F.uniforms(cols=cols, rows=rows).items()},
        "wasm": f"fountain_{tag}.wasm", "wasm_abi": f"fountain_{tag}.abi.json",
        "wasm_buffers": len(wasm_abi["order"]),
        "source_digest": digest(tag, cols, rows),
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    return record


def build(web_dir: Path, sizes: tuple[str, ...] = tuple(SIZES)) -> dict:
    """Write the page products; returns the manifest."""
    from src.compiler.deployment_classification import WebGPUComputeCostModel
    web_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = web_dir / "fountain.manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = {}
    for tag in sizes:
        cols, rows = SIZES[tag]
        previous = (old.get("sizes") or {}).get(tag)
        files = (web_dir / f"fountain_{tag}.wgsl", web_dir / f"fountain_{tag}.wasm", web_dir / f"fountain_{tag}.abi.json")
        if previous and previous.get("source_digest") == digest(tag, cols, rows) and all(p.exists() for p in files):
            records[tag] = previous
            continue
        records[tag] = build_size(tag, cols, rows, web_dir)
        print(f"[fountain] {tag}: {records[tag]['n']} particles -> {records[tag]['route']} "
              f"(predicted {records[tag]['verdict']['speedup']:.2f}x), built in {records[tag]['build_seconds']}s", flush=True)
    model = WebGPUComputeCostModel()
    manifest = {
        "schema": SCHEMA, "sizes": records, "cost_model": model.__dict__,
        "sweep": [{"n": n, **verdict_for(n).as_record()} for n in SWEEP],
        "law": {"arguments": list(F.ARGUMENTS), "state": list(F.STATE), "constants": list(F.CONSTANTS),
                "uniforms": list(F.UNIFORMS), "outputs": list(F.OUTPUTS), "dt": F.DT},
        "view": list(F.VIEW), "emit_span": F.EMIT_SPAN, "nozzle": list(F.NOZZLE),
        # how the page's generic compute runtime presents and starts the product
        "initial": {"x": F.NOZZLE[0], "y": F.NOZZLE[1]},
        "points": {"x": "x", "y": "y", "rgb": ["cr", "cg", "cb"], "live": ["age", "t_emit"],
                   "landed": "landed", "size": "radius"},
        "width": F.WIDTH, "basin_y0": F.BASIN_Y0, "layout_columns": list(F.LAYOUT_COLUMNS),
        "species": [list(item[:4]) + [bool(item[4])] for item in F.SPECIES],
        "settle_seconds": F.EMIT_SPAN + F.FLIGHT[1] + 3.0,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--web-dir", default=str(ROOT / "build" / "meowdoku" / "web"))
    parser.add_argument("--sizes", default=",".join(SIZES))
    args = parser.parse_args(argv)
    manifest = build(Path(args.web_dir), tuple(args.sizes.split(",")))
    for item in manifest["sweep"]:
        print(f"  verdict n={item['n']:>6}: pays={item['pays']} speedup {item['speedup']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
