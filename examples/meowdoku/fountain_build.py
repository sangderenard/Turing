"""Build the fountain's page products from the one authored law.

``fountain.make_law(n)`` is traced ONCE per particle count through the SSA
tensor backend (the same trace ``LawPiece._compile_native`` and
``graph_wasm._compile`` use), and from that one trace the compiler produces:

* the DEPLOYMENT VERDICT -- ``webgpu_compute_payoff`` over the traced module,
  charged for every dispatch the program needs;
* the WEBGPU COMPUTE PROGRAM -- ``ssa_webgpu_backend.emit_tensor_program_passes``
  cuts the traced elementwise-and-gather kernel calls into dispatches over one
  arena buffer (the sort's stages are chained dispatches; the search and the
  contact loop fuse into one) and emits the WGSL.  Nothing of the law is
  hand-written in WGSL;
* the WASM PRODUCT -- ``compile_graph_forward_to_wasm`` (``graph_wasm._compile``),
  the route when the verdict says the dispatch does not pay or the browser has
  no WebGPU.

Files written to the page's directory (rebuilt only when their source digest
changes):

    fountain.manifest.json        sizes, verdicts, ABIs, the declared cost model
    fountain_<tag>.wgsl           the compute program for that size
    fountain_<tag>.wasm / .abi.json   the wasm product

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

SCHEMA = "turing.fountain-products.v2"
#: tag -> particle count (a power of two: the sort is a bitonic network).  ``full`` is the
#: game; ``small`` is a handful of particles on which the compiler's verdict is NO.
SIZES = dict(F.SIZES)
#: extra counts the verdict is published for (no products are built for them)
SWEEP = (64, 256, 1024, 4096, 16384)
WGSL_CONTRACT = "fast"           # float64 storage -> f32 through a declared backend identity
WASM_CONTRACT = "develop"


def _sources() -> str:
    from src.compiler import deployment_classification, llvm_training_runtime, ssa_webgpu_backend
    return "\0".join(inspect.getsource(m) for m in (F, sys.modules[__name__], ssa_webgpu_backend,
                                                     deployment_classification, llvm_training_runtime))


def digest(tag: str, n: int) -> str:
    return hashlib.sha256(f"{tag}:{n}\0{_sources()}".encode()).hexdigest()[:16]


def shapes_for(n: int) -> dict:
    return {name: ((n,) if name in F.STATE + F.CONSTANTS else ()) for name in F.ARGUMENTS}


def trace(n: int, contract: str):
    """The law traced at ``n`` particles: (program, output values, bindings)."""
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    law = F.make_law(n)
    with scoped_work_contract(PRESETS[contract]):
        program = SSATensorProgram("fountain_step")
        shapes = shapes_for(n)
        bindings = {name: SSATensorOperations.input(program, shapes[name]) for name in F.ARGUMENTS}
        with program.activate():
            produced = law(*(bindings[name] for name in F.ARGUMENTS))
        outputs = [item.data.value for item in produced]
    return program, outputs, bindings


def passes_for(n: int):
    """The compiler-generated dispatch program at ``n`` particles (and the trace it came from)."""
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    program, outputs, bindings = trace(n, WGSL_CONTRACT)
    with scoped_work_contract(PRESETS[WGSL_CONTRACT]):
        compiled = webgpu.emit_tensor_program_passes(
            program.module, "fountain_step", outputs, count=n, name=f"fountain_{n}")
    ids = {name: int(getattr(binding.data, "value").id) for name, binding in bindings.items()}
    return compiled, program, outputs, ids


def verdict_for(n: int, model=None):
    """The compiler's WebGPU-compute verdict for the law at ``n`` particles, charged for the
    dispatches its generated program needs."""
    from src.compiler.deployment_classification import WebGPUComputeCostModel, webgpu_compute_payoff
    compiled, program, outputs, _ids = passes_for(n)
    return webgpu_compute_payoff(program.module, "fountain_step", outputs=outputs,
                                 model=model or WebGPUComputeCostModel(), dispatches=compiled.dispatches)


def build_size(tag: str, n: int, web_dir: Path) -> dict:
    from meowdoku import graph_wasm
    started = time.perf_counter()
    compiled, program, outputs, ids = passes_for(n)
    if not compiled.complete:
        raise RuntimeError("fountain shader is incomplete: " + "; ".join(s.format() for s in compiled.shortfalls))
    from src.compiler.deployment_classification import WebGPUComputeCostModel, webgpu_compute_payoff
    payoff = webgpu_compute_payoff(program.module, "fountain_step", outputs=outputs,
                                   model=WebGPUComputeCostModel(), dispatches=compiled.dispatches)
    (web_dir / f"fountain_{tag}.wgsl").write_text(compiled.source, encoding="utf-8", newline="\n")
    law = F.make_law(n)
    wasm_abi = graph_wasm._compile(
        f"fountain_{tag}", shapes_for(n),
        lambda b: law(*(b[name] for name in F.ARGUMENTS)),
        F.OUTPUTS, WASM_CONTRACT, web_dir,
        {"argument_names": list(F.ARGUMENTS), "count": n})
    by_id = {ids[name]: name for name in F.ARGUMENTS}
    out_ids = [int(v.id) for v in outputs]
    # the ABI: arena slot of every formal and every output, by name
    feed_slots = {name: compiled.feed_slots[ids[name]] for name in F.ARGUMENTS}
    out_slots = {name: compiled.output_slots[out_ids[k]] for k, name in enumerate(F.OUTPUTS)}
    del by_id
    return {
        "tag": tag, "n": n,
        "route": "webgpu-compute" if payoff.pays else "wasm",
        "verdict": payoff.as_record(),
        "wgsl": f"fountain_{tag}.wgsl",
        "wgsl_abi": {
            "count": n, "workgroup_size": list(compiled.workgroup_size), "groups": list(compiled.groups),
            "entries": list(compiled.entries), "slots": compiled.slots, "dispatches": compiled.dispatches,
            "arguments": list(F.ARGUMENTS), "uniforms": list(F.UNIFORMS), "outputs": list(F.OUTPUTS),
            "feed_slots": feed_slots, "output_slots": out_slots,
            "backend_identities": [d.as_record() for d in compiled.backend_identity_decisions],
        },
        "uniforms": {k: float(v) for k, v in F.uniforms().items()},
        "wasm": f"fountain_{tag}.wasm", "wasm_abi": f"fountain_{tag}.abi.json",
        "wasm_buffers": len(wasm_abi["order"]),
        "geometry": F.basin(n),
        "source_digest": digest(tag, n),
        "build_seconds": round(time.perf_counter() - started, 1),
    }


def build(web_dir: Path, sizes: tuple[str, ...] = tuple(SIZES), sweep: bool = True) -> dict:
    """Write the page products; returns the manifest."""
    from src.compiler.deployment_classification import WebGPUComputeCostModel
    web_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = web_dir / "fountain.manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = {}
    for tag in sizes:
        n = SIZES[tag]
        previous = (old.get("sizes") or {}).get(tag)
        files = (web_dir / f"fountain_{tag}.wgsl", web_dir / f"fountain_{tag}.wasm", web_dir / f"fountain_{tag}.abi.json")
        if previous and previous.get("source_digest") == digest(tag, n) and all(p.exists() for p in files):
            records[tag] = previous
            continue
        records[tag] = build_size(tag, n, web_dir)
        print(f"[fountain] {tag}: {n} particles -> {records[tag]['route']} "
              f"(predicted {records[tag]['verdict']['speedup']:.2f}x, {records[tag]['wgsl_abi']['dispatches']} dispatches), "
              f"built in {records[tag]['build_seconds']}s", flush=True)
    model = WebGPUComputeCostModel()
    cached_sweep = old.get("sweep") if old.get("schema") == SCHEMA and old.get("sweep_digest") == digest("sweep", 0) else None
    sweep_rows = cached_sweep or ([{"n": n, **verdict_for(n).as_record()} for n in SWEEP] if sweep else [])
    manifest = {
        "schema": SCHEMA, "sizes": records, "cost_model": model.__dict__,
        "sweep": sweep_rows, "sweep_digest": digest("sweep", 0),
        "law": {"arguments": list(F.ARGUMENTS), "state": list(F.STATE), "constants": list(F.CONSTANTS),
                "uniforms": list(F.UNIFORMS), "outputs": list(F.OUTPUTS), "dt": F.DT},
        "view": list(F.VIEW), "emit_span": F.EMIT_SPAN, "nozzle": list(F.NOZZLE),
        # how the page's generic compute runtime presents and starts the product
        "initial": {},
        "points": {"x": "x", "y": "y", "vx": "vx", "vy": "vy", "rgb": ["cr", "cg", "cb"], "live": ["age", "t_emit"],
                   "size": "radius", "rest_speed": 0.15},
        "layout_columns": list(F.LAYOUT_COLUMNS),
        "species": [list(item) for item in F.SPECIES],
        "settle_seconds": F.SETTLE_SECONDS,
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
