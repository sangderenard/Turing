"""Build the fountain's page products from the one authored law.

``fountain.make_law(pad, active)`` is traced ONCE per size through the SSA
tensor backend (the same trace ``LawPiece._compile_native`` and
``graph_wasm._compile`` use), and from that one trace the compiler produces:

* the DEPLOYMENT VERDICT -- ``webgpu_compute_payoff`` over the traced module,
  charged for every dispatch the program needs;
* the WEBGPU COMPUTE PROGRAM -- ``ssa_webgpu_backend.emit_tensor_program_passes``
  cuts the traced elementwise-and-gather kernel calls into dispatches over one
  arena buffer and emits the WGSL.  Nothing of the law is hand-written in
  WGSL, and no per-id data ships: the law derives everything from the id
  (``pid`` is the shader's own invocation index);
* for sizes up to ``WASM_MAX`` ids, the WASM PRODUCT
  (``compile_graph_forward_to_wasm``, ``graph_wasm._compile``): the route when
  the verdict says the dispatches do not pay or the browser has no WebGPU.
  A size above it has no wasm route and needs WebGPU.

Files written to the page's directory (rebuilt only when their source digest
changes):

    fountain.manifest.json        sizes, verdicts, ABIs, memory per id, the declared cost model
    fountain_<tag>.wgsl           the compute program for that size
    fountain_<tag>.wasm / .abi.json   the wasm product (small sizes)

    python examples/meowdoku/fountain_build.py [--web-dir DIR] [--sizes 4k,1M]
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _path in (str(ROOT), str(ROOT / "examples")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from meowdoku import fountain as F  # noqa: E402

SCHEMA = "turing.fountain-products.v3"
SIZES = dict(F.SIZES)                    # tag -> (active ids, sort length)
WASM_MAX = 4096                          # sizes up to this also get a wasm product
WGSL_CONTRACT = "fast"                   # float64 storage -> f32 through a declared backend identity
WASM_CONTRACT = "develop"
DISPLAY_MAX = 262144                     # the most points the page draws: the display samples every k-th id


def _sources() -> str:
    from src.compiler import deployment_classification, llvm_training_runtime, ssa_webgpu_backend
    return "\0".join(inspect.getsource(m) for m in (F, sys.modules[__name__], ssa_webgpu_backend,
                                                     deployment_classification, llvm_training_runtime))


def digest(tag: str) -> str:
    return hashlib.sha256(f"{tag}:{SIZES[tag]}\0{_sources()}".encode()).hexdigest()[:16]


def shapes_for(pad: int) -> dict:
    return {name: ((pad,) if name in F.STATE + F.CONSTANTS else ()) for name in F.ARGUMENTS}


def trace(pad: int, active: int, contract: str):
    """The law traced at ``pad`` ids: (program, output values, bindings)."""
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    law = F.make_law(pad, active)
    with scoped_work_contract(PRESETS[contract]):
        program = SSATensorProgram("fountain_step")
        shapes = shapes_for(pad)
        bindings = {name: SSATensorOperations.input(program, shapes[name]) for name in F.ARGUMENTS}
        with program.activate():
            produced = law(*(bindings[name] for name in F.ARGUMENTS))
        outputs = [item.data.value for item in produced]
    return program, outputs, bindings


def passes_for(tag_or_pad, active: int | None = None):
    """The compiler-generated dispatch program (and the trace it came from) for a size tag,
    or for ``(pad, active)``."""
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    pad, act = SIZES[tag_or_pad][1], SIZES[tag_or_pad][0] if isinstance(tag_or_pad, str) else (int(tag_or_pad), int(active or tag_or_pad))
    program, outputs, bindings = trace(pad, act, WGSL_CONTRACT)
    ids = {name: int(getattr(binding.data, "value").id) for name, binding in bindings.items()}
    with scoped_work_contract(PRESETS[WGSL_CONTRACT]):
        compiled = webgpu.emit_tensor_program_passes(
            program.module, "fountain_step", outputs, count=pad, name=f"fountain_{pad}",
            iota_args=[ids[name] for name in F.IOTA])
    return compiled, program, outputs, ids


def verdict_for(tag_or_pad, active: int | None = None, model=None):
    """The compiler's WebGPU-compute verdict, charged for the dispatches its generated program needs."""
    from src.compiler.deployment_classification import WebGPUComputeCostModel, webgpu_compute_payoff
    compiled, program, outputs, _ids = passes_for(tag_or_pad, active)
    return webgpu_compute_payoff(program.module, "fountain_step", outputs=outputs,
                                 model=model or WebGPUComputeCostModel(), dispatches=compiled.dispatches)


def build_size(tag: str, web_dir: Path, wasm: bool = True) -> dict:
    from meowdoku import graph_wasm
    from src.compiler.deployment_classification import WebGPUComputeCostModel, webgpu_compute_payoff
    started = time.perf_counter()
    from src.compiler import ssa_webgpu_backend as webgpu
    active, pad = SIZES[tag]
    compiled, program, outputs, ids = passes_for(tag)
    if not compiled.complete:
        raise RuntimeError("fountain shader is incomplete: " + "; ".join(s.format() for s in compiled.shortfalls))
    payoff = webgpu_compute_payoff(program.module, "fountain_step", outputs=outputs,
                                   model=WebGPUComputeCostModel(), dispatches=compiled.dispatches)
    (web_dir / f"fountain_{tag}.wgsl").write_text(compiled.source, encoding="utf-8", newline="\n")
    out_ids = [int(v.id) for v in outputs]
    geo = F.basin(active)
    stride = max(1, -(-pad // DISPLAY_MAX))
    record = {
        "tag": tag, "n": active, "pad": pad,
        "route": "webgpu-compute" if payoff.pays else "wasm",
        "verdict": payoff.as_record(),
        "wgsl": f"fountain_{tag}.wgsl",
        "wgsl_abi": {
            "count": pad, "workgroup_size": list(compiled.workgroup_size), "groups": list(compiled.groups),
            "entries": list(compiled.entries), "slots": compiled.slots, "dispatches": compiled.dispatches,
            "lane_base": compiled.lane_base, "arena_elements": compiled.arena_elements,
            "bindings": compiled.bindings, "slots_per_binding": compiled.slots_per_binding,
            # the logical->physical map, declared once: slot s -> lane s // slots_per_binding, element (s % slots_per_binding) * count;
            # the uniform head is binding `bindings`.  Every pass (step, commit, pack) binds bindings + 1 storage buffers.
            "storage_bindings_per_pass": compiled.bindings + 1, "binding_budget": webgpu.MAX_STORAGE_BINDINGS_PER_STAGE,
            "binding_bytes": min(compiled.slots_per_binding, compiled.slots) * pad * 4,
            "arguments": list(F.ARGUMENTS), "uniforms": list(F.UNIFORMS), "iota": list(F.IOTA),
            "outputs": list(F.OUTPUTS),
            "feed_slots": {name: compiled.feed_slots[ids[name]] for name in F.ARGUMENTS if ids[name] in compiled.feed_slots},
            "uniform_index": {name: compiled.uniform_index[ids[name]] for name in F.UNIFORMS},
            "output_slots": {name: compiled.output_slots[out_ids[k]] for k, name in enumerate(F.OUTPUTS)},
            "backend_identities": [d.as_record() for d in compiled.backend_identity_decisions],
        },
        "memory": {"arena_bytes": compiled.arena_elements * 4, "bytes_per_id": compiled.arena_elements * 4 / pad,
                   "display_ids": -(-pad // stride), "display_bytes": 3 * 16 * -(-pad // stride)},
        "display_stride": stride, "emit_rate": F.EMIT_SPAN / active,
        "geometry": {k: (list(v) if isinstance(v, tuple) else v) for k, v in geo.items()},
        "uniforms": {k: float(v) for k, v in F.uniforms(n=active).items()},
        "source_digest": digest(tag),
    }
    if pad <= WASM_MAX and wasm:
        wasm_abi = graph_wasm._compile(
            f"fountain_{tag}", shapes_for(pad),
            lambda b: F.make_law(pad, active)(*(b[name] for name in F.ARGUMENTS)),
            F.OUTPUTS, WASM_CONTRACT, web_dir,
            {"argument_names": list(F.ARGUMENTS), "count": pad})
        record.update({"wasm": f"fountain_{tag}.wasm", "wasm_abi": f"fountain_{tag}.abi.json",
                       "wasm_buffers": len(wasm_abi["order"])})
    record["build_seconds"] = round(time.perf_counter() - started, 1)
    return record


def build(web_dir: Path, sizes: tuple[str, ...] = tuple(SIZES), wasm: bool = True) -> dict:
    """Write the page products; returns the manifest.  Sizes whose source digest is unchanged are kept."""
    from src.compiler.deployment_classification import WebGPUComputeCostModel
    web_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = web_dir / "fountain.manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = {}
    for tag in sizes:
        previous = (old.get("sizes") or {}).get(tag)
        files = [web_dir / f"fountain_{tag}.wgsl"]
        if SIZES[tag][1] <= WASM_MAX and wasm:
            files += [web_dir / f"fountain_{tag}.wasm", web_dir / f"fountain_{tag}.abi.json"]
        if previous and previous.get("source_digest") == digest(tag) and all(p.exists() for p in files):
            records[tag] = previous
            continue
        records[tag] = build_size(tag, web_dir, wasm)
        r = records[tag]
        print(f"[fountain] {tag}: {r['n']} ids (sort {r['pad']}) -> {r['route']} (predicted {r['verdict']['speedup']:.2f}x, "
              f"{r['wgsl_abi']['dispatches']} dispatches, {r['memory']['bytes_per_id']:.0f} B/id), built in {r['build_seconds']}s", flush=True)
    model = WebGPUComputeCostModel()
    geo = F.basin(F.N_FULL)
    manifest = {
        "schema": SCHEMA, "sizes": records, "cost_model": model.__dict__,
        "law": {"arguments": list(F.ARGUMENTS), "state": list(F.STATE), "constants": list(F.CONSTANTS),
                "uniforms": list(F.UNIFORMS), "outputs": list(F.OUTPUTS), "diagnostics": list(F.DIAGNOSTICS), "dt": F.DT,
                "runtime_inputs": list(F.RUNTIME_INPUTS), "gravity": F.G},
        "emit_span": F.EMIT_SPAN, "initial": {},
        "points": {"x": "x", "y": "y", "vx": "vx", "vy": "vy", "rgb": ["cr", "cg", "cb"], "live": ["age"],
                   "radius": "rad", "rest_speed": 0.15},
        "reference_view": list(geo["view"]),
        "species": [list(item) for item in F.SPECIES],
        "settle": F.SETTLE,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--web-dir", default=str(ROOT / "build" / "meowdoku" / "web"))
    parser.add_argument("--sizes", default=",".join(SIZES))
    args = parser.parse_args(argv)
    manifest = build(Path(args.web_dir), tuple(args.sizes.split(",")))
    for tag, r in manifest["sizes"].items():
        print(f"  {tag:>6}: pays={r['verdict']['pays']} speedup {r['verdict']['speedup']:.2f}x  {r['memory']['arena_bytes'] / 1e6:.0f} MB arena")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
