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
ROOT = HERE.parents[2]
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


_SOURCES: str | None = None


def _sources() -> str:
    """The source text the products are keyed by, read ONCE per process and kept.  inspect.getsource re-reads the files
    from disk, so a file edited while a long build ran stamped the build's later sizes with the digest of code it never
    compiled (2026-11-06: small and 4k keyed to one fountain.py, 16k+ to the next; the next start rebuilt both fallbacks)."""
    global _SOURCES
    if _SOURCES is None:
        from src.compiler import deployment_classification, llvm_training_runtime, ssa_webgpu_backend
        _SOURCES = "\0".join(inspect.getsource(m) for m in (F, sys.modules[__name__], ssa_webgpu_backend,
                                                            deployment_classification, llvm_training_runtime))
    return _SOURCES


def digest(tag: str, body: bool = False) -> str:
    return hashlib.sha256(f"{tag}:{SIZES[tag]}:{'body' if body else ''}\0{_sources()}".encode()).hexdigest()[:16]


def layout(body: bool = False) -> dict:
    """The law's names for the default law or its hinged-body (waterwheel) variant."""
    if not body:
        return {"arguments": F.ARGUMENTS, "state": F.STATE, "outputs": F.OUTPUTS, "diagnostics": F.DIAGNOSTICS,
                "uniforms": F.UNIFORMS, "runtime_inputs": F.RUNTIME_INPUTS, "prefix": "fountain"}
    return {"arguments": F.BODY_ARGUMENTS, "state": F.STATE + F.BODY_STATE, "outputs": F.BODY_OUTPUTS,
            "diagnostics": F.DIAGNOSTICS + F.BODY_DIAGNOSTICS, "uniforms": F.UNIFORMS + F.BODY_UNIFORMS,
            "runtime_inputs": F.RUNTIME_INPUTS + F.BODY_RUNTIME_INPUTS, "prefix": "fountain_wheel"}


def shapes_for(pad: int, body: bool = False) -> dict:
    lay = layout(body)
    return {name: ((pad,) if name in lay["state"] + F.CONSTANTS else ()) for name in lay["arguments"]}


def trace(pad: int, active: int, contract: str, body: bool = False):
    """The law traced at ``pad`` ids: (program, output values, bindings)."""
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    law = F.make_law(pad, active, body=body)
    names = layout(body)["arguments"]
    with scoped_work_contract(PRESETS[contract]):
        program = SSATensorProgram("fountain_step")
        shapes = shapes_for(pad, body)
        bindings = {name: SSATensorOperations.input(program, shapes[name]) for name in names}
        with program.activate():
            produced = law(*(bindings[name] for name in names))
        outputs = [item.data.value for item in produced]
    return program, outputs, bindings


def passes_for(tag_or_pad, active: int | None = None, body: bool = False):
    """The compiler-generated dispatch program (and the trace it came from) for a size tag,
    or for ``(pad, active)``."""
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    pad, act = SIZES[tag_or_pad][1], SIZES[tag_or_pad][0] if isinstance(tag_or_pad, str) else (int(tag_or_pad), int(active or tag_or_pad))
    program, outputs, bindings = trace(pad, act, WGSL_CONTRACT, body)
    ids = {name: int(getattr(binding.data, "value").id) for name, binding in bindings.items()}
    with scoped_work_contract(PRESETS[WGSL_CONTRACT]):
        compiled = webgpu.emit_tensor_program_passes(
            program.module, "fountain_step", outputs, count=pad, name=f"{layout(body)['prefix']}_{pad}",
            iota_args=[ids[name] for name in F.IOTA])
    return compiled, program, outputs, ids


def verdict_for(tag_or_pad, active: int | None = None, model=None, body: bool = False):
    """The compiler's WebGPU-compute verdict, charged for the dispatches its generated program needs."""
    from src.compiler.deployment_classification import WebGPUComputeCostModel, webgpu_compute_payoff
    compiled, program, outputs, _ids = passes_for(tag_or_pad, active, body)
    return webgpu_compute_payoff(program.module, "fountain_step", outputs=outputs,
                                 model=model or WebGPUComputeCostModel(), dispatches=compiled.dispatches)


def build_size(tag: str, web_dir: Path, wasm: bool = True, body: bool = False) -> dict:
    from meowdoku import graph_wasm
    from src.compiler.deployment_classification import WebGPUComputeCostModel, webgpu_compute_payoff
    started = time.perf_counter()
    from src.compiler import ssa_webgpu_backend as webgpu
    active, pad = SIZES[tag]
    lay = layout(body)
    prefix = lay["prefix"]
    compiled, program, outputs, ids = passes_for(tag, None, body)
    if not compiled.complete:
        raise RuntimeError("fountain shader is incomplete: " + "; ".join(s.format() for s in compiled.shortfalls))
    payoff = webgpu_compute_payoff(program.module, "fountain_step", outputs=outputs,
                                   model=WebGPUComputeCostModel(), dispatches=compiled.dispatches)
    (web_dir / f"{prefix}_{tag}.wgsl").write_text(compiled.source, encoding="utf-8", newline="\n")
    out_ids = [int(v.id) for v in outputs]
    geo = F.basin(active)
    stride = max(1, -(-pad // DISPLAY_MAX))
    record = {
        "tag": tag, "n": active, "pad": pad,
        "route": "webgpu-compute" if payoff.pays else "wasm",
        "verdict": payoff.as_record(),
        "wgsl": f"{prefix}_{tag}.wgsl",
        "wgsl_abi": {
            "count": pad, "workgroup_size": list(compiled.workgroup_size), "groups": list(compiled.groups),
            "entries": list(compiled.entries), "slots": compiled.slots, "dispatches": compiled.dispatches,
            "lane_base": compiled.lane_base, "arena_elements": compiled.arena_elements,
            "bindings": compiled.bindings, "slots_per_binding": compiled.slots_per_binding,
            # the logical->physical map, declared once: slot s -> lane s // slots_per_binding, element (s % slots_per_binding) * count;
            # the uniform head is binding `bindings`.  Every pass (step, commit, pack) binds bindings + 1 storage buffers.
            "storage_bindings_per_pass": compiled.bindings + 1, "binding_budget": webgpu.MAX_STORAGE_BINDINGS_PER_STAGE,
            "binding_bytes": min(compiled.slots_per_binding, compiled.slots) * pad * 4,
            "arguments": list(lay["arguments"]), "uniforms": list(lay["uniforms"]), "iota": list(F.IOTA),
            "outputs": list(lay["outputs"]),
            "feed_slots": {name: compiled.feed_slots[ids[name]] for name in lay["arguments"] if ids[name] in compiled.feed_slots},
            "uniform_index": {name: compiled.uniform_index[ids[name]] for name in lay["uniforms"]},
            "output_slots": {name: compiled.output_slots[out_ids[k]] for k, name in enumerate(lay["outputs"])},
            "backend_identities": [d.as_record() for d in compiled.backend_identity_decisions],
        },
        "memory": {"arena_bytes": compiled.arena_elements * 4, "bytes_per_id": compiled.arena_elements * 4 / pad,
                   "display_ids": -(-pad // stride), "display_bytes": 3 * 16 * -(-pad // stride)},
        "display_stride": stride, "emit_rate": F.EMIT_SPAN / active,
        "geometry": {k: (list(v) if isinstance(v, tuple) else v) for k, v in geo.items()},
        "uniforms": {k: float(v) for k, v in F.uniforms(n=active, shape="waterwheel" if body else "box").items()},
        "source_digest": digest(tag, body),
    }
    record["build_seconds"] = round(time.perf_counter() - started, 1)
    if pad <= WASM_MAX and wasm:
        build_fallback(tag, web_dir, record, body)
    return record


def build_fallback(tag: str, web_dir: Path, record: dict, body: bool = False) -> dict:
    """The CPU fallback of one size (the wasm product, for a browser without WebGPU), added to its ``record``.  It is the
    slow part of a build (the size-specialised law lowered whole: minutes, ~30 at 4k), so ``build`` makes it AFTER every
    WebGPU product is written."""
    from meowdoku import graph_wasm
    active, pad = SIZES[tag]
    started = time.perf_counter()
    lay = layout(body)
    wasm_abi = graph_wasm._compile(
        f"{lay['prefix']}_{tag}", shapes_for(pad, body),
        lambda b: F.make_law(pad, active, body=body)(*(b[name] for name in lay["arguments"])),
        lay["outputs"], WASM_CONTRACT, web_dir,
        {"argument_names": list(lay["arguments"]), "count": pad})
    record.update({"wasm": f"{lay['prefix']}_{tag}.wasm", "wasm_abi": f"{lay['prefix']}_{tag}.abi.json",
                   "wasm_buffers": len(wasm_abi["order"]), "wasm_seconds": round(time.perf_counter() - started, 1)})
    return record


def build(web_dir: Path, sizes: tuple[str, ...] = tuple(SIZES), wasm: bool = True, on_gpu_ready=None) -> dict:
    """Write the page products; returns the manifest.  Sizes whose source digest is unchanged are kept.

    Two passes: first every size's WebGPU product (seconds each), then the manifest is written and ``on_gpu_ready``
    called -- the page can run -- and only then the slow CPU fallbacks (``build_fallback``), the manifest rewritten as
    each one lands."""
    web_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = web_dir / "fountain.manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = {}
    for tag in sizes:
        previous = (old.get("sizes") or {}).get(tag)
        if previous and previous.get("source_digest") == digest(tag) and (web_dir / f"fountain_{tag}.wgsl").exists():
            if previous.get("wasm") and not all((web_dir / previous[k]).exists() for k in ("wasm", "wasm_abi")):
                previous = {k: v for k, v in previous.items() if not k.startswith("wasm")}
            records[tag] = previous
            continue
        records[tag] = build_size(tag, web_dir, wasm=False)
        r = records[tag]
        print(f"[fountain] {tag}: {r['n']} ids (sort {r['pad']}) -> {r['route']} (predicted {r['verdict']['speedup']:.2f}x, "
              f"{r['wgsl_abi']['dispatches']} dispatches, {r['memory']['bytes_per_id']:.0f} B/id), built in {r['build_seconds']}s", flush=True)
    manifest = _manifest(records)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if on_gpu_ready is not None:
        on_gpu_ready(manifest)
    if wasm:
        for tag in sizes:
            r = records[tag]
            if SIZES[tag][1] > WASM_MAX or r.get("wasm"):
                continue
            build_fallback(tag, web_dir, r)
            print(f"[fountain] {tag}: CPU fallback (wasm) built in {r['wasm_seconds']}s", flush=True)
            manifest = _manifest(records)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


#: the sizes the hinged-body (waterwheel) law is built at: the 10M size is not offered (two more state columns = 44 slots per id)
WHEEL_SIZES = tuple(tag for tag in SIZES if SIZES[tag][0] < 10_000_000)


def build_wheel(web_dir: Path, sizes: tuple[str, ...] = WHEEL_SIZES, wasm: bool = True, on_gpu_ready=None) -> dict:
    """The hinged-body (waterwheel) variant's page products, beside the default ones and independent of them:
    ``fountain_wheel.manifest.json`` and ``fountain_wheel_<tag>.wgsl`` / ``.wasm``.  Same two passes as ``build`` (WebGPU products first,
    then the CPU fallbacks); sizes whose source digest is unchanged are kept."""
    web_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = web_dir / "fountain_wheel.manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = {}
    for tag in sizes:
        previous = (old.get("sizes") or {}).get(tag)
        if previous and previous.get("source_digest") == digest(tag, True) and (web_dir / f"fountain_wheel_{tag}.wgsl").exists():
            if previous.get("wasm") and not all((web_dir / previous[k]).exists() for k in ("wasm", "wasm_abi")):
                previous = {k: v for k, v in previous.items() if not k.startswith("wasm")}
            records[tag] = previous
            continue
        records[tag] = build_size(tag, web_dir, wasm=False, body=True)
        r = records[tag]
        print(f"[fountain wheel] {tag}: {r['n']} ids -> {r['route']} ({r['wgsl_abi']['dispatches']} dispatches, "
              f"{r['memory']['bytes_per_id']:.0f} B/id), built in {r['build_seconds']}s", flush=True)
    manifest = _manifest(records, True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if on_gpu_ready is not None:
        on_gpu_ready(manifest)
    if wasm:
        for tag in sizes:
            r = records[tag]
            if SIZES[tag][1] > WASM_MAX or r.get("wasm"):
                continue
            build_fallback(tag, web_dir, r, True)
            manifest = _manifest(records, True)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _manifest(records: dict, body: bool = False) -> dict:
    from src.compiler.deployment_classification import WebGPUComputeCostModel
    model = WebGPUComputeCostModel()
    geo = F.basin(F.N_FULL)
    return {
        "schema": SCHEMA, "sizes": records, "cost_model": model.__dict__,
        "law": {"arguments": list(layout(body)["arguments"]), "state": list(layout(body)["state"]), "constants": list(F.CONSTANTS),
                "uniforms": list(layout(body)["uniforms"]), "outputs": list(layout(body)["outputs"]),
                "diagnostics": list(layout(body)["diagnostics"]), "dt": F.DT,
                "runtime_inputs": list(layout(body)["runtime_inputs"]), "gravity": F.G, "dof": F.DOF},
        "emit_span": F.EMIT_SPAN, "initial": {},
        "points": {"x": "x", "y": "y", "vx": "vx", "vy": "vy", "rgb": ["cr", "cg", "cb"], "live": ["age"], "placed": ["x", "y"],
                   "radius": "rad", "rest_speed": 0.15},
        "reference_view": list(geo["view"]),
        "species": [list(item) for item in F.SPECIES],
        "settle": F.SETTLE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--web-dir", default=str(ROOT / "build" / "meowdoku" / "web"))
    parser.add_argument("--sizes", default=",".join(SIZES))
    parser.add_argument("--wheel", action="store_true", help="build the hinged-body (waterwheel) variant instead")
    parser.add_argument("--no-wasm", action="store_true", help="WebGPU products only (skip the slow CPU fallbacks)")
    args = parser.parse_args(argv)
    sizes = tuple(args.sizes.split(","))
    manifest = build_wheel(Path(args.web_dir), tuple(t for t in sizes if t in WHEEL_SIZES), wasm=not args.no_wasm) if args.wheel else build(Path(args.web_dir), sizes, wasm=not args.no_wasm)
    for tag, r in manifest["sizes"].items():
        print(f"  {tag:>6}: pays={r['verdict']['pays']} speedup {r['verdict']['speedup']:.2f}x  {r['memory']['arena_bytes'] / 1e6:.0f} MB arena")
    return 0


# <purrline>
# ------------------------------------------------------------------------------------------------------------------------
# THE PURRLINE SAND VARIANT (fountain.make_law_purrline): its own products, ``fountain_purrline.manifest.json`` and ``fountain_purrline_<tag>.wgsl``
# (/ ``.wasm``), built after the default ones and independent of them.  Everything of the variant is between ``# <purrline>`` markers in
# fountain.py and in this file; the source digest of the DEFAULT and wheel products leaves those regions out (``_sources`` below), so they are
# byte for byte the products they were: same digests, nothing of them rebuilds when only this variant changes.
import re as _re

_MARKED = _re.compile(r"^[ \t]*# <purrline>\n.*?^[ \t]*# </purrline>\n", _re.S | _re.M)
_SOURCES_PLAIN: str | None = None


def _module_sources(strip: bool) -> str:
    from src.compiler import deployment_classification, llvm_training_runtime, ssa_webgpu_backend
    texts = [inspect.getsource(m) for m in (F, sys.modules[__name__])]
    if strip:
        texts = [_MARKED.sub("", t) for t in texts]
    return "\0".join(texts + [inspect.getsource(m) for m in (ssa_webgpu_backend, deployment_classification, llvm_training_runtime)])


def _sources() -> str:
    """The default / wheel products' source text: the same as before the variant, the purrline regions left out (read ONCE per process)."""
    global _SOURCES_PLAIN
    if _SOURCES_PLAIN is None:
        _SOURCES_PLAIN = _module_sources(True)
    return _SOURCES_PLAIN


def purrline_digest(tag: str) -> str:
    """The variant's own digest: its sources WHOLE (the markers' contents included)."""
    return hashlib.sha256(f"{tag}:{SIZES[tag]}:purrline\0{_module_sources(False)}".encode()).hexdigest()[:16]


def purrline_layout() -> dict:
    return {"arguments": F.PURRLINE_ARGUMENTS, "state": F.STATE, "outputs": F.OUTPUTS, "diagnostics": F.DIAGNOSTICS, "uniforms": F.PURRLINE_ALL_UNIFORMS,
            "runtime_inputs": F.PURRLINE_RUNTIME_INPUTS, "prefix": "fountain_purrline"}


def purrline_shapes(pad: int) -> dict:
    lay = purrline_layout()
    return {name: ((pad,) if name in lay["state"] + F.CONSTANTS else ()) for name in lay["arguments"]}


def purrline_passes(tag: str):
    """The compiler-generated dispatch program of the variant at a size tag: (compiled, program, outputs, ids)."""
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    active, pad = SIZES[tag]
    law, names, shapes = F.make_law_purrline(pad, active), purrline_layout()["arguments"], purrline_shapes(pad)
    with scoped_work_contract(PRESETS[WGSL_CONTRACT]):
        program = SSATensorProgram("fountain_step")
        bindings = {name: SSATensorOperations.input(program, shapes[name]) for name in names}
        with program.activate():
            produced = law(*(bindings[name] for name in names))
        outputs = [item.data.value for item in produced]
        ids = {name: int(getattr(binding.data, "value").id) for name, binding in bindings.items()}
        compiled = webgpu.emit_tensor_program_passes(program.module, "fountain_step", outputs, count=pad, name=f"fountain_purrline_{pad}",
                                                     iota_args=[ids[name] for name in F.IOTA])
    return compiled, program, outputs, ids


def build_size_purrline(tag: str, web_dir: Path, wasm: bool = True) -> dict:
    from src.compiler.deployment_classification import WebGPUComputeCostModel, webgpu_compute_payoff
    from src.compiler import ssa_webgpu_backend as webgpu
    started = time.perf_counter()
    active, pad = SIZES[tag]
    lay = purrline_layout()
    compiled, program, outputs, ids = purrline_passes(tag)
    if not compiled.complete:
        raise RuntimeError("fountain purrline shader is incomplete: " + "; ".join(s.format() for s in compiled.shortfalls))
    payoff = webgpu_compute_payoff(program.module, "fountain_step", outputs=outputs, model=WebGPUComputeCostModel(), dispatches=compiled.dispatches)
    (web_dir / f"fountain_purrline_{tag}.wgsl").write_text(compiled.source, encoding="utf-8", newline="\n")
    out_ids = [int(v.id) for v in outputs]
    geo = F.basin(active)
    stride = max(1, -(-pad // DISPLAY_MAX))
    record = {
        "tag": tag, "n": active, "pad": pad, "route": "webgpu-compute" if payoff.pays else "wasm", "verdict": payoff.as_record(),
        "wgsl": f"fountain_purrline_{tag}.wgsl",
        "wgsl_abi": {
            "count": pad, "workgroup_size": list(compiled.workgroup_size), "groups": list(compiled.groups), "entries": list(compiled.entries),
            "slots": compiled.slots, "dispatches": compiled.dispatches, "lane_base": compiled.lane_base, "arena_elements": compiled.arena_elements,
            "bindings": compiled.bindings, "slots_per_binding": compiled.slots_per_binding,
            "storage_bindings_per_pass": compiled.bindings + 1, "binding_budget": webgpu.MAX_STORAGE_BINDINGS_PER_STAGE,
            "binding_bytes": min(compiled.slots_per_binding, compiled.slots) * pad * 4,
            "arguments": list(lay["arguments"]), "uniforms": list(lay["uniforms"]), "iota": list(F.IOTA), "outputs": list(lay["outputs"]),
            "feed_slots": {name: compiled.feed_slots[ids[name]] for name in lay["arguments"] if ids[name] in compiled.feed_slots},
            "uniform_index": {name: compiled.uniform_index[ids[name]] for name in lay["uniforms"]},
            "output_slots": {name: compiled.output_slots[out_ids[k]] for k, name in enumerate(lay["outputs"])},
            "backend_identities": [d.as_record() for d in compiled.backend_identity_decisions],
        },
        "memory": {"arena_bytes": compiled.arena_elements * 4, "bytes_per_id": compiled.arena_elements * 4 / pad,
                   "display_ids": -(-pad // stride), "display_bytes": 3 * 16 * -(-pad // stride)},
        "display_stride": stride, "emit_rate": F.EMIT_SPAN / active,
        "geometry": {k: (list(v) if isinstance(v, tuple) else v) for k, v in geo.items()},
        "uniforms": F.purrline_spec(tag)["uniforms"], "source_digest": purrline_digest(tag),
    }
    record["build_seconds"] = round(time.perf_counter() - started, 1)
    if pad <= WASM_MAX and wasm:
        build_fallback_purrline(tag, web_dir, record)
    return record


def build_fallback_purrline(tag: str, web_dir: Path, record: dict) -> dict:
    """The CPU fallback (wasm) of one size of the variant, added to its ``record``."""
    from meowdoku import graph_wasm
    active, pad = SIZES[tag]
    started = time.perf_counter()
    lay = purrline_layout()
    wasm_abi = graph_wasm._compile(f"fountain_purrline_{tag}", purrline_shapes(pad),
                                   lambda b: F.make_law_purrline(pad, active)(*(b[name] for name in lay["arguments"])),
                                   lay["outputs"], WASM_CONTRACT, web_dir, {"argument_names": list(lay["arguments"]), "count": pad})
    record.update({"wasm": f"fountain_purrline_{tag}.wasm", "wasm_abi": f"fountain_purrline_{tag}.abi.json", "wasm_buffers": len(wasm_abi["order"]),
                   "wasm_seconds": round(time.perf_counter() - started, 1)})
    return record


#: the sizes the variant is built at: not the 10M one (the extra uniforms are free, but a 10M board is not a thing the page offers)
PURRLINE_SIZES = tuple(tag for tag in SIZES if SIZES[tag][0] < 10_000_000)


def purrline_manifest(records: dict) -> dict:
    from src.compiler.deployment_classification import WebGPUComputeCostModel
    lay, geo = purrline_layout(), F.basin(F.N_FULL)
    return {
        "schema": SCHEMA, "sizes": records, "cost_model": WebGPUComputeCostModel().__dict__,
        "law": {"arguments": list(lay["arguments"]), "state": list(lay["state"]), "constants": list(F.CONSTANTS), "uniforms": list(lay["uniforms"]),
                "outputs": list(lay["outputs"]), "diagnostics": list(lay["diagnostics"]), "dt": F.DT, "runtime_inputs": list(lay["runtime_inputs"]),
                "gravity": F.G, "dof": F.DOF},
        "emit_span": F.EMIT_SPAN, "initial": {},
        "points": {"x": "x", "y": "y", "vx": "vx", "vy": "vy", "rgb": ["cr", "cg", "cb"], "live": ["age"], "placed": ["x", "y"], "radius": "rad", "rest_speed": 0.15},
        "reference_view": list(geo["view"]), "species": [list(item) for item in F.SPECIES], "settle": F.SETTLE,
        "purrline": {"boxes": F.MB_COUNT},
        "presets": {"sdf_purrline": {tag: F.purrline_spec(tag) for tag in records}},
    }


def build_purrline(web_dir: Path, sizes: tuple[str, ...] = PURRLINE_SIZES, wasm: bool = True, on_gpu_ready=None) -> dict:
    """The purrline variant's page products, beside the default ones: ``fountain_purrline.manifest.json`` and ``fountain_purrline_<tag>.wgsl`` /
    ``.wasm``.  Two passes like ``build`` (WebGPU products first, the CPU fallbacks after); sizes whose source digest is unchanged are kept."""
    web_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = web_dir / "fountain_purrline.manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = {}
    for tag in sizes:
        previous = (old.get("sizes") or {}).get(tag)
        if previous and previous.get("source_digest") == purrline_digest(tag) and (web_dir / f"fountain_purrline_{tag}.wgsl").exists():
            if previous.get("wasm") and not all((web_dir / previous[k]).exists() for k in ("wasm", "wasm_abi")):
                previous = {k: v for k, v in previous.items() if not k.startswith("wasm")}
            records[tag] = previous
            continue
        records[tag] = build_size_purrline(tag, web_dir, wasm=False)
        r = records[tag]
        print(f"[fountain purrline] {tag}: {r['n']} ids -> {r['route']} ({r['wgsl_abi']['dispatches']} dispatches, "
              f"{r['memory']['bytes_per_id']:.0f} B/id), built in {r['build_seconds']}s", flush=True)
    manifest = purrline_manifest(records)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if on_gpu_ready is not None:
        on_gpu_ready(manifest)
    if wasm:
        for tag in sizes:
            r = records[tag]
            if SIZES[tag][1] > WASM_MAX or r.get("wasm"):
                continue
            build_fallback_purrline(tag, web_dir, r)
            manifest_path.write_text(json.dumps(purrline_manifest(records)), encoding="utf-8")
    return manifest
# </purrline>
# <mobroad>
# ------------------------------------------------------------------------------------------------------------------------
# THE MOBROAD VARIANT (fountain.make_law_mobroad): its own products, ``fountain_mobroad.manifest.json`` and ``fountain_mobroad_<tag>.wgsl`` (/ ``.wasm``),
# built after the default, wheel and purrline ones and independent of them.  Everything of it is between ``# <mobroad>`` markers (here and in fountain.py).
# The source digest of the DEFAULT and wheel products leaves BOTH variants' regions out and the purrline digest leaves this one out (the three functions
# just below re-bind the helpers the purrline region defined), so none of those products is touched by this variant: same digests, nothing rebuilds.
_MARKED = _re.compile(r"^[ \t]*# <(?:purrline|mobroad)>\n.*?^[ \t]*# </(?:purrline|mobroad)>\n", _re.S | _re.M)
_MARKED_MOB = _re.compile(r"^[ \t]*# <mobroad>\n.*?^[ \t]*# </mobroad>\n", _re.S | _re.M)


def _module_sources(strip: bool, drop=None) -> str:
    """The sources a digest is a function of: ``strip`` leaves out BOTH variants' regions (the default / wheel products), ``drop`` a given pattern's, else whole."""
    from src.compiler import deployment_classification, llvm_training_runtime, ssa_webgpu_backend
    texts = [inspect.getsource(m) for m in (F, sys.modules[__name__])]
    pattern = drop if drop is not None else (_MARKED if strip else None)
    if pattern is not None:
        texts = [pattern.sub("", t) for t in texts]
    return "\0".join(texts + [inspect.getsource(m) for m in (ssa_webgpu_backend, deployment_classification, llvm_training_runtime)])


def purrline_digest(tag: str) -> str:
    """The purrline variant's own digest: its sources whole, the mobroad regions left out (so it is the digest it was)."""
    return hashlib.sha256(f"{tag}:{SIZES[tag]}:purrline\0{_module_sources(False, _MARKED_MOB)}".encode()).hexdigest()[:16]


def mobroad_digest(tag: str) -> str:
    """The mobroad variant's own digest: its sources WHOLE."""
    return hashlib.sha256(f"{tag}:{SIZES[tag]}:mobroad\0{_module_sources(False)}".encode()).hexdigest()[:16]


def mobroad_layout() -> dict:
    return {"arguments": F.MOBROAD_ARGUMENTS, "state": F.STATE, "outputs": F.MOBROAD_OUTPUTS, "diagnostics": F.DIAGNOSTICS + F.MOBROAD_DIAGNOSTICS,
            "uniforms": F.MOBROAD_ALL_UNIFORMS, "runtime_inputs": F.MOBROAD_RUNTIME_INPUTS, "prefix": "fountain_mobroad"}


def mobroad_shapes(pad: int) -> dict:
    lay = mobroad_layout()
    return {name: ((pad,) if name in lay["state"] + F.CONSTANTS else ()) for name in lay["arguments"]}


def mobroad_passes(tag: str):
    """The compiler-generated dispatch program of the variant at a size tag: (compiled, program, outputs, ids)."""
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    active, pad = SIZES[tag]
    law, names, shapes = F.make_law_mobroad(pad, active), mobroad_layout()["arguments"], mobroad_shapes(pad)
    with scoped_work_contract(PRESETS[WGSL_CONTRACT]):
        program = SSATensorProgram("fountain_step")
        bindings = {name: SSATensorOperations.input(program, shapes[name]) for name in names}
        with program.activate():
            produced = law(*(bindings[name] for name in names))
        outputs = [item.data.value for item in produced]
        ids = {name: int(getattr(binding.data, "value").id) for name, binding in bindings.items()}
        compiled = webgpu.emit_tensor_program_passes(program.module, "fountain_step", outputs, count=pad, name=f"fountain_mobroad_{pad}",
                                                     iota_args=[ids[name] for name in F.IOTA])
    return compiled, program, outputs, ids


def build_size_mobroad(tag: str, web_dir: Path, wasm: bool = True) -> dict:
    from src.compiler.deployment_classification import WebGPUComputeCostModel, webgpu_compute_payoff
    from src.compiler import ssa_webgpu_backend as webgpu
    started = time.perf_counter()
    active, pad = SIZES[tag]
    lay = mobroad_layout()
    compiled, program, outputs, ids = mobroad_passes(tag)
    if not compiled.complete:
        raise RuntimeError("fountain mobroad shader is incomplete: " + "; ".join(s.format() for s in compiled.shortfalls))
    payoff = webgpu_compute_payoff(program.module, "fountain_step", outputs=outputs, model=WebGPUComputeCostModel(), dispatches=compiled.dispatches)
    (web_dir / f"fountain_mobroad_{tag}.wgsl").write_text(compiled.source, encoding="utf-8", newline="\n")
    out_ids = [int(v.id) for v in outputs]
    geo = F.mobroad_geometry(F.cell_needed(F.species_params({"fl_mu": F.MOB_MARCH["fl_mu"]})))
    stride = max(1, -(-pad // DISPLAY_MAX))
    record = {
        "tag": tag, "n": active, "pad": pad, "route": "webgpu-compute" if payoff.pays else "wasm", "verdict": payoff.as_record(),
        "wgsl": f"fountain_mobroad_{tag}.wgsl",
        "wgsl_abi": {
            "count": pad, "workgroup_size": list(compiled.workgroup_size), "groups": list(compiled.groups), "entries": list(compiled.entries),
            "slots": compiled.slots, "dispatches": compiled.dispatches, "lane_base": compiled.lane_base, "arena_elements": compiled.arena_elements,
            "bindings": compiled.bindings, "slots_per_binding": compiled.slots_per_binding,
            "storage_bindings_per_pass": compiled.bindings + 1, "binding_budget": webgpu.MAX_STORAGE_BINDINGS_PER_STAGE,
            "binding_bytes": min(compiled.slots_per_binding, compiled.slots) * pad * 4,
            "arguments": list(lay["arguments"]), "uniforms": list(lay["uniforms"]), "iota": list(F.IOTA), "outputs": list(lay["outputs"]),
            "feed_slots": {name: compiled.feed_slots[ids[name]] for name in lay["arguments"] if ids[name] in compiled.feed_slots},
            "uniform_index": {name: compiled.uniform_index[ids[name]] for name in lay["uniforms"]},
            "output_slots": {name: compiled.output_slots[out_ids[k]] for k, name in enumerate(lay["outputs"])},
            "backend_identities": [d.as_record() for d in compiled.backend_identity_decisions],
        },
        "memory": {"arena_bytes": compiled.arena_elements * 4, "bytes_per_id": compiled.arena_elements * 4 / pad,
                   "display_ids": -(-pad // stride), "display_bytes": 3 * 16 * -(-pad // stride)},
        "display_stride": stride, "emit_rate": F.EMIT_SPAN / active,
        "geometry": {k: (list(v) if isinstance(v, tuple) else v) for k, v in geo.items()},
        "uniforms": F.mobroad_spec(tag)["uniforms"], "source_digest": mobroad_digest(tag),
    }
    record["build_seconds"] = round(time.perf_counter() - started, 1)
    if pad <= WASM_MAX and wasm:
        build_fallback_mobroad(tag, web_dir, record)
    return record


def build_fallback_mobroad(tag: str, web_dir: Path, record: dict) -> dict:
    """The CPU fallback (wasm) of one size of the variant, added to its ``record``."""
    from meowdoku import graph_wasm
    active, pad = SIZES[tag]
    started = time.perf_counter()
    lay = mobroad_layout()
    wasm_abi = graph_wasm._compile(f"fountain_mobroad_{tag}", mobroad_shapes(pad),
                                   lambda b: F.make_law_mobroad(pad, active)(*(b[name] for name in lay["arguments"])),
                                   lay["outputs"], WASM_CONTRACT, web_dir, {"argument_names": list(lay["arguments"]), "count": pad})
    record.update({"wasm": f"fountain_mobroad_{tag}.wasm", "wasm_abi": f"fountain_mobroad_{tag}.abi.json", "wasm_buffers": len(wasm_abi["order"]),
                   "wasm_seconds": round(time.perf_counter() - started, 1)})
    return record


#: the sizes the road is built at: the 256-id wasm demonstration and the 4k mob (a larger mob would not fit the road)
MOBROAD_SIZES = ("small", "4k")


def mobroad_manifest(records: dict) -> dict:
    from src.compiler.deployment_classification import WebGPUComputeCostModel
    lay = mobroad_layout()
    return {
        "schema": SCHEMA, "sizes": records, "cost_model": WebGPUComputeCostModel().__dict__,
        "law": {"arguments": list(lay["arguments"]), "state": list(lay["state"]), "constants": list(F.CONSTANTS), "uniforms": list(lay["uniforms"]),
                "outputs": list(lay["outputs"]), "diagnostics": list(lay["diagnostics"]), "dt": F.DT, "runtime_inputs": list(lay["runtime_inputs"]),
                "gravity": F.G, "dof": F.DOF},
        "emit_span": F.EMIT_SPAN, "initial": {},
        "points": {"x": "x", "y": "y", "vx": "vx", "vy": "vy", "rgb": ["cr", "cg", "cb"], "live": ["age"], "placed": ["x", "y"], "radius": "rad", "rest_speed": 0.15},
        "reference_view": [-F.MOB_ROAD_HALF, F.MOB_ROAD_HALF, 0.0, F.MOB_ROAD_TOP], "species": [list(item) for item in F.SPECIES], "settle": F.SETTLE,
        "mobroad": F.mobroad_map(),
        "presets": {"sdf_mobroad": {tag: F.mobroad_spec(tag) for tag in records}},
    }


def build_mobroad(web_dir: Path, sizes: tuple[str, ...] = MOBROAD_SIZES, wasm: bool = True, on_gpu_ready=None) -> dict:
    """The mob road's page products, beside the default ones: ``fountain_mobroad.manifest.json`` and ``fountain_mobroad_<tag>.wgsl`` / ``.wasm``.  Two passes
    like ``build`` (WebGPU products first, the CPU fallbacks after); sizes whose source digest is unchanged are kept."""
    web_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = web_dir / "fountain_mobroad.manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = {}
    for tag in sizes:
        previous = (old.get("sizes") or {}).get(tag)
        if previous and previous.get("source_digest") == mobroad_digest(tag) and (web_dir / f"fountain_mobroad_{tag}.wgsl").exists():
            if previous.get("wasm") and not all((web_dir / previous[k]).exists() for k in ("wasm", "wasm_abi")):
                previous = {k: v for k, v in previous.items() if not k.startswith("wasm")}
            records[tag] = previous
            continue
        records[tag] = build_size_mobroad(tag, web_dir, wasm=False)
        r = records[tag]
        print(f"[fountain mobroad] {tag}: {r['n']} ids -> {r['route']} ({r['wgsl_abi']['dispatches']} dispatches, "
              f"{r['memory']['bytes_per_id']:.0f} B/id), built in {r['build_seconds']}s", flush=True)
    manifest = mobroad_manifest(records)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if on_gpu_ready is not None:
        on_gpu_ready(manifest)
    if wasm:
        for tag in sizes:
            r = records[tag]
            if SIZES[tag][1] > WASM_MAX or r.get("wasm"):
                continue
            build_fallback_mobroad(tag, web_dir, r)
            manifest_path.write_text(json.dumps(mobroad_manifest(records)), encoding="utf-8")
    return manifest
# </mobroad>
if __name__ == "__main__":
    raise SystemExit(main())
