"""Traced graph forwards as page WebAssembly products (buffer-table ABI).

Two products beside the engine laws of ``entity_motion.py``, both built with
``compile_graph_forward_to_wasm`` (the forward traced through the SSA tensor
backend, re-emitted for ``wasm32-freestanding``, libm calls as host ``math``
imports) and both loaded in the page by ``TuringGraphProgram`` in
``turing-wasm-shell.js``:

    law.wasm   + law.abi.json                        the bullet-hell law ``scroller_step``
                                                     (traced as ``LawPiece._compile_native``
                                                     does), work contract ``develop``
    pilot.wasm + pilot.abi.json + pilot.weights.json the token pilot network at batch 1
                                                     (traced as ``NativePolicy`` does), the
                                                     weights being INPUTS

The ABI JSON is written from the ``BrowserGraphForward`` fields (order,
shapes, dtypes, extents) plus the traced named inputs and outputs; nothing is
hand-written.  A product is rebuilt only when the digest of its sources
(the law/forward text, the shapes, the contract, the emitter) changes.

    python examples/meowdoku/graph_wasm.py parity   # node loader vs native DLL / Python forward
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _path in (str(ROOT), str(ROOT / "examples")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

SCHEMA = "turing.graph-buffer-abi.v1"
CONTRACT = "develop"
TRACE_HOME = ROOT / "build" / "meowdoku" / "graph_wasm"
SHELL = ROOT / "examples" / "mobile_reference_app" / "web" / "turing-wasm-shell.js"


def _emitter_sources() -> str:
    from src.compiler import llvm_training_runtime
    return inspect.getsource(llvm_training_runtime)


def _digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def _cached(web_dir: Path, stem: str, digest: str) -> dict | None:
    abi_path, wasm_path = web_dir / f"{stem}.abi.json", web_dir / f"{stem}.wasm"
    if abi_path.exists() and wasm_path.exists():
        abi = json.loads(abi_path.read_text(encoding="utf-8"))
        if abi.get("source_digest") == digest:
            return abi
    return None


def _trace(name: str, shapes: dict, run, contract: str):
    """Trace ``run(bindings)`` through the SSA tensor backend under the work
    contract; returns (outputs, bindings)."""
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    with scoped_work_contract(PRESETS[contract]):
        program = SSATensorProgram(name)
        bindings = {key: SSATensorOperations.input(program, shape) for key, shape in shapes.items()}
        with program.activate():
            produced = run(bindings)
        return produced, bindings


def _compile(stem: str, shapes: dict, run, output_names, contract: str, web_dir: Path, extra: dict) -> dict:
    """Trace, compile to wasm, and write ``<stem>.wasm`` + ``<stem>.abi.json``."""
    from src.compiler.llvm_training_runtime import compile_graph_forward_to_wasm
    produced, bindings = _trace(stem, shapes, run, contract)
    outputs = tuple(produced) if isinstance(produced, (tuple, list)) else (produced,)
    if len(outputs) != len(output_names):
        raise RuntimeError(f"{stem}: traced {len(outputs)} outputs, declared {len(output_names)}")
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    with scoped_work_contract(PRESETS[contract]):
        forward = compile_graph_forward_to_wasm(produced, name=stem, directory=TRACE_HOME / stem)
    # Value ids of the traced graph, exactly as compile_native_graph_forward reads them.
    inputs = {str(n): int(getattr(getattr(v, "data", v), "value").id) for n, v in bindings.items()}
    output_ids = [int(getattr(item, "data", item).value.id) for item in outputs]
    fields = dataclasses.asdict(forward)
    wasm_path = Path(fields.pop("wasm_path"))
    abi = {
        "schema": SCHEMA,
        "wasm": f"{stem}.wasm",
        "entry": fields.pop("name"),
        "order": list(fields["buffer_order"]),
        "shapes": [list(s) for s in fields["buffer_shapes"]],
        "dtypes": list(fields["buffer_dtypes"]),
        "extents": [list(e) for e in fields["extent_order"]],
        "inputs": inputs,
        "outputs": [{"name": n, "id": i} for n, i in zip(output_names, output_ids)],
        "contract": contract,
        **extra,
    }
    web_dir.mkdir(parents=True, exist_ok=True)
    (web_dir / f"{stem}.wasm").write_bytes(wasm_path.read_bytes())
    (web_dir / f"{stem}.abi.json").write_text(json.dumps(abi), encoding="utf-8")
    return abi


# ----------------------------------------------------------------------- law

def law_shapes() -> dict:
    from meowdoku.s2_titles.scroller import scroller as sc
    columns = sc.initial_columns()
    piece = sc.ScrollerPiece
    shapes = {n: (tuple(np.shape(columns[n])) if n in columns else ()) for n in piece.argument_names}
    # the law's PARAMETER inputs (the pilot network's weights) are named inputs too
    shapes.update({n: tuple(shape) for n, shape in piece.parameter_shapes.items()})
    return shapes


def build_law(web_dir: Path, contract: str = CONTRACT) -> dict:
    """``law.wasm`` + ``law.abi.json``: ``scroller_step``, traced as
    ``LawPiece._compile_native`` traces it."""
    from meowdoku.s2_titles.scroller import scroller as sc
    piece = sc.ScrollerPiece
    shapes = law_shapes()
    digest = _digest(piece._captured_sources or piece._read_sources(),
                     ",".join((*piece.argument_names, *piece.parameter_shapes)), json.dumps(shapes, sort_keys=True),
                     contract, _emitter_sources())
    abi = _cached(web_dir, "law", digest)
    if abi is not None:
        return abi
    started = time.perf_counter()
    names = (*piece.argument_names, *piece.parameter_shapes)
    abi = _compile("law", shapes, lambda b: piece.law(*(b[n] for n in names)), piece.output_names,
                   contract, web_dir, {"source_digest": digest, "argument_names": list(piece.argument_names),
                                       "parameter_names": list(piece.parameter_shapes)})
    abi["build_seconds"] = round(time.perf_counter() - started, 1)
    (web_dir / "law.abi.json").write_text(json.dumps(abi), encoding="utf-8")
    return abi


# --------------------------------------------------------------------- pilot

def pilot_params() -> dict:
    from meowdoku.s3_train import autopilot
    params = autopilot.live_params("bullet_hell")
    if params is None or autopilot.arch_of(params) != "token":
        raise RuntimeError("no token pilot weights for bullet_hell")
    return params


def write_pilot_weights(web_dir: Path, params: dict) -> Path:
    from meowdoku.s3_train import autopilot
    payload = {n: {"shape": list(np.asarray(params[n]).shape),
                   "data": np.asarray(params[n], dtype=np.float64).reshape(-1).tolist()}
               for n in autopilot.param_names(params)}
    path = Path(web_dir) / "pilot.weights.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def build_pilot(web_dir: Path, params: dict | None = None, contract: str = CONTRACT) -> dict:
    """``pilot.wasm`` + ``pilot.abi.json`` (+ ``pilot.weights.json``): the token
    forward at batch 1, traced as ``NativePolicy`` traces it; the weights are
    inputs, so new weights rewrite only the weights file."""
    from meowdoku.s3_train import autopilot
    params = params if params is not None else pilot_params()
    names, forward = autopilot.param_names(params), autopilot.forward_of(params)
    features, outputs = autopilot.widths_of(params)
    shapes = {"x": (1, features), **{n: tuple(np.asarray(params[n]).shape) for n in names}}
    constants = "".join(f"{n}={int(getattr(autopilot, n))};" for n in ("TOKENS", "TOKEN_F", "SHIP_G", "TOKEN_H"))
    digest = _digest(inspect.getsource(forward), constants, json.dumps(shapes, sort_keys=True), contract, _emitter_sources())
    write_pilot_weights(web_dir, params)
    abi = _cached(web_dir, "pilot", digest)
    if abi is not None:
        return abi
    started = time.perf_counter()
    abi = _compile("pilot", shapes, lambda b: forward(b["x"], *(b[n] for n in names)), ("action",),
                   contract, web_dir, {"source_digest": digest, "features": features, "outputs_width": outputs,
                                       "weights": list(names)})
    abi["build_seconds"] = round(time.perf_counter() - started, 1)
    (web_dir / "pilot.abi.json").write_text(json.dumps(abi), encoding="utf-8")
    return abi


# -------------------------------------------------------------------- parity

NODE_PARITY = r"""
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const [dir, shell, job] = process.argv.slice(2);
const {TuringGraphProgram} = await import(pathToFileURL(shell).href);
const spec = JSON.parse(fs.readFileSync(`${dir}/${job}.job.json`, 'utf8'));
const abi = JSON.parse(fs.readFileSync(`${dir}/${spec.stem}.abi.json`, 'utf8'));
const program = await TuringGraphProgram.fromBytes(fs.readFileSync(`${dir}/${spec.stem}.wasm`), abi);
const raw = fs.readFileSync(`${dir}/${job}.in.bin`);
const all = new Float64Array(raw.buffer, raw.byteOffset, raw.length / 8);
let at = 0;
for (const [name, size] of spec.inputs) { program.setInput(name, all.subarray(at, at + size)); at += size; }
const t0 = performance.now(); program.run(); const first = performance.now() - t0;
const N = spec.timing; const t1 = performance.now();
for (let i = 0; i < N; i++) program.run();
const per = (performance.now() - t1) / N;
const out = {};
for (const o of abi.outputs) out[o.name] = Array.from(program.output(o.name));
fs.writeFileSync(`${dir}/${job}.out.json`, JSON.stringify({out, memory: program.memory.buffer.byteLength, used: program.used, first, per}));
"""


def _node_run(work: Path, stem: str, job: str, inputs: list, timing: int = 200) -> dict:
    import subprocess
    work.mkdir(parents=True, exist_ok=True)
    (work / "node_parity.mjs").write_text(NODE_PARITY, encoding="utf-8")
    flat = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for _n, a in inputs])
    flat.tofile(work / f"{job}.in.bin")
    (work / f"{job}.job.json").write_text(json.dumps(
        {"stem": stem, "timing": timing, "inputs": [[n, int(np.asarray(a).size)] for n, a in inputs]}))
    done = subprocess.run(["node", str(work / "node_parity.mjs"), str(work), str(SHELL), job],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-2000:])
    return json.loads((work / f"{job}.out.json").read_text())


def parity_law(work: Path) -> dict:
    """The page loader's ``law.wasm`` against the native DLL (``develop``) on
    a real state: a bullet-hell game flown a few seconds with fire held."""
    from meowdoku.s2_titles.scroller import scroller as sc
    build_law(work)
    game = sc.BulletScroller(level=1)
    for _ in range(40):
        game._land(0.1, {"move_axis": 0.4, "move_axis_y": 0.2, "fire": 1.0})
    piece = game.session.piece
    names = piece.argument_names
    # the network flies the player in this step: marked, with the live weights as parameter inputs
    game.session.column("net_pilot")[sc.PILOT_ROWS[0]] = 1.0
    weights = game.parameters()
    columns = {n: np.array(game.session.column(n), dtype=np.float64) for n in names if n != "dt"}
    columns["dt"] = np.asarray(float(game.session.dt))
    piece._compile_native({n: columns[n] for n in columns}, CONTRACT)
    piece.set_parameters(weights)
    native = piece.product(*(columns[n] for n in names))
    got = _node_run(work, "law", "law", [*((n, columns[n]) for n in names),
                                          *((n, weights[n]) for n in piece.parameter_names)])
    report = {"lanes": sc.LANES, "alive": int(columns["alive"].sum()), "outputs": {}}
    worst = 0.0
    for name, ref in zip(piece.output_names, native):
        diff = float(np.max(np.abs(np.asarray(got["out"][name]).reshape(-1)[:np.size(ref)] - np.asarray(ref).reshape(-1)))) if np.size(ref) else 0.0
        report["outputs"][name] = diff
        worst = max(worst, diff)
    report.update(max_abs_diff=worst, memory_bytes=got["memory"], used_bytes=got["used"],
                  first_ms=got["first"], ms_per_step=got["per"])
    return report


def parity_pilot(work: Path, samples: int = 5) -> dict:
    """The page loader's ``pilot.wasm`` against Python ``forward_of`` (eager)."""
    from meowdoku.s3_train import autopilot
    params = pilot_params()
    abi = build_pilot(work, params)
    rng = np.random.default_rng(0)
    worst, per = 0.0, 0.0
    for k in range(samples):
        x = rng.normal(0.0, 1.0, (1, abi["features"]))
        got = _node_run(work, "pilot", f"pilot{k}", [("x", x), *((n, params[n]) for n in abi["weights"])], timing=2000)
        ref = autopilot.eager_forward(params, x)
        worst = max(worst, float(np.max(np.abs(np.asarray(got["out"]["action"]).reshape(ref.shape) - ref))))
        per = got["per"]
    return {"max_abs_diff": worst, "samples": samples, "memory_bytes": got["memory"], "used_bytes": got["used"],
            "ms_per_forward": per}


def main(argv=None) -> int:
    work = Path(argv[1]) if argv and len(argv) > 1 else TRACE_HOME / "parity"
    print("law  :", json.dumps(parity_law(work), indent=1))
    print("pilot:", json.dumps(parity_pilot(work), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
