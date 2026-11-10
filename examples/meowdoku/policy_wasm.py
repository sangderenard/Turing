"""The autopilot policy network as a WebAssembly core for the page.

The same route as the engine laws of ``entity_motion.py``: the policy's own
source text (``autopilot.policy_forward`` / ``token_policy_forward``) is
lowered by ``lower_ast_source_to_ssa`` -- every operand a declared rank-2
float64 span, the weights among them -- and emitted by
``emit_ssa_module_to_wasm_core``.  Nothing is rewritten.  The weights are
INPUTS of the module (not baked in), so the page fetches ``<stem>.weights.json``
and the live weights can be replaced without a rebuild.

    python examples/meowdoku/policy_wasm.py --game space_invaders --parity 20

builds into ``build/meowdoku/policy_wasm`` and reports the max abs difference
of the emitted module against the Python forward (``autopilot.eager_forward``).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

STEMS = {"mlp": "policy", "token": "policy_token"}


def _source(forward) -> str:
    from meowdoku import autopilot
    # The module constants the token forward reads are literals in the text.
    constants = "".join(f"{name} = {int(getattr(autopilot, name))}\n"
                        for name in ("TOKENS", "TOKEN_F", "SHIP_G", "TOKEN_H"))
    return constants + inspect.getsource(forward)


def build_policy(out_dir: Path, params: dict, *, batch: int = 1, force: bool = False) -> dict:
    """Lower and emit the forward of ``params``' architecture into
    ``out_dir`` (``<stem>.wasm``, ``<stem>.abi.json``, ``<stem>.weights.json``);
    rebuilt only when the source or the weight shapes change.  Returns the
    contract."""
    from meowdoku import autopilot
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.native_law_kernels import _CONTRACTS
    from src.compiler.ssa_wasm_backend import emit_ssa_module_to_wasm_core
    from src.compiler.wasm_core_product import core_contract

    arch = autopilot.arch_of(params)
    forward, names = autopilot.forward_of(arch), autopilot.param_names(arch)
    features, outputs_width = autopilot.widths_of(params)
    shapes = {"x": (int(batch), features), **{n: tuple(np.asarray(params[n]).shape) for n in names}}
    stem = STEMS[arch]
    out_dir = Path(out_dir)
    source = _source(forward)
    digest = hashlib.sha256((source + json.dumps(shapes, sort_keys=True)).encode("utf-8")).hexdigest()[:16]
    abi_path, wasm_path = out_dir / f"{stem}.abi.json", out_dir / f"{stem}.wasm"
    contract = None
    if not force and abi_path.exists() and wasm_path.exists():
        contract = json.loads(abi_path.read_text(encoding="utf-8"))
        if contract.get("source_digest") != digest:
            contract = None
    if contract is None:
        values = [{"function": forward.__name__, "parameter": name, "storage": "span", "dtype": "float64",
                   "rank": len(shape), "shape": list(shape),
                   "python_type": "src.common.tensors.abstraction.AbstractTensor"}
                  for name, shape in shapes.items()]
        extraction = ExtractionContract(_CONTRACTS / "program_extraction.yaml").with_program_abi(
            {"records": {}, "bindings": [], "values": values}
        ).with_execution_file(_CONTRACTS / "vehicle_full_native_execution.yaml")
        module, outputs, exports = lower_ast_source_to_ssa(
            source, forward.__name__, python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(), name=stem,
            runtime_closure_only=True, extraction_contract=extraction, progress_bars=False)
        entry = exports[0]
        wasm = emit_ssa_module_to_wasm_core(module, entry, entry_name=entry)
        if not wasm.complete:
            raise RuntimeError(f"{forward.__name__} WebAssembly emission shortfalls: {wasm.shortfalls}")
        function = module.functions[entry]
        input_names = {int(v.id): n for v, n in zip(function.args, shapes)}
        output_names = {int(v.id): "action" for v in outputs[entry]}
        contract = core_contract(wasm, module, entry, input_names=input_names, output_names=output_names)
        contract.update({"schema": "turing.mobile-wasm-abi.v1", "source_digest": digest, "arch": arch,
                         "features": features, "outputs_width": outputs_width, "batch": int(batch),
                         "weights": list(names)})
        out_dir.mkdir(parents=True, exist_ok=True)
        wasm_path.write_bytes(wasm.binary)
        abi_path.write_text(json.dumps(contract, indent=1), encoding="utf-8")
        wasm.api.write(out_dir / f"{stem}.api.yaml")
        build_policy.artifacts[stem] = (wasm, module, entry)
    write_weights(out_dir, params)
    return contract


build_policy.artifacts = {}


def write_weights(out_dir: Path, params: dict) -> Path:
    """``<stem>.weights.json``: each weight as {shape, data} -- the live
    weights, which the page feeds as the module's inputs."""
    from meowdoku import autopilot
    arch = autopilot.arch_of(params)
    payload = {n: {"shape": list(np.asarray(params[n]).shape),
                   "data": np.asarray(params[n], dtype=np.float64).reshape(-1).tolist()}
               for n in autopilot.param_names(arch)}
    path = Path(out_dir) / f"{STEMS[arch]}.weights.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def parity(out_dir: Path, params: dict, samples: int = 20, seed: int = 0) -> float:
    """Max abs difference between the emitted module (run by the node
    harness of ``ssa_wasm_backend``) and ``autopilot.eager_forward``."""
    from meowdoku import autopilot
    from src.compiler.ssa_wasm_backend import prepare_wasm_core_execution
    contract = build_policy(out_dir, params, force=True)     # the artifact itself is run, so it is rebuilt
    stem = STEMS[contract["arch"]]
    wasm, _module, _entry = build_policy.artifacts[stem]
    by_name = {p["name"]: p["value_id"] for p in contract["parameters"]}
    rng = np.random.default_rng(seed)
    features = contract["features"]
    worst = 0.0
    for _ in range(samples):
        x = rng.normal(0.0, 1.0, (1, features))
        feeds = {by_name["x"]: x.reshape(-1)}
        feeds.update({by_name[n]: np.asarray(params[n], dtype=np.float64).reshape(-1) for n in contract["weights"]})
        execution = prepare_wasm_core_execution(wasm, feeds)
        try:
            execution.run()
            got = np.asarray(execution.buffers[by_name["action"]], dtype=np.float64).reshape(1, -1)
        finally:
            execution.close()
        worst = max(worst, float(np.max(np.abs(got - autopilot.eager_forward(params, x)))))
    return worst


def main(argv=None) -> int:
    from meowdoku import autopilot
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--game", default="space_invaders", choices=autopilot.GAMES)
    parser.add_argument("--out", default=str(ROOT / "build" / "meowdoku" / "policy_wasm"))
    parser.add_argument("--parity", type=int, default=20)
    parser.add_argument("--random-weights", action="store_true", help="use fresh random weights, not the live ones")
    parser.add_argument("--random-biases", action="store_true", help="also draw the biases at random (they init to zero)")
    args = parser.parse_args(argv)
    params = None if args.random_weights else autopilot.live_params(args.game)
    if params is None:
        params = autopilot.init_token_params() if autopilot.GAME_ARCH[args.game] == "token" \
            else {k: v for k, v in autopilot.init_params(seed=1).items()}
    if args.random_biases:
        rng = np.random.default_rng(7)
        params = {k: (rng.normal(0.0, 0.5, np.shape(v)) if k.startswith("b") else np.asarray(v)) for k, v in params.items()}
    worst = parity(Path(args.out), params, args.parity)
    print(f"[policy_wasm] {args.game} ({autopilot.arch_of(params)}): max abs diff over {args.parity} random features = {worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
