"""Split lowering: JavaScript holds control flow and memory, WebAssembly holds
the numeric sections.

Nothing here is a new compiler.  ``lower_ast_source_to_ssa`` already cuts a
program into its host function (loops, Phi, loads, calls) and its planned
regions (straight-line arithmetic functions).  This tool only decides where
each planned region runs and wires the two existing backends together:

* a region goes to WebAssembly when ``emit_ssa_module_to_wasm_core`` emits it
  COMPLETE (the wasm lane itself is the judge of what it can express exactly
  -- e.g. an unbounded Python ``int`` add refuses, so it stays in JS) and it
  carries at least ``min_ops`` arithmetic instructions (below that the
  JS->wasm call costs more than the arithmetic it moves);
* everything else -- the host and the small or refused regions -- is printed
  by ``emit_ssa_module_to_javascript``, which sends the chosen regions through
  its ``external_functions`` / ``runtime.call`` seam;
* each wasm region gets a ``turing.mobile-wasm-abi.v1`` contract, so the page
  binds ``runtime.call`` with the existing ``TuringWasmProgram`` loader
  (``turing-wasm-shell.js``) -- see ``web/split-runtime.mjs``.

Output (``out_dir``): ``host.mjs``, ``regions/<name>.wasm`` +
``regions/<name>.abi.json``, and ``split.json`` (the plan: which region went
where and why, plus the entry's argument names in ``bufferOrder``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Instructions a numeric section is made of.  A region is a candidate only if
# it is one block of these; the wasm lane then decides if it can emit it.
_ARITHMETIC = frozenset({"Add", "Sub", "Mul", "Div", "Abs", "Neg", "Min", "Max", "Sqrt",
                         "Lt", "Le", "Gt", "Ge", "Eq", "Ne", "Select", "Cast", "FMA"})
_NUMERIC = _ARITHMETIC | {"Const", "Ret"}


def _region_plan(module, min_ops: int):
    from src.compiler.ssa_wasm_backend import emit_ssa_module_to_wasm_core
    plan, wasm = [], {}
    for name, fn in module.functions.items():
        ops = [i.op for b in fn.blocks.values() for i in b.instrs]
        if len(fn.blocks) != 1 or not all(op in _NUMERIC for op in ops):
            continue
        arithmetic = sum(op in _ARITHMETIC for op in ops)
        artifact = emit_ssa_module_to_wasm_core(module, name, entry_name=name)
        if not artifact.complete:
            where, why = "javascript", "wasm lane refused: " + "; ".join(
                f"{s.operation}: {s.reason}" for s in artifact.shortfalls[:2])
        elif arithmetic < min_ops:
            where, why = "javascript", f"{arithmetic} arithmetic op(s) < min_ops={min_ops}: the call would cost more"
        else:
            where, why = "wasm", f"{arithmetic} arithmetic ops, wasm emission complete"
            wasm[name] = artifact
        plan.append({"function": name, "target": where, "reason": why, "arithmetic_ops": arithmetic})
    return plan, wasm


def _region_contract(name: str, artifact) -> dict:
    dtypes = dict(artifact.output_dtypes)
    parameters, offset = [], 8
    for kind, value_id in artifact.parameters:
        entry = {"kind": kind, "value_id": int(value_id), "name": f"value_{value_id}",
                 "role": "output" if value_id in dtypes else "input"}
        if kind == "buffer":
            entry["offset"], entry["dtype"] = offset, dtypes.get(value_id, "float64")
            offset += 8
        parameters.append(entry)
    return {"schema": "turing.mobile-wasm-abi.v1", "entrypoint": name, "memory_bytes": offset,
            "parameters": parameters}


def build_split(source: str, entry: str, out_dir: Path, *, name: str = "split",
                min_ops: int = 8) -> dict:
    """Lower ``entry`` of ``source`` and write the JS host + wasm regions."""
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_javascript_backend import emit_ssa_module_to_javascript

    out_dir = Path(out_dir)
    digest = hashlib.sha256(f"{min_ops}\n{source}".encode("utf-8")).hexdigest()[:16]
    split_path = out_dir / "split.json"
    if split_path.exists():
        cached = json.loads(split_path.read_text(encoding="utf-8"))
        if cached.get("source_digest") == digest:
            return cached
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, entry, name=name,
        extraction_contract=ExtractionContract(ROOT / "extraction_contracts/program_extraction.yaml"),
        progress_bars=False)
    host_entry = exports[0]
    plan, wasm = _region_plan(module, min_ops)
    js = emit_ssa_module_to_javascript(module, host_entry, external_functions=tuple(wasm))
    if not js.complete:
        raise RuntimeError(f"JavaScript host emission shortfalls: {js.shortfalls}")
    (out_dir / "regions").mkdir(parents=True, exist_ok=True)
    (out_dir / "host.mjs").write_text(js.source, encoding="utf-8")
    # The same program with every region in JavaScript: the baseline the
    # split is measured against (and a fallback with no wasm at all).
    pure = emit_ssa_module_to_javascript(module, host_entry)
    if pure.complete:
        (out_dir / "host.pure.mjs").write_text(pure.source, encoding="utf-8")
    for region, artifact in wasm.items():
        (out_dir / "regions" / f"{region}.wasm").write_bytes(artifact.binary)
        (out_dir / "regions" / f"{region}.abi.json").write_text(
            json.dumps(_region_contract(region, artifact), indent=1), encoding="utf-8")
    # The entry's formals by source name, in the host's bufferOrder.
    # Names come from the function's own declared metadata (the JS backend's
    # ``_source_names``): formal order is the compiler's, not the source's.
    from src.compiler.ssa_javascript_backend import _source_names
    names = _source_names(module.functions[host_entry])
    abi = json.loads(js.source.split("export const PROGRAM_ABI = Object.freeze(", 1)[1].split(");", 1)[0])
    split = {"schema": "turing.split-js-wasm.v1", "entry": host_entry, "source_digest": digest,
             "min_ops": min_ops, "inputs": [names.get(int(i), f"value_{i}") for i in abi["bufferOrder"]],
             "wasm_regions": sorted(wasm), "plan": plan}
    split_path.write_text(json.dumps(split, indent=1), encoding="utf-8")
    return split


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description="JS control + wasm numeric split lowering")
    parser.add_argument("source"); parser.add_argument("entry"); parser.add_argument("out")
    parser.add_argument("--min-ops", type=int, default=8)
    args = parser.parse_args()
    result = build_split(Path(args.source).read_text(encoding="utf-8"), args.entry, Path(args.out),
                         name=Path(args.source).stem, min_ops=args.min_ops)
    print("inputs:", result["inputs"])
    for row in result["plan"]:
        print(f"{row['target']:10s} {row['function']}  -- {row['reason']}")
