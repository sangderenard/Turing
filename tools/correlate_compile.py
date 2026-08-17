"""Correlate one compiled value across every layer of the pipeline.

Run one fresh compile of the symbolic-fluid advance function and, for a
given set of value ids (in the advance function's own top-level numbering),
print what each layer says about it side by side:

  - repository SSA: the producing/consuming instruction in the advance
    function itself, and in any planned region that touches it
  - LLVM IR text: every line mentioning that id's register name
  - runtime: the actual value read out of the execution buffer after a run

This exists because manually grepping IR text and cross-referencing pickled
SSA by hand, one id at a time, does not scale -- this makes one fresh
compile answer "what does every layer believe about id N" in one shot.

Usage:
    python tools/correlate_compile.py 116 47 117 187 188 192 193 206 207 208
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState
from src.compiler.symbolic_fluid_native_runtime import (
    compile_native_symbolic_fluid_advance,
)

ADVANCE_NAME = "symbolic_fluid_control__symbolic_fluid_advance"


def _instr_repr(instr: Any) -> str:
    args = [int(a.id) for a in instr.args]
    res = int(instr.res.id) if instr.res is not None else None
    attrs = {
        k: v for k, v in instr.attributes.items()
        if k not in {"region"}
    }
    return f"{res} = {instr.op}{args}  {attrs}" if res is not None else f"{instr.op}{args}  {attrs}"


def find_instructions(fn: Any, value_id: int) -> list[tuple[str, str]]:
    """(block_name, instr repr) for every instruction that produces OR
    consumes value_id in this function."""
    hits = []
    for block_name, block in fn.blocks.items():
        for instr in block.instrs:
            res_hit = instr.res is not None and int(instr.res.id) == value_id
            arg_hit = any(int(a.id) == value_id for a in instr.args)
            if res_hit or arg_hit:
                marker = "PRODUCES" if res_hit else "consumes"
                hits.append((block_name, f"[{marker}] {_instr_repr(instr)}"))
    return hits


def find_llvm_lines(llvm_ir: str, value_id: int) -> list[str]:
    needles = (f"%value.{value_id} ", f"%value.{value_id}\n", f"%value.{value_id},",
               f"%value.{value_id})", f"%phi.{value_id} ", f"%phi.{value_id},",
               f"%phi.{value_id})", f"%out.{value_id} ")
    lines = []
    for line in llvm_ir.splitlines():
        stripped = line.strip()
        for needle in needles:
            key = needle.rstrip(" ,)\n")
            if key and (
                stripped.startswith(key) or f" {key} " in f" {stripped} "
                or f" {key}," in f" {stripped} " or f" {key})" in f" {stripped} "
            ):
                lines.append(line.rstrip())
                break
    # de-dup, preserve order
    seen = set()
    out = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def main() -> int:
    value_ids = [int(a) for a in sys.argv[1:]]
    if not value_ids:
        print(__doc__)
        return 1

    tmp_dir = ROOT / "build" / "correlate-tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"compiling into {tmp_dir} ...")
    adv = compile_native_symbolic_fluid_advance(tmp_dir)
    fn = adv.function
    module_functions = adv._module_functions or {}
    llvm_ir = adv.artifact.llvm_ir

    state = SymbolicFluidGridState.initial(4, 4)
    before = float(np.asarray(state.height, dtype=float).sum())
    ok, metrics = adv(state, 0.2)
    after = float(np.asarray(state.height, dtype=float).sum())
    print(f"ran: mass before={before:.10f} after={after:.10f} ok={ok}")
    print(f"metrics: max_vel={metrics.max_vel} mass_err={metrics.mass_err} "
          f"dt_limit={metrics.dt_limit}")
    print("=" * 100)

    for vid in value_ids:
        print(f"\n### id={vid} " + "#" * 80)

        print("-- repository SSA (advance function) --")
        hits = find_instructions(fn, vid)
        if not hits:
            print("  (not found in advance's own body)")
        for block_name, text in hits:
            print(f"  [{block_name}] {text}")

        print("-- repository SSA (planned regions) --")
        for region_name, region_fn in module_functions.items():
            if "planned_region" not in region_name:
                continue
            hits = find_instructions(region_fn, vid)
            if hits:
                short_name = region_name.split("__")[-1]
                for block_name, text in hits:
                    print(f"  [{short_name}/{block_name}] {text}")

        print("-- LLVM IR lines --")
        lines = find_llvm_lines(llvm_ir, vid)
        if not lines:
            print("  (no matching lines)")
        for line in lines[:12]:
            print(f"  {line}")
        if len(lines) > 12:
            print(f"  ... ({len(lines) - 12} more)")

        print("-- runtime value --")
        if adv.observable(vid):
            print(f"  {adv._read(vid)}")
        else:
            print(
                "  NOT OBSERVABLE -- this id is not in the artifact's public "
                "buffer ABI (internal alloca). It has no readable value; do "
                "NOT treat this as zero."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
