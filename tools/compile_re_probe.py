"""Standalone script: compile ``re._compile`` through the real, non-deprecated
compiler entry point (``lower_ast_source_to_ssa``) and print what happened.

No pytest, no test harness -- just run it directly:

    python tools/compile_re_probe.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_fortran_backend import emit_module

CONTRACT = ROOT / "extraction_contracts" / "program_extraction.yaml"

SOURCE = (
    "def entry(pattern, flags):\n"
    "    return compile_re(pattern, flags)\n"
)


def _progress(t0: float):
    def report(message: str) -> None:
        print(f"  [+{time.time() - t0:.1f}s] {message}", flush=True)
    return report


def main() -> int:
    print(f"contract: {CONTRACT}")
    print(f"source:\n{SOURCE}")
    t0 = time.time()
    module, outputs, exports = lower_ast_source_to_ssa(
        SOURCE,
        "entry",
        python_bindings={"compile_re": re._compile},
        name="re_compile_probe",
        extraction_contract=CONTRACT,
        progress=_progress(t0),
    )
    print(f"lower_ast_source_to_ssa: {time.time() - t0:.2f}s")
    print(f"functions compiled: {len(module.functions)}")

    compile_name = "re_compile_probe___compile"
    if compile_name not in module.functions:
        print(f"MISSING: {compile_name!r} was not compiled.")
        print("compiled function names:")
        for name in sorted(module.functions):
            print(f"  {name}")
        return 1

    compile_function = module.functions[compile_name]
    operations = [
        instruction.attributes.get("ssa_sequence_operation")
        for block in compile_function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation")
    ]
    print(f"{compile_name} sequence operations: {operations}")

    t1 = time.time()
    emitted = emit_module(
        module, name="re_compile_probe", outputs=outputs, extra_roots=exports,
        progress=_progress(t1),
    )
    print(f"emit_module: {time.time() - t1:.2f}s")
    print(f"emitted.complete: {emitted.complete}")
    if not emitted.complete:
        for item in emitted.shortfalls:
            print(f"  shortfall: {item.format()}")
    return 0 if emitted.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
