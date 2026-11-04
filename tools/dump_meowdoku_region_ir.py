"""Count bare rank-2 ops in the saved SSA and dump planned_region_8's emitted LLVM IR to %TEMP%.

Report only; edits nothing.  Run from the compressor-validation checkout with
PYTHONPATH set to it and the system Python: the saved SSA pickles in %TEMP% were
produced there and unpickle only against its compiler.  The game itself is read
from this repository.  See HANDOFF_MEOWDOKU_WASM_GAME_2026-11-04.md.
"""
import collections
import os
import pickle
import re
import sys
from pathlib import Path

from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm  # noqa: E402

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 25
PKL = Path(os.environ["TEMP"]) / f"meowdoku_validation_{BATCH}_module.pkl"
with PKL.open("rb") as handle:
    loaded = pickle.load(handle)
module = loaded[0] if isinstance(loaded, tuple) else loaded
entry = next(n for n in module.functions if n.endswith("__meowdoku_step"))

# 1. every op in every function whose operands are rank>=2 and that is NOT a linked Call
bare = collections.Counter()
for fname, function in module.functions.items():
    for block in function.blocks.values():
        for instr in block.instrs:
            op = str(instr.op)
            if op in ("Call", "call", "Const", "Ret", "ret", "GetElementPtr", "Load", "Store", "Cast", "string_token"):
                continue
            shapes = [getattr(a, "shape", None) for a in instr.args]
            if any(s and len(s) >= 2 for s in shapes):
                bare[(fname.split("__")[-1], op, tuple(shapes))] += 1
print("[ir] bare ops on rank>=2 operands:")
for key, count in bare.items():
    print(f"[ir]   {key} x{count}")

# 2. the IR text of region 8
artifact = emit_ssa_function_to_llvm(module, entry)
text = None
for attr in ("llvm_ir", "ir", "text", "module_text", "source"):
    value = getattr(artifact, attr, None)
    if isinstance(value, str) and "define" in value:
        text = value
        break
if text is None:
    print(f"[ir] artifact attrs: {[a for a in dir(artifact) if not a.startswith('_')]}")
    sys.exit(1)
out = Path(os.environ["TEMP"]) / "meowdoku_region8.ll"
match = re.search(r"define[^\n]*planned_region_8[^\n]*\{.*?\n\}", text, re.S)
if not match:
    print("[ir] region 8 not found in IR text")
    sys.exit(1)
body = match.group(0)
out.write_text(body, encoding="utf-8")
lines = body.splitlines()
print(f"[ir] region 8 IR: {len(lines)} lines, saved to {out}")
for i, line in enumerate(lines):
    if re.search(r"fcmp|icmp|@binary_double|@broadcast_double|@unary_double|@binary_scalar_double|call ", line):
        print(f"[ir] {i:4d}: {line.strip()[:170]}")
