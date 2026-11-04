"""Emit the saved Meowdoku SSA through the module-level C lane. Report only.

Run from the validation checkout with PYTHONPATH set to it (the pickle only
unpickles against that compiler):

    PYTHONPATH=$PWD python C:/dev/Powershell/turing/tools/probe_meowdoku_c_module.py 25

Legacy ``emit_ssa_function_to_c`` is the scalar ``(const double *in, double
*out)`` adapter and refuses GetElementPtr/Load/Store by design; this probe
uses ``emit_ssa_module_to_c``, the canonical module-preserving route.
"""
import os
import pickle
import sys
import time
from pathlib import Path

from src.compiler.ssa_c_backend import emit_ssa_module_to_c  # noqa: E402

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 25
PKL = Path(os.environ["TEMP"]) / f"meowdoku_validation_{BATCH}_module.pkl"
with PKL.open("rb") as handle:
    loaded = pickle.load(handle)
module = loaded[0] if isinstance(loaded, tuple) else loaded
entry = next(n for n in module.functions if n.endswith("__meowdoku_step"))

started = time.time()
artifact = emit_ssa_module_to_c(module, entry, batch=BATCH)
print(f"[c-module] complete={artifact.complete} shortfalls={len(artifact.shortfalls)} "
      f"in {time.time() - started:.1f}s; buffers={len(artifact.buffer_order)} "
      f"roles={artifact.buffer_roles}", flush=True)
for item in artifact.shortfalls:
    print(f"[c-module]   {item.operation}: {item.reason}", flush=True)
if artifact.complete:
    print(f"[c-module] source: {len(artifact.source.splitlines())} lines (not compiled here)")
