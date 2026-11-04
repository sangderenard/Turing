"""Run the Codex lane's saved pairwise slice natively: sum |r1 - r2| must be 1000.0 for a 5x5 board.

Report only; edits nothing.  Run from the compressor-validation checkout with
PYTHONPATH set to it and the system Python: the saved SSA pickles in %TEMP% were
produced there and unpickle only against its compiler.  The game itself is read
from this repository.  See HANDOFF_MEOWDOKU_WASM_GAME_2026-11-04.md.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

from src.compiler.ssa_llvm_backend import (  # noqa: E402
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)

PKL = Path(r"C:\Users\alber\.codex\worktrees\compressor-validation\turing\build\meowdoku_pairwise25_ssa.pkl")
with PKL.open("rb") as handle:
    module, outputs, exports = pickle.load(handle)
entry = exports[0]
function = module.functions[entry]
print(f"[slice] entry={entry} functions={len(module.functions)}", flush=True)
print(f"[slice] parameter_names={function.metadata.get('parameter_names')}", flush=True)
print(f"[slice] named_outputs={function.metadata.get('named_outputs')} outputs={[int(v.id) for v in outputs[entry]]}", flush=True)
src = function.metadata.get("source_function_reference") or ""
print(f"[slice] source ref: {str(src)[:200]}", flush=True)

artifact = emit_ssa_function_to_llvm(module, entry)
print(f"[slice] LLVM complete={artifact.complete} shortfalls={artifact.shortfalls[:3]}", flush=True)
if not artifact.complete:
    sys.exit(1)
artifact = compile_artifact(artifact)
print(f"[slice] buffer_order={artifact.buffer_order}", flush=True)
names = {str(name): int(vid) for name, vid in function.metadata["parameter_names"]}
N = 5
cells = np.arange(N * N)
feeds = {}
for name, vid in names.items():
    if name == "row":
        feeds[vid] = (cells // N).astype(np.float64)
    elif name == "col":
        feeds[vid] = (cells % N).astype(np.float64)
    else:
        feeds[vid] = np.zeros(N * N)
        print(f"[slice] input {name!r} fed zeros", flush=True)
execution = prepare_artifact_execution(artifact, feeds)
execution.run()
result_ids = [int(v.id) for v in outputs[entry]]
for vid in result_ids:
    print(f"[slice] native output %{vid} = {np.asarray(execution.buffers[vid])}", flush=True)
row = (cells // N).astype(float)
print(f"[slice] python sum |r1-r2| = {np.abs(row[:, None] - row[None, :]).sum()}", flush=True)
print(f"[slice] if the two views collapsed to one orientation the sum would be 0.0", flush=True)
