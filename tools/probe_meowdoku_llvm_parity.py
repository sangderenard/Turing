"""Emit, compile and play the saved Meowdoku SSA through the LLVM lane against the Python piece; prints the first diverging move.

Report only; edits nothing.  Run from the compressor-validation checkout with
PYTHONPATH set to it and the system Python: the saved SSA pickles in %TEMP% were
produced there and unpickle only against its compiler.  The game itself is read
from this repository.  See HANDOFF_MEOWDOKU_WASM_GAME_2026-11-04.md.
"""
import faulthandler
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

faulthandler.dump_traceback_later(540, exit=False)
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 25
N = int(round(BATCH ** 0.5))
PKL = Path(os.environ["TEMP"]) / f"meowdoku_validation_{BATCH}_module.pkl"
CANON = Path(r"C:\dev\Powershell\turing")
sys.path.insert(0, str(CANON / "examples"))  # the game, from canonical (unchanged there)

from src.compiler.ssa_llvm_backend import (  # noqa: E402
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)
from meowdoku.program import (  # noqa: E402
    PLACE_CAT, TOGGLE_X, MeowdokuPiece, initial_columns, intent_column,
)
from meowdoku.puzzle import generate  # noqa: E402
from src.common.tensors import AbstractTensor  # noqa: E402

for backend in ("numpy", "pure_python"):
    AbstractTensor._preferred_backend = backend
    try:
        AbstractTensor.get_tensor(np.zeros(2))
        print(f"[run] tensor backend: {backend}", flush=True)
        break
    except Exception as error:
        print(f"[run] backend {backend} failed: {type(error).__name__}: {error}", flush=True)
        traceback.print_exc(limit=3)

with PKL.open("rb") as handle:
    loaded = pickle.load(handle)
module = loaded[0] if isinstance(loaded, tuple) else loaded
entry = next(n for n in module.functions if n.endswith("__meowdoku_step"))
function = module.functions[entry]
print(f"[run] metadata keys: {sorted(function.metadata)}", flush=True)
print(f"[run] args: {[(int(v.id), getattr(v, 'name', None), getattr(v, 'dtype', None), getattr(v, 'shape', None)) for v in function.args]}", flush=True)
names = function.metadata.get("parameter_names")
print(f"[run] parameter_names: {names}", flush=True)

artifact = emit_ssa_function_to_llvm(module, entry)
assert artifact.complete, artifact.shortfalls
print(f"[run] buffer_order: {artifact.buffer_order}", flush=True)
t = time.perf_counter()
artifact = compile_artifact(artifact)
print(f"[run] compiled in {time.perf_counter() - t:.1f}s: {type(artifact).__name__}", flush=True)

# --- map input value ids to column names
piece = MeowdokuPiece(BATCH)
outputs = dict((name, int(vid)) for name, vid in function.metadata["named_outputs"])
input_ids = [vid for vid in artifact.buffer_order if vid not in outputs.values()]
if names:
    by_id = {int(vid): str(name) for name, vid in (names.items() if isinstance(names, dict) else names)}
    id_of = {by_id[vid]: vid for vid in input_ids if vid in by_id}
else:
    id_of = {}
print(f"[run] input ids -> names: {id_of}", flush=True)
missing = [n for n in piece.argument_names if n not in id_of and n != "dt"]
if missing or len(id_of) != len(input_ids):
    print(f"[run] STOP: cannot name inputs; unnamed={[v for v in input_ids if v not in id_of.values()]} missing={missing}", flush=True)
    sys.exit(2)

# --- play the same scripted game through both
puzzle = generate(N, seed=1)
cols = puzzle.solution_columns()
py = initial_columns(puzzle)
nat = {k: v.copy() for k, v in py.items()}
dt = np.ones(BATCH)
moves = [(0, cols[0], PLACE_CAT), (2, 2, TOGGLE_X), (2, 2, TOGGLE_X)]
moves += [((r, c, PLACE_CAT)) for r in range(N) for c in range(N) if cols[r] != c and r > 1][:1]
moves += [(1, cols[0] + 1, PLACE_CAT), (1, cols[1], TOGGLE_X), (1, cols[1], PLACE_CAT)]
moves += [(r, cols[r], PLACE_CAT) for r in range(2, N)]
moves += [(0, 0, PLACE_CAT)]
worst = 0.0
t = time.perf_counter()
for k, (r, c, action) in enumerate(moves):
    intent = intent_column(BATCH, r * N + c, action)
    py["intent"][...] = intent
    nat["intent"][...] = intent
    py_out = dict(zip(piece.output_names, piece(*[py[n] if n != "dt" else dt for n in piece.argument_names])))
    execution = prepare_artifact_execution(artifact, {
        vid: np.asarray(nat[name], dtype=np.float64) for name, vid in id_of.items()
    } | ({id_of["dt"]: dt} if "dt" in id_of else {}))
    execution.run()
    nat_out = {name: np.asarray(execution.buffers[vid], dtype=np.float64) for name, vid in outputs.items()}
    for name in piece.output_names:
        a, b = py_out[name], nat_out[name]
        diff = float(np.max(np.abs(np.broadcast_to(a, b.shape) - b))) if b.size else float("nan")
        worst = max(worst, diff)
        if diff != 0.0:
            print(f"[run] move {k} {name}: py={a[:6]} native={b[:6]} maxdiff={diff}", flush=True)
    for name in ("cat", "x_mark", "hearts"):
        py[name][...] = py_out[name + "_next"]
        nat[name][...] = np.broadcast_to(nat_out[name + "_next"], (BATCH,))
print(f"[run] {len(moves)} moves in {time.perf_counter() - t:.2f}s, worst |py-native| = {worst}", flush=True)
print(f"[run] final: hearts={py['hearts'][0]} residual={py_out['residual'][0]} native hearts={nat['hearts'][0]} "
      f"solved={bool(np.array_equal(nat['cat'], puzzle.solution))}", flush=True)
