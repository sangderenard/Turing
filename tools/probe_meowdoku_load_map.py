"""Map where native thinks `load` is nonzero for one cat (args: batch, cat cells, 'identity' for identity geometry).

Report only; edits nothing.  Run from the compressor-validation checkout with
PYTHONPATH set to it and the system Python: the saved SSA pickles in %TEMP% were
produced there and unpickle only against its compiler.  The game itself is read
from this repository.  See HANDOFF_MEOWDOKU_WASM_GAME_2026-11-04.md.
"""
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 25
N = int(round(BATCH ** 0.5))
PKL = Path(os.environ["TEMP"]) / f"meowdoku_validation_{BATCH}_module.pkl"
CANON = Path(r"C:\dev\Powershell\turing")
sys.path.insert(0, str(CANON / "examples"))

from src.compiler.ssa_llvm_backend import (  # noqa: E402
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)
from src.common.tensors import AbstractTensor  # noqa: E402
from meowdoku.program import PLACE_CAT, MeowdokuPiece  # noqa: E402

AbstractTensor._preferred_backend = "numpy"

with PKL.open("rb") as handle:
    loaded = pickle.load(handle)
module = loaded[0] if isinstance(loaded, tuple) else loaded
entry = next(n for n in module.functions if n.endswith("__meowdoku_step"))
function = module.functions[entry]
id_of = {str(name): int(vid) for name, vid in function.metadata["parameter_names"]}
outputs = {str(name): int(vid) for name, vid in function.metadata["named_outputs"]}
artifact = compile_artifact(emit_ssa_function_to_llvm(module, entry))
piece = MeowdokuPiece(BATCH)

cells = np.arange(BATCH)
row = (cells // N).astype(float)
col = (cells % N).astype(float)
region = np.zeros(BATCH)          # one region: every cell conflicts by region with every other
region_distinct = cells.astype(float)   # every cell its own region: region never conflicts


def hearts_after(cat_cell, target, region_column):
    """hearts_next[0] from both lanes with solution all-ones, hearts 3, one cat, one placement."""
    cat = np.zeros(BATCH); cat[cat_cell] = 1.0
    intent = np.zeros(BATCH); intent[target] = PLACE_CAT
    columns = {"row": row, "col": col, "region": region_column, "solution": np.ones(BATCH),
               "cat": cat, "x_mark": np.zeros(BATCH), "hearts": np.full(BATCH, 3.0), "intent": intent}
    py = dict(zip(piece.output_names, piece(*[columns[n] if n != "dt" else np.ones(BATCH)
                                               for n in piece.argument_names])))
    execution = prepare_artifact_execution(artifact, {vid: np.asarray(columns[name], dtype=np.float64)
                                                      for name, vid in id_of.items()})
    execution.run()
    native = np.asarray(execution.buffers[outputs["hearts_next"]], dtype=np.float64)
    return float(py["hearts_next"][0]), float(native[0])


GEOM = sys.argv[3] if len(sys.argv) > 3 else "board"
if GEOM == "identity":
    row = 2.0 * cells; col = 2.0 * cells
    print("[map] geometry: identity (row=col=2*cell, region=cell): conflict is the identity, load == cat)", flush=True)
for label, region_column in (("distinct regions", region_distinct),):
    for cat_cell in ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else (0, (N // 2) * N + N // 2)):
        py_grid = np.zeros((N, N), dtype=int)
        nat_grid = np.zeros((N, N), dtype=int)
        for target in range(BATCH):
            if target == cat_cell:
                py_grid.flat[target] = nat_grid.flat[target] = 9
                continue
            py_h, nat_h = hearts_after(cat_cell, target, region_column)
            py_grid.flat[target] = int(py_h < 3.0)       # 1 = placement refused (load > 0)
            nat_grid.flat[target] = int(nat_h < 3.0)
        print(f"[map] {label}; cat at cell {cat_cell}. 1 = load>0 (placement costs a heart), 9 = the cat")
        print("[map] python:\n" + "\n".join("  " + " ".join(str(v) for v in r) for r in py_grid))
        print("[map] native:\n" + "\n".join("  " + " ".join(str(v) for v in r) for r in nat_grid))
        print(f"[map] agree={bool(np.array_equal(py_grid, nat_grid))}", flush=True)
