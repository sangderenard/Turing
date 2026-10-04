"""Lower `torch.linalg.solve` and run it on the C lane or the LLVM lane.

usage: probe_solve_c_lane.py {c|llvm} {2x2|3x3tie}

Same entry and contract as probe_solve_numeric.py and
tests/test_native_shaped_view_lowering.py: `lower_ast_source_to_ssa` with the
program_extraction contract plus a declared ABI, `-O0`, and the executed
result compared with NumPy.  Output buffers are poisoned with SENTINEL (not
NaN) so a NaN written by the program and a buffer never written are told
apart.  Prints shortfalls, per-buffer sentinel classification, max abs error.
"""

from pathlib import Path
import sys
import time

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

SYSTEMS = {
    "2x2": (
        np.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64),
        np.asarray([1.0, 2.0], dtype=np.float64),
    ),
    # |pivot| tie in column 0 (rows 0 and 1 both 1.0).
    "3x3tie": (
        np.asarray([[1.0, 2.0, 3.0], [1.0, 0.0, 1.0], [0.0, 1.0, 4.0]],
                   dtype=np.float64),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
    ),
}
SENTINEL = -12345.0

lane, system = sys.argv[1], sys.argv[2]
MATRIX, RHS = SYSTEMS[system]
tag = f"abstract_solve_{lane}_{system}"

SOURCE = """
import torch

def solve_sys(matrix: torch.Tensor, rhs: torch.Tensor):
    return torch.linalg.solve(matrix, rhs)
"""

contract = ExtractionContract(
    root / "extraction_contracts" / "program_extraction.yaml"
).with_program_abi({
    "records": {},
    "bindings": [],
    "values": [
        {
            "function": "solve_sys", "parameter": "matrix",
            "storage": "span", "dtype": "float64", "rank": 2,
            "shape": list(MATRIX.shape), "python_type": "AbstractTensor",
        },
        {
            "function": "solve_sys", "parameter": "rhs",
            "storage": "span", "dtype": "float64", "rank": 1,
            "shape": list(RHS.shape), "python_type": "AbstractTensor",
        },
    ],
})

t0 = time.time()
module, outputs, _exports = lower_ast_source_to_ssa(
    SOURCE, "solve_sys", name=tag, extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda message: print("PROGRESS", message, flush=True),
)
print(f"LOWERED in {time.time() - t0:.1f}s", flush=True)
qualified = f"{tag}__solve_sys"
function = module.functions[qualified]
expected = np.linalg.solve(MATRIX, RHS)
parameters = dict(function.metadata["parameter_names"])
published = [int(value.id) for value in outputs[qualified]]

t0 = time.time()
if lane == "c":
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    artifact = emit_ssa_module_to_c(module, qualified)
    print("C_COMPLETE", artifact.complete, flush=True)
    print("C_SHORTFALLS", len(artifact.shortfalls), flush=True)
    for _s in artifact.shortfalls:
        print("  C_SHORTFALL", _s.operation, str(_s.reason)[:600], flush=True)
    if not artifact.complete:
        raise SystemExit(1)
    artifact.compile(root / "build" / tag, optimization="O0")
    prepare = artifact.prepare_execution
else:
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm,
        prepare_artifact_execution,
    )
    artifact = emit_ssa_function_to_llvm(module, qualified, entry_name=tag)
    print("LLVM_SHORTFALLS", len(artifact.shortfalls), flush=True)
    for _s in artifact.shortfalls:
        print("  LLVM_SHORTFALL", _s, flush=True)
    if artifact.shortfalls:
        raise SystemExit(1)
    native = compile_artifact(artifact, directory=root / "build" / tag,
                              optimization="O0")
    prepare = lambda feeds: prepare_artifact_execution(native, feeds)
print(f"BUILT in {time.time() - t0:.1f}s", flush=True)

feeds = {parameters["matrix"]: MATRIX.copy(), parameters["rhs"]: RHS.copy()}
for value_id in published:
    feeds[value_id] = np.full(np.shape(expected), SENTINEL)
execution = prepare(feeds).run()

untouched, wrote_nan, wrote_real = [], [], []
for bid, buf in execution.buffers.items():
    a = np.asarray(buf)
    if not a.size:
        continue
    if np.all(a == SENTINEL):
        untouched.append(int(bid))
    elif np.any(np.isnan(a)):
        wrote_nan.append(int(bid))
    else:
        wrote_real.append(int(bid))
print(f"SENTINEL untouched={len(untouched)} wrote-NaN={len(wrote_nan)} "
      f"wrote-real={len(wrote_real)}", flush=True)
produced = [np.asarray(execution.buffers[v]).reshape(np.shape(expected))
            for v in published if v in execution.buffers]
print("PUBLISHED", published, "in buffers:",
      [v in execution.buffers for v in published], flush=True)
print("EXPECTED", expected, flush=True)
print("PRODUCED", produced, flush=True)
for value in produced:
    print("PUBLISHED untouched:", bool(np.all(value == SENTINEL)),
          "has NaN:", bool(np.any(np.isnan(value))), flush=True)
    err = float(np.max(np.abs(value - expected)))
    print(f"MAX_ABS_ERR {err:.3e}", flush=True)
    print("NUMERIC_MATCH" if err < 1e-9 else "NUMERIC_MISMATCH", flush=True)
