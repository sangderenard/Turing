"""Lower an OUTER static `for outer in range(3)` driver around a dt-controlled
substep `while` loop (run_superstep's skeleton, dt_controller.py:677: `while
(round_max - total) > eps: dt_try = min(dt_cap, remainder) ...; total += dt_used;
dt_cap = clamp(next proposal)`).  Each substep builds A, b from dt_try and
calls `torch.linalg.solve`; x is carried.  The accepted-step proposal is a
halving clamped to dt_min (the controller's dt_min floor).  Run on the C or
LLVM lane and compare with NumPy running the same loops.

usage: probe_solve_in_dt_loop.py {c|llvm}

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

N_STEPS = 4
A0 = np.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
B0 = np.asarray([1.0, 2.0], dtype=np.float64)
DIAG = np.eye(2, dtype=np.float64)


WINDOW = 1.0
DT_INIT = 0.5
DT_MIN = 0.1
EPS = 1e-15


def numpy_loop():
    x = np.zeros_like(B0)
    dt_cap = DT_INIT
    substeps = 0
    for outer in range(3):
        total = 0.0
        while WINDOW - total > EPS:
            remainder = WINDOW - total
            dt_try = min(dt_cap, remainder)
            A = A0 + DIAG * (dt_try + 0.25 * outer)
            b = B0 + dt_try
            x = x + np.linalg.solve(A, b) * dt_try
            total = total + dt_try
            dt_cap = max(dt_try * 0.5, DT_MIN)
            substeps += 1
    print("NUMPY_SUBSTEPS", substeps)
    return x


SENTINEL = -12345.0

lane = sys.argv[1]
tag = f"abstract_solve_dt_loop_{lane}"

SOURCE = """
import torch

def solve_dt_loop(a0: torch.Tensor, b0: torch.Tensor, diag: torch.Tensor):
    x = b0 * 0.0
    dt_cap = 0.5
    for outer in range(3):
        total = 0.0
        while 1.0 - total > 1e-15:
            remainder = 1.0 - total
            dt_try = min(dt_cap, remainder)
            A = a0 + diag * (dt_try + 0.25 * outer)
            b = b0 + dt_try
            x = x + torch.linalg.solve(A, b) * dt_try
            total = total + dt_try
            dt_cap = max(dt_try * 0.5, 0.1)
    return x
"""

contract = ExtractionContract(
    root / "extraction_contracts" / "program_extraction.yaml"
).with_program_abi({
    "records": {},
    "bindings": [],
    "values": [
        {
            "function": "solve_dt_loop", "parameter": name,
            "storage": "span", "dtype": "float64", "rank": value.ndim,
            "shape": list(value.shape), "python_type": "AbstractTensor",
        }
        for name, value in (("a0", A0), ("b0", B0), ("diag", DIAG))
    ],
})

t0 = time.time()
module, outputs, _exports = lower_ast_source_to_ssa(
    SOURCE, "solve_dt_loop", name=tag, extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda message: print("PROGRESS", message, flush=True),
)
print(f"LOWERED in {time.time() - t0:.1f}s", flush=True)
qualified = f"{tag}__solve_dt_loop"
function = module.functions[qualified]
expected = numpy_loop()
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

feeds = {parameters["a0"]: A0.copy(), parameters["b0"]: B0.copy(),
         parameters["diag"]: DIAG.copy()}
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
