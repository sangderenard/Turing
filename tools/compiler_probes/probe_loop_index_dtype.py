"""A loop index passed to a callee that compares it with a float tensor.

usage: probe_loop_index_dtype.py [ssa|run]

`for k in range(n): total = total + pick(t, k) * (k + 1)` with
`pick(t, k) = (t == k).cast_like(t)` and t = [0, 1, 2] (float64).
Expected total = [1, 2, 3].  `ssa` prints, for every function, the dtype of
each formal and of every call actual (no clang); `run` builds natively with
LLVM and compares with the eager value.  Contract/entry style as
probe_solve_c_lane.py.
"""
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

mode = sys.argv[1] if len(sys.argv) > 1 else "ssa"
T = np.asarray([0.0, 1.0, 2.0])
SOURCE = """
from src.common.tensors.abstraction import AbstractTensor

def pick(t, k):
    return (t == k).cast_like(t)

def stage(t, count):
    total = t * 0
    for k in range(count):
        total = total + pick(t, k) * (k + 1)
    return total
"""
contract = ExtractionContract(
    root / "extraction_contracts" / "program_extraction.yaml"
).with_program_abi({"records": {}, "bindings": [], "values": [{
    "function": "stage", "parameter": "t", "storage": "span",
    "dtype": "float64", "rank": 1, "shape": [3],
    "python_type": "AbstractTensor"},
    {"function": "stage", "parameter": "count", "storage": "scalar",
     "dtype": "int64", "rank": 0, "python_type": "int"}]})
module, outputs, _ = lower_ast_source_to_ssa(
    SOURCE, "stage", name="lid", extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda _m: None,
)
qualified = "lid__stage"
if mode == "ssa":
    for fname, function in module.functions.items():
        short = fname.split("__", 1)[-1]
        if "pick" not in short and not fname.endswith("__stage"):
            continue
        print("FUNCTION", fname)
        print("  parameter_names", dict(function.metadata.get("parameter_names") or {}))
        for block in function.blocks.values():
            for ins in block.instrs:
                if ins.op in {"Call", "RegionCall"} or "cmp" in str(ins.op).lower() \
                        or "Eq" in str(ins.op):
                    print("  ", ins.op, [(a.id, str(getattr(a, "dtype", None)))
                                          for a in ins.args],
                          "->", getattr(ins.res, "dtype", None))
    raise SystemExit(0)

from src.common.tensors.abstraction import AbstractTensor
from src.compiler.ssa_llvm_backend import (
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)
import os
artifact = emit_ssa_function_to_llvm(module, qualified, entry_name="lid_stage")
print("LLVM_SHORTFALLS", len(artifact.shortfalls), flush=True)
if artifact.shortfalls:
    print(artifact.shortfalls); raise SystemExit(1)
if os.environ.get("PROBE_DUMP_IR"):
    (root / "build").mkdir(exist_ok=True)
    (root / "build" / "lid.ll").write_text(artifact.llvm_ir)
native = compile_artifact(artifact, directory=root / "build" / "lid", optimization="O0")
function = module.functions[qualified]
parameters = dict(function.metadata["parameter_names"])
published = [int(v.id) for v in outputs[qualified]]
feeds = {parameters["t"]: T.copy(), parameters["count"]: np.asarray([3], dtype=np.int64)}
for v in published:
    feeds[v] = np.full((3,), -12345.0)
execution = prepare_artifact_execution(native, feeds).run()
produced = np.asarray(execution.buffers[published[0]]).reshape(3)
expected = np.asarray([1.0, 2.0, 3.0])
print("EXPECTED", expected, "PRODUCED", produced)
print("MAX_ABS_ERR", float(np.max(np.abs(produced - expected))))
print("NUMERIC_MATCH" if np.allclose(produced, expected) else "NUMERIC_MISMATCH")
