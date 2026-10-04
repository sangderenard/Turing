"""A loop-carried tensor update that is lost natively.

usage: probe_loop_update_lost.py VARIANT      (VARIANT: a|b|c|d|e)

Each variant is `_forward_substitute` (linalg.py) shrunk; b = [[1],[2]] (2,1),
lu = [[4,1],[0.5,2.5]] (2,2).  Output buffer poisoned with SENTINEL -12345.0.
Compared against the same source run eagerly.  Seconds long; no solve.
"""
from pathlib import Path
import os
import sys

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)

SENTINEL = -12345.0
LU = np.asarray([[4.0, 1.0], [0.5, 2.5]])
B = np.asarray([[1.0], [2.0]])
PRE = """
from src.common.tensors.linalg import _axis_index, _row, _one_hot_axis, _forward_substitute
"""
BODIES = {
    # the real thing
    "a": "    return _forward_substitute(lu, b)\n",
    # no helper call: the same body inline
    "b": """
    dtype = lu.get_dtype()
    n = lu.get_shape()[-1]
    index = _axis_index(lu, n)
    y = b.clone()
    for i in range(n):
        hot_i = _one_hot_axis(lu, n, i)
        prefix = (index < i).to_dtype(dtype)
        coefficients = _row(lu, hot_i) * prefix
        product = (coefficients.unsqueeze(-1) * y).sum(dim=-2)
        solved = _row(y, hot_i) - product
        y = (y * (1 - hot_i.unsqueeze(-1)) + hot_i.unsqueeze(-1) * solved.unsqueeze(-2))
    return y
""",
    # plain accumulate, nothing but y = y + c
    "c": """
    y = b.clone()
    for i in range(2):
        y = y + 1
    return y
""",
    # y = b.clone() replaced by y = b (no clone)
    "d": """
    n = lu.get_shape()[-1]
    index = _axis_index(lu, n)
    y = b * 1
    for i in range(n):
        hot_i = _one_hot_axis(lu, n, i)
        y = y + hot_i.unsqueeze(-1) * 10
    return y
""",
    "e": """
    n = lu.get_shape()[-1]
    y = b.clone()
    for i in range(n):
        y = y + 10
    return y
""",
}
BODIES["f"] = """
    y = lu.clone()
    for i in range(2):
        y = y + 1
    return y
"""
BODIES["g"] = """
    y = b + 0
    for i in range(2):
        y = y + 1
    return y
"""
BODIES["h"] = """
    y = b.clone()
    y = y + 1
    y = y + 1
    return y
"""
variant = sys.argv[1]
source = PRE + "\ndef stage(lu, b):\n" + BODIES[variant]
values = [{"function": "stage", "parameter": p, "storage": "span",
           "dtype": "float64", "rank": 2, "shape": list(a.shape),
           "python_type": "AbstractTensor"} for p, a in (("lu", LU), ("b", B))]
contract = ExtractionContract(
    root / "extraction_contracts" / "program_extraction.yaml"
).with_program_abi({"records": {}, "bindings": [], "values": values})
module, outputs, _ = lower_ast_source_to_ssa(
    source, "stage", name=f"lul_{variant}", extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda _m: None,
)
qualified = f"lul_{variant}__stage"
artifact = emit_ssa_function_to_llvm(module, qualified, entry_name=f"lul_{variant}")
print("LLVM_SHORTFALLS", len(artifact.shortfalls), flush=True)
if artifact.shortfalls:
    print(artifact.shortfalls); raise SystemExit(1)
(root / "build").mkdir(exist_ok=True)
if os.environ.get("PROBE_DUMP_IR"):
    (root / "build" / f"lul_{variant}.ll").write_text(artifact.llvm_ir)
native = compile_artifact(artifact, directory=root / "build" / f"lul_{variant}",
                          optimization="O0")
function = module.functions[qualified]
parameters = dict(function.metadata["parameter_names"])
published = [int(v.id) for v in outputs[qualified]]
if os.environ.get("PROBE_SSA"):
    for bl in function.blocks.values():
        for ins in bl.instrs:
            print("  ", ins.op, [a.id for a in ins.args], "->", getattr(ins.res, "id", None), dict(getattr(ins, "attrs", {}) or {}) if os.environ.get("PROBE_SSA") == "2" else "")
    print("OUTPUTS", published)
feeds = {parameters[k]: a.copy() for k, a in (("lu", LU), ("b", B)) if k in parameters}
for v in published:
    feeds[v] = np.full((2, 2) if variant == "f" else (2, 1), SENTINEL)
execution = prepare_artifact_execution(native, feeds)
for buffer_id, buffer in execution.buffers.items():
    if buffer_id in feeds:
        continue
    try:
        buffer[...] = SENTINEL
    except Exception:
        pass
execution.run()
produced = np.asarray(execution.buffers[published[0]]).reshape(-1)
from src.common.tensors.abstraction import AbstractTensor
ns = {}
exec(compile(source, "<eager>", "exec"), ns)
eager = np.asarray(ns["stage"](AbstractTensor.tensor(LU.tolist()),
                               AbstractTensor.tensor(B.tolist())).tolist(),
                   dtype=float).reshape(-1)
print("EAGER", eager, "NATIVE", produced)
print("MAX_ABS_ERR", float(np.max(np.abs(produced - eager))))
print("MATCH" if np.allclose(produced, eager) else "DIFFERS")
