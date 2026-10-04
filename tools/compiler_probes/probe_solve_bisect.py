"""Bisect the compiled `AT.linalg.solve` against eager, one stage per run.

usage: probe_solve_bisect.py {2x2|3x3tie} STAGE [STAGE ...]

Each STAGE compiles the same authored helpers `solve` uses (imported from
src.common.tensors.linalg), returns ONE intermediate as the program's output,
runs it natively and compares with the same source run eagerly through
AbstractTensor.  Output buffers are poisoned with SENTINEL, not NaN.
Contract and entry are those of probe_solve_c_lane.py.
"""


from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)

SYSTEMS = {
    "2x2": (np.asarray([[4.0, 1.0], [2.0, 3.0]]), np.asarray([1.0, 2.0])),
    "3x3tie": (np.asarray([[1.0, 2.0, 3.0], [1.0, 0.0, 1.0], [0.0, 1.0, 4.0]]),
               np.asarray([1.0, 2.0, 3.0])),
}
MATRIX, RHS = SYSTEMS[sys.argv[1]]
N = MATRIX.shape[0]
SENTINEL = -12345.0
root = Path(__file__).resolve().parents[2]


PRE = """
from src.common.tensors.linalg import (
    _axis_index, _column, _row, _one_hot_axis, _pivot_mask,
    _masked_pivot_rows, _lu_decompose_inplace, _forward_substitute,
    _back_substitute,
)
"""

LU_STEP0 = """
    n = matrix.get_shape()[-1]
    U = matrix.clone()
    index = _axis_index(U, n)
    permutation = (index.unsqueeze(-1) == index.unsqueeze(-2)).to_dtype(U.get_dtype())
    pivot = _pivot_mask(U, 0)
    U, parity = _masked_pivot_rows(U, 0, pivot)
    permutation, _ = _masked_pivot_rows(permutation, 0, pivot)
    hot_k = _one_hot_axis(U, n, 0)
    after_k = (index > 0).to_dtype(U.get_dtype())
    pivot_row = _row(U, hot_k)
    pivot_value = (pivot_row * hot_k).sum(dim=-1)
    factor = (_column(U, hot_k) / pivot_value.unsqueeze(-1)) * after_k
"""

UPD0 = """
    U = U - (factor.unsqueeze(-1) * pivot_row.unsqueeze(-2) * after_k.unsqueeze(-2))
    stored = after_k.unsqueeze(-1) * hot_k.unsqueeze(-2)
    U = U * (1 - stored) + factor.unsqueeze(-1) * stored
"""
STEP1 = """
    pivot = _pivot_mask(U, 1)
    U, parity = _masked_pivot_rows(U, 1, pivot)
    permutation, _ = _masked_pivot_rows(permutation, 1, pivot)
    hot_k = _one_hot_axis(U, n, 1)
    after_k = (index > 1).to_dtype(U.get_dtype())
    pivot_row = _row(U, hot_k)
    pivot_value = (pivot_row * hot_k).sum(dim=-1)
    factor = (_column(U, hot_k) / pivot_value.unsqueeze(-1)) * after_k
    U = U - (factor.unsqueeze(-1) * pivot_row.unsqueeze(-2) * after_k.unsqueeze(-2))
    stored = after_k.unsqueeze(-1) * hot_k.unsqueeze(-2)
    U = U * (1 - stored) + factor.unsqueeze(-1) * stored
"""
BODIES_EXTRA = {
    "u_after_update0": ("matrix", LU_STEP0 + UPD0 + "    return U\n", (N, N)),
    "u_unrolled_k1": ("matrix", LU_STEP0 + UPD0 + STEP1 + "    return U\n", (N, N)),
    "factor_k1": ("matrix", LU_STEP0 + UPD0 + STEP1.split("    U = U - (")[0] + "    return factor\n", (N,)),
}
LOOP = """
    n = matrix.get_shape()[-1]
    U = matrix.clone()
    index = _axis_index(U, n)
    permutation = (index.unsqueeze(-1) == index.unsqueeze(-2)).to_dtype(U.get_dtype())
    sign = (U * 0 + 1).sum(dim=-1).sum(dim=-1) * 0 + 1
    for k in range(n):
        pivot = _pivot_mask(U, k)
        U, parity = _masked_pivot_rows(U, k, pivot)
        permutation, _ = _masked_pivot_rows(permutation, k, pivot)
        sign = sign * parity
        hot_k = _one_hot_axis(U, n, k)
        after_k = (index > k).to_dtype(U.get_dtype())
        pivot_row = _row(U, hot_k)
        pivot_value = (pivot_row * hot_k).sum(dim=-1)
        factor = (_column(U, hot_k) / pivot_value.unsqueeze(-1)) * after_k
        U = U - (factor.unsqueeze(-1) * pivot_row.unsqueeze(-2) * after_k.unsqueeze(-2))
        stored = after_k.unsqueeze(-1) * hot_k.unsqueeze(-2)
        U = U * (1 - stored) + factor.unsqueeze(-1) * stored
"""
LOOPX = {
    "loop_pivot": ("pivot", (N,)), "loop_hot": ("hot_k", (N,)),
    "loop_pivot_row": ("pivot_row", (N,)), "loop_pivot_value": ("pivot_value", ()),
    "loop_parity": ("parity", ()), "loop_factor": ("factor", (N,)),
    "loop_after": ("after_k", (N,)),
}
BODIES_EXTRA.update({
    name: ("matrix", LOOP + "    return " + var + "\n", shape)
    for name, (var, shape) in LOOPX.items()
})
LOOPK = """
    n = matrix.get_shape()[-1]
    U = matrix.clone()
    index = _axis_index(U, n)
    for k in range(n):
        kv = index * 0 + k
        dk = index - k
        eq = (index == k).to_dtype(U.get_dtype())
        ge = (index >= k).to_dtype(U.get_dtype())
"""
BODIES_EXTRA.update({
    "loopk_val": ("matrix", LOOPK + "    return kv" + "\n", (N,)),
    "loopk_diff": ("matrix", LOOPK + "    return dk" + "\n", (N,)),
    "loopk_eq": ("matrix", LOOPK + "    return eq" + "\n", (N,)),
    "loopk_ge": ("matrix", LOOPK + "    return ge" + "\n", (N,)),
})
LU = "    upper, sign, permutation = _lu_decompose_inplace(matrix)\n"
PB = "    LU, sign, permutation = _lu_decompose_inplace(matrix)\n"
BODIES = {
    "pivot_mask": ("matrix", "    return _pivot_mask(matrix, 0)\n", (N,)),
    "masked_rows_U": ("matrix", "    rows, parity = _masked_pivot_rows(matrix, 0, _pivot_mask(matrix, 0))\n    return rows\n", (N, N)),
    "perm_after_pivot0": ("matrix", LU_STEP0 + "    return permutation\n", (N, N)),
    "pivot_value0": ("matrix", LU_STEP0 + "    return pivot_value\n", ()),
    "factor0": ("matrix", LU_STEP0 + "    return factor\n", (N,)),
    "lu_U": ("matrix", LU + "    return upper\n", (N, N)),
    "lu_P": ("matrix", LU + "    return permutation\n", (N, N)),
    "lu_sign": ("matrix", LU + "    return sign\n", ()),
    "pb": ("matrix,rhs", LU + "    return permutation.matmul(rhs.unsqueeze(-1))\n", (N, 1)),
    "y_forward": ("matrix,rhs", PB + "    return _forward_substitute(LU, permutation.matmul(rhs.unsqueeze(-1)))\n", (N, 1)),
    "x_back": ("matrix,rhs", PB + "    y = _forward_substitute(LU, permutation.matmul(rhs.unsqueeze(-1)))\n    return _back_substitute(LU, y)\n", (N, 1)),
}
BODIES.update(BODIES_EXTRA)
STAGES = tuple(
    (name, PRE + "\ndef stage(" + params + "):\n" + body, tuple(params.split(",")), shape)
    for name, (params, body, shape) in BODIES.items()
)
STAGES = tuple(s for s in STAGES if s[0] in sys.argv[2:])


def run_stage(name, source, parameter_names, out_shape):
    import os
    values = []
    for parameter in parameter_names:
        array = MATRIX if parameter == "matrix" else RHS
        values.append({
            "function": "stage", "parameter": parameter, "storage": "span",
            "dtype": "float64", "rank": len(array.shape),
            "shape": list(array.shape), "python_type": "AbstractTensor",
        })
    contract = ExtractionContract(
        root / "extraction_contracts" / "program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": values})

    module, outputs, _exports = lower_ast_source_to_ssa(
        source, "stage", name=f"sb_{name}",
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _m: None,
    )
    qualified = f"sb_{name}__stage"
    function = module.functions[qualified]
    artifact = emit_ssa_function_to_llvm(
        module, qualified, entry_name=f"sb_{name}_stage",
    )
    if artifact.shortfalls:
        print(f"{name}: SHORTFALLS {artifact.shortfalls}", flush=True)
        return
    if os.environ.get('PROBE_DUMP_IR'):
        open(root / 'build' / f'sb_{name}.ll', 'w').write(artifact.llvm_ir)
    native = compile_artifact(
        artifact, directory=root / "build" / f"sb_{name}",
        optimization="O0",
    )
    parameters = dict(function.metadata["parameter_names"])
    published = [int(v.id) for v in outputs[qualified]]
    feeds = {parameters["matrix"]: MATRIX.copy()}
    if "rhs" in parameters:
        feeds[parameters["rhs"]] = RHS.copy()
    for value_id in published:
        feeds[value_id] = np.full(out_shape, SENTINEL)
    execution = prepare_artifact_execution(native, feeds)
    import os
    for buffer_id, buffer in ([] if os.environ.get('PROBE_NOPOISON') else execution.buffers.items()):
        if buffer_id in feeds:
            continue
        try:
            buffer[...] = SENTINEL
        except Exception:
            pass
    execution.run()
    produced = np.asarray(execution.buffers[published[0]]).reshape(out_shape)
    state = (
        "UNTOUCHED" if np.all(produced == SENTINEL)
        else "NaN" if np.any(np.isnan(produced))
        else "real"
    )
    # The same authored source, run eagerly, is the truth to compare against.
    from src.common.tensors.abstraction import AbstractTensor

    namespace = {}
    exec(compile(source, f"<{name}>", "exec"), namespace)
    eager_args = [AbstractTensor.tensor(MATRIX.tolist())]
    if "rhs" in parameter_names:
        eager_args.append(AbstractTensor.tensor(RHS.tolist()))
    try:
        eager = np.asarray(
            namespace["stage"](*eager_args).tolist(), dtype=float
        ).reshape(-1)
    except Exception as error:
        eager = None
        print(f"{name}: EAGER FAILED {type(error).__name__}: {error}",
              flush=True)
    flat = produced.reshape(-1)
    if eager is None:
        verdict = "?"
    elif np.allclose(flat, eager, rtol=1e-9, atol=1e-12, equal_nan=True):
        verdict = "MATCH"
    else:
        verdict = "DIFFERS"
    print(f"{name}: {verdict} [{state}] native={flat} eager={eager}",
          flush=True)


for stage_name, stage_source, stage_parameters, stage_shape in STAGES:
    try:
        run_stage(stage_name, stage_source, stage_parameters, stage_shape)
    except Exception as error:
        print(f"{stage_name}: FAILED {type(error).__name__}: {error}", flush=True)
