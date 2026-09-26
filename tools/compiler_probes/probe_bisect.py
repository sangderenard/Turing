"""Compile one stage of the solver at a time and compare it with NumPy.

The published output is written and is NaN, so the program runs and computes
one.  Every intermediate lives in an alloca, so nothing downstream of the
entry is visible from the buffer table -- the only way to see where the NaN
enters is to make an intermediate BE the output.

Each stage below returns a different value from the same authored code, so a
stage that matches NumPy exonerates everything it depends on and a stage that
does not is where to look.  Buffers are poisoned with a sentinel rather than
NaN, because poisoning with NaN cannot detect a write OF NaN.
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

MATRIX = np.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
RHS = np.asarray([1.0, 2.0], dtype=np.float64)
SENTINEL = -12345.0
root = Path(__file__).resolve().parents[2]

STAGES = (
    (
        "clone",
        """
def stage(matrix):
    return matrix.clone()
""",
        ("matrix",),
        (2, 2),
    ),
    (
        "axis_index",
        """
from src.common.tensors.linalg import _axis_index

def stage(matrix):
    return _axis_index(matrix, 2)
""",
        ("matrix",),
        (2,),
    ),
    (
        "identity_seed",
        """
from src.common.tensors.linalg import _axis_index

def stage(matrix):
    index = _axis_index(matrix, 2)
    return (index.unsqueeze(-1) == index.unsqueeze(-2)).to_dtype(
        matrix.get_dtype()
    )
""",
        ("matrix",),
        (2, 2),
    ),
    (
        "pivot_mask",
        """
from src.common.tensors.linalg import _pivot_mask

def stage(matrix):
    return _pivot_mask(matrix, 0)
""",
        ("matrix",),
        (2,),
    ),
    (
        "pm_candidates",
        """
from src.common.tensors.linalg import _axis_index, _column, _one_hot_axis

def stage(matrix):
    dtype = matrix.get_dtype()
    eligible = (_axis_index(matrix, 2) >= 0).to_dtype(dtype)
    column = abs(_column(matrix, _one_hot_axis(matrix, 2, 0)))
    return column * eligible - (1 - eligible)
""",
        ("matrix",),
        (2,),
    ),
    (
        "pm_largest",
        """
from src.common.tensors.linalg import _axis_index, _column, _one_hot_axis

def stage(matrix):
    dtype = matrix.get_dtype()
    eligible = (_axis_index(matrix, 2) >= 0).to_dtype(dtype)
    column = abs(_column(matrix, _one_hot_axis(matrix, 2, 0)))
    candidates = column * eligible - (1 - eligible)
    return candidates.max(dim=-1, keepdim=True)
""",
        ("matrix",),
        (1,),
    ),
    (
        "pm_eqmask",
        """
from src.common.tensors.linalg import _axis_index, _column, _one_hot_axis

def stage(matrix):
    dtype = matrix.get_dtype()
    eligible = (_axis_index(matrix, 2) >= 0).to_dtype(dtype)
    column = abs(_column(matrix, _one_hot_axis(matrix, 2, 0)))
    candidates = column * eligible - (1 - eligible)
    largest = candidates.max(dim=-1, keepdim=True)
    return (candidates == largest).to_dtype(dtype) * eligible
""",
        ("matrix",),
        (2,),
    ),
    (
        "cumsum_only",
        """
from src.common.tensors.linalg import _axis_index, _one_hot_axis

def stage(matrix):
    dtype = matrix.get_dtype()
    hot = _one_hot_axis(matrix, 2, 0)
    return hot.cumsum(dim=-1)
""",
        ("matrix",),
        (2,),
    ),
    (
        "cumsum_eq",
        """
from src.common.tensors.linalg import _axis_index, _one_hot_axis

def stage(matrix):
    dtype = matrix.get_dtype()
    hot = _one_hot_axis(matrix, 2, 0)
    return (hot.cumsum(dim=-1) == 1).to_dtype(dtype)
""",
        ("matrix",),
        (2,),
    ),
    (
        "cmp_plain",
        """
from src.common.tensors.linalg import _one_hot_axis

def stage(matrix):
    hot = _one_hot_axis(matrix, 2, 0)
    return (hot == 1).to_dtype(matrix.get_dtype())
""",
        ("matrix",),
        (2,),
    ),
    (
        "cumsum_cast",
        """
from src.common.tensors.linalg import _one_hot_axis

def stage(matrix):
    hot = _one_hot_axis(matrix, 2, 0)
    return hot.cumsum(dim=-1).to_dtype(matrix.get_dtype())
""",
        ("matrix",),
        (2,),
    ),
    (
        "fo_inline",
        """
from src.common.tensors.linalg import _axis_index, _column, _one_hot_axis

def stage(matrix):
    dtype = matrix.get_dtype()
    eligible = (_axis_index(matrix, 2) >= 0).to_dtype(dtype)
    column = abs(_column(matrix, _one_hot_axis(matrix, 2, 0)))
    candidates = column * eligible - (1 - eligible)
    largest = candidates.max(dim=-1, keepdim=True)
    eqmask = (candidates == largest).to_dtype(dtype) * eligible
    return eqmask * (eqmask.cumsum(dim=-1) == 1).to_dtype(dtype)
""",
        ("matrix",),
        (2,),
    ),
    (
        "fo_call",
        """
from src.common.tensors.linalg import (
    _axis_index, _column, _first_occurrence, _one_hot_axis,
)

def stage(matrix):
    dtype = matrix.get_dtype()
    eligible = (_axis_index(matrix, 2) >= 0).to_dtype(dtype)
    column = abs(_column(matrix, _one_hot_axis(matrix, 2, 0)))
    candidates = column * eligible - (1 - eligible)
    largest = candidates.max(dim=-1, keepdim=True)
    eqmask = (candidates == largest).to_dtype(dtype) * eligible
    return _first_occurrence(eqmask, dtype)
""",
        ("matrix",),
        (2,),
    ),
    (
        "one_hot",
        """
from src.common.tensors.linalg import _one_hot_axis

def stage(matrix):
    return _one_hot_axis(matrix, 2, 0)
""",
        ("matrix",),
        (2,),
    ),
    (
        "row0",
        """
from src.common.tensors.linalg import _one_hot_axis, _row

def stage(matrix):
    return _row(matrix, _one_hot_axis(matrix, 2, 0))
""",
        ("matrix",),
        (2,),
    ),
    (
        "column0",
        """
from src.common.tensors.linalg import _one_hot_axis, _column

def stage(matrix):
    return _column(matrix, _one_hot_axis(matrix, 2, 0))
""",
        ("matrix",),
        (2,),
    ),
    (
        "masked_rows",
        """
from src.common.tensors.linalg import _masked_pivot_rows, _pivot_mask

def stage(matrix):
    rows, parity = _masked_pivot_rows(matrix, 0, _pivot_mask(matrix, 0))
    return rows
""",
        ("matrix",),
        (2, 2),
    ),
    (
        "lu_U",
        """
from src.common.tensors.linalg import _lu_decompose_inplace

def stage(matrix):
    upper, sign, permutation = _lu_decompose_inplace(matrix)
    return upper
""",
        ("matrix",),
        (2, 2),
    ),
    (
        "lu_P",
        """
from src.common.tensors.linalg import _lu_decompose_inplace

def stage(matrix):
    upper, sign, permutation = _lu_decompose_inplace(matrix)
    return permutation
""",
        ("matrix",),
        (2, 2),
    ),
)


def expected_for(name):
    import scipy.linalg as sla  # noqa: F401  (only if available)


def run_stage(name, source, parameter_names, out_shape):
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
        source, "stage", name=f"bisect_{name}",
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _m: None,
    )
    qualified = f"bisect_{name}__stage"
    function = module.functions[qualified]
    artifact = emit_ssa_function_to_llvm(
        module, qualified, entry_name=f"bisect_{name}_stage",
    )
    if artifact.shortfalls:
        print(f"{name}: SHORTFALLS {artifact.shortfalls}", flush=True)
        return
    native = compile_artifact(
        artifact, directory=root / "build" / f"bisect_{name}",
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
    for buffer_id, buffer in execution.buffers.items():
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
        print(f"{stage_name}: FAILED {type(error).__name__}: {error}",
              flush=True)
