from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from src.common.tensors.blas import GEMM_SOURCE
from src.compiler.loop_interchange import interchange_reduction_loops


def _run_gemm(source: str, a, b, c, *, size: int = 5):
    namespace: dict = {}
    exec(compile(source, "<gemm>", "exec"), namespace)
    return namespace["gemm"](
        a.copy(), b.copy(), c.copy(), 1.7, 0.3, size, size, size,
    )


def test_exact_contract_refuses_profitable_interchange():
    result = interchange_reduction_loops(GEMM_SOURCE, licensed=False)

    assert result.source == GEMM_SOURCE
    assert len(result.decisions) == 1
    assert not result.decisions[0].interchanged
    assert "forbids inexact identities" in " ".join(
        result.decisions[0].reasons
    )


def test_licensed_gemm_interchanges_to_unit_stride_inner_loop():
    result = interchange_reduction_loops(GEMM_SOURCE, licensed=True)

    assert result.decisions[0].interchanged
    assert "for p in range(k):\n            for j in range(n):" in result.source
    assert "C[i * n + j] = beta * C[i * n + j]" in result.source
    assert "total" not in result.source


def test_interchanged_gemm_matches_the_authored_program():
    rng = np.random.default_rng(12)
    a = rng.standard_normal(25)
    b = rng.standard_normal(25)
    c = rng.standard_normal(25)
    transformed = interchange_reduction_loops(
        GEMM_SOURCE, licensed=True,
    ).source

    expected = _run_gemm(GEMM_SOURCE, a, b, c)
    produced = _run_gemm(transformed, a, b, c)

    assert np.max(np.abs(produced - expected)) < 2.0e-14


def test_accumulator_in_the_term_is_refused():
    source = GEMM_SOURCE.replace(
        "A[i * k + p] * B[p * n + j]",
        "total * A[i * k + p] * B[p * n + j]",
    )
    result = interchange_reduction_loops(source, licensed=True)

    assert not result.decisions[0].interchanged
    assert "accumulator" in " ".join(result.decisions[0].reasons)


def test_reduction_variable_in_the_store_remainder_is_refused():
    source = GEMM_SOURCE.replace(
        "beta * C[i * n + j]", "p",
    )
    result = interchange_reduction_loops(source, licensed=True)

    assert not result.decisions[0].interchanged
    assert "promotion is not equivalence-safe" in " ".join(
        result.decisions[0].reasons
    )


def test_effectful_reduction_term_is_refused():
    source = GEMM_SOURCE.replace(
        "A[i * k + p] * B[p * n + j]",
        "sample(A, i * k + p) * B[p * n + j]",
    )
    result = interchange_reduction_loops(source, licensed=True)

    assert not result.decisions[0].interchanged
    assert "side-effect-free" in " ".join(result.decisions[0].reasons)


def test_reduction_that_reads_its_destination_is_refused():
    source = GEMM_SOURCE.replace(
        "A[i * k + p] * B[p * n + j]",
        "C[i * n + j] * B[p * n + j]",
    )
    result = interchange_reduction_loops(source, licensed=True)

    assert not result.decisions[0].interchanged
    assert "destination buffer" in " ".join(result.decisions[0].reasons)


def test_non_range_iterator_is_refused():
    source = GEMM_SOURCE.replace("for p in range(k):", "for p in indices:")
    result = interchange_reduction_loops(source, licensed=True)

    assert not result.decisions[0].interchanged
    assert "range loops" in " ".join(result.decisions[0].reasons)


def test_source_and_decisions_are_deterministic():
    first = interchange_reduction_loops(GEMM_SOURCE, licensed=True)
    second = interchange_reduction_loops(GEMM_SOURCE, licensed=True)

    assert first.source == second.source
    assert tuple(map(asdict, first.decisions)) == tuple(
        map(asdict, second.decisions)
    )


def _lower_under_contract(contract: str):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.work_contract import set_active_contract

    set_active_contract(contract)
    try:
        return lower_ast_source_to_ssa(
            GEMM_SOURCE, "gemm", name=f"interchange_{contract}",
        )[0]
    finally:
        set_active_contract(None)


def test_canonical_entry_records_exact_contract_refusal():
    module = _lower_under_contract("develop")
    receipt = module.metadata["loop_interchange"]

    assert receipt["contract"] == "develop"
    assert not receipt["licensed"]
    assert not receipt["changed"]
    assert receipt["authored_source_sha256"] == receipt[
        "transformed_source_sha256"
    ]
    assert not receipt["decisions"][0]["interchanged"]


def test_canonical_entry_records_fast_interchange_on_the_source_function():
    module = _lower_under_contract("fast")
    receipt = module.metadata["loop_interchange"]
    function = module.functions["interchange_fast__gemm"]

    assert receipt["contract"] == "fast"
    assert receipt["licensed"]
    assert receipt["changed"]
    assert receipt["decisions"][0]["interchanged"]
    assert function.metadata["loop_interchange_decisions"] == (
        receipt["decisions"][0],
    )


def test_fast_interchanged_gemm_compiles_and_matches_numpy():
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution
    from tools.benchmark_blas_vs_numpy import compile_kernel

    size = 9  # Stay above the separately pinned tiny-trip evaporator defect.
    rng = np.random.default_rng(19)
    a = rng.standard_normal(size * size)
    b = rng.standard_normal(size * size)
    c = rng.standard_normal(size * size)
    expected = (
        1.7 * (a.reshape(size, size) @ b.reshape(size, size)).reshape(-1)
        + 0.3 * c
    )
    native, identifiers, _outputs, _returns = compile_kernel(
        "gemm", GEMM_SOURCE, "fast",
        Path(tempfile.mkdtemp(prefix="interchange_native_")),
        tag_suffix="_equivalence",
    )
    execution = prepare_artifact_execution(native, {
        identifiers["A"]: a,
        identifiers["B"]: b,
        identifiers["C"]: c,
        identifiers["alpha"]: 1.7,
        identifiers["beta"]: 0.3,
        identifiers["m"]: size,
        identifiers["n"]: size,
        identifiers["k"]: size,
    })
    execution.run()
    produced = np.asarray(execution.buffers[identifiers["C"]])

    assert np.allclose(produced, expected, rtol=2.0e-13, atol=2.0e-13)


def test_fast_entry_identity_and_decisions_are_stable_across_processes():
    script = r'''
import json
from src.common.tensors.blas import GEMM_SOURCE
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.work_contract import set_active_contract

set_active_contract("fast")
try:
    module, _outputs, _exports = lower_ast_source_to_ssa(
        GEMM_SOURCE, "gemm", name="interchange_process",
    )
finally:
    set_active_contract(None)
print(json.dumps({
    "receipt": module.metadata["loop_interchange"],
    "functions": {
        name: {
            "args": [value.id for value in function.args],
            "instructions": [
                [instruction.op, [value.id for value in instruction.args],
                 None if instruction.res is None else instruction.res.id]
                for block in function.blocks.values()
                for instruction in block.instrs
            ],
        }
        for name, function in module.functions.items()
    },
}, sort_keys=True))
'''
    first = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True,
        text=True,
    ).stdout
    second = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True,
        text=True,
    ).stdout

    assert first == second
