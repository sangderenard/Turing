"""Linear-algebra-shaped kernels through the LLVM lane: what works, pinned;
what fails, pinned by its exact failure.

This suite is the test-side of ``docs/FUNCTION_TO_DEPLOYMENT_HANDOFF.md``:
the eigh compilation effort proved the pipeline can turn plainly-written
Python linear algebra into fast native code, but its verification lived and
died inside one session.  These tests make the envelope it mapped a
permanent, executable claim.

Two kinds of test live here:

* **Envelope guards** -- kernel shapes measured to compile AND compute
  correctly (dependent inner bounds, the full Jacobi rotation-angle
  arithmetic with its branch-free gate, a four-deep sweep nest).  If one of
  these starts failing, the proven envelope shrank.

* **A defect pin, strict-xfail** -- the smallest reproduction of the defect
  that currently stops a full Jacobi eigh (and any kernel of its shape):

      for i0 in range(n):
          a[i0] = a[i0] * 2.0
      for i1 in range(n):
          a[i1] = a[i1] + 1.0

  Two sequential loops storing to the SAME array lose the first loop's
  stores entirely -- native returns ``a + 1``, not ``2a + 1`` -- while
  ``artifact.shortfalls`` stays ``()``.  Loop variables are distinct, so
  this is NOT the reused-loop-variable defect
  (``test_reused_loop_variable_dominance.py``): that one emits invalid IR
  the verifier rejects; this one emits valid IR that silently computes the
  wrong thing.  The lost first loop also surfaces as an unnamed float64
  extra formal on the emitted signature (its orphaned array version).
  Measured 2026-08-20.  In-place row-rotate then column-rotate -- the core
  of every two-sided Jacobi, LU, QR -- is exactly this shape, which is why
  the full eigh test below is xfail until this is fixed.

Compile hygiene (from ``tests/test_llvm_inplace_store_aliasing.py``): each
artifact is compiled at most once per session (zig's compiler_rt cache races
when the same artifact compiles twice in quick succession), and every
unknown extra formal is fed an OVERSIZED zero buffer -- an orphaned array
version fed a length-1 scratch cell writes out of bounds and kills the
whole pytest process, which an xfail cannot contain.
"""
from __future__ import annotations

import pathlib
import tempfile
import warnings

import numpy as np
import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)


def _run_native(source, entrypoint, name, arrays, scalars, reads, scratch_len):
    """Lower, emit, compile, run; return {name: buffer} for ``reads``.

    Extra formals beyond the authored signature are fed zeros sized
    ``scratch_len`` (floats) -- large enough that an orphaned array version
    cannot write out of bounds -- or a single 0 (ints: leaked loop
    induction variables, verified harmless in the aliasing suite).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            source, entrypoint, name=name
        )
    qualified = f"{name}__{entrypoint}"
    function = module.functions[qualified]
    parameters = dict(function.metadata["parameter_names"])

    artifact = emit_ssa_function_to_llvm(module, qualified)
    assert artifact.shortfalls == (), (
        "these kernels are expected to emit cleanly; a shortfall here means "
        f"the envelope changed shape, not size: {artifact.shortfalls}"
    )
    native = compile_artifact(
        artifact, directory=pathlib.Path(tempfile.mkdtemp()) / name
    )

    feed = {parameters[key]: value.copy() for key, value in arrays.items()}
    feed.update(
        {parameters[key]: np.array([value]) for key, value in scalars.items()}
    )
    for formal in function.args:
        identifier = int(formal.id)
        if identifier not in feed:
            feed[identifier] = (
                np.array([0])
                if formal.dtype == "int"
                else np.zeros(scratch_len)
            )

    execution = prepare_artifact_execution(native, feed)
    execution.run()
    return {
        key: np.asarray(execution.buffers[parameters[key]]).copy()
        for key in reads
    }


# ---------------------------------------------------------------------------
# Envelope guards: measured to compile and compute correctly.
# ---------------------------------------------------------------------------

# The full Jacobi rotation-angle arithmetic -- relative-threshold gate
# expressed branch-free as bool->float (handoff section 5: the ABSOLUTE
# cutoff was a real numerical bug on graded matrices; the RELATIVE criterion
# is the one high relative accuracy requires), abs, **0.5, a dependent inner
# bound -- applied through ONE in-place rotation loop.  One loop, not the
# full row+column pair, because of the sequential-store defect pinned below.
ROTATION_WITH_GATE = """
def rotate(a, n, eps):
    for p0 in range(n):
        for q0 in range(p0 + 1, n):
            apq = a[p0 * n + q0]
            app = a[p0 * n + p0]
            aqq = a[q0 * n + q0]
            gate = (abs(apq) > eps * (abs(app * aqq)) ** 0.5) * 1.0
            denom = 2.0 * apq + (1.0 - gate)
            theta = (aqq - app) / denom
            sgn = (theta >= 0.0) * 2.0 - 1.0
            t = sgn / (abs(theta) + (theta * theta + 1.0) ** 0.5)
            cr = 1.0 / (t * t + 1.0) ** 0.5
            sr = t * cr
            c = gate * cr + (1.0 - gate)
            s = gate * sr
            for k1 in range(n):
                apk = a[p0 * n + k1]
                aqk = a[q0 * n + k1]
                a[p0 * n + k1] = c * apk - s * aqk
                a[q0 * n + k1] = s * apk + c * aqk
    return a
"""


def _rotation_with_gate_reference(a, n, eps):
    a = a.copy()
    for p0 in range(n):
        for q0 in range(p0 + 1, n):
            apq = a[p0 * n + q0]
            app = a[p0 * n + p0]
            aqq = a[q0 * n + q0]
            gate = (abs(apq) > eps * (abs(app * aqq)) ** 0.5) * 1.0
            denom = 2.0 * apq + (1.0 - gate)
            theta = (aqq - app) / denom
            sgn = (theta >= 0.0) * 2.0 - 1.0
            t = sgn / (abs(theta) + (theta * theta + 1.0) ** 0.5)
            cr = 1.0 / (t * t + 1.0) ** 0.5
            sr = t * cr
            c = gate * cr + (1.0 - gate)
            s = gate * sr
            for k1 in range(n):
                apk = a[p0 * n + k1]
                aqk = a[q0 * n + k1]
                a[p0 * n + k1] = c * apk - s * aqk
                a[q0 * n + k1] = s * apk + c * aqk
    return a


def test_jacobi_rotation_arithmetic_computes_natively():
    n = 5
    rng = np.random.default_rng(3)
    m = rng.standard_normal((n, n))
    a0 = (m.T @ m).reshape(-1)
    produced = _run_native(
        ROTATION_WITH_GATE, "rotate", "linrot",
        {"a": a0}, {"n": n, "eps": 1e-15}, ("a",), n * n,
    )
    assert np.allclose(
        produced["a"], _rotation_with_gate_reference(a0, n, 1e-15)
    )


# A four-deep nest: sweeps -> p -> dependent q -> k, one in-place rotation
# loop.  Depth itself is not the blocker; the sequential same-array store is.
SWEEP_NEST = """
def sweep(a, n, sweeps):
    for s0 in range(sweeps):
        for p0 in range(n):
            for q0 in range(p0 + 1, n):
                apq = a[p0 * n + q0]
                c = apq * 0.001 + 1.0
                s = apq * 0.001
                for k1 in range(n):
                    apk = a[p0 * n + k1]
                    aqk = a[q0 * n + k1]
                    a[p0 * n + k1] = c * apk - s * aqk
                    a[q0 * n + k1] = s * apk + c * aqk
    return a
"""


def _sweep_nest_reference(a, n, sweeps):
    a = a.copy()
    for _ in range(sweeps):
        for p0 in range(n):
            for q0 in range(p0 + 1, n):
                apq = a[p0 * n + q0]
                c = apq * 0.001 + 1.0
                s = apq * 0.001
                for k1 in range(n):
                    apk = a[p0 * n + k1]
                    aqk = a[q0 * n + k1]
                    a[p0 * n + k1] = c * apk - s * aqk
                    a[q0 * n + k1] = s * apk + c * aqk
    return a


def test_four_deep_sweep_nest_computes_natively():
    n = 4
    a0 = np.arange(1.0, n * n + 1.0)
    produced = _run_native(
        SWEEP_NEST, "sweep", "linsweep",
        {"a": a0}, {"n": n, "sweeps": 3}, ("a",), n * n,
    )
    assert np.allclose(produced["a"], _sweep_nest_reference(a0, n, 3))


# ---------------------------------------------------------------------------
# The defect pin.
# ---------------------------------------------------------------------------

SEQUENTIAL_SAME_ARRAY = """
def twice(a, n):
    for i0 in range(n):
        a[i0] = a[i0] * 2.0
    for i1 in range(n):
        a[i1] = a[i1] + 1.0
    return a
"""


@pytest.mark.xfail(
    strict=True,
    reason=(
        "two sequential loops storing to the same array lose the FIRST "
        "loop's stores: native computes a+1 instead of 2a+1, with zero "
        "shortfalls and distinct loop variables (measured 2026-08-20). "
        "This is the defect that blocks a full in-place Jacobi (row-rotate "
        "then column-rotate). A fix must flip this test."
    ),
)
def test_sequential_stores_to_one_array_both_land():
    n = 4
    a0 = np.arange(1.0, n + 1.0)
    produced = _run_native(
        SEQUENTIAL_SAME_ARRAY, "twice", "linseq",
        {"a": a0}, {"n": n}, ("a",), n,
    )
    assert np.allclose(produced["a"], a0 * 2.0 + 1.0)


# ---------------------------------------------------------------------------
# Size-baked (specialized) kernels: correct at real sizes, pinned-broken at
# tiny trip counts.
# ---------------------------------------------------------------------------

BAKED_GEMM_TEMPLATE = """
def gemm(A, B, C, alpha, beta):
    m = {size}
    n = {size}
    k = {size}
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total = total + A[i * k + p] * B[p * n + j]
            C[i * n + j] = alpha * total + beta * C[i * n + j]
    return C
"""


def _baked_gemm_expected(a, b, c, alpha, beta, size):
    return (
        alpha * (a.reshape(size, size) @ b.reshape(size, size)).reshape(-1)
        + beta * c
    )


def test_fully_size_baked_gemm_is_exact_above_the_unroll_limit():
    """All three sizes baked to literals -- the kernel bank's specialization
    shape -- computes exactly, natively, at any size whose trip count
    exceeds the loop unroll limit (8). This is the variant the bank's
    admission gate previously refused; the refusal traced to store-version
    alias chains resolving one level instead of to their root
    (ir_indexing.py), fixed alongside this test."""

    size = 16
    rng = np.random.default_rng(5)
    a0 = rng.standard_normal(size * size)
    b0 = rng.standard_normal(size * size)
    c0 = rng.standard_normal(size * size)
    produced = _run_native(
        BAKED_GEMM_TEMPLATE.format(size=size), "gemm", "linbake",
        {"A": a0, "B": b0, "C": c0},
        {"alpha": 1.7, "beta": 0.3}, ("C",), size * size,
    )
    assert np.allclose(
        produced["C"],
        _baked_gemm_expected(a0, b0, c0, 1.7, 0.3, size),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "at trip counts within the unroll limit the loop evaporator "
        "unrolls the inner loops but drops the OUTER loop's iteration: its "
        "induction variable leaks out as a free formal and only the "
        "iteration where it equals the scratch-fill value computes "
        "(measured 2026-08-20: 2x2 baked gemm computes row 0 exactly and "
        "leaves row 1 untouched). The kernel bank's admission gate catches "
        "this for bank users; this pin covers everyone else. Checked via "
        "the SSA reference evaluator, NOT natively: the native build of "
        "this shape hard-crashes (access violation), which no xfail can "
        "contain -- the evaluator reproduces the same wrong values safely "
        "and pins the defect at the layer it lives, lowering."
    ),
)
def test_fully_size_baked_gemm_is_exact_within_the_unroll_limit():
    from src.compiler.ssa_reference_evaluator import SSAReferenceEvaluator

    size = 2
    a0 = np.array([1.0, 2.0, 3.0, 4.0])
    b0 = np.array([10.0, 20.0, 30.0, 40.0])
    c0 = np.zeros(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            BAKED_GEMM_TEMPLATE.format(size=size), "gemm", name="lintiny"
        )
    function = module.functions["lintiny__gemm"]
    parameters = dict(function.metadata["parameter_names"])
    feed = {
        parameters["A"]: a0.copy(),
        parameters["B"]: b0.copy(),
        parameters["C"]: c0.copy(),
        parameters["alpha"]: 1.0,
        parameters["beta"]: 0.0,
    }
    for formal in function.args:
        if int(formal.id) not in feed:
            feed[int(formal.id)] = 0.0
    result = SSAReferenceEvaluator(module).run("lintiny__gemm", feed)
    produced = np.asarray(result.values[parameters["C"]]).reshape(-1)
    assert np.allclose(
        produced, _baked_gemm_expected(a0, b0, c0, 1.0, 0.0, size)
    )


# ---------------------------------------------------------------------------
# The goal this suite exists for: the full compiled eigh, xfail until the
# sequential-store defect is fixed.  When the pin above flips, this is the
# test that promotes the handoff's 2030x eigh precedent into a permanent,
# CI-checked claim.
# ---------------------------------------------------------------------------

JACOBI_EIGH = """
def jacobi_eigh(a, v, w, n, sweeps, eps):
    for i0 in range(n):
        for j0 in range(n):
            v[i0 * n + j0] = 0.0
        v[i0 * n + i0] = 1.0
    for s0 in range(sweeps):
        for p0 in range(n):
            for q0 in range(p0 + 1, n):
                apq = a[p0 * n + q0]
                app = a[p0 * n + p0]
                aqq = a[q0 * n + q0]
                gate = (abs(apq) > eps * (abs(app * aqq)) ** 0.5) * 1.0
                denom = 2.0 * apq + (1.0 - gate)
                theta = (aqq - app) / denom
                sgn = (theta >= 0.0) * 2.0 - 1.0
                t = sgn / (abs(theta) + (theta * theta + 1.0) ** 0.5)
                cr = 1.0 / (t * t + 1.0) ** 0.5
                sr = t * cr
                c = gate * cr + (1.0 - gate)
                s = gate * sr
                for k1 in range(n):
                    apk = a[p0 * n + k1]
                    aqk = a[q0 * n + k1]
                    a[p0 * n + k1] = c * apk - s * aqk
                    a[q0 * n + k1] = s * apk + c * aqk
                for k2 in range(n):
                    akp = a[k2 * n + p0]
                    akq = a[k2 * n + q0]
                    a[k2 * n + p0] = c * akp - s * akq
                    a[k2 * n + q0] = s * akp + c * akq
                for k3 in range(n):
                    vkp = v[k3 * n + p0]
                    vkq = v[k3 * n + q0]
                    v[k3 * n + p0] = c * vkp - s * vkq
                    v[k3 * n + q0] = s * vkp + c * vkq
    for i1 in range(n):
        w[i1] = a[i1 * n + i1]
    return w
"""


def test_the_jacobi_kernel_is_correct_in_plain_python():
    """The kernel is the spec; prove it against numpy BEFORE compiling.

    This is the handoff's rule 3 (prove equality against the definitional
    version) and its diagnostic rule (run the kernel in plain Python first,
    so a compiled failure indicts the compiler, not the algorithm).
    Deliberately independent of the compile pipeline: it must keep passing
    while the xfail below stands.
    """

    n = 8
    rng = np.random.default_rng(7)
    m = rng.standard_normal((n, n))
    matrix = m.T @ m

    namespace: dict = {}
    exec(compile(JACOBI_EIGH, "<jacobi>", "exec"), namespace)
    a = matrix.reshape(-1).copy()
    v = np.zeros(n * n)
    w = np.zeros(n)
    namespace["jacobi_eigh"](a, v, w, n, 12, 1e-15)

    eigenvectors = v.reshape(n, n)
    reference_w, _ = np.linalg.eigh(matrix)
    assert np.abs(np.sort(w) - reference_w).max() < 1e-12
    assert np.abs(
        eigenvectors.T @ eigenvectors - np.eye(n)
    ).max() < 1e-13
    assert np.abs(
        matrix @ eigenvectors - eigenvectors @ np.diag(w)
    ).max() < 1e-12


@pytest.mark.xfail(
    strict=True,
    reason=(
        "blocked by the sequential same-array store defect: the row-rotate "
        "and column-rotate loops both store into `a`, so the compiled "
        "kernel loses the row rotations and returns a wrong decomposition "
        "(V comes back identity). Flips when the defect pin above flips."
    ),
)
def test_the_compiled_eigh_matches_numpy():
    n = 8
    rng = np.random.default_rng(7)
    m = rng.standard_normal((n, n))
    matrix = m.T @ m

    produced = _run_native(
        JACOBI_EIGH, "jacobi_eigh", "lineigh",
        {
            "a": matrix.reshape(-1),
            "v": np.zeros(n * n),
            "w": np.zeros(n),
        },
        {"n": n, "sweeps": 12, "eps": 1e-15},
        ("w", "v"), n * n,
    )
    eigenvalues = produced["w"]
    eigenvectors = produced["v"].reshape(n, n)

    reference_w, _ = np.linalg.eigh(matrix)
    assert np.abs(np.sort(eigenvalues) - reference_w).max() < 1e-11
    assert np.abs(
        eigenvectors.T @ eigenvectors - np.eye(n)
    ).max() < 1e-12
    assert np.abs(
        matrix @ eigenvectors - eigenvectors @ np.diag(eigenvalues)
    ).max() < 1e-11
