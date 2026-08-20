# Task 5 — post-BLAS `eigh` profiling and backward boundary

## Observed code state

- `tests/test_compiled_linalg.py::test_the_compiled_eigh_matches_numpy` is a
  normal passing test.  It compiles the explicit `JACOBI_EIGH` source through
  repository SSA and native LLVM and validates eigenvalues, orthogonality, and
  residuals.
- `AbstractTensor.eigh` points to the Jacobi implementation in
  `src/common/tensors/abstraction_methods/eigen.py`; it is not currently a
  single compiled operator dispatch.
- `GradTape.backward_overrides` is implemented and used by capture/gradient
  tests, but no `eigh` override is installed.
- The BLAS benchmark tool does not profile the Jacobi kernel.

## Work sequence

1. After tasks 1–4, add an `eigh` profiling tool that uses the exact passing
   test kernel and separately profiles compile, prepare, first call, and warm
   calls over a size ladder.
2. Validate eigenvalue error, `V.T @ V`, and `A @ V - V @ diag(w)` for every
   measured size before retaining a timing row.
3. Attribute time by emitted function/loop where available; do not reopen GEMM
   performance unless the profile demonstrates it remains dominant.
4. Treat backward as a distinct deliverable.  Specify an opaque `eigh`
   operation name, saved forward values, degeneracy policy, and a
   `GradTape.backward_overrides` implementation, with finite-difference tests.

## Acceptance

- Profiling never weakens the existing compiled correctness test.
- Results state algorithm, sweeps, tolerance, contract, and matrix spectrum.
- Repeated/near-repeated eigenvalues have an explicit backward policy; no
  silent unstable gradient is installed.

## Result — 2026-08-20

EIGH offers the definitional AbstractTensor Jacobi path and an explicit
`method="blas"` path using the repository's compiled, admission-verified
`rot` kernel. Small matrices zero-pad to a specialized width-nine module to
The original run stayed above the then-standing tiny-trip gate; that gate was
removed after nested carried-recurrence preservation was added (Task 7).
The report carries the actual module key and deterministic name-to-ID binding.
The default remains Jacobi.

| n | compiled whole Jacobi | AbstractTensor Jacobi | bank-backed native rot |
|---:|---:|---:|---:|
| 3 | 0.009 ms | 88.349 ms | 2.035 ms |
| 4 | 0.012 ms | 175.641 ms | 3.404 ms |
| 6 | 0.024 ms | 431.680 ms | 9.673 ms |

All eigenvalue, orthogonality, and residual errors are at or below 4.0e-15.
The rot path is much faster than eager operator dispatch, while compiling the
whole Jacobi program remains the performance frontier. No opaque EIGH backward
override is installed; the report states that boundary rather than providing
an unstable repeated-eigenvalue gradient.
