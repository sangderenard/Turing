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
