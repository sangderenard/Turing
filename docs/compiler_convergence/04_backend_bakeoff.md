# Task 4 — correctness-gated backend bake-off

## Observed code state

- `tools/benchmark_blas_vs_numpy.py` lowers through repository SSA, emits LLVM,
  compiles one native artifact, and compares it with NumPy.  Its “operator”
  section measures eager AbstractTensor dispatch on the NumPy backend.
- `benchmark_control_shell_variants.py` profiles control-shell variants, not
  the BLAS kernel matrix.
- No current tool lowers one canonical BLAS SSA module through five backends
  and reports comparable correctness/timing rows.

## Work sequence

1. Define the backend set from callable code, not names in reports.  For each
   candidate, identify its emitter, executable runtime, supported dtype/shape,
   synchronization requirement, and whether timing measures compile, launch,
   or steady compute.
2. Lower one source/contract/specialization to one repository SSA authority.
   Backends consume that module; they do not each re-ingest Python.
3. Require a backend-specific correctness oracle before timing.  Unsupported
   operations become explicit `unsupported` rows, not zeroes or omissions.
4. Separate compilation, first launch, relaunch, and steady compute.  GPU rows
   must synchronize before stopping the clock.
5. Emit JSON/CSV plus a human chart with machine, toolchain, source digest,
   contract, dimensions, repetitions, error, and status.

## Acceptance

- Every timed row passed the same numerical oracle.
- Every row names the same SSA/source identity and contract.
- Timings with different semantics are separate columns, never compared as if
  equivalent.
- Re-running one backend does not overwrite another backend's artifact.
