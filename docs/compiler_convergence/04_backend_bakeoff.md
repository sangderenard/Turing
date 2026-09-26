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

## Result — 2026-08-20

`tools/blas_backend_bakeoff.py` lowers one literal-stride GEMM source once to
repository SSA and feeds that module to every backend. At 64 cubed:

| backend | status | worst error | steady compute | GF/s |
|---|---|---:|---:|---:|
| repository SSA evaluator | timed | 4.26e-14 | diagnostic only | — |
| LLVM | timed | 4.26e-14 | 0.390 ms | 1.345 |
| Fortran | timed | 4.26e-14 | 107.374 ms | 0.00488 |
| direct scalar C | unsupported | — | — | — |
| WebGPU | unsupported | — | — | — |

The SSA evaluator retains its global 5,000,000-step runaway guard; this tool
raises it in proportion to the known finite GEMM loop. Direct C refuses the
multi-block function and WebGPU refuses its multi-output region. Refusals
remain rows rather than disappearing. Exact 16- and 64-cubed JSON reports are
in the evidence directory.
