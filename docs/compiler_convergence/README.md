# Compiler convergence program

Baseline: `f18680ba9ad2b7d8470518dcde1616eae69f5de4` (2026-08-20).

This directory is the execution plan for converging the compiler performance,
native deployment, deterministic identity, training-corpus, and difficult
translation lanes.  It was written from the code at the baseline above.  Older
reports and handoffs are evidence leads, not statements of current behavior.

## One goal

Produce a stable compiler in which legal loop restructuring improves the
BLAS-shaped native floor, performance evidence selects baked variants, those
choices survive into Python-free native products, cross-backend measurements
compare the resulting program honestly, and deterministic structural identity
feeds a compiler-teacher corpus without destabilizing compilation.

## Execution order

The critical path is sequential because every later measurement depends on the
loop shape produced by the earlier stage.

1. [Loop-interchange integration](01_loop_interchange_integration.md): test and
   repair the standalone pass, wire it before ProcessGraph ingestion, retain
   decisions, verify deterministic IDs, and measure before/after.
2. [Tile-choice re-baseline](02_tile_choice_rebaseline.md): rebuild the real
   32/64/128/256 fast-contract ladder and prove the chart-driven chooser reads
   the new evidence.
3. [Native tiling seam](03_native_tiling_seam.md): carry workers and chunks from
   deployment planning into emitted C and `turing_pool.c`, with serial fallback.
4. [Backend bake-off](04_backend_bakeoff.md): add a common, correctness-gated
   measurement surface instead of calling the LLVM/NumPy script a backend
   matrix.
5. [Post-BLAS `eigh` profiling](05_eigh_profile.md): profile the now-correct
   compiled Jacobi program and separately scope the absent backward override.

Two lanes may advance independently after task 1 has a checkpoint, provided
they do not edit its compiler-entry files:

- [Training and spectral faculty](06_training_and_spectral_faculty.md).
- [Deterministic-token performance watch](10_token_identity_performance.md).

The remaining lanes are explicitly staged behind reproductions or demand:

- [Tiny-trip evaporation](07_tiny_trip_evaporation.md).
- [`re._compile` region ownership](08_re_compile_region_ownership.md).
- [Shoal native publication/OOB investigation](09_shoal_native_lane.md).

## Program invariants

- No stash/pop, destructive checkout, or hidden cache comparison.  Use a clean
  worktree when a baseline binary is required.
- The `develop` and `prove` contracts retain authored floating-point order.
  Interchange may fire only when `inexact_identities` is true (`deploy`/`fast`
  in `src/compiler/work_contract.py`).
- Same source plus same contract must produce byte-equivalent interchange
  decisions, dense IDs, and `ssa_identity_tokens` across Python processes.
- Every performance row follows a correctness gate and records source digest,
  contract, compiler fingerprint, dimensions, warmup, repetitions, and timing
  statistic.
- A chart is evidence only for the compiler shape that generated it.  Wiring
  interchange invalidates the existing kernel-bank fingerprint and requires a
  new ladder.
- Native finished products may not require a Python host pool.
- Learned translation never silently substitutes for an unavailable compiler
  transform.  It either uses a ready weight cell or records a compiler-teacher
  command.

## Regression floor

Use the bounded test policy in `TEST_BASELINE_AND_HAZARDS.md`; never run the
whole suite.  The cross-cutting compiler floor is:

```powershell
python tools\translation_scorecard.py
python -m pytest tests\test_precompile_to_ssa.py -q --tb=short
python -m pytest tests\test_symbolic_fluid_native_runtime.py -q --tb=short
python -m pytest tests\test_abstract_tensor_indexing.py -q --tb=short
python -m pytest tests\test_ssa_fusion_regions.py -q --tb=short
python -m pytest tests\test_region_kernel_dedup.py -q --tb=short
python -m pytest tests\test_compiled_linalg.py -q --tb=short
```

Task documents add narrower tests.  A stage ends with `git diff --check`, a
clean or fully explained `git status`, and a checkpoint commit before the next
stage changes the measurement floor.

## Convergence checkpoint — 2026-08-20

Tasks 1, 2, 4, 5, and 6 have executable evidence. Task 3 has a proven native
span ABI, consumed prebake matrix, manifest propagation, and an explicit
remaining product-renderer adoption seam. Tasks 7–10 remain bounded by their
documents; no speculative fixes were mixed into this checkpoint.

The touched-feature gate passed 73 tests plus the end-to-end prebake demo. The
regression floor remains 17/19 scorecard journeys, 34 precompile tests, native
fluid runtime, indexing, fusion, dedup, and compiled linear algebra at 6
passed / 1 strict xfail.
