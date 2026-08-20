# Task 7 — tiny-trip loop evaporation defect

## Observed code state

- `KernelBank.get()` refuses any specialized size at or below
  `LoopBackendCapabilities.unroll_limit` before compilation.
- `tests/test_compiled_linalg.py` contains a strict xfail using the SSA
  reference evaluator.  The stated native form access-violates, so it is not a
  safe xfail subprocess inside pytest.
- `evaporate_unrolled_loops()` already requires canonical value IDs.  The
  defect therefore survives the deterministic-ID migration and is not fixed by
  renumbering alone.

## Work sequence

1. Use the reference-evaluator pin as the primary reproducer and inspect the
   outer-loop induction ownership through `loop_composer.py`.
2. Repair cloned membership/publication so every outer iteration survives
   inner-loop evaporation.
3. Flip the strict xfail, then add a separately contained native subprocess
   check before relaxing the bank refusal.
4. Remove or narrow the precompile refusal only after the native proof is safe.

## Acceptance

- The 2³ baked GEMM reference-evaluator pin passes exactly.
- The native subprocess exits normally and matches NumPy.
- Bank admission safely accepts newly legal tiny variants without exposing the
  parent pytest process to an access violation.
