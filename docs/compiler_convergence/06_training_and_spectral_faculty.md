# Task 6 — compiler-teacher training and spectral analysis faculty

## Observed code state

- `CompilerTrainingDatabase` persists programs, views, per-position tokens,
  transformations, compiler commands, and weight-set records in SQLite.
- `GraphTranslationNetwork.densify()` only enqueues commands.  No worker maps
  command names such as `lower_repository_ssa` to compiler functions and then
  calls `complete_command()`.
- All 110 matrix cells created by `tools/compiler_training_network.py` are
  `UnfinishedTransformerCell`; prediction correctly raises instead of
  fabricating output.
- AbstractNN has linear/matmul/autograd/softmax/loss machinery but no completed
  embedding, normalization, positional encoding, attention mask, or sequence
  batching implementation.
- `compile_spectral_dye_trace.py` records real spectral artifacts into the
  corpus, providing one end-to-end teacher chain.

## Work sequence

1. Implement the compiler-command worker for one narrow route first
   (`process_graph -> ssa`).  Commands must use registered callables, validate
   inputs, create the target view, link provenance, and atomically mark the
   command complete.
2. Store source/target node alignment, verification outcome, diagnostics,
   timing, and compiler identity—not merely two serialized payloads.
3. Make retries and failures explicit states.  A crashed teacher command must
   not remain indistinguishable from pending work.
4. Define the first learning problem as constrained alignment/rewrite
   prediction with compiler verification.  Keep the transformer itself a stub
   until the missing AbstractNN sequence primitives have focused tests.
5. Extend spectral target selection to deterministic name/token correlation so
   traces remain resolvable after recompilation and corpus upgrade.

## Acceptance

- A queued route can be fulfilled idempotently into a linked, verified view.
- Failure/retry provenance is retained.
- No untrained cell can emit a compiler result.
- Spectral reports resolve targets by persisted structural identity rather than
  process-local IDs.
