# Task 1 — loop-interchange integration and measurement

## Observed code state

- `src/compiler/loop_interchange.py` contains a narrow AST rewrite and decision
  records, but `rg` finds no caller and no repository test importing it.
- `lower_ast_source_to_ssa` in `src/compiler/fortran_c_shell.py` parses `source`
  immediately and builds the ProcessGraph from that tree.  Interchange must run
  before this parse to affect all canonical callers (kernel bank, BLAS probes,
  autogenesis, absorption, and native shells).
- The pass consults `active_contract().inexact_identities` when `licensed` is
  omitted.  This correctly separates `develop`/`prove` from `deploy`/`fast`.
- The accumulator safety condition is presently ineffective:
  `accumulator in _names_in(term) - {accumulator}` can never be true.  It must
  be repaired before wiring.
- `_loop_bound_names()` is computed and discarded.  It currently proves
  nothing and should either support a tested legality rule or be removed.
- `IRModule` has no module-level metadata mapping.  `Function.metadata` is the
  existing durable receipt surface; an integration must not invent an
  unconsumed side channel.
- `_loop_carried_storage_aliases()` operates on canonical reduced graphs.  Its
  current behavior is already pinned by `tests/test_compiled_linalg.py`; a new
  guard is justified only by a failing test, not by the pre-deterministic patch.

## Work sequence

1. Add `tests/test_loop_interchange.py` covering exact refusal, fast/deploy
   acceptance, nonmatching shapes, accumulator leakage, source-location
   stability, decision serialization, and numerical equivalence.
2. Repair legality defects found by those tests before touching the entry
   point.  Preserve the intentionally narrow recognized grammar.
3. Capture the untouched `f18680b` 64³ fast-contract result in a separate
   build root, recording command, source digest, compiler fingerprint, error,
   median steady-state time, and GF/s.
4. Call the pass exactly once at the beginning of `lower_ast_source_to_ssa`.
   Parse and ingest `result.source`, not the authored string.
5. After `_class_surface_ssa_program` returns, attach JSON-shaped decision
   receipts to the corresponding lowered function metadata.  Include source
   function, line, verdict, reasons, and contract name.  Do not store AST
   objects or process-local IDs.
6. Add entry-point tests proving exact contracts retain source shape, fast
   contracts transform, decisions survive into SSA metadata, and two clean
   Python processes emit identical IDs and decisions.
7. Run the regression floor, then repeat the same 64³ measurement in a fresh
   post-integration build root.  Only then run the 1024³ demonstration.

## Acceptance

- No rewrite under `develop` or `prove`.
- The exact GEMM form rewrites under `fast`; an adversarial term mentioning the
  accumulator or unsafe store remainder refuses.
- Compiled outputs agree with NumPy within the fast contract's stated bound.
- Same source/contract gives identical decision records and deterministic SSA
  identity across processes.
- Before/after measurements use identical source, inputs, compiler flags,
  warmup, repetition count, and statistic.
- The focused compiler floor and `tests/test_loop_interchange.py` pass.

## Stop conditions

Do not proceed to tile re-baselining if the transform changes exact-contract
results, loses decision provenance, breaks cross-process identity, or if the
64³ post-transform result is not actually the intended unit-stride loop shape
in emitted LLVM/native code.
