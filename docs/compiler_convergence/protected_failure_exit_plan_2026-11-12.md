# Protected failure exit (plan B), 2026-11-12

Goal: a safety-armed operator that fails inside a protected `try` body stops
the body before any later effect, as Python does. Program:
`tests/test_source_c_exceptions.py::PROTECTED_LOOP`, n=0 must give 150 (it
gives 179), n=2 must give 132.

Why it is wrong today (traced, see
`docs/hazards/20261112T193712Z_claude-specialized-guard-projection.md`):
the C division arm fills the thread-local error slot and continues with a
placeholder 0. Only the protected body's trailing `ExceptionTake` reads the
slot. The constant loop is evaporated into one straight-line planned region,
so nothing can branch.

Design note this follows:
`ssa_operator_failure_arms_and_thread_exceptions_2026-10-31.md`
("failure exits the callee before subsequent effects"; "Next establish
failure-exit loop carries").

Decision (owner, 2026-11-12): B. The failure exit lives in the control
builder, not inside planned regions. A failable operator inside a protected
body ends its region; the exit is a check after the region call.

## Records (who writes what)

1. Ingestion declares each protected guard. `FailureArms.guard` already
   marks the If (`_turing_exception_guard`). Graph ingestion posts it as a
   book row beside `source_exception_node`, caused by the conversion cell.
2. The reducer records a failure site when it reduces a statement inside a
   protected guard that contains an operator whose SSA kind
   (`source_operator_kind`) has `OPERATOR_FAILURE_ARMS`. The row carries the
   lexical (name, value) bindings visible at the statement, as a `break`
   site does (`SOURCE_LOOP_CONTROL_SITE`), the operator occurrence(s), the
   innermost enclosing loop inside the guard, and the guard. Whether the arm
   is enabled stays the instruction's compile policy.

## Lowering

3. Region partition: a failure site ends its region; nodes after it in
   source order are a separate partition (`_control_partition_keys`).
4. Evaporation: a loop that contains a failure site is not evaporated; it
   stays a loop whose body region ends at the site.
5. Control program: a failure-site block follows the region that holds the
   operator.
6. SSA builder: the block lowers to `ExceptionPending` then `CondBr` to the
   exit. Inside a loop the exit is a break-like loop exit carrying the
   site's bindings (it skips the loop `else`). A construct that contains a
   site is itself a site at its end. The guard arm's exit edges join the
   guard's merge with the site bindings.
7. Backends: `ExceptionPending` = `turing_validation_error_pending() != 0`
   in C and LLVM (existing runtime symbol).
8. Concordance: every new row and edge posts its writer; the audit checks
   the site against its operator and guard.

## Status

Measured: see docs/hazards/20261112T234500Z_claude-protected-failure-exit.md.
The shell's last control pass (`_install_external_reference_calls`) and
`_control_block_consumes_values` also know the block.

- [x] 1 guard declaration: `graph_express2` posts `source_exception_guard`
  beside `source_exception_node`.
- [x] 2 failure-site records: `topological_reducer.record_failure_site`
  (before each statement's reduction) posts `source_failure_site`.
- [x] 3 partition: `_failure_site_frontiers` adds a key to
  `_control_partition_keys`.
- [x] 4 evaporation: `evaporate_unrolled_loops` keeps loops that hold a site.
- [x] 5 control block: `FailureExitBlock`, placed by
  `_place_failure_exit_blocks` / `place_failure_exits_after_regions`.
- [x] 6 SSA: `_lower_failure_exit` (ExceptionPending + CondBr) and
  `_join_failure_exits` (Phi at the end of the guard's true arm, bound to
  each carried alias's arm id). Exits go straight to the guard arm's end:
  an SSA branch out of a loop body needs no break edge and skips `else`.
- [x] 7 C backend: `turing_validation_error_pending() != 0`, row
  `c_failure_exit`. LLVM follows the LLVM exception port (review a14927e5),
  which is not on this branch's base.
- [x] 8 audit: `failure-site-provenance`, `c-failure-exit-provenance`.

Known limits: AugAssign (`x //= n`) has no operator-kind row and records no
site; effects in the same statement after the failing operator run before
the test; a guard whose merge carries results, sequences, fields or optional
glyphs refuses.
