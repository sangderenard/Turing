2026-11-12 Protected failure exit (MEASURED, branch claude/protected-failure-exit,
stacked on claude/specialized-guard-projection 51609ab2):

Wrong record (traced in 20261112T193712Z_claude-specialized-guard-projection.md):
`try: for i in range(4): total = total + 12 // (i - n) ... else: total += 7`
returned 179 for Python's 150 at n = 0. The C division arm filled the
native error slot and continued with a placeholder 0. The constant loop was
evaporated into one straight-line planned region, and only the protected
body's trailing ExceptionTake read the slot.

Rule (owner decision B: the exit lives in the control builder, not inside
planned regions). Plan and records:
docs/compiler_convergence/protected_failure_exit_plan_2026-11-12.md.
- Ingestion's guard declaration reaches the book: `source_exception_guard`.
- The reducer records a `source_failure_site` for each statement inside a
  protected guard that holds an operator whose SSA kind has failure arms,
  with the (name, value) bindings visible before the statement, the loops
  inside the guard that hold it, and the guard.
- The work after a site inside its guard is a separate partition; a loop
  holding a site is not evaporated.
- `FailureExitBlock` follows the site's region. It lowers to
  `ExceptionPending` (`turing_validation_error_pending() != 0`) and a CondBr
  straight to the end of the guard's true arm. A `failure_join` block there
  Phis each carried name's normal value with its site version, and binds it
  to the alias's arm id, so the guard's own merge is unchanged.
- Audit: `failure-site-provenance`, `c-failure-exit-provenance`.

Measured (one process each, `pytest tests/test_source_c_exceptions.py
tests/test_linked_accepted_c_emission_owner.py`):
- Base 51609ab2 (guard projection): 15 failed, 32 passed.
- This branch, first run: 15 failed, 32 passed, but a different set:
  loop_inside_try [0] [2] [5] and worker_has_loop_inside_try newly passed;
  returns_from_try_and_handler_run_finally [0] [3] and
  worker_return_runs_observable_finally_store [0] [3] newly failed with
  "control block has no activity contract: FailureExitBlock"
  (`_install_external_reference_calls`).
- Fixed: the block's activity position is its operator region's position,
  and `_control_block_consumes_values` counts its binding ids. Those four,
  the three loop cases and the two new tests then passed (10 of 10).
- Final suite, same command (50 tests: the 47 plus three new): 11 failed,
  39 passed. Every failure also fails on the base: try_except_else_finally
  _in_loop x3 and while_loop_handlers x4 (scalar-physical-layout-unproven),
  loop_and_try_in_an_authored_worker, protected_ordinary_call (DID NOT
  RAISE), standalone_raise_dynamic_message, linked_split_c_emission.
  Newly passing against the base: loop_inside_try [0] [2] [5],
  worker_has_loop_inside_try, and the new failure_skips_later_statements
  [0] [3] and failure_site_records tests.

Limits: AugAssign (`x //= n`) has no operator-kind row and records no site;
an effect later in the same statement as the failing operator runs before
the test; a guard whose merge carries results, sequences, fields or optional
glyphs refuses. LLVM `ExceptionPending` waits for the LLVM exception port
(review a14927e5), which is not on this base. The varying upstream
refusals (`carried-name-arm-missing`, an occasional single final-concordance
finding) are untouched.
