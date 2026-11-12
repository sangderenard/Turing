2026-11-12 A structurally decided If no longer guards its kept arm (MEASURED, branch claude/specialized-guard-projection from main 488f8de5):

Wrong record, traced on `test_exception_handler_does_not_catch_base_exception_itself`
(the full chain is in 20261112T161424Z_claude-llvm-exception-records.md on
claude/llvm-exception-records):
- Exception ingestion spells each handler test as
  `h = not False; if h: h = __turing_exception_active(e)`.
- `_fold_callsite_structural_values` proves that If constant-true and selects
  its body. It posts `source_control_specialization_concordance`
  (function, If) = {selected 'body', rejected 'orelse',
  proof 'structural_constant_predicate'}, lists the If in
  `structurally_specialized_conditional_node_ids`, and removes it.
- `_ordinary_conditional_control_programs` builds no block for it, by design.
- `control_activity.memberships` still projected the body's
  `exception_active` as guarded by that If, through its
  retained-source-control branch.
- `_install_external_reference_calls` found no block for the guard and
  refused with "dispatcher operation's control owner is not retained".

Rule, in `control_activity.memberships`:
- An owner this graph's fold decided guards nothing in its selected arm or
  its test. "Decided" means absent from the graph and listed in the graph's
  `structurally_specialized_conditional_node_ids`; the graph-local receipt
  matters because several copies of one function graph coexist, and one
  still held the live If after another decided it. The node's enclosing
  owners are already in its activity fact.
- A resident node in the rejected arm refuses.
- A receipt with no specialization row naming the selected arm refuses.

Measured:
- `test_exception_handler_does_not_catch_base_exception_itself`: 3 of 3
  passed. On clean main it passed about 1 run in 15.
- Same command, `pytest tests/test_source_c_exceptions.py
  tests/test_linked_accepted_c_emission_owner.py`, one process per tree:
  - This branch: 15 failed, 32 passed.
  - Clean main 488f8de5: 23 failed, 24 passed.
  - Every failure on this branch also fails on main.
  - Now passing: base-exception handler, raise-from-None,
    worker-catches-its-own-error x2, worker-return-runs-finally x2,
    zero-iteration-loop-inside-try x2.
  - Main's counts vary run to run (34, 23 earlier today), so the exact
    numbers are one sample each.

EXPOSED, not caused — a wrong answer the refusal used to hide:
- `test_loop_inside_try_exits_to_handler_and_finally[0]` now compiles and
  returns 179 where Python returns 150. On main it refuses at the
  dispatcher owner.
- 179 = 12 + 6 + 4 + 7 + 50 + 100: the `for` loop did not leave on the
  ZeroDivisionError at i = 0. It ran the other iterations and its `else`,
  then the handler and `finally`.
- In this program the fold decided exactly one If (control 31, the
  constant-true handler test). The guard this change drops covers only
  `__turing_exception_16 = __turing_exception_active(e)` inside it. The loop
  lives in `root__root__planned_region_0`, called under "no error active".
  So the lost early exit is in the loop region's failure arm, not in this
  projection.
- Not traced. Four further lowerings of the same program stopped earlier at
  `carried-name-arm-missing` (`root.sequence[2].body.sequence[4|6].body`),
  another run-to-run varying refusal upstream. The node numbering of the
  exception operations changes between runs; the identity-ordered
  iteration behind that is not found.

Other remaining failures, unchanged from main: `scalar-physical-layout-unproven`
in C emission (`try_except_else_finally_in_loop`, `while_loop_handlers...`), and
`test_protected_ordinary_call_requires_a_failure_return_contract` DID NOT RAISE.
