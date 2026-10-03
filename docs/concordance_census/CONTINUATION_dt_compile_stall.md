# CONTINUATION: native dt-system compile stall (dt_system_over -> run_superstep callsite)

Repro: scratchpad `lower_two_pieces_spans.py` (air + pool b1, `lowered_system`,
C backend).  Probe copy: scratchpad `stall_probe_tdesc.py` (monkeypatches
`gds._tensor_descriptor`/`_tensor_descriptor_rule` and
`IdentityPage.history` from the script; no src edits) prints call counts,
distinct (function, node) pairs, recursion depth, and the mean `columns`
length each `history` scan walks.

## 2026-10-03 -- instrumentation launched (11:16)

## 2026-10-03 -- finding 1: the stall is a revision churn on `shape_transformation_state`, not new work

Run A (11:16, killed at +6 min by Windows pid).  The run reaches
`callsite-progress ... caller=dt_system_over callee=run_superstep` at +52 s.
From then on (probe lines every 20,000 `_tensor_descriptor` calls):

| t | td calls | distinct (fn, node) | `shape_transformation_state` columns / cells | mean columns per `history` scan |
|---|---|---|---|---|
| +62 s | 60k | 4112 | 1115 / 16,117 | 49 |
| +108 s | 80k | 4112 | 2665 / 36,025 | 208 |
| +211 s | 100k | 4112 | 4237 / 56,013 | 437 |
| +369 s | 120k | 4112 | 5797 / 75,177 | 705 |

- No new node is visited after +62 s; the same (step_0, 28/20/10) nodes are
  re-queried thousands of times, recursion depth stays <= 26.
- Every 20k re-queries append ~1,500 columns and ~20k cells to
  `shape_transformation_state`: the same rows are REVISED on every visit
  (a revision is only written when the fact differs from `latest`).
- `IdentityPage.history` walks the page's whole `columns` list, so every
  read gets slower as the churn grows: 20k calls took 10 s, then 46 s,
  103 s, 158 s.  That is the "slow" the dumps saw; the cause is the churn.

## 2026-10-03 -- finding 2: the churned rows; verdict = spinning on the book, amplified by an O(columns) read

Run B (11:23, same probe + per-row dump, killed at +12.5 min by Windows pid).
Top revised rows on `shape_transformation_state` (at td=90k):

    ('step_0', 42) revisions=4239 distinct_facts=5
      ('resolved', ((1,), 'float64', 1, 'static', None), (.., 'step_0', 20, .., 'Pow', 'lhs'))
      ('resolved', ((1,), 'float64', 1, 'static', None), (.., 'step_0', 30, .., 'Pow', 'rhs'))
      ... alternating lhs / rhs ...
    ('step_0', 40) revisions=4217   Mul lhs (28) / rhs (1), alternating
    ('step_0', 41) revisions=4205   Mul lhs (28) / rhs (9), alternating

- The TARGET SHAPE never changes ((1,) float64 static).  What flips is the
  edge_row inside the state fact: `_tensor_descriptor` (gds, the publication
  block after `_tensor_descriptor_rule`) calls `record_shape_transformation`
  once per semantic source, and `record_shape_transformation`
  (identity_concordance.py, the `state_fact = ("resolved", target, edge_row)`
  block) revises the state whenever `latest != state_fact` -- so a binary
  operator's two agreeing edges overwrite each other's projection on every
  visit.  Two writers that agree on the fact but each claim the projection.
- Distinct (fn, node) frozen at 4101; 140k re-queries in 10 min, each one
  appending 2 revisions per binary node.  The book grows without bound and
  `IdentityPage.history` walks the whole page `columns` list (7,351 by
  +595 s), so each read is O(total revisions): 20k queries took 10 s, then
  31, 42, 61, 60, 89, 114, 117 s.
- Verdict: no progress counter moves and the book grows per visit -- spinning
  on book churn, with the per-read cost growing linearly in the churn.

## 2026-10-03 -- finding 3: bisect

Worktree `Temp\wtd`, each commit's own tree + the a2020e0b `ssa.py`
`__setstate__` fix applied (fuzzy; the pieces do not unpickle without it),
pieces from the main checkout.  Verdict per commit (scratchpad
`bisect_step.sh`): CHURN = `shape_transformation_state` passes 1,500 columns;
PASS = a progress line after the run_superstep callsite.

| commit | date | verdict |
|---|---|---|
| fa4b127a | 09-28 | PASS (callsite in 8 s, 241k descriptor queries) |
| 42a689a2 | 09-29 | PASS (callsite in 24 s) |
| 9cfd81a4 | 09-29 | CHURN at +70 s |
| e8d3c4d5, 7a678092, 4249738b, baded93f | 09-29 .. 10-03 | CHURN |
| 5787ebf9 | 09-29 | n/a (ExtractionContractError before the callsite) |

First bad: **9cfd81a4** "Bind shape-transformation names before first use in
_tensor_descriptor".  It did not write the churn; it switched it on.  Before
it, the `_tensor_descriptor` preamble raised UnboundLocalError on every call,
the except set `row = None`, and the whole publication block (one
`record_shape_transformation` per semantic source) never ran.  The writer
itself came in 42a689a2.  Today's commits (095a3c0c .. c03ae4e6) are not the
cause.  Note the re-query volume (~240k descriptor queries for this
callsite) is the same at the good commits; it was cheap there because the
queries wrote nothing.  Worktree removed.

## 2026-10-03 -- the fix (identity_concordance.py only)

1. Root cause, `record_shape_transformation` (identity_concordance.py, the
   state-row block after the dependents post): the state row
   `(scope, id)` -> `("resolved", target, edge_row)` was revised whenever the
   fact differed, and the fact names the edge.  `_tensor_descriptor` posts one
   edge per semantic source with the same answer, so a multi-source target's
   agreeing edges overwrote each other's projection on every query.  Rule now:
   one projection per identity, many edges into it.  When the incumbent is
   resolved to the SAME target state through a different, still-live edge into
   this identity (the edge page's latest fact for it ends in the same target),
   the new edge corroborates it: the edge and its dependents row are posted
   (with the edge's own source cell) and the state row is not revised.  A
   different target state still revises and withdraws as before.
2. Data structure, `IdentityPage`: `history` walked every column the page had
   ever had (O(all revisions on the page) per read).  The page now keeps
   `column_positions` and `row_columns`, maintained by `_stamp` (the only cell
   writer; checked with git grep), so `history` reads one row's columns in the
   same `columns` order.  Same cells, same order, same semantics; pages
   pickled before the index are reindexed in `__setstate__`/`__post_init__`.
- Targeted tests: test_sequence_contract_concordance.py +
  test_callsite_formal_shape_concordance.py -> 3 failed / 27 passed, the SAME
  3 failures on a clean HEAD worktree (822a753b; int32/int64 feed dtype and
  projection enrichment) -- pre-existing.

## 2026-10-03 -- after the fix: the callsite clears

Run C (11:52, working tree with the fix, probe on): reaches
dt_system_over -> run_superstep at +59 s and clears it at about +89 s
(the next callsite, step_with_dt_control_used -> update_dt_max, is printed
at +89 s).  Before the fix it was still inside the callsite after 30 min.
About 233k descriptor queries in roughly 30 s, the same query volume as the
good commits fa4b127a/42a689a2.  `shape_transformation_state` is no longer
among the four largest pages, and the mean `history` scan is 10 columns
(it was 1,013 and growing).  The lowering continues; result below.

## 2026-10-03 -- run C ends at the next wall (+313 s), in another lane's in-flight code

Run C (working tree, which carries other agents' uncommitted edits) raises at
+313 s, in `function=publish_window` (depth 1), at
`propagate-callsite-planner-specializations`:

    _propagate_callsite_planner_specializations (gds 17286)
      -> _post_planner_specialization -> _post_if_changed -> IdentityBook.post
    ConcordanceRefusal: post planner_specialization
      (('lexical_reads:_propose_dt_pen', 0), 'distribution'): REVISE without
      a changed source; every Derived cell is stamped at or before the row's
      previous revision (3259961) and the cell set is the previous revision's

- So the post proposes a DIFFERENT fact for `_propose_dt_pen`'s formal
  `distribution` from the same cells, with nothing changed: two writers
  disagree on that row.  The cells are argument/formal identity cells
  (`_argument_identity_cells`, `_formal_identity_cells`, and now
  `role_cells`), not shape-state cells.
- This function holds the other agent's uncommitted `role_cells` /
  `declared_argument_role` edit (git diff gds ~17158-17216).  Next: run the
  same lowering on a clean HEAD worktree + this fix only, to separate the two.

## 2026-10-03 -- the next wall is on clean HEAD too (not the WIP, not this fix's rows)

Run on a clean HEAD worktree (822a753b) + this identity_concordance.py
only: same ConcordanceRefusal, same row, at +348 s.  Hooking
`_post_if_changed` from the probe (scratchpad `stall_probe_tdesc.py`,
PROBE_HOOK_POST=1) shows the two claims on
`(('lexical_reads:_propose_dt_pen', 0), 'distribution')`:

- col 1: `Unresolved(specialization_dynamic_argument)` DERIVED from
  {canonical_value (step_with_dt_control_used, 8), proven_literal
  (step_with_dt_control_used, 8)}
- next round: `SpecializationFact(value=None, LITERAL)` from the SAME two
  cells, neither restamped -> refused (correctly: the cause of the change is
  not a cited cell).
- The verdict flips because `_source_static_value(caller, 8)` reads graph
  state (node type Constant, `planner_specializations` graph dict), which
  changed between rounds without a cell on the row.  Observing node 8's
  type per round now (run 12:14).
- The shape-state rows this fix touches are not among the cited cells.

## 2026-10-03 -- the next wall's chain, observed, and its fix

Hooking `_source_static_value(step_with_dt_control_used, 8)` (run 12:14):
node 8 stays `Input` (`binding_name='distribution'`) throughout.  It returns
False while the caller's `planner_specializations` graph dict is
['failures', 'retries'], and True once the same pass has added
'distribution' (the pass writes `callee.G.graph["planner_specializations"]
[parameter]` right after posting step_with_dt_control_used's own
`planner_specialization` row).  So the cause of `_propose_dt_pen`'s flip is
step_with_dt_control_used's `planner_specialization` row for
`distribution`, and `_argument_identity_cells` did not cite it.

Fix (gds `_argument_identity_cells`, outside the catalogue-fold read site):
for an `Input` argument, also cite the caller's
`planner_specialization` row `(lexical_read_scope, binding_name)` when the
book has one.  The callee row's flip then derives from a changed cell (the
caller row's new column) and is admitted as a revision with its edge.
Applied to the main tree and to the HEAD worktree; rerun 12:2x on HEAD +
both fixes.

## 2026-10-03 -- HEAD + both fixes: the deployment stage completes

Run 12:21 (clean HEAD worktree + the two fixes): `propagate-callsite-planner-
specializations` passes, deployment-select ends at +496 s (`stage=deployment
state=end` for `<source-catalogue>`), `ssa-source: instantiating complete
control/operator deployment`, then callsite-plan (dt_system_over ->
run_superstep -> step_with_dt_control_used ...) from +533 s.  2.0M
descriptor queries by +510 s, mean history scan 10 columns.
- +969 s: all 32 AOT shells prepared; `ssa-source: lowering full planned
  source to repository SSA` begins (30 planned shells); ABI settlement
  round 1 at +969 s with 275 contracts, round 2 at +1092 s with 406.
  Progress is real; the run continues past the 30-minute mark per the rule
  (record where it is and keep going).

## 2026-10-03 -- HEAD + both fixes: result (NOT full-native; next wall is in precompile_to_ssa)

Run 12:21 ended at +1323 s (22 min) with a raise during SSA control
lowering of `step_with_dt_control_used__specialized_*` (region 99, inside a
while loop):

    fortran_c_shell._class_surface_ssa_program (19502)
      -> precompile_to_ssa.lower_control_sections_to_ssa (16283)
      -> ... lower_while (10427) -> ... _carried_field_arm (4474) -> book.post
    ConcordanceRefusal: post ssa_field_version
      (('lexical_reads:step_with_dt_control_used|fork', 5),
       Ref('reducer_field_state', (((..step_with_dt_control_used, 0), 'ingestion'),
           <id>, 'unresolved_report'), 0)):
      REVISE without a changed source

- That is `Metrics.unresolved_report`, the field the frontier memory already
  lists as open (table storage, no physical ABI columns).  precompile_to_ssa.py
  belongs to another lane; not touched here.
- Peak private memory of the run: about 6.8 GB (2 GB+ more than earlier
  runs; the probe's per-call counters add little).  The contract was the
  program's own (`lowered_system` -> `dt_system_contract`, with the full-native
  execution file), not None.
- Targeted tests, HEAD + both fixes: test_pruned_loop_return,
  test_shell_reference_tables, test_sequence_contract_concordance,
  test_callsite_formal_shape_concordance: 5 failed / 38 passed.  The same 5
  fail on clean HEAD (3 from the earlier run, 2 shell-table tests re-run on
  clean HEAD 12:5x).  Worktree removed.

Timeline for this lowering: before the fix, the dt_system_over ->
run_superstep callsite never finished (30 min and more, still stuck).  After
the fix it clears in about 30 s, deployment ends at +496 s, and SSA lowering
runs until the precompile_to_ssa wall at +1323 s.
