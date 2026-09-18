# Continuation: full-native DT compiler frontier

## Consolidated implementation sequence — September 7

Start with `PATCH_SEQUENCE_2026-09-07_ALL_19.md`, supported by
`REPAIR_ACTIONS_2026-09-07_COMPLETE_FRONTIER.md`. These specify the complete
repair series, cover all 19 baseline findings exactly once, and include the
non-formal defects and optional input domain required for native parity.
They are specifications, not evidence that the repairs are implemented.

Additional working edits in the mapping materialization/scheduling paths are
unfinished. `tests/test_native_mapping_copy_effects.py` currently fails:
lowering and native compilation succeed, but the public return is empty
instead of the four requested lookup values. The strict gate currently misses
these structural output shortfalls. Preserve this regression and repair the
source-position lookup/return path; do not weaken or mark the test xfail.
The 19-finding diagnostic below predates these unfinished mapping edits.

## Latest repair status: checked replacement prerequisite

See `REPAIRS_2026-09-07_KEYED_REPLACEMENT.md`. Existing conditional sequence
replacement now checks source/destination capacity before writing, copies rows
directly, and publishes length last. Self-aliasing and failed replacement
preserve contents. Focused native/caller tests4 passed15.68s; broader42 passed,
1 failed41.39s at an append fixture's empty-C-shim assertion (retained).

Fresh source diagnostic `build/keyed_replace_full_formals_20260907.log`:
**19 formals, zero undefined operands/unresolved calls**, terminal exit1 at
strict gate. The authoritative saved artifacts contain this run. All four
scalar publications remain, including1482 -> Cast9564 ->593; early read1378
is unchanged. No full native build/parity, live jobs, authored DT edit, commit,
or push.

Keyed publication is still open. The floor branch lacks the dict constructor
copy and both keyed stores494/495. The table resolver refuses materializer493
without a lexical binding name, and chained store aliases need resolution.
Fix recognition together with lexical effect scheduling: the current fallback
can emit unscheduled stores at entry. Then bind the field's actual triplet and
capacity, copy at assignment, and preserve subsequent alias behavior. Do not
publish the current unpopulated RHS arena or remove496/497. The repair notes
give the exact saved-SSA evidence and next bounded source regression.

## Prior repair status: scalar return publication

See `REPAIRS_2026-09-06_RECORD_RETURN_STATE.md`. The fresh source diagnostic
`build/record_return_publication_full_formals_20260906.log` completed at the
strict gate with **19 formals, zero undefined operands/unresolved calls**.
The authoritative diagnostic pickles and formals.json contain this run.

Four checked scalar versions now reach the physical step return: div_inf 553,
mass_err 573, dt_limit 555, and hard_failure through
`1482 = Phi(9564)`, `9564 = Cast(593)` on return_control_next. The early bool
region at loop_exit still consumes 1378. Final publication occurs after
reachability and signature cleanup; repeating it makes zero changes.
This preserves the return slot identities and intermediate Boolean-ledger ABI.

Focused tests: 18 passed, 1 xfailed in 26.38s, including native execution of
both scalar dtype variants through the actual publication pass. The authored
child-record test still hits existing unaccounted ABI formals and is not an
end-to-end proof. A separately reproduced static-reference cache eviction
crash is repaired by checking cached node liveness and reference identity.

Next: keyed error_channels ownership/copy assignment and authored child-record
ABI recovery/native proof. Keep duplicate effect Phis 496/497 until those
returned fields are proven. Optional presence storage remains necessary for
the real dt_min=None/dt_max=None validator inputs and will reopen reachability
paths and their producers. No full native build/parity, no live jobs, no
authored DT changes, commit, or push.

## Prior repair status: effect ordering

See `REPAIRS_2026-09-06_EFFECT_ORDER.md`. The fresh post-repair full diagnostic
is **19 formals, zero undefined operands/unresolved calls**, terminal exit 1
at the provenance gate. Log: `build/effect_order_full_formals_20260906.log`.
The authoritative diagnostic pickles and formals.json have been regenerated.

The linked hard-failure test now reads incoming field 1378 instead of later
Phi 593, surviving reason appends precede the truth query, and accept calls
follow the rejection continue guard. On-demand prerequisite scheduling,
resident-sequence effect constraints, and exclusion of proven-pure regions
from obsolete hierarchy ranks fix that ordering. Non-loop clear effects are
retained. Contradictory scheduling cycles now diagnose rather than fall back.

The controller's None-to-number feed substitution was unsound for `is None`.
Numeric record fields now reject absent inputs without an optional-presence
ABI; implementing that ABI remains necessary. Scalar record writes and a
call-only for-loop repro still drop their guards upstream and are explicit
expected failures. Return-layout/ledger wiring remains the next record step;
do not delete 496/497 until the returned fields are proven. No full native
build or parity run, no live jobs, no commit or push.

## Prior repair status: reachability

The user authorized implementation. See `REPAIRS_2026-09-06_SECOND_OPINION.md`.
Conservative CFG reachability now runs after late source literals and before
paired caller/callee signature pruning. The fresh full driver completed with
**19 unaccounted formals, zero undefined operands, zero unresolved calls**.
It exited 1 at the strict provenance gate; no full native build was launched.
The diagnostic process is terminal. Log: `build/reachability_full_formals_20260906.log`.

`build/full_formal_diagnostic/{formals.json,repository-ssa.pkl,resolved-process-graph.pkl}`
now contains this fresh run. The old 23-finding ids and second-opinion review
below are historical; do not cross-reference those ids against the new files.
The current groups are run_superstep 6, step 10, and the three singleton helpers.
The replay on the old saved SSA removed five formals (23 -> 18); that replay
and this fresh source run, which includes the existing experimental field
ledger, are distinct measurements. Full native parity is still unproved.

The C backend takes scalar formals through storage pointers; scalar SSA dtype
does not prove a by-value ABI. Correct advance-result storage identity remains
unverified. Other review findings remain a repair frontier, not accepted fixes.

This is the concise resumption document for the next agent. The objective is
one Python-authored vehicle validator compiled end to end as one native
program. Python callbacks, validator-specific native insertions, source edits
that evade compiler defects, and after-the-fact wrapper orchestration do not
satisfy it.

## Repository and process state

- Repository: `C:\dev\Powershell\turing`.
- Branch: `codex/recursive-reduction-bridge`.
- Last pushed commit: `d50015b4`.
- The worktree is intentionally dirty with accumulated compiler work. Do not
  reset, stash, or replace it. No work described here is committed or pushed.
- No heavy build, parity, or timing process is running.
- Read `AGENTS.md` and `TEST_BASELINE_AND_HAZARDS.md` before testing. Run only
  one heavy process at a time.

## Historical gate before the second-opinion repairs

The authoritative full driver is `build/diagnose_full_formals.py`; run it from
the repository root with `PYTHONPATH=.`. It writes its evidence under
`build/full_formal_diagnostic/`.

The latest complete full driver run is now `FORMAL-DIAGNOSTIC 23`, with
`undefined_operands=0`. This is a strict native provenance count: every one of
the 23 is a local Python computation or state transition that is still being
misrepresented as a native function input. Zero is required. Never allow-list
or ignore them.

The exact current groups are:

- `run_superstep` (6): boundary sequence construction; `callable(hint)`;
  `hint()`; keyed `.get`; `first.unresolved_report`; slice `[1:]`.
- `step_with_dt_control_used` (14): one flag BoolOp; the two record-field
  assignments to `metrics.error_channels` and `metrics.hard_failure`;
  `len(floor_reasons)`; `tuple(reasons)`; six formatted/sequence values; two
  attempt-log mapping rows; and the final loop-exit BoolOp.
- `_no_exchange_observed` (1): guarded keyed load.
- `_propose_dt_pen` (1): starred generator reduction.
- `coerce_metrics` (1): dictionary-comprehension row.

The last run's generated ids were `run_superstep`
`[284,174,57,240,124,126]`, step
`[559,516,517,422,421,444,452,470,478,248,488,518,408,207]`, and singleton
ids 11, 49, and 0. Regenerated graphs can renumber them; identify findings by
source role.

No full native build, frame parity, or performance run is justified while this
count is nonzero. The old trace3 executable predates these compiler changes and
does not establish current correctness.

## Accepted general compiler changes

The work accepted in this latest segment is general compiler behavior, with no
DT symbol names or source ids embedded in it.

### Exact conditional Phi provenance at graph ingestion

`topological_reducer.py` now puts the exact pre-conditional value in
`initial_value_id` on ordinary reducer-authored Phi nodes. Canonical relabeling
updates that receipt. `_ordinary_conditional_control_programs` in
`glsl_deployment_strategy.py` consumes those exact Phi parents instead of
reconstructing a carried value from flat identity history.

This fixed the nested `dt_cap` recurrence at its source. The real full compile
now directly emits:

```
if_merge.1: Phi [196,178] -> 214
if_merge:   Phi [214,178] -> 215
while:      Phi [215,250] -> 368
```

The former post-link `_repair_late_loop_initial_conditional_phis` pass and its
historical test fixture were removed. The 23 count is a fresh whole-program
result, not an in-memory repair estimate.

### Earlier accepted compiler work still in this dirty tree

The strict formal-provenance gate, source-faithful snapshot/restore methods,
optional tuple presence and member representation, conditional aggregate
members, loop-current scalar expression recovery, tensor `all`, terminal
`continue` recurrence, and call-result record plumbing are all general
compiler work. Their focused native/eager regressions are recorded in
`HANDOFF_2026-09-06.md`.

None of those changes license Python execution inside the native product. The
formal gate exists to catch exactly that kind of missing native computation.

## Experimental work: do not call this accepted yet

There is a lean record-field control-state experiment in
`topological_reducer.py`:

- explicit `GetAttr` records the current field value;
- `SetAttr` records its RHS as the new value and retains a separate effect
  identity;
- each `if` arm gets its own field-state ledger;
- a field already projected before the branch can receive a reducer-authored
  Phi with `record_field_state`, exact `initial_value_id`, and the two arm
  values.

The focused reducer regression
`test_conditional_attribute_write_phis_rhs_with_existing_projection` passes,
as do the nearby attribute and nested-conditional tests. The real bounded step
repro also lowers and was saved as `build/step_field_state.pkl`.

This is not accepted because a graph-level field Phi is insufficient unless
the linked physical record descriptor and final return layout use that selected
field version. The next agent must prove that both branches return the correct
record field natively. Do not remove formals 516/517 merely because their old
effect-token Phis appear unused.

An earlier, broader experiment that synthesized missing initial `GetAttr`
values was completely backed out. It created undefined numerical operands and
must not be restored. A field with no proven incoming projection must remain a
loud diagnostic.

## Record-field evidence and next bounded test

In step, the `coerce_metrics(metrics)` call publishes 14 physical Metrics
fields. The call source is 437 and its forwarded caller values were
1344..1357. The final returned Metrics surface was 1312 with fields 1447..1460.
`error_channels` occupies the length/keys/values triplet 1353..1355;
`hard_failure` is 1356.

Source 516 assigns the keyed `error_channels` arena through an `IndexedStore`.
Source 517 assigns scalar `hard_failure = False`. The old fabricated inputs are
effect identities, not physical field values. Correct lowering must create new
field versions on the relevant control edge and make the returned record use
those versions.

First inspect `build/step_field_state.pkl` compactly: print formal parity, the
Metrics `record_return_layouts`, definitions and uses for the two field
surfaces, and only Phis touching them. Then add a small native/eager regression
whose child returns a record, whose caller conditionally changes one scalar
field, and whose caller returns the whole record. Test both branch outcomes.

For scalar fields, linking can version the record descriptor to the selected
Phi result and alias later projections to it. For `error_channels`, preserve the
actual resident keyed arena and its store. `_promote_conditional_sequence_aliases`
currently handles list/bytes/bytearray only; do not pretend a dictionary handle
is a scalar Phi.

## Known regression warning

A broader selected test run currently has three red existing record tests:

- `test_record_field_assignment_is_a_real_inout_value`
- `test_record_field_storage_identity_crosses_the_call_frame`
- `test_returned_record_fields_feed_structural_call_argument`

The experimental ledger was removed and reintroduced in narrower form without
changing those failures. Treat them as part of the accumulated dirty-tree
frontier; do not claim a broad green gate until they pass or their pre-existing
status is established from saved evidence.

## Recommended order

1. Prove or reject the lean record-field experiment with the whole-record
   native/eager branch regression. Repair physical record versioning generally.
2. Re-run the full formal driver after a material fix. Record the exact count;
   zero remains the only acceptable end state.
3. Attach `_no_exchange_observed`'s keyed load to its already compiled
   membership branch instead of introducing a second control hierarchy.
4. Add general native producers for tuple, slice, comprehension/generator,
   mapping-row, and formatted sequence values.
5. Lower the remaining BoolOps after their exact operands are resident and
   preserve Python short-circuit control where evaluation is guarded.
6. At zero unaccounted formals, run one detached native build, then frame
   parity. Measure optimized equivalent performance only after parity passes.

The compiler has moved downstream of the earlier “it lowers” state into a
stricter correctness audit. That is real progress rather than a regression:
previous lowering silently promoted missing local values into ABI inputs. The
23 findings expose those unsound promotions. The system is not yet a verified
native validator.

## Second-opinion review brief

Please begin as a reviewer, without editing the compiler. Independently answer
these questions from the saved graph and SSA:

1. Does `formals.json` account for every non-authored formal, and does it omit
   any equivalent fabricated value hidden behind call or record forwarding?
2. Are the 23 findings truly missing native producers, or are any existing
   exact graph/SSA definitions disconnected by aliasing, placement, or ABI
   publication?
3. For step values 516 and 517, is record-field SSA versioning the correct
   model? Trace each authored write through the call-return Metrics surface and
   final `Ret` before recommending deletion, aliasing, or a new operator.
4. Does the experimental `record_field_state` ledger preserve Python semantics
   for nested conditionals, terminal arms, loops, aliasing, and mutable keyed
   fields? Identify any case in which receiver identity alone is an unsafe key.
5. Can the remaining producer problems be grouped into a smaller number of
   general compiler mechanisms without adding DT-specific names, ids, source
   rewrites, Python callbacks, or fabricated ABI inputs?

Primary evidence:

- `build/full_formal_diagnostic/formals.json`: the exact 23 findings, their
  consumers, authored parameter lists, and current accounting.
- `build/full_formal_diagnostic/repository-ssa.pkl`: failed whole-program SSA.
- `build/full_formal_diagnostic/resolved-process-graph.pkl`: corresponding
  resolved graph.
- `build/step_field_state.pkl`: bounded step lowering with the experimental
  field ledger.
- `src/common/dt_system/dt_controller.py`: authored functions at
  `_no_exchange_observed` line 99, `_propose_dt_pen` line 173,
  `step_with_dt_control_used` line 207, and `run_superstep` line 532.
- `src/common/dt_system/dt_scaler.py`: `coerce_metrics` line 62.

Use `build/diagnose_full_formals.py` only if the saved artifacts cannot answer
the review question; a fresh run takes several minutes. Do not start a native
build. The useful deliverable is a written disagreement or confirmation for
each category, especially record-field state, plus the smallest general order
of repairs that can drive 23 to zero.
