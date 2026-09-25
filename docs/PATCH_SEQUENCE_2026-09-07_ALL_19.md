# One patch sequence for all19 baseline findings

**2026-09-14 continuation:** see
[Native managed tire and dt controller](CONTINUATION_2026-09-14_NATIVE_DT.md).
The v167 raw-snapshot rewrite is repaired, the current replay passes its SSA
shape gate, and the rebuilt `-O0` one-step artifact matches all 48 buffers.
The continuation records the full-window verification separately.

## 2026-09-08 in/out scalar return snapshots

A scalar field may be updated through caller-owned storage and also returned
inside a new record. The settled return ABI now selects the unique latest
dominating `ssa_inout_write_version` for the returned value while retaining the
resident formal on ambiguity or equal priority. Return operands, record layouts,
record descriptors, and caller result contracts change together, with a
provenance receipt for each accepted replacement.

Replay v111 changes managed advance's two Metrics velocity positions from the
mutable material slot 194 to snapshot 295 and retains zero structural findings.
The focused batch passed 5 tests, followed by a 2-test placement regression.
One explicit `-O0` native build completed with 78 buffers. Its generated call
now writes the snapshot to controller slot 1741 and all three Metrics consumers
read that slot. One-frame parity remains 54/78, so this closes a real aliasing
defect but does not move the measured parity count. The next frontier is inside
the controller calculation after the now-correct snapshot handoff.

## 2026-09-08 returned-record call-frame settlement

The managed controller frontier exposed an exact record handoff that still
used the pre-call `Metrics` fields. Call-result aliases are now resolved to the
late returned-record descriptor before consumer frames are reconciled.
Repeated callee output identities settle by aggregate position, with the first
resident retained on equal priority. The descriptor, call-table receipt, and
emitted call are updated together and record their provenance. Exact returned
record descriptors also remain incumbent over equivalent later read-only
projections, including nested sequence handles.

Saved pre-frame replay `build/patch_sequence_replay_v107` converged in six
frame rounds and three result-type rounds with zero incompatible contracts and
zero structural findings. In the real controller, `coerce_metrics`,
`_propose_dt_pen`, and `_apply_energy_sidechain` now receive `max_vel` through
produced aggregate slot 1741 instead of the zero-initialized field 1654.
Focused record/call tests passed 3 tests in 4.66 seconds. A later explicit
`-O0` v107 build established the prior 54/78 one-frame parity baseline.

This is the consolidated implementation sequence. Apply the complete series
as one coordinated task, then debug and validate the integrated result. Shared
mechanisms are implemented once, rather than making19 overlapping fixes to the
same compiler paths. The coverage ledger below maps every finding exactly once.
This specification is not a claim that these patches have already been applied.

Detailed evidence, contracts, and tests:
`REPAIR_ACTIONS_2026-09-07_COMPLETE_FRONTIER.md`.

**Frontier checkpoint (2026-09-07, replay v24):** explicit planned-region
capture/output ownership and write-only scalar-field provenance remove the
emergent `balloon_tire_managed_advance` `%182` formal collision. The current
strict frontier is 18 findings: two formal groups, one optional merge, thirteen
dominance findings, and two call-result conflicts. The full sequence remains
open.

**Frontier checkpoint (2026-09-07, fresh v25):** the managed tire's dt-limit
hint is now a total scalar contract: positive values pass through and `0.0`
means inactive, matching the controller's existing finite-positive admission
rule. Fresh extraction removes the sole optional-merge finding. The current
strict frontier is 17 findings: two formal groups, thirteen dominance findings,
and two call-result conflicts.

**Frontier checkpoint (2026-09-07, fresh v26):** the managed tire now returns
its exact `advanced_dt` and a finite neutral `dt_limit`, removing both
`NoneValue`/float64 call-result conflicts. The current strict frontier is 15
findings: two formal groups and thirteen dominance findings.

**Frontier checkpoint (2026-09-07, replay v27):** predicated `continue` edges
now complete each loop-carried binding at their control join and again at the
latch. An authored update wins only on edges where it is resident; otherwise
the equal-priority incumbent is retained, with both choices recorded on the
Phi. This removes the `%570` and `%280` loop-header dominance findings. The
current strict frontier is 13 findings: two formal groups and eleven
definition-dominance findings, all eleven at the function-exit merge.

**Frontier checkpoint (2026-09-07, replay v28):** literal Boolean while
predicates now select a provenance-marked direct header edge. For `while True`,
the impossible false header edge no longer turns the normal loop exit into a
synthetic function-return path; authored `break` edges still reach that exit.
This removes all eleven remaining definition-dominance findings together. The
strict frontier is now 2 findings, both formal-parity groups: `run_superstep`
and `step_with_dt_control_used`.

**Frontier checkpoint (2026-09-08, fresh v32):** controller diagnostics now
store stable rule/report tokens while numeric measurements remain in `Metrics`,
`error_channels`, and the optional attempt log. The source graph contains zero
`JoinedStr` nodes, controller planned regions fall from 44 to 37, and the final
native gate's anonymous slots fall from 12 to 7. The finding count remains 2
because both affected functions still have other unnamed values. The measured
post-pruning frontier is `run_superstep` `[124, 126]` and
`step_with_dt_control_used` `[428, 347, 429, 477, 458]`.

**Frontier checkpoint (2026-09-08, replay v35):** scalar record-field writes
now publish their `SetAttr` effect identity as the value committed at that
lexical site. The lowering indexes each effect's resident field as its
pre-write incumbent, captures the true and false branch versions separately,
and versions a join when the graph reuses an arm identity as the join identity.
This removes `%347` (`ctrl.clamp_events`) without creating a self-referential
Phi. The converged frame falls from 4,283 to 4,280 formals; the raw structural
surface falls from 18 to 17 unnamed values and remains two formal-parity groups.
The production post-pruning gate was not rerun in this lower-only chunk.

**Frontier checkpoint (2026-09-08, fresh v40):** exact reducer-authored
`record_field_state` Phis now take precedence over flat dotted attribute
histories for the same `(receiver, field)` slot. The flat histories contained
`SetAttr` event IDs and created duplicate carried chains with producerless
incumbents even though the graph already held the correct field-state merge.
This removes `%428` (`metrics.error_channels`) and `%429`
(`metrics.hard_failure`) together. The converged frame falls from 4,280 to
4,274 formals. After production pruning, the controller frontier is now
`[477, 458]`; `run_superstep` remains `[124, 126]`. The full-native gate still
rejects these four anonymous values, so no C compilation or parity is claimed.

**Frontier checkpoint (2026-09-08, fresh v41):** invocation-site call linking
now reconstructs an exact source `BoolOp` even when earlier control lowering
has provisionally exposed that expression as an unaccounted caller formal.
Graph node keys are normalized through their semantic `value_id`; the
placeholder remains incumbent until both ordered operands resolve, then the
materialized `LAnd`/`LOr` takes its exact source identity and retires only that
placeholder. This removes `%458` (`metrics.osc_flag or metrics.stiff_flag`).
The converged frame falls from 4,274 to 4,273 formals, with zero incompatible
result contracts and no dominance finding. The raw unnamed surface is 14
values across two formal-parity groups. After production pruning,
`step_with_dt_control_used` has only `%477`; `run_superstep` remains `%124` and
`%126`. The full-native gate still rejects those three values, so no C
compilation or parity is claimed.

## Patch01 — collection producers and reductions

**Findings: coerce_metrics0, _propose_dt_pen49, run_superstep291.**

**Implementation checkpoint (2026-09-07, replays v21-v22):** comprehension calls
now follow exact PlanCall loop ancestry when persisted AST controls lack source
spans. This places `coerce_metrics`'s `_scalar(channel)` after its projected row
load and removes its definition-dominance finding. Exact defensive mapping
iteration then reuses the declared key/value spans and canonical key token,
removing its three unnamed projected-storage/string values. Broader Patch01
runtime coverage remains open, so this patch is not complete.

- Retain dict-comprehension loops, their row variables, conversions, and keyed
  stores. Connect coerce_metrics's result to its actual receiver/record storage.
- Lower the starred error-ratio generator into a loop-carried max accumulator;
  preserve source comparison order and run the consumer after completion.
- Materialize the boundary filter/set/sort/tuple chain. Fold the proved empty
  baseline input into an actual empty arena; retain the general nonempty path.
- Use one collection schema/iteration mechanism for these three cases.

Touch: reducer, Control IR retention, collection/table lowerers, region producer
and dependency ownership, call/record linkage.

Required tests: empty and multirow comprehensions; missing channel defaults;
generator completion before max; NaN/order cases; unsorted/duplicate/filtered
event boundaries and resulting step sizes.

## Patch02 — snapshots and their lengths

**Findings: step417, step418.**

- Materialize tuple(reasons) at the source statement with its own contents and
  logical length; do not alias the mutable reasons arena.
- Load len(floor_reasons) from that snapshot and preserve it across the following
  clear and branch merge.

Touch: collection materialization, lexical sequence effects/queries, descriptor
carry lowering.

Required tests: snapshot -> clear -> append; both floor outcomes; repeated calls;
violation count equals the number of reasons before clear.

## Patch03 — dictionary producers, field ownership, and duplicate keyed carry

**Finding: step496.**

- Finish the current unnamed dict materializer path: correct schema, one
  constructor owner, copied source contents, and exactly one lexical owner for
  each indexed store. Resolve chained store and same-arena Phi identities.
- Fix direct keyed lookup return operands. The current new native mapping test
  must return all four values instead of silently returning nothing.
- Bind returned record projections to real key/value/length storage and capacity.
- Implement alias-correct mapping field assignment: distinguish object identity
  from the receiver field's binding; preserve aliases to the old object and all
  aliases to the RHS. Use owned arena copy only where equivalent semantics are
  proved; otherwise retain descriptor indirection or equivalent alias-aware state.
- After that proof, replace the duplicate effect-token carry1901 and remove its
  fabricated initial496 through ordinary paired formal pruning.

Touch: `_field_slot_ops`, mapping mutation/query recovery, lexical scheduler,
record/storage identity metadata, caller/callee physical record publication.

Required tests: the new source-level mapping copy test; two writes inside a
conditional; old-field alias; two RHS aliases; post-assignment mutation;
returned helper record; empty/capacity/self-copy/repeated-buffer cases. Inspect
the real floor copy493 and stores494/495 at their source positions.

## Patch04 — scalar versions, edge state, and first observation

**Findings: step497, step554.**

- Distinguish observing a field from assigning it. Seed branch state from the
  actual physical field, including first observation inside an arm.
- Resolve554's dt_limit initial and merge only actual differing versions; do
  not create a producerless read-history Phi.
- Complete loop-header/backedge, break/continue, and terminal-return field state.
  Publish constant terminal assignments as well as conditional ledger Phis.
- Preserve early reads; publish augmented-assignment Add results, including
  clamp_events, rather than collapsing them to input aliases.
- Remove duplicate effect-token carry1902 and its fabricated initial497 after
  the scalar return and identity-returning child native tests pass.

Touch: reducer field/effect ledgers and liveness, projection alias remapping,
`ssa_record_return_state.py`, physical return layouts, paired signature pruning.

Required tests: first-arm write, nested conditionals, retries, all terminal
returns, clamp increments, identity-returning child, and unchanged early read1378.

## Patch05 — bound method capability and invocation

**Findings: run_superstep164, run_superstep57.**

- Resolve state.dt_limit_hint against its authored class and receiver ABI.
- Produce callable(hint) from that proven capability and classify hint() as the
  bound method call, removing the incorrect NoneType intrinsic classification.
- Preserve default/absent capability and optional method result behavior.

Touch: method/reference resolution, callable lowering, callsite specialization
and receiver binding.

Required tests: positive, absent, None, nonpositive, and nonfinite hints; verify
the first attempted dt respects the valid hint.

## Patch06 — value-position Boolean semantics

**Finding: step526.**

- Emit the value of osc_flag or stiff_flag at the pi_update call.
- Implement general selected-operand and/or semantics with proper short-circuit
  control; do not eagerly execute an effectful RHS.

Touch: structural Boolean lowering, control-expression/value bridge, source
dependency scheduling.

Required tests: all Boolean pairs plus an RHS with an observable effect or
invalid lookup that must remain unexecuted on the short-circuit path.

## Patch07 — keyed reads through returned records and guards

**Findings: run_superstep245, _no_exchange_observed11.**

**Implementation checkpoint (2026-09-07):** exact `mapping or {}` identity,
defaulted/indexed resident lookup, and source-call ownership are implemented.
Replay v18 removes `_no_exchange_observed11` and the related `run_superstep`
defensive-read structural identities. Guard-local placement and native/eager
behavior across absent/present keys still require the Patch07 runtime coverage
below, so this patch is not marked complete.

- Use Patch03's exact record-field/table binding for get(default), membership,
  and indexed reads through returned records and mapping aliases.
- Place the power_w lookup and its _scalar consumer inside the actual source
  guard. Remove unreachable operations only after their placement is correct.
- Retain the dt_unresolved lookup result as a real input to its scalar consumer.

Touch: record projection linker, lexical keyed queries, source short-circuit
placement, existing lookup/default/contains helper binding.

Required tests: absent/present keys, returned record aliases, disabled exchange
fraction, and native/eager agreement without hoisted RHS effects.

## Patch08 — exact formatted diagnostic strings

**Findings: step440, step448, step570, step550.**

- Implement one exact native JoinedStr/FormattedValue path for all four source
  sites, including the authored `.3e` formats and rollback operands.
- Retain canonical bytes or collision-safe exact interning; do not substitute
  hashes of unformatted numbers or template identity for Python string equality.
- Support the same mechanism for failure headers/report lines made live by
  optional presence in Patch10.

Touch: native string representation/formatting lowerers, JoinedStr dispatch,
reason sequence row schema, exact string comparison/deduplication.

Required tests: rounding, equal formatted strings from unequal numbers, signed
zero, exponent spelling, NaN/inf, and exact reason equality behavior at all four
current sites and the newly reachable failure sites.

## Patch09 — report presence and slices

**Findings: run_superstep115, run_superstep117.**

- Resolve getattr(first,'unresolved_report',()) using actual attribute presence
  and record-row identity. Use an empty default only with a valid absence proof.
- Lower the report sequence's `[1:]`, iteration, and output effects.
- Reapply reachability after presence and source short circuit are valid;
  never hide a producer by assuming all report paths are permanently dead.

Touch: record optional/default projection, sequence slicing, report iteration,
string output effects, reachability.

Required tests: absent, empty, single-line, and multiline reports, including
first-record extraction from the unresolved sequence.

## Patch10 — integrate the real input domain, loop state, and completion gates

**Required to make the19 repairs an actual completed native controller.**

**Implementation checkpoint (2026-09-07, replay v20):** structured numerical
operands of a while predicate now execute on entry and on the latch using the
updated carried state. This removes the two `run_superstep` `%71` dominance
findings. The remaining loop-return and carried-state dominance findings below
are still open, so Patch10 is not complete.

- Implement presence+payload ABI for optional controller fields. Update schema,
  caller/callee expansion, feed packing, loads, and decoding together. Admit the
  real dt_min=None/dt_max=None managed input without numeric substitutions.
- Reopen and implement the resulting exhaustion/failure paths; feed additional
  live findings back into the shared mechanisms above in this same series.
- Restore run_superstep's current-total condition, last_metrics record carry,
  remaining-window annotations, optional clamps, and exact break-edge
  last_dt_next behavior. Preserve zero-iteration and failed-attempt semantics.
- Replace unproven advance-result aliases with exact return/storage links.
- Repair source guards/bodies behind the existing scalar-record, child-record,
  and call-only-loop xfails; require genuine native passes.
- Gate missing/duplicate effects, stale reads, and missing structural return
  slots as well as formals, undefined operands, unresolved calls, and ABI gaps.
- Resolve the append test's C-shim contract assertion while retaining its
  actual native mutation verification.

Touch: ProgramABI and native adapter; loop/control recovery; record return
linker; self-check/provenance gate; focused native regression tests.

Required tests: all optional presence combinations; zero/one/multiple substeps;
retry/floor/exhaustion; iteration cap and zero-duration break; exact-window
completion; distinct advance return vs state field; full output/state parity.

## Exact coverage check

| Patch | Baseline findings | Count |
|---|---|---:|
| 01 | coerce0; propose49; superstep291 | 3 |
| 02 | step417,418 | 2 |
| 03 | step496 | 1 |
| 04 | step497,554 | 2 |
| 05 | superstep164,57 | 2 |
| 06 | step526 | 1 |
| 07 | superstep245; no_exchange11 | 2 |
| 08 | step440,448,570,550 | 4 |
| 09 | superstep115,117 | 2 |
| **Total** | **Every baseline finding, once** | **19** |

Patch10 covers defects that produce no baseline formal and paths restored by
the correct input ABI. It is not optional finishing work.

## Integrated validation after the series

1. Run the focused source-to-native regressions for each shared mechanism;
   resolve failures without introducing placeholders or broad xfails.
2. Regenerate the full SSA/formal report from the real optional input domain.
   Require all strengthened gates to pass. The intermediate count may rise
   when previously discarded source operations become real consumers.
3. Build the full native validator only after that gate, then compare every
   returned field and mutated state against Python across the scenarios above.
4. Fix mismatches within this series, run the relevant regression checks, and
   report completion only after native parity. No commit/push is implied.

## Total report field and finite sequence-schema propagation (02:50 update)

`Metrics.unresolved_report` is now an authored `list[str]` field with an empty
default. The unresolved path assigns `list(lines)`, so every Metrics record has
one resident, variable-length report arena instead of encoding dynamic
attribute absence as an anonymous scalar input. Imported dataclass defaults and
annotations now publish the aggregate kind and physical column dtype on a
synthesized field-state seed.

Sequence column contracts now propagate through exact materialization and
aggregate-Phi edges to a fixed point. Edges are admitted only when row width
and key policy match. Existing contracts are never overwritten, so equal or
conflicting evidence retains the incumbent and remains visible to the ordinary
replacement validator. The focused reducer/controller batch passes 3 tests in
2.45 seconds and checks the complete `list[str]` seed -> Phi -> materialized
arm chain.

Fresh checkpoint v45 passes the v44 sequence-replacement failure, converges in
six frame rounds at 4,276 formals, and reports zero incompatible result
contracts. Production pruning removes the controller `%477` blocker entirely;
the full-native gate now reports only `run_superstep` `[124, 126]`. The raw
self-check still reports eleven controller formals
`[174, 177, 196, 199, 204, 207, 212, 263, 264, 401, 431]` and exposes one
controller dominance defect: a call in `if_merge.11` reads report seed `%478`
whose definition is in `while_exit`. Artifact:
`build/patch_sequence_fresh_v45/repository-ssa.pkl`. The remaining two
production formals reject the build before C compilation; no native parity is
claimed.

## Resident report-tail loop and record ABI (03:51 update)

The loop composer now proves a constant nonnegative tail slice of a resident
sequence as loop bounds over the original arena. For
`first.unresolved_report[1:]`, the loop starts at one and no longer requires a
separate slice value. Fresh v48 reduced the planned root regions from 196 to
195 while retaining all 19 `run_superstep` regions.

`Metrics.unresolved_report` is also declared as a mutable one-column `int64`
token table in ProgramABI. If specialization removes the authored list
producer but retains a demanded record-field occurrence, parameter lowering
now allocates the table's column, length, capacity, and status cells from that
schema instead of degrading the field to an opaque reference. Exact returned
records correlate all sequence members. Later record-frame reconciliation
uses the caller's resident descriptor as the incumbent when a callee publishes
an equal-priority view of the same storage identity.

Checkpoint replay v54 completes six frame rounds, reaches result-type fixed
point in three rounds with zero incompatible result contracts, and saves
`build/patch_sequence_replay_v54/repository-ssa.pkl`. The raw self-check has
seven findings. In `run_superstep`, the old anonymous slice bound `%126` is
gone; `%127` remains as an anonymous indexed loop value, and sequence 69 still
lacks the physical `Metrics` row descriptor needed for append/storage. The
controller contributes twelve anonymous formals, one missing report column in
a `Metrics` row table, and two `%478` dominance findings. The window wrapper
contributes `%47`. This replay does not apply production pruning, compile C,
or claim native parity.

The focused batch passes 5 tests in 6.22 seconds, covering resident tail-slice
planning, the ProgramABI report table, reducer dtype propagation, controller
behavior, and incumbent-preserving complete sequence-member binding. No
execution deadline was used.

## Tail-domain ownership and returned-record aliases (04:23 update)

Loop-domain regions now have an explicit ownership channel separate from loop
body emission. Once a resident tail slice has become `start=1` over its base
arena, the old Indexed/Slice numerical region is excluded from flat scheduling
even if a separate shell-output effect prevents emitting the complete loop
control program. This removes the region call that kept `%127` alive.

Deferred record-row expansion also follows the compiler's exact
`output_identity_aliases` and `value_aliases` to their resident record before
checking the physical layout. This resolves `run_superstep` record result
`%242` to its proven resident descriptor instead of reporting that the record
is unavailable.

Fresh v56 saves
`build/patch_sequence_fresh_v56/repository-ssa.pkl`, converges after six frame
rounds at 4,287 formals (six fewer than v55) and after three result-type rounds
with zero incompatible contracts. Raw findings fall from seven in v54 to six:
the `run_superstep` formal-parity group is gone. Its remaining finding is now
the truthful row-layout frontier: the resident `Metrics` record supplies 11
physical columns while sequence 69 requires 15. The missing surface is the
three-column `error_channels` mapping plus one nested `unresolved_report`
child-table handle. The production gate likewise contains no anonymous
`run_superstep` formal; its `run_superstep` failure is only the unresolved
record append. No C compilation or native parity is claimed.

The exact focused regression passes in 4.37 seconds. A broader loop-filtered
run had 65 passes and three existing dirty-suite failures, including a
test-local `NameError`; it is recorded as non-green rather than used as
validation. No execution deadline was used.

## Late returned-record surface completion (04:35 update)

Native call linking now resolves an aggregate result through the caller's
exact output/value identity aliases before looking up its resident record.
When a later fixed-point round publishes additional flat fields on the callee
record, it allocates caller-owned slots only for the missing storage identities
and merges them into that incumbent descriptor. Sequence and nested-record
fields still require an existing exact resident descriptor; they are not
misrepresented as scalar slots. Equal-priority alias evidence retains the
first incumbent.

Replay v57 saves
`build/patch_sequence_replay_v57/repository-ssa.pkl`, converges after six frame
rounds and three result-type rounds, and reports zero incompatible result
contracts. The `run_superstep` row advances from 11 to 14 of 15 physical
columns: all three `error_channels` columns are now present. The sole missing
column is the `unresolved_report` child-table handle. Raw checking still has
six findings, but completing the record surface exposes 23 previously hidden
anonymous formals in `step_with_dt_control_used`; the frame count rises from
4,287 to 4,365. The remaining structural frontier is therefore the nested
report-table row handle/copy, followed by the exposed callee formals, `%478`
dominance, and window `%47`. This replay did not run production pruning, C
compilation, or native parity, and used no execution deadline.

## Resident returned sequence arena (04:52 update)

A planned `GetAttr` region could return a mutable record sequence field as if
the arena were a scalar aggregate result. In the controller this defined
`unresolved_report` arena `%478` only in `while_exit`, while calls in
`while_body` and `if_merge.11` already consumed it. After frame linking, the
compiler now recognizes only an exact one-output record-field projection whose
record and sequence descriptors agree, removes that redundant region
call/projection, and promotes the same arena identity to accounted resident
storage. The transformation records its source region in
`resident_record_sequence_promotions`.

The sequence runtime also has a focused fixed-capacity child-copy append
lowering. It checks both outer and flattened child capacity, snapshots a leaf
source sequence into the destination row's stride, publishes child
length/status, stores the row index as the handle, and advances the outer
length only after the copy completes. It is tested independently; connecting
it to deferred record-row expansion remains the next chunk.

Replay v58 saves
`build/patch_sequence_replay_v58/repository-ssa.pkl`. It converges after six
frame rounds at 4,365 formals and after three result-type rounds with zero
incompatible contracts. Both `%478` dominance findings are gone, reducing raw
findings from six to four. The `step_with_dt_control_used` anonymous-formal set
also falls from 23 to 20. Remaining findings are window `%47`, those 20 callee
formals, and the two nested-record row boundaries (`run_superstep` 14/15 and
controller `unresolved_report`). The focused batch passes 3 tests in 4.78
seconds. No execution deadline, production-pruning run, C compilation, or
native parity claim was used.

## Optional predicate and local mutation lowering (08:11 update)

Optional record identity tests now lower before control planning. Every field
uses one canonical Boolean presence input per function; `is not None` reads it
and `is None` uses an explicit scalar `logical_not`. The C module lane now
spells that operation. Synthetic single-exit return Phis use the exact
conditional-result path when no pre-branch incumbent exists, preventing the
first arm from replacing the second. A mutable optional scalar write stores
into the caller-owned payload cell and then stores true into its presence cell;
the receipts name the field and retain the incumbent on ties.

Structural folding also repairs an aggregate ledger if its selected producer
survives while projections from a rejected arm are removed. It keeps resident
leaf IDs, recreates only missing positions from exact stored output
descriptors, records old-to-new IDs and the incumbent tie rule, and is
idempotent. This removes the fresh v71-v74 `snapshot` dangling-leaf failure.
The final related batch passes 17 tests in 8.12 seconds, including two native
artifacts compiled at `-O0`; absence, present zero, present nonzero, and an
absent-to-present local mutation all execute correctly. No deadline or
optimized compilation was used.

Replay v76, based on the fresh v74 resolved graph, completes planning, all 194
root regions, six frame rounds, and two result-type rounds with zero result
contract conflicts. It saves
`build/patch_sequence_replay_v76/repository-ssa.pkl`. The raw audit reports six
findings: four formal-parity groups (`run_superstep` 1,
`step_with_dt_control_used` 3, `balloon_tire_managed_advance` 24, and
`balloon_tire_vector_step` 388) plus two function-exit dominance findings for
controller values `%263` and `%264`. The local optional path is proven, but the
real `update_dt_max` specialization does not retain its optional presence
argument or presence-write instructions after cross-call frame linking. That
propagation boundary, together with the newly live downstream formal surface,
is the next chunk. No full C emission, full native compile, or parity claim was
made.

## Late aggregate return and C correctness frontier (06:36 update)

Late call-frame linking now materializes a complete returned record even when
it contains nested sequence storage. It recursively installs complete nested
descriptors, clones the caller-owned scalar and sequence arena surface, and
records the exact result binding. Replay v68 therefore gives window record
`%46` its 15-field `Metrics` descriptor and binds `hard_failure` to resident
caller scalar `%301`. The structural checker reports zero findings.

A fresh production lowering in
`build/patch_sequence_fresh_v69/repository-ssa.pkl` also passes the strict
structural gate with zero findings. Its first C emission exposed 18 backend
shortfalls. Scalar `item` and the C99 `isfinite`/`isinf`/`isnan` predicates
removed 16; the remaining two were duplicate initial
`controller.clamp_events` projections inside planned region 0.

Planned numerical regions now replace such projections only when the owner
proves a finite alias chain to one resident scalar in the receiver's record
descriptor. The field crosses the region boundary as one explicit capture;
two equal projections reuse the incumbent capture, provenance retains both
alterations, and the unused conceptual record receiver is pruned atomically
with every call operand. Replay v70 converges in six frame rounds and three
result-type rounds, reports zero structural findings, and emits C with zero
shortfalls.

The first correctness compile then exposed three repeated C declarations for
stable mutable scalar identities. The module backend now declares each such
local once and emits later loads as assignments. Its executable regression
returns the second loaded value. The full v70 artifact compiles successfully
at `-O0` to
`build/patch_sequence_replay_v70/native-o0/balloon_tire_managed_native_c.dll`
(1,650,688 bytes). The related focused batches pass 5 tests in 4.53 seconds
and 3 tests in 3.55 seconds. No execution deadline was used. Optimized builds
are deferred until correctness and native parity are complete; full native
execution/parity remains the next frontier.

## Optional ProgramABI boundary foundation (06:59 update)

The first real-input execution attempt stopped before entering the DLL because
`controller.dt_min` was `None` while the root exposed only a `float64` payload.
Artifact inspection also proved the semantic consequence: the saved controller
had compiled `dt_min is not None` as `Const True`.

ProgramABI scalar fields can now declare `optional: true`. Such a field
materializes a Boolean `field.__present` slot beside its typed payload, and the
managed feed adapter accepts absence only when both slots exist. An absent
payload receives canonical inactive storage while the presence bit alone
controls semantics; a present numeric zero remains distinguishable from
absence. `STController.dt_min` and `dt_max` now use this schema. The old guard
that rejects `None` without a presence ABI remains green.

The focused contract, materialization, and packing batch passes 9 tests in
4.32 seconds. No compilation or execution deadline was used. This establishes
the physical boundary only. Source `is None`/`is not None` predicates and the
mutable transition when `update_dt_max` changes absence to presence are the
next required chunk before another fresh controller build.

## Controller formal closure and dead specialized CFG (05:41 update)

The exposed 23-value controller group had three general causes, repaired as
one lowering batch. Definition-free `GetAttr` placeholders on an exact
resident record now resolve to that record's established field state. Source
mutation lowering retains scalar and string literal arguments as local
constants, with strings represented by stable tokens. Finally, the completed
whole-object CFG now removes blocks unreachable from entry, repairs each
predecessor-labelled Phi in the same transaction, and only then prunes callee
formals and corresponding caller operands. This ordering prevents dead
specialized return/loop compartments from keeping phantom ABI captures. Live
calls remain ordered effects. Equal-priority field evidence retains the
incumbent.

Replay v67 saves
`build/patch_sequence_replay_v67/repository-ssa.pkl`. It converges after six
frame rounds at 4,344 formals and after three result-type rounds with zero
incompatible contracts. The controller's 23 anonymous formals are all gone:
12 exact field projections, seven retained diagnostic tokens, and four values
owned only by dead specialized control paths. The CFG pass records 18 changes
and removes `return_edge`, `while_exit`, and both compiler-generated
unreachable control blocks from this specialization. The raw structural check
now has one finding total: window `%47`.

The six focused reachability/ABI tests pass in 8.57 seconds, including native
execution of both retained return paths, with no execution deadline. The next
frontier is precise: `%47` is `metrics.hard_failure` after `run_superstep`;
receiver `%46` has no resident `Metrics` record descriptor in the window
function, so its planned Boolean region still receives an anonymous scalar.
No C compilation or native parity is claimed.

## Deferred record rows with child snapshots (05:19 update)

Deferred whole-record rows and record slots inside wider rows now use authored
ProgramABI column order. A sequence field occupies one integer handle column;
its leaf arena, length, and capacity remain separate resident storage. Exact
returned-record sequence bindings replace provisional call-frame scratch and
are reconciled into already-linked incoming calls after resident-arena
promotion. Equal exact evidence retains the incumbent, while exact result
identity outranks provisional storage.

For each supported single nested sequence, lowering creates caller-owned
flattened child data and length arenas, derives total child capacity as
`outer_capacity * child_stride`, installs a `SSAChildTablePoolDescriptor`, and
replaces the original append with the tested child-copy helper. The helper
snapshots the child before publishing the handle and outer length. This works
both for a complete record row (`run_superstep` sequence 69, handle column 14)
and for a record embedded in a wider row (controller sequence 39, handle
column 15).

Replay v64 saves
`build/patch_sequence_replay_v64/repository-ssa.pkl`. Frame linking converges
after six rounds and result types after three rounds with zero incompatible
contracts. Both record-row findings are gone; raw findings fall from four to
two and now consist only of formal-parity groups: window `%47` and 23
`step_with_dt_control_used` values. Artifact checks prove both child pools have
complete resident storage, both helper calls exactly match their 37- and
41-formal ABIs, every source-linked call in the module has exact arity, and no
deferred record-row marker remains. The focused batch passes 4 tests in 5.60
seconds. No execution deadline, production-pruning run, C compilation, or
native parity claim was used.

## Cross-call optional presence identity (08:45 update)

ProgramABI materialization no longer assumes that a ProcessGraph presence ID
owns the same object in precompiled SSA. If that integer is already owned by
an instruction result, materialization allocates a fresh Boolean formal while
retaining both the requested graph ID and chosen physical ID in provenance.
An existing formal remains the incumbent on an equal identity tie. Optional
payload and presence roles are also distinct frame-storage keys, so they cannot
collapse merely because they name the same record field.

Replay v80 from the saved v76 pre-frame checkpoint proves the real
`update_dt_max` boundary. Graph presence `%24` becomes physical formal `%37`;
the presence write stores true to `%37`, the callee has five formals, and its
controller call has five operands with `callee_input_ids` ending in `%37`.
The caller operand has exact `ctrl.dt_max` optional-presence accounting. Frame
linking converges after six rounds at 4,353 formals and result typing after two
rounds. The raw frontier remains six findings: formal-parity groups of 1, 3,
24, and 388 values, plus controller function-exit dominance errors `%263` and
`%264`. Focused optional tests pass at `-O0`. No full C emission, full compile,
optimized compile, execution deadline, or native parity claim was used.

## Edge-local return expressions (09:03 update)

Path-correlated control had placed two pure numerical return expressions in
`if_merge.16`, while their synthesized `return_edge` occurred after the paths
rejoined. The return guard implied that the definitions ran whenever the edge
was selected, but repository SSA requires ordinary graph dominance and cannot
use that predicate correlation.

The final target-neutral composition seam now recognizes only an exact
`return_merge` slot whose return-edge receipt names the same source value and
whose producer is a pure planned-region projection. It clones the minimal pure
dependency slice onto that physical edge with fresh SSA identities. A
dominating operand remains the incumbent; accepted replacements record source
and physical IDs, source block, operation count, priority, and the incumbent
tie rule. Once repaired, the edge definition dominates and the pass reaches a
fixed point without another alteration.

Replay v81 saves `build/patch_sequence_replay_v81/repository-ssa.pkl`.
Controller `%263` becomes edge-local `%3390` through three instructions and
`%264` becomes `%3395` through five. Definition-dominance findings fall from
two to zero. The raw frontier falls from six to four findings, all
formal-parity groups: 1 value in `run_superstep`, 3 in the controller, 24 in
managed advance, and 388 in vector step. The focused CFG/return-state batch
passes 14 tests with two unrelated native tests deselected. No compilation,
optimization, execution deadline, or parity claim was used.

## Missing Phi incumbent identity (09:20 update)

A surviving conditional Phi can retain an `initial_value_id` for an
intermediate spelling removed by graph reduction. Control lowering previously
treated that absent integer as an external value and fabricated a function
formal. The repair now searches only the Phi binding's authored identity
history and requires one unique nearest value which is a common dataflow
ancestor of both arms. A resident incumbent is never challenged. Equal nearest
candidates remain unresolved, retain the incumbent policy, and are recorded
once so repeated fixed-point passes do not grow the ledger.

The same receipt is projected into an already-planned immutable ControlProgram
before SSA construction, because saved pre-frame checkpoints legitimately
contain Control IR built before this final graph normalization. Final function
metadata keeps the missing ID, resident ID, graph distance, priority, and tie
policy.

Replay v84 proves `run_superstep` Phi `%262` changes from phantom incumbent
`%260` to resident `%256`; its operands are now `(%261, %256)`. Formal `%260`
and the corresponding caller operand disappear together, call arity remains
exact, and total frame formals fall from 4,353 to 4,351. Raw structural
findings fall from four to three: controller 3, managed advance 24, and vector
step 388 anonymous formals. The focused control-compartment file passes all 7
tests. No compilation, optimization, execution deadline, or parity claim was
used.

## Post-reachability pure-feed recovery (09:38 update)

The controller's last three anonymous formals were exact scalar source
expressions: two `dt_tensor.item()` values and one `item() * 0.5` closure.
Late recovery already required ordinary dominance, but it ran before static
reachability pruning and therefore retained stale predecessor paths. Recovery
now accepts a scalar `item()` from its unique resident producer when no
loop-carried Phi exists, refuses multiple carried versions, and runs again over
the pruned CFG before callee/caller signature pruning. The second application
changes the proof surface; it does not weaken dominance.

Replay v86 defines `%149` and `%231` from resident `%43`, and defines `%147`
through `%145 = item(%43)`, constant `%146 = 0.5`, and `Mul`. All three formals
and their caller operands disappear; the controller and its source-linked call
both have exact arity 1,061. Final module formals fall 5,622 -> 5,619 and raw
structural findings fall three -> two. The remaining frontier is managed
advance's 24 anonymous formals and vector step's 388. The focused recovery file
passes 5 tests. No native compilation, optimization, execution deadline, or
parity claim was used.

## Static slice selector closure (09:49 update)

The remaining 412 anonymous formals were one source family: static `Slice`
syntax had crossed numerical region boundaries as floating runtime data. Each
accepted node is now required to be an exact `ast.Slice`, used exclusively as
an index, with non-negative static bounds and unit stride. It materializes as
the integer lower-bound offset consumed by `GetElementPtr`; the complete
`(lower, upper, step)` tuple and source value ID remain on the Const receipt.
Dynamic, negative, non-unit, aliased, or ordinary value slices remain
unresolved.

Replay v87 recovers 24 selectors in `balloon_tire_managed_advance` and 388 in
`balloon_tire_vector_step`. Their callee formals and every matching caller
operand are pruned atomically. Managed advance has exact arity 438 and vector
step exact arity 26. Final module formals fall 5,619 -> 5,207 and the raw
repository SSA checker reports zero findings. The focused static/dynamic slice
batch passes 4 tests.

The next fresh-source `-O0` build stopped before C emission or compilation at
the stronger physical-input gate: controller `%109` was seen as `float64`
while `pi_update` formal `%94` is the Boolean `dt_min.__present` slot. Replay
v87 carries both as Boolean, so this is now the precise fresh-source frontier.
No native compiler, optimized build, execution deadline, or parity run was
used.

## Optional-presence physical priority (10:12 update)

Fresh planning can first expose an optional-presence graph identity as a
numerical-region capture with the provisional physical default `float64`.
ProgramABI materialization already changed its logical dtype to Boolean but
merged the stale physical dtype. It now makes the explicit presence boundary
authoritative for both logical and physical dtype. If it displaces a
non-Boolean provisional dtype, accounting records the old dtype and reason;
an existing Boolean incumbent remains unchanged under the incumbent tie rule.

Fresh checkpoint v89 preserves resolved and pre-frame graphs. Replay v90
converges after six frame rounds and three result-type rounds with zero physical
input conflicts and zero structural findings. Controller `%109` and
`pi_update` `%94` are both physical Boolean values, and their call has exact
arity 12. The two focused optional-presence tests pass at `-O0`, covering
absence, present zero/nonzero, and mutation from absence.

The saved v90 module emits C with zero shortfalls and compiles successfully at
`-O0` to
`build/patch_sequence_replay_v90/native-o0/balloon_tire_managed_native_c.dll`
(1,669,632 bytes). No optimization, execution deadline, or full native parity
run was used. Standalone execution and state/output parity are the next gate.

## Standalone material contract and first live frontier (10:31 update)

The standalone packer now treats an optional field's physical presence slot as
the identity for all exact linked storage aliases of that parameter and field.
When absent, each alias receives a typed filler while the presence bit remains
false. Hoisted linked result-record storage with no authored parameter
provenance is initialized as private typed-zero workspace; ordinary unnamed
inputs still fail closed. The focused material-contract suite passes 11 tests.

`tools/build_managed_checkpoint_native.py` validates the saved checkpoint,
emits with zero shortfalls, and builds a 78-buffer standalone executable at
`-O0`. `tools/managed_dt_parity.py` now follows the checkpoint path recorded in
the manifest and runs native and eager processes to completion without a
deadline.

The first one-frame run is useful failure evidence: eager completes, while
native exits with a Zig alignment panic. Root planned region 3 declares feed
`%47` as `float64` and calls `cast_double_to_bool_values`, but its exact caller
operand `%308` is the Boolean `Metrics.hard_failure` result. Generated C passes
the byte-aligned Boolean result buffer as a `double *`, producing the panic.
The next frontier is therefore call-edge dtype reconciliation for this
redundant Boolean conversion. No optimized compile or numerical parity claim
was made.

## Final-seam physical adapters and live execution (11:07 update)

Planned-region captures may be separate `SSAValue` objects with the same
unshadowed formal ID. They are now interned to the declared formal before
physical adaptation; the formal is the incumbent and stale occurrence
accounting is not merged. This lets read-only buffer edges receive explicit
numeric adapters without changing shared storage. A second idempotent adapter
transaction runs after result typing, aggregate legalization, and precision
settlement, because those passes can expose new physical edges. Replay v94
records 17 conversions, five at the final seam; an immediate repeat makes zero
changes.

ProgramABI keyed members now establish physical dtype at every materialization
path. Generated keyed lookup helpers use that contract for keys, values,
length, capacity, status, and query storage, retaining the incumbent on equal
evidence and recording displaced provisional types. This removes the two
hidden string-token `int64 -> float64` conflicts. Focused results are 7 passing
adapter/keyed tests (including an `-O0` native test) and 3 passing keyed lookup
tests.

Replay v94 converges in six frame rounds and three result-type rounds with zero
physical conflicts and zero structural findings. Its 78-buffer standalone
build succeeds at `-O0`. Both native and eager one-frame executions now return
code 0: the alignment panics are closed. Numerical parity is not yet reached:
54 buffers match and 24 differ. Native state contains 3,456 non-finite values,
`advanced` is `0.000244140625` versus eager `0.008333333333333333`, and
`dt_next` is `0.0` versus eager `1.856126911235753e-05`. The next frontier is
the earliest native producer of the non-finite state/data divergence. No
optimized compile or parity claim was made.

## Collapsed-proposal dead end (11:43 update)

The v94 trace locates the first NaN at the second managed tire advance: the
controller's next proposal has collapsed to exactly zero, but `run_superstep`
still invokes authored physics, whose energy-rate calculation divides by
`dt`. `run_superstep` now checks the final bounded proposal immediately before
dispatch and exits through its existing incomplete-window path when that value
is zero or negative. This makes a no-progress controller state a terminal edge
instead of a state-mutating cycle.

The focused controller file passes 16 tests, including zero and negative
collapsed proposals. Fresh v96 converges in six frame rounds and three result
type rounds with zero incompatible contracts and zero structural findings. Its
78-buffer standalone builds at explicit `-O0`; both parity processes return 0.
Parity remains 54/78: the native state still has 2,056 non-finite elements, all
in batch lane 4, reduced from v94's 3,456. `advanced` and `dt_next` remain
`0.000244140625` and `0.0`, so the next frontier is the earlier first-substep
numerical divergence that collapses the proposal. No optimized compile or
parity claim was made.

## Indexed-store RHS extent (13:11 update)

The first-substep trace isolated a general indexed-store contract error.
`inputs[:, 0] = dt` selected eight destination cells, and tensor SSA lowering
incorrectly published that selection size as the RHS `value_count`. The native
kernel consequently read eight doubles from a one-element `dt` tensor. Lowering
now derives `value_count` from the RHS value's known shape and records a
shortfall rather than guessing when that extent is unknown.

The focused lowering and linked-record regression batch passes 22 tests,
covering rank-zero scalars, singleton tensors, and eight-element RHS storage.
Fresh v97 converges in six frame rounds and three result-type rounds with zero
incompatible contracts and zero structural findings. An audit of all 37
lowered indexed stores finds zero RHS-count mismatches.

The 78-buffer standalone builds at explicit `-O0`; native and eager both return
0. Native state is now finite in all 27,648 elements, with all eight batch
lanes following the same first substep. This removes v96's 2,056 non-finite
elements. Overall parity remains 54/78 because native stops after one
`0.000244140625` substep with `dt_next == 0`, while eager advances the full
`0.008333333333333333` window. The next frontier is the native first-substep
metric/reduction path that reports displacement `0.0009428783322241102` and
velocity `3.8620296487899624`, collapsing the controller proposal. Artifact:
`build/patch_sequence_fresh_v97/repository-ssa.pkl`; report:
`build/patch_sequence_fresh_v97-standalone-o0/managed-dt-parity.json`. No
optimized compile or full-parity claim was made.

## In-pass canonical shape settlement (14:02 update)

The inflated first-substep force traced through the scatter matmuls to the gas
law. A reciprocal whose producer had already established shape `(8, 4)` was
lowered from a stale same-ID scalar occurrence, so `binary_scalar_double`
processed one element and left the other 31 gas-pressure cells zero. Tensor
lowering now fills a missing occurrence shape from its already resident static
descriptor during the same rewrite. Existing shaped occurrences remain the
incumbent. Canonical fills record their identity provenance and the incumbent
tie policy.

The combined focused batch passes 24 tests. Fresh v98 converges in six frame
rounds and three result-type rounds with zero incompatible contracts and zero
structural findings. Its gas reciprocal is `(8, 4) -> (8, 4)`, has count 32,
and records `resident_descriptor_incumbent` provenance. The 78-buffer
standalone builds at explicit `-O0`; native and eager return 0.

Parity remains 54/78 because the native controller still emits `dt_next == 0`
after one `0.000244140625` substep. The force repair is measured: the native
first-step displacement and velocity now match the eager single-step values
`5.84721565e-7` and `0.00239501953`, and the final state worst error falls from
v97's `3.8615154563` to `0.00890879321`. The next frontier is the controller
continuation/result path that zeros the next proposal. Report:
`build/patch_sequence_fresh_v98-standalone-o0/managed-dt-parity.json`. No
optimized compile or full-parity claim was made.

## Settled forwarded results and atomic-region completion (15:55 update)

Identity-return calls now reconcile their forwarded record fields after the
callee input frame settles. The pass maps exact callee formals to caller
actuals, updates the call/result record descriptors, and rewrites only
dominated consumers when one latest forwarder is unambiguous. Equal-priority
or ambiguous candidates retain the incumbent, and every alteration records
its callsite and result-record provenance.

Applied to saved replay v111, the pass makes 37 reconciliations with zero
structural findings. In v112, `coerce_metrics.max_vel` and its downstream
`update_dt_max` argument use snapshot `%1741` instead of stale `%1654`.
The one-frame `-O0` native run remains 54/78, but the controller is no longer
inert: native `max_vel_ever` moves from `9.5e-31` to
`0.0023950195322857593`, and `dt_max` from `3e27` to
`1.252599387837501`. `dt_next` remains zero.

The next trace found a second general ordering defect in `pi_update`. Its
final accumulator region also contains early input casts. Although hierarchy
planning orders atomic regions by their data dependencies, the later scalar
field scheduler re-sorted that region by its earliest member and moved it
ahead of the normalization write it reads. Region/effect interleaving now
uses each atomic region's final authored member, the point where the complete
region can execute. The focused combined batch passes 5 tests. This planning
change requires a fresh controller replay; no additional native compile or
optimized build was performed for it.

## Optional guards and conditional continuations (16:44 update)

Fresh v113 confirms the atomic-region completion fix with zero structural
findings. Its explicit `-O0` run moves native controller `acc` from `0.0` to
`0.42714935345574623` (eager `0.601198735259513`), while parity remains 54/78.
The generated `pi_update` then exposed erased `is not None` tests: typed filler
values survived but their exact ProgramABI presence bits did not control the
optional `dt_min` and `dt_max` regions. Lowering now recovers only an exact
field/presence-slot match; ambiguity retains the incumbent and the decision is
recorded as `exact_program_abi_optional_presence` provenance.

Restoring those guards revealed four stale, non-dominating continuation uses.
Two finite reconciliation rules close them. Conditional arm objects now flow
through the unique latest dominating `conditional_carried` Phi, using object
identity because distinct SSA values may share a source id. A branch-local
same-physical-type cast result may flow back to its dominating call actual only
when the callee return slot, caller aggregate projection, dtype, shape, and
device all match exactly. Both rules record provenance and retain the incumbent
without a unique winner.

Saved replay v116 converges in six frame rounds and three result-type rounds,
records eight Phi continuations and one identity-cast forwarding, and has zero
structural findings (down from four before the batch). Seven focused tests pass.
Its 78-buffer standalone builds at explicit `-O0`; native and eager both finish
one frame. Parity remains 54/78 and the key values are unchanged from v113:
native `acc=0.42714935345574623`, `dt_max=1.252599387837501`,
`advanced=0.000244140625`, and `dt_next=0.0`. The next frontier is the
value/presence propagation feeding the controller result, not structural
control validity. Report:
`build/patch_sequence_replay_v116-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

## 2026-09-08 — exact optional scalar return and structural-formal closure

The next helper gate is closed generally. A finite SSA pass recognizes a
`return_merge` Phi whose alternatives are one scalar representation and
`None`, preserves the scalar identity as its payload, adds one Boolean
presence Phi, and expands both the callee Ret and every linked caller Call to
the same `(payload, presence)` physical contract. Exact caller `is None` and
`is not None` uses read the presence projection. Every alteration records its
priority and `tie_policy=incumbent`; a second pass makes no changes.

Two adjacent structural gaps were fixed in the same batch. ProgramABI record
discovery now recognizes intrinsic `getattr(obj, "field", default)` calls as
field reads, including optional-presence tests. Anonymous control-predicate
formals with exact graph producers are recovered even when they are not data
ancestors of Ret, direct repository unary predicates such as `isfinite` are
eligible, and each recovered dependency closure is placed before its unique
consumer block rather than after the function-exit Phi.

On the real `_energy_time_limit` source, the four false formals are gone: both
channel-membership tests are native contains scans, the energy finite check is
a local `isfinite`, and `energy_exchange_fraction` is the accounted optional
ProgramABI payload paired with its presence input. Formal parity and optional
merge checks are empty, C emission has zero shortfalls, and the exact helper
artifact compiles successfully at explicit `-O0`. The focused batch passes
3 tests in 19.46s, and an adjacent structural/call/optional batch passes 10
tests in 8.44s, without a process deadline. The saved v118 controller and its
55/78 parity count predate this lowering change, so full-controller replay
remains the next measurement frontier.

## Repeated SSA definition identity (17:01 update)

The clean v116 graph exposed a stricter SSA violation in emitted `pi_update`:
the first optional merge and a later branch-local aggregate projection reused
the exact same `SSAValue` object as a result definition. Consequently the final
Phi's true and false arms were indistinguishable, and C emission selected the
same uninitialized call slot on both edges.

A bounded canonicalization pass now keeps the first definition as incumbent,
freshens every later definition of that object, and rewrites only uses dominated
by the later definition, including each Phi operand on its own incoming edge.
The C emitter also retains exact-object bindings for formals, aggregate
projections, and Phi results so distinct same-source-id values cannot shadow one
another. Replay v118 records eight repeated-definition freshenings, six Phi
continuations, one identity-cast forwarding, and zero structural findings.

The combined focused batch passes nine tests. The explicit `-O0` 78-buffer build emits the corrected
Phi copies: the true `dt_max` arm takes `callout8686_0`, while the false arm
takes prior `t87`. One-frame parity improves from 54/78 to 55/78. `advanced`
now exactly matches eager at `0.008333333333333333`, and native `dt_next` is no
longer zero (`0.013326566940131533`). The remaining adaptive trajectory differs:
native final `max_vel_ever=0.3159734017361334` versus eager
`0.009045569831777825`. The next frontier is the earliest multi-substep value
divergence. Report:
`build/patch_sequence_replay_v118-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.
## 2026-09-08 — first substep divergence: missing Targets ABI fields

The earliest measured numerical divergence is now explained. Eager
`_energy_time_limit` caps the first proposal to `2.44140625e-05`; native had
reduced that helper to `return None` and took only two substeps. The shared
ProgramABI did not declare the authored optional
`Targets.energy_exchange_fraction` field, so `getattr(..., None)` legally
folded to `None`. `energy_exchange_fraction` and `shadow_growth_max` are now
declared as optional float64 scalar fields. The default exact-source audit
retains the full helper and reports zero structural shortfalls; two focused
checks pass in 4.45s without deadlines. An `-O0` native attempt advances to the
next gate: optional return payload/presence representation plus accounting for
three recovered structural inputs. This is measured progress, not a full
controller parity claim.

## Exact structural membership negation (23:20 update)

The capacity chain is live. Trace evidence shows `_energy_time_limit` receives
correct energy and power keys and values, but recovered source `not in` used
`contains != False`, which preserves rather than negates the contains result.
Structural recovery now emits unary `LNot` with exact-source priority and the
incumbent tie rule, matching the normal lowering path.

The combined membership/capacity batch passes 5 tests. Replay v128 converges
in 6 frame rounds and 3 result rounds with zero incompatible contracts and
zero structural findings. Its 82-buffer standalone builds at explicit `-O0`.
Native and eager both return 0. Parity stays 59/82, while native
`max_vel_ever` improves from `0.3159734017361334` to `0.017680656115454992`
against eager `0.009045569831777825`, and `dt_next` moves from
`0.013326566940131533` to `6.104353894230951e-06` against eager
`1.85612691e-05`. `advanced` remains exact. The newly live many-substep native
path finishes naturally in about 12 minutes 22 seconds. The next frontier is
the traced first mapping store occurring before a later sequence clear, which
erases `maximum_substep_displacement_m`. Report:
`build/patch_sequence_replay_v128-standalone-o0/managed-dt-parity.json`. No
optimization or process deadline was used.

## Dynamic mapping literal effect order (00:01 update)

Dynamic dictionary rows used their stored value producer as the effect
position. A value computed before the literal could therefore store its row
before the literal's later clear. Rows now use their exact key AST positions;
data dependencies still schedule value producers first, while the literal's
clear and rows retain authored order. Lexical sequence placement also uses the
shared last-member position for atomic regions. Equal positions retain the
incumbent order.

The focused batch passes 3 tests, including one explicit `-O0` native artifact
reused across three input cases. Replay v129 converges in 6 frame rounds and 3
result rounds with zero incompatible contracts and zero structural findings.
Production sequence 213 now contains `clear, displacement, energy, power` in
one block, with no early row store. The 82-buffer standalone builds at explicit
`-O0`; native and eager both return 0. Parity remains 59/82 and measured
controller values are byte-for-byte unchanged from v128, so displacement is
not the active limiter in this fixture. The next frontier is a short-window
trace of the first divergent substep. Report:
`build/patch_sequence_replay_v129-standalone-o0/managed-dt-parity.json`. No
optimization or process deadline was used.

## Local returned-value binding in C (00:31 update)

The C emitter formerly bound an exact returned SSA object to its caller output
buffer before its local producer ran. An internal consumer of that same object
therefore kept reading stale caller storage even after the id-level binding
advanced to `tN`. Locally defined outputs now remain local until `Ret`
publishes them; formal aliases and aggregate projections keep their exact
bindings.

A focused batch passes 4 tests, including an explicit `-O0` native Boolean
producer/consumer regression. Reusing the clean v129 SSA checkpoint, the
82-buffer v130 standalone builds at explicit `-O0`, and the production finite
test now negates local `t307`. Both parity processes return 0. The match count
remains 59/82, but native telemetry improves from 0 successful / 158 critical
substeps to 158 successful / 0 critical. The next frontier is the completion
boundary: native takes 158 successful substeps without setting completion,
while eager takes 157 and completes. Report:
`build/patch_sequence_replay_v130-standalone-o0/managed-dt-parity.json`. No
optimization or process deadline was used.

## Exact loop-record views and state-based fixed points (03:26 update)

A record returned by a source-linked call inside a loop was collapsed to the
incumbent loop-result identity before its physical fields existed. The linker
now retains the exact call result long enough to construct checked header and
exit record Phis. When the return is richer than the incumbent, the loop merge
uses a separate exact-field projection of the incumbent schema, while
sequence-row consumers retain the richer exact record. Each merge and
projection records its identities and `incumbent_on_equal_priority` policy.

Frame linking also has a generic state-based fixed-point guard. It hashes the
complete governed function, record, sequence, and call-table state. A repeated
state records its digest, rounds, and cycle period, then retains the incumbent;
there is no round cap or elapsed-time deadline. Exact frame receipts are
reapplied after aggregate legalization so stale positional operands cannot
override proven physical bindings.

The focused record/frame batch passes 2 tests. Replay v131 converges after 6
frame rounds and 3 result-type rounds with zero incompatible contracts and zero
structural findings. Its standalone builds at explicit `-O0` with 70 public
buffers. Native and eager both return 0. Parity is 49/70, not directly
comparable with v130's 59/82 because the public surface changed. Native
telemetry is byte-for-byte unchanged: completion remains false after 158
successful attempts, while eager completes after 157. The next frontier is the
managed-window completion/result projection after `run_superstep`. Artifact:
`build/patch_sequence_replay_v131/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v131-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

## Terminal record-field Phis and declared scalar storage (05:10 update)

The completion failure was a positional ABI error. A richer loop record was
correctly projected onto the incumbent 11-field schema, but the next frame
round treated each emitted physical field Phi as another conceptual record.
Loop-alias propagation then registered the complete richer descriptor under
every scalar field id, and return expansion published eleven repeated 14-field
groups. Generated C consequently wrote `error_channels.length` into the public
`hard_failure` position.

Fieldwise record Phis now carry explicit terminal provenance on both the
instruction and SSA result. Record materialization and loop-alias propagation
both honor that marker, so neither pass can consume its own physical output as
a conceptual record in a later fixed-point round. Richer records still use the
exact incumbent-schema projection, and equal priority still keeps the
incumbent.

The compact graph exposed one previously hidden optional edge. Late result
typing gave the declared scalar `advanced_dt` field a stale singleton tensor
shape. A completed-module invariant now applies record descriptors as the
authoritative physical ABI, normalizes declared scalar fields after result
refinement, and records the prior shape, owning record fields, priority, and
incumbent tie policy. Optional lowering then emits the explicit payload and
presence representation instead of leaving a `None`/array Phi.

The focused batch passes 4 tests and both changed compiler modules pass
`py_compile`. Replay v136 converges in 6 frame rounds and 3 result-type rounds
with zero incompatible contracts and zero structural findings. Its
`run_superstep` layouts are exactly three conceptual 11-field layouts; the
public return is 13 values, and `advanced_dt` has explicit optional presence.
The explicit `-O0` standalone build succeeds with 70 buffers.

One-frame parity remains 49/70, but the targeted behavior is fixed: native
completion is now `1.0`, matching eager, instead of v131's false completion.
Native still takes 158 successful attempts versus eager 157. The next frontier
is numerical/controller state: native `controller.dt_max` is `3e27` versus
eager `0.3316540644527175`, and native `dt_next` is
`6.104353894230951e-06` versus eager `1.856126911235753e-05`. Artifact:
`build/patch_sequence_replay_v136/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v136-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

## Deduplicated record-result identity and formal passthrough (06:27 update)

The first numerical controller divergence was a record-result identity error.
`balloon_tire_managed_advance` returns `max_vel` and `max_flux` from the same
SSA value. Aggregate legalization correctly removes redundant physical outputs
and folds output views of a callee formal back to the caller actual, but the
later record reconciliation required every logical position to remain emitted.
It therefore left the exact call-result record stale and shifted all following
field names relative to the surviving outputs. `coerce_metrics` received zero
for `max_vel`, so `update_dt_max` computed `dx / 1e-30 = 3e27`.

Aggregate output legalization now publishes an ordered receipt for every
formal passthrough. A completed-module reconciliation combines retained output
positions with those receipts, maps record fields by their declared storage
identity, and then reapplies the settled record to later call frames. Repeated
callee result identities use the first physical resident; equal-priority ties
keep the incumbent. The production chain now maps both `max_vel` and
`max_flux` to `%1657`, leaves `div_inf` at `%1660`, feeds
`coerce_metrics` with `(1657, 1657, 1660, 1661, ...)`, and passes `%1657` to
`update_dt_max`.

Five focused tests pass and the three involved compiler modules pass
`py_compile`. Replay v138 converges in 6 frame rounds and 3 result-type rounds
with zero incompatible contracts and zero structural findings. Its 70-buffer
standalone builds at explicit `-O0`; native and eager both complete one frame.
Parity remains 49/70 because the report is buffer-granular, but the targeted
values materially improve: native `controller.dt_max` moves from `3e27` to
`0.1696769611042683`, and `controller.max_vel_ever` moves from
`3.022244776478256e-34` to `0.017680656115454992`. Eager values remain
`0.3316540644527175` and `0.009045569831777825` respectively.

The next frontier is the optional-presence edge. The payload now updates, but
public SSA `%121` remains false because the `step_with_dt_control_used` call
still receives split intermediate `%1784` instead of the incumbent controller
presence resident. Artifact:
`build/patch_sequence_replay_v138/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v138-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

## Stable optional-field ownership across linked frames (07:18 update)

The false public `dt_max` presence bit came from an invalid ownership split,
not from optional lowering itself. `pi_update` and `update_dt_max` both name
the same scalar ProgramABI field, `STController.dt_max.__present`, but only one
callee-local formal was already represented in a record descriptor. The frame
owner check therefore classified the other by its local numeric SSA id,
declared a conflict, and allocated a separate slot that could never rejoin
without violating the transformation-priority graph.

Linked-frame ownership now recognizes the complete stable identity of scalar
optional fields: declared record, field, payload/presence role, dtype, and
shape. Receiver parameter spelling and callee-local SSA ids do not split that
identity. This rule is deliberately limited to scalar optional storage;
ordinary fields and aggregate members still require a record descriptor or
retain their callee-local owner. The first broad implementation exposed a
conflicting sequence descriptor during v139, so it was rejected and narrowed
before producing a build artifact.

The focused optional/priority batch passes 12 tests and `py_compile` passes.
Replay v140 converges in 6 frame rounds and 3 result rounds with zero
incompatible contracts and zero structural findings. It has 4595 formals,
three fewer than v138. The exact route is now
`update_dt_max %37 -> step %520 -> run %51 -> root %121`; no corrective
lower-priority rejoin is needed. The 70-buffer standalone builds at explicit
`-O0`, and both native and eager executions complete.

Parity improves from 49/70 to 50/70. Public controller presence SSA `%121`
now exactly matches eager `True`. Numerical controller values are unchanged,
which makes the next frontier the duplicate root controller inputs: `%394-396`
and `%451-453` are argument-only buffers passed to later run-superstep formals,
while the authoritative controller record owns `%120`, `%122`, and `%123`.
Artifact: `build/patch_sequence_replay_v140/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v140-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

## Canonical physical shape for declared scalar frame storage (07:57 update)

The duplicate root controller inputs were created because frame candidate
matching used the incidental SSA shape. Propagated scalar formals carried
stale shape `(1,)`, while the declared `STController` fields used shape `()`.
That prevented an exact candidate match and materialized new receiver-field
arguments for `dt_max`, `acc`, and `max_vel_ever` at each outer call frame.

Linked-frame identity now derives physical shape from the declared ProgramABI
storage contract. A field whose storage is `scalar` and rank is zero has shape
`()`, including incomplete legacy metadata with no explicit rank. The same
canonical shape is used for candidate lookup, owner identity, cloned receiver
members, and frame-ledger proof. Sequence, span, keyed-table, and other
aggregate storage retain their descriptor or callee-local identity. This is
the boundary learned from the rejected v139 generalization: sharing every
ProgramABI field across callees incorrectly merged unrelated sequence
descriptors. Existing priority rules remain directional, and equal priority
continues to retain the incumbent.

Ten focused scalar/optional/priority regressions pass and the compiler module
passes `py_compile`. Replay v141 converges in 6 frame rounds and 3 result
rounds with zero incompatible contracts and zero structural findings. It has
4542 formals, 53 fewer than v140. The stale `(1,)` controller arguments are
gone; the remaining controller formals all have declared scalar shape `()`.

The explicit `-O0` standalone build succeeds with 52 public buffers, down
from 70. Native and eager both complete one frame without a process deadline.
The report moves from 50/70 matches with 20 mismatches to 40/52 matches with
12 mismatches. Authoritative numerical values are byte-for-byte unchanged
from v140, so this is a pure identity/surface correction rather than a hidden
semantic change. The next frontier is the authoritative result path:
`controller.dt_max` `%120` is native `0.1696769611042683` versus eager
`0.3316540644527175`, and `dt_next` `%45` is native
`6.10435389e-06` versus eager `1.85612691e-05`. Artifact:
`build/patch_sequence_replay_v141/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v141-standalone-o0/managed-dt-parity.json`.

## Sequential conditional joins publish every arm identity (08:39 update)

Tracing the authoritative `dt_next` path exposed a separate legal-control
defect in `STController.pi_update`. Its `dt_max` conditional computed a merged
value, but the following `osc` conditional still read the earlier pre-clamp
SSA identity. The control plan reuses branch identities for later lexical
versions; the lowerer published a conditional Phi only under its initial and
merge ids, leaving its true and false arm ids mapped to branch-local values.
A later conditional naming one of those ids could therefore bypass the
preceding join.

Conditional lowering now publishes the merged resident as the current lexical
version of the true arm, false arm, incoming value, and merge identity. A real
later definition overwrites that map entry normally. Until such a definition,
all paths read the dominating join. This is a forward, finite version update;
it adds no reverse rewrite, and equal-priority ownership remains with the
incumbent.

Thirteen focused sequential, nested, reused-identity, dominance, and priority
tests pass, including a new two-conditional regression, and `py_compile`
passes. Replay v142 converges in 6 frame rounds and 3 result rounds with zero
incompatible contracts and zero structural findings. In production
`pi_update`, the `dt_max` false arm now consumes preceding merge `%87`, and
the `osc` false arm consumes post-clamp merge `%89`.

The explicit `-O0` 52-buffer build succeeds and both one-frame executions
complete without a process deadline. Parity remains 40/52, and every stored
mismatch array is byte-for-byte identical to v141. The corrected branch is
dormant for this fixture because the active proposal does not bind at that
conditional, so this chunk repairs the legal control graph without claiming a
numerical gain. The next frontier remains the earliest active source of the
controller/material divergence, upstream of the final PI result. Artifact:
`build/patch_sequence_replay_v142/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v142-standalone-o0/managed-dt-parity.json`.

## Explicit-axis min/max remain reductions (09:52 update)

The first-step material-output mismatch was a tensor-operation arity error.
`output[:, :, 10] = next_position[:, :, :, 1].min(dim=2)` reached tensor SSA
with two operands: the data tensor and the captured literal axis. The lowering
treated every two-operand `min`/`max` as Python's two-value built-in and
therefore emitted elementwise `minimum(next_position_y, 2)` with shape
`(8, 4, 144)`. The following `(8, 4)` store consumed only its first 32
elements, publishing one vertex coordinate instead of the reduction.

Two-value `min(left, right)` and `max(left, right)` now select elementwise or
scalar comparison only when no explicit `axis` or `dim` attribute exists. An
explicit axis retains reduction identity and reaches `reduce_dim_double`.
This is a direct source distinction and requires no shape guess or reverse
rewrite.

All 26 tensor-SSA metadata tests pass, including new `dim=` and `axis=`
regressions, and `py_compile` plus `git diff --check` pass for the changed
files. Replay v143 converges in 6 frame rounds and 3 result rounds with zero
incompatible contracts and zero structural findings. Production value `%1881`
is now reduced along dimension 2 to `%1883` with shape `(8, 4)` before the
column-10 store.

The explicit `-O0` one-step artifact builds with 52 buffers. Native and eager
both finish naturally. Parity improves from 49/52 to 50/52: `material.output`
now matches, leaving only read-only `controller.dt_max` aliases `%386` and
`%435`, each initialized to zero while authoritative `%120` matches eager.
Artifact: `build/patch_sequence_replay_v143/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v143-one-step-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

## Canonical read-only scalar frame aliases (10:36 update)

The final two one-step mismatches were not independent controller results.
They were generated read-only copies of the already-correct
`STController.dt_max` payload. Optional lowering had marked the writable
resident as an optional payload, while propagated read-only scalar formals
sometimes carried the same declared ProgramABI field without the redundant
payload marker. Frame matching treated the missing marker as a third storage
role and kept separate zero-initialized slots.

Declared scalar fields now infer payload role when they are not the explicit
Boolean presence slot. Candidate settlement prefers writable storage, then an
authored/non-callsite resident; original argument order retains the incumbent
on an equal-priority tie. This selection is scalar-only. Span and other
aggregate storage retain every physical candidate. A conservative final
entry-signature cleanup can remove a generated duplicate only when the
function has no repository caller and neither an instruction nor a call-frame
receipt refers to it.

Three focused role, priority, and pruning regressions pass, as do
`py_compile` and `git diff --check`. An adjacent 15-test selection has 14
passes and the existing span-accounting failure
`test_record_field_storage_identity_crosses_the_call_frame`; monkeypatching
both new selection mechanisms back to their prior behavior reproduces that
failure, so it is not attributed to this chunk.

Replay v145 converges in 6 frame rounds and 3 result rounds with zero
incompatible contracts and zero structural findings. It has 4,532 formals,
ten fewer than v143. The root retains only writable `dt_max` payload `%120`
and presence `%121`, and the `run_superstep` edge consumes those exact
residents. The explicit `-O0` one-step build exposes 48 buffers, removing the
two read-only `dt_min` and two read-only `dt_max` aliases from v143. Native and
eager finish naturally and all 48 buffers match. Artifact:
`build/patch_sequence_replay_v145/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v145-one-step-o0/managed-dt-parity.json`. The
next measurement frontier is the complete managed window on v145.

## Full-window v145 measurement (10:54 update)

The full managed-window executable also builds at explicit `-O0` with the
same 48-buffer public surface. Native and eager both finish naturally and all
27,648 material-state values remain finite. Both advance exactly the requested
`0.008333333333333333` window.

Full-window parity is 38/48. The four redundant controller aliases removed by
the scalar ownership fix remain absent; the ten mismatches are authoritative
trajectory results: telemetry, `controller.dt_max`, `controller.acc`,
`controller.max_vel_ever`, material inputs/state/output, the two last-step
material metrics, and `dt_next`. Native reports 158 successful substeps versus
eager 157. Representative final values are native/eager
`dt_max=0.1696769611042683/0.3316540644527175`,
`acc=0.43072264312898484/0.601198735259513`,
`max_vel_ever=0.017680656115454992/0.009045569831777825`, and
`dt_next=6.104353894230951e-06/1.856126911235753e-05`.

This measurement closes the alias chunk without claiming full-window
correctness. The next frontier is the earliest repeated-substep controller or
metric divergence; it requires trace evidence rather than another public
surface rewrite. Report:
`build/patch_sequence_replay_v145-standalone-o0/managed-dt-parity.json`.

## Unique field provenance across source call edges (11:46 update)

The adjacent span regression exposed a provenance gap before numerical-region
lowering. An exact call such as `inner(state.height, i, j)` propagated the
declared float64 rank-two span contract, but dropped the `height` field
identity from the callee and its planned region. The later frame settlement
could not restore that identity because it deliberately refuses to label a
generic helper from an incidental storage-backed actual.

The pre-lowering call-edge ABI fixed point now carries the declared
`(record, field)` identity while it remains unique. If different caller fields
feed the same formal, the destination records permanent ambiguity and drops
the field identity while retaining compatible dtype, rank, shape, and storage
facts. A uniquely proved identity annotates the physical formal before later
ownership settlement. This is monotonic: exact becomes ambiguous at most
once, and an ambiguous incumbent cannot be repopulated by a later edge.

Five focused span, shape, scalar-owner, and indexed-write tests pass. They
include the formerly failing single-field span test and a new two-field generic
callee regression proving that first-arrival order cannot label an ambiguous
formal. `py_compile` and `git diff --check` also pass.

Replay v146 reaches the same production fixed point as v145: 6 frame rounds,
4,532 formals, 3 result rounds, zero incompatible contracts, and zero
structural findings. The explicit `-O0` one-step artifact retains 48 buffers;
native and eager both finish and all 48 match. Artifact:
`build/patch_sequence_replay_v146/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v146-one-step-o0/managed-dt-parity.json`.

The v145 full native trace separately identifies the long-window numerical
boundary. Native and eager start with maximum state error `6.37e-13`; their
controller and metric traces remain within the parity tolerances through
31 substeps. Native `max_vel` first crosses tolerance at substep 32, then
`dt_pen` and the PI result cross at substep 33. Continuous floating drift is
therefore amplified by the adaptive trajectory before native completes in 158
steps and eager in 157; the trace contains no earlier discrete wrong-value
edge. Trace:
`artifacts/compiler_evidence/patch_sequence_replay_v145-trace-o0/native-trace.log`.

## Two-day continuity ledger and current frontier (2026-09-09 update)

This section is the continuation point after roughly two days of integrated
work.  A checked item below signs off only the named invariant and its cited
evidence.  It does not turn a focused test into a claim of full-window native
correctness.  The original nineteen-item structural audit is at zero findings,
but the complete patch sequence remains open until the runtime and required
coverage gates below pass.

### Signed-off invariants

- [x] **The original structural frontier remains closed at the repository-SSA
  gate.** Accepted replays through v166 each converge after six frame rounds
  and three result-type rounds with zero incompatible result contracts and
  zero structural findings.  The repository contains 4,538 formals.  Latest
  completed diagnostic artifact:
  `build/patch_sequence_replay_v166/repository-ssa.pkl`.
- [x] **One-step native execution has reached a completely matching public
  surface at an earlier stable checkpoint.** Replays v145 and v146 both build
  at explicit `-O0`, finish naturally, expose 48 buffers, and match 48/48.
  This establishes that the full compiler/runtime route can be correct for one
  step; later stricter identity repairs intentionally reopened two truthful
  values rather than preserving an accidental match.
- [x] **Terminal-only branch assignments do not reach a conditional merge.**
  `enrich_represented_conditionals` now asks whether each arm can fall through
  to its enclosing join.  An update whose only authored path ends in an
  unconditional break/continue/return is suppressed with outcome
  `terminal_only_update_not_merged`, priority `exact_lexical_fallthrough`, and
  tie policy `incumbent`.  A conditional loop control retains its untaken path;
  controls inside a nested loop stay local to that loop.
- [x] **Exact reducer field-state Phis outrank flat dotted `SetAttr`
  histories.** A flat identity history for `(receiver, field)` is now
  suppressed whenever the reducer already owns that exact field state, even
  when the authoritative Phi belongs to an earlier conditional.  The receipt
  is `exact_field_state_phi_retained` with priority
  `exact_reducer_record_field_state`; equal priority retains the incumbent.
  Replay v158 removes producerless `%328` completely: zero definitions and
  zero uses, while the exact lattice Phi `(286, 282) -> 564` remains.
- [x] **Control-defined values survive the projection that carries them.**
  Conditional merge/result identities are collected before numerical-region
  projection.  A loop carry defined by Control IR is retained even if no
  numerical region owns it.  Initial source-loop projection may defer carry
  pruning until the ordinary conditional overlay has supplied its metadata;
  the final projection still performs ordinary liveness pruning.
- [x] **Shared loop incumbents split into distinct logical bindings during
  local SSA lowering.** When several carries share one initial source
  identity, loop header, latch, and post-loop maps select the first incumbent.
  Conditional publication protects branch updates participating in those
  distinct bindings, so the local raw snapshot is not overwritten by a later
  conditional merge.  The focused SSA regression proves distinct backedges
  and definition dominance.  A later repository pass still collapses the
  production snapshot operand; that separate frontier is recorded below.
- [x] **The continuation search itself is finite, directional, and
  provenance-recorded.** For a snapshot carry, the Control IR enrichment pass
  follows only a unique chain of conditional `initial -> merged` edges.  It
  stops on a missing successor, records and rejects multiple successors, and
  records and rejects a repeated identity.  Unique continuations precede the
  lower-priority snapshot carry; ambiguous, cyclic, and equal-priority cases
  retain the incumbent.  The focused chain regression passes.
- [x] **The current focused batch is green.** `py_compile` passes for
  `control_source.py`, `glsl_deployment_strategy.py`,
  `precompile_to_ssa.py`, and `fortran_c_shell.py`.  The combined conditional,
  loop-result, break-boundary, field-state, projection, shared-incumbent, and
  continuation selection is 19 passed / 111 deselected.  It includes the
  regression proving that a raw update reused as a conditional initial remains
  the raw loop backedge while the recurrence uses the final conditional Phi.
  `git diff --check` reports no whitespace errors (only the repository's
  existing LF-to-CRLF notices).

### Runtime evidence after the stricter identity repairs

Replay v158 builds a 48-buffer one-step executable at explicit `-O0`.  Native
and eager match 46/48.  The only public mismatches are `material.telemetry` and
`dt_next`.  The exact `dt_next` values are:

- native: `2.4413922801613808e-05`
- eager: `2.44140625e-05`
- lattice quantum: `2.3283064365386963e-10`

The native value is exactly 104857 lattice quanta; the eager value is 104857.6
quanta.  Source semantics retain both values: the lattice-constrained value is
`dt_cap`, used as the next iteration's limit, while `last_dt_next` snapshots
the raw proposal and is returned.  This reduced the broad numerical symptom to
one exact compiler identity boundary.  Artifact:
`build/patch_sequence_replay_v158-one-step-o0/managed-dt-parity.json`.

Replays v159 and v160 tested projection retention and shared-incumbent lowering
but still produced one logical carry.  They were useful negative evidence:
retaining an existing source carry cannot create the second recurrence binding
when the source conditional reuses one numeric identity for its arm result and
its merge spelling.

### v161-v166: corrected evidence and exact resumption point

The first inspection after v161 looked at `step_with_dt_control_used`.  Its
source IDs 238 and 24 belong to the integer retry state, not to the floating
`dt` recurrence.  The resulting same-spelling hypothesis was tested in v162;
it split retry identity 238 into a synthetic identity and therefore changed the
wrong state.  That implementation and both supporting tests were removed.
v162 remains only as rejected diagnostic evidence and must not be used as an
accepted checkpoint.

The correct production function is `run_superstep`.  Its v161 repository SSA
contains two loop carries, but continuation enrichment ran before the late
ordinary conditional overlay.  The recurrence stopped at update 282 instead of
the lattice computation 286; both header Phis used final conditional Phi 565
as their backedge.  Thus v161 was a structurally valid partial improvement, not
a no-op and not the final identity separation.

Continuation solving is now deferred until after
`overlay_scheduled_control` and `_nest_lexical_conditionals_in_loops`.  Replay
v163 then gives the recurrence carry update 286 and preserves the authored
chain `274 -> 280 -> 282 -> 286 -> Phi 565`; `%328` remains absent.  The raw
snapshot carry still receives backedge 565 in the final repository artifact.

Conditional publication now publishes a merge under its initial identity only
when that initial is not a protected source for another loop binding.  The
focused regression proves the rule locally.  Replays v164 and v165 then show
that `run_superstep` local lowering has distinct carried updates:

```text
recurrence: source update 286 -> SSA Phi 565
raw snapshot: source update 274 -> SSA value 274
```

Opt-in provenance output under `TURING_DEBUG_LOOP_ALIAS_PUBLICATION=1` records
the protected identities and publication decisions without changing normal
compilation.  The v166 run adds latch-choice provenance.  Its live evidence
shows the recurrence latch choosing 565 and the snapshot latch choosing 274.
Therefore conditional enrichment, local publication, and latch completion are
all behaving correctly.  The remaining collapse from snapshot operand 274 to
565 occurs later, during repository/frame reconciliation.  v166 then finishes
after six frame rounds and three result-type rounds, with 4,538 formals, zero
incompatible result contracts, and zero structural findings.  Artifact:
`build/patch_sequence_replay_v166/repository-ssa.pkl`.

No C build has been run since v158.  This is deliberate: the exact SSA shape
gate is still wrong, and another native compile would only measure the already
known downstream symptom.

### Checkpoint journey and disposition

| Checkpoint | Disposition | What it established |
|---|---|---|
| v145-v146 | accepted runtime evidence | Explicit `-O0` one-step native and eager execution completed with 48/48 public buffers; v145 also located long-window divergence at substep 32. |
| source v147 | trusted replay input | `pre-frame-link.pkl` is the stable source-side checkpoint used to avoid repeating ingestion and planning work. |
| v158 | accepted diagnostic runtime evidence | Exact field-Phi authority removes `%328`; explicit `-O0` one-step parity is 46/48 and isolates telemetry plus raw `dt_next`. |
| v159-v160 | accepted negative evidence | Projection retention and shared-incumbent lowering alone cannot produce the two required logical carry histories. |
| v161 | accepted partial structural evidence | Early continuation solving reaches update 282, before the late lattice conditional is available. |
| v162 | rejected | A same-spelling synthetic-ID hypothesis changes integer retry state 238.  The implementation and tests were removed. |
| v163 | accepted partial structural evidence | Deferred continuation solving reaches recurrence update 286 and keeps the lattice chain, but both loop backedges still become 565. |
| v164 | accepted partial structural evidence | Protected conditional publication passes focused tests; the final repository artifact still collapses the raw snapshot. |
| v165 | accepted diagnostic evidence | Local conditional publication reports recurrence `286 -> 565` and raw snapshot `274 -> 274`; final repository SSA still has `274 -> 565`. |
| v166 | accepted diagnostic evidence | Latch completion independently reports recurrence 565 and raw snapshot 274; replay finishes with 4,538 formals, zero incompatible contracts, and zero structural findings. |
| v167 | in progress | Five stage probes bracket the first post-latch rewrite that changes the raw snapshot operand. |

This table is a disposition ledger, not a list of releases.  Rejected evidence
is retained so the same false path is not retried; accepted partial evidence
signs off only the boundary named in its final column.

### Generous remaining frontier

The next implementation chunk begins after local loop lowering, where the
snapshot operand is demonstrably still correct.  Work through this frontier in
order, signing off each boundary only with direct evidence:

1. [x] Let v166 finish naturally and retain its completed repository artifact
   and full diagnostic log.  It has zero structural findings and zero
   incompatible result contracts.
2. [ ] Locate the first post-latch pass that changes the `run_superstep` snapshot
   header-Phi backedge from object/value 274 to Phi 565.  Inspect the repository
   rewrite stages around operand refresh, frame linking, storage-alias rebinding,
   and result reconciliation.  If static inspection is insufficient, place
   several opt-in probes in one replay so a single expensive run brackets the
   mutation.
3. State the violated identity rule in terms of source identity, SSA object
   identity, owning loop carry, and provenance.  Do not special-case function
   names or numeric IDs.
4. Repair only the pass that performs the invalid canonicalization.  A loop
   carried feed with explicit ownership must retain its selected SSA object;
   equivalent numeric/source spellings must not redirect it to a conditional
   Phi owned by a different logical carry.  Existing higher-priority evidence
   may replace an incumbent; equal priority retains the incumbent.
5. Preserve finite behavior.  Operand reconciliation must be an ordered,
   monotone choice over existing candidates.  Re-visiting the same evidence
   must be idempotent, and ambiguous/cyclic provenance must dead-end without
   generating identities or oscillating between objects.
6. Add a focused regression at the late repository/frame boundary that failed
   in production.  Keep the local lowerer regression, but do not treat it as a
   substitute for exercising the rewriting pass that caused the collapse.
7. Run the combined focused batch and syntax checks once after the known edits
   are assembled.  Record counts and any expected deselections.
8. Replay the trusted checkpoint once more.  Before compiling, require two
   `run_superstep` header Phis: recurrence update 286 must use backedge 565;
   raw snapshot update 274 must use its raw SSA definition.  Require the chain
   `274 -> 280 -> 282 -> 286 -> 565`, zero `%328` uses/definitions, zero
   incompatible contracts, and zero structural findings.
9. Only after that SSA shape passes, build the one-step artifact at explicit
   `-O0` and run parity.  The immediate target is 48/48, including raw
   `dt_next` and telemetry.
10. Build and run the complete managed window at explicit `-O0`.  Compare all
   48 public buffers, completion counts, full state arrays, telemetry, and the
   controller trajectory.  If the prior substep-32 divergence remains, locate
   the earliest causal trace value and open it as the next bounded chunk.
11. Run the required coverage matrix in Patch01 through Patch10.  The current
    focused control tests and production fixture do not by themselves prove
    every listed empty/multirow collection, snapshot/clear, mapping alias,
    first-arm field write, method capability, short-circuit side effect, keyed
    read, diagnostic string, report slice, retry, terminal return, and repeated
    full-window case.
12. Re-run the full managed window enough times to establish that completion
    and buffer parity are stable rather than a single-run accident.  Keep all
    builds at `-O0` until every correctness gate is signed off.
13. Audit every generated artifact and every checklist item before declaring
    completion.  Update this ledger with the final commands and measured
    outputs.  Optimization remains deferred until correctness and full native
    parity are proved.

The trusted source checkpoint is
`build/patch_sequence_source_v147/pre-frame-link.pkl`.  The latest completed
accepted diagnostic repository-SSA artifact is
`build/patch_sequence_replay_v166/repository-ssa.pkl`; v167 is the current
natural-running stage-probe replay at the time of this update.  No process in
this sequence was killed for elapsed time, and no optimized build was used.
