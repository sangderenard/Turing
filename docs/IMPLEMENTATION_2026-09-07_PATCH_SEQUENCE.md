# Patch sequence implementation — 2026-09-07

## 2026-09-08: scalar record returns preserve snapshots

Returned scalar fields now use the unique latest dominating in/out write
version instead of reusing mutable caller-owned storage. Ambiguous candidates
and equal-priority candidates retain the incumbent. The pass updates the Ret,
record layout and descriptor, and each caller's positional native result
contract in one transaction and records its provenance.

Replay v111 publishes managed advance value 295 into both Metrics velocity
positions, remains structurally clean, and produces an explicit native output
to controller slot 1741. Focused validation passed 5 tests plus a 2-test
placement check. The explicit `-O0` build completed with 78 buffers. One-frame
parity remains 54/78, so the remaining defect is downstream in controller
calculation rather than at this return/snapshot boundary.

## 2026-09-08: returned records feed subsequent calls

Resolved caller output aliases before matching a returned record to a later
record-valued argument. Exact aggregate positions now settle repeated output
identities, update the resident record descriptor, and publish the same field
mapping to the call receipt and emitted call. Provenance records both the
exact-position decision and the incumbent-on-equal-priority policy. Nested
sequence handles travel with their physical members, while the exact returned
descriptor wins an equal-priority conflict with a later read-only projection.

Validation: focused record/call batch 3 passed in 4.66 seconds; saved full
controller replay v107 completed 6 frame rounds and 3 result rounds with zero
incompatible result contracts and zero structural findings. The three real
controller consumers of `Metrics.max_vel` now use produced slot 1741. The v107
`-O0` baseline subsequently completed at 54/78 one-frame parity.

This is an implementation checkpoint, not a claim that the complete 19-finding
series or the full native controller passes. The repository already contained
substantial uncommitted repairs when this work began; those were preserved.

## Implemented mechanisms

- Local dictionary construction has a lexical owner; copied tables retain source
  contents and indexed writes. Direct keyed return reads are emitted and preserve
  all four authored result slots. Empty-copy schemas and finite constant-key
  capacities are inferred without substituting runtime contents.
- Tuple snapshots use separate row storage and logical length. Copy reads are
  ordered against the source arena's mutations; they are not SSA dependencies on
  a future append. Local finite append bounds supply actual private capacity.
- Scalar record reads no longer overwrite the assignment ledger. First-arm reads
  get entry state, synthetic augmented writes inherit source guards, and scalar
  RHS results are publications rather than invented in/out formals. Numerical
  regions preserve declared field dtypes and authored read/write order.
- Constant terminal record writes are eligible for exact return receipts. An
  identity-returning child carries its exact receiver alias and field contract.
- Finite loops containing calls or terminal edges survive value-only unrolling.
  Existing call-frame bindings are reconciled with the caller's physical field
  using exact receiver provenance; this prevents accept mutations going to an
  orphaned duplicate buffer.
- Constant and/or folding returns the selected operand, and a later constant
  cannot erase an earlier unresolved operand's effects. Runtime ABI facts are
  not evaluated as truthy Python objects.
- The full-native gate rejects retained structural-output and required-source
  shortfalls even when the emitted Ret has no undefined operands.
- The append regression now verifies the real packed-shim contract and keeps
  its native mutation assertions.

## Validation evidence

Before the final batching pass: mapping copy variants passed (ordinary, empty,
overwrite); native record capture, first-arm, nested, augmented-assignment and
identity-child cases passed (5 in 44.72 seconds); native snapshot length and
sequence observations passed (2 in 19.15 seconds). The authored fixed rejection
loop passed natively (13.14 seconds), and its xfail was removed. The native
scalar and child record xfails were also removed.

Surrounding checks passed: sequence replacement 3, record-return receipts 12,
formal provenance 9, and control ordering 5 (before the additional snapshot
ordering case). A later batch passed 31 checks; two new Boolean cases failed
because assertions belonging to a neighboring test were accidentally left in
those cases. That test placement was corrected before the final batch.

A fresh lower-only controller diagnostic was externally terminated at 300
seconds during instantiation, before a new formal report. A second diagnostic
uses a 600-second bound and the isolated build/patch_sequence_diagnostic_v2
artifact directory. Its terminal result and final focused batch are recorded
below when available. No full native parity claim is established.

## Remaining implementation scope

The complete patch specification remains authoritative. Collection/generator
producers, mapping field rebinding with aliases, general effectful short circuit,
bound-method capability/default behavior, exact native diagnostic strings,
report slices/presence, and the real optional presence+payload input ABI still
need integrated implementation and validation. Loop/return edge cases beyond
the passing native fixtures remain open. Do not remove the old 19 formal
findings by aliasing them to arbitrary storage or treating missing output as
success. No full native build should proceed until the strengthened gate passes.

The latest user preference is to batch known fixes before validation, and to
exercise many runtime scenarios using one compiled artifact where possible.

## Execution policy correction

The second diagnostic was also terminated, at 600 seconds during repository
SSA lowering. Neither attempt yielded a fresh formal count. The user explicitly
instructed: “YOU CANNOT KEEP USING TIMEOUTS”. Subsequent runs have no execution
time ceiling; tool yields are progress polling only. The fresh v3 diagnostic
runs directly and logs to artifacts/compiler_evidence/patch_sequence_diagnostic_v3.log. Test subprocess
time limits in the five native regression files expanded in this batch were
removed as well. Do not repeat the timeout-based workflow.

## Identity priorities and saved method repair

Read-only CPython stack samples found the v3 run in frame reconciliation with
`next_value_id` increasing from 70510 to 71570. Field reconciliation was undoing
explicit ownership splits; each later ownership pass then allocated again.
Stopped that obsolete process (PID 11756) for the diagnosed rewrite cycle, not
because of elapsed time. The standard-library reader is
`speaktome/AGENTS/tools/read_cpython311_stack.py`.

The frame linker now uses `transformation_priority.py`: identities are caller,
callsite, callee, and callee formal; physical allocation IDs are decision payload.
Receiver-field reconciliation has priority 1, distinct ownership 2, and exclusive
result storage 3. Registered replacement edges must strictly increase priority.
Equal priority retains the incumbent, including its physical slot, as explicitly
requested. Accepted changes and rejected challengers retain rule, proof, priority,
and before/after provenance in `frame_transformation_provenance`. Repeated losing
proposals do not grow the receipt indefinitely. Existing explicit split proofs
survive reconciliation. Field lookup is indexed per caller rather than scanning
every caller formal for every callee binding. Frame rounds report progress and
the converged round count is retained; there is no execution deadline or round cap.

This is a termination contract for the registered frame rules, not a claim that
all compiler passes or arbitrary source programs have a termination proof.

Saved `getattr` method capabilities now retain receiver and original source
identity. Callable folding, single-exit return aliases, and readonly scalar field
publication were repaired together. The native saved-method fixture reuses one
compiled artifact for positive, zero, negative, infinite, and NaN field values.
Absent capability and optional method-result coverage are still incomplete.

The broad batch passed 18 checks in 108.71 seconds: mappings, sequence snapshots
(contents and length), scalar reads/writes/returns, child identity, saved method,
rejection loop, and priority checks. Later equal-priority physical-incumbent
refinements are covered by an additional focused batch recorded below.

That focused batch passed 9 checks in 20.57 seconds. A separate saved-graph
dependency regression passed in 3.91 seconds with planning only, no DLL compile.

The v4 saved-graph replay stopped at an explicit missing deployment shell for
`BalloonTireManagedState.dt_limit_hint` (reference 12). Resolve grounded method
identities before rebuilding a saved graph's dependency closure; resolving them
only during shell instantiation is too late. The v5 replay applies that repair
and attempts a cloudpickle checkpoint immediately before SSA/frame lowering,
using the repository's existing vendored cloudpickle. These old source-graph
replays measure the linker; they cannot establish parity for newer reducer edits.

## Checkpoint replay and physical contracts (09:03 update)

The v5 replay is terminal. Late callsite specialization required registering the
grounded method's deployment dependency at that point as well. The expanded
native fixture covers root -> helper -> saved bound method. The v6 run produced
`build/patch_sequence_replay_v6/pre-frame-link.pkl`; replay it with
`tools/replay_ssa_checkpoint.py` to skip extraction and planning. The real saved
controller's frame linker now converges in six rounds (4269 final formals).
This is a linker measurement, not a fresh-source formal-completeness report.

Aggregate result propagation now uses the same finite priority ledger. Explicit
physical storage outranks inferred callee results, which outrank provisional
types. Ties keep the incumbent. Incompatible physical outputs remain structural
findings and prevent C emission; logical Boolean/double storage and equal-size
contiguous views are compared using their physical contracts.

Read-only call inputs receive explicit scalar or elementwise span conversions
where compiled numerical buffer contracts differ. Typed pointer load/store
operations supply pointee evidence. Fresh helper outputs use private storage
and convert into the semantic result afterwards. C emission allocates these
output-pointer destinations before rendering call arguments. Public storage is
never relabelled to silence a mismatch. This corrected four scalar-read native
regressions exposed during the type-pass replacement.

Latest combined validation: 24 passed in 55.69 seconds, covering all four scalar
read cases, native scalar/span conversions, saved bound methods, result-type
propagation and incumbent priority rules. Replay v11 is running from the saved
pre-frame checkpoint with no execution deadline. Full controller parity and the
remaining all-19 implementation scope above are still open.

## Completed lowering checkpoint and collection batch (09:30 update)

Replay v11 terminated on a rejected-incumbent provenance assumption, now fixed.
Replay v12 completed and saved `build/patch_sequence_replay_v12/repository-ssa.pkl`.
Its strict scan reported 47 findings: five formal-parity groups, optional merge,
dominance, result-layout, and structural-output findings. This is a diagnostic
frontier, not permission to compile a malformed controller. Two apparent Boolean
result conflicts were missing inherited physical-buffer metadata; propagation
now preserves that metadata when the callee proof wins. Optional None results
remain conflicts. The latest priority/scalar/storage batch passed 24 in 46.54s.

Dictionary comprehensions now publish key/value rows using the shared iteration
output mechanism. Unrolled rows retain graph dependencies, lexical construction,
constant-key capacity and overwrite semantics. Retained loops preserve dict/set
schema across their collection result ports, derive arena capacity from the input
span, and use the appropriate unique update/add operation. Dictionary literals
also use update so duplicate keys keep the final value. Comprehension controls
get a source position from their target, placing keyed consumers after completion.
Regions claimed by structural filter lowering are excluded from the later
resident-value rewrite; an opaque value reference cannot itself claim ownership
of a removed numerical producer.

The combined collection batch passed 6 tests in 46.16s: native dictionary
comprehensions, mapping copies and sequence snapshots. One comprehension DLL is
reused for constant and runtime iterables, filters, empty results, duplicate keys
with different values, missing defaults and repeated invocations.

Fresh full-controller lowering is now session43484, output
`build/patch_sequence_fresh_v13`, log `artifacts/compiler_evidence/patch_sequence_fresh_v13.log`.
`tools/checkpoint_managed_ssa.py` saves source, pre-frame-link, and completed SSA
checkpoints before the strict native gate. It does not compile C or impose an
execution deadline. General generator reductions, alias-correct field rebinding,
optional presence/payload, strings/report slices and full state parity remain open.

The follow-up structural collection/storage batch passed 12 tests in 3.39s.
Its tuple-publication fixture now verifies the resident append writer and exact
collection owner; the old indexed-publication binding list is no longer the
storage interface. Guestbook validation and `git diff --check` passed.

Fresh v13 is now terminal: source extraction, planning and repository SSA
completed. Frame linking converged in 6 rounds (4305 formals at that stage),
result types in 3 rounds, with only the two genuine optional-result conflicts.
The pre-gate structural scan reported 45 findings; the full-native gate rejected
the controller. All checkpoints were preserved. Its later formal-pruning gate
reports 18 unaccounted formals across five functions, plus optional and structural
output failures. Do not compare that post-pruning count directly with earlier
pre-gate scans. No complete native controller was compiled or run.

The next generator-reduction reproduction is recorded in
`artifacts/compiler_evidence/star_max_diagnostic.log`: `max(initial, *(value * 2 for value in values),
suffix)` leaves the Starred wrapper as an unaccounted formal. No generator-max
fix is claimed. The query/reduction lowering must preserve prefix/generator/suffix
order and incumbent behavior for equal values and NaNs.

The starred-generator maximum is now lowered through the resident sequence
query mechanism. Prefix operands are compared in source order, the completed
generator arena is traversed with an explicit SSA accumulator, and suffix
operands run last. Every selection uses strict candidate-greater-than, so an
equal or unordered candidate retains the incumbent. Query dependency accounting
publishes scalar prefix producers from their numerical region, and the obsolete
flattened Max chain no longer retains the starred wrapper as a scalar formal.

`tests/test_native_generator_max.py` compiles one library and reuses it for
complete and filtered generators, empty filtered input, multiple values,
suffix wins, signed zero, and NaNs in incumbent and candidate positions. The
adjacent collection/control batch passed 8 tests in 28.95s; the final focused
lowering/native batch passed 3 tests in 16.24s, both without execution deadlines.
Saved checkpoint replay `artifacts/compiler_evidence/patch_sequence_replay_v15.log` completed without
source extraction or native compilation. The real `_propose_dt_pen` formal
parity finding is absent and the pre-gate structural scan moved from 45 to 44.
The remaining controller findings are unchanged families and no full native
controller compile or execution is claimed.

The next bounded chunk closes region-local reshape publication findings using
the tensor table as a proof receipt. A finding is discharged only when exactly
one numerical region owned by the wrapper records the reshape/view as a
non-owning alias, the aliased storage has a descriptor and a real SSA producer
in that region, and dtype and byte extent agree. The receipt preserves the
source identity, operation, owning region, resident storage identity and target
shape. A semantic function output still requires a wrapper definition;
ambiguous equal claims retain the incumbent open finding.

Saved checkpoint replay v16 completed and wrote
`build/patch_sequence_replay_v16/repository-ssa.pkl`. It records all recovered
region-view provenance and removes the 17 vector-step reshape findings plus the
one managed-advance reshape finding. The strict scan is now 26 findings, down
from v15's 44. The remaining families are four formal-parity findings, one
optional merge, sixteen definition-dominance findings, two call-result contract
conflicts, and three `run_superstep` structural outputs. This was a lower-only
checkpoint replay; no full controller C emission or native execution is claimed.

## Scalar identity and defensive keyed-read ownership (21:55 update)

The three remaining `run_superstep` structural outputs were one identity
family. Scalar `.item()` now records that its authored result is the resident
scalar rather than fabricating another formal. A declared keyed field followed
by the exact defensive spelling `mapping or {}` keeps the field's resident
table identity for membership, indexed reads, and `get(default)`. The fallback
is recognized only when the right operand is literally an empty dictionary.

Lookup ownership now includes exact source-call argument bindings as well as
numerical-region consumption. This closes the case where
`channels['power_w']` feeds `_scalar(...)` directly: the lookup is emitted in
the caller, and its instruction and function metadata retain the owning
callsite receipt. Unconsumed lookups from other linked frames remain excluded.

The focused scalar-item, defaulted-get, and call-owned-indexed-read batch passed
3 tests in 7.06 seconds. Saved replay v17 removed all three structural-output
findings and reduced the strict scan from 26 to 24, while exposing the
call-owned lookup as one new unnamed formal. Replay v18 applied the ownership
repair, removed that formal, and wrote
`build/patch_sequence_replay_v18/repository-ssa.pkl`; its strict scan reports
23 findings. The remaining frontier is four formal-parity groups (27 unnamed
values total), one optional merge, sixteen dominance findings, and two genuine
call-result contract conflicts. This is still a lower-only checkpoint replay;
the full controller gate remains closed and no native parity is claimed.

## Structured while-predicate region ownership (22:20 update)

Numerical operations nested under a structural while predicate now execute at
the predicate's dynamic evaluation site. Fresh loop composition follows the
predicate expression's computed identities while leaving captured leaf values
at their existing lexical owners. The class-surface boundary applies the same
rule to persisted shell plans: it rehomes a region only when every deployment
node is a computed identity in exactly one while predicate. Mixed regions and
equal competing owners retain their incumbent placement. The receipt records
the source loop, region, complete deployment-node set, and ownership rule.

The first replay (v19) showed that the fresh-composer repair alone could not
alter the cached control program in the saved checkpoint; its strict frontier
remained 23. After adding the persisted-plan reconciliation, the five-test
fresh/persisted/SSA while batch passed in 5.80 seconds. Replay v20 emits region
2 both before `run_superstep`'s while and on its latch; the latch feed uses the
updated carried `total` identity. Its two `%71` dominance findings are gone and
the strict frontier is 21. The artifact is
`build/patch_sequence_replay_v20/repository-ssa.pkl`. Remaining families are
four formal-parity groups, one optional merge, fourteen dominance findings,
and two call-result conflicts. No native controller compilation or parity run
has occurred.

## Exact PlanCall loop ancestry for comprehension calls (22:30 update)

Planned source calls now use `PlanCall.enclosing_loop_ids` as their primary
lexical loop ownership proof. Source spans remain a fallback. This matters for
persisted comprehensions, whose `ast.comprehension` control node may have no
line span even though the hierarchy has already recorded its exact ancestry.
Previously `_scalar(channel)` in `coerce_metrics` was hoisted to function entry,
where it read the loop-target `%0` before the projected row load.

The focused persisted-loop placement batch passed 2 tests in 4.60 seconds.
Replay v21 places callsite 9 in `loop_body`, after `%0` is loaded, and removes
the isolated `coerce_metrics` definition-dominance finding. The strict frontier
is now 20: four formal-parity groups, one optional merge, thirteen dominance
findings, and two call-result conflicts. The three `coerce_metrics` unnamed
formals remain a separate projected-storage/string-materialization family.
Replay artifact: `build/patch_sequence_replay_v21/repository-ssa.pkl`. No full
native compilation or parity run is claimed.

## Defensive mapping iteration and canonical key tokens (22:45 update)

The keyed-iterable resolver now follows only the exact defensive
`mapping or {}` form: a binary `or` whose right operand is a parentless empty
dictionary materializer. It binds `.items()`, `.keys()`, and `.values()` to the
declared mapping length and parallel key/value spans. The indexed address and
load remain intact; only their storage base is rebound.

When an `.items()` key comes from a Program ABI mapping declared with
`key_encoding: string_token`, `str(key)` retains that canonical token identity.
This is recorded separately from the mapping-slot receipt. Other values and
other encodings do not receive the identity rule.

The final focused mapping batch passed 2 tests in 4.01 seconds; the preceding
four-test batch also passed both adjacent lookup/call-placement checks while a
new assertion exposed and led to correction of an over-broad projected-load
rewrite. Replay v22 removes `%15`, `%114`, and `%66` from `coerce_metrics`'s
signature (25 formals to 22) and eliminates that formal-parity group. Frame
propagation removes the same three operands from affected callers. The strict
frontier is 19: three formal-parity groups, one optional merge, thirteen
dominance findings, and two call-result conflicts. Artifact:
`build/patch_sequence_replay_v22/repository-ssa.pkl`. No native controller
compilation or parity run is claimed.

## Total scalar dt-limit hint contract (23:04 update)

`BalloonTireManagedState.dt_limit_hint` now has one physical scalar result
contract. A positive declared integration step is returned unchanged; an
inactive zero or negative value returns `0.0`. This preserves `run_superstep`'s
authored behavior because that consumer accepts a hint only when it is finite
and strictly positive, while removing a payload/`None` union that carried no
additional controller state.

The focused scalar-contract, controller, and tire integration batch passed 4
tests in 68.62 seconds. Because the v24 pre-frame checkpoint contained the old
source graph, validation used a fresh extraction and plan. Fresh checkpoint v25
has no optional-merge finding and reduces the strict frontier from 18 to 17:
two formal-parity groups, thirteen dominance findings, and two call-result
conflicts. The repository SSA was saved before the full-native gate correctly
rejected those remaining findings:
`build/patch_sequence_fresh_v25/repository-ssa.pkl`. No native code compilation
or parity run is claimed.

## Declared region in/out ownership and write-only scalar fields (22:55 update)

Planned-region metadata now acts as the exact ownership receipt when one value
is declared in both `capture_value_ids` and `output_value_ids`. The collision
interning pass attaches that region's producer to the incumbent formal instead
of freshening the producer away when later record propagation has lost its field
label. Equal ownership therefore keeps the incumbent storage object.

Program ABI materialization also retains the exact `SetAttr` value identity for
a mutable scalar field that is written but never read when the write could not
be lowered as a scalar control expression. This supplies the field provenance
needed by the enclosing function instead of leaving the numerical result as an
unnamed propagated formal.

The focused ownership, generator-maximum, and scalar-record batch passed 8 tests
in 11.09 seconds. Replay v24 removes the `balloon_tire_managed_advance` `%182`
formal-parity group and reduces the converged frame from 4,330 to 4,290 formals.
The strict frontier is 18: two formal-parity groups, one optional merge, thirteen
dominance findings, and two call-result conflicts. Artifact:
`build/patch_sequence_replay_v24/repository-ssa.pkl`. No full native controller
compilation or parity run is claimed.

## Concrete managed-tire Metrics results (23:19 update)

`balloon_tire_managed_advance` now publishes both optional-shaped Metrics
defaults as concrete values it knows. `advanced_dt` is the exact authored `dt`
that the function advances. `dt_limit` is the largest finite float, the neutral
element for the controller's `minimum` clamps, because this core publishes no
additional post-step limit. This replaces two `NoneValue` results with one
stable float64 record ABI without changing timestep selection.

The focused eager-tire and controller batch passed 2 tests in 15.18 seconds.
Fresh source checkpoint v26 reports zero incompatible call-result contracts and
reduces the strict frontier from 17 to 15: two formal-parity groups and thirteen
dominance findings. The full-native gate's structural-output list is now empty.
The repository SSA was saved before the expected formal-parity rejection at
`build/patch_sequence_fresh_v26/repository-ssa.pkl`. No native code compilation
or parity run is claimed.

## Edge-complete loop-carried continue values (23:39 update)

Control SSA now records the carried values visible at every source `continue`.
When a predicated continue sits after a control join, lowering inserts an
edge-completion Phi: the recorded update is selected on incoming edges where
its producer dominates, and the loop-header incumbent is retained elsewhere.
The latch then resolves all reachable continue and fall-through predecessors
to one backedge value per carried identity. Both generated Phi kinds record
`tie_policy="incumbent"`.

The focused loop batch passed 4 tests in 2.55 seconds, including a new
partial-update/predicated-continue dominance regression. Saved-checkpoint replay
v27 converged in six frame rounds with zero incompatible result contracts and
reduced the strict frontier from 15 to 13 by removing the real controller's
`%570` and `%280` while-header findings. The remaining eleven dominance
findings are all at `function_exit`; the other two findings are formal-parity
groups. Artifact: `build/patch_sequence_replay_v27/repository-ssa.pkl`. This was
a lower-only replay, with no native compilation or parity claim.

## Constant while predicates and impossible exit edges (23:52 update)

`lower_while` now recognizes an exact Boolean `ControlExpression("const")` and
emits a direct, provenance-marked header branch. The condition block and its
SSA predicate remain available for authored evaluation and loop metadata, but
the CFG no longer claims that `while True` can leave through a false header
edge. A real `break` continues to target the loop exit normally. The symmetric
`False` case branches directly to the exit and leaves the body unreachable.

This fixes the shared cause of eleven controller findings: the synthetic
`while_exit -> function_exit` fallthrough had been asked to supply record fields
and scalars defined only inside the loop body. The focused while/return batch
passed 8 tests in 2.48 seconds, and the adjacent branch-compartment file passed
3 tests in 2.05 seconds. Replay v28 converged in six frame rounds, retained zero
incompatible result contracts, and reduced the strict frontier from 13 to 2.
Both remaining findings are formal-parity groups. Artifact:
`build/patch_sequence_replay_v28/repository-ssa.pkl`. This is lower-only
evidence; no native compilation or parity run is claimed.

## Stable diagnostic tokens and specialized graph provenance (00:15 update)

The controller's mass/divergence soft and rollback reasons now use stable rule
tokens. Both unresolved-report loops likewise append fixed report tokens rather
than constructing runtime f-strings; their numeric status remains in the
Metrics/error-channel payload and in `attempt_log` when requested. This removes
all `JoinedStr` nodes from the specialized controller graph instead of treating
formatted strings as unnamed ABI inputs.

Late graph-backed recovery now resolves generated specialized symbols through
the exact `source_qualified_name` receipt and indexes planned graphs by their
repository, local, and qualified spellings. Its focused provenance/formal batch
passed 7 tests in 4.08 seconds. The controller behavior batch passed 21 tests in
1.26 seconds, including exact stable-token assertions.

Fresh checkpoint v32 reduces `step_with_dt_control_used` from 44 to 37 planned
regions and the converged frame from 4,320 to 4,283 formals. The raw structural
surface shrinks from 23 unnamed values across two groups to 18; after the
production literal and dead-control passes, the full-native gate shrinks from
12 anonymous formals to 7. The two remaining groups are `run_superstep`
`[124, 126]` and `step_with_dt_control_used`
`[428, 347, 429, 477, 458]`. Artifact:
`build/patch_sequence_fresh_v32/repository-ssa.pkl`. The gate still rejects
these values, so no native compilation or parity claim is made.

## Scalar field effect versions at conditional joins (00:43 update)

`ScalarFieldWriteBlock` lowering now binds the authored `SetAttr` effect ID to
the scalar value stored at that site. Before lexical emission, each such effect
is indexed to its resident field so a conditional false arm receives the real
incumbent instead of inventing an anonymous formal. Conditional lowering also
captures arm environments independently. If a graph identity names both an arm
version and its enclosing join, the join receives a fresh SSA ID with the graph
ID retained in `source_value_id` provenance, preventing a self-referential Phi.

The final conditional-focused slice passes 6 tests in 1.96 seconds;
`py_compile` and `git diff --check` also pass (line-ending warnings only).
Saved-checkpoint replay v35 converges after six frame rounds at 4,280 formals,
reports zero incompatible result contracts, and has only the two pre-existing
formal-parity groups. It removes `%347` from the controller group and introduces
no dominance finding. Artifact:
`build/patch_sequence_replay_v35/repository-ssa.pkl`. This is lower-only
evidence; the production pruning/native gate and C execution were not run.

## Exact record-field state precedence (01:14 update)

Conditional control construction now correlates graph values by their exact
`(receiver, field)` state key. When the reducer has already emitted a
`record_field_state` Phi for that slot, the generic dotted attribute history is
not reconstructed as a second carried chain. This resolves the paradoxical
case where `metrics.hard_failure` already had the correct `%384 -> %495 ->
%496` field-state path while the flat history independently treated `SetAttr`
event `%429` as an input value. The same rule applies to the keyed
`metrics.error_channels` slot and removes `%428` in the same planning pass.

Scalar field effect blocks also support state-only publication for records
produced inside the function. Such records have no parameter in/out cell, so
the block publishes the exact lexical effect version without inventing a Store
destination; physical parameter-record writes retain their existing Store.

The combined focused batch passes 7 tests in 3.68 seconds, covering exact field
correlation, local record state, conditional versioning, and existing physical
record writes. Fresh checkpoint v40 retains 37 controller regions, converges in
six frame rounds at 4,274 formals, and reports zero incompatible result
contracts. Raw checking has the same two formal-parity groups; after production
pruning, only `run_superstep` `[124, 126]` and `step_with_dt_control_used`
`[477, 458]` remain. Artifact:
`build/patch_sequence_fresh_v40/repository-ssa.pkl`. The full-native gate still
rejects those four values; no C compilation or native execution occurred.

## Invocation-site structural BoolOp feeds (01:27 update)

Source-linked calls can be installed after ordinary control lowering has
already created a provisional formal for a complete Boolean expression. The
call-feed resolver now maps structural graph nodes through their semantic
`value_id`, resolves the exact ordered `BoolOp` operands, and replaces an
unaccounted placeholder only after reconstruction succeeds. A same-ID
instruction already resident in the caller wins immediately; ABI and frame
storage remain ineligible for reclamation. Equal-priority failure therefore
retains the incumbent instead of deleting a usable value midway through
resolution.

The focused call-feed, scalar-field, and conditional-state batch passes 5 tests
in 3.13 seconds. Fresh checkpoint v41 converges in six frame rounds at 4,273
formals, reports zero incompatible result contracts, and introduces no
dominance finding. It removes controller `%458`; the raw
formal-parity surface is now 14 values across two groups, while production
pruning leaves `run_superstep` `[124, 126]` and
`step_with_dt_control_used` `[477]`. Artifact:
`build/patch_sequence_fresh_v41/repository-ssa.pkl`. At that gate,
unmaterialized boundaries, unresolved calls, undefined operands, optional
merges, structural-output failures, and non-native findings are all empty. The
three anonymous values still reject full-native lowering before C compilation.
A broader run of `test_process_graph_function_linking.py` is not
green: 47 passed and 9 failed in 16.97 seconds. Those failures are outside the
new invocation-site BoolOp case and remain part of the dirty-tree suite
frontier; this update does not claim that file passes as a whole.

## Total report sequence contract (02:50 update)

The unresolved report is now a total `Metrics` field backed by `list[str]`.
Reducer metadata carries the imported dataclass field's aggregate kind and
`int64` text-token column through its synthesized incumbent state. Repository
sequence typing applies explicit node contracts and reaches a finite fixed
point across matching materialization and aggregate-Phi edges; established
contracts win ties and conflicting contracts remain untouched for validation.

The focused reducer/controller batch passes 3 tests in 2.45 seconds. Fresh v45
passes sequence replacement, converges after six frame rounds at 4,276
formals, and has zero incompatible result contracts. It removes controller
`%477` from the production gate. The only production-pruned anonymous formals
are now `run_superstep` `[124, 126]`. Raw checking also exposes a separate
dominance defect at controller `%478` (`while_exit` definition read in
`if_merge.11`), which is the next controller-side structural frontier.
Artifact: `build/patch_sequence_fresh_v45/repository-ssa.pkl`. No C compilation
or native parity run occurred.

## Resident report tail and sequence-field frame binding (03:51 update)

Constant resident tail slices now become loop start bounds over their base
sequence. `Metrics.unresolved_report` has an explicit mutable one-column token
table ABI, including schema-owned storage when specialization erases the list
producer. Returned-record and record-argument linking correlate every sequence
member, and an equal-priority incoming view keeps the caller's incumbent
storage.

The focused batch passes 5 tests in 6.22 seconds. Replay v54 completes without
an execution deadline, converges after six frame rounds, reaches result-type
fixed point after three rounds with zero incompatible contracts, and saves
`build/patch_sequence_replay_v54/repository-ssa.pkl`. Raw checking reports
seven findings. `run_superstep` `%126` is removed; `%127` and the missing
physical `Metrics` row descriptor are its next paired frontier. Production
pruning and native compilation were not rerun.

## Tail-domain region ownership (04:23 update)

Structured loop domains now own numerical regions removed by domain rewriting
independently of whether the complete loop body can be emitted. Deferred
record-row lookup follows exact compiler identity aliases before validating
the row layout. The focused regression passes in 4.37 seconds.

Fresh v56 converges at 4,287 frame formals and zero incompatible result
contracts. It removes the complete raw `run_superstep` formal-parity group;
`%127` no longer reaches the parent signature. The remaining `run_superstep`
finding is sequence 69's complete-row contract: the resident `Metrics` result
has 11 physical columns and the destination requires 15. Artifact:
`build/patch_sequence_fresh_v56/repository-ssa.pkl`. The production gate stops
before C compilation.

## Late result-record surface merge (04:35 update)

Each native result binding now resolves to the caller's exact resident record
identity during every frame-link fixed-point round. If the callee record has
grown, missing nonaggregate fields receive distinct caller-owned storage and
are merged by stable storage identity. Existing fields and equal-priority
aliases retain their incumbent mappings. Existing sequence fields also bind
their full storage members; an absent sequence field is left for explicit
child-table materialization rather than being flattened into a false scalar.

Replay v57 converges in six frame rounds at 4,365 formals and in three result
type rounds with zero incompatible contracts. The `run_superstep` resident
`Metrics` row now has all 14 flat columns, leaving only the fifteenth
`unresolved_report` child-table handle. The raw finding count remains six and
the fuller surface exposes a 23-value formal-parity group in
`step_with_dt_control_used`. Artifact:
`build/patch_sequence_replay_v57/repository-ssa.pkl`. No production-pruning or
native run was performed.

## Resident sequence-field projection and child snapshot helper (04:52 update)

After linked record descriptors settle, an exact one-output planned `GetAttr`
of a mutable sequence field is no longer emitted as a branch-local scalar
aggregate. The compiler removes its Call/GEP/Load projection and promotes the
descriptor's existing column identity to accounted resident arena storage.
This makes the arena dominate every mutation and call while preserving its
record-owned length, capacity, and status cells. Provenance records the retired
region rather than erasing the history of the alteration.

`lower_record_sequence_append_with_child_copy` now supplies the matching
runtime primitive for one leaf child sequence: it copies into the outer row's
fixed-stride child slice and atomically publishes the child handle and outer
length. Replay v58 removes both `%478` dominance findings and reduces the raw
surface from six findings to four. It still reports the two record-row
boundaries because link-time deferred-row wiring to this helper is not yet
installed. Artifact: `build/patch_sequence_replay_v58/repository-ssa.pkl`.

## Link-time child-pool installation (05:19 update)

Deferred record expansion now reads the authored ABI column sequence instead
of descriptor registration order. Scalar/keyed fields contribute their flat
values; a leaf sequence field contributes one handle position. The caller
realizes missing resident sequence members, replaces provisional result-frame
slots with the exact record field storage, creates flattened child pool
storage, and rewrites the append to
`lower_record_sequence_append_with_child_copy`. The same path handles a record
as the complete row and a record occupying one slot of a wider row.

Replay v64 removes both remaining record-row findings. Sequence 69 uses child
sequence 66 at handle column 14; sequence 39 uses child sequence 478 at handle
column 15. Their linked calls have arities 37 and 41 respectively, and a full
artifact scan finds no source-linked call/callee arity mismatch. The raw
frontier is now only window `%47` and the 23-value controller formal-parity
group. Artifact: `build/patch_sequence_replay_v64/repository-ssa.pkl`.

## Whole-object dead-CFG and controller formal closure (05:41 update)

Exact resident record fields now replace definition-free projection formals,
and scalar/string arguments preserved by source mutation lowering become local
constants instead of caller inputs. After whole-object call linking finishes,
`prune_constant_control_flow` evaluates only proven scalar Boolean expressions,
deletes blocks unreachable from entry, removes dead predecessor-labelled Phi
operands, and then lets the existing atomic signature transaction remove the
unused callee formals and matching actual operands. It does not discard calls
or stores on reachable paths.

Replay v67 converges in six frame rounds at 4,344 formals and three result-type
rounds with zero incompatibilities. The raw controller formal set falls from
23 in v64 to zero. The reachability pass records 18 CFG/Phi changes; its focused
suite passes 6 tests in 8.57 seconds without an execution deadline. The sole
remaining structural finding is window `%47`, the `hard_failure` projection
from returned `Metrics` receiver `%46`. That receiver lacks a resident record
descriptor in the caller, which is the next independently repairable boundary.
Artifact: `build/patch_sequence_replay_v67/repository-ssa.pkl`. No native
parity is claimed.

2026-09-08 native-frontier update: recursive late returned-record
materialization gives window `%46` the complete `Metrics` surface and binds
`hard_failure` to caller storage `%301`. Replay v68 and fresh production v69
both pass structural checking with zero findings. C emission from v69 moved
from 18 shortfalls to two after adding scalar `item` and C99 finiteness
predicates.

Those final projections are now lowered at the owner/region boundary. A
planned region may capture a record field only when the owner alias chain is
finite and terminates at a resident scalar named by the receiver record's
matching field. Equal evidence reuses the incumbent capture, and receipts keep
the projection result, receiver, field name, and resident identity. Replay
v70 has zero structural findings and zero C emission shortfalls. The C backend
also treats later definitions of a stable mutable scalar identity as
assignments to its first local declaration, fixing the three compile errors
revealed after emission became complete. The exact v70 artifact compiles at
`-O0`; its DLL is
`build/patch_sequence_replay_v70/native-o0/balloon_tire_managed_native_c.dll`.
Focused batches pass 5 tests in 4.53 seconds and 3 tests in 3.55 seconds. No
deadline was used, and optimized compilation is deferred until correctness
and native parity are complete.

2026-09-08 optional-boundary foundation: the first real-input execution probe
did not enter native code. Feed construction correctly rejected
`controller.dt_min=None` because v70 had only a float payload, and inspection
showed its `is not None` predicate had become `Const True`. ProgramABI fields
now support `optional: true`; lowering creates a Boolean presence slot and a
typed payload slot, and managed feed packing admits absence only for that
pair. The absent payload is inactive storage, never the absence discriminator,
so present zero remains distinct. `dt_min` and `dt_max` are declared optional.
The focused schema/materialization/packing batch passes 9 tests in 4.32
seconds. Predicate binding and mutable `dt_max` presence propagation remain
open before fresh full-controller lowering.

2026-09-08 optional-control update: source `is None`/`is not None` tests are
rewritten on canonical post-reduction function graphs before deployment
planning. One ProgramABI presence identity owns every test in the function;
negative tests become `logical_not`, which now has a scalar C spelling.
Optional mutable scalar assignment keeps the payload source separate from the
field slot and emits an adjacent true-store to the presence cell. Synthetic
single-exit Phis without an incumbent use conditional-result lowering.

The same planning run exposed stale aggregate ledgers after branch folding.
The fold fixed point now retains resident members and republishes only missing
call-result projections from exact stored descriptors, with replacement and
incumbent-tie provenance. Fresh v71-v74 stopped at the dangling `snapshot`
ledger while this rule was localized; replay v76 passes that point and saves
repository SSA. The final related batch passes 17 tests in 8.12s, including
the optional read and mutation executions compiled at `-O0`. No deadlines or
optimized compiles were used.

The current raw v76 frontier is six findings: formal-parity groups of 1, 3,
24, and 388 values in `run_superstep`, `step_with_dt_control_used`, managed
advance, and vector step, plus two controller function-exit dominance errors.
The saved `update_dt_max` function still loses optional flags/presence storage
while its mutable payload field crosses the linked call frame. Local mutation
lowering is green; cross-call optional provenance is the next repair.

2026-09-08 cross-call optional-presence update: graph value IDs and precompiled
SSA result IDs occupy independent identity domains. Optional materialization
now reuses a requested ID only when its resident object is already a function
argument; an instruction result with the same integer causes allocation of a
fresh physical Boolean formal. Provenance records requested and physical IDs,
and frame correlation includes the optional payload/presence role. Equal ties
retain the incumbent formal.

Replay v80 proves `update_dt_max` now owns presence formal `%37` for requested
graph ID `%24`, writes true to that slot, and receives it through an exact
five-operand source-linked call. The caller operand is accounted as the
`ctrl.dt_max` presence slot. Replay converges in six frame rounds at 4,353
formals and two result-type rounds. The remaining six findings are unchanged:
four formal-parity groups (1, 3, 24, and 388 values) and function-exit
dominance for `%263` and `%264`. No full or optimized compile was run.

2026-09-08 return-edge dominance update: synthesized return guards can retain
a source path correlation which standard SSA does not encode. At final module
composition, an exact `return_merge` slot may now recompute its pure
planned-region projection on the matching physical return edge. The repair
requires the edge receipt's source ID to equal the Phi operand, clones only a
whitelisted dependency slice with fresh identities, keeps an already
dominating incumbent, records its priority/tie provenance, and is idempotent.

Replay v81 applies exactly two repairs: `%263 -> %3390` with three cloned
instructions and `%264 -> %3395` with five. Definition dominance is clean and
the full raw structural frontier is now four formal-parity groups (1, 3, 24,
388). The non-native return-state/dominance batch passes 14 tests. No compile,
optimized compile, deadline, or native parity run was used.

2026-09-08 missing-Phi-incumbent update: a removed ProcessGraph spelling can
survive as a conditional Phi's `initial_value_id`, which previously became an
anonymous formal. Recovery now requires a unique nearest common dataflow
ancestor from the same authored identity history. Resident initial identities
win immediately; equal candidates are recorded once and left unresolved. The
repair is projected into saved Control IR and retained in final SSA metadata.

Replay v84 changes `run_superstep` Phi `%262` from missing `%260` to resident
`%256`, yielding operands `(%261, %256)`. Both specialized frame copies of the
phantom slot disappear, reducing total formals from 4,353 to 4,351 and the raw
frontier from four findings to three. Remaining formal-parity groups contain
3, 24, and 388 values. The focused file passes 7 tests; no compile or deadline
was used.

2026-09-08 post-reachability pure-feed update: scalar `item()` recovery now
uses a unique resident producer when no loop-carried Phi exists and continues
to reject ambiguous carried versions. Static reachability pruning is followed
by a second exact recovery pass before atomic signature pruning, because dead
predecessors can be the only reason the earlier dominance proof fails.

Replay v86 recovers controller `%149` and `%231` directly from `%43`, plus the
three-operation `%145`, `%146`, `%147` closure for `item() * 0.5`. The three
callee formals and matching caller operands are removed together; both sides
have arity 1,061. Final module formals fall 5,622 -> 5,619 and the raw frontier
falls from three findings to two: groups of 24 and 388. The focused file passes
5 tests. No compilation, optimization, deadline, or parity run was used.

2026-09-08 static-slice closure: exact source `ast.Slice` selectors with static
non-negative bounds and unit stride now become integer address offsets. The
full bounds and source identity remain in provenance. Dynamic bounds,
non-unit strides, negative normalization, aliases, and slices used as values
are rejected rather than guessed.

Replay v87 recovers all 24 managed-advance selectors and all 388 vector-step
selectors. Exact callee/caller arities are 438 and 26 respectively; final
module formals fall 5,619 -> 5,207 and raw structural findings fall two ->
zero. The focused batch passes 4 tests.

A fresh-source build requested explicitly at `-O0` then stopped before C
emission/compilation on one immutable physical call-input conflict: caller
`step_with_dt_control_used` `%109` was typed `float64`, while `pi_update` `%94`
is the Boolean `dt_min` optional-presence slot. Saved replay v87 has both sides
as Boolean. That fresh/replay disagreement is the next correctness frontier;
no optimized build, execution deadline, or native parity run occurred.

2026-09-08 optional-presence physical-priority update: an exact ProgramABI
presence slot now establishes physical `bool` as well as logical `bool`.
Inherited provisional physical types are replaced with an explicit receipt
containing the displaced dtype and reason. A Boolean incumbent is retained on
an equal tie.

Fresh checkpoint v89 reproduces the old `%109 float64 -> %94 bool` conflict.
Replay v90 removes it: both values are physical Boolean, their source-linked
call has exact arity 12, result typing converges in three rounds, and the raw
checker reports zero findings across 202 functions. Two focused optional tests
pass at `-O0`.

Direct emission from the saved v90 module has zero shortfalls, and one `-O0`
correctness compile produces a 1,669,632-byte DLL. No optimized compile,
deadline, or runtime parity claim was used. Artifact:
`build/patch_sequence_replay_v90/native-o0/balloon_tire_managed_native_c.dll`.

2026-09-08 standalone material-contract update: exact linked aliases of a
field with optional-presence storage now receive a typed filler when absent,
and private linked result-record workspace is typed-zero initialized when it
has no authored root parameter provenance. Unnamed ordinary inputs remain an
error. The focused contract suite passes 11 tests.

The saved v90 checkpoint now builds a 78-buffer standalone executable at
`-O0`. The parity harness resolves the manifest's checkpoint reference and has
no subprocess deadline. Its first one-frame run completes eagerly but native
exits at the new precise frontier: root planned region 3 passes Boolean result
`%308` (`Metrics.hard_failure`) to float64 feed `%47`, whose generated helper
loads it through `double *` before converting it back to Boolean. Zig detects
the misaligned load. No optimized compile or parity claim was made. Artifact:
`build/patch_sequence_replay_v90/standalone-o0/balloon_tire_managed_native_c.exe`;
report: `build/patch_sequence_replay_v90/standalone-o0/managed-dt-parity.json`.

2026-09-08 final physical-edge update: unshadowed same-ID capture occurrences
are interned to their incumbent formal before adaptation, without importing
stale occurrence provenance. Physical adaptation is repeated after final
result/aggregate/precision settlement; replay v94 records five new late casts
and a second immediate application records zero. Keyed ProgramABI members and
generated lookup helpers now carry authoritative physical element types, with
replacement provenance and incumbent tie policy. This closes the complete
two-conflict string-token frontier.

Replay v94 has zero physical conflicts and zero structural findings after six
frame rounds and three result-type rounds. The 78-buffer standalone executable
builds at `-O0`; native and eager both complete one frame with return code 0,
so the two alignment panics are resolved. Buffer comparison reports 54 matches
and 24 mismatches. Native state has 3,456 non-finite elements; `advanced` is
`0.000244140625` rather than `0.008333333333333333`, and `dt_next` is `0.0`
rather than `1.856126911235753e-05`. The next work is producer tracing for the
first numerical divergence. No optimized compile or parity claim. Replay:
`build/patch_sequence_replay_v94`; parity report:
`build/patch_sequence_replay_v94-standalone-o0/managed-dt-parity.json`.

2026-09-08 collapsed-proposal update: v94 tracing showed that a zero
controller proposal was dispatched as a second tire step and reached the
authored `(kinetic_after - kinetic_before).abs() / dt` expression. The
controller now dead-ends zero and negative final proposals before calling
physics, using the established incomplete-window return path. The focused
controller file passes 16 tests.

Fresh v96 has zero structural findings after six frame-link and three
result-type rounds. Its 78-buffer standalone compiles at `-O0`, and both sides
of one-frame parity return 0. The comparison remains 54 matches and 24
mismatches; native state non-finites fall from 3,456 to 2,056, all in batch lane
4. The earlier numerical divergence that drives `dt_next` to zero is still the
frontier. Checkpoint: `build/patch_sequence_fresh_v96/repository-ssa.pkl`;
report: `build/patch_sequence_fresh_v96-standalone-o0/managed-dt-parity.json`.

2026-09-08 indexed-store extent update: repository tensor lowering now passes
the RHS element count to `index_assign_double`. It previously passed the
destination selection count, so scalar broadcasts read beyond their one-element
source. Unknown RHS extents fail closed as lowering shortfalls. The focused
scalar/singleton/elementwise and linked-record tests pass (22 tests), and a
full v97 scan finds all 37 indexed stores consistent with their RHS shapes.

Fresh v97 converges after six frame-link and three result-type rounds with zero
structural findings. Its 78-buffer standalone compiles at `-O0`; both parity
processes return 0. Native state now has zero non-finite elements and identical
behavior across all eight batch lanes. Parity remains 54/78 because the native
controller stops after its first `1/4096` substep with a zero next proposal.
The next producer frontier is the inflated first-substep displacement/velocity
metric path. Checkpoint: `build/patch_sequence_fresh_v97/repository-ssa.pkl`;
report: `build/patch_sequence_fresh_v97-standalone-o0/managed-dt-parity.json`.

2026-09-08 in-pass shape update: a later same-ID elementwise occurrence now
inherits a missing shape from the static resident descriptor established by an
earlier instruction in the same lowering pass. A nonempty occurrence remains
the incumbent; canonical fills record identity provenance and the incumbent
tie policy. This corrects the gas-law reciprocal from count 1 to count 32.

The focused batch passes 24 tests. Fresh v98 converges after six frame-link and
three result-type rounds with zero incompatible contracts and zero structural
findings. Its 78-buffer standalone compiles at `-O0`, and both parity processes
return 0. Parity remains 54/78 because the controller stops after one substep,
but first-step tire displacement/velocity now match eager and worst state error
falls from `3.8615154563` to `0.00890879321`. The controller path producing
`dt_next == 0` is the next frontier. Report:
`build/patch_sequence_fresh_v98-standalone-o0/managed-dt-parity.json`.

2026-09-08 forwarded-result and atomic-region update: calls whose result is an
identity forwarding of their inputs now carry the settled formal-to-actual
frame into result bindings, record descriptors, and uniquely dominated uses.
The transaction records provenance and retains the incumbent on ties. Replay
v112 performs 37 reconciliations with zero structural findings. Its explicit
`-O0` build and one-frame parity run remain 54/78, while native controller
`max_vel_ever` improves from `9.5e-31` to `0.0023950195322857593` and
`dt_max` from `3e27` to `1.252599387837501`.

Tracing the remaining zero `dt_next` showed the final `pi_update` accumulator
region executing before an earlier resident-field normalization. A late
source-order pass had keyed atomic regions by their earliest member, undoing
the hierarchy planner's dependency order when a region spans early casts and
a later update. It now keys each region by its completion position. The
focused combined batch passes 5 tests. The change is upstream of the saved
SSA seam, so its measured controller result awaits one fresh replay; no second
native compile was used in this chunk.

2026-09-08 optional-control continuation update: fresh v113 validates the
atomic-region completion ordering with zero structural findings and moves
native controller `acc` from zero to `0.42714935345574623`. Exact erased
`Name is [not] None` controls now bind to the matching ProgramABI optional
presence slot. Equal-priority or ambiguous matches retain the incumbent and
all accepted bindings record provenance.

Restored branches initially exposed four dominance errors. A new continuation
pass maps exact conditional arm objects to the unique latest dominating
`conditional_carried` Phi. A companion rule proves when a callee return slot is
a same-dtype, same-shape, same-device cast of one formal, traces that exact
aggregate projection in the caller, and replaces only non-dominating uses with
the dominating actual. Replay v116 records eight Phi repairs and one exact cast
repair; structural findings fall from four to zero. The focused seven-test
batch passes and the 78-buffer executable builds at explicit `-O0`.

The one-frame run remains 54 matches / 24 mismatches. Both processes complete,
but native values remain `acc=0.42714935345574623`,
`dt_max=1.252599387837501`, `advanced=0.000244140625`, and `dt_next=0.0`.
Thus the next producer frontier is controller value/presence propagation after
valid control reconstruction. Replay: `build/patch_sequence_replay_v116`;
report: `build/patch_sequence_replay_v116-standalone-o0/managed-dt-parity.json`.
No optimized compile or process deadline was used.

2026-09-08 repeated-definition update: v116 showed one `SSAValue` object serving
as both an earlier Phi result and a later branch-local Load result. The final
Phi therefore contained two references to the same object even though its edges
required different runtime versions. A finite pass now preserves the first
definition, gives every later definition a fresh identity, and rebinds only its
dominated uses and incoming Phi edges. C emission keeps exact-object bindings
for formals, projections, and Phis in addition to authored integer ids.

Replay v118 records eight definition freshenings, six continuation repairs, and
one exact cast forwarding with zero structural findings. The combined focused
batch passes nine tests. The explicit `-O0` build emits `callout8686_0` on the true optional edge
and prior `t87` on the false edge. Parity improves to 55 matches / 23 mismatches:
`advanced` now matches the complete `0.008333333333333333` window and `dt_next`
becomes nonzero at `0.013326566940131533`. The remaining controller/physics
trajectory diverges across its adaptive substeps, which is the next producer
frontier. Report:
`build/patch_sequence_replay_v118-standalone-o0/managed-dt-parity.json`. No
optimized compile or process deadline was used.

## 2026-09-08 — sequence storage authority and constant dominance

Compiler-owned sequence arenas now receive their physical-storage identity at
descriptor creation. This prevents later call-result reconciliation from
replacing a list arena with a loop-local scalar that merely participates in the
same source transformation. The storage receipt records
`compiler_storage_identity` priority and the equal-priority incumbent rule.

A finite dominance repair also moves a unique operand-free `Const` to entry
when its original compartment cannot dominate an exact-object use. Reused
numeric identities are left for the separate identity-freshening pass, and
every accepted move records its original block/index and incumbent tie policy.

The focused sequence/dominance batch passes 7 tests in 2.68s. Saved pre-frame
replay v122 converges in 6 frame rounds and 3 result rounds with zero
incompatible result contracts. Structural findings fall from 7 to 2: both
dominance findings, the `_propose_dt_pen` arena/result collision, the
`coerce_metrics` carried result, and one `run_superstep` call result are gone.
The remaining frontier is exactly `run_superstep` call-result-unavailable IDs
228 and 298. No native compile, optimization, or execution deadline was used.
## 2026-09-08 — Targets sidechain ABI frontier

The first native/eager timestep divergence was traced to
`_energy_time_limit`: native returned `None` immediately, so the first raw PI
proposal was not capped from `0.011408210247870237` to
`2.44140625e-05`. The compiler was following its schema: the shared `Targets`
ProgramABI omitted the authored optional `energy_exchange_fraction` field and
therefore folded `getattr(..., None)` to its default.

The shared contract now declares `energy_exchange_fraction` and
`shadow_growth_max` as optional float64 scalars. The default exact-source audit
now retains the presence inputs, guards, keyed energy/power reads, finite checks,
and final arithmetic with zero structural shortfalls. Two focused
contract/presence checks pass in 4.45s. The first `-O0` native attempt no longer
emits the incorrect constant-None helper; it stops at the next correct gate:
optional function returns require explicit payload/presence representation, and
three recovered structural values still lack caller accounting. Full controller
parity remains open.

## 2026-09-08 — optional scalar ABI and exact predicate recovery

Mixed scalar/`None` return Phis now lower to a physical `(payload, presence)`
pair. The linked Call receives the same aggregate result, and exact None tests
consume its Boolean presence projection. The transformation is finite and
idempotent, records provenance for the callee and caller rewrites, and applies
the established equal-priority rule: the incumbent wins.

Record-field discovery now gives intrinsic
`getattr(obj, "field", default)` the same ProgramABI identity as attribute
syntax. Structural recovery also claims anonymous formals whose exact graph
nodes are recoverable control predicates, supports direct repository unary
predicates such as `isfinite`, and moves each recovered closure before its
unique consumer. This removes `_energy_time_limit`'s two membership-predicate
formals, finite-predicate formal, and unaccounted fraction formal.

The exact helper has zero formal-parity findings, zero optional-merge findings,
and zero C-emission shortfalls, and it compiles successfully at explicit
`-O0`. The focused optional/extraction batch passes 3 tests in 19.46s, and an
adjacent structural/call/optional batch passes 10 tests in 8.44s, without an
execution deadline. A fresh full-controller replay is still required before
changing the saved 55/78 parity measurement.

## 2026-09-08 — live structural settlement and call-feed identity

Saved replay v123 proved that `run_superstep` structural findings 228 and 298
were failed recursive probes whose identities had disappeared from every final
SSA surface. Structural settlement now retains a probe whenever its id remains
a formal, operand, required source, or authoritative output, and records only
the non-live cases as settled. Replay v123 converged in 6 frame rounds and 3
result rounds with zero incompatible contracts and zero structural findings.

The first v123 native attempt exposed `float('inf')` in storage-index analysis.
Storage requirements now use the shared exact-integer classifier, so nonfinite
and nonintegral floating constants cannot become static address indices. The
next C-emission gate exposed four stale scalar `item()` identities and one
operand-only BoolOp call feed. Proven scalar identities now rebind every stale
operand object to one unambiguous formal or definition, update call feed ids,
and restore exact source-parameter naming when that makes a pruned parameter
live. Source-linked BoolOp feeds reconstruct from their ordered source operands
instead of accepting an operand-only placeholder. Every alteration records its
priority and provenance; ambiguity retains the incumbent.

The combined focused batches pass 9 tests and then 7 tests. Replay v125 again
converges in 6 frame rounds and 3 result rounds with zero incompatible result
contracts and zero structural findings. Exact inspection shows `%459` defined
by `LOr` immediately before `pi_update`; the stale `%84`, `%224`, `%15`, and
`%21` operands are absent. The 82-buffer standalone builds successfully at
explicit `-O0`.

Both sides of the one-frame live check return 0. Parity is 59/82: 23 buffers
remain different, concentrated in material telemetry/state, controller
`dt_max`/`acc`/`max_vel_ever`, and `dt_next`. Native `max_vel_ever` is
`0.3159734017361334` versus eager `0.009045569831777825`; native `dt_next` is
`0.013326566940131533` versus eager `1.85612691e-05`. The next frontier is the
first material/telemetry divergence within the adaptive substeps. Report:
`build/patch_sequence_replay_v125-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

## 2026-09-08 — returned keyed-record capacity closure

Trace inspection localized the empty adaptive-step energy metrics to a zero
capacity cell. `balloon_tire_managed_advance` authored and lowered all three
keyed values (`maximum_substep_displacement_m`, `energy_j`, and `power_w`), but
record-constructor analysis treated packaging the dictionary as a possible
mutation and discarded its exact three-key capacity. Linked-frame propagation
also admitted only input ABI parameters, so the returned `Metrics` descriptor
could not reach the outer activation that initializes private capacity cells.

Schema-known record construction is now classified as nonmutating for static
sequence bounds. Exact returned-record sequence descriptors propagate with
their bounds and provenance. Because call-table iteration order is not a
semantic order, transitive propagation closes over the linked-call graph to a
finite fixed point. Each edge records its source sequence, destination
sequence, exact ABI priority, and incumbent tie policy; an equal bound keeps
the incumbent.

The focused batch passes 4 tests. Replay v127 converges in 6 frame rounds and
3 result rounds with zero incompatible contracts and zero structural findings.
The capacity-3 chain is present at advance sequence 213, step sequence 1668,
superstep sequence 733, and root-window sequence 349. The 82-buffer standalone
build succeeds at explicit `-O0`, and generated C initializes the root capacity
to 3.

The one-frame live result remains 59/82, with the same 23 mismatches and native
values as v125/v126. The descriptor/capacity closure is therefore necessary
but not sufficient: the next frontier is the runtime alias path from root
sequence 349 through superstep 733 and step 1668 to advance sequence 213, or
the first store/length transition after that path. Report:
`build/patch_sequence_replay_v127-standalone-o0/managed-dt-parity.json`. A small
three-entry synthetic constructor also exposed a separate control-effect
scheduler cycle; it is recorded as a later frontier rather than mixed into
this storage fix. No optimized build or process deadline was used.

## 2026-09-08 — exact structural membership negation

The capacity-3 alias chain was live: a native trace showed `_energy_time_limit`
receiving the correct `energy_j` and `power_w` keys and values. The remaining
early return came from structural recovery of source `not in`. It emitted
`contains != False`, which is identical to `contains`, so both present keys
were misreported as absent.

Structural membership recovery now emits unary `LNot` over the exact native
contains result, matching the normal source-to-SSA path. Each recovered
negation records exact-source priority and incumbent tie policy. The existing
real `_energy_time_limit` regression proves both membership scans feed two
`LNot` instructions directly. The combined capacity/membership batch passes 5
tests.

Replay v128 converges in 6 frame rounds and 3 result rounds with zero
incompatible result contracts and zero structural findings. Both production
membership negations retain their exact contains operands and provenance. The
82-buffer standalone builds successfully at explicit `-O0`.

Native and eager both finish one frame and return 0. Parity remains 59/82, but
the adaptive trajectory changes substantially: native `max_vel_ever` moves
from `0.3159734017361334` to `0.017680656115454992` versus eager
`0.009045569831777825`, and native `dt_next` moves from
`0.013326566940131533` to `6.104353894230951e-06` versus eager
`1.85612691e-05`. `advanced` remains an exact match. The native correctness
run now takes about 12 minutes 22 seconds because the live energy limit opens
the many-substep path; it completed naturally without a process deadline.
Report: `build/patch_sequence_replay_v128-standalone-o0/managed-dt-parity.json`.

The next concrete correctness frontier is the authored mapping effect order
seen in the trace: the first `maximum_substep_displacement_m` store occurs
before a later sequence clear, which erases that entry while the subsequent
energy and power stores survive.

## 2026-09-09 — dynamic mapping literal effect order

Dynamic dictionary rows were scheduled at each stored value's producer
position. That position can be much earlier than the dictionary literal: in
managed advance, `maximum_displacement` was computed before later conditionals,
so its row store ran in `entry`, while the literal's clear ran afterward in
`if_merge.1`. The later energy and power rows survived, but displacement was
erased.

Each dynamic mapping row is now anchored to its exact key AST node inside the
literal. Data dependencies still require the value producer first, while the
key positions preserve clear-before-rows and authored row order. Lexical
sequence installation also reuses the shared atomic-region completion rule
instead of independently ordering a region by its earliest member. Equal
positions retain stable incumbent order.

The focused batch passes 3 tests. It includes one explicit `-O0` native
artifact executed three times with different inputs, proving clear-before-all
rows and correct reuse of the same resident buffers. Replay v129 converges in
6 frame rounds and 3 result rounds with zero incompatible contracts and zero
structural findings. Production sequence 213 now has exactly `clear`,
displacement store, energy store, and power store in that order in
`if_merge.1`; no row store remains in `entry`. The 82-buffer standalone builds
at explicit `-O0`.

Native and eager both return 0. Parity remains 59/82, and the measured native
controller values are byte-for-byte unchanged from v128. The restored
displacement channel is therefore correct but is not the active limiter for
this fixture. Report:
`build/patch_sequence_replay_v129-standalone-o0/managed-dt-parity.json`. The
next measurement frontier is a short-window trace of the first divergent
substep. No optimization or process deadline was used.

## 2026-09-09 — local returned-value binding in C

The module C emitter prebound every native output object to its caller-owned
`outN` buffer before emitting the function body. When one SSA value was both
consumed internally and returned, exact-object lookup kept selecting `outN`
after the numeric binding advanced to the producer's local `tN`. Managed
advance consequently emitted `t308 = !*out307` instead of `t308 = !t307`,
turning a finite-state result into a critical rejection from stale output
storage.

Locally defined output objects no longer receive that premature exact-object
binding. Internal consumers follow the local producer, while `Ret` remains the
sole publication step. Formal aliases and preallocated aggregate projections
retain their existing exact bindings. A focused batch passes 4 tests,
including an explicit `-O0` compiled regression which returns a Boolean and
negates it before publication.

The already clean v129 SSA checkpoint was reused because this change is solely
in C emission. Its 82-buffer v130 standalone builds at explicit `-O0`; the
production branch now emits `t308 = !t307`. Native and eager both return 0.
Parity remains 59/82, but native telemetry changes from 0 successful and 158
critical substeps to 158 successful and 0 critical substeps. Advanced time
still matches exactly. Native now performs 158 successful substeps and leaves
completion false, while eager performs 157 and completes, making the adaptive
loop completion boundary the next frontier. Report:
`build/patch_sequence_replay_v130-standalone-o0/managed-dt-parity.json`. No
optimization or process deadline was used.

## 2026-09-09 — exact loop-record views and state-based fixed points

Source-linked record results inside loops now retain their exact physical
surface until checked header and exit field Phis are formed. Richer returns use
a separate incumbent-schema projection for the loop merge, while exact
sequence-row consumers keep the full record. Receipts record the involved
identities and `incumbent_on_equal_priority`.

Frame linking now detects repeated complete states by hashing its function,
record, sequence, and call-table ledgers. A recurrence records its digest and
cycle period and retains the incumbent state, without a round limit or process
deadline. Final source-linked call operands are also reconciled from exact
frame receipts after aggregate legalization.

The focused batch passes 2 tests. Replay v131 converges in 6 frame rounds and
3 result-type rounds with zero incompatible contracts and zero structural
findings. The explicit `-O0` standalone has 70 public buffers; native and eager
both return 0, with 49/70 matching. Native telemetry remains unchanged:
completion is false after 158 successful attempts, versus eager completion in
157. The next frontier is the managed-window completion/result projection.
Artifact: `build/patch_sequence_replay_v131/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v131-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

## 2026-09-09 — terminal record-field Phis and scalar record ABI

The managed completion mismatch came from recursive record expansion. The
loop merge correctly projected a richer returned `Metrics` record onto the
incumbent schema, then loop-alias propagation mistook each physical field Phi
for another conceptual record and registered the full descriptor under that
field id. Return expansion therefore multiplied the physical layout and
shifted `error_channels.length` into the public `hard_failure` slot.

Every fieldwise record Phi now carries terminal provenance on its instruction
and SSA result. Both record materialization and loop-alias propagation stop at
that state. This preserves ordinary conceptual record discovery while making
the transformation graph finite under repeated frame-link rounds. The
existing incumbent-schema projection and equal-priority incumbent rule remain
unchanged.

After removing the multiplied layout, structural checking exposed
`advanced_dt` as a `None`/singleton-array Phi. Late result refinement had
restored a `(1, 1, 1)` shape on storage declared scalar by its record
descriptor. A final idempotent normalization now makes declared scalar record
storage authoritative immediately before optional lowering and records the
old shape and ownership provenance. Optional lowering consequently produces
an explicit scalar payload and presence Phi.

Four focused tests pass, including a direct idempotence/provenance regression,
and the changed modules pass `py_compile`. Replay v136 converges after 6 frame
rounds and 3 result-type rounds with zero incompatible contracts and zero
structural findings. The final `run_superstep` return has 13 values and only
three 11-field conceptual layouts. Its 70-buffer standalone builds at explicit
`-O0`.

Native and eager both finish one frame. Parity is 49/70, while completion now
matches at `1.0`; v131 produced false completion. Native still uses 158
successful attempts versus eager 157. The next frontier is the numerical
controller trajectory, beginning with native `controller.dt_max=3e27` versus
eager `0.3316540644527175` and native
`dt_next=6.104353894230951e-06` versus eager
`1.856126911235753e-05`. Artifact:
`build/patch_sequence_replay_v136/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v136-standalone-o0/managed-dt-parity.json`. No
optimization or process deadline was used.

## 2026-09-09 — deduplicated record results and formal passthrough

The native controller's `dt_max=3e27` came from a stale returned `Metrics`
descriptor. `balloon_tire_managed_advance` legitimately returns `max_vel` and
`max_flux` from one SSA value. Aggregate output legalization removes those
redundant result projections and folds a view of callee formal `%194` back to
caller actual `%1657`. The record pass ran before that rewrite and later
resolved an output-identity alias instead of the exact call-result record, so
the `max_vel` field stayed on a zero-filled synthetic slot and later fields
were positionally shifted.

The aggregate legalizer now retains provenance for each removed output:
logical position, caller projection, canonical callee formal, caller actual,
priority, and tie policy. A final record transaction joins those receipts with
the surviving aggregate positions, updates the exact result descriptor and
its resident alias by storage identity, and rebuilds dependent call-frame
bindings. Repeated result identities select the first physical resident and
equal priority retains the incumbent. In the production artifact, records
`%370` and `%1624` now both bind `max_vel/max_flux` to `%1657`, `div_inf` to
`%1660`, and the `coerce_metrics` and `update_dt_max` calls consume those exact
residents.

Five focused tests pass, including direct repeated-position and formal-
passthrough frame regressions. The changed compiler modules pass `py_compile`.
Replay v138 reaches the same finite fixed point in 6 frame rounds and 3 result
rounds, with zero incompatible contracts and zero structural findings. The
explicit `-O0` 70-buffer build succeeds and both parity executions complete.

The buffer match count remains 49/70, but the repaired numerical path is
measurable: native `controller.dt_max` changes from `3e27` to
`0.1696769611042683`, and native `max_vel_ever` changes from approximately
zero to `0.017680656115454992`. The next defect is the independent optional
presence resident: SSA `%121` remains false while eager is true, and the step
call still routes split `%1784` rather than the incumbent controller presence.
Artifact: `build/patch_sequence_replay_v138/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v138-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

## 2026-09-09 — stable optional-field ownership across linked frames

The remaining false `dt_max` presence was introduced by frame ownership.
`pi_update` formal `%95` and `update_dt_max` formal `%37` both describe the
same `STController.dt_max` presence slot, but only the former was already in a
record descriptor. The latter fell back to its callee-local numeric identity,
won a `distinct_owner` transition over the shared slot, and propagated that
synthetic split through the outer call graph.

The owner classifier now uses record identity, field name, optional role,
dtype, and scalar shape when that complete ProgramABI evidence exists. This
unifies payload/presence storage across receiver spelling and local IDs while
keeping the two roles distinct. It does not generalize the rule to ordinary or
aggregate fields. A first broad version did so and v139 correctly failed on a
conflicting sequence descriptor; the implementation was narrowed rather than
adding a reverse transition to the priority graph.

Twelve focused optional and transformation-priority tests pass, as does
`py_compile`. Replay v140 reaches 6 frame rounds and 3 result rounds with zero
incompatible contracts and zero structural findings. It removes three
synthetic formals and proves the direct route
`update_dt_max %37 -> step %520 -> run %51 -> root %121`. The explicit `-O0`
70-buffer build succeeds and both parity executions complete.

Parity advances from 49/70 to 50/70 because public presence `%121` now equals
eager `True`. The numerical buffers do not move. The next concrete defect is
the duplicate root controller storage: argument-only `%394-396` and
`%451-453` feed later `run_superstep` formals even though root record `%0`
declares `%120`, `%122`, and `%123` as the authoritative fields. Artifact:
`build/patch_sequence_replay_v140/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v140-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

## 2026-09-09 — canonical scalar shape across linked call frames

Frame candidate matching previously treated incidental SSA shape as physical
identity. Declared rank-zero `STController` fields had shape `()`, but
propagated formals sometimes retained `(1,)`; the mismatch allocated duplicate
receiver-field arguments for `dt_max`, `acc`, and `max_vel_ever` in successive
outer frames.

The linker now canonicalizes declared ProgramABI scalar rank-zero storage to
shape `()` everywhere that frame identity is established: owner keys,
candidate keys, cloned receiver members, and provenance receipts. This rule
does not apply to sequences, spans, keyed tables, or other aggregate storage.
That narrow storage condition preserves the lesson from v139, where treating
all fields as globally shared caused a conflicting sequence descriptor.
Transformation direction is unchanged and equal-priority ties keep the
incumbent.

Ten focused scalar, optional, and priority tests pass, and `py_compile`
passes. Replay v141 converges in 6 frame rounds and 3 result rounds with zero
incompatible contracts and zero structural findings. Its 4542 formals are 53
fewer than v140, and the stale singleton controller formals are absent.

The explicit `-O0` native build exposes 52 buffers instead of 70. Native and
eager both execute one frame to completion. Parity is 40/52 with 12
mismatches, compared with v140's 50/70 with 20 mismatches. All authoritative
numerical values are unchanged from v140, proving that this chunk removed
redundant public identities without altering execution. The next measured
frontier is the authoritative controller/result divergence: `%120` is
`0.1696769611042683` native versus `0.3316540644527175` eager, and `%45`
(`dt_next`) is `6.10435389e-06` native versus `1.85612691e-05` eager.
Artifact: `build/patch_sequence_replay_v141/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v141-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

## 2026-09-09 — sequential conditional arm identities join lexically

The next authoritative `dt_next` trace found a distinct conditional-version
bug in `STController.pi_update`. The `dt_max` branch produced a valid Phi, but
the following `osc` branch consumed the older pre-clamp identity. Source graph
ids can be reused across conditional arms; publishing the merge only under
the incoming and merged ids left the arm ids pointing at branch-local values
after control rejoined.

The SSA lowerer now records the merge as the current lexical version of all
four identities in a carried alias: true arm, false arm, incoming value, and
merge. Later definitions still replace their own entries. This ensures a
following conditional sees the dominating join and preserves the existing
forward transformation order and incumbent-on-equal-priority rule.

Thirteen focused tests pass, including sequential and nested conditionals,
reused graph identities, definition dominance, and transformation priority;
`py_compile` also passes. Replay v142 converges in 6 frame rounds and 3 result
rounds with zero incompatible contracts and zero structural findings. The
emitted production C now passes post-`dt_max` resident `%89` into the `osc`
branch and uses it directly when `osc` is false.

The explicit `-O0` build retains 52 buffers, and native and eager both finish
one frame without a process deadline. Parity stays 40/52; all mismatch arrays
are byte-for-byte unchanged from v141 because this fixture does not activate
the corrected clamp transition. This is a verified general control-flow fix,
not a numerical-parity claim. The next measured frontier is still the first
active divergence feeding the controller and material state. Artifact:
`build/patch_sequence_replay_v142/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v142-standalone-o0/managed-dt-parity.json`.

## Explicit-axis min/max remain reductions

Tensor SSA no longer classifies a two-operand `min` or `max` as a Python
two-value comparison when the call carries an explicit `axis` or `dim`.
Captured structural axis operands now retain reduction semantics and lower to
`reduce_dim_double`; two actual data operands without an axis retain the
existing elementwise/scalar path.

The production tire column-10 expression now reduces its `(8, 4, 144)` Y
slice to `(8, 4)` before storing. The complete tensor metadata test file passes
(26 tests), replay v143 converges in 6 frame and 3 result rounds with no
structural findings, and its explicit `-O0` one-step parity improves from
49/52 to 50/52 by making `material.output` match eager. The two remaining
one-step failures are stale read-only `controller.dt_max` frame aliases.
Artifact: `build/patch_sequence_replay_v143/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v143-one-step-o0/managed-dt-parity.json`.

## Canonical read-only scalar frame aliases

Linked-frame scalar identity now treats a declared ProgramABI field without
an optional payload marker as the ordinary payload slot. Explicit presence
storage remains separate. Writable candidates outrank read-only copies,
non-callsite candidates break the next priority level, and equal priority
retains the first incumbent. Candidate collapse is limited to scalar storage;
aggregate candidates remain plural.

A final entry-signature safeguard removes only generated duplicate scalar
fields in callerless functions when no instruction or call-frame receipt uses
them. Focused role, priority, and liveness tests pass. An adjacent span test
still fails under the prior mechanisms as well and is recorded as an existing
dirty-worktree failure rather than evidence against this change.

Replay v145 converges in 6 frame and 3 result rounds with no incompatible
contracts or structural findings. The production root has one `dt_max`
payload `%120`, one presence `%121`, and passes both directly to
`run_superstep`. Its explicit `-O0` one-step artifact removes four redundant
controller aliases and reaches 48/48 native/eager buffer parity. Artifact:
`build/patch_sequence_replay_v145/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v145-one-step-o0/managed-dt-parity.json`.

The full-window v145 artifact also builds at `-O0` with 48 buffers. Both
executions complete the requested window and remain finite. Parity is 38/48:
the remaining mismatches are accumulated controller, telemetry, material, and
`dt_next` results after native takes 158 successful substeps and eager takes
157. The canonical alias correction therefore survives the full trajectory,
while the earliest repeated-substep numerical divergence remains open.
Report: `build/patch_sequence_replay_v145-standalone-o0/managed-dt-parity.json`.

## Unique field provenance across source call edges

The linked value-ABI fixed point now carries a declared `(record, field)`
identity through an exact source call edge while that identity is unique.
Conflicting fields feeding one formal permanently mark its field provenance
ambiguous; dtype, rank, shape, and storage continue to settle independently.
Unique field provenance is attached before region lowering so a span such as
`state.height` keeps its identity in both the callee and its planned region,
without labeling a generic helper from first-arrival order.

Five focused call-frame and storage tests pass, including unique and ambiguous
span cases. Replay v146 retains the v145 production counts and zero-finding
gate, and its explicit `-O0` one-step artifact remains 48/48. Artifact:
`build/patch_sequence_replay_v146/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v146-one-step-o0/managed-dt-parity.json`.

The v145 native trace shows that the remaining full-window pointwise mismatch
does not begin at a discrete compiler edge: first-step state error is at most
`6.37e-13`, and traced values stay within parity tolerances through 31
substeps. `max_vel` crosses tolerance at substep 32 and perturbs the adaptive
controller at substep 33, after which the two valid trajectories shadow
different step schedules. Numerical reproducibility policy remains open.
