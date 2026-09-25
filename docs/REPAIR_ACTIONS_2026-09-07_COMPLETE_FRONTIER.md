# Complete remaining repair work package

## Scope and evidence boundary

This is one coordinated implementation work package, with sequential validation
checkpoints, not a series of optional follow-up tasks. Completion means the real
managed input can enter the native adapter, all required operations and returns
survive lowering, the strict gates pass, and native results/state match Python.
Zero fabricated formals alone is not completion.

Baseline: `artifacts/compiler_evidence/keyed_replace_full_formals_20260907.log` and
`build/full_formal_diagnostic/{formals.json,repository-ssa.pkl,resolved-process-graph.pkl}`.
That completed run has19 formals, zero undefined operands/unresolved calls.
IDs below refer to that baseline, not an anticipated new lowering.

Current working edits go beyond that baseline. They recover unnamed dictionary
materializer copies and chained writes, prevent duplicate numerical-region
ownership, preserve mapping mutation kinds, propagate row schemas, and correlate
same-arena dictionary Phis. They are **not yet validated as a complete repair**.
The new `test_native_mapping_copy_effects.py` currently fails because direct
`.get` return expressions disappear: the emitted root returns no values while
metadata lists four structural-output shortfalls. This must be repaired, not
xfail-marked or hidden by checking only internal buffers. The literal dictionary
path also currently shows both literal initialization and an add for the same
source row; reconcile ownership rather than relying on deduplication.

Already verified and to preserve: source-ordered rejection reads/accept calls;
four scalar return publications, including1482 <- Cast9564 <-593 with earlier
read1378; and checked resident replacement that preserves contents on failure
and self-copy. Those successes are regression obligations, not proof of the
remaining mechanisms below.

## Every baseline formal, traced

| Function / ID | Source operation and actual failure | Action |
|---|---|---|
| step440 | First formatted mass-error reason has no string producer. | A11 |
| step448 | First formatted divergence reason has no string producer. | A11 |
| step570 | Formatted mass rollback reason has no string producer. | A11 |
| step550 | Formatted divergence rollback reason has no string producer. | A11 |
| step496 | SetAttr effect identity used as the initial of duplicate Phi1901; actual keyed state is576/577. | A2–A4 |
| step497 | SetAttr effect identity used as the initial of duplicate Phi1902; scalar ledger592/593 now reaches the return. | A4 |
| step417 | `floor_reasons = tuple(reasons)` has no snapshot producer; aliasing the mutable list would be incorrect because clear follows. | A7 |
| step418 | `len(floor_reasons)` consumes the missing snapshot. | A7 |
| step554 | Field ledger's dt_limit Phi has no emitted definition;555 merges554 on both edges. Source inputs516/521 are reads of the same receiver field. | A4 |
| step526 | Value-position `metrics.osc_flag or metrics.stiff_flag` lacks its Boolean producer at pi_update. | A10 |
| run_superstep291 | `tuple(sorted({float(value) for value in event_boundaries if ...}))` has no resident producer. Baseline event_boundaries is empty. | A8, A6 |
| run_superstep164 | `callable(hint)` is unresolved despite the state class's authored dt_limit_hint method. | A9 |
| run_superstep57 | `hint()` is misclassified as a NoneType intrinsic instead of a bound method call. | A9 |
| run_superstep245 | Returned metrics field `.get('dt_unresolved',0.0)` is disconnected from physical keyed storage. | A3 |
| run_superstep115 | `first.unresolved_report` lacks a record/default projection. | A11, A12 |
| run_superstep117 | Slice `[1:]` of that report lacks a resident slice representation. | A11, A12 |
| _no_exchange_observed11 | `channels['power_w']` is disconnected and consumed by an already-hoisted _scalar call. A later false Boolean operand does not authorize deleting that call's effects. | A3, A10 |
| _propose_dt_pen49 | Starred generator of error ratios is passed as one fabricated scalar to max; no completed reduction reaches the consumer. | A6 |
| coerce_metrics0 | Dict-comprehension iteration variable `channel` escapes as a formal because its producing loop is absent. | A6, A5 |

The table accounts for exactly19 baseline formals: step10, run_superstep6,
and three singleton helpers. Optional-presence restoration may expose additional
live findings; do not promise a fixed monotone count reduction.

## Specific actions

### A1. Make completion checks cover effects and returns

In `ssa_self_check.py` and the final `fortran_c_shell.py` gate, reject unresolved
required structural outputs, missing source return slots, and dropped required
effects, alongside undefined operands/unaccounted formals. Retain source-to-SSA
ownership receipts for mutations and return slots. Detect duplicate ownership
and check CFG placement/dominance against source scopes. A source query or store
must not disappear merely because its old consumer was also dropped.

Validation: the current empty-return mapping regression must fail the compiler
gate until four outputs are defined. Separate probes must catch a missing
field write and a hoisted read even when every formal is accounted for.

### A2. Finish dictionary construction and keyed effects

In `_field_slot_ops`, `_source_mapping_mutations`, lexical mutation placement,
and `precompile_to_ssa`, use exact declared arena identities for materializers;
do not require a lexical spelling. Chase IndexedStore versions to their storage
root. Preserve mapping key/value contracts through `dict(source)` and the
proven `mapping or {}` case. Claim each store once, remove its duplicate region
instruction, and recompute region dependency signatures from retained operations.
Keep mapping updates as mapping updates during argument expansion.

Schedule constructor initialization/copy before its indexed writes, on every
execution of the authored statement, including inside loops. Give literal rows
one owner; do not initialize them both eagerly and through an add effect.
Place keyed reads at their authored position, including direct return operands;
do not restrict them to numerical-region consumers or loop-owned keys.

Validation: make `test_native_mapping_copy_effects.py` pass without weakening
its four Python-result comparisons; extend with construction inside a loop,
empty input, repeated writes to one key, and read-before/read-after checks.
The real floor branch must contain copy493 followed by stores494/495.

### A3. Preserve keyed-field assignment and all Python aliases

Link GetAttr and keyed operations on call-returned records to the actual
length/key/value storage. Baseline record433/434 has1375/1376/1377; source
projection461 currently names an independent anonymous arena. Correlate through
receiver and storage receipts, including calls, returned records, and record
rows in sequences. Reuse existing lookup/default/contains helpers.

Represent mapping object identity independently from a receiver field's current
binding. `field = rhs` must bind the field to rhs's logical object: later writes
through *any* rhs alias must be visible; aliases retained to the previous field
object must still see the previous object. Use descriptor indirection/alias-aware
publication or a proven equivalent representation. The checked copy helper is
usable for physical ABI publication only when ownership/escape analysis proves
that copy preserves those semantics. Rebinding one local is not sufficient.
Obtain capacity from the actual bound storage contract, never an unrelated
compiler-local arena. Publish on the assignment/return edges required by the ABI.

Validation: old-field alias, two rhs aliases, mutation after assignment,
conditional assignment, return through a helper, and repeated native calls.
Assert both field content and alias-visible state against Python.

### A4. Complete scalar field SSA and eliminate duplicate effect carries

In the reducer, distinguish field reads from field writes. Resolve first reads
to the existing physical field; do not fabricate an initial for an observation
inside an arm. Merge actual field versions at joins and loop headers/backedges,
and carry separate state on break, continue, and each return edge. Retain required
write values through graph cleanup. Seed from the field's exact storage identity.

Finish `ssa_record_return_state.py` beyond the current conditional-Phi-only
selection: terminal constant assignments and first-observed fields must publish
their edge-specific state while earlier reads keep their earlier version.
Repair clamp_events augmented assignment so the Add result is stored/published,
not replaced by the input alias. Repair554 using its resident dt_limit initial
and actual writes; coalesce redundant read-only versions where proven.

Once field semantics are validated, remove the effect-token path that treats
SetAttr as a data value, remove duplicate1901/1902, and run paired signature
pruning. Remove496/497 because they are redundant, not by accounting exemption.

Validation: all three clamp sites, multiple retry iterations, first write inside
an arm, nested conditionals, every terminal outcome, and the existing scalar
read-order and returned-field tests with their xfails removed when fixed.

### A5. Repair record call/return identity and physical ABI correlation

Propagate identity-returning relationships such as coerce_metrics and keep
through caller/callee signatures and record-field layouts. Repair
`test_child_record_conditional_write_reaches_return` so it passes natively,
including the pre-write captures and returned field, without an xfail.

Replace the `split_from_unproven_alias` advance-result substitution with an exact
callee-return-slot/storage proof or load the actual returned component. Do not
infer unsoundness merely from scalar dtype: internal native arguments are
`void *` and are loaded through storage addresses. The unresolved issue is the
correct identity/version on each retry, not scalar-by-value calling convention.

Validation: a returned metric deliberately differs from the similarly named
state field; repeat across retries and aliases. Native values must follow the
actual return, never coincidental equality in the current advance function.

### A6. Lower comprehensions and starred generator reductions

Retain comprehension iteration domains, row variables, filters, and effects in
Control IR. Dict-comprehension lowering must load keys/values in its loop,
execute _scalar on each value, and store the resulting rows in the actual
destination. Preserve coerce_metrics's authored identity and normalization
semantics rather than replacing the whole function with an assumed no-op.

For `_propose_dt_pen`, fuse the generator into a loop-carried max accumulator
with the authored scalar operands and final1.0 seed/argument semantics. Preserve
Python's order-sensitive floating-point/NaN comparison behavior. Its consumer
must run after the loop, and keys/values must come from the bound error_limits
and error_channels tables. Do not pass a generator handle as a scalar.

Validation: empty/multiple channels, non-unit limits, absent keys/defaults,
filtering, and exceptional floating-point cases; native/eager comparison.

### A7. Materialize tuple snapshots and lengths

Lower `tuple(reasons)` as a snapshot of the contents and length at that statement,
not an alias of reasons. Its lifetime must survive the following clear.
Lower len from that snapshot's logical length. Merge its descriptor/contents
across the floor arm without an opaque fabricated scalar Phi.

Validation: several reasons, snapshot, clear original, subsequent append,
and both floor outcomes across reused buffers. The violation count must equal
the pre-clear snapshot length, and417/418 must have real producers.

### A8. Resolve boundary collection specialization and general lowering

For the baseline's proven empty event_boundaries, produce a real empty boundary
sequence and prune its iteration; do not pass291 as an unexplained arena.
For nonempty inputs, use A6's filtered collection lowering, unique numeric rows,
and deterministic sorting before tuple materialization. Preserve the source
filter and boundary order rather than specializing all callers to empty.

Validation: empty, duplicate, unsorted, out-of-window, and several valid
boundaries. Assert the sequence and actual attempted step sizes.

### A9. Resolve bound method capability and calls

Resolve state.dt_limit_hint through the authored state class and receiver ABI.
Fold callable only when that exact callable capability is proved; retain the
method body and receiver binding for hint(). Remove the false NoneType intrinsic
classification. Preserve absent/default behavior for receivers without the
capability and optional returns from the method.

Validation: positive hint caps the first trial, absent/None/nonpositive/nonfinite
hint follows source behavior. Resolve164/57 through actual producers.

### A10. Lower value-position BoolOps with source short circuit

Produce526 from the actual osc_flag/stiff_flag fields at the accepted call.
For general and/or expressions, use control and merge the selected operand
value; do not eagerly run a potentially effectful RHS. In _no_exchange_observed,
place the guarded keyed load and _scalar call inside the proper source guard.
Only then may sound reachability remove them for absent exchange fractions.

Validation: RHS has an observable effect or invalid lookup; verify it is skipped
when short-circuited. Also compare all Boolean combinations passed to pi_update.

### A11. Implement exact diagnostic strings and optional report sequences

Lower JoinedStr/FormattedValue into native formatting with the authored format
specifiers, including `.3e` and `.6g`, backed by a real string representation.
If interning is used, equality must be based on exact formatted bytes with
collision handling; hash-combining raw operands is not equivalent. Support
the failure headers and report lines that A12 makes reachable again.

Resolve getattr(first,'unresolved_report',()) through proven attribute presence.
Prune the report path only with a valid closed-world absence proof; otherwise
retain its string sequence, `[1:]` slice, iteration, and output effects.

Validation: distinct numbers formatting identically, rounding boundaries,
signed zero, exponent spelling, NaN/inf, empty/multiple report lines, and exact
reason equality/deduplication behavior. Covers the four current formatted
formals plus newly reachable formatted paths and115/117.

### A12. Add optional presence to the ABI before final pruning

Give optional controller fields explicit presence storage alongside their
payload. Update ProgramABI metadata, caller/callee field expansion, native feed
packing, result decoding, and field loads together. Test `is None` against
presence, with payload accessed only on present paths. Keep the existing refusal
of None-to-numeric substitution until this full path works.

The real managed builder constructs STController(dt_min=None,dt_max=None).
Re-lower that input after presence exists. The numerical-exhaustion/failure
return path becomes live; regenerate findings and repair its required producers
under A4/A7/A10/A11. Never preserve an old fold derived from an invalid numeric
specialization. Apply the same mechanism wherever other optional field payloads
cross this product's ABI.

Validation: all four min/max presence combinations, changing presence between
invocations where supported, and actual managed input admission without defaults
invented by the adapter.

### A13. Repair run_superstep's loop-current state and record result

Recompute `(round_max_t-total).item()>eps` from the current loop-carried total
at the header/latch, not the entry subtraction. Retain `last_metrics=metrics`
as a loop-carried record identity with its complete keyed layout. Preserve the
zero-iteration initial record and the last real metrics after any iteration.
Retain the remaining-window condition and its keyed annotations.

Preserve the mid-round optional min/max clamps on dt_cap. Put last_dt_next
updates only on the source edge that executes them: the dt_used<=0 break keeps
the previous value while last_metrics already holds that attempt's metrics.
Preserve max_iters and all break/continue ports and incomplete-window reporting.

Validation: zero, one, and multiple substeps; exact window completion without
an extra zero-duration attempt; failed advance; iteration cap; pinned/steered
modes; increasing/decreasing proposals and optional clamps. Compare total,
next dt, final metrics, and mutated controller/state together.

### A14. Close the source/native validation loop

Make the call-only rejection-loop and scalar/child-record xfails into passing
native tests by repairing their missing source bodies/guards. Preserve the
already-fixed reason and accept-call ordering. Resolve the old append test's
C-shim assertion against the actual intended ABI contract while retaining its
native mutation checks; do not simply remove the failing assertion to report
a clean suite.

Run focused tests after their coupled changes, then one fresh whole-source
diagnostic with optional presence enabled. Require zero formal/undefined/call/
structural-output/effect-order/ABI shortfalls. Only then perform the full native
build and frame parity using the actual managed builder, covering acceptance,
rejection/retry, floor retention, exhaustion, and repeated frames. Compare all
returned fields and mutated state, not just a summary success flag. Repair any
new mismatch within this work package. Performance measurement follows parity.

## Execution order within one continuous implementation run

1. A1 and the failing source/native regressions establish honest gates.
2. A2/A3/A4/A5 together settle effects, object/storage identity, and return ABI.
3. A12 restores the real input domain; regenerate the live-path inventory.
4. A6–A11 supply collection, method, Boolean, and string producers for that domain.
5. A13 closes loop-current values, record carries, clamps, and exit semantics.
6. A14 proves the full result and closes newly surfaced failures.

These are sequential dependencies inside one task, not reasons to stop after
each item. A2–A5 share compiler files and identity contracts and should not be
edited independently without integration. There is no evidence supporting a
promise that all patches can safely be applied simultaneously or that the19
baseline findings are the final count after optional paths are restored.
