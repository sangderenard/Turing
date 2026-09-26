# Status: DT-managed compile on `codex/recursive-reduction-bridge`

Short page, updated in place. History lives in the dated continuation
notes; sequencing in `ACTION_PLAN_2026-09-05_EVENING.md`.

## Gates (2026-09-06, early morning)

| Gate | Cost | Status |
|---|---|---|
| `python tools/audit_break_in_if_trace.py {single,nested,param,carried,while_break}` | seconds | pass (exit 0 x5) |
| `python tools/repro_return_merge_toplevel.py` | seconds | pass |
| `python tools/repro_return_merge.py` | 2 s | pass (bb3cf3d9) |
| `python tools/repro_record_row_effects.py` | 2 s | pass (e1f94cc3 .. dbf77f32) |
| four control-region pytest files + two threading tests + five record-row tests | 15 s | 17 pass |
| `python tools/scan_managed_duplicates.py` | 5.5 min | LOWERED OK, 178 functions, 0 duplicates, INOUT-REDEFINED pi_update only (dbf77f32) |
| `python tools/repro_record_field_mutation_native.py` (native, real material ABI) | 15 s | pass (8dc639b3) |
| native DT build `build/managed_dt_record_rows10_20260906` (O0, batch 8, window = dt = 2^-20) | 10 min | builds and runs to completion (8dc639b3) |
| `python tools/managed_dt_parity.py build/managed_dt_record_rows10_20260906 --frames 1 --timeout 60` | 1 min | **FAIL**: 10 mismatches, root cause below |
| `tests/test_ir_sequence_tables.py::test_compiled_retained_loop_mutates_caller_sequence_record` | 5 s | FAIL, pre-existing at f90b36ea (expects an empty C shell source) |

## Identity rules landed 2026-09-05/06

- A scheduled region is the single physical producer of the values it
  owns (plan builder never folds a loop-carried seed into a capture;
  `_consume_resident_control_values`).
- A record at one column of a tuple row stands for its member fields in
  the ABI's physical column order; the annotation (parameters included)
  declares the row, the append defers the record column
  (`ssa_deferred_record_slots`), native-call linking expands it.
- Everything authored inside a loop body is lowered at its authored
  position: loop-owned guarded effects, region-less conditionals
  (`_nest_lexical_conditionals_in_loops`), planned callsites
  (`_place_plan_callsites_lexically`, scheduler respects placed markers),
  local truth queries; positions of pruned conditionals come from the
  source control record; ordered insertion flattens nested sequences.
- Scheduler dependencies see a call's result projections and a
  predicate's operands; `and`-with-False / `or`-with-True branch tests
  fold by short-circuit; a parameter the caller left at `None` is known.
- The linker refuses, loudly, to relocate a call off its scheduled marker.
- C emitter: a callee's output slot is typed under the callee; a `None`
  record default carries the field's physical dtype
  (`TURING_DEBUG_PHYSICAL` prints type-union conflicts).

## Current blocker: native parity (build/managed_dt_record_rows10_20260906)

The binary runs to completion; `advanced` matches eager; everything else
the retry loop touches is wrong in one recognisable shape (telemetry 0,
inputs rows 2-7 uninitialised, controller untouched, dt_next 0).  Traced
through the C: the material record's buffers pass by pointer through
window -> run_superstep -> step -> advance, and advance's regions store
through them.  What undoes those writes is the rollback path, and two
silent identity defects sit under it:

1. `saved = state.copy_shallow()` (a source-linked record method returning
   a tuple of array copies) was never lowered.  `saved` became a FABRICATED
   float64 scalar input of `step_with_dt_control_used` (formal 46), passed
   up as an invented input of `run_superstep` (589) and of the window, so
   `state.restore(saved)` (lowered as step's planned_region_2, guarded by
   `rejected`) copies inputs/state/output/telemetry from garbage.  The
   full-native gate reported `unresolved_calls=0`: the call vanished before
   the gate, exactly the class the plan forbids.
2. `targets.mass_max`, `targets.error_limits`, `ctrl.dt_min` projections in
   step are recorded as `unresolved_record_projection_receivers`
   (function metadata), i.e. the rejection test compares metrics against
   receivers the linker could not bind.  Whether native rejects the first
   attempt because of this or for another reason needs a native trace
   (`--trace` is not implemented for the managed C host).

Both are record/method resolution defects (the "opaque material methods"
frontier), not placement, not numerics.  The record plumbing itself is
proven by tools/repro_record_field_mutation_native.py (writes two calls
deep land, returned scalar right).  Two further backend gaps met on the
way and left alone: region `less`/`greater` over loop-carried scalars have
no module-lane C spelling, which blocks any while-loop variant of the
native repro.

## Handoff

See `HANDOFF_2026-09-06.md` (narrative, evidence, tools, open defects).

## Next step

1. Make the gate loud: a call result or record projection that becomes a
   fabricated ABI input of a linked function is a hard finding, never a
   silent input (formal 46 / 589 above would have failed the scan).
2. Lower `copy_shallow`/`restore` as what they are: a snapshot is the
   record's mutable arrays copied into caller-owned storage, restore is
   the copy back; seconds-long repro first (`tools/repro_record_field_
   mutation_native.py` shape plus a snapshot/restore pair).
3. Resolve the `targets`/`ctrl` projection receivers in step, then rebuild
   and rerun parity.  Then section 5 of the evening plan
   (`DESIGN_DISPATCHER_THREADS_2026-09-05.md`, regressions before code).

## Current resumption (after d50015b4, uncommitted)

The full-native gate now checks formal provenance using check_formal_parity.
Re-auditing saved trace3 SSA rejects unaccounted formals in five functions,
including step's fabricated saved46. Three targeted provenance cases pass;
return-merge repro stays green. No fresh full lowering or native/parity run.
The previous trace3 build finished and its executable exists. See the handoff's
resumption section for snapshot semantic mismatch and projection-metadata caveat.

Subsequent targeted work retains call-only branches and conditional-expression
result merges; native branch/result regressions pass. Record span `copy()` now
preserves shape (native 2x3 check passes). The authored snapshot diagnostic no
longer fabricates saved, but remains incorrect: restore's IndexedStore is present
before SSA and absent from both its region and control program; snapshot effect
ordering and the production normalizer's field selection remain open. The record
row effects repro also fails the new provenance gate for step value55, independently
of the IfExp addition. Full DT parity has not been rerun.

Latest: the lost slice store is fixed at dispatch classification and normalized
index metadata. Native 2x3 restore passes with DT's repository tensor provider.
The combined snapshot now executes but remains red: rollback produces [7,garbage]
instead of [2,3]. Its conditional payload is still scalar and capture follows
mutation. Earlier diagnostic harnesses omitted the tensor provider; the current
snapshot driver includes it. See the handoff's latest section for exact evidence.

2026-09-06 further producer audit: structural recovery no longer turns graph
loop inputs into function parameters. Exact aggregate-parameter member receipts
are accepted by the native provenance gate; local names/view labels are not,
and explicitly zero-argument source functions are audited. Seven gate cases
pass (1.87 s). Singleton tuple/call round-trip compiled native execution passes
(2x3, six distinct values); nested call-output shape/formal regression passes
(4.13 s), return merge repro passes (1.75 s). Authored conditional rollback still
has capture ordering and optional aggregate representation defects. Full DT
parity and performance remain unverified. Details in the latest handoff section.

2026-09-06 call-only conditional ordering: planner now publishes the existing
source anchor, preventing empty-region branches from being appended at the end.
Native capture-before-mutation regression passes (10.60 s); authored snapshot
SSA now orders capture, mutation, restore, final read. Optional snapshot tuple
representation remains open; no full DT parity or performance claim.

2026-09-06 method-container follow-up: descriptor propagation now includes
resolved methods and derives results from their declared record ABI. Native
method tuple capture/recover regression passes (10.18 s, six 2x3 values, no extra
inputs). Optional snapshot now fails the provenance gate at capture aggregate
id 3: its IfExp still expects a scalar handle. This is the next representation
defect, not a reason to admit another input. Full DT parity remains unverified.

2026-09-06 array-merge prerequisite: fixed matching-shape conditional result
metadata and C Phi lowering, which previously selected only an array's first
scalar. Native 2x3 selection and selection feeding compiled multiplication pass;
scalar conditional-call regression remains green. Optional tuple presence/member
representation is still open, and DT parity/performance remain unverified.

2026-09-06 optional representation audit: full-native now rejects bare mixed
None/payload Phis with an explicit presence/payload diagnostic. Four focused
gate cases and a real Python-source rejection test pass. This is a safeguard,
not optional support. DT restore is guarded by rejected, while snapshot creation
uses rollback; optional presence cannot be inferred from matching branch guards.

2026-09-06 authored snapshot semantics: removed the snapshot-specific AST rewrite
that captured/restored all mutable ABI fields. Ordinary compiled methods now
perform the exact authored operations. Direct capture/mutate/restore matches
eager execution of the same source for state, telemetry and return (10.51 s).
The conditional snapshot still fails at optional merge Phi 5/container input 3;
full DT parity and performance remain unverified.

2026-09-06 tuple merge progress: conditional tuples now merge matching members
individually under compiled control. Mixed array/scalar tuples from calls and
literal arms match eager execution natively (10.46/10.96 s), with no invented
inputs. Authored scalar shape survives padded helper views; plain array selection
still passes (10.43 s). None/presence handling is still the next frontier.

2026-09-06 guarded optional snapshot now native/eager verified in either arm
(10.63/10.98 s), including both flag outcomes and telemetry preservation. Exact
presence/payload receipts prevent invented inputs; three unsafe consumer cases
are rejected (4.08 s). General optional ABI/None tests and DT's early-return
presence proof remain open. Full DT parity/performance are still unverified.

2026-09-06 early-return optional snapshot now matches eager execution for both
runtime branches, including state, telemetry and return (10.79 s). Source
fallthrough facts establish presence; source-placed calls no longer get reordered
by flat hierarchy rank. Inactive span nulls retain pointer dtype through scalar
inference and C emission. Unsafe guard cases still reject (3.95 s). This is a
bounded regression, not a whole-DT parity or performance result.

2026-09-06 evening full audit: the strict provenance gate started at 29 illegal
internal formals and is now at 25. General late recovery supplies native tensor
`all` and loop-current scalar `item` producers. Terminal `continue` bindings now
create exact loop-carried state, restoring the step retry loop's `dt_tensor` and
`retries` Phis. Saved SSA proves scalar reads consume the current header Phis,
not entry values. A competing BoolOp overlay (32 findings) and an incorrect
entry-hoist (25 but stale loop values) were both removed. Five functions remain
red. The final focused gate is 13 passed; the last full audit preceded only a
metadata-key rename. No native build/parity/performance run, commit, or push.
See the handoff's "full formal frontier 29 -> 25" section for IDs, evidence, and
next order.

Latest continuation: the authoritative full driver saved 24 unaccounted
formals after a general loop-aware scalar expression closure reconstructed
step's current-dt `item() * 0.5`. A tested nested-conditional incoming-edge
repair removes run_superstep's fabricated loop initializer from that saved
module, giving an effective 23; a fresh full driver run has not yet certified
that count. The current frontier is general record-field SSA state versioning
for Metrics assignments. See `CONTINUATION_2026-09-06_FULL_NATIVE_FRONTIER.md`.

Correction after the next full driver: 23 is now authoritative, with zero
undefined operands. The nested conditional is fixed upstream: reducer-authored
Phis carry their exact incoming value and ordinary control lowering consumes
those exact Phi parents. The post-link repair and historical fixture were
removed. A narrower record-field branch-state implementation is experimental;
it has reducer coverage and lowers the bounded step repro, but is not accepted
until native/eager whole-record return proves that its selected field version
reaches physical record storage. The concise continuation document supersedes
the stale “effective 23” wording above.

The continuation now ends with a read-only second-opinion brief covering the
formal audit, possible disconnected existing definitions, record-field SSA
semantics, and opportunities to collapse the 23 findings into general compiler
mechanisms. It points directly at the four saved evidence artifacts and authored
source functions.
