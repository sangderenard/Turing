# Effect-order repair following the 19-finding review

## Current result

Fresh whole-program lowering completed at **FORMAL-DIAGNOSTIC 19**, with
**zero undefined operands, unresolved calls, and unmaterialized boundaries**.
It exited 1 at the strict provenance gate. Log:
`build/effect_order_full_formals_20260906.log`. The authoritative files in
`build/full_formal_diagnostic/` now contain this post-repair linked SSA.

The linked step confirms that the hard-failure bool region reads **1378**,
the coerce_metrics field, rather than the later **593** Phi. All four surviving
reason append sites precede the query in `loop_exit.1` (checked with loop
backedges removed). `_propose_dt_pen` and `pi_update` are in
`loop_control_next`; `_apply_energy_sidechain` and `update_dt_max` follow in
`if_merge.18`. The later scalar field Phi is in `if_merge.8` and is still not
wired into the returned record. The formal count is intentionally unchanged.

No full native build or validator parity run was launched. All processes from
this repair are terminal. Nothing was committed or pushed.

## Verified causes and changes

The late `_schedule_loop_callsites` scheduler selected any ready statement
when an earlier statement was waiting for a producer. That could postpone a
conditional append until after a later truth read, or execute an accept call
before the earlier guarded `continue`. It now schedules prerequisites on
demand immediately before their consumer.

Resident sequence identity is no longer treated as a fresh value defined by
the next mutation. An initial truth read therefore cannot acquire a dependency
on a later `clear`. Accesses to the same sequence are ordered through their
containing control blocks. Source-placed calls also retain preceding terminal
control. A contradictory effect/dataflow cycle now raises a diagnostic with
control/region identities instead of silently emitting the remaining order.

The captured whole-step plan exposed a second cause: pure region 35 (a
rejection comparison) inherited an old hierarchy-rank dependency on region 34
(accept-side result conversion), which depends on `_apply_energy_sidechain`.
That rank forced rejection work after accept work. Proven-pure regions now
use their SSA dependencies without that obsolete rank. Purity uses the
existing `_PURE_REGION_OPS` inventory; memory traffic and unknown operations
retain their prior ordering protection.

The native sequence regression exposed a separate missing effect: lexical
mutation recovery admitted `append` and `add`, but omitted non-loop `clear()`.
Recovery now retains zero-argument clear effects on the known resident
sequence surface, using the sequence's policy. No authored DT source changed.

## Controller optional-presence correction

The managed input builder creates `STController(dt_min=None, dt_max=None)`.
Its feed adapter previously replaced None by field-name-dependent numerical
sentinels (`1e-30`, infinity, or zero). That is not a declared specialization
preserving `is None`, and the rejection-exhaustion distinction makes the loss
observable. The adapter now rejects None in a numeric record field whose ABI
has no optional presence storage. Actual numeric values remain unchanged.

This is a refusal of an unrepresentable input, not an implementation of
optional record-field presence. A genuine presence representation (including
mutation when needed) remains required before this controller can run natively.
The reachability pass itself continues to consume only what the SSA declares.

## Validation and evidence

- Three new scheduler tests reproduced the wrong orders before the repair.
  The expanded targeted scheduler/region gate passed **12 tests in 2.67s**,
  including the existing mutation-alias hierarchy-order regression and a new
  cycle-refusal test.
- The optional-field admission checks first failed at all four field names;
  after the guard, all six managed-wrapper tests passed in **3.47s**.
- The native sequence test covers all eight combinations of two independent
  appends and a conditional clear, comparing both observations to Python and
  reusing native execution storage. It passed in **12.87s**.
  Before the clear-recovery fix, native emission omitted clear and its guard;
  after the fix, both observations match the authored Python function.
- The existing sequence-truth and two conditional method/capture native
  checks passed during the intermediate scheduler run (**3 passed**; the new
  test in that run failed because its fixture omitted required ABI bindings,
  corrected before the final sequence run above).
- The actual full-step captured Control IR replays through the repaired
  section lowering with **zero shortfalls**. Its hard-failure bool region now
  consumes source field 458 before the floor merge instead of future field
  Phi 593. The reasons query follows the rejection appends. Accept call 523
  is in `loop_control_next`, and call 537 follows in the accept region.
  This is control/section evidence, not linked native validator parity.

Captures are in `build/effect_order_full_capture/`; `final-control.pkl` is the
pre-repair source-positioned control and lowering arguments. The replay is
`replayed-sections.pkl`, with `replay.log` reporting zero shortfalls. The old
full capture finished at the existing 19-formal gate, without a native build.
The helper scripts live in `speaktome/AGENTS/tools/`.

## Explicit remaining failures

Two broader native tests are retained as narrowly detected expected failures:
the scalar-record assignment loses its guard before scheduling, and a
call-only `for` loop loses its reject guard and body. These are **not passing
native proofs**. They stop at the missing authored parameter and remain visible
for the record/effect-retention work. Other exceptions still fail the tests.
Both were confirmed as **2 xfailed in 5.32s**, with assertions that conditional
branches really are absent before taking the narrow xfail path.

Return-merge wiring of the ledger versions, removal of duplicate effect-token
Phis, first-observation handling, mutable aliases, generator reduction, and
optional-field presence are not fixed here. In particular, do not drop the
duplicate formals solely because the early read now observes the right value.
The full validator's linked output/state parity remains unproved.
Ordering checks currently live in the scheduler; a general post-link
source-aware stale-read audit has not been added.
