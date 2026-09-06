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
