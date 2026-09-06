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
| native DT build `build/managed_dt_record_rows9_20260906` (O0, batch 8, window = dt = 2^-20) | 10 min | builds and runs to completion (1445946f) |
| `python tools/managed_dt_parity.py build/managed_dt_record_rows9_20260906 --frames 1 --timeout 60` | 1 min | **FAIL**: 10 mismatches, see below |
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

## Current blocker: native parity

The fresh binary runs to completion (no crash, no unresolved calls, no
undefined operands) but every value the callee chain writes is invisible
at the root buffers the parity tool reads: `material.telemetry` stays 0,
`material.inputs[:, 0] = dt` never lands, the controller's `dt_max/acc/
max_vel_ever` keep their initial values, `last_maximum_*` are NaN and the
returned `dt_next` is 0. Verified on this binary's repository SSA:

- the retry loop runs (`while_condition` Phi of two `True` constants);
- `advance` is called first in `while_body` with the caller's own record
  formals (telemetry 222, state 221, inputs 219, output 220);
- inside `advance` the telemetry/inputs stores (source lines 382-392) are
  regions 0/1/6/7 at the top of `entry`, unconditional, as authored;
- `copy_shallow`/`restore` are correctly absent (the managed window runs
  the `rollback=False` lane).

So the remaining defect is below the DT controller: the material record's
mutable fields (and the window's returned scalar) are not the root
storage in the native chain `window -> run_superstep ->
step_with_dt_control_used -> advance`. This is the still-open
"material_state param-mutation" half of the store-chain identity
(`project-store-chain-identity`), the same class as the previous binary's
11 mismatches. It is not a placement or record-identity defect.

## Next step

Prove which frame breaks the aliasing: a seconds-long repro with a record
parameter whose field is written two calls deep and read at the root
(`window -> a -> b` with `b` doing `rec.field[k] = v`), asserting the root
buffer changes natively. Then section 5 of the evening plan
(`DESIGN_DISPATCHER_THREADS_2026-09-05.md`, regressions before code).
