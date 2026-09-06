# Status: DT-managed compile on `codex/recursive-reduction-bridge`

Short page, updated in place. History lives in the dated continuation
notes; sequencing in `ACTION_PLAN_2026-09-05_EVENING.md`.

## Gates (2026-09-05, evening)

| Gate | Cost | Status |
|---|---|---|
| `python tools/audit_break_in_if_trace.py {single,nested,param,carried,while_break}` | seconds | pass (exit 0 x5) |
| `python tools/repro_return_merge_toplevel.py` | seconds | pass |
| `python tools/repro_return_merge.py` | 2 s | pass (bb3cf3d9) |
| `python tools/repro_record_row_effects.py` | 2 s | pass (e1f94cc3, e4c04407) |
| four control-region pytest files + two threading tests + five record-row tests | 15 s | 17 pass |
| `python tools/scan_managed_duplicates.py` | 5.5 min | LOWERED OK, 180 functions, 0 duplicates, INOUT-REDEFINED pi_update only (e4c04407) |
| `tests/test_ir_sequence_tables.py::test_compiled_retained_loop_mutates_caller_sequence_record` | 5 s | FAIL, pre-existing at f90b36ea (expects an empty C shell source) |
| native DT build `build/managed_dt_record_rows_20260905` + `tools/managed_dt_parity.py --frames 1` | ~10 min | in progress, see the continuation |

## Identity rules landed this evening

- A scheduled region is the single physical producer of the values it
  owns: the plan builder never folds a loop-carried seed into a region
  capture, and control expressions consume region-owned values instead of
  re-deriving them (`_consume_resident_control_values`).
- A record named at one column of a tuple row stands for its member
  fields: the authored annotation declares the physical row, the append
  defers the record column (`ssa_deferred_record_slots`), native-call
  linking expands it from the caller's record entry.
- Effects, calls and conditionals authored inside a loop body are lowered
  at their authored position: loop-owned guarded effects installed
  lexically, conditionals nested by authored span
  (`_nest_lexical_conditionals_in_loops`), arm-owned callsites stamped,
  trailing markers ahead of terminal edges, and a marker the linker cannot
  honour is a loud error naming the undominated consumers.
- A parameter the caller left at `None` is a known fact even when the name
  is mutated; only collection initializers are withheld from the fold.

## Current blocker

None in the lowering. Parity of the fresh native DT binary is the open
question; the previous binary (`build/managed_dt_return_order_20260905`)
is not a baseline.

## Next step

Section 4 of the evening plan: parity on the fresh build, then section 5
(the Thread/Condition contract design note before any code).
