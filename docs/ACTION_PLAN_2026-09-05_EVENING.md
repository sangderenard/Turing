# Action plan, 2026-09-05 evening: preserve the gains, then unblock DT

> Progress and the live gate table: [STATUS.md](STATUS.md). Sections 0-3
> landed as f90b36ea (checkpoint), bb3cf3d9, e1f94cc3, e4c04407.

Supersedes `ACTION_PLAN_NEXT_AGENT.md` (morning) for sequencing. The ground
rules in that document still apply verbatim: fix the compiler not the
sources, no heuristics, seconds-long repros before six-minute compiles,
native builds and parity only in the sequence the user authorized, one heavy
run at a time, kill by Windows pid, report with verification output.

## 0. Preserve first

The working tree holds 56 modified and 34 new files from the 12.5-hour
session. All of it is intended work toward one goal: the complete
Python-authored validator, its DT window included, compiled and run
natively with the dispatcher owning threads and the viewer fed by the
authored geometry. Commit it AS A CHECKPOINT before touching anything, so
every later step is a small diff against a known point and can be bisected
or reverted individually. Do not stash, do not checkout paths, do not
"clean up" first.

```bash
cd /c/dev/Powershell/turing
git add -A
git commit -m "Checkpoint 2026-09-05: DT compiler, native emission, dispatcher/threads, geometry and viewer work in progress"
git push origin codex/recursive-reduction-bridge
```

Record in the commit message that `tools/repro_return_merge.py` and
`tools/scan_managed_duplicates.py` fail at this checkpoint (section 1), so
nobody later reads the commit as a green state.

## 1. Gates, and their status at the checkpoint

| Gate | Cost | Status now |
|---|---|---|
| `python tools/audit_break_in_if_trace.py {single,nested,param,carried,while_break}` | seconds | pass |
| `python tools/repro_return_merge_toplevel.py` | seconds | pass |
| `python tools/repro_return_merge.py` | seconds | **FAIL**: id 35 defined twice |
| `python -m pytest tests/test_call_order_across_conditional.py tests/test_ssa_definition_dominance.py tests/test_pruned_loop_return.py tests/test_control_region_dependencies.py -q` | 4 s | 10 pass |
| `python tools/scan_managed_duplicates.py` | ~6 min | **FAIL**: full-native contract, undefined operand 459 in `step_with_dt_control_used` |

Every change below is done against these; a step is not done until the
gates that were green stay green and the one it targets turns green.

## 2. Fix the return-merge regression first (cheap, isolates the scheduler edits)

`repro_return_merge.py` runs in 2 s and passed at `bce4f5da`. On the
checkpoint it reports id 35 produced both by a control-expression
`binary_scalar_double` call in `if_merge` (which also consumes 35) and by a
region output load in `return_control_next`. The candidates are the
session's precompile_to_ssa.py edits, each a separate hunk:

1. `emit_plan_callsite`: replacing a non-dominating argument by the unique
   dominating `conditional_carried` Phi.
2. `dependency_signature` for `LoopControlBlock` (return slot and guard
   inputs) and for `ConditionalBlock` (atomic arm with external inputs).
3. The while predicate getting a fresh id when a predicate expression exists.
4. Region functions gaining an explicit `Ret` publication terminator.
5. loop_composer: post-order return values, return placement waiting for
   slots, specialized dead-arm handling.

Disable one at a time (in memory, the way `build/check_scheduler_baseline.py`
does) until the repro passes, then fix that hunk at its identity source: a
control expression must never define an id a region publishes. Keep the
hunk's intent; do not just delete it.

## 3. The DT blocker: record identities used as physical operands

Diagnosed on the checkpoint with the new dump hook
(`TURING_DEBUG_DUMP_FUNCTION=step_with_dt_control_used:<file>`, written at
the full-native contract check in fortran_c_shell.py):

- 459 is the whole `Metrics` record returned by `balloon_tire_managed_advance`
  at plan callsite 326 (binding callee 214 -> caller 459). The advance call
  publishes member fields 1564..1575; the record id has no definition, by
  design.
- `ssa_sequence_39_append [39, parts, 325, 459, 1498]` in
  `sequence_mutation_selected.11` is `failures.append((float(dt_for_advance),
  metrics, tuple(reasons)))`, dt_controller.py line 449. A record inside a
  tuple must be appended as its row fields.
- `ssa_sequence_540_add [540, parts, 745, 459]` with
  `source_effect_node_id` 460 (the `coerce_metrics` callsite) is emitted in
  `entry`, before the while loop and before advance runs. Its exact source
  statement was not confirmed; it reads as the keyed write-back of the
  normalized `error_channels`.

Steps:

1. Seconds-long repro in the `repro_return_merge_toplevel.py` harness (real
   Metrics/Targets ABI): a callee returning a record; the caller appends
   `(float(x), record, tuple(names))` to a list inside a loop, and passes the
   record to a second callee that rewrites its keyed field. Assert no
   undefined operands and that the effect calls sit inside the loop.
2. Where the mutation's argument ids become operands:
   `_lower_sequence_mutation_body` in precompile_to_ssa.py (the
   `ControlSequenceMutation.argument_value_ids` path) and the append/row
   deferral (`ssa_deferred_record_row` in fortran_c_shell.py). The rule to
   establish: an argument that is a record identity resolves to that record's
   published member formals (or keyed part slots), from the record table
   entry that the producing call's result publication must register. The
   session's note "record_table.records 239 ABSENT" is this same gap for a
   sibling site; register the entry, do not invent row fields.
3. Placement: the mutation block for effect 460 is built at
   fortran_c_shell.py `SequenceMutationBlock(mutation)` (~line 5733) and
   installed by the lexical sequence installers the session extended. It
   must anchor at its callsite marker (`__plan_callsite_460__`, which the
   scheduler does emit inside `loop616`), never fall back to `entry`. Make
   the fallback loud.
4. Re-run the dump plus `scan_managed_duplicates.py`; expect `LOWERED OK`,
   170 or 169 functions, zero duplicates, zero control shortfalls, the one
   `INOUT-REDEFINED pi_update`.

## 4. Native DT and parity (in the order already authorized)

Only after sections 2 and 3 are green. One build at a time, detached, then
`python -u tools/managed_dt_parity.py <build dir> --frames 1 --timeout 60`.
The previous binary (`build/managed_dt_return_order_20260905`) was built from
the broken placement above; its 11 mismatches (`inf`, `1e-30`, zeros) are
consistent with an effect running on uninitialised storage before advance.
Do not use it as a baseline for anything. Parity green is the prerequisite
for any timing claim.

## 5. The whole validator native

After DT parity. The last whole-program lowering got through hierarchy
instantiation and stopped at `CompilationSubdivisionRequired` on loops 535
(wheel initialisation) and 587 (viewer loop): raw `with status_lock` and
`Condition.notify_all`. Direction from the user: these are dispatcher-owned
regions, the dispatcher owns threads, the worker/viewer handshake is kept.
What exists: `DispatchBlock` and `ResourceScopeBlock` lowering in
precompile_to_ssa.py, dispatch ops in `python_special_cases.py`,
turing_pool.c/h with barrier-joined frames, Nodus `thread_pool` as the async
submission model, the geometry wire format and GL consumer for the viewer
side of that handshake. What is missing: the Thread.start/join plus
Condition/Event contract itself. Write that contract as a short design note
before code, then a seconds-long regression per construct (`with lock`,
`notify_all`, `Thread(target=...).start()`), then the whole-program lowering
(`tools/lower_vehicle_validator_program.py`, ~10 min).

## 6. Keep the record usable

- Freeze `docs/CONTINUATION_2026-09-05_RECORD_RETURN_IDENTITY.md` as
  history (2,300 lines). Start `docs/STATUS.md`, at most ~100 lines: the
  gate table above with dates, the current blocker, the next step. Update it
  instead of appending to the continuation.
- Debug hooks are env-gated and stay: `TURING_DEBUG_DUMP_FUNCTION`,
  `TURING_DEBUG_BREAK_EDGE`, `TURING_DEBUG_REGION_OUTPUTS`,
  `TURING_DEBUG_OUTPUT_LOADS`, `TURING_DEBUG_ALIAS_BINDING`,
  `TURING_DEBUG_GRAPH_NODES`, `TURING_DEBUG_LINKED_CALLS`.
- `build/` is ignored; the saved scheduler snapshot
  (`build/step-call-schedule.pkl` plus `inspect_step_call_schedule.py`) is a
  cheap replay of the step scheduler. Keep it until section 2 is done.
- Commit after each green step with a one-line message naming the gate that
  turned green. Push at the end of each session.
