# Continuation 2026-09-17 — identity concordance

One day, one fault family, seven instances, one instrument.  Every compiler
fault traced today had the same shape: the process graph already held the
exact identity of a value, and a later pass rebuilt that identity from a
weaker proxy (a name, a signature position, a source coordinate, a dtype,
"it was explicitly passed") and disagreed with the graph.  The fixes are at
the passes; the instrument is a master correlation table that checks the
compiler's own records against each other in seconds.

Everything below is verified by a seconds-long repro unless marked
otherwise.  Nothing here was verified by the seven-minute product lowering;
that run is the user's to make (see "Product run").

## 1. Faults traced and fixed (compiler)

Each entry: symptom → observed chain → fix (file).

### 1.1 `aggregate call binding for 'metrics' ... 3 != 0`
`_propose_dt_pen(metrics, ...)` was handed the return tuple.
Chain: the reducer names output slot 0 `metrics` because every `return`
spells it so → structural specialization of the constant `if not rollback:`
arm wrote the whole return Tuple under **every** output name, clobbering the
variable `metrics`'s identity history → the planner's name-history repair
consulted the table, found only the tuple, and overrode the correct arg:0
edge.
Fix: the specialization writes only positional slots, from
`return_slot_values[span of the selected return]`; name-spelled slots keep
the variable history (glsl_deployment_strategy.py, `selected_return_id`
block).  Repro: `tools/repro_step_with_dt_control_used.py` assembly with
`rollback=False` (~41 s).

### 1.2 `control effect order conflicts with value dependencies` (cycle)
A `return` inside `while True` was scheduled before the calls inside its own
expression.  Chain: `_place_plan_callsites_lexically` (and three sibling
passes) position a return `LoopControlBlock` at the **start** column of its
returned Tuple; the `__plan_callsite_N__` markers for calls inside that tuple
sort after it; the "call after a terminal waits for the terminal" rule then
closes the cycle.
Fix: shared `_loop_control_block_position` — a return sits at the **end** of
its expression / slot values (fortran_c_shell.py, four callers).

### 1.3 undefined operand: `_apply_energy_sidechain(dt_next, ...)` read the arm-local `minimum`
Chain: raw edge was the merge Phi (correct); the planner's name-history
repair picks the latest definition by source position; a Phi has no source
position so it can never win and the arm-local `minimum` was substituted.
Fix: the repair never overrides a `Phi`, exactly as it already never
overrode `LoopResult`/`LoopExit` (glsl_deployment_strategy.py ~2170).

### 1.4 `restore`: 416 storage formals + 104 nameless (`self.field[...] = x`)
Chain: ingestion rewrites `obj[...]` to
`obj[tuple([slice(None)] * (obj.ndim - k))]`
(node_special_cases.expand_ellipsis_subscripts); the fold must collapse
`obj.ndim`; its shape walk reached the record receiver `self`, found no
tensor descriptor, and returned unresolved although the field's own
descriptor was declared → the expansion stayed live → a resident-sequence
"replace" helper with hidden length formals plus a one-element store.
Fix: `_declared_record_span_descriptor` — an ABI-declared record span is a
shape boundary in the walk (glsl_deployment_strategy.py,
`_fold_callsite_structural_values`).  Probe: the view-on-memory PieceState
model (`tools/audit_identity_concordance.py view`, and the scratch
`probe_view_state2.py`) now restores memory exactly.  **Note:** the tire's
`restore` compiled through the same defect before today (rollback path).

### 1.5 `conflicting SSA sequence descriptor 24` (`tools/repro_return_merge_toplevel.py`)
Chain: the keyed-lookup rewrite rebinds helper calls to the owner's parts
(`error_channels.keys/values/length`) but left the function's sequence
descriptor over the abandoned anonymous cells; caller-side propagation
mapped those onto fresh frame slots and registered a second descriptor for
a sequence the caller already owned.
Fix: the rewrite also remaps the descriptors onto the parts
(`keyed_storage_remap`), strips the abandoned cells' bookkeeping
(`_strip_dead_cell_bookkeeping`), drops the freed formals with their call
operands (`_drop_formals_and_call_operands`, shared), claims the keyed
status cell (`program_abi_keyed_part="status"`); propagation keeps the
incumbent when only scratch cells differ (fortran_c_shell.py).

### 1.6 dead local sequence from `channels = metrics.error_channels or {}`
Chain: specialization selects the field and drops the `{}`; the descriptor
minted for the literal survives with leased length/status formals and an
entry clear-store.  Fix: `_prune_dead_local_sequences` (fortran_c_shell.py).
Found by the audit, not by a failure.

### 1.7 `_propose_dt_pen` specialized into an empty function (`tools/repro_targets_expansion.py`)
Chain: the tool declares `distribution` as a scalar with
`python_type: builtins.NoneType`; only shape/dtype crossed the call, the
callee re-derived `builtins.int` from the dtype, and the fold's "an ABI fact
is never None" rule selected `return distribution(...)` (an opaque
callable) — the whole body vanished, leaving a bare `__plan_callsite_5__`.
Fix: `_tensor_descriptor` carries `python_type`; the callee's fact prefers
it; the fold's None-identity rule honors a declared `NoneType`; the
hierarchy predicate does the same (glsl_deployment_strategy.py).

## 2. The instrument: master correlation table

`src/compiler/identity_concordance.py` — one row per `(function, value_id)`
with every claim the compiler's records make about it (parameter names,
storage/closure/member formals, ABI field/parameter accounting, keyed parts,
leased frame storage, projected rows, sequence descriptor roles, record
fields, aliases, definitions, uses).  `findings()` reports disagreements:
`multiple-definition`, `unaccounted-formal`, `conflicting-storage-claims`,
`descriptor-member-unknown`, `descriptor-member-shared`,
`helper-operand-outside-descriptor`, `duplicate-storage-across-call`,
`use-not-dominated`, `alias-target-missing`.  Authored functions only;
regions/helpers are skipped; the region out-pointer convention and Phi
operands are exempt from dominance; identity is per record **instance**.

`tools/audit_identity_concordance.py [view|toplevel|mapping] | --pickle P`
runs it in 0.2–12 s.  As of this report all three cases report **0
findings**.  On the untouched HEAD checkout the `view` case reports the
1.4 defect in ten seconds.

## 3. Tools and evaluator

- `tools/repro_loop_dominance.py` accepts the projected-row `(keys, values)`
  ABI of a bare mapping parameter.
- `tools/repro_targets_expansion.py` feeds keyed handles, keyed status cells,
  frame storage, unset optionals; prints the concordance report and the
  formals it cannot feed.
- `src/compiler/ssa_reference_evaluator.py`: `extent` (row count) and
  `Deploy`/`Join` receipts.  `isfinite` is **not** added: it has no scalar
  spelling in the planner's table and no LLVM likeness entry; the
  evaluator's vocabulary audit correctly refuses invented opcodes.

## 4. Green / red

Green (seconds): `tools/audit_identity_concordance.py` (3/3 clean),
`repro_return_merge_toplevel`, `repro_return_merge`, `repro_metrics_rebind`,
`repro_no_exchange_observed`, `repro_keyed_get/construct`,
`repro_shared_record_two_callees`, `repro_record_field_mutation_native`,
`repro_loop_dominance`, `audit_break_in_if_trace` (5 cases),
`audit_result_binding_nested_return`, `audit_stale_ids_retained_ports`,
`audit_deferred_row_append_in_loop`; tests: native conditional call/tuple
result, record span restore, authored snapshot, sequence replace, container
store lowering.

Red, pre-existing at HEAD (verified on the clean worktree
`C:\Users\alber\AppData\Local\Temp\wtb`):
- `tools/repro_record_row_effects.py`: `Metrics.unresolved_report` is a
  `storage: table` field; `_record_row_physical_columns` expands keyed fields
  into columns but not tables, so a receiver record has no physical columns
  for it and a returned record that publishes it cannot be bound.  Needs a
  table ABI (length + column spans).  An honest refusal; a fabricated
  allocation was tried and reverted (it would zero host data at entry).
- `tools/repro_targets_expansion.py`: after 1.7, root reads
  `coerced.hard_failure` off the `coerce_metrics` result; the callee's
  returned record publishes only the fields the callee references, so the
  read becomes a nameless bool formal (the audit reports it).  Needs the
  result-direction counterpart of `_propagate_record_field_demand`.  Then the
  evaluator needs `isfinite` (see §3).
- `tools/repro_run_superstep.py`: `CompilationSubdivisionRequired` after
  ~60 s, untraced.
- `tests/test_declared_parameter_shape.py::test_an_extent_must_be_a_positive_integer[bad0]`.

## 5. Product run

The route is `examples/llvm_dt_system.lowered_system` over two stored pieces
(air + pool, batch 1).  Before today's fixes it cleared planning and all 47
shells and was refused by the final full-native contract on exactly 1.3
(undefined operand in `loop_control_next`) and 1.4 (`restore` formals).
Both are fixed on small repros; the product lowering has **not** been re-run
since.  The user runs it:

```
cd C:/dev/Powershell/turing && python -u "<scratch>/lower_two_pieces.py" c
```

(`lower_two_pieces.py` = `lowered_system([voxel_air_step/b1, pool_step/b1])`
with the compiler's progress reporter wired to stdout; recreate from
`examples/llvm_dt_system.py` if the scratch copy is gone.)

## 6. Working method (what the day settled)

- Trace the whole chain before speaking: repro, value at ingestion, every
  rewrite with its stack, the consumer that trusted the wrong record, the
  raise.  Observe with scratch hooks (`monkeypatch` module helpers, trap
  dict writes, read the raising frame's locals).  Never confirm by patching
  src and rerunning.
- Fix identities, not intermediaries.  One identity, one key.
- Seconds-long repros fit to the problem; the user launches long
  lowerings; output inline, unbuffered (`python -u`), never to a file.
- Baseline on the clean worktree at the short path, never via stash.
- Run the concordance audit before and after every compiler change.

See `AGENTS.md` § "The Kalto Engineer".

## 7. Handoff prompt

```
You are continuing the turing compiler work of 2026-09-17.  Read, in order:
turing/AGENTS.md ("The Kalto Engineer" at the end),
turing/docs/CONTINUATION_2026-09-17_IDENTITY_CONCORDANCE.md, and the memory
notes project-identity-concordance-audit and
project-dt-managed-compile-frontier.

State: seven identity faults fixed at their passes (uncommitted on
nogodsnomasters/turing until the checkpoint commit of this report); the
master correlation table (src/compiler/identity_concordance.py) reports zero
findings on its three cases; the product lowering (air + pool through
examples/llvm_dt_system.lowered_system) has NOT been re-run since the fixes
and is the user's to launch.

First action: run `python tools/audit_identity_concordance.py` (seconds) and
confirm 0 findings on all three cases before touching anything.

Next targets, in order:
1. Result-direction record field demand: a caller reading `result.field`
   off a returned record must materialize that field on the callee's
   returned record (mirror of `_propagate_record_field_demand`).  Repro:
   `tools/repro_targets_expansion.py` (root's nameless bool formal
   `coerced.hard_failure`; the audit lists it as unaccounted-formal).
2. Table-storage record fields (`Metrics.unresolved_report`, `storage:
   table`) need physical ABI columns on receiver records.  Repro:
   `tools/repro_record_row_effects.py`.  Do not fabricate storage.
3. `tools/repro_run_superstep.py` (60 s): trace `CompilationSubdivisionRequired`.
4. Make passes READ the correlation table instead of private maps: the
   planner's name-history repair, lexical callsite placement, frame-tail
   completion.  Each one removed is one fewer key for one identity.

Rules that are not optional: user runs anything longer than seconds; show
output inline; trace the full chain before reporting; fix the identity, not
the symptom; baseline on C:\Users\alber\AppData\Local\Temp\wtb; run the
concordance audit before and after; when told something exists, it exists.
```
