# Action plan for the next agent: DT-managed compile, remaining frontier

Written 2026-09-05 at the end of the session that landed control-aware result
merging (returns and breaks) and the three planned-region identity rules.
Read this before touching the compiler. Everything below was verified in
this tree; nothing is assumed.

## 0. Ground rules (the user has stated these; they are not optional)

- Fix the compiler, never rewrite the compiled sources to dodge a defect.
  Two earlier-session source edits exist and are still uncommitted and
  flagged: `src/common/dt_system/dt_controller.py` (`last_metrics =
  Metrics(...)`) and `dt_scaler.py` (`coerce_metrics` all-12-fields). They
  are NOT approved as the fix; the compiler defects they dodged are items
  2 and 3 below.
- No guessing, no heuristics matching things, no bespoke patches at an
  intermediary (linker, mid-emission). Read the code until the identity
  source is known, then fix it there.
- Never run the vehicle native build or `tools/frame_parity.py` unless the
  user has just said to. Verify with the seconds-long tools first, then the
  ~6 min managed compile.
- Never run whole test files or sweeps; one targeted test or one detached
  run at a time. Kill stale runs by WINDOWS pid (`Get-Process`/`taskkill`),
  not the Cygwin pid `$!` prints.
- Do not report a step as done without its verification output. Do not
  shelter the user from what is wrong; say it plainly.
- There is no time crunch. Do it right and thoroughly.
- Commit only when asked. The whole working tree is uncommitted (21
  modified files, ~4.6k insertions, plus new docs/tools); do not commit,
  stash, or checkout paths to baseline anything.

## 1. Current verified state

`lower_balloon_tire_managed_python_ssa()` (the whole DT system: 
`balloon_tire_managed_window` -> `run_superstep` -> `step_with_dt_control_used`
-> `_apply_energy_sidechain` -> `_no_exchange_observed`, 170 functions)
passes the full-native execution contract and the identity scan:

```bash
cd /c/dev/Powershell/turing && python tools/scan_managed_duplicates.py
```

Expected (fix30, ~350 s): `LOWERED OK`, `duplicate functions=0`, no
`loop-control-site` / `break-edge-value` shortfalls, exactly one
`INOUT-REDEFINED ... pi_update ... [(11, 2)]` (by design: a written mutable
scalar record field's formal id is redefined by every producer; the C backend
publishes the last definition). Anything else in that output is a regression.

Seconds-long checks that must all pass before any full compile:

```bash
python tools/repro_return_merge.py            # return inside loop -> OK
python tools/repro_return_merge_toplevel.py   # return in top-level if -> OK
python tools/audit_break_in_if_trace.py single
python tools/audit_break_in_if_trace.py nested
python tools/audit_break_in_if_trace.py param
python tools/audit_break_in_if_trace.py carried
python tools/audit_break_in_if_trace.py while_break   # run_superstep shape
```

Each audit case must end with a `loop_exit` Phi tagged `loop_result_port`
that merges the pre-loop value with one incoming per break edge, and a `Ret`
of that Phi (not of an arm-only value).

Design record: `docs/PLAN_CONTROL_AWARE_RESULT_MERGING.md` (sections
"Done since", "Break path: implemented"). Memory files:
`project-region-identity-rules`, `project-break-path-control-merging`,
`project-dt-managed-compile-frontier`.

## 2. Open items, in the order to take them

### 2.1 `opaque-state-effect` blocker (multi-field mutation of one object)

- Symptom: `tools/repro_run_superstep.py` fails with
  `CompilationSubdivisionRequired ... blockers=('opaque-state-effect',)`
  (loop node 614, all 39 regions). The managed compile itself passes, so this
  is the standalone-repro artifact of the documented "param-mutation half
  still open" (memory `project-store-chain-identity`).
- Where: `src/compiler/loop_composer.py` ~4499 appends the blocker when any
  `loop.state_effects` entry has `mode is LoopStateEffectMode.OPAQUE`. State
  effects are collected in the reducer's loop handling
  (`src/common/tensors/topological_reducer.py`, `state_effect_calls` /
  `state_effects` around the `loop_carried_bindings` block) and classified in
  the composer (`LoopStateEffect`, `LoopStateEffectMode`).
- Approach: find which `ctrl.*` mutations are classified OPAQUE and why the
  repro's stub ABI differs from the managed contract (the repro builds its
  policy from `balloon_tire_managed_extraction_contract` but only for a
  subset of records). The fix is either a missing record/field declaration
  path in the repro harness (then fix the harness, not the compiler) or a
  real classification gap (then fix the classifier so the effect is a
  sequence/record write, not opaque). Do not widen `ct_value`-style
  filters to make it pass.

### 2.2 `coerce_metrics` receiver identity

- Symptom (team-6 finding): a single-result call that returns its own record
  parameter (`coerce_metrics(metrics) -> metrics`) mints a fresh surface for
  the result instead of publishing the receiver identity, so callers see a
  copy and field updates on the returned record can be lost.
- Repro shape: `tools/repro_metrics_rebind.py` and
  `tools/audit_ancestry_retained_loop_graph.py` (imports `coerce_metrics`).
  Build a seconds-long repro first (same harness as
  `tools/repro_return_merge_toplevel.py`: real Metrics/Targets ABI via
  `balloon_tire_managed_extraction_contract(stub).program_abi.receipt()`).
- Where to look: result publication for calls in
  `src/compiler/fortran_c_shell.py` (`emit_outputs`, the collision freshener
  ~21290 "Prefer the unique real producer", `materialize_parameter_record_abi`
  ~12768 which coalesces record field storage) and the aggregate call
  identity rules (memory `project-aggregate-call-identity-rules`, tests in
  `tests/test_aggregate_call_identity.py`). The identity rule to establish:
  a call whose returned value IS a parameter record publishes that
  parameter's identity (member formals), never a fresh id.

### 2.3 zip audit sites #4 to #10

- Earlier agents audited every `zip(...)` in the compiler that silently
  truncates on arity mismatch. Sites #1 to #3 were fixed (arity guards in
  `precompile_to_ssa.py` ~21439 / ~23073 and `ssa_call_storage.py`). Sites
  #4 to #10 were left as "raise-style" candidates: each should become a
  loud arity check (`strict=True` or an explicit shortfall), not a silent
  truncation. Find them with:

```bash
grep -n 'zip(' src/compiler/fortran_c_shell.py src/compiler/precompile_to_ssa.py src/compiler/loop_composer.py | grep -v strict
```

  Judge each: if both operands are contracts that must agree (formals vs
  actuals, outputs vs slots), make it strict; if one side is legitimately a
  prefix, leave it and say so in a comment.

### 2.4 team-2 capture-coordinator walks (#2) and `has_path` sites (#4)

- Ancestry walks in `glsl_deployment_strategy.py` that ignore the
  retained-loop control-member discount already applied in
  `_topological_region_order` and `_atomic_region_node_order`
  (`recursion_table[...]["control_members"]`). Any raw `nx.ancestors` /
  `nx.has_path` over `graph.G` that can cross a retained loop's state-effect
  cycle is a candidate. List them:

```bash
grep -n 'nx.ancestors\|nx.has_path\|nx.descendants' src/compiler/glsl_deployment_strategy.py
```

  Apply the same discount (or a shared helper) instead of inventing a second
  rule; `tools/repro_step_with_dt_control_used.py` is the diagnosing repro.

### 2.5 Native build and frame parity (needs explicit approval)

- Only after 2.1 to 2.4, and only when the user says so: the vehicle native
  build, then `tools/frame_parity.py` (N-frame Python vs native, 1-ULP
  baseline; memory `project-frame-parity-harness`). Expect the build to take
  10+ minutes; run it detached, one at a time, and read its log instead of
  re-running.

## 3. How the fixed machinery works (so you do not re-derive it)

- Region identity: `expand_plan_regions` / `copy_region_instructions`
  (`hierarchical_plan.py`) expand each planned region once per function from
  a function-wide id watermark; all consumers in
  `lower_control_sections_to_ssa` share it. Literals in `constant_values`
  are never region outputs. The plan walk orders regions atomically
  (`_atomic_region_node_order`, `glsl_deployment_strategy.py`).
- Return merging: reducer `return_slot_values` (span-keyed) ->
  `LoopDescriptor.return_controls` / conditional arm return controls ->
  `LoopControlBlock(action="return")` -> `function_exit` Phis with
  `binding: return_merge`.
- Break/continue: reducer `loop_control_site_bindings` -> loop node
  `loop_break_sites` / `loop_break_bindings` (Break node consumes its site's
  initial/value ids so pruning cannot strip them) ->
  `LoopDescriptor.control_sites` / `break_bindings`, break-bound LoopResult
  ports (`result_ports` entry with updated == initial) -> arm-owned placement
  iff a carried site value is region-produced inside the arm (composer
  `arm_owned_site`, builder `arm_loop_control`, same rule), else lexical
  under `guarded_expression(chain)` -> SSA builder break edge carries site
  values; `control_site_ids` makes an unplaced site a shortfall.
- Dominance (`_value_dominates_current_edge`): two-write carried slots need
  "any producer dominates"; unreachable blocks are excluded from the
  dominator fixpoint.

## 4. Debug hooks (all env-gated, all kept)

| Variable | Prints |
|---|---|
| `TURING_DEBUG_BREAK_EDGE=1` | arm-ownership decisions on both sides; block dump at a failing break site |
| `TURING_DEBUG_REGION_OUTPUTS=1` | per-region produced/required/outputs and watermarks |
| `TURING_DEBUG_OUTPUT_LOADS=<fn>` | region-output loads whose result id drifted, at several pass boundaries |
| `TURING_DEBUG_ALIAS_BINDING=<fn>` | every `external_values[k] = v` with `v.id != k`, with stack |
| `TURING_DEBUG_GRAPH_NODES=<fn>:<id,id>` | graph node data for those ids at the shell's lowering call |
| `TURING_DEBUG_STRUCTURAL_OUTPUTS=1`, `TURING_DEBUG_CONTROL_OVERLAY=1`, `TURING_DEBUG_REGION_ORDER=1` | earlier hooks for structural outputs, conditional overlay, scheduled marker structure |

`tools/dump_region_schedule.py <fn-substring>` prints a function's plan
order, dispatch regions and field `after_write` edges during the managed
compile.

## 5. Practical hazards observed this session

- Bash heredocs containing Python triple quotes intermittently fail with
  "unexpected EOF while looking for matching `''`". Write patch scripts to
  the scratchpad with the Write tool and run them; do not retry heredocs.
- Bash `cd` does not persist reliably; prefix commands with
  `cd /c/dev/Powershell/turing`.
- A patch that asserts and fails partway may already have written earlier
  files; check what landed before re-running (count matches with leading
  newlines when a 4-space pattern is a substring of an 8-space line).
- The managed compile is ~6 min; never start one to answer a question a
  seconds-long repro can answer. Never leave two running.
- Specialization prunes dead bindings AFTER the reducer records ids in
  node attributes; any new attribute holding ids needs either a consuming
  edge or a presence filter at consumption (both patterns are in the code).
- Cached id copies (attributes, pending plans, descriptors) drift when
  edges are rewired: extend `_retarget_cached_value_ids` and
  `_retarget_plan_value_ids` for any new id-bearing field.

## 6. Reporting

End every work chunk with: what was found, what changed (files), the exact
verification command and its output, and what is still open. If a step is
blocked or skipped, say which and why.
