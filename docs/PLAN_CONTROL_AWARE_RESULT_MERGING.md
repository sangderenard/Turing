# Plan: control-aware result merging (multiple `return` statements)

Status: design, 2026-09-04. Blocks correctness of the managed DT-system compile
(`lower_balloon_tire_managed_python_ssa`), which now passes the full-native
execution contract but publishes the *last* return's values on every path.

## The defect, as measured

`step_with_dt_control_used` (dt_controller.py) has three
`return metrics, dt_next, dt_used` statements inside `while True:`. After the
body-wide `function_outputs` scan (topological_reducer) the callee publishes
`('metrics', 'result_1', 'result_2')` and callers bind all 15 outputs. But the
lowered SSA has ONE `Ret` in `while_exit` reading ids 594/331, and those ids
are **defined twice** (a degenerate `conditional_carried` Phi `[329,329]` in
`if_merge.10`, and a `Load` in `if_merge.15` from the final return's regions);
no `break`/`loop-return` edge exists. `_propose_dt_pen` (`if distribution is
not None: return distribution(...)`) and `pi_update` show the same duplicate-id
signature. The hierarchy plan says it outright
(`_direct_return_value_ids`): "Multiple lexical returns need control-aware
result merging; never guess".

Root: every source `return` records its values under the SAME output-slot
identity (`identity_bindings[slot].append(value)`), and every consumer takes
`identity[-1]`; the early return's arm is a terminal arm whose environment the
`if` merge drops ("guard clause" rule), and nothing branches to the function
exit.

## Design: a `return` is "bind the slots, then branch to the function exit"

One mechanism for both in-loop and in-`if` returns; no per-return special
casing.

1. **Reducer** (`src/common/tensors/topological_reducer.py`, Return handler
   ~3175): keep recording per-slot identities, and ALSO record, per return
   statement, its slot value ids in a graph-level table keyed by source span
   (`graph.G.graph["return_slot_values"][(line, col, end_line, end_col)] =
   (ids...)`). Add that table to the canonical relabel's remap list
   (`_normalize_lexical_values`, ~4185-4271). Spans are the correlation the
   hierarchy plan already uses for returns (`_captured_return_value_ids`).

2. **Planner** (`src/compiler/loop_composer.py`):
   - `_loop_plans` / `collect_loop_controls` (~2150-2200): classify EVERY
     in-loop `return` with a value as a `loop-return` control — drop the
     sole-root requirement (`tuple(graph.roots) == (return_value_id,)` can
     never hold for a tuple return; `graph.roots` is top-level-only anyway).
   - Conjoin nested guards instead of replacing them (`next_true = guard if
     predicate_id is None else (predicate_id, True)` loses the outer guard for
     any control two `if`s deep — a latent defect for `break` too). Represent
     the guard as a chain and build the `ControlExpression` as nested
     `"and"`/`"not"` (both already in `lower_control_expression`'s table).
   - `analyze_shader_loop_reductions` (~4296-4400): place loop-returns in
     `body_items` at their lexical position like `break`, NOT in
     `terminal_controls` (which lower after the whole body and therefore let
     the non-returning path's work run on the returning path).
   - `LoopControlBlock` (`control_source.py`): add
     `return_value_ids: tuple[int, ...] = ()`; `source_action="loop-return"`.

3. **SSA builder** (`src/compiler/precompile_to_ssa.py`):
   - `LoopControlBlock` lowering (~3550-3600): when `return_value_ids` is
     set, capture `self.external_value(id)` for each slot at the branch point
     and record `(block, values)` in a builder-level `function_return_edges`;
     branch to a dedicated `function_exit` block (created lazily), not the
     loop exit. Carried loop Phis at the loop exit simply do not receive this
     edge — correct, the post-loop code does not run on a return path.
   - `finish()` (~6048-6233): if `function_return_edges` is non-empty, make
     the fallthrough epilogue one more edge (if the current block is
     reachable), emit one `Phi` per output slot in `function_exit` with
     `incoming_blocks`, and `Ret` the Phi results; `named_returns` name the
     Phis. With no return edges, behaviour is unchanged.
   - Same for a `return` inside a top-level `if` (`_propose_dt_pen`): a
     `ReturnBlock`-shaped control at the arm's lexical position branching to
     `function_exit`. Prefer reusing `LoopControlBlock` with
     `action="return"` (extend the action set) over a new class, so
     `lower()` dispatch and the guard handling stay shared.

4. **Reducer merge rule** (~3315-3384): the terminal-arm "guard clause" rule
   stays for ordinary names. Output slots are excluded from it only in the
   sense that their merge point is the function exit, which step 3 provides;
   nothing else changes.

## Verification (fast, no build)

- `tools/audit_result_binding_nested_return.py` (2 s): extend with a case
  whose early return yields a DIFFERENT value than the fall-through, and
  assert the lowered function has exactly one `Ret` reading a `Phi` with one
  incoming per return site (today: `identity = (9, 1)`, last wins).
- Real: `_propose_dt_pen` via `tools/repro_metrics_rebind.py` shape; then
  `step_with_dt_control_used` via the full managed compile with
  `TURING_DEBUG_STRUCTURAL_OUTPUTS=1` — the `DUPLICATE-RES` scan
  (scratchpad script in the session log) must report 0 duplicate ids.
- Only then: native build + `tools/frame_parity.py` (needs approval).

## Out of scope here (known, separate)

`opaque-state-effect` (multi-field mutation of one object), single-result
calls returning their own record parameter (`coerce_metrics`), team audits'
remaining zip/ancestry sites.

## Status update (same day)

Implemented for returns nested in loops: reducer `return_slot_values` span
table (+ canonical remap); `LoopDescriptor.return_controls`; conjoined guard
chains for return controls; `LoopControlBlock(action="return",
return_value_ids=...)` placed at lexical position; SSA builder branches to a
lazily-created `function_exit` and `finish()` merges every return edge per
slot (`Phi` with `binding: return_merge`). Verified by
`tools/repro_return_merge.py`: one `Ret` in `function_exit`, one incoming per
return site, zero duplicate ids.

Done since: a `return` in a top-level `if` arm (`_propose_dt_pen`,
`pi_update`) -- the return control is appended to the arm's SequenceBlock in
`_ordinary_conditional_control_programs`, a terminal Return counts as a
structural branch effect, arm-owned callsites are scheduled inside the arm,
and a region-less conditional program is anchored before the first region
whose source line follows the `if` (`ControlProgram.anchor_region`).
Verified by `tools/repro_return_merge_toplevel.py` and the managed compile
(`tools/scan_managed_duplicates.py`: 170 functions, zero duplicate SSA
definitions, fix22).

Three identity defects the duplicate scan then exposed, all fixed at source:

- Region temporaries coined from a region-local watermark
  (`plan_region_to_ssa_instrs` decomposing a variadic `max` into a binary
  chain) collided with authored ids (`_propose_dt_pen` 50 x2). Now every
  planned region is expanded ONCE per function (`expand_plan_regions`,
  `copy_region_instructions` in hierarchical_plan.py) from a function-wide
  watermark above every id in play; all six consumers share that expansion.
- A canonical literal rematerialized in several regions was published by
  each of them (`step_with_dt_control_used` 342 x4). Literals known to
  `constant_values` are control-owned (`_materialize_control_constants`) and
  are never region outputs.
- The hierarchy plan emitted a region at its FIRST member's node position,
  so a region straddling another region's output ran before its producer
  (`pi_update`: the `self.acc` update read the stale field). The plan walk
  now orders regions atomically (`_atomic_region_node_order`).
- Informational: a written mutable scalar record field is in/out storage
  whose formal id every producer deliberately redefines (C backend publishes
  the last definition); the scan reports these as `INOUT-REDEFINED`.

## Break path: implemented (2026-09-05)

All three defects below are fixed; `tools/audit_break_in_if_trace.py`
(cases single/nested/param/carried/while_break, ~8 s each) and the managed
compile (`tools/scan_managed_duplicates.py`, fix29: 170 functions, zero
duplicate definitions, zero `loop-control-site`/`break-edge-value`
shortfalls) verify it.

- Reducer: every `break`/`continue` records the environment at the site
  (`loop_control_site_bindings`, span-keyed); the innermost loop claims
  its sites at loop end and publishes `loop_break_sites` (span ->
  {pre-loop id: value at site}) and `loop_break_bindings` (names bound
  only on break paths: pre-loop id, continuation id). Relabeled with the
  canonical mapping like `loop_carried_bindings`.
- Loop composer: `LoopDescriptor.control_sites` (node, action, guard chain,
  site values, enclosing-arm span) and `break_bindings`; break-only names
  get a `LoopResult` port (`result_kind: break_bound`, published as
  `result_ports` entry with updated == initial) and their continuation is
  rewired onto it. Lexical positions are pre-order (`source_order_walk`),
  not BFS. A site is ARM-OWNED iff a value it carries is produced by a body
  region inside the arm; otherwise it is placed lexically under the full
  conjoined guard chain (`guarded_expression`). Pending plans and node
  attributes retarget these ids on port rewires.
- Conditional builder: the same arm-ownership rule (region-produced value
  with arm membership) appends the `LoopControlBlock` to the arm; terminal
  arms contribute no carried aliases; loop ports are never a conditional's
  merge candidate. Pruned ids (dead bindings) are dropped on both sides.
- SSA builder: `LoopControlBlock.site_node_id/site_values`; the break edge
  carries the site's value for each carried/break-bound port (a known
  literal is materialized through `external_value`); exit Phis for
  break-bound ports merge the pre-loop value with every break edge;
  `control_site_ids` on `LoopBlock`/`WhileBlock` make an unplaced site a
  loud shortfall. `_value_dominates_current_edge` accepts a two-write
  carried slot (any producer block dominating) and ignores unreachable
  blocks (the continuation after an arm-owned break) in the dominator
  fixpoint -- both had made the exit edge carry the stale header value.

## Related defects found in the existing `break` path (audit_break_in_if_trace.py) -- historical

Same program, `run_superstep` lines 620-626 (`for boundary in
boundary_values: if boundary > total_value + eps: dt_try = ...; break`):

- (a) nested `if` guards for `break`/`continue` are REPLACED, not conjoined
  (`collect_loop_controls`): a break two ifs deep fires when the outer
  predicate is false. Fixed for returns via `chain`; apply the same
  `guarded_expression(chain)` to break/continue.
- (b1) `lexical_position` is BFS `ast.walk` order, so the `Break` node sorts
  BEFORE its arm's assignment region -- the exit is taken before `dt_try` is
  assigned. Statement-level controls must be positioned by source order
  (lineno, col), not walk order.
- (b2) the reducer's terminal-arm rule (`environment.update(else_environment)`
  when the body arm ends in Break) discards the arm's bindings, so `dt_try`
  is never loop-carried and `loop_exit` returns a value defined only in
  `if_true` (non-dominating use, measured). A `break` arm's bindings DO reach
  the loop exit through the break edge. Generalise the return machinery:
  record per-break-site carried values (span table) and let the break edge
  carry them explicitly instead of "updated-if-dominates-else-current".
