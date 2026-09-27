# Regulation through the concordance — report C (2026-09-26)

Continues [report B](CONCORDANCE_REGULATION_REPORT_2026-09-26_B.md).  Driver:
the woodshop dt system (`engine_toy/woodshop.py`,
`WorldRules.lower_newton_dt_system(..., backend="llvm", optimization="O0",
piece_mode="link")`).  At the start of this round it refused; it now lowers
completely.  Its concordance report went **22 → 14 findings**.  Every defect
below was reproduced in seconds on a small program against the authored
Python (native LLVM, C where noted) before the compiler was touched, and each
has a guard that fails at `f9432d80` and passes now.  The per-fix record is
in [`CONCORDANCE_MASTER_LIST.md`](CONCORDANCE_MASTER_LIST.md) §A3a–A3d.

## Working rule adopted this round

Operations on identities are operators recorded on the book, and passes
consume graph facts rather than rebuild them from names, source order or
value ids.  Three heuristics were replaced by graph facts instead of being
patched (§A3d); the gate and both linking suites were unchanged by each
deletion.

## Fixed (in order found)

| Defect (symptom) | Identity fact now recorded / consumed |
|---|---|
| Stale law-piece pickles (`SSASequenceTable` has no `sequences`) | engine_toy `_LLVM_PIECE_CACHE_SCHEMA` → v2 |
| Call rebuild renamed `args`→`arg:0`; carried read unattributed (refused) | `_set_operands`: the one writer of operand lists; page `identity_transition` (move/retire/fork) |
| `TransformationLedger.propose` O(E²) scan stalled call-result settlement | page `transformation_rejection`; `scope_row_count` |
| `total.item()` in a loop froze at the pre-loop value (wrong answer) | fork of the receiver read; planner page `item_operand`; control lowering reads the operand by binding (merge) |
| Keyword / method receiver reads had no row (refused) | read committed at `(scope, "occurrence", id)`; call positions fork from it |
| LLVM tuple `Const` wrote `i32 int(item)` into one slot (float tuples read 0.0) | typed, payload-sized emission (C lane's rule) |
| Carried updates read only by the loop were deleted by two pruners (`dt_cap`: body read pre-loop value) | loop node consumes `carried_update` / `break_value` edges |
| Continue arm: fall-through update taken as the if-merge; continue value unconsumed; call argument overridden by source-latest name (wrong answers) | continue sites recorded like break sites; history-synthesized merges and the name-history argument override **deleted**; non-falling-through arms restore bindings |
| Specialized copies shared one read scope | `fork_read_scope` in `extract_clean_process_subgraph` |
| `x is None` → `not presence` orphaned the read (audit) | presence lowering and all glsl working-graph writers migrated to `_set_operands` |
| `if` scheduled before its predicate's input region (`exchange_time_bound`) | `ConditionalBlock.predicate_region_indices`; ordering treats them as prerequisites |
| `float(x.item())` scheduled before `x`'s producer (`_propose_dt_pen`) | dependency signatures resolve an item feed through `item_operand` |

Audit: `operand-position-orphan` (planner records any read row at a
position its consumer no longer has; `CorrelationTable` reports it).

## Evidence

- Guards: `tests/test_loop_identity_operators_native.py` (7 programs, LLVM),
  `tests/test_loop_binding_reads_native.py` (+1), 
  `tests/test_conditional_predicate_order.py` (3),
  `tests/test_transformation_priority.py` (+1).  All fail at `f9432d80`.
- Gate (report B's focused files + linking + guards): **23 failed / 285
  passed**; all 23 fail identically at `f9432d80` on Windows (15 gate +
  8 in `test_process_graph_function_linking.py`).
- `tests/test_fortran_c_shell.py`: 20 failed / 65 passed, identical at
  `f9432d80`.
- Scorecard 18/19 throughout.
- Baselines taken in the clean worktree `C:\Users\alber\AppData\Local\Temp\wtb`
  at `f9432d80`.

## Follow-up: `run_superstep` use-not-dominated (14 → 13)

Not the predicate-region class.  `metrics, dt_next, dt_used =
step_with_dt_control_used(...)` then `if float(metrics.control_values[0]
.item()) > 0.0`.  `_plan_callsite_projection_ids` walked GetAttr/Indexed
transitively from the call's results and claimed the element
`control_values[0]` for the call; region 17 computes it (book:
`region_feed_consumer`, `consumer_operand` base).  One id, two producers:
`dependency_order` bound region 18's feed to the call, so 18, 19 and the
`if` ran before 17.  The walk now descends only through declared aggregates
(the call's result bindings, projections carrying `result_class_ref`).
`step_with_dt_control_used`'s direct field reads of `coerce_metrics`'
record stay the call's.  A first attempt (region publications override
projections) broke that case and was reverted.

- Repro (1.4 s) and guards: `test_conditional_predicate_order.py::
  test_call_projection_element_is_its_region_publication`,
  `test_loop_identity_operators_native.py[call-projection-element-predicate]`
  (LLVM matches Python; 2 findings at `f9432d80`).
- All 14 woodshop findings reproduce by lowering `examples/llvm_dt_system.py`
  `dt_system_over` over the one-law drift piece of
  `tests/dt_system/test_llvm_dt_system.py` (~165 s).  After: 13, lowering
  completes.
- Gate + linking + guards: 26 failed / 280 passed; all 26 fail at
  `f9432d80` (`test_fixed_width_sequence_append_passes_every_row_column`
  is order-dependent, fails in-file on both trees).
- Scorecard is **17/19** on this tree with or without the change (HEAD's
  walk swapped in memory): level 16, conditional assignment of a
  comparison, stops at EXECUTE (`UnboundLocalError: t9`).  18/19 at
  `f9432d80`.  So `d0991f81` regressed level 16; not yet confirmed on a
  clean `d0991f81` checkout.
- Also logged: the class page puts the first tuple member's class on the
  tuple call (`source_value_class_concordance (run_superstep, call) =
  Metrics`; `topological_reducer.py` ~10983 writes a single returned class
  onto the call when one of several outputs resolves); the member itself
  has no class row.  Value ids drift by ±1 between identical runs.

## Remaining woodshop findings (13)

1. ~~`use-not-dominated` ×1~~ — fixed above.
2. **`alias-target-missing` ×9** — `run_superstep` (71, 272 → 360/361) and
   `step_with_dt_control_used` (376–430 → minted ids).  Not investigated.
3. **`descriptor-member-shared` ×2, `descriptor-member-unknown` ×2** —
   `step_with_dt_control_used` handles 347/348, 592/593 sharing `column0`.
   Not investigated.

## Other defects found, logged, not fixed

- LLVM loop Phis are `phi ptr` over the backedge slot (lost copy):
  `grow(cap, factor=f)` read `f` after `f = f + 1` in the same iteration;
  C is correct.  Needs a copy only where the backedge value is defined
  before a later read of the Phi.
- C lane refuses `while` + call + `break` (`%tN is unavailable`) though the
  SSA dominates; LLVM correct.
- Kernel-emitter tuple `Const` (`@const.vec`, i32) has the same shape as
  the fixed one.
- Name-keyed class/field resolution: a test class named `Targets`, and a
  field named `energy_exchange_fraction` on another class, both resolved to
  `src.common.dt_system.dt_controller.Targets`.
- `Targets(...)` without the optional `energy_exchange_fraction` refuses at
  the `_propose_dt_pen` callsite ("bound to caller record ... which has no
  such field").
- `_set_operands` still pairs positions by operand identity and order;
  writers should declare their moves.  ~30 raw `parents` writes remain in
  `process_graph_fusion`, `loop_composer`, autograd and others; the audit
  flags any read they orphan.
- A dead `Cast` of a pre-inner-loop value still reads the outer generation.

## Commands

Woodshop (≈3 min with warm law cache):

```
cd engine_toy
python -u -c "from woodshop import WoodshopSimulation; from src.compiler.identity_concordance import concordance_report; sim=WoodshopSimulation(); system=sim.world_rules.lower_newton_dt_system('build/woodshop_newton_llvm', backend='llvm', optimization='O0', piece_mode='link'); print(concordance_report(system.module, limit=40))"
```

Guards (≈2 min):

```
python -m pytest -q tests/test_loop_identity_operators_native.py tests/test_loop_binding_reads_native.py tests/test_conditional_predicate_order.py tests/test_control_branch_compartments.py
```
