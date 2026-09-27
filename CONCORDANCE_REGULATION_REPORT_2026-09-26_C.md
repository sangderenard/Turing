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

## Remaining woodshop findings (14)

1. **`use-not-dominated` ×1** — `run_superstep` value 273.  Diagnosed, not
   fixed: `loop_control_next.1` (after the inner `for`) runs region 18, the
   predicate producer of the following conditional (`if_true.4`), which
   reads `%273`; region 17 produces `%273` in `if_merge.4`, after that
   conditional.  Same class as `exchange_time_bound`, for a conditional in
   the while body: find the builder of that conditional and make it declare
   its predicate regions (or find why region 17 is not ordered before 18).
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
