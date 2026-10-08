# Frontier audit, 2026-10-07

Audited from disk, not memory: branches, worktrees, scratch logs, the audit gate run
fresh on `main`. "Verified" means I ran it or read the artifact today; "worker-reported"
means a lane reported it before the previous session died (2026-10-06 auth failure
killed every background worker mid-task).

## 1. Verified state of `main` (e5a52f50, pushed)

- The 2-piece orbital dt system lowers with 0 shortfalls, emits C, builds, and runs one
  round window natively. **69 columns bit-exact** against the Python `dt_system` lane
  (67 state columns + `dt` + `telemetry`); the Python lane changes 12 of them over the
  window, so the match is not vacuous. Last run `n2_e2.txt`, 2026-10-06 14:38, on the
  committed HEAD. Driver: scratchpad `native_round.py`.
- Audit gate on HEAD: findings `0,1,0,1,5,0,0` (baseline), run today.
- Pre-existing failing tests, NOT from this week (fail identically at c0781c7a):
  4 in `tests/test_process_graph_function_linking.py` (`test_record_field_storage_identity_crosses_the_call_frame`,
  `..._uses_fields_without_python_receiver_handle`, `..._refreshes_completed_physical_field_surface`,
  `..._feed_structural_call_argument`) and 3 in `tests/test_declared_parameter_shape.py`.
  Unowned; nobody has bisected them.

### What landed this week (all on `main`, all pushed)

| Theme | Commits |
|---|---|
| Solve: C lane, loops, dt loop, matrix (24 cases both lanes), float32 lowering | 3fefab8d 05bf36a3 ad72f1f9 3ca49b46 ed658793 |
| Loop guards: `BoundedFixedPoint` + receipts, adopted in 5 unbounded compiler loops | 4f0da89f 6df8c300 e01ad889 eaaa7d42 36d6f3b0 6dc1e13b 4320ce28 |
| Superlinear costs found by py-spy: graph copy pickled bindings; N^3 operand match; ABI publication per recursion level; value-id rescans; descriptor reuse; `latest` reads last column; polymorphic scan memo; repeated-post short circuit | c5a88c58 80a64f87 73a4b501 f9106c5e 2c811ee9 2b28e675 0931589b 0cd2beab 6b19f07c b578b5cd |
| Link mode: a compiled piece enters as its declared signature only (graph -36..-44%); callsite dedup per (callee, signature) under a declared policy | 15fceab4 03b2ee32 |
| 2-piece lowering errors cleared (6) + loose ends (2 of 3) | f580e6f4 e8681aae 3b772af9 4f742eae 141b6a07 c0781c7a 529fd9ed e5a52f50 |
| Detector false positive; identity log levels/lzma; compression tooling | 8b0caf51 4241444f |

### Measured curve (new code; before -> after the stub)

| Pieces | Graph nodes | Closure build | `callsite-tensor-specialization` |
|---|---|---|---|
| 2 | 21,704 -> 13,787 | 9.1 s -> ~4 s | — (whole run to the same point 984 s -> 280 s) |
| 4 | 43,631 -> 26,578 | — -> 8.6 s | was 448 s |
| 8 | 83,448 -> 46,521 | — -> ~18 s | DNF (45 min) -> 329 s, finished |

## 2. Loose ends on the 2-piece window (user: close before advancing)

| # | Item | Status |
|---|---|---|
| 1 | Stale `initial_value_id` on return-merge field Phis | **FIXED** 529fd9ed: the pass-through fold posts an `aggregate_passthrough_rebinding` row DERIVED from both `ssa_value` cells; the Phi's initial is read from that row. Probe `probe_aggregate_passthrough_initial.py`. |
| 2 | Silent skip in `repair_non_dominating_record_phi_uses` | **FIXED** e5a52f50: now an `Unresolved(RECORD_PHI_FALLBACK_NOT_DEFINED)` row and a raise. Probe `probe_record_phi_fallback_refusal.py`. N=2 unchanged (54/84 repairs, loud path not reached). |
| 3 | The unchecked `LNot` use + whole-module dominance | Probe WRITTEN (`tools/compiler_probes/probe_module_dominance.py`, untracked): three checks over the whole lowered module (undefined operands, definition dominance incl. Phi incomings on their edge, stale `initial_value_id`). RESULT: see §2a. |
| — | `clang -Wuninitialized` on emitted C = 0 | **UNVERIFIED** (worker died before recording it). |
| — | attempts/rejections per window | not exposed by the driver; inferred one accepted attempt (full 1.0 window, `hard_failure` 0). |

### 2a. Dominance probe result (N=2)

Run today on `main` e5a52f50 (459 functions): **10 violations, NOT clean.**

- `undefined_operand`: 0.
- `definition_does_not_dominate_use`: 1 — the flagged `LNot` is real:
  `step_with_dt_control_used`, `function_exit[33]` reads a `Load` defined in
  `if_true.19`, which does not dominate `function_exit`. Same bug class as c0781c7a
  (a read of a value not defined on every path); the native run passed only because
  clang happened not to exploit it.
- `stale_initial_value_id`: 9 — all on NON-record Phis (`record_field: None`), so the
  529fd9ed fix (record-field pass-through fold) did not cover them: `run_superstep`
  `if_merge`/`while_header` (2), `step_with_dt_control_used` `while_header`,
  `if_merge.11` (2), `if_merge.13`, `if_merge.14` (4), `_apply_energy_sidechain`
  `if_merge.2` (1), `exchange_time_bound` `if_merge` (1). Each names an initial the
  function does not define — a use waiting for a repair that cannot be made.

**CLOSED 2026-10-08** (commits 36849e6a, aa78fa7a on `main`): the probe now reports
458 functions, 0 violations; `native_round.py 2` still `PARITY 69 column(s) bit-exact`;
clang 20.1.7 `-Wuninitialized -Wsometimes-uninitialized` on the emitted C: 0 warnings
(sanity-checked against a deliberately uninitialized file).
- Fix A (the LNot): `recover_structural_source_outputs` rebuilds each anonymous
  control-predicate formal as a chain before `Ret`; the arm's own `control_expression`
  already computes it, so the chain is dead, but the dead-pure sweep's vocabulary
  (`ir_identities._PURE_REGION_OPS`) lacked `LNot`/`Select`/`isfinite`, so the tail
  survived in `function_exit` reading an arm-local Load. Vocabulary widened; each
  retirement is now a `dead_structural_retirement` row derived from the value's
  `ssa_value` cell. Repro `probe_dead_structural_dominance.py`.
- Fix B (the 9 Phis): `initial_value_id` was the PLANNER's spelling of the pre-merge
  version, not the SSA id the lowering holds (they differ for planning aliases, minted
  literals and builder stand-ins); only record-field Phis carried the SSA id. Every
  carried-Phi emission now routes through `_phi_initial_binding`, which posts a
  `phi_initial_binding` row (spelled id, resident id) derived from both `ssa_value`
  cells and reads `initial_value_id` back from it; alias settlement rebinds the initial
  through the same `alias_application_concordance` row. Repro `probe_phi_initial_resident.py`
  (hand-built IR, the audit's oscillator/mapping lowerings, and a specialized re-lowering).
- Found in passing, untouched: the audit's `toplevel`/`controller`/`controller_untyped`
  lowerings still report `undefined_operand` in `Phi`/`restore_type` functions; a retry
  loop with a `break` hits a `break-edge-value ... does not dominate the exit edge`
  emission error.

## 3. Lanes: landed / in progress / never started

All lanes are Sonnet workers in their own worktree branches; none is merged. Every
worker died at 2026-10-06 ~15:00 (OAuth revoked). Branch state is from `git` today.

| Branch (worktree) | Commits ahead of main | Uncommitted | Status |
|---|---|---|---|
| `split/glsl-leaves` (wtv) | 9: shared digest list `compiler_implementation_files.py` (07560d7e) + 8 leaf modules out of `glsl_deployment_strategy.py` (source_node_facts, call_argument_binding, region_scheduling, deployment_profiler, formal_shape_ledger, planner_concordance_posts, planned_shell_tree, receiver_resolution_prepasses) | 9th move in progress: `source_control_retention.py` untracked, facade edited (-462 lines) | worker-reported verified per commit (imports, memo test, product test, audit); NOT re-verified after the crash |
| `split/shell-leaves` (wtc) | 9 leaf modules out of `fortran_c_shell.py` (declared_piece_signature, ast_control_normalization, native_link_audit, native_shell_abi, authored_parameter_abi, frame_identity_book, frame_storage_roles, resident_sequence_materialization, sequence_storage_binding); digests added to the existing explicit lists | 10th move in progress: `lexical_control_placement.py` untracked, facade edited (-1598 lines) | `extra_wtc.txt`: 20 passed; `cur_fail.txt` = the 4 pre-existing failures; NOT re-verified after the crash |
| `fix/region-metadata-copy` (wtm) | 1 (9df413c7): a dispatch region copies only its own nodes' identity tokens, shares read-only tables; test added | clean | audit CASES: fresh items per region copy 3.52M -> 0.145M; **N=8 NOT re-measured** (the 37 GB wall) |
| `fix/state-machine-tick-rows` (wts) | 0 | clean | never started |
| `fix/host-layout-rows` (wth) | 0 | clean | never started |
| `fix/per-copy-shape-proofs` (wtp) | 0 | clean | never started |
| `fix/float32-result-dtype` (wtf) | 0 | clean | never started |
| `fix/ellipsis-store-ingestion` (wte) | 0 | clean | never started |
| raw_primitive sweep, cross-book stamps, closure-build costs | no branch, no worktree | — | never started (died before `git worktree add`) |

### 3a. Lanes restarted 2026-10-07 (all 8, Sonnet, off `main` 9297df18+)

| Branch | Landed (unmerged) |
|---|---|
| `fix/cross-book-stamps` | 129c20fa: the cross-book bug was already fixed on main; the lane fixed a root-graph double visit in callsite tensor specialization (VJP test red on its own). 3 test files green together (17 passed, 1 pre-existing xfail). |
| `fix/host-layout-rows` | 5 commits (51b7a473..52e76de1): slot rows, API_CONTRACT, `<entry>_layout.h`, `bind_column`, probe bit-exact. See §5. |
| `fix/closure-build-costs` | 3 commits (3617c136, ec29c88f, 5d0760c7): run-length edge batching in `connect` (whole-node batching would reorder shared sources' `children`, so it was rejected), append-only `_set_operands` shortcut with position facts read only where positions move, hub-children index. 900-arg call `build_from_ast` 3.30 s -> 0.55 s (now ~linear: N=100 0.03 s, N=300 0.17 s); hub cases 1.8 s -> 0.48 s. Equivalence test: every page's rows identical in order (two global logs as multisets); build-stage output identical to base on 4 sources (37,600 rows). Also fixed a latent bug: every new edge of one call shared the same `extra` set object. **Next target it names:** the reduce stage is unchanged (~3 s on the 900-arg call): `_redirect_value` calls `_set_operands` ~5,400 times through the general path, and `deduplicate_node` scans all graph nodes per non-AST node. |
| `fix/state-machine-tick-rows` | 7 commits (46be24b0..b643e4f3): the tick is on the book end to end — `control_block` row keyed by the tick's canonical cell, DERIVED from the owner, the state VALUE cell and each case-literal cell; `state_case/state_next/state_merge` `ssa_block` rows routed (the `SSA_BLOCK_OWNER_UNROUTED` and `CONTROL_OWNER_UNKNOWN` exemptions retired for ticks); carried merges as `carried_snapshot` + `CONDITIONAL_MERGE` rows; two ticks in one function get distinct rows; the producer wire exists (a planned `match` ingests as one dispatch node, `install_state_machine_control` builds the tick with per-arm callsite markers). Probe: a 2-state `Ramp(AbstractTensorStateMachine)` lowers with a real record contract and runs on C and LLVM bit-exact with the Python object over 5 steps incl. the flip. **Four hunks sit in the two files the split branches move** (isolated in 416893fd: the install hook in `_class_surface_ssa_program`, a `_place_plan_callsites_lexically` filter, `ast.Match` in `_dispatch_metadata_rule`, `replace()` in `fix_aggregate_loop_bounds`) — fae98f5a and 416893fd go together or not at all. Not done: returning arms (`translation_shortfall` instead), `match int(x.item())` subject, `MatchSingleton` literal cells, field-carried merges, the autograd reversal still rebuilds ticks ownerless, `StateMachineDomain` still dead. |
| **NEW baseline miscompile** (found by the tick lane on unmodified main) | `if cond: self.phase = 1` on a scalar record field writes the literal ONCE at function entry (the store is placed after the constant's producer, `_inject_field_slot_access`), so the field flips on step 1 even when `cond` is false. `self.phase = self.phase + 1` is placed correctly. Lane `fix/conditional-field-store` dispatched 2026-10-07. |
| `fix/ellipsis-store-ingestion` | 1e7f040c: a sole `x[...] = v` ingests as ONE `IndexedStore` (`node_special_cases._EllipsisExpander.visit_Subscript` leaves a whole-`...` index as authored; `_dispatch_metadata_rule` now classes `Ellipsis` as a basic-index literal — it was silently dropped as dispatch metadata; `ir_indexing` no longer treats `...` as a scalar selector; an Ellipsis reaching LLVM is a loud shortfall). Rows identical to `[:]` and to the explicit expansion (`reducer_field_state` ELEMENT_WRITTEN, `record_storage_alias`, `dispatch_store`). Ingestion-only counts, N=2 cached pieces: `PieceState.restore` 2,184 -> 756, `advance_pieces` 1,034 -> 698, whole graph 14,005 -> 12,134 (21 -> 6 nodes per store). New test (6) bit-exact on C/LLVM vs eager and vs the expansion. Behaviour change: `x[...]` on a tensor with NO declared extent now fails loudly at LLVM emission, exactly as `x[:]` already did. |
| `fix/float32-result-dtype` | 0c3d8be0 (partial): planner/SSA now keep float32 (widest-declared-float rule with weak literals, in `_value_shape_dtype`, `hierarchical_plan.promoted_numeric_dtype`, `ir_indexing._propagate_scalar_dtypes`; `arange` reads its `dtype` kw; the float64 `physical_dtype` stamp on f32 kernel results removed), but the public buffer is still `double` because the dtype authority declares only bool/int32/int64/float64 storage classes. Decision given 2026-10-08: add float32 as a declared storage class. **DONE** (6a9ca446): float32 declared once in `dtype_layout.py` (joins the C and LLVM lane tables; `llvm_abi_numpy_table()` derives the LLVM public-buffer dtypes from the same layouts — no second hand-kept list); LLVM `float` spans with `fpext`/`fptrunc` at loads, C `float` buffers, an input Cast where a float32 span feeds a double-backed kernel. **f32 cases now return float32 on both lanes and MATCH NumPy's float32 solve: 2.4e-7, 7.5e-9, 2.1e-7, 7.2e-7**; all 20 float64 cases unchanged; solve probes NUMERIC_MATCH; audit unchanged; 14 test files same failing set as base. Honest limits (kernels still compute in double, rounded once): +,−,×,÷,abs,max,broadcast are correctly-rounded float32; `sum`/`reduce_dim`/`cumsum`/`matmul` accumulate in double (more accurate than float32 accumulation, not bit-identical); transcendentals/fma/sqrt untested at float32; float32 SCALARS stay double. Follow-up: float kernel variants in the bank. Queued as merge item 11. |
| `fix/conditional-field-store` | 7f345f24: root cause `_inject_field_slot_access` (precompile_to_ssa.py ~12600) put every field write right after its SOURCE's producer — a literal's producer is the hoisted entry-block `Const`, so `if cond: self.phase = 1.0` stored on every path. Control lowering already emitted the store inside the right arm (`ScalarFieldWriteBlock`), but its destination was the field's read value (a Load result), so it was dead. Fix: `_field_slot_ops` records one WRITTEN cell per write op; the injection finds the arm Store by that cell and rewrites it in place into the real slot store; placement row `control_block_placement (scope, Ref(control_block, (scope, SCALAR_FIELD_WRITE, <WRITTEN cell>)))`, and the minted index/GEP ids take it as an operand (edge on the book). Probe variants scalar / int_literal / elif_chain / scalar_else: MISMATCH on main -> MATCH on both lanes; element / increment / outside_value were already right. Same failing-set as main on 15 test files; audit unchanged. **Two defects it left (lane `fix/field-write-leftovers` dispatched 2026-10-08):** (1) `return self.phase` after a conditional write returns a Load defined only in `if_true` — the C lane returns garbage on the false path; this IS the baseline `controller` audit finding ("Ret reads a value defined at if_true"); (2) the after-producer path still misplaces reference-field writes (`self.x = None` in an `if`) and writes whose source id equals a receiver column id. |
| `fix/per-copy-shape-proofs` | 3 commits (183b216d, 2440defa, 33b135c4): shape proofs keyed per COPY — `shape_scope_of(copy)` is minted once per graph copy, forked per callsite specialization (`fork_shape_scope` posts a `scope_origin` row derived from the source scope's registry cell), stamped on every IR function; pages re-keyed: `proven_shape`, `shape_transformation_state` (+concordance/dependents), `sequence_row_layout_concordance`, `callsite_descriptor_reuse`, `callsite_return_specialization`, `callsite_tensor_result_specialization`, `tensor_shape_settlement_concordance`, `shape.{node,linked,ssa}`; the authored name stays reachable through a `shape_scope_function` page; `value_shape`/`formal_shape` stay name-keyed by design. Child shells whose scope is a fork now run the callsite tensor propagation (recursing into transient copies, memoized per book). **Result: the mixed 2x2/3x3 solve in one function lowers and MATCHes on both lanes** (size 2 err 0, size 3 2.2e-16); all 24 default cases unchanged; solve probes NUMERIC_MATCH; audit unchanged; fixes one previously failing test. Also changed dtype promotion (`_promoting_sides`: an authored literal is a weak scalar against a floating tensor) — overlaps the f32 lane's identical change; reconcile at merge. Queued as merge item 10. |
| `fix/field-write-leftovers` | 2 commits (e81ac88f, 8712435c) on a cherry-pick of 7f345f24. Defect 1 root cause was NOT the merge: after injection the IR is right (`Load` at `if_merge` after the arm's Store, `Ret` of it); `coalesce_record_field_storage` (fortran_c_shell.py ~23765-23925) then posts a `record_storage_alias` from every getter onto a resident chosen BY ORDER, which after injection is the first pre-write Load (arm-local) — rewriting `Ret` to it. Same bug in straight-line code (`self.phase = self.phase + x; return self.phase` returned the pre-write value). Fix: no alias for a field whose reads are slot-loaded — first done as function metadata, sent back 2026-10-08; **now a posted row** (f1753bce): page `receiver_field_slot_read` keyed (function scope, read value id) → (slot, load id), DERIVED from the slot address's `ssa_value` cell, the read's `ssa_value` cell and the field's `reducer_field_state` write cell; the coalescer reads it with `latest_ref`; the metadata key is gone. Queued as merge item 12. Defect 2: reference-field writes had no `ScalarFieldWriteBlock` (only scalar-storage fields got one), so the after-producer path stored at entry on every path; declared mutable reference fields now get a block with dtype `opaque_ref` and the arm-owned placement row; a shadowed local named `identity` (~8899) crashed every static-reference field write; the C lane had no `StaticRef` spelling (added). The `receiver_alias` case has no source-level repro (only reachable when a region output reuses the receiver's id) — covered by a unit test on hand-built IR; the exclusion removed. Probe now 14 variants, all MATCH on both lanes (5 were MISMATCH/FAIL on base). **Correction to §2/§3a:** the baseline `controller` audit finding is NOT the `Ret`-after-conditional read; it is `operand-never-written` on a `CondBr` operand in `AbstractTensor.__restore_type__` — a separate open defect. Hunks in fortran_c_shell.py outside 7f345f24's footprint: ~8899, ~19534, the coalescer (~23818 and two removed filters), ~25300. |
| `fix/raw-primitive-posts` | 19 commits: 24 of the 54 detector groups migrated to `book.post` with real sources; **unsourced facts per audit program down ~90%**: view 4506→353, toplevel 2888→417, energy 3638→294, controller 4311→584, controller_untyped 4206→502, mapping 188→16, oscillator 2818→220; audit findings unchanged; solve probes NUMERIC_MATCH; same failing test sets as main. To ZERO: `transformation_event`/`_decision` (settle_call_result_types; 16 left are values with no `ssa_value` cell), `identity_transition` (fork_read_scope posts `scope_origin`), `lexical_read_binding`, `consumer_operand`, `call_argument_operand` (newly declared), `ssa_call_shape(+evidence)`, `shape.node/linked`, `call_edge`, `scheduled_call_argument`, `kernel_by_value_formal_concordance`, `pruned_callee_formal_concordance`, `argument_binding_resolution`, `loop_scope(+inner_transition)`, three `source_*_concordance` pages, `operator_result_type`/`scalar_kernel_operand`, `callsite_return_specialization`, `callsite_tensor_result_specialization`, `tensor_shape_settlement`, `ssa_value` adoptions (fallback to `ingestion_value` cell), `graph_tensor_descriptor` edges (2 left), `ingestion_value[reduction]` (6 left; unrolled clones now post canonical rows). Reduced: `loop_result_reconciliation` 7138→875 (restated after IR rewrites with no cell for the rewrite), `shape_transformation_state` 1606→44, `repository_ssa_enrichment` edges 341→88, `aggregate_ledger` 74→6, `call_record` 88→43, `cross_function_references` 206→83. **Stopped on (28 groups):** `structural_specialization_fixed_point` 703 (raw tensor annotations on node data have no cell), `proven_shape` 296 + `tensor_shape_enrichment` 242 (their columns are dependency levels / (round, step) and readers depend on them — need a redesign), `control_block_placement` 17 + `control_program` 3 (`_post_control_rewrite` called without its cause cells), `call_record` rewrites 43 + `call_link_order` 15 (the linker rewrites whole lists with no cell for the pass), `assignment_projection_*` 14 (normaliser emits tuples without cells), `formal_shape` 23, `linked_caller_member` 150, `transformation_rejection`, and a few singletons. Recording bug found, not changed: `enrich()` records every shape edge's `source_scope` as the TARGET's function even for caller/callee values (fixing it changes invalidation). Heavy overlap with the glsl split and the per-copy branch in the descriptor query / callsite fixed point / consumer-operand posts: **to be rebased onto main by its author after merge item 12**, then merged as item 13. |
| loose end 3 (main tree) | running: 10 dominance violations at N=2 (§2a); the probe is committed (78a69087). |

Both in-progress worktrees import cleanly today (`import src.compiler.glsl_deployment_strategy,
src.compiler.fortran_c_shell` ok in wtv and wtc), so the uncommitted moves are not broken,
only unfinished. The three detached baseline worktrees (`wtb2`, `wtmb`, `wtv2`, clean at
c0781c7a) were removed today. Disk: 57 GB free.

## 3b. Merge status (2026-10-08)

All twelve lane branches are merged into `main` (488bc9a0, pushed), one at a time, in
the order region copy, cross-book, closure costs, ellipsis, conditional store, host
layout, tick rows, shell leaves, glsl leaves, per-copy shape proofs, float32 storage
class, field-write leftovers (the last as a cherry-pick of its three commits because it
carried a duplicate of 7f345f24). After EVERY merge: audit findings 0,1,0,1,5,0,0,
solve-in-dt-loop NUMERIC_MATCH on both lanes, the branch's own probe/tests with no new
failures, `probe_module_dominance.py 2` = 458 functions / 0 violations, and
`native_round.py 2` = 69 columns bit-exact. Conflicts resolved: the audit doc (both
texts kept), the digest list (all nine shell-leaf modules added to the ONE shared
`compiler_implementation_files.py`; the three readers read only that list — so the
site-bundle and GLSL digests now also cover the shell-leaf modules), and the f32/per-copy
weak-literal helper (one `_promoting_sides` kept; the f32 call site uses it).
Not merged: `fix/raw-primitive-posts` (19 commits; being rebased by its author onto
488bc9a0, then merge item 13); the two uncommitted in-progress moves in the split
worktrees (`lexical_control_placement.py`, `source_control_retention.py`).

**N=4 native: DONE (2026-10-08).** `PARITY 305 column(s) bit-exact, 0 mismatched`
(303 state columns + `dt` + `telemetry`; Python and native telemetry identical). The
driver takes every column from `MachineCraft._initial_columns`, the same method
`OrbitalJumper.__init__` uses, with the arguments `OrbitalGame` passes (`orbital_game.py:761`:
`GravityCenter` at the origin with `MU_EARTH`, `_circular_state(MU_EARTH, CRAFT_RADIUS_M, 0.0)`,
identity attitude, zero angular velocity, `green_coast=True`) — 1,332 columns from the 35
pieces' outputs, none unprovided. End to end 1,360 s (22.7 min), ~21 GB.

| Stage | Time |
|---|---|
| piece load + authored columns | 87 s |
| source closure | 7 s |
| topology reduction | 10 s |
| deployment select | 139.5 s (`extract-dispatch-subgraphs`: restore 68 s, copy_shallow 26 s; callsite tensor specialization on the 18,738-node catalogue 18.6 s) |
| deployment instantiate | 141 s |
| call-topology planning (40 shells) | 16.5 s |
| SSA lowering (1,066 functions, 35 exports) | **850 s — of which ABI settlement round 1 = 681 s** |
| pre-native dominance repair | 18 s |
| C emission + compile/link | 30 s + 10 s |
| Python lane + native prepare (346 buffers) + run | 7.5 s (native window 2.4 s) |

**The N=4 wall:** ABI settlement round 1 sits in `_fold_callsite_structural_values`
(glsl_deployment_strategy.py ~21921): the dead-metadata loop removes ONE node per
whole-graph scan and restarts (`while dead_metadata: for node in tuple(G.nodes): ... break`),
quadratic in nodes × removals; rounds 2+ take 13-27 s. Invisible at N=2 (whole settlement
~55 s). Lane `fix/dead-metadata-worklist` dispatched. Note: the cached pieces are now
STALE against the merged compiler (the driver loaded them without rebuilding; the game's
own path would rebuild all 35, ~26 min).

### 3c. Lanes running after the merge (dispatched 2026-10-08, all Sonnet, off `main` 887a858b)

| Branch (worktree) | Task |
|---|---|
| `fix/raw-primitive-posts` (wtr) | **REBASED** onto 887a858b (20 commits; five conflict stops: shape-page withdrawals re-keyed with `_shape_key(scope)`, `enrich` keeps main's `shape_scope_of` recording plus the lane's `source_owner` for source cells, `_frame_call_cells` re-added in `frame_identity_book.py`, `_record_aggregate_ledger_lookup`'s post re-applied in `call_argument_binding.py`, `callsite_return_specialization`'s callee field declared as a scope). Gates pass: audit 0,1,0,1,5,0,0; solve probes NUMERIC_MATCH; `mixed_refused` MATCH; 20 touched test files same failing sets as main. Unsourced per program after the rebase: view 390, toplevel 443, energy 304, controller 660 (85%, the one below 90%), controller_untyped 524, mapping 16, oscillator 225 — the rise over the pre-rebase numbers is per-copy keying creating more shape rows (`proven_shape` 296→1248; withdrawals with no source state cell 44→~106; `graph_tensor_descriptor` edges 2→14, uninvestigated). Detector groups 54→27. **Ready to merge as item 13 once the build slot is free.** |
| `split/shell-leaves-2` (wtc) | next `fortran_c_shell.py` moves: lexical_control_placement, late_source_recovery, shortfall_settlement, dead_storage_pruning, then the non-leaf areas in dependency order up to `whole_source_lowering` (public wrapper stays in the facade; giant function stays whole) |
| `split/glsl-leaves-2` (wtv) | next `glsl_deployment_strategy.py` moves: source_control_retention, tensor_descriptor_query, callsite_literal_specialization, captured_region_programs, validation_control, conditional_control_programs, dispatch_metadata_classifier, dispatch_region_partition, dispatch_subgraph_extraction, then structural_fold, scheduled_capture_coordinator, shell_hierarchy_builder, callsite_descriptor_application, callsite_tensor_specialization last |
| `fix/audit-findings-zero` (wta) | The 7 findings decoded: all `operand-never-written` — a branch/Phi reads a stand-in for `isinstance(...)`, a compile-time type test the structural fold could not decide. (A) 2 fixed (one commit): `isinstance(value, AbstractTensor)` where `value` is a callee formal every caller binds to a computed tensor — a new `formal_tensor_class` page keyed (callee read scope, parameter), one vote per call edge in the callsite fixed point, posted when all edges agree, derived from every callsite actual's identity cell, restated per copy; the fold decides the guard from that row. Audit now `0,1,0,0,4,0,0`. (B) 4 in `controller_untyped` are the audit correctly reporting an under-declared root (`dx`/`dt_prev` undeclared by design of that case) — decision: keep as the declared expected baseline, stated per case in the audit tool. (C) `toplevel` 1 is a DEFECTIVE CASE: it calls `_propose_dt_pen` with 4 of 5 args (also `tools/repro_return_merge_toplevel.py`, `repro_targets_expansion.py`) and leaves `distribution` undeclared — decision: fix the case and declare it; and the lane found a real compiler bug behind it: `_tensor_descriptor_rule` re-derives a declared scalar Input's python type from its dtype because an ABI-declared Input already carries a `tensor` dict, so a `NoneType`-declared `distribution` emptied `_propose_dt_pen` from 69 nodes to 16 — fix at the source (declared type wins). Expected end state `0,0,0,0,4,0,0` with the 4 declared. Untyped unsourced +31 from other writers' raw posts on the newly lowered arm (noted). Lane continuing. |
| `fix/break-edge-value` (wtbk) | **LANDED** fd122682, two root causes: (1) terminal-arm placement — a `break` ending one arm of an `if` whose other arm runs retained regions was left lexical (placed after the merge), so the surviving arm's value did not dominate the exit edge; now arm-owned (`loop_composer.sibling_arm_span` / `arm_owned_site`, and `arm_loop_control` in `_ordinary_conditional_control_programs`); (2) the reducer keyed the per-site break value table by pre-loop identity ALONE, so `tried = dt` (two names seeded from one id) overwrote the first — the exit Phi for `dt` took `tried`'s value: **a silently wrong result on BOTH lanes** (`body-value-after`: 1.2/−0.8/8.2 vs the right 0.625/−0.375/4.12); now keyed by the carried pair (initial, update), the same pair `_post_loop_carried_binding` uses. No new row needed: the exit-Phi `loop_result_port_binding` / `carried_port_value` rows were already correct; the defect was which value each site selected. Probe `probe_break_edge_value.py`: 9 variants MATCH on both lanes (3 were FAIL/MISMATCH on main). Same failing test sets as main (42/278, 15/66). **OPEN:** `break-nested-guard` (`if c: dt = …; if dt < x: break`) is still placed at loop level, reads the arm-local predicate out of dominance and drags the `isfinite` chain into `while_exit`; LLVM prints MATCH on it despite the violation, so an LLVM match there is not evidence. Queued as merge item 14. |
| merge worker (main tree, build slot) | N=4 done (§3b); now merging item 13 (raw-primitive) with the full gate |
| `fix/dead-metadata-worklist` (wtd) | **LANDED** 8abd4997: the restart-from-zero scan is one call to `_remove_dead_metadata_nodes` (seed scan, then a min-heap keyed by original node position so the same first-dead-in-node-order sequence is removed — only predecessors of a removed node can newly become dead). Equivalence test (48): random graphs incl. shuffled order and protected nodes, a 1,500-node chain, and all 7 audit lowerings run under both implementations with identical removal lists, node lists and book rows in write order. Synthetic worst case (every removal exposes the next): 9,000 nodes 41.3 s → 0.055 s; audit CASES unchanged (too small to show). Measured counts on the CASES: scans = removals + 1 per call, as expected. The sibling `while pruned` branch-pruning loop (~21602) left alone: each pass changes the graph the next reads, so it is not a pure re-scan; not the sampled hot spot. Queued as merge item 15; the N=4 re-run after it is the real measurement. |
| `fix/progress-and-memory-budget` (wtpg) | user request 2026-10-08 ("self regulate down instantaneous memory and offer progress indication"): done/total progress through the existing callback on every silent stage, RSS on stage-boundary lines; a declared `memory_budget_bytes` work-contract policy whose releases (recomputable caches, retained region copies, in-RAM logs) post `memory_release_receipt` rows; unconditional releases where a copy is dead after its program exists (an architectural fault, fixed as such); never a kill |
| `fix/fork-read-scope-projection` (wtfr) | `fork_read_scope` re-posts every row of the source scope per copy and never retires: project to the copy's own nodes as a declared relation on the `scope_origin` row (the region-metadata-copy precedent), loud miss for non-member reads, retirement if the book supports it |
| profile lane (read-only, waits for the build slot) | N=2 lowering under py-spy per stage + tracemalloc/RSS at stage boundaries + object histogram at the peak: the measured basis for the next efficiency lanes (deployment instantiate 141 s and select 139 s at N=4 are unprofiled) |

## 4. Next walls (measured)

1. **N=8 `extract-dispatch-subgraphs`**: 37 GB private / swapping at the `restore` shell
   (per-region deep copy of `graph.G.graph`, 2.18M items x ~68 regions). Fix exists on
   `fix/region-metadata-copy`, unmerged, unmeasured at N=8.
2. **N=4 / N=8 full native runs** have not been done since the stub (only closure and the
   specialization stage were timed). The 2-piece window is the only one proven native.
3. **35 pieces**: not attempted since the stub. Closure at 35 was 323 s / 225k nodes
   before the stub (expect ~-58% nodes).
4. Known compile-time items still unfixed from the efficiency catalogue: `fork_read_scope`
   re-posts every row per copy; `publish_program_abi_graph_identities` receipt invalidates
   on any node/edge change; invalidation BFS by whole-graph scans; closure-build per-edge
   operand rewrite; `specialization_state` repr-hash per round; generated `PieceState`
   stores at ~32 nodes/column (ellipsis-index expansion ingested per column).

## 5. Open concordance gaps (known, unfixed; the user's absolute rule)

- 55 unsourced-fact groups on the solve_dt probe: 43 `RAW_PRIMITIVE` writers unmigrated
  to the single post API (`loop_result_reconciliation` 7138 cells, `transformation_event`
  2222, `transformation_decision` 2208, `identity_transition` 457, `consumer_operand` 418,
  `aggregate_ledger` 74, `argument_binding_resolution` 32, `lexical_read_binding` 273,
  `exact_region_feed_dtype` 85, call pages ~40), 4 `shape_source_not_on_book` groups
  (~900 cells), 8 adoption groups with no canonical cell (~1100 cells); 150 unsourced
  `linked_caller_member` rows the detector does not report.
- Link mode: the native `PIECE_FILE` row is an unsourced root (`PIECE_ARTIFACT_UNROUTED`);
  the stub's `source_span` rows are `INGEST_SOURCE` roots; the call has no edge to the
  piece's build book (one-anchor-per-piece not done).
- The marked state-machine shell (`StateMachineTick`) is off the book (owner falls back to
  the program cell; block rows `SSA_BLOCK_OWNER_UNROUTED`; case literals not cells; no
  merge Phis) and unconnected (`G.graph["state_machine_controls"]` has no reader;
  `StateMachineDomain` is dead vocabulary; Python `match` is never ingested).
- Host-facing layout facts are not on the book (`NativeSystem.layout()` is Python-only,
  no callers); the written-slot-wins rule exists twice (compiler scalars vs
  `NativeSystem.state_field_ids` spans). **Lane `fix/host-layout-rows` (5 commits,
  unmerged, 2026-10-07):** `program_abi_field_slot` rows per (entry, backend, parameter,
  field, role) DERIVED from the resident formal's `ssa_value`, `function_parameter` and
  BUFFER_ORDER cells; `NativeSystem` reads them (its duplicate rule deleted);
  `API_CONTRACT` row from the function-header/formal units, `function_output` cells,
  BUFFER_ORDER and the slot rows; `<entry>_layout.h` posted as a `SOURCE_FILE` part
  derived from it; `bind_column` adapter; probe bit-exact. **New gap it exposed:** a
  formal's `program_abi_*` accounting (`written`, `storage`, parameter, field) is NOT a
  book cell — passes set `formal.accounting` as attributes — so those parts of the slot
  fact are payload, not derived. Still C-only (LLVM gets slot rows, no contract/header);
  the scene adapter still waits on declared column roles; `batch` is host payload.
- Pre-existing failing tests are more extensive than the 7 named in §1: the lane's
  baseline at 78a69087 shows 24 failures across the whole of
  `test_process_graph_function_linking.py` + `test_declared_parameter_shape.py`, and
  17 failed / 4 errors across the 12 emission-related test files
  (`test_perforated_network_llvm`, `test_process_graph_autograd`,
  `test_ssa_c_aggregate_constants`, `test_ssa_glsl_compute_backend`). None from this
  week's work; all unowned.
- Host-facing layout facts: now book rows (`program_abi_field_slot`, the entry's
  API_CONTRACT, `<entry>_layout.h` printed from it; probe
  `tools/compiler_probes/probe_host_layout_rows.py`). Still open: the compiler-side scalar
  copy of the written-slot-wins rule (`_preferred_linked_field_candidates`) is a second
  copy of `post_program_abi_field_slots`'s rule; the contract has no declared column roles
  (scene adapter); a formal's program_abi accounting is not a book cell;
  `NativeSystem.layout()` is still Python-only.
- Shape proofs keyed by authored function name, shared across specialized copies
  (blocks mixed-shape solve in one function).
- ~~Cross-book `_turing_source_cells` stamps on cached AST objects.~~ Already fixed on
  `main` before this audit (stamps are `SourceDefinition(page, row, fact)` identities,
  resolved per book by `resolve_source_definition`; `test_source_definition_cross_book.py`
  is real and green). The lane that re-checked it (2026-10-07) found and fixed a
  different bug instead: the root graph sits in its own function table, so callsite
  tensor specialization visited each root callsite twice per round and double-posted
  `callsite_descriptor_reuse` (VJP test red on its own). One hunk on
  `fix/cross-book-stamps` (129c20fa), unmerged.

## 6. Other known compiler items (from the 2026-10-03 continuation; untouched)

name-arm write-back miscompile (momentum `[0,0,0]` vs `[0.5,1,1.5]`, Woodshop ~17 min
build); `tools/repro_step_with_dt_control_used.py` fails on an `unresolved_report`
REVISE refusal on main and baseline alike; float32 solve returns a float64 buffer
(value correct).

## 7. Orbital game (`engine_toy`, root repo `nogodsnomasters` at 49d2de1)

No commits since 2026-10-05. One uncommitted 78-line addition (another agent's): a
Helmholtz light/colour law section (Planck radiance -> CIE XYZ -> linear RGB) in
`honorary_engine_equation_catalogue.py`. Game items in `ORBITAL_GAME_STATUS_2026-10-03.md`
are unchanged: `--planner collocation` broken, reaction wheels / dt metrics unverified,
planner row cache stale. The craft's 27 physics pieces + 8 Green-coast pieces are the
35 cached pieces the native window compiles.

## 8. Decisions already made (do not re-ask)

- Per-copy shape-proof keying: yes. float32 solve returns float32 (eager parity).
- The tick lives inside a piece; `run_superstep` and `llvm_dt_system.py` stay as they
  are (own lowered source). Dependents of the state-machine shell convert; none exempt.
- Host-facing facts are book rows first; `<entry>_layout.h` is an artifact derived from
  them (API_CONTRACT part); scene adapter waits on declared column roles.
- `_class_surface_ssa_program` (26,115 lines) stays whole; only helpers move.
- Workers: Sonnet, one narrow task each. New code first, small runs before large; never
  re-run a known timing. Memory below ~20 GB is normal. Nothing valid without edges.

## 9. Ranked plan for a week of full service

1. Re-verify and merge the three landed branches (region copy, shell leaves, glsl
   leaves): audit + solve probes + `native_round.py 2` after each merge; push.
   Finish the two in-progress moves or drop their dirty state.
2. Close loose end 3 (dominance probe = 0 violations, clang warnings = 0) if §2a is not
   clean; then N=4, then N=8 with the region-copy fix, then 35.
3. Restart the dead lanes, highest value first: raw_primitive sweep (the rule), per-copy
   shape proofs (unblocks mixed solve and the cross-query descriptor memo), tick rows
   (dt-system state machine), host layout rows, cross-book stamps, f32 dtype,
   closure-build costs, ellipsis stores.
4. Remaining module moves (non-leaf areas of both files), then the giant-function phase
   split with a context object.
5. Pre-existing failing tests (7) bisected and owned.
