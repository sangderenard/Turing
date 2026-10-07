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

So loose end 3 is open: these 10 must be fixed at their writers (which pass leaves the
stale initial / places the Load's definition on one arm) and the probe must report 0
before N=4. The 69-column parity stands, but it is parity on a module with 10 latent
undefined reads.

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

Both in-progress worktrees import cleanly today (`import src.compiler.glsl_deployment_strategy,
src.compiler.fortran_c_shell` ok in wtv and wtc), so the uncommitted moves are not broken,
only unfinished. The three detached baseline worktrees (`wtb2`, `wtmb`, `wtv2`, clean at
c0781c7a) were removed today. Disk: 57 GB free.

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
  `NativeSystem.state_field_ids` spans).
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
