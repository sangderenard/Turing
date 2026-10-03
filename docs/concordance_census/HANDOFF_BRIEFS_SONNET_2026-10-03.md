# Handoff briefs for Sonnet workers (drafted 2026-10-03)

The Opus lanes below are still finishing. Each brief is the narrow continuation
that a fresh Sonnet worker takes when its Opus lane reports. Before dispatching,
append that lane's final report under "State at handoff".

Rules for every worker (the user's, absolute or standing):
- Nothing is valid unless it goes through the concordance and leaves edges. A decision
  kept only as an attribute, flag, side table or private key is invalid even when correct.
- Never hand-edit compiler output. Fix the source or the compiler and recompile.
- Never compile with extraction_contract=None or a hand-built contract. If RSS passes
  ~6 GB, kill it by Windows pid and check the contract.
- A compile running ~30 min is a suspected loop: faulthandler dumps every 600-900 s,
  compare, kill and trace. Dumps crashed with an access violation twice; fall back to progress prints.
- One build at a time. Search with `git grep`, not recursive filesystem search.
- Baseline is turing HEAD a7f57c1e (not a2020e0b). The audit baseline findings are
  0,1,0,1,5,0,0; unsourced facts at HEAD: view 4022, toplevel 2831 (ids 6), energy 3800,
  controller 4546, controller_untyped 4431, mapping 197, oscillator 2952.
- Do not commit; the coordinating session commits.

## 1. Name-arm / Woodshop (replaces Opus lane ad585b)
Objective: take the Woodshop whole-program native build (build/woodshop_outer_native_probe.py,
~17 min, real contract) past its current wall.
Decided by the user: (A) join annotation to contract record through book rows; keyed-lookup
result is the row handle (leaves are element pointers into declared row columns, derived on
the book); (B) a declared fixed-shape span leaf is a span value, never a sequence;
no linker-side grows (remove grow_pooled_row_column and the field-demand grow if the
upstream form works).
First proof: `rhmod/rhworld.py` (scratchpad) natively in C and LLVM must give momentum
[0.5, 1.0, 1.5] like CPython (was [0,0,0]).
Files: precompile_to_ssa.py (storage_root, parameter_abi_kind, payload classification),
fortran_c_shell.py (materialize_parameter_record_abi, contract-record selection),
topological_reducer.py (indexed_store_site). Notes: CONTINUATION_name_arm_alias.md.
State at handoff: (fill from the Opus report)

## 2. Edges C continuation (replaces Opus lane a783c7)
Objective, in order: (1) cross-book stamps: `_turing_source_cells` stamps carrying Refs on
cached AST objects must become identity stamps (registry key / module+qualname) that each book
resolves to its own definition cell. Proof: VJP test then test_record_return_site_identity.py
in ONE pytest process, all green (was 6 failed + 8 errors). (2) Store rows from
glsl_deployment_strategy: derive from the stored value's cell and the destination's
storage_root cell. (3) Stragglers: static_constant (3), 3 Name nodes, `*.__present` Inputs.
Notes: CONTINUATION_edges_lane_C.md.
State at handoff: (fill from the Opus report)

## 3. Tracker burn cutoff and re-render (replaces Opus lane a1ed34)
Objective: burns end when the delivered delta-v reaches the plan (request a round that ends at the
predicted cutoff, account for throttle spool-down, RCS for the residual under the main engine's
0.4 minimum throttle); fly(t0->t2) equals fly(t0->t1) then fly(t1->t2) to integration tolerance.
DEFER until the reaction-wheel lane lands: gain retuning and the game re-render, because
wheels change torque sourcing in orbital_actuation.allocate_wrench.
Re-render command (from engine_toy): `python orbital_game.py --click 0 --frames 24 --every 60
--burn-shots --out shots/orbital_game_machine`.
Files: orbital_tracker.py, tests/test_orbital_tracker.py. Notes: CONTINUATION_orbital_step5_tracker.md.
State at handoff: (fill from the Opus report)

## 4. Planner row cache (new, unassigned)
Objective: rebuild %TEMP%\orbital_collocation_rows on current HEAD once (~15 min slice compile)
and make the cache record and check PieceCompilerRecord (native_law_kernels.py) so a compiler
change invalidates it. Verify the slice rows entry by entry against sympy if feasible; they were
verified only through flights before. Files: engine_toy/orbital_collocation.py only.
