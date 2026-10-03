# Session continuation report, 2026-10-03 (orbital demo push) — FINAL, all activity stopped

Everything below is the state at shutdown. All agents were stopped, all compile
processes were killed, and nothing is running. Unverified work is on the branch
`wip/2026-10-03-inflight` in BOTH repos (turing and the root repo); `main` and
`nogodsnomasters` hold only verified work plus these notes. Each lane keeps its own
`CONTINUATION_*.md` in this folder; three were condensed (jacobian, step4, item3).
Older history: `git show a7f57c1e:<path>`.

Standing rules (user): see "Rules" at the end. The first is absolute.

## 1. Landed and verified (on main)

### turing (HEAD at shutdown: a7f57c1e + notes commit)
- **Compiled collocation Jacobian = SymPy to 4.138e-16** (7x19, 55 nnz): 095a3c0c, 0519095d,
  ccba325c, 51b4cebe, ce587b82. Root causes: plan_callsites skipped adjoint calls with no AST;
  eps marker; argument roles; rank-0 formal shapes never published (a spin); one identity book
  per reverse compile; int32 shape vectors widened to int64 (NaN). Re-verified on the whole tree
  at integration: lowering 47 s (was 235-340 s).
- **Orbital item 3 + 4** (c03ae4e6, a7f57c1e): undefined sympy functions compile as runtime-slot
  externals in C and LLVM; derivatives of externals resolve to declared derivatives (loud refusal
  otherwise); Integral lowering; `constant_role` rows. The WHOLE orbital set compiles as written
  and matches the reference: tests/test_orbital_transfer_compile.py 12 passed / 0 xfailed
  (errors 0, 2.2e-16, 1.4e-16, 1.8e-15). Bound to a live OrbitalJumper it reproduces the craft
  physics (tools/compiler_probes/probe_orbital_craft_binding.py, ~3 min).
- **Uploaded patches:** 854e145c (NumPy scatter precision); 408155a7 (return-site identity, ported
  onto our mechanisms; native probe_record_return_merge now matches CPython).
- **Batch-4 Piecewise miscompile** (822a753b): elementwise Select in the AbstractTensor stage is
  `where()`, not one if/else for the whole column. Multi-element branch conditions are now a named
  shortfall in C and LLVM.
- **Phasing regression** (a7f57c1e): `Mod`/`FloorDiv` are integer only on integer evidence (an
  unconditional seed since 4f14a372c spread int64 into float sign-selects; wait 0 -> 4995.3 s).
- **Concordance edges:** argument roles, AbstractTensor forward scopes, backward_rule_definition
  roots (rule-graph spans 3,626 rootless -> 0), integer widths, repository-kernel roots,
  value_shape_polymorphism rows, return versions Unresolved without a link, reducer rows
  (398 -> 16 in the rule graph, edges lane C), constant_role, `_set_operands` for sympy.
  Audit findings unchanged 0,1,0,1,5,0,0; unsourced facts at HEAD: view 4022, toplevel 2831
  (6 ids), energy 3800, controller 4546, controller_untyped 4431, mapping 197, oscillator 2952.
- **dt compile stall** (in a7f57c1e): spinning, not slow. `record_shape_transformation` kept one
  shape state per value naming its edge, so lhs/rhs edges overwrote each other (one Pow revised
  4,239 times) and `IdentityPage.history` scanned every column. One state per value, indexed
  per-row history. Bisect: first bad commit 9cfd81a4.
- **llvm_dt_system** (a2020e0b): per-state program binding (the "every state runs the last state's
  program" bug), per-state owned pieces, in-place outputs, one contiguous eager span; parity
  bit-identical over 7,428 field-rounds; station frame 1.55 s -> 1.0 s.
- Deploy preset `-O3 -march=native` (fd68458d).

### root repo (engine_toy), branch nogodsnomasters
- Craft as an engine_toy machine (6b20ed8): geometry from the production graph, 4 kN gimballed
  main, 2 brake, 16 RCS, mass properties from the machine reduction (dry 249.9 / wet 945.9 kg).
- Role allocator (c36e6a2): main XOR brake, gimbal trims first, RCS couples only; 5-15 ms/call.
- Jumper (3d6d74c): synchronized r(), mean-step kick (orbit radius error 2093 m -> 102 m),
  BIND dt contract, propellant-mass pricing. Steps 1-3, 7 (Isp mass loss, fuel cutoff law,
  rotation-matrix attitude: 861ccb4).
- Tracker (7fae275, 348cb5d): wrench seam, off-plan hysteresis, full-trip re-plan, honest deadbands,
  machine-craft routing. Machine transfer 7000->8000 km: 0.91 m / 0.034 m/s, biprop 0.997x ideal.
- Planner (77f4496, 3afe283): collocation converges (1.0000x Hohmann fuel, 0.6 s); live re-plan
  after a 300 m/s kick 1.2-2.0 s (was 205 s), 409-536 m/s vs Hohmann 690-918 m/s.
- Game (2f3666d, 607a3b0): click -> phased transfer, machine craft, tilting plume, batched
  stations, 1.65 s/frame. Cache lock + compiler provenance in equation_piece (95e1e8d).

## 2. In flight: UNVERIFIED, on branch wip/2026-10-03-inflight (both repos)
Stopped mid-edit. Treat every item as unverified and re-run its gates before merging.

| Lane (Opus, stopped) | What the edits do | Last known state |
|---|---|---|
| Name-arm / Woodshop | `record_field_incoming_slot`; storage_root, parameter_abi_kind, parameter_record_class, indexed_store_site pages; annotation join; row-handle materialization (was mid-edit) | Cause 1 of the write-back miscompile fixed (written-never-read field lost its Store). Cause 2 OPEN: the caller passes the callee a slot minted from nothing, so natively momentum is [0,0,0] where CPython gives [0.5,1,1.5]. Woodshop build stopped at callsite 84 -> then `center_xyz` / `orientation_deg_xyz` (fixed-shape span leaf lowered as a sequence). |
| dt full-native | control_source.py guard placement (`overlay_scheduled_control(lexical_encloses=)`, `nested_root`) | Variants b, d, e of the repro now lower with the merge Phi and no invented formal. Open: `unresolved_report` call-result record arena has no ABI columns. Previous run: deployment +380 s, SSA lowering +792 s, 5.7 GB. |
| Edges C | reducer/graph_express2/node_special_cases row work | Done and gated (26 passed, orbital 12). Found, NOT fixed: `_turing_source_cells` stamps (Refs) persist on cached AST objects across books -> `source cell does not exist: Ref('backward_rule_definition', ('BACKWARD_RULES','clamp'), 0)`; VJP test then return-site tests in one process = 6 failed + 8 errors. |
| Symbolic cache + external rows | sympy_dual_ir_cache.py (item 1 DONE, gated); aot_checkpoint temp-name fix and native_law_kernels cache record (follow-ups, unverified); fortran_c_shell opt-in for one book | Item 2 approved by the user (lowering resumes the caller's ambient book) but not verified. 234 external-leaf unsourced ids are the measure. |
| Tracker burn cutoff | orbital_tracker.py cutoff timing | No gravity: a cut costs 1.3 mm / 2.7e-4 m/s (integrator fine). Gravity case, frame-boundary-independence test and re-render NOT done. |
| Wheels + dt metrics | 3 reaction wheels as rotors, wheel dynamics, allocation, dt bounds for slews/spool/propellant draw in orbital_actuation / craft_machine / jumper (+ propellant_supply fix by a Sonnet worker, test passed in isolation) | No numbers reported. The dt-metrics audit table was not delivered. |

## 3. Open decisions for the user
1. llvm_dt_system per-participant publication fields (9 single-element stores per law): option 1 whole-field
   store per field, or option 2 a declared publication array (needs compiler support for shared storage).
2. `get_tensor` lowers through the operator catalogue alias, not a backend table entry (recommended).
3. Patch-port gap: `static_branch` variant (pruned field is not a record output) is a strict xfail;
   fixing it needs the patch's "project every declared field at entry".
4. Planner cost weights: keep the current scale-free defaults (recommended), or re-pick.
5. Invented defaults to review: Isp kinds (70/230/310 s), craft masses/dimensions, throttle floor 0.4,
   deadband 1e-6 r / 1e-5 v, 0.125 rad attitude step, 0.05/0.2 rad burn light/hold gates.
6. Merge order for the WIP branch (see 6).

## 4. Known failures and traps
- Fail identically at clean HEAD: test_precompile_to_ssa 13 failed / 91 passed; test_ir_sequence_tables 3;
  test_process_graph_autograd 4 failed / 20 passed when measured early in the day (not re-measured; the
  linear-motion test in it passes now);
  test_llvm_dt_system_python_lane 12 (stand-in Piece lacks `instantiate`); test_native_record_read_order
  `first_write`; test_public_abstract_tensor_api_is_explicitly_classified;
  test_child_record_conditional_write_reaches_return (ArgumentBindingFact pickling).
- Native dt-system lowering is NOT yet full-native (see dt full-native row). It is a big compile:
  ~6-7 GB with its real contract. Never `extraction_contract=None`.
- Planner Jacobian row cache `%TEMP%\orbital_collocation_rows` is keyed on laws only and was built
  10:42-10:45, before ~7 compiler fixes. Rebuild once on HEAD (~15 min); make it check PieceCompilerRecord.
- Piece cache now rebuilds stale pieces (seconds-minutes) whenever any loaded src.* module changes; set
  ENGINE_TOY_PIECE_SERVE_STALE=1 to serve stale pieces with a logged row. Other on-disk caches without
  compiler identity: native_law_kernels.py:610, opportunistic_pipeline.py, project_compilation_product.py,
  vehicle_validator_simulation.py, build_math_cache.py (partial: kernel_bank, host_code_modules,
  symbolic_fluid_direct_control, perforated_network_llvm, aot_compile).
- Game screenshots in engine_toy/shots/orbital_game_machine_*.png were rendered with the broken phasing
  piece; re-render after wheels land.
- Faulthandler dumps crashed twice with a Windows access violation during compiles; use progress prints.
- The audit needs system Python 3.11 (turing/.venv has no yaml). Free RAM was down to 5 GB with 6 agents.
- Hand-edits: never. Fix the source or the compiler and recompile.

## 5. Next steps, in priority order
1. **Edge-less decisions still open** (the user's absolute rule): the integer-index inference in
   precompile_to_ssa (`region_value_meta`) never posts a row; remaining reducer Constant/Indexed/Name rows;
   Store rows (30-85 per audit case); `*.__present` Inputs; `static_constant`. Edges C owns the first three.
2. Finish the name-arm write-back (row handle) and re-run the Woodshop build; then dt full-native.
3. Fix the cross-book stamp bug (edges C) before anything else touches the VJP/return-site tests.
4. Verify and merge the symbolic one-book opt-in; then rebuild the planner row cache.
5. Wheels + dt metrics, then tracker retune and game re-render.
6. Sonnet handoff briefs: HANDOFF_BRIEFS_SONNET_2026-10-03.md (name-arm, edges C, tracker, row cache).

## 6. How to resume
```
cd C:\dev\Powershell\turing && git fetch && git switch wip/2026-10-03-inflight   # unverified edits
cd C:\dev\Powershell      && git switch wip/2026-10-03-inflight                  # engine_toy edits
```
Verify from the turing repo root before merging each lane (run one at a time):
`probe_collocation_jacobian.py jacobian --compile` (4.138e-16), tests/test_native_scalar_loss_adjoint.py,
test_llvm_training_runtime.py::test_graph_reverse_is_a_compiled_parametric_vjp,
test_process_graph_autograd.py::test_linear_forward_loss_backward_is_one_parametric_graph_motion,
test_ssa_record_return_state.py, test_record_return_site_identity.py, test_orbital_transfer_compile.py (12),
and `python tools/audit_identity_concordance.py` (findings 0,1,0,1,5,0,0; unsourced at or below the numbers above).
Probe needs `PYTHONPATH=C:\dev\Powershell\engine_toy`.

## Rules (user)
- ABSOLUTE: "nothing is valid in any way that doesn't go through the concordance leaving edges." A decision kept as an
  attribute, flag, side table or private key is invalid even when correct.
- A compile running ~30 min is a suspected loop: dump stacks, compare, kill by Windows pid and trace.
- Never compile with extraction_contract=None or a hand-built contract; kill any run past ~6 GB and check.
- One build at a time. Search with `git grep`. Show output inline. Never run the interpreter lane.
- New workers are Sonnet with narrow tasks; let Opus workers finish, then hand follow-ups to Sonnet.
