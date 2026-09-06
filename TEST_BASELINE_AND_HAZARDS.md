# Test baseline and hazards — read before running any test

2026-09-05: latest whole-validator lowering failed at Pygame draw loop effects
after611.818s, session91926 terminal. Subsequent full-native extension ABI admission
tests and existing gates:5 passed2.82s. Pygame->Abstract UI AST ingestion:5 passed
2.66s; transformation only, not geometry backend or native execution proof.

2026-09-05: authored support-stage placement/binding, component placement/law,
and vehicle state feedback: 5 passed in3.01s. First attempt caught an editing
indentation error in the constructor, corrected before the passing run.
The support-stage check executes the actual AST block with19 supports/2 lanes;
it does not initialize tires, run the whole validator, or execute native code.
Corrected full dually allocation/ABI test completed successfully in106.92s;
session50509 is terminal, superseding the active-run note below.

2026-09-05: `test_dually_program_allocates_and_declares_requested_rig_point_count`
requires expensive full dually preparation: first run 575.12 s, failing on a
test-only ABI accessor typo after buffer assertions passed. Correct accessor is
`binding.field.shape`. Corrected rerun active (see continuation); do not treat
this as a seconds-long test or rerun for cosmetic verification.

2026-09-05: variable rig-point law regression failed before generalizing the
hard-coded 16-point reshape (3.54 s), passed after (2.95 s). Existing declared
rig SSA/C/library compilation test passed with 19 points in 9.99 s; no native
execution in that test. Eager checks 0/1/3/19 points and balanced reactions.

2026-09-05: common player/validator placement action and stale revision regression
passed 0.60 s; existing IssuedAction table count check passed 0.52 s.

2026-09-05: general four-port attachment placement regression passed 0.58 s.
Graph edits and routing validation only; not mechanical constraint execution.

2026-09-05: authored dually vehicle feedback regression reproduced missing state
carry (0.69 s), then passed after fix (0.59 s). Expanded actual mapping/rollback/
replay check passed 0.60 s. Deterministic test plant, not native vehicle solver.

2026-09-05: fixture mechanics test passed in 1.89 s after correcting pillar
reaction to the actual scalar-y output shape. Previous vector fixture was not
representative of the numerical graph output.

2026-09-05: identified fixture command/reaction/pose test and authored viewer
acknowledgment regression passed together in 2.17 s. No native physics execution.

2026-09-05: selected fixture/graph geometry regression passed in 0.57 s after
converting graph edges to identified AbstractUI objects with visibility state.

2026-09-05: three selected vehicle geometry checks passed in 0.70 s:
Abstract UI object/child identities and graph relationships across poses,
fixture/graph geometry, and existing seven-primitive geometry after relocation.

2026-09-05: `tests/test_vehicle_part_geometry.py::test_tire_geometry_preserves_winding_material_depth_and_center_surface`
passed in 0.57 s. Numerical authored geometry only; no native/pixel parity.

2026-09-05: `tests/test_vehicle_part_geometry.py::test_authored_part_geometry_preserves_shape_width_and_world_coordinates`
passed in 0.54 s. Covers seven authored shape primitives and skipped membrane
records; no GPU rendering or compilation involved.

2026-09-05: two focused `tests/test_geometry_consumer.py` checks passed in
0.91 s (frame completion/failure/camera/empty layers and persistent mesh buffers).
Mesh check passed again in 0.84 s after adding explicit color-presence handling.
GL calls mocked; actual shader output remains unverified.

2026-09-05: geometry packet ownership/invalid-native-span regression in
`tests/test_geometry_display.py` passed in 0.52 s. Geometry capability opt-in
and rejection by pixel-only shell test plus existing shared-mailbox ABI test
passed together in 1.91 s. These exercise codec/contracts, not rendering.

2026-09-05: `tests/test_vehicle_viewer_frame_ack.py::test_viewer_does_not_acknowledge_revision_published_during_draw`
reproduced premature revision acknowledgment (0.70 s) and passed after fixing
the authored loop (0.57 s). Executes the production loop AST with controlled
publication during drawing; excludes model startup and native compilation.

2026-09-05: `tests/test_glrenderer_host.py::test_external_context_renders_without_pygame_and_propagates_present_failure`
passed in 1.19 s. Mocked GL, denied Pygame import: external context host,
overlay ordering, resize dimensions, core-context enables and presentation
failure propagation. Does not verify pixels or native validator parity.

This document exists because the same expensive mistake kept being repeated:
an agent sees a failing test, assumes it broke it, and spends two full suite
runs plus a stash/pop to find out the failure was already there. That costs
hours and has damaged working state in this tree. **The baseline below is the
answer to "is this mine?" — consult it before running anything.**

## The rule

1. **A failure in the manifest below is NOT yours.** Do not re-derive it. Do
   not run a second time to confirm. Say "pre-existing, see
   TEST_BASELINE_AND_HAZARDS.md" and move on.
2. **Never baseline with `git stash` or `git checkout -- <path>`.** Both have
   destroyed uncommitted work in this tree. If you genuinely must compare
   against a commit, use a worktree, which touches nothing you have:
   `git worktree add /c/dev/turing-head <commit>` — and remove it when done
   with `git worktree remove /c/dev/turing-head`.
3. **Never run a test purely to collect names for a report.** If you already
   have a pass/fail count or a progress string that answers your question,
   that IS the measurement. Re-running for cosmetics is not free here.
4. **Update this file** when you legitimately run something and learn a name
   or a count the manifest is missing. Filling a gap as a side effect of work
   you were doing anyway is right; running to fill it is not.

## Hazards, measured

| hazard | detail |
|---|---|
| `pytest tests` (whole suite) | **Does not finish.** Observed 2026-08-19: 5 h wall, **zero bytes** of output, then decayed to ~3 CPU-seconds per 5 minutes and had to be killed. Do not run it. |
| Output is buffered | pytest writes nothing until it exits when its stdout is a pipe. `-u` / `PYTHONUNBUFFERED` do not help — pytest buffers its own progress. A silent run is not a hung run, and you cannot tell them apart by watching. |
| `pytest-timeout` is **not installed** | There is no per-test timeout. One hanging test eats the whole run. Bound it from outside instead: `timeout 100 python -m pytest tests/<one_file>.py -q --tb=no`, one file at a time. Exit code 124 means it hung. |
| Killing a run | Several tests spawn native toolchains (gfortran, zig cc). Killing the pytest process can leave children; check for stray `python`/`zig` processes afterward. |
| Cached artifacts | A stale `control_repository_ssa.pkl` presents a program built by a compiler that no longer exists, so your change appears to do nothing. `_cache_is_stale` guards the fluid path; other paths may not. |

## The manifest — known-failing at `af00599`

Verified 2026-08-19 by running each file in the working tree AND in a clean
`git worktree` at `af00599`. Identical results in both, so these are
pre-existing and independent of the namespace/indexing fixes on top.

| file | result | notes |
|---|---|---|
| `tests/test_ast_indexing_aot.py` | **13 failed, 10 passed** | progress string `FF.F.FFFF.F.F.....F.FFF`, byte-identical in both trees. Individual test names NOT captured — collecting them costs a full run of an expensive file, which rule 3 forbids doing for its own sake. If you run this file for your own reasons, paste the `-rf` names here. |
| `tests/test_index_set_scatter.py` | **2 failed, 7 passed** | `test_index_set_emits_a_complete_scatter_module`, `test_index_set_scatter_runs_correctly` |
| `tests/test_extraction_contract.py` | **2 failed, 14 passed** | `test_default_contract_draws_python_native_and_decompile_lines`, `test_print_host_boundary_uses_existing_stream_operator` — both assert `print` decides `python_host_call`; it now decides `use_native`. Verified 2026-08-20 in a clean worktree at `6cb148b`: identical there, so pre-existing. |
| `tests/test_process_graph_autograd.py` | **1 failed, 21 passed** | `test_real_abstract_nn_xor_has_exact_native_adjoint_and_training_loop` — `NotImplementedError: SSATensorOperations must implement tolist_()` (`abstraction.py:752`). The three former aggregate-linking xfails were fixed and promoted to passing regression tests on 2026-08-20; two multi-output/contract tests were added. The remaining failure was verified 2026-08-19 in a clean worktree at `2ee2fd1`. |
| `tests/test_process_graph_function_linking.py` | **7 failed, 17 passed** | `test_record_field_assignment_is_a_real_inout_value`, `test_direct_source_lowers_declared_record_literal_and_bool_return`, `test_record_return_call_refreshes_completed_physical_field_surface`, `test_record_parameter_call_uses_fields_without_python_receiver_handle`, both comprehension publication tests, and `test_returned_record_fields_feed_structural_call_argument`. Verified 2026-08-24 with identical results in the working tree and a clean worktree at `57b5e25`; independent of class-emitter receiver wiring. |
| `tests/test_site_bundle.py` (focused bundle pair) | **2 failed** | `test_program_bundle_owns_page_source_wasm_manifest_and_inventory`, `test_one_shot_bundle_packages_the_discovery_numeric_trace` — both reach WASM fidelity and fail with `ValueError: a compiled program needs at least one output`. Verified 2026-08-20 in a clean worktree at `735409d`; unrelated to the shader-region deployment stage. |
| `tests/test_webgpu_ssa_backend.py` (deprecated-AOT quartet) | **4 failed** | `test_ast_generated_float32_program_emits_wgsl_compute`, both `test_ast_generated_loop_uses_structured_wgsl` cases, and `test_float64_is_a_named_webgpu_core_shortfall`. The deprecated `compile_ast_aot` path no longer presents the captured numerical program/cycle shape these tests expect. The first was verified 2026-08-20 in a clean worktree at `ab9a078`; the other three were exposed by a focused run the same day and do not touch direct repository-SSA GEMM/WebGPU emission. |
| `tests/test_machine_target_languages.py` (focused pair) | **2 failed** | `test_existing_backend_operator_lists_are_exposed_without_a_fifth_copy` hard-codes 40 C operators but the shared inventory now has 50; `test_fortran_and_desktop_glsl_print_the_same_numeric_program` expects `cos(` in a Fortran artifact that no longer retains it. Verified 2026-08-20 in a clean worktree at `ab9a078`; unrelated to WebGPU benchmark emission. |
| `tests/test_precompile_to_ssa.py` (current tree, 2026-08-30) | **2 failed, 64 passed** | `test_native_fortran_ops_keep_mean_and_span_fill_in_ssa` now sees the span fill lowered to a `Call`; `test_index_dtype_propagation_is_scoped_per_function_identity` sees a `float64` load instead of `int64`. Discovered during the vehicle call-shape propagation regression gate; neither exercises call metadata propagation, and no clean-worktree ancestry comparison was commissioned. The older 2026-08-23 repeat/API-publication failure no longer appears in this run. |
| `tests/test_wasm_class_modules.py` (current tree, 2026-08-24) | **1 failed, 27 passed** | `test_describe_process_graph_api_resolves_the_real_source_parameter_name` sees no logical input from the deprecated `compile_ast_aot` path. Discovered while testing state-feedback inventory wiring; the new focused regression passes and does not exercise that compilation path. No clean-worktree ancestry comparison was commissioned. |
| `tests/test_wasm_class_coordinator.py` (current tree, 2026-08-24) | **1 failed, 10 passed** | `test_wasm_coordinator_calls_cards_internally_and_honors_latched_ranges` broadcasts the first element (`seam=[4,4,4]`, `result=[16,16,16]`) instead of preserving the three-element input. The passing count includes the new control-region state-naming regression. No clean-worktree ancestry comparison was commissioned. |
| `tests/test_symbolic_equation_compiler.py` (current tree, 2026-08-25) | **2 failed, 4 passed** | `test_compiled_scalar_equation_can_run_in_fortran` expects integer literal `2` but the backend emits `2.0_c_double`; `test_fluid_equation_compiler_builds_the_full_default_model` expects 30 inputs but receives 28. Discovered while adding the independent float32 SSA-to-WebGPU vehicle contact path. No clean-worktree ancestry comparison was commissioned. |

| `tests/test_ssa_c_aggregate_constants.py` (current tree, 2026-09-01) | **6 failed, 29 passed** | The six failures (`const`-qualified aggregate constants, invocation-local `frame_N[...]` storage instead of malloc'd root storage, imported-LLVM literal decode, planned-aggregate caller-storage binding, physical-dtype preservation across planned outputs, static extent shape emission) are the prior session's SPEC for the still-unfinished native activation-storage contract. They are the same defect family as the managed-window emission shortfalls (`dynamic temporary %t96 has no native activation-storage contract in ...balloon_tire_managed_window__planned_region_2`) that currently block re-emitting `balloon_tire_appendage_step.c` at ANY window rate (verified identical at 1/120 and 1/1024). Not regressions from the 2026-09-01 deployment-outlining work: the failing arms (Const, root storage) were not touched, and the identical shortfall set reproduces on the un-outlined module. |
| `tests/test_deployment_native_emission.py` (current tree, 2026-09-01) | **1 failed, rest passed** | `test_frame_plan_workers_and_chunk_are_literal_in_native_source` dies inside `shader_region_pipeline` ("typed shader hole lacks shape/dtype metadata for boundary values") while constructing its plan — a dirty-tree interaction in plan construction, not in pooled C rendering, whose compile-and-run proofs in the same file pass. |

| `tests/test_compiled_linalg.py` (current tree, 2026-09-01) | **`test_jacobi_rotation_arithmetic_computes_natively` fails numerically; a later test in the same file hard-crashes the process** | The test runs the LLVM lane (`compile_artifact`), which the 2026-09-01 deployment/extent work does not touch (its only LLVM edit is an IndexError->shortfall guard). The signature — a compiled rotation kernel with partially-wrong elements — matches the long-open "native re-reads a load across an in-place store" aliasing defect already pinned for rotation/swap kernels. The 2026-08-19 "6 passed, 1 xfailed" row above predates substantial dirty-tree movement. No clean-worktree ancestry comparison was commissioned. |
| `tests/test_ir_sequence_tables.py` (current tree, 2026-09-01) | **1 failed, 35 passed** | `test_compiled_retained_loop_mutates_caller_sequence_record` asserts `artifact.c_source_path.read_text() == ""` — an in-progress/spec expectation (source file consumed?) from commit 839a40d. Unrelated to deployment outlining (which never runs without an explicit pass call or `deployment=auto` contract). |

## The manifest — known-good at `af00599` plus the current working tree

2026-09-05 focused threading/loop checks: `test_literal_seeded_counter_executes_native_iterations`
passed in 9.17 s after repairing carried scalar region capture and C singleton
publication. Native execution is isolated in a subprocess with a 20-second timeout;
earlier versions hung, so keep that bound. `test_resource_wait_retains_loop_scope_and_unique_ssa_definitions`
passed in 2.71 s with current-phi and nonidentity increment assertions. These are
bounded regression proofs, not full-validator native parity.

2026-09-05 native Event addition: `test_native_event_broadcast_clear_and_future_waits`
passed in 2.95 s (Windows, subprocess timeout=20). The selected-loop diagnostic
regression in `test_process_graph_call_diagnostic.py` passed in 0.55 s.

2026-09-05 source pursuit fixes: focused registered-method, constructor and
module-field receipt regressions in `test_ast_parent_ingestion.py` passed
together (3 passed, 2.55 s). The existing reachable-root exclusion regression
passed in 2.12 s after repairing lexical-body activation; nested-helper plus
parameter-shadowing coverage passed in 2.27 s. No full-suite run.

2026-09-05 compiler-ledger coverage: focused fingerprint regression passed in
2.34 s after adding shared AST context, deployment, C emission and dispatcher
runtime sources. Saved-call extraction-parameter diagnostic passed in 0.55 s.

These passed on 2026-08-19 and are the cheap, high-signal set. Prefer them.

| check | result | cost |
|---|---|---|
| `tools/translation_scorecard.py` | 18/19 journeys equivalent; level 18 stops at materialization | ~6 s |
| `tests/test_precompile_to_ssa.py` | Historical baseline: 34 passed; current-tree result is recorded above | ~7 s |
| `tests/test_symbolic_fluid_native_runtime.py` | 1 passed | ~17 s |
| `tests/test_abstract_tensor_indexing.py` | 2 passed | ~1 s |
| `tests/test_ssa_fusion_regions.py` | 1 passed | ~3 s |
| `tests/test_region_kernel_dedup.py` | 2 passed | ~3 s |
| `tests/test_compiled_linalg.py` | 6 passed, 1 xfailed (strict) | ~9 s |
| `tests/test_ir_sequence_tables.py` | 23 passed | ~3 s |
| Full-native vehicle contract/link helpers (focused, 2026-08-30) | 3 passed: contract boundary rejection, ordered-view rebinding, allocator namespace filtering | ~4 s |
| Canonical balloon full-native lowering + C emission (batch 8, 2026-08-30) | 52 linked functions; gate complete; no unresolved/unmaterialized/non-native boundary; no id-scale finding; C complete with 0 shortfalls | ~48 s |
| `tests/test_aggregate_call_identity.py` (new, 2026-09-02) | **5 passed** (~8 s) | Synthetic ~1 min reproductions of the dually call-boundary rules: tuple-of-span-parameters stays dataflow (was folded to a Constant of ProgramABI facts -> `aggregate call binding for 'tire_history' ... 4 != 0`), member formals bound by exact index, nested-tuple results/returns correlated by structural path, unpacked multi-result call carried inside a retained loop (the "carried update value 251 has no producer" shape), non-recycled value ids, identity-keyed dependency-order cache. Run these before any ~8 min full dually lowering. |
| Planner/linking/loop batch on the 2026-09-02 aggregate-identity tree (`test_hierarchical_plan`, `test_loop_composer`, `test_vehicle_python_graph_source`, `test_process_graph_function_linking`, `test_loop_carried_producers`, `test_state_loop_deployment`, `test_precompile_to_ssa`) | **31 failed, 221 passed** (4.5 min) | All 31 reproduce identically on a pre-patch copy of `glsl_deployment_strategy.py` (verified by swap for the 11 `test_loop_carried_producers` failures -- `loopresult value-source identity conflicts with its semantic edge`, the dirty-tree `value_source_id` port work -- and for the 4 callsite-descriptor/nested-loop tests; the remaining 16 are the unpack-arity, `NameError: plans`, source-text and record/loop identity failures whose messages do not touch the patched paths). Not regressions from the aggregate-identity work. |
| Full dually repository-SSA lowering (`lower_vehicle_python_graph_ssa(inputs=dually_vehicle_python_compilation_inputs())`, 2026-09-02, ~7 min) | **plan + control deployment complete; SSA emission stops on 2 shortfalls** | The `tire_history` arity error and the `vehicle_tire_recurrence` loop-carried `251` shortfall are gone. Remaining: `balloon_tire_vector_step__specialized...__planned_region_12::where` operands `((8,4,1,1,1),(8,1,1,1,2),(8,4,800,2,3))` and `planned_region_13::add` `((8,4,1,2),(8,4,1,3))` -- shape specialization inside the balloon step under the dually profile (present in every run today, independent of the identity fixes). Known pre-existing ABI leak still visible in synthetic programs: loop variable `step` and loop-body scalars appear as `linked_call_frame_storage` root arguments (same in a fully positional program). |
| Full dually repository-SSA lowering with strict `tensor_data_descriptors` (2026-09-02, later, ~7 min) | **plan + control + region SSA complete; frame linking stops** | The two balloon `where`/`add` shape shortfalls above are GONE. Linking now stops in `_prune_unused_callee_formals`-style cleanup: `call has fewer operands than the callee signature while pruning 'vehicle_tire_recurrence__specialized...': operands=85 formals=501`. Probe (`scratch: probe_recurrence_formals.py`): of the 501 formals, 402 are `linked_call_frame_storage` slots propagated outward from the `balloon_tire_vector_step` callsite (463 of 501 are rank-0 float64; 13 carry `unbound_variant_source_id`/`variant_column=row`), i.e. the balloon step's own frame scalars become the recurrence's public formals and the tick->recurrence call was linked before that growth. This is the activation-storage contract family (`dynamic temporary ... has no native activation-storage contract`), not the aggregate-identity family: `tire_history` members `(8,4,3)`, `(8,4,3,3)` arrive as proper formals. Next identity source to fix: why the balloon step's loop-body scalars are frame storage instead of locals. |
| `tests/test_deployment_outlining.py` (2026-09-01) | 6 passed — includes a COMPILED pooled loop linked against `turing_pool.c` matching serial numerics bit-for-bit, the invariant-append guard, and the ordered-join refusal | ~7 s |
| `tests/test_repository_ssa_dispatch.py` (2026-09-01) | 3 passed — planner treats outlined single-lane iteration regions as launchable; un-outlined ones name the pass | ~3 s |
| Managed vehicle deployment region outline (2026-09-01) | `step_with_dt_control_used...` region 0 outlines (19 live-ins, guarded append block), plans launchable+parallel, and emits `turing_pool_deploy_span` + effect locks in module C | ~4 min (lowering dominates) |
| Managed vehicle COMPLETE module emission (2026-09-01, later) | `complete=True` after three `ssa_c_backend` repairs: requirement-backed dynamic temporaries, the `resolve_span_origin` extent walk (output edges, `ssa_call_result_from` hops, cast_like/broadcast shape transfer, binary broadcast-combine, local-def-before-edge), and flat `max/min/all/any` scan spellings. Unresolvable extent origins refuse again instead of registering unfillable slots. The six `test_ssa_c_aggregate_constants.py` spec failures above remain as the storage-contract *style* spec (const qualifiers, local frame arrays) — the emission-blocking subset is fixed. O3+pool compile of the monolithic TU takes ~27 min. | ~25 s emit |
| Compute-shader lane selection (2026-09-01) | `deployment_compute_selection.select_compute_lanes` — vehicle lane refuses with 4 named reasons; synthetic straight-line lane judges eligible (`tests/test_deployment_outlining.py`, 7 passed) | ~6 s |

**That table is the recommended regression gate for compiler changes.** It is
~40 seconds total and it caught nothing false in this session. Reach past it
only when your change plausibly touches something it does not cover, and then
reach for single files with an external `timeout`, never the whole tree.

## Marking expected failures in code

The manifest is the cheap fix. The better fix is `@pytest.mark.xfail(reason=
"pre-existing at af00599, see TEST_BASELINE_AND_HAZARDS.md", strict=False)` on
the known-bad tests, so a green run means green and nobody has to cross-check
a document. That needs the 13 names from `test_ast_indexing_aot.py`, which
per rule 3 should be collected the next time someone runs that file for a
real reason — not by a run commissioned for this purpose.

`strict=False` matters: these should announce themselves as XPASS when
somebody finally fixes them, rather than failing the suite for being fixed.

2026-09-05 DT control fixes: targeted native None/callable(None)/variadic Boolean
checks3passed29.70s; independent-flag sequential branch native check1passed11.98s;
existing equal-region nesting/shared predicate/nested conditional overlay3passed
2.72s. Native subprocesses20s timeout. Actual DT energy-sidechain C/DLL parity
passed for3exact dyadic dt values with optional limits disabled and emptychannels.
Broader call with conditionally rebound input remains broken; see continuation.
Full managed build session91655 active13:53:35, do not overlap heavy operations.

2026-09-05 terminal correction: managed session91655 exited1 at C emission.
No live heavy build. Current saved SSA/C/shortfalls in
build/managed_dt_control_fixes_20260905. Four native call-predicate regressions
passed30.22s after catching/fixing zero versus None equality (session15880terminal).

2026-09-05 native input receipts: 2receipt+4nativepredicate checks6passed30.88s.
Existing loop-carried record-return plus2receipt checks3passed2.89s afterlate
record-merge refresh. Known baseline test_returned_record_fields_feed_structural_call_argument
failed asalreadydocumented above; no testchanged. ActualDT proposal helper C
emissionclean1.08s. Fullmanagedsession83504 TERMINALfailed receiptconsistency;
fixedunupdatedIDs inkeyed-storage removal. Newfullbuildsession78547 PID5848 live
14:26:24; do notoverlapheavywork; latestcontinuationhascommand/log.

2026-09-05 14:37: three tests/test_native_call_input_receipts.py tests passed in2.17s, including tensor-output identity without field-name metadata. Prior full managed session78547 terminalexit1, three emission shortfalls. New full managed session90068/PID16520 active; see continuation before launching another build. Plain conditional append then bool(list) reproduces region0 arity0/1; no native execution performed.

2026-09-05 14:49: tests/test_native_sequence_truth.py plus three native call
receipt tests and two existing query scheduling tests passed6 in11.33s.
New test first failed emission0/1, then failed native assertion due private
capacity0; both corrected. Native test subprocess timeout20s. Session90068
terminalexit1 (only bool(reasons) region arity failure); replacement full build
session41778/PID19476 active, managed_dt_sequence_truth_20260905. See continuation.

2026-09-05: full managed session41778 TERMINAL exit1, now zero C emission
shortfalls. Stops before native compilation on unnamed public47/57. No live
build. tools/managed_dt_parity.py syntax checked only. Minimal completed Boolean
passed to float becomes extra public input12; direct return of the same Boolean
has correct produced Select12. See continuation for exact source and next seam.

2026-09-05 15:07: native Boolean intermediate regression passed12.65s; after
native result-contract refresh, record-return physical-surface baseline test
(run explicitly while fixing it), scalar-result-type test, and Boolean test
passed3 in12.79s. Full managed replacement session8623/PID2844 active, directory
managed_dt_result_contract_refresh_20260905; inspect continuation before builds.

2026-09-05 15:15: test_native_scalar_index_store.py passed13.39s (actual repository
tensor provider, C/DLL native writes and unchanged neighbor, subprocess20s).
Fullmanaged session8623 terminalexit1: aggregate result types fail convergence;
failed-link snapshot has coerce/advance Loads defining same fieldSSA objects.
No full build active. See continuation before starting next expensive build.

2026-09-05 15:24: native exact-forwarded-record-fields test plus existing
record-return physical-surface test passed2 in12.31s. Native test includes
callee mutation retained through void/inout call and downstream result.
New fullmanaged session86224/PID15832 active, managed_dt_forwarded_fields_20260905.

2026-09-05 15:39: fullmanaged session86224 terminalexit1, passed typefixedpoint,
only Cshortfall was restoredforwardedCallres. Clearingexactone suchres in saved
fullSSA givesCshortfalls empty. Source marker fallback corrected. No new native
test/executable in this turn. Freshfullbuild session76288/PID16400 active,
managed_dt_forwarded_marker_20260905. Seecontinuationbeforestartingwork.

2026-09-05: tests/test_managed_native_output_contract.py: 2 passed in3.81s.
Wrapper-only checks: allocate actual scalar return storage; reject missing root
inputs before invoking native toolchain. Not a native execution/parity test.
Fresh DT source build session76288 is terminal with C shortfalls=[]; wrapper
output44/45 corrected. Native toolchain continuation session44475 remains live.

2026-09-05 17:01: all earlier compile/test sessions terminal. Native cap/physical
storage checks: 7 passed, 1 obsolete typed-C-signature assertion failed (45.84s).
Control source tests: 20 passed (2.04s). Full fresh source DT build session43311
active, O0, managed_dt_predicate_partition_20260905, microstep runtime fixture.
No overlapping native jobs. See continuation before starting builds or timing.

2026-09-05 17:10: full source43311 terminal0. Parity67705 terminal1: native60s
timeout; eager setup15.446909s, stepping0.276230s, exit0. No native/build job live.
Repeated-bounds native regression58263 passed10.22s. Remaining full cap producer
still after loop; see continuation. Predicate AST helper filtering untested.

2026-09-05 17:12: final predicate syntax refinement verified by49988 terminal0:
25 passed,3 inapplicable parameter combinations skipped,39.05s. No live jobs.

2026-09-05 17:22: atomic control dependency regression and prior control/native
cap cases:28passed3skipped39.52s,24126terminal0. Capture15882terminal0. Full fresh
build11416 active in managed_dt_atomic_control_order_20260905, no overlap.

2026-09-05 17:34: fullbuild11416terminal0, cap producer now beforewhile in C.
Parity45565terminal1 native60stimeout/eager0; trace48147terminal0(copy only).
Carried-call regression86189failed then fixed;39229terminal0,5passed17.68s.
No live jobs. Remaining fullstep returns use unproduced locals before producers
in unreachable_return_control. See continuation before next build.

2026-09-05 17:55: return-slot/planner and post-conditional call fixes:7passed
39.92s (73613), direct-truth variants2passed17.61s (20540). All earlier jobs
terminal. Full fresh source build75251 ACTIVE, managed_dt_return_order_20260905,
O0 microstep fixture; no overlap. See continuation before starting another job.

2026-09-05 current terminal correction: build75251 passed; parity6606 finished
(native0/eager0, no timeout) but11 mismatches. managed_dt_return_order_20260905
is latest full DT diagnostic; no parity/performance claim. Dominance tests3passed.
Active lower-only trace9550 writes build/coerce-placement-trace.log; no concurrent
heavy work. See continuation report for call460 loop-ownership contradiction.

Trace9550 TERMINAL0: missing coerce460 marker/anchor precedes fallback insertion;
late ABI refresh keeps already-wrong entry placement. New focused scheduler
capture86652 ACTIVE (lower only), build/step-call-schedule-capture.log.

18:45 scheduler capture86652 terminal0; no live jobs. Conditional call-order
regression passes. Scheduler test file has5 pre-existing failures verified by
in-memory baseline disabling only the new change (75pass5fail; with new test
76pass5fail). Full saved DT plan still orders advance too late; no fullbuild or
parity rerun. See continuation for exact evidence and replay artifact.
