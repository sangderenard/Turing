# Test baseline and hazards — read before running any test

2026-09-14 full-cycle perforated engine training: the complete focused LLVM
file passes 4 tests in 63.38s. Its three-minibatch regression exercises
gradient accumulation 2, a 0.15 global-norm clip, and a trailing partial group,
matching an independent reference for loss, five parameters, ten moment
tensors, beta powers, iteration, and pre/post-clip norms. The actual headless
LDT path (`423 -> 212`, batch 2) compiled an 18-batch cyclic bank and executed
72 forward/loss/generated-VJP/Adam motions in one native call in 0.999s;
training loss moved 1.11127 -> 0.02979. Adam is presently a backend-native
cycle around the pre-LLVM-composed forward/loss/VJP motion; it is not yet a
linked functional-Adam ProcessGraph.

2026-09-14 adaptive engine sample refresh: a forced-trigger real LDT headless
run completed 2 native epochs / 36 motions with accumulation 2 and clipping
1.0. After epoch 1, a widening 0.26246 validation gap captured 54 new
real-simulator transitions, grew the training pool 33 -> 87 rows, refilled the
same 18-batch LLVM bank without recompilation, and completed epoch 2. Training
loss moved 0.89212 -> 0.13489 and validation improved 0.73089 -> 0.64798.

2026-09-14 perforated startup cache: the real-dataset NPZ round-trip regression
passed in 28.25s, and the native cold-then-cached forward/VJP regression passed
in 22.68s. A direct 50-transition LDT preparation measured 9.537s cold and
0.037s cached. These checks do not cover a full-catalogue capture or long
training run.

2026-09-14 engine capture neutral/cooling/friction: the full
`tests/test_perforated_multifuel_engine.py` passes 11 tests in 250.73 s.
The exact Camry allocation (256 named samples per fuel, 104 random rows,
seed 3747) makes this file take about four minutes; it is not a quick gate.
Before the neutral fix that regression failed with load speed
78,539,816,339.74483 rad/s. The engine_toy friction/stand/oxidation group
passes 16 tests in 11.15 s; the station cooling/stand group passed 8 in
9.97 s. Validation covers real teacher capture and energy accounting, not
an all-profile 30-epoch training run or complete per-bearing calibration.

2026-09-08 sequence-arena/dominance settlement: compiler-owned sequence arenas
now claim storage provenance when created, before result reconciliation can
replace them with a loop-local scalar. Unique operand-free constants whose
original block does not dominate an exact-object use are hoisted to entry with
priority/tie provenance. The focused sequence/dominance batch passed 7 tests
in 2.68s. Saved pre-frame replay v122 converged in 6 frame rounds and 3 result
rounds with zero incompatible contracts; structural findings fell 7 -> 2.
Only `run_superstep` call-result-unavailable IDs 228 and 298 remain. No native
compile, optimization, or execution deadline was used.

2026-09-08 exact energy-limit helper closure: optional scalar returns now use
an explicit payload/presence pair through callee Ret and linked caller Call.
ProgramABI field discovery treats `getattr(obj, "field", default)` as the same
field read as `obj.field`; structural recovery replaces anonymous predicate
formals with their exact membership/isfinite producers and places each private
dependency closure before its unique consumer. `_energy_time_limit` now has
zero formal-parity findings, zero optional-merge findings, and zero C-emission
shortfalls. Its exact-source artifact compiled successfully as a DLL at `-O0`.
The focused optional/extraction batch passed 3 tests in 19.46s; an adjacent
structural/call/optional regression batch passed 10 tests in 8.44s. Neither
used an execution deadline. No helper execution or full-controller parity
claim was made in this chunk.

2026-09-08 Targets sidechain ABI repair: the shared ProgramABI now declares
`energy_exchange_fraction` and `shadow_growth_max` as optional float64 scalar
fields. Before the change, the default structural audit folded
`getattr(targets, "energy_exchange_fraction", None)` to `None`, reducing
`_energy_time_limit` to an unconditional `return None`; afterward the default
audit retains both record inputs, explicit presence slots, all three guards,
keyed channel reads, and the final multiply/divide result with zero structural
shortfalls. Contract/presence checks passed 2 tests in 4.45s, without an
execution deadline. An explicit `-O0` native attempt now reaches the next
honest gate: `_energy_time_limit` needs optional return payload/presence
representation and caller accounting for three recovered structural values.
No optimized build was used.

2026-09-08 scalar return snapshot settlement: focused batch passed 5 tests in
17.74s, followed by a 2-test placement check in 3.64s, without execution
deadlines. Replay v111 replaces managed advance's returned Metrics velocity
positions 194 with the unique dominating write version 295, records two
priority/tie-policy receipts, and has zero structural findings. Its explicit
`-O0` native build exposes snapshot output 295, writes caller slot 1741, and
keeps all three downstream consumers on 1741. One-frame parity remains 54/78;
this is a verified alias-boundary repair, not full controller parity.

2026-09-08 returned-record argument settlement: focused record identity/call
batch passed 3 tests in 4.66s with no execution deadline. Saved full-controller
pre-frame replay v107 converged in 6 frame rounds and 3 result rounds, with zero
incompatible result contracts and zero structural findings. `coerce_metrics`,
`_propose_dt_pen`, and `_apply_energy_sidechain` now receive the produced
`Metrics.max_vel` aggregate slot 1741 instead of stale input field 1654. The
later v107 `-O0` baseline completed at 54/78 one-frame parity.

2026-09-07 starred-generator maximum repair: one compiled native artifact
covered the complete and filtered forms across empty-filter, multi-value,
suffix-wins, signed-zero incumbent, incumbent-NaN, and candidate-NaN cases.
The adjacent generator/dict/control batch passed 8 tests in 28.95s; the final
focused lowering/native batch passed 3 tests in 16.24s. No execution deadline
was used. Saved full-controller replay v15 completed in about 87s without source
extraction or native compilation: `_propose_dt_pen` no longer has a formal
parity finding, and the pre-gate structural count moved 45 -> 44. The controller
still fails later native gates; this is not a full-controller execution proof.

2026-09-07 checked sequence replacement: native keyed copy/self-alias/empty/
capacity-failure/source-bounds regressions plus conditional lowering:
4 passed15.68s (outer timeout90s, native subprocess20s).
Broader sequence-table/conditional-tuple group:42 passed,1 failed41.39s.
`test_compiled_retained_loop_mutates_caller_sequence_record` fails at its
line811 empty-C-file assertion because the generated ABI shim is nonempty;
that fixture uses append, not replacement. The assertion was retained.
See docs/REPAIRS_2026-09-07_KEYED_REPLACEMENT.md for the full diagnostic and
missing keyed constructor/store/owned-field binding frontier.
Fresh lower-only diagnostic `build/keyed_replace_full_formals_20260907.log`:
19 formals, zero undefined operands/unresolved calls, scalar publication
1482 ->9564 ->593 and earlier read1378 preserved. Terminal exit1 at strict
gate; no full native build/parity, no live jobs.

2026-09-06 scalar return publication: focused return-state, reachability,
and authored child-record tests: 18 passed, 1 xfailed in 26.38s, external
timeout90s. Both Boolean and provisional-float64 native variants invoke the
actual final publication pass, preserve output IDs, check idempotence, and
exercise repeated branch outcomes/buffer reuse (native subprocess timeout20s).
The authored child-record case still fails the preexisting ABI provenance
gate and is not an end-to-end native proof. Static-reference cache eviction
reproduced the missing-node crash; the guard fix and related reducer checks
passed 4 tests in 2.26s. See docs/REPAIRS_2026-09-06_RECORD_RETURN_STATE.md
for the fresh whole-source result and remaining keyed/optional-field limits.
Fresh source diagnostic `build/record_return_publication_full_formals_20260906.log`:
four scalar return publications, early hard_failure read1378 unchanged,
returned1482 -> Cast9564 -> ledger593; 19 formals, zero undefined operands or
unresolved calls. Terminal exit1 at strict gate, no full native build/parity.
All launched processes are terminal.

2026-09-06 effect-order repair: targeted scheduling/alias-order gate 12 passed
2.67s; optional-field admission + wrapper checks 6 passed 3.47s. New native
two-append/conditional-clear truth test passed 12.87s across all eight flag
combinations after retaining non-loop clear. Native executions bounded at20s.
Two narrowly detected expected failures (2 xfailed5.32s):
`test_scalar_field_capture_precedes_conditional_write` loses scalar write/guard;
`test_rejected_attempt_never_calls_accept_mutation` loses call-only for-loop/guard.
These are not native proofs. Saved whole-step control replay has zero shortfalls.
Fresh whole diagnostic19 formals, zero undefined operands/unresolved calls;
terminal exit1 at strict gate. Linked field read1378 now precedes laterPhi593;
accept calls follow continue. No full native build/parity; all sessions terminal.
See docs/REPAIRS_2026-09-06_EFFECT_ORDER.md for intermediate failing runs and limits.

2026-09-06 second-opinion reachability repairs: conservative CFG/Phi checks,
formal-provenance gate, and existing paired caller/callee pruning regression:
13 passed, 1 native test deselected in 4.62s. New native return-path test plus
existing conditional-call-result and sequence-truth native checks: 3 passed
27.88s, execution subprocesses bounded at 20s. Saved-SSA replay removed five
private formals (23 -> 18), undefined operands zero. Fresh full diagnostic
completed with 19 formals, zero undefined operands and unresolved calls; exited
1 at the strict gate. It includes the existing field ledger, unlike the old
saved artifact. No full native build or parity run. All sessions terminal.

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


2026-09-07 patch-sequence implementation: mapping copy/empty/overwrite native
cases pass; scalar capture/first-arm/nested/increment plus identity-child native
cases passed 5 in 44.72s. Fixed call-only rejection loop now passes zero/one/three
accepted attempts, rejection and early break on reused buffers in one compile.
Snapshot copy has separate length and contents; the private-frame watch API now
publishes its arena without changing private initialization. Latest mixed batch
passed 9 with one saved-bound-method return failure under repair. Earlier
structural Boolean/control/provenance/receipt batch passed 31, with two test
placement errors subsequently corrected; Boolean cases and control rendering
then passed. The append fixture validates the packed shim rather than expecting
an empty C file. Three former native xfails have passing implementations.

IMPORTANT USER OVERRIDE: “YOU CANNOT KEEP USING TIMEOUTS”. Do not impose further
execution time limits on this work. Two lower-only runs were previously killed
at 300 and 600 seconds and produced no fresh formal report; that workflow was
wrong for this task. The no-deadline full diagnostic is session42101 / PID11756,
log build/patch_sequence_diagnostic_v3.log (started 07:06:28). It predates the
latest bound-method and explicit-split persistence repairs. Do not mistake
microtest passes for completion of the all-19 series or full controller parity.
See docs/IMPLEMENTATION_2026-09-07_PATCH_SEQUENCE.md for the implementation ledger.

2026-09-07 priority repair update: session42101/PID11756 is TERMINAL, explicitly
stopped after read-only stack samples and rule inspection diagnosed a join/split
allocation cycle (next value 70510 -> 71570). This was not an elapsed-time cutoff.
Frame rule priorities now retain incumbents on ties and log transformation
provenance. Broad native/priority batch: 18 passed in 108.71s. Latest focused
priority + bound-method + rejection batch: 9 passed in 20.57s. Saved dependency
closure regression: 1 passed in 3.91s without native compilation.
Saved graph replay v4 terminated on the missing dt_limit_hint deployment shell;
method normalization now precedes dependency closure for persisted graphs.
Current lower-only replay is session85685 / PID11580, log
build/patch_sequence_replay_v5.log. It attempts a pre-frame-link checkpoint.
It uses an older reduced graph, so even successful lowering cannot establish
parity for the latest source reduction edits. Full all-19 implementation remains open.

2026-09-07 09:03 update: v5 through v10 are terminal. The reusable v6
pre-frame-link checkpoint is available; `tools/replay_ssa_checkpoint.py` skips
extraction/planning. Frame linking converges in six rounds. Result types now
use finite proof priorities with incumbent ties; incompatible physical layouts
remain hard findings. Explicit scalar/span input and fresh-output conversions
preserve physical storage. Four transient scalar-read regressions are repaired.
Latest combined batch: 24 passed in 55.69s. Current lower-only replay v11 is
session18036, log build/patch_sequence_replay_v11.log, without an execution
deadline. This cached graph still cannot establish fresh-source controller parity.

2026-09-07 09:30 update: v11 is terminal; v12 completed repository SSA and saved
build/patch_sequence_replay_v12/repository-ssa.pkl, with 47 strict findings.
Result provenance rejection and physical metadata propagation are repaired;
latest priority/scalar/storage batch passed 24 in 46.54s. Dictionary producer,
overwrite, schema, capacity, source ordering and filter-ownership repairs pass a
6-test native collection batch (46.16s), including one reused DLL for constant and
runtime comprehensions. Fresh controller lowering is session43484; logs and
checkpoints are under build/patch_sequence_fresh_v13. No execution deadline.
Full all-19 implementation and controller parity remain open.

2026-09-07 final v13 checkpoint update: session43484 is TERMINAL. Fresh source
lowering completed and saved its pre-frame-link and repository-SSA checkpoints.
Frame/result convergence: 6/3 rounds, two optional-result conflicts. The strict
pre-gate scan reported 45 findings; subsequent gate pruning reports 18 unaccounted
formals across five functions. The controller is still rejected, not native-ready.
Follow-up structural collection/storage tests: 12 passed in 3.39s. No test or
lowering process from this batch remains active.

2026-09-07 region-view accounting update: saved pre-frame replay v16 is
terminal and wrote `build/patch_sequence_replay_v16/repository-ssa.pkl` plus
`build/patch_sequence_replay_v16.log`. An internal reshape/view now reaches an
accounted fixed point only when one owned planned region supplies an exact
tensor descriptor, its resident storage descriptor and SSA producer agree,
and byte extent and dtype match. The receipt records region, storage identity
and target shape. Public outputs, wrapper-visible identities, missing storage,
and equal competing region claims retain the incumbent open finding. This
closed the 17 `balloon_tire_vector_step` reshape findings and the one
`balloon_tire_managed_advance` reshape finding: the strict replay scan moved
from 44 to 26 findings without source extraction or native compilation. The
focused receipt test passed. In the adjacent three-test batch, two tests passed
and the existing structural-boolean opcode-spelling assertion failed even
though its unresolved-call and structural-shortfall checks passed; that
unrelated behavior was not changed. Full controller native parity remains open.

2026-09-07 scalar/keyed ownership update: scalar `.item()` identities and exact
`mapping or {}` keyed reads now retain proof receipts. Lookups consumed only by
a linked source call use that call's exact argument binding as local ownership;
the result no longer becomes a public formal. Focused batch: 3 passed in 7.06s,
without execution deadlines. Replay v17 removed the final three structural
outputs (26 -> 24 strict findings but exposed `_no_exchange_observed` value 11);
replay v18 materialized that lookup at its source-call owner and reports 23
strict findings. Artifacts:
`build/patch_sequence_replay_v18/repository-ssa.pkl` and
`build/patch_sequence_replay_v18.log`. Remaining: four formal-parity groups,
one optional merge, sixteen dominance findings, and two call-result conflicts.
No full native controller compile or parity claim.

2026-09-07 structured while-predicate update: computed numerical regions under
a structural predicate are evaluated at while entry and latch; captured leaves
keep their existing owner. Persisted plans use the same exact unique-owner rule
and record `(loop, region, deployment nodes, rule)` provenance. Focused batch:
5 passed in 5.80s. Replay v19 was unchanged at 23 because it proved the saved
control needed reconciliation as well as fresh composition. Replay v20 then
removed both `run_superstep` `%71` dominance findings and reports 21 strict
findings: four formal groups, one optional merge, fourteen dominance findings,
and two call-result conflicts. No execution deadlines or native parity claim.

2026-09-07 PlanCall loop-ancestry update: lexical source-call placement now
prefers the hierarchy's exact `enclosing_loop_ids` and uses source spans only as
a fallback. This keeps calls inside persisted comprehension loops even when the
comprehension AST has no line span. Focused batch: 2 passed in 4.60s. Replay v21
moves `coerce_metrics` callsite 9 after its loop-target projection, removes its
sole dominance finding, and reports 20 strict findings. The remaining three
unnamed `coerce_metrics` formals were not reclassified or hidden. No execution
deadline, full native compilation, or parity claim.

2026-09-07 defensive keyed-iteration update: exact `mapping or {}` wrappers now
reuse the declared mapping length/key/value ABI slots. Projected iteration keeps
its GEP and indexed Load; only the arena base changes. `str(key)` is an identity
only for a key projected from a declared `string_token` mapping, with separate
provenance receipts. Final focused batch: 2 passed in 4.01s; adjacent direct
mapping, defaulted lookup, and loop-call checks also passed. Replay v22 removes
all three `coerce_metrics` unnamed formals (25 -> 22 function formals) and its
formal-parity finding. Strict frontier: 19 findings—three formal groups, one
optional merge, thirteen dominance findings, and two call-result conflicts.
No execution deadline, full native compilation, or parity claim.

2026-09-07 region in/out ownership update: a planned region's explicit
capture/output intersection now proves that its result and incumbent formal are
the same writable storage, even if later record linking has no field label to
repeat. Mutable scalar fields that are only written now retain their exact
`SetAttr` source identity when scalar control lowering cannot represent the
producer. Focused batch: 8 passed in 11.09s. Replay v24 removes managed advance
`%182`, reduces the converged frame from 4,330 to 4,290 formals, and reports 18
strict findings: two formal groups, one optional merge, thirteen dominance
findings, and two call-result conflicts. Artifact:
`build/patch_sequence_replay_v24/repository-ssa.pkl`. No execution deadline,
full native compilation, or parity claim.

2026-09-07 total scalar dt-limit hint update: the managed tire returns its
positive declared integration step or `0.0` when inactive. `run_superstep`
already admits only finite positive hints, so this removes an unneeded
payload/`None` union without changing which bound is applied. Focused batch: 4
passed in 68.62s. A fresh source build was required because v24's pre-frame
checkpoint contained the old graph. Fresh v25 reports 17 strict findings—two
formal groups, thirteen dominance findings, and two call-result conflicts—and
no optional merge. The repository SSA was saved before the expected remaining
full-native gate rejection at
`build/patch_sequence_fresh_v25/repository-ssa.pkl`. No execution deadline,
native code compilation, or parity claim.

2026-09-07 concrete managed-tire Metrics update: the tire advance now reports
`advanced_dt=dt` and uses the largest finite float as the neutral `dt_limit` for
the controller's minimum clamps. Focused batch: 2 passed in 15.18s. Fresh v26
reports zero incompatible result contracts, an empty structural-output gate,
and 15 strict findings: two formal groups plus thirteen dominance findings.
Artifact: `build/patch_sequence_fresh_v26/repository-ssa.pkl`. No execution
deadline, native code compilation, or parity claim.

2026-09-07 predicated-continue edge completion: focused loop lowering passed 4
tests in 2.55s. Replay v27 removes both while-header definition-dominance
findings and reports 13 strict findings: two formal groups plus eleven
function-exit dominance findings. Artifact:
`build/patch_sequence_replay_v27/repository-ssa.pkl`. A broader three-file run
reported 125 passed and 16 failed. Two failures are the recorded
`test_precompile_to_ssa.py` baseline above; the other 14 reflect unresolved
dirty-tree suite drift and are not treated as green validation for this chunk.
No execution deadline, native compilation, or parity claim.

2026-09-07 constant-while CFG update: exact Boolean while predicates now use a
direct header branch while preserving real break exits. The focused
while/return batch passed 8 tests in 2.48s; the branch-compartment file passed 3
tests in 2.05s. Replay v28 removes all eleven function-exit dominance findings
and reports only two strict findings, both formal-parity groups. Artifact:
`build/patch_sequence_replay_v28/repository-ssa.pkl`. No execution deadline,
native compilation, or parity claim.

2026-09-08 scalar-field conditional update: the final focused conditional slice
passes 6 tests in 1.96s. Replay v35 completes without an execution deadline,
converges at 4,280 frame formals, reports zero incompatible result contracts,
and removes anonymous `%347` (`ctrl.clamp_events`) from the raw controller
formal-parity group. The raw surface is now 17 unnamed values across the same
two groups. Artifact: `build/patch_sequence_replay_v35/repository-ssa.pkl`.
The production pruning gate and native execution were not rerun, so this is not
a parity claim.

2026-09-08 exact record-field-state precedence: the combined focused batch
passes 7 tests in 3.68s. Fresh v40 completes source extraction, planning, and
repository SSA lowering without an execution deadline. It retains 37
controller regions, converges at 4,274 frame formals, and reports zero
incompatible result contracts. Exact reducer field-state Phis now suppress
duplicate flat attribute histories, removing controller formals `%428` and
`%429`. The production-pruned frontier is `run_superstep` `[124, 126]` and
`step_with_dt_control_used` `[477, 458]`. Artifact:
`build/patch_sequence_fresh_v40/repository-ssa.pkl`. The gate rejects those
four remaining anonymous values, so no native compilation or parity claim.

2026-09-08 stable controller diagnostic tokens: the dt controller graph now
contains zero `JoinedStr` nodes. The focused controller batch passed 21 tests in
1.26s and specialized source-provenance/formal recovery passed 7 tests in
4.08s. Fresh v32 reduces controller regions 44 -> 37, converged formals 4,320 ->
4,283, and final-gate anonymous slots 12 -> 7. Two formal-parity groups remain:
`run_superstep` `[124, 126]` and `step_with_dt_control_used`
`[428, 347, 429, 477, 458]`. Artifact:
`build/patch_sequence_fresh_v32/repository-ssa.pkl`. No execution deadline,
native compilation, or parity claim.

2026-09-08 invocation-site structural BoolOp feeds: call linking now
reconstructs an exact source Boolean expression after provisional-formal
creation, normalizes graph node keys through semantic `value_id`, and retires
the unaccounted placeholder only after every ordered operand resolves. ABI or
frame storage cannot be reclaimed, and a resident same-ID producer wins. The
focused batch passes 5 tests in 3.13s. Fresh v41 completes source extraction,
planning, and repository SSA lowering without an execution deadline, converges
at 4,273 frame formals, reports zero incompatible result contracts, and has no
dominance finding. It removes controller `%458`.
The raw surface is 14 unnamed values across two formal-parity groups; production
pruning leaves `run_superstep` `[124, 126]` and
`step_with_dt_control_used` `[477]`. Artifact:
`build/patch_sequence_fresh_v41/repository-ssa.pkl`. The gate rejects those
three remaining values; its unmaterialized-boundary, unresolved-call,
undefined-operand, optional-merge, structural-output, and non-native families
are empty. No native compilation or parity is claimed. The broader
call-linking file reports 47 passed and 9 failed in 16.97s; the failures are in
record aliasing, keyed mapping/default selection, span propagation, and
callsite shape behavior rather than the new structural BoolOp case. This is a
recorded dirty-tree suite frontier, not a whole-file pass.

2026-09-08 total unresolved-report sequence: `Metrics.unresolved_report` now
has an empty `list[str]` default, and exact column dtype evidence propagates
through matching materialization and aggregate-Phi edges to a finite fixed
point. Existing contracts retain precedence. The focused batch passes 3 tests
in 2.45s. Fresh v45 completes repository SSA lowering without an execution
deadline, converges at 4,276 formals, and removes controller `%477` from the
production-pruned gate. That gate now contains only `run_superstep`
`[124, 126]`; all other gate families are empty. Raw self-checking exposes one
separate controller dominance defect where `if_merge.11` reads `%478` from a
`while_exit` definition. Artifact:
`build/patch_sequence_fresh_v45/repository-ssa.pkl`. The gate stops before C
compilation, so this is not a native parity claim.

2026-09-08 resident report-tail and table ABI update: constant nonnegative
tail slices over resident sequences now lower as loop starts over the base
arena. ProgramABI table fields allocate complete resident sequence storage
when specialization removes their aggregate producer, and record-frame
linking correlates the complete sequence while retaining the caller incumbent
on equal-priority identity matches. The focused batch passes 5 tests in 6.22s.
Replay v54 converges after six frame rounds and three result-type rounds, with
zero incompatible result contracts and seven raw structural findings. It
removes `run_superstep` `%126`; `%127` plus the missing physical `Metrics` row
descriptor remain the direct `run_superstep` frontier. Artifact:
`build/patch_sequence_replay_v54/repository-ssa.pkl`. No execution deadline,
production-pruning rerun, native compilation, or parity claim.

2026-09-08 tail-domain ownership update: regions consumed by structured loop
bounds are excluded from flat scheduling independently of complete loop-body
emission. Deferred record rows resolve exact compiler aliases before layout
checking. The exact focused regression passes in 4.37s. Fresh v56 converges at
4,287 formals with zero incompatible result contracts and six raw findings.
It removes the `run_superstep` formal-parity group entirely; `%127` is gone.
The remaining `run_superstep` issue is a measured 11-versus-15 `Metrics` row
layout, requiring the three `error_channels` columns and a nested report-table
handle. Artifact: `build/patch_sequence_fresh_v56/repository-ssa.pkl`. The
broader loop-filtered batch remains non-green at 65 passed / 3 failed. No
execution deadline, C compilation, or native parity claim.

2026-09-08 late returned-record surface update: call linking follows exact
caller result aliases and merges callee fields discovered in later fixed-point
rounds into the resident caller descriptor. It adds only missing flat storage;
aggregate fields require real resident descriptors, and equal-priority alias
ties keep the incumbent. Replay v57 converges after six frame rounds and three
result-type rounds with zero incompatible contracts. The `run_superstep`
`Metrics` append improves from 11/15 to 14/15 columns; only the nested
`unresolved_report` handle remains. The fuller record surface also exposes 23
anonymous `step_with_dt_control_used` formals and raises the frame total to
4,365, so the raw finding count remains six. The two focused flat-record tests
pass; the existing full `Metrics` row test remains non-green at the known
nested-table boundary. Artifact:
`build/patch_sequence_replay_v57/repository-ssa.pkl`. No execution deadline,
production-pruning run, C compilation, or native parity claim.

2026-09-08 optional-control/local-mutation update: pre-planning presence
rewriting, scalar C `logical_not`, exact synthetic return merges, distinct
optional payload storage, adjacent mutable presence stores, canonical
ProcessGraph allocation, and idempotent aggregate-leaf ledger repair pass 17
related tests in 8.12s. The two executable optional tests compile only at
`-O0` and cover absent, present-zero, present-nonzero, and absent-to-present
local mutation. No execution deadline was used.

Fresh v71-v74 are diagnostic failures at a stale `snapshot` aggregate ledger;
do not use them as SSA baselines. Replay v76 from the fresh v74 resolved graph
passes that planner boundary, converges frames in six rounds and result types
in two, and saves `build/patch_sequence_replay_v76/repository-ssa.pkl`. Its raw
audit has six findings: four formal-parity groups (1, 3, 24, and 388 unnamed
values) and two controller function-exit dominance errors. The linked
`update_dt_max` specialization still drops optional-presence provenance, so
the full controller is not ready for C emission or native parity. Do not run
an optimized compile while these correctness findings remain.

2026-09-08 zero-structural/zero-emission checkpoint: recursive late record
materialization closes the window `Metrics.hard_failure` boundary. Replay v68
and fresh production v69 report zero structural findings. C emission initially
reported 18 shortfalls, fell to two after scalar `item` and finiteness support,
and reaches zero after exact initial record-field projections become explicit
planned-region scalar captures. Replay v70 records both `clamp_events`
projection receipts against incumbent `%124` and removes the obsolete record
receiver from the region ABI.

The first full compile exposed repeated declarations of mutable scalar SSA
identities (`t5` and `t11`). Later loads now assign the first C local rather
than redeclaring it. The focused executable regression and the related record
capture/item tests pass: 3 tests in 3.55 seconds, following a 5-test batch in
4.53 seconds. The full zero-shortfall v70 source compiles successfully at
`-O0` to a 1,650,688-byte DLL. No execution deadline was used. Do not use an
optimized full compile until correctness execution and native parity are
complete; parity is not yet claimed.

2026-09-08 optional ABI checkpoint: attempting to construct real v70 native
feeds stopped on `controller.dt_min=None` before the DLL ran. This is a useful
guard failure, not a native result. ProgramABI now represents optional scalar
fields with separate Boolean presence and typed payload slots; feed packing
rejects absence unless the pair exists and preserves present zero. The default
controller schema marks `dt_min` and `dt_max` optional. Contract receipt,
lowering materialization, absent/present packing, and the legacy rejection path
pass 9 focused tests in 4.32 seconds. No deadline was used. Do not treat this
as optional-control correctness: source predicates and mutable presence writes
still need to be wired and verified in a fresh build.

2026-09-08 controller formal-closure update: exact resident-field projections,
retained mutation literals, and whole-object unreachable-CFG/Phi pruning remove
all 23 anonymous controller formals. Replay v67 converges after six frame rounds
at 4,344 formals and three result-type rounds with zero incompatible contracts;
the pass records 18 CFG/Phi changes. The raw checker now reports one finding in
the entire saved module: window `%47`, the `hard_failure` field read from
returned `Metrics` receiver `%46`, whose record descriptor is absent in the
window caller. The focused reachability/ABI suite passes 6 tests in 8.57s,
including native execution of both retained return paths, without an execution
deadline. Artifact: `build/patch_sequence_replay_v67/repository-ssa.pkl`. No C
compilation of the full controller or native parity is claimed.

2026-09-08 resident returned sequence update: exact one-output planned
projections of a descriptor-backed record sequence field become accounted
resident arenas after frame linking. Replay v58 proves `%478` has no local
definition, remains a live consumed argument, and carries provenance naming
planned region 34. Both dominance findings disappear; raw findings fall from
six to four, while the controller anonymous-formal set shrinks from 23 to 20.
The new fixed-capacity child-copy append helper passes its direct CFG test and
the combined focused batch passes 3 tests in 4.78s. Deferred record rows do not
yet invoke that helper, so `run_superstep` remains 14/15 and the controller row
still lacks `unresolved_report`. Artifact:
`build/patch_sequence_replay_v58/repository-ssa.pkl`. No execution deadline,
production-pruning run, C compilation, or native parity claim.

2026-09-08 deferred child-row update: whole-record and embedded-record rows
now expand in ProgramABI order and snapshot one nested leaf sequence into a
caller-owned fixed-stride child pool. Replay v64 removes both record-row
findings, leaving two raw formal-parity groups only: window `%47` and 23
controller values. Sequence 69/66 uses handle column 14 and a 37-argument
helper; sequence 39/478 uses handle column 15 and a 41-argument helper. Every
source-linked call has exact callee arity, every child-pool identity names a
resident value, and deferred row markers are absent. The focused batch passes
4 tests in 5.60s. Artifact:
`build/patch_sequence_replay_v64/repository-ssa.pkl`. No execution deadline,
production-pruning run, C compilation, or native parity claim.

2026-09-08 optional cross-call checkpoint: never treat equality between a
ProcessGraph presence ID and an SSA instruction-result ID as ownership. The
presence slot must be a real formal; otherwise its Store may survive while the
call frame silently omits the target. Materialization now allocates a distinct
physical ID on that collision, records requested and physical IDs, and uses
payload/presence roles in frame keys. Equal-priority formal evidence keeps the
incumbent.

Replay v80 saves `build/patch_sequence_replay_v80/repository-ssa.pkl` and
proves requested presence `%24` is physical `update_dt_max` formal `%37`, the
presence Store targets `%37`, and the source-linked call carries the matching
fifth operand. It converges at 4,353 formals in six rounds and reports the same
six raw findings as v76: formal groups 1/3/24/388 and dominance `%263`/`%264`.
Focused optional tests compile only at `-O0`. No full compile, optimized
compile, execution deadline, or parity claim was used.

2026-09-08 return-edge dominance checkpoint: exact direct return expressions
from pure planned regions are recomputed on a synthesized physical return edge
when their original path-correlated definition does not dominate it. The rule
requires matching return-slot provenance, allocates fresh SSA identities,
retains a dominating incumbent, records priority and tie policy, and dead-ends
after one accepted repair.

Replay v81 records `%263 -> %3390` (three operations) and `%264 -> %3395`
(five operations). Definition-dominance findings are zero; the raw checker now
reports four formal-parity groups only, sized 1/3/24/388. The focused
return-state and dominance batch passes 14 tests with the two pre-existing
native subprocess tests deselected. No compilation or execution deadline was
used. Artifact: `build/patch_sequence_replay_v81/repository-ssa.pkl`; audit:
`build/patch_sequence_replay_v81/return-edge-audit.txt`.

2026-09-08 missing-Phi-incumbent checkpoint: when reduction removes the value
named by `initial_value_id`, control lowering must not externalize its integer
identity. The graph repair accepts only one nearest same-binding common
ancestor of both Phi arms, leaves equal candidates unresolved, records the
incumbent tie policy once, and projects accepted repairs into already-planned
Control IR.

Replay v84 records `run_superstep` `%262: initial %260 -> %256`; final operands
are `(%261, %256)`, formal `%260` is absent, and its only source-linked caller
still has exact arity. Frame formals fall 4,353 -> 4,351. The raw checker now
reports three formal-parity groups sized 3/24/388. The focused control file
passes 7 tests. No compilation, optimization, execution deadline, or parity
claim. Artifact: `build/patch_sequence_replay_v84/repository-ssa.pkl`; audit:
`build/patch_sequence_replay_v84/phi-initial-audit.txt`.

2026-09-08 post-reachability pure-feed checkpoint: scalar `item()` may consume
a unique dominating resident producer when no loop-carried Phi exists; multiple
carried candidates remain unresolved. Repeat exact pure-expression recovery
after static reachability pruning and before atomic call-signature pruning so a
dead CFG predecessor cannot preserve an internal expression as an ABI input.

Replay v86 removes controller formals `%147`, `%149`, and `%231` together with
their caller operands. Their definitions are exact source operations rooted at
resident `%43`, and controller/caller arity is 1,061. The raw checker reports
two remaining formal-parity groups sized 24 and 388. The focused recovery file
passes 5 tests. No compile, optimization, execution deadline, or parity claim.
Artifact: `build/patch_sequence_replay_v86/repository-ssa.pkl`; log:
`build/patch_sequence_replay_v86.log`.

2026-09-08 static-slice checkpoint: a source slice used only as an index may
become its integer lower-bound address selector only when its bounds are static
and non-negative and its stride is one. The Const receipt retains complete
bounds and source identity; unsupported slice semantics stay unresolved.

Replay v87 recovers 24 managed-advance and 388 vector-step selectors, giving
exact arities 438 and 26. Final formals fall 5,619 -> 5,207 and `run_all`
reports zero findings. Focused result: 4 passed, 81 deselected. Artifact:
`build/patch_sequence_replay_v87/repository-ssa.pkl`; log:
`build/patch_sequence_replay_v87.log`.

The subsequent fresh `-O0` build reached frame convergence and stopped before
C emission/compilation at one physical-input mismatch: caller `%109`
`float64` versus Boolean optional-presence formal `%94` in `pi_update`. No
deadline or optimized compile was used. Log:
`build/patch_sequence_fresh_v88_o0.log`.

2026-09-08 optional-presence physical-priority checkpoint: ProgramABI presence
materialization now writes physical `bool`, retaining the incumbent tie policy
and recording any displaced provisional physical dtype. This prevents an
earlier numerical-region capture from reclassifying the optional-presence
storage as `float64`.

Fresh pre-frame artifact: `build/patch_sequence_fresh_v89/pre-frame-link.pkl`.
Replay v90 gives controller `%109` and `pi_update` `%94` matching physical
Boolean contracts, exact call arity 12, zero result conflicts, and zero
`run_all` findings. Both focused optional tests pass and compile only at `-O0`.
Saved-module C emission reports zero shortfalls; the `-O0` DLL is 1,669,632
bytes. No runtime parity or optimized build was used. Replay log:
`build/patch_sequence_replay_v90.log`; compile log:
`build/patch_sequence_replay_v90/native-o0-build.log`.

2026-09-08 standalone checkpoint: optional-field linked aliases and private
linked result-record workspace are now packed deterministically. The focused
native material-contract file passes 11 tests. Saved v90 emits and builds a
78-buffer standalone executable at `-O0`; the parity tool reads the checkpoint
reference from its manifest and runs both processes without a deadline.

One-frame eager execution completes. Native exits with a Zig misalignment
panic before producing final buffers because root planned region 3 declares
its input `%47` as `float64` while the caller supplies Boolean
`Metrics.hard_failure` result `%308`; generated `cast_double_to_bool_values`
therefore reads byte storage through `double *`. This exact call-edge dtype
conflict is the current runtime frontier. Report and logs:
`build/patch_sequence_replay_v90/standalone-o0/managed-dt-parity.json`,
`parity-native.log`, and `parity-eager.log`. No optimized build or parity claim.

2026-09-08 physical-edge checkpoint: formal/capture occurrences sharing an
unshadowed ID are interned to the incumbent formal, and physical input adapters
run again after final result/aggregate/precision settlement. The transaction
is finite: replay v94 adds five late adapters and an immediate repeat adds
zero. Keyed ProgramABI members and generated lookup helpers carry explicit
physical dtypes; the prior two string-token conflicts are zero. Focused
adapter/keyed results are 7 passing tests (one small native compile at `-O0`)
plus 3 passing lookup tests.

Replay v94 converges in 6 frame rounds and 3 result-type rounds with zero
structural findings. Its 78-buffer standalone builds at `-O0`. Native and eager
both complete the one-frame run with return code 0, closing the misalignment
panic frontier. Numerical comparison has 54 matches and 24 mismatches; native
material state contains 3,456 non-finite elements. `advanced` and `dt_next`
also diverge (`0.000244140625`/`0.0` native versus
`0.008333333333333333`/`1.856126911235753e-05` eager). The next gate is the
earliest native producer of that numerical divergence. Report:
`build/patch_sequence_replay_v94-standalone-o0/managed-dt-parity.json`. No
optimized build or parity claim.

2026-09-08 collapsed-proposal checkpoint: `run_superstep` now stops before
calling physics when its fully bounded proposal is zero or negative. This
closes the traced zero-step division cycle while preserving the existing
incomplete-window report. `tests/dt_system/test_dt_superstep.py` passes 16
tests, including both collapsed cases. Fresh v96 reports zero structural
findings after six frame rounds and three result-type rounds.

The v96 78-buffer standalone builds only at explicit `-O0` and both one-frame
processes return 0. Parity is still 54 matches / 24 mismatches. Native state has
2,056 non-finite elements, all in batch lane 4, versus 3,456 in v94; the first
substep's earlier numerical divergence remains open. Report:
`build/patch_sequence_fresh_v96-standalone-o0/managed-dt-parity.json`.

2026-09-08 indexed-store RHS checkpoint: tensor SSA lowering now computes
`index_assign_double`'s `value_count` from the RHS shape. The old selection-size
value made scalar broadcasts overread adjacent memory. The 22-test focused
batch passes, and fresh v97 audits all 37 indexed stores with zero RHS-count
mismatches. Frame linking converges in six rounds, result typing in three, with
zero incompatible contracts and zero structural findings.

The v97 78-buffer standalone builds at explicit `-O0`; native and eager return
0. Native state is fully finite (27,648/27,648) and all eight batch lanes now
agree, eliminating v96's 2,056 non-finite elements. Comparison remains 54
matches / 24 mismatches: native advances one `0.000244140625` substep and then
produces `dt_next == 0`. The active gate is the inflated native first-substep
metric/reduction result. Report:
`build/patch_sequence_fresh_v97-standalone-o0/managed-dt-parity.json`.

2026-09-08 same-pass shape-settlement checkpoint: tensor lowering now fills a
missing SSA occurrence shape from an already resident static descriptor of the
same identity. It does not replace shaped incumbents, and records provenance
plus the incumbent tie policy. The 24-test focused batch passes. Fresh v98 has
zero structural findings after six frame rounds and three result-type rounds;
the formerly scalar gas reciprocal is audited at shape `(8, 4)` and count 32.

The v98 78-buffer standalone builds at explicit `-O0`; native and eager return
0. Native first-step displacement and velocity now match the eager direct-step
reference. One-frame parity remains 54 matches / 24 mismatches because native
still stops at `advanced == 0.000244140625` with `dt_next == 0`; worst state
error improves from `3.8615154563` to `0.00890879321`. The controller
continuation/result path is the active frontier. Report:
`build/patch_sequence_fresh_v98-standalone-o0/managed-dt-parity.json`.

2026-09-08 v112 checkpoint: settled identity-forward results replace 37 stale
record-field uses with exact caller residents and leave zero structural
findings. The focused combined batch passes 5 tests. The `-O0` one-frame run
still has 54 matches / 24 mismatches, but native `max_vel_ever` is now
`0.0023950195322857593` and native `dt_max` is `1.252599387837501`, proving
that the corrected metric result reaches controller state. Report:
`build/patch_sequence_replay_v112-standalone-o0/managed-dt-parity.json`.

The next zero-`dt_next` producer is a region/effect ordering error: the later
scalar-field scheduler placed an atomic region by its earliest source member,
which can move a region containing early casts and a late field update before
the field write it reads. Atomic regions now use their completion position.
The regression is included in the same 5-test batch. Because this affects
pre-SSA planning, it has not been claimed against v112 and requires a fresh
controller replay. No optimized build or process deadline was used.

2026-09-08 v116 checkpoint: fresh v113 first confirmed the atomic-region
completion repair by moving native controller `acc` from `0.0` to
`0.42714935345574623`. Exact optional `is not None` controls are now recovered
from matching ProgramABI presence slots. Restoring those branches exposed four
non-dominating stale values; object-identity Phi continuation and exact
same-physical-type cast-result forwarding reduce them to zero. Replay v116
records eight and one repairs respectively, with provenance and incumbent tie
policy.

The focused batch passes 7 tests. Replay v116 converges in 6 frame rounds and 3
result-type rounds with zero incompatible contracts and zero structural
findings. Its 78-buffer standalone builds at explicit `-O0`, and native/eager
both complete. Numerical parity is still 54/78: native controller
`acc=0.42714935345574623`, `dt_max=1.252599387837501`,
`advanced=0.000244140625`, and `dt_next=0.0`. The active frontier is the
controller value/presence propagation that still yields a zero next proposal.
Report: `build/patch_sequence_replay_v116-standalone-o0/managed-dt-parity.json`.
No optimized build or process deadline was used.

2026-09-08 v118 checkpoint: exact reuse of one `SSAValue` object by an earlier
Phi and a later branch projection is now freshened per definition. Dominance
selects the uses of each fresh version, including separate incoming Phi edges;
the first definition remains incumbent. The C backend also preserves
exact-object bindings for formals, aggregate projections, and Phi locals.
Replay v118 records eight freshenings, six Phi continuations, one identity-cast
forwarding, and zero structural findings. The focused batch passes 9 tests.

The 78-buffer executable builds at explicit `-O0`. One-frame parity improves
from 54/78 to 55/78: `advanced` now matches eager at
`0.008333333333333333`, while `dt_next` moves from zero to
`0.013326566940131533`. Native `max_vel_ever=0.3159734017361334` still differs
from eager `0.009045569831777825`, so the active frontier is the earliest
multi-substep numerical divergence. Report:
`build/patch_sequence_replay_v118-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

2026-09-08 v125 checkpoint: speculative structural recovery findings are now
settled only when their identities are absent from every final live SSA
surface. Static address-index analysis ignores nonfinite/nonintegral floats.
Proven rank-zero `item()` identities rebind stale operand objects to a unique
incumbent, and linked BoolOp call feeds reconstruct from exact ordered source
operands rather than accepting an operand-only placeholder. Proven changes
record priority, source identity, and the incumbent tie rule.

The focused batches pass 9 tests and 7 tests. Replay v125 converges in 6 frame
rounds and 3 result-type rounds with zero incompatible contracts and zero
structural findings. Its 82-buffer standalone builds successfully at explicit
`-O0`. Native and eager one-frame processes both return 0; parity is 59/82.
The 23 mismatches are material telemetry/state, controller state, and
`dt_next`. Native/eager `max_vel_ever` are `0.3159734017361334` /
`0.009045569831777825`; native/eager `dt_next` are `0.013326566940131533` /
`1.85612691e-05`. The active gate is the earliest material/telemetry producer
divergence inside the adaptive substeps. Report:
`build/patch_sequence_replay_v125-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

2026-09-08 v127 checkpoint: returned keyed record descriptors and exact static
capacity bounds now close transitively over linked call frames. Record
constructors preserve their literal mapping universe because schema packaging
is nonmutating. The propagation is a monotonic finite fixed point and records
exact ABI provenance plus incumbent tie policy. A focused batch passes 4 tests.
Replay v127 converges in 6 frame rounds and 3 result rounds with zero
incompatible contracts and zero structural findings. Capacity 3 is present on
the full advance 213 -> step 1668 -> superstep 733 -> root 349 chain. The
82-buffer standalone builds at explicit `-O0`, and generated C initializes the
root capacity cell to 3. Live parity remains 59/82 with the same 23 mismatches
and values, so the next measured frontier is the runtime alias/store/length
path across that chain. Artifact:
`build/patch_sequence_replay_v127/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v127-standalone-o0/managed-dt-parity.json`. A
separate synthetic three-entry record fixture exposed a control-effect
scheduler cycle and was left for a later focused chunk. No optimized build or
process deadline was used.

2026-09-08 v128 checkpoint: structural recovery of `not in` now emits unary
`LNot` over the exact native contains result. The old recovered form used
`contains != False`, inverted the source meaning, and made
`_energy_time_limit` return absent when both energy keys were present. Both
recovered negations record exact-source priority and incumbent tie policy. The
combined capacity/membership batch passes 5 tests. Replay v128 converges in 6
frame rounds and 3 result rounds with zero incompatible contracts and zero
structural findings; its 82-buffer standalone builds at explicit `-O0`.
Native and eager both return 0. Parity remains 59/82, but native
`max_vel_ever` improves from `0.3159734017361334` to `0.017680656115454992`
against eager `0.009045569831777825`, and native `dt_next` moves from
`0.013326566940131533` to `6.104353894230951e-06` against eager
`1.85612691e-05`; `advanced` still matches. The newly live many-substep native
path completes naturally in about 12 minutes 22 seconds. The next frontier is
the traced store-before-clear loss of `maximum_substep_displacement_m`.
Artifact: `build/patch_sequence_replay_v128/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v128-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

2026-09-09 v129 checkpoint: dynamic mapping literal rows are anchored to their
key AST nodes inside the literal instead of to possibly much earlier value
producers. Value dependencies still run first; authored key positions preserve
the literal's clear-before-row order. Lexical mutation installation now uses
the shared atomic-region completion-position rule. A focused batch passes 3
tests, including one native `-O0` artifact reused for three executions. Replay
v129 converges in 6 frame rounds and 3 result rounds with zero incompatible
contracts and zero structural findings. Managed advance sequence 213 contains
`clear, displacement, energy, power` in one block, and the 82-buffer standalone
builds at explicit `-O0`. Both parity processes return 0. The result remains
59/82 and controller values are byte-for-byte unchanged from v128, proving the
restored displacement row is not the active limiter in this fixture. Artifact:
`build/patch_sequence_replay_v129/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v129-standalone-o0/managed-dt-parity.json`. No
optimization or process deadline was used.

2026-09-09 v130 checkpoint: the module C emitter no longer prebinds a locally
defined returned SSA object to caller-owned output storage. Internal consumers
now read the producer's local `tN`; `Ret` publishes it afterward. Formal
aliases and aggregate projections retain exact-object bindings. A focused
batch passes 4 tests, including an explicit `-O0` compiled regression for a
returned Boolean consumed before publication. The clean v129 SSA checkpoint
was reused because SSA is unchanged. Its 82-buffer v130 standalone builds at
explicit `-O0`, and the production criticality condition emits `!t307` instead
of reading `out307`. Both parity processes return 0. Parity remains 59/82, but
native telemetry moves from 0 successful / 158 critical substeps to 158
successful / 0 critical; advanced time remains exact. The next frontier is the
loop completion boundary: native performs 158 successful substeps and leaves
completion false, while eager performs 157 and completes. Report:
`build/patch_sequence_replay_v130-standalone-o0/managed-dt-parity.json`. No
optimization or process deadline was used.
