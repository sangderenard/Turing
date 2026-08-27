# Test baseline and hazards — read before running any test

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
| `tests/test_precompile_to_ssa.py` (current tree, 2026-08-23) | **1 failed, 55 passed** | `test_repeat_lowers_as_native_fortran_axis_tiling` reaches `ssa_fortran_backend.emit_module` with its documented mapping input and fails when API publication reads `module.metadata`. Discovered during the native shell-boundary regression gate; no clean-worktree ancestry comparison was commissioned. |
| `tests/test_wasm_class_modules.py` (current tree, 2026-08-24) | **1 failed, 27 passed** | `test_describe_process_graph_api_resolves_the_real_source_parameter_name` sees no logical input from the deprecated `compile_ast_aot` path. Discovered while testing state-feedback inventory wiring; the new focused regression passes and does not exercise that compilation path. No clean-worktree ancestry comparison was commissioned. |
| `tests/test_wasm_class_coordinator.py` (current tree, 2026-08-24) | **1 failed, 10 passed** | `test_wasm_coordinator_calls_cards_internally_and_honors_latched_ranges` broadcasts the first element (`seam=[4,4,4]`, `result=[16,16,16]`) instead of preserving the three-element input. The passing count includes the new control-region state-naming regression. No clean-worktree ancestry comparison was commissioned. |
| `tests/test_symbolic_equation_compiler.py` (current tree, 2026-08-25) | **2 failed, 4 passed** | `test_compiled_scalar_equation_can_run_in_fortran` expects integer literal `2` but the backend emits `2.0_c_double`; `test_fluid_equation_compiler_builds_the_full_default_model` expects 30 inputs but receives 28. Discovered while adding the independent float32 SSA-to-WebGPU vehicle contact path. No clean-worktree ancestry comparison was commissioned. |

## The manifest — known-good at `af00599` plus the current working tree

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
