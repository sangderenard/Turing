# Continuation: compiled collocation Jacobian (condensed)

Condensed 2026-10-03 from a 1,057-line working log. Full text: `git show a7f57c1e:docs/concordance_census/CONTINUATION_jacobian_compile_fixes.md`
(or any earlier commit touching this file).

## Status

Lane CLOSED, fully committed. Headline: the compiled collocation Jacobian matches SymPy to
max_rel 4.138e-16 (median 0, Jacobian shape (7,19), nnz 55, forward max rel 2.39e-15, LLVM
shortfalls (), lowering ~235 s). Repro, from the turing repo root:
`python tools/compiler_probes/probe_collocation_jacobian.py jacobian --compile`
(prints `RESULT jacobian max_rel ...`). The log does not mention a PYTHONPATH requirement for
engine_toy; the probe imports the engine_toy actuation law, so if it fails to import, put engine_toy
on PYTHONPATH (not verified in the log). Task source: `docs/DIFFERENTIATION_FEASIBILITY_2026-10-02.md` item 1; base 25257ac5.

## Defects found (symptom / root cause / fix / commit)

Commit map: de609156 (Fix 1 + Fix 2), 095a3c0c (items 3-7), 0519095d (item 8), ccba325c (items 9-10),
51b4cebe (items 11-14), ce587b82 (item 15). Per-item mapping below is from the commit stat + log
order; the log itself names hashes only loosely.

1. Sympy `Min`/`Max` had no graph op. `symbolic_process_graph.py`: now ingest as `minimum`/`maximum`
   (pairwise left fold). Every SSA reader resolves ops through
   `hierarchical_plan.TENSOR_OPERATION_SCALAR_SPELLING` (`maximum`->`Max`): `ssa_python_materializer`
   `_BodyMaterializer.step`, `ssa_c_backend.emit_ssa_function_to_c`, `ssa_llvm_backend.emit_ssa_function_to_llvm`.
   Ops with no scalar form keep their name (else `isfinite`/`isnan`/`isinf` lose the catalogue route). (de609156)
2. Backward graph and sympy forward graph had no ingestion scope, so adjoint rows had nothing to derive
   from ("node 3 has no ingestion scope"). Sympy ingestion now mints `ingestion:sympy`; adjoint builder
   mints `ingestion:adjoint` (NOVEL `ADJOINT_OF` from the forward cell); fusion mints
   `ingestion:training_motion` (DERIVED). Files: `symbolic_process_graph.py`, `symbolic_equation_compiler.py`,
   `process_graph_autograd.py`, `topological_reducer.node_ingestion_scopes`, `concordance_declarations.py`. (de609156)
3. Backward `Call bw_*` silently dropped (grads and seeds missing; KeyError 7/806; `buffer_order (0,1,2)`).
   Cause: `glsl_deployment_strategy.plan_callsites` required `ast.Call` in `expr_obj` (added 42a689a2);
   graph-native calls (adjoint builder) have `expr_obj=None`. Fix: no authored AST -> admit iff declared
   op casefolds to "call"; AST nodes still need `ast.Call`. left*right VJP now `(0,1,3,2,7,8)`, grads [55,91]/[22,39]. (095a3c0c)
4. `__plan_callsite_N__` marker left in bw_pow/bw_log after the `eps` decomposition (two definitions of one id,
   LLVM "operation has no repository LLVM emission"). Fix: `fortran_c_shell.py` eps decomposition retires the
   marker with matching `plan_callsite_id`. (095a3c0c)
5. `where(c, 0, 0.5)` typed int64 (0.5 truncated). Fix: `glsl_deployment_strategy.py` where descriptor uses
   `np.result_type` of the two value arms. (095a3c0c)
6. `where_double` read int64 literal arms as doubles (grad 1e-323). Fix: `tensor_ssa_lowering.py` ~:4958
   respells shape-matched scalar non-float64 literal arms as float64 constants. (095a3c0c)
7. Probe fixture drift: actuation law gained `attitude_{ab}` and `propellant_supply`; reference Jacobian was
   non-numeric. Fix: probe `slice_laws` binds attitude = identity, propellant_supply = 1. (095a3c0c)
8. Catalogue-level literal fold of `bw_pow`'s `p` dropped its tensor fact, leaving `unbroadcast(..., p.shape)`
   with an unlowerable `extent` Call. Fix = decision (b)/(i) below; adjoint builder declares
   `argument_roles` on backward-rule Calls. (095a3c0c for the tensor-argument block, 0519095d for roles)
9. Non-terminating `_propagate_callsite_tensor_specializations` fixed point (scalar-loss test, 184+ rounds,
   members flipping () vs (2,3)). Cause: rank-0 formal shapes were never published, so `unbroadcast` G=() vs (2,3)
   recorded no `FORMAL_SHAPE_CONFLICT` and copies shared one authored-name row. Fix: publication gates on
   `descriptor_states_a_shape` (fixed point and `_callsite_specialized_shell_type`). (ccba325c)
10. `expand_reduction 'axis'` "REVISE without a changed source" refusal. Cause: `replace()` posted
    PROVEN_LITERAL not for `None`, and read a missing row as `None == None`. Fix: post for `None`, and post when
    `book.latest_ref(PROVEN_LITERAL, row)` is None. (ccba325c)
11. `_publish_callsite_return_members` wrote a raw undeclared page. Fix (D1): declared page
    `CALLSITE_RETURN_MEMBER`, posts DERIVED from the callee return value's cells; fold re-derivation posts on the same row. (51b4cebe)
12. Second reverse compile in one process refused (state leak: no book begun, lru_cached rule graph kept scopes).
    Fix (D2): `reverse_compile_book()` in `process_graph_autograd.py`, per-book rule-graph memo (lru_cache gone),
    both `llvm_training_runtime` entries run inside it. (51b4cebe)
13. Scalar-loss NaN grads: `_structured_output_descriptor` used the shared authored proof `('unbroadcast',1)`=(2,3)
    cemented by bw_mul copies over bw_add's correct (). Fix: skip the shared proof when the authored function has a
    `FORMAL_SHAPE_CONFLICT`. (51b4cebe)
14. "concorded source/SSA shape disagreement" for two unbroadcast copies. Cause: `fortran_c_shell.note_shape` keyed
    rows by the shared authored name, last writer won. Fix: a different graph stating a different shape marks the
    row polymorphic (sticky); the completed-module seam skips it. (51b4cebe)
15. Linear motion test NaN: shape vectors minted int32 by `tensor_ssa_lowering.int_vector` were widened to int64 by
    `ir_indexing._propagate_scalar_dtypes`; the kernel reads them as i32 ([2,2] -> [2,0], tmp never written).
    Fix: a Const keeps its declared integer width; only inferred ints widen. (ce587b82)

## Decisions and why

- Min/Max: option (a), every reader goes through the one spelling table (documented as the single table); not a per-route bridge.
- Per-callsite specialization with declared argument roles (decision (b) refined by (i)): do not literal-fold a formal at
  catalogue level when the caller passes a tensor/gradient there; the callsite copy keeps literal + descriptor, and its fold
  posts PROVEN_LITERAL DERIVED from the formal's cell and the planner_specialization row (itself derived from the caller's
  argument cell; traced by observation). Roles come from the adjoint builder's `argument_forward_sources`: index 0 "gradient",
  bound forward source "operand", constants "metadata". Reason: the first form ("has a `_tensor_descriptor`") was unreliable,
  because `_annotate_numeric_metadata` gives every motion const `tensor {shape ()}`; it blocked axis/keepdim/shape/target_shape.
  Undeclared (AST) calls fold as before.
- Integer-width rule: a declared Const dtype is its identity; only inferred integers widen to int64 (the Sep-26 rule beat the older 839a40d1 widening).
- One identity book per reverse compile (D2): opens at `obtain_graph_reverse` / `compile_native_graph_reverse` /
  `compile_native_training_schedule`, unless an enclosing reverse compile or a forward compile (sympy Derivative folding) owns the current book.
- Graph-native nodes with no AST are identified by declared op, not AST class (item 3).
- A decomposition that inserts the replacing instruction retires its own callsite marker (item 4).

## Still open

- `ssa_wasm_backend.py:257` tests the exact `{"Max","Min"}`; sympy fluid laws reach it as `maximum`, so
  `test_sympy_fluid_emits_and_runs_direct_repository_ssa_wasm` and the four-lane fluid test are expected to fail. Never run.
  Same one-table fix. The C module lane (`ssa_c_backend.py` ~4991/5027 `{"max","min"}`) takes planner-respelled SSA, left alone;
  Fortran and WebGPU already accept both spellings.
- `tensor_numel` (`_count`) decomposition in `fortran_c_shell.py` has the same marker-survives shape as eps; not observed failing, unchanged.
- `tests/test_precompile_to_ssa.py`: 13 fail / 91 pass; `test_ir_sequence_tables.py`: 3 fail / 36 pass. Identical failure sets at clean 51b4cebe
  (so not caused by this lane).
- The frame-linker NaN suspicion was wrong: no call-arity mismatch (root frame storage is wrapper-owned, dead formals), cause was item 15.
- Unclear in the log: the actual hunk-to-commit assignment is by commit stat only; one of the original hunks (item 8's
  `replace()` descriptor edit) was reverted as ineffective.

## Final gate (after ce587b82)

Probe max_rel 4.138e-16; `tests/test_llvm_training_runtime.py::test_graph_reverse_is_a_compiled_parametric_vjp` passes (~8-13 s; red
with KeyError 7 before item 3); `tests/test_orbital_transfer_compile.py` 4 passed / 7 xfailed; `tests/test_native_scalar_loss_adjoint.py`
2 passed in one process (two compiles); `test_linear_forward_loss_backward_is_one_parametric_graph_motion` passed; left*right VJP
`(0,1,3,2,7,8)`; engine_toy `test_orbital_actuation.py::test_throttles_are_clamped_to_their_declared_range` passed; fluid direct C test passed.

## Traps and gotchas

- The clean-HEAD failure of the linear motion and scalar-loss tests (ConcordanceRefusal at `expand_reduction 'axis'`) was
  pre-existing; (b) alone moved scalar-loss from a fast refusal to a non-termination that the refusal had been masking. Check
  each new fix against a baseline on a clean worktree (short path, e.g. Temp\wtb), never stash.
- Authored-name rows are shared by every callsite copy (shape_transformation_state, proven_shape_contract, `value_shape`): the
  recurring bug class here. Check for cross-copy collisions first (items 9, 13, 14).
- Reverse compiles share one book per process unless D2's `reverse_compile_book` owns it; the rule graph is rebuilt (~2.5 s) per new book.
- Every motion const carries `tensor {shape (), float64}` from `_annotate_numeric_metadata`; never infer "caller passes a tensor" from a descriptor.
- The probe's sympy reference raises "Cannot convert expression to float" if the actuation law gains new free symbols and `slice_laws` does not bind them.
- Bisecting 534a4941..de609156 is not possible (every commit raises RecursionError or the unscoped-backward ValueError); 534a4941 first broke the small VJP.
- Scratch scripts in a shared scratchpad get overwritten by other agents; use uniquely named files.
- Lowering takes ~235-340 s once backward calls are no longer dropped (it was ~37 s when they were silently lost); build a seconds-long
  sympy slice (e.g. `x**3*y + log(x)`, Max/clamp, left*right VJP) instead of the full probe.
- Other lanes' uncommitted edits shared the tree (`precompile_to_ssa.py`, `work_contract.py`, `llvm_dt_system.py`); the binary_value
  collision fix (`module_metadata["tensor_ssa_reference"]`, `process_graph_autograd.py` ~:2723) belongs to the planner lane; keep it.
