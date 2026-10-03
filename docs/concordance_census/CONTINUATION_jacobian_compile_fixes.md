# Continuation: compiled collocation Jacobian (Max/Min ops + backward-graph book scope)

Source of the task: `docs/DIFFERENTIATION_FEASIBILITY_2026-10-02.md` item 1,
probe `tools/compiler_probes/probe_collocation_jacobian.py jacobian --compile`
(commit 25257ac5). Base: turing 25257ac5.

## 2026-10-02 session start

Uncommitted at start (NOT this lane; belongs to CONTINUATION_name_arm_alias.md):
`src/compiler/precompile_to_ssa.py`, `src/compiler/work_contract.py`,
`tools/compiler_probes/probe_nested_inplace_arm.py`. Leave them out of this
lane's commit. No partial edits from a previous jacobian lane were present.

## 2026-10-02 Fix 1 applied (not yet probed)

`src/compiler/symbolic_process_graph.py` translation table (~:76-84):
`sympy.Min` -> `SympyProcessGraphRule("minimum")`, `sympy.Max` ->
`SympyProcessGraphRule("maximum")`. The existing left-associated pairwise
fold (~:1176) now tests `{"Add","Mul","minimum","maximum"}`. Reverse table
(`_CANONICAL_FUNCTIONS`) already accepted both spellings; graph->SSA spelling
`maximum`->`Max` is in `hierarchical_plan.TENSOR_OPERATION_SCALAR_SPELLING`.
No other `"Max"`/`"Min"` string checks in that file.

## 2026-10-02 Fix 1 REVERTED: stopped on a design question

Observed by reading (no runs): the sympy forward compile route carries the
graph op verbatim into SSA and then into a consumer that knows only the
scalar opcodes `Max`/`Min`:

- `symbolic_equation_compiler._compile_sympy_equations_uncached` (~:313)
  schedules the ingested graph with `ssa_builder.process_graph_to_ssa_instrs`,
  which copies `data["op"]` into `Instr.op` unchanged (ssa_builder.py ~:52).
- `native_package.piece_from_law` -> `vehicle_python_compilation.
  symbolic_abstract_tensor_source` -> `ssa_python_materializer.
  materialize_function_body`, whose `_BINARY_SPELLING` (:84-108) has
  `Max`/`Min` only; any other op reaches
  `MaterializationError("no Python form for 'maximum'")` (:633).
- `engine_toy/orbital_actuation.py:167` (the jumper throttle clamp
  `sp.Min(sp.Max(...))`) is compiled by exactly this route.

So with the table edit the graph-native backward works but every sympy law
with Min/Max stops compiling forward. Bridging `maximum`->`Max` for that
route is a choice the code does not settle: the documented one table is
`hierarchical_plan.TENSOR_OPERATION_SCALAR_SPELLING` (:90, "anything
interpreting repository SSA must read them through this one table"); the
reference evaluator reads it, the materializer does not. Candidate sites:
(a) the materializer reads its binary ops through that table, or
(b) the symbolic compiler respells scheduled ops through it.
Question raised to the user; nothing else edited. Tree restored to base
for this lane.

Fix 2 not started. Open point found while reading, for the next agent:
the probe's forward graph is sympy-ingested and has NO
`ingestion_value_scope` (`ingest_sympy_expression` mints none; only
`graph_express2.py:5247` does). So a backward node posted NOVEL needs a
forward operand cell that does not exist. `book.post` Novel requires
`len(operands) == transform.arity` and every operand an existing cell
(identity_concordance.py:3435-3449). Either sympy ingestion mints a scope
(rows `Unsourced(SYNTHESIZED_NO_SOURCE)`, as `ensure_node` does for a
span-less node) or the adjoint rows fall back to Unsourced when the forward
graph is unscoped.

## 2026-10-02 Coordinator: user approved option (a); this agent owns FIX 2 only

Fix 1 (Max/Min mapping + materializer reading TENSOR_OPERATION_SCALAR_SPELLING)
is another agent's; do not edit those regions. Do NOT commit (main session
integrates). Fix 2 scope: sympy ingestion mints its own ingestion_value_scope
and posts per-node rows from the law's equation/subexpression; backward graph
gets its own scope with NOVEL adjoint rows (operand = forward cell); fused
motion rows DERIVED from the copied cell.

Audit baseline (tools/audit_identity_concordance.py, all cases, shared tree
with other lanes' edits to precompile_to_ssa.py / ssa_python_materializer.py):
view 493 rows 0 findings (unsourced-fact x57); toplevel 322 rows 1 finding
(operand-never-written x1, unsourced-fact x46, unsourced-identity x10);
energy 490/0 (x48, x10); controller 530/1 (onw x1, x52, x1);
controller_untyped 533/5 (onw x5, x52, x1); mapping 24/0 (x15);
oscillator 331/0 (x28).

## 2026-10-02 Fix 2 edits written (not yet run)

- `src/compiler/concordance_declarations.py` (end, before `__all__`): new
  section "SymPy ingestion and graph-native differentiation": stages
  ADJOINT, FORWARD_LOSS_BACKWARD_FUSION; transform ADJOINT_OF (arity 1);
  reason FORWARD_GRAPH_UNSCOPED; pages `symbolic_ingested_expression`
  (scope, expression) SymbolicExpressionFact(output) and
  `symbolic_subexpression` (scope, expression, path) SymbolicSubexpressionFact(head).
- `src/compiler/symbolic_process_graph.py`: class
  `_SympyIngestionProvenance` (before `ingest_sympy_expression`) mints
  `ingestion:sympy` scope (reuses an existing one), posts expression rows
  (DERIVED from caller's equation cell, else NOVEL(INGEST_SOURCE) root) and
  one subexpression row per distinct authored subexpression (path of first
  occurrence, DERIVED from parent cell). `ingest_sympy_expression` gains
  private `_provenance=`; `make_node` posts each node's INGESTION_VALUE row at
  birth (DERIVED from its authored subexpression's occurrence cells, else the
  innermost authored context; envelope Tuple Unsourced(SYNTHESIZED_NO_SOURCE));
  `add_node` became a context-pushing wrapper over `ingest_value` (the old
  body); Derivative fold passes `source_cells=(node_identity_cell(backward,
  id),)`. `ingest_sympy_expressions` gains `expression_sources=`.
- `src/compiler/symbolic_equation_compiler.py`: `_symbolic_program_key`,
  `_post_symbolic_program` (returns output cells) factored out of
  `_post_symbolic_outputs`; the uncached compile posts equations+outputs
  BEFORE ingestion and passes the output cells as `expression_sources`.
- `src/common/tensors/topological_reducer.py`: `node_ingestion_scopes(graph)`
  extracted from `node_identity_cell` (same tuple, same order).
- `src/compiler/process_graph_autograd.py`: `_graph_node_cell`,
  `_post_copied_node` helpers; `_AdjointBuilder.__init__` mints
  `ingestion:adjoint` (stage ADJOINT) and `add` posts NOVEL(ADJOINT_OF,
  (forward cell,)) per node (raises if a node names no source_forward_id);
  `fuse_forward_loss_backward` mints `ingestion:training_motion` and posts
  every copied node DERIVED from its forward/backward cell.

## 2026-10-02 Fix 2 verified (rows + audit); not committed (main session integrates)

Targeted check (scratch script, not in repo): probe `_ingest_slice()` ->
`differentiate_process_graph(all 7 outputs, wrt all 19)` ->
`fuse_forward_loss_backward(unit_loss_seed=False)`. Fix 1 was already in
the shared tree (forward ops: maximum 6, minimum 6). No refusals, 9.4 s.
- forward `('ingestion:sympy', 0)`: 154/154 node rows DERIVED; plus the
  removed Tuple envelope's row Unsourced(SYNTHESIZED_NO_SOURCE) (1).
  symbolic_ingested_expression 7 rows NOVEL(INGEST_SOURCE) (the probe has no
  equations); symbolic_subexpression 323 rows DERIVED.
- backward `('ingestion:adjoint', 0)`: 821/821 NOVEL(ADJOINT_OF, forward cell).
- motion `('ingestion:training_motion', 0)`: 828/828 DERIVED.
Audit (all 7 cases) after edits: identical to the baseline above in rows,
findings and unsourced counts.

Not run by this lane (main session): `probe_collocation_jacobian.py jacobian
--compile`, tests/test_orbital_transfer_compile.py. The compile_sympy_equations
path (equation rows now posted before ingestion, output cells passed as
expression_sources) is verified only by reading; the orbital test exercises it.

## 2026-10-02 Fix 1 re-applied under approved option (a)

The user approved option (a): every SSA reader resolves the op through
`hierarchical_plan.TENSOR_OPERATION_SCALAR_SPELLING`, whose spelling is
`TABLE.get(op.casefold(), op)`, the same as `ssa_reference_evaluator._operation_name`.
The table already maps `maximum`->`Max` and `minimum`->`Min`. No table edit.

- `symbolic_process_graph.py`: `sympy.Min`/`sympy.Max` now map to graph ops
  `minimum`/`maximum`. The existing pairwise left fold (~:1170) now tests those
  names. A 3-argument Max ingests as two `maximum` nodes.
- `ssa_python_materializer.py` `_BodyMaterializer.step`: the binary and unary
  lookups read `scalar_operation` through the table. An op with no scalar form
  keeps its own name for the tensor-catalogue fallback. Without that guard,
  `isfinite`/`isnan`/`isinf` would respell to names absent from `_UNARY_SPELLING`
  and lose their catalogue route.
- `ssa_c_backend.py` `emit_ssa_function_to_c` (the direct scalar lane): `op` is
  respelled through the table at the top of the instruction loop.
- `ssa_llvm_backend.py` `emit_ssa_function_to_llvm`: `operation` is respelled
  right before `scalar_likeness`. Tensor work in this emitter arrives as `Call`
  plus a `tensor_operation` attribute, so `_TENSOR` dispatch is unaffected.

Verified:
- engine_toy `test_orbital_actuation.py::test_throttles_are_clamped_to_their_declared_range`
  (jumper Min(Max) clamp, piece_from_law route): 1 passed.
- `tests/test_orbital_transfer_compile.py`: 4 passed / 7 xfailed (baseline),
  before and after the backend edits.
- Direct `compile_sympy_equations` of `Min(Max(t,lo,c),hi)`: SSA
  `maximum, maximum, minimum`. C is complete (`fmax`/`fmax`/`fmin`), and LLVM has no
  shortfalls (`llvm.maxnum` x2, `llvm.minnum`).
- `test_symbolic_fluid_direct_backends.py::test_sympy_fluid_emits_and_runs_direct_repository_ssa_c`
  (fluid law with Max, compiled and run): 1 passed.

Open, not touched (out of scope):
- WASM direct lane `ssa_wasm_backend.py:257` tests the exact `{"Max","Min"}`.
  The sympy fluid law (which uses Max) now reaches it as `maximum`, so
  `test_sympy_fluid_emits_and_runs_direct_repository_ssa_wasm` and the four-lane
  fluid test are expected to fail there. This was not run. It is the same one-table fix.
- The C module lane (`ssa_c_backend.py` ~4991/5027 `{"max","min"}`) consumes
  AST-lane SSA that the planner already respells, so it was left alone.
- Fortran (`maximum` key at :133) and WebGPU (:34) already accept both spellings.

## 2026-10-03 KeyError 806 lane: findings before the reset (no source edits)

This lane edited NO source file. `git diff` hunks in process_graph_autograd.py
are Fix 2's only (adjoint/fusion rows, :1502-2390); nothing in
`lower_training_motion_to_repository_ssa`. Scratch scripts only
(scratchpad trace_seed.py / trace_unit.py / trace_at.py).

Observed (scratch hook after `lower_training_motion_to_repository_ssa`, probe slice):
- Seeds 154..160 are `input` nodes, each consumed only by one `Call bw_add`
  (393..399). Root SSA function args = the 19 variables + ~120 BACKWARD value
  ids (475, 495, ...), no seed. Seeds have no def and no use anywhere in the module.
- Root metadata: `settled_nonlive_structural_shortfalls` lists every backward
  `Call bw_*` (393.., 414..) as `call-result-unavailable` /
  `absent_from_final_ssa`; `structural_output_shortfalls` and
  `unresolved_required_source_values` list exactly the 12 missing grads
  (806, 809, ... 448), all `Indexed(base=Call bw_*, index=const)`,
  reason `basic-index-contract`. `value_aliases` is empty: nothing is renamed.
- So the 12 grads are DROPPED, not aliased, and the 7 "present" grads
  (add of two Indexed) are computed from Indexed values that became formals.
- NOT seed-specific: the same slice with 1 output gives identical args,
  shortfalls and settled calls under unit_loss_seed=True and False.
- The tiny AbstractTensor VJP of tests/test_llvm_training_runtime.py
  (left*right) shows the same shape at this stage: args [0,1], seed 3 absent,
  Call 6 settled absent, grads 7/8 in structural_output_shortfalls, Ret [2].
  Whether that test's artifact recovers them at LLVM emission is NOT yet
  observed; next step is to read its buffer_order after emit/compile.

## 2026-10-03 The small working-lane VJP is ALSO broken in the current tree

`compile_native_graph_reverse(left*right, unit_output_seed=False)` (the body
of tests/test_llvm_training_runtime.py::test_graph_reverse_is_a_compiled_parametric_vjp),
run from a scratch script in the current shared tree: artifact
buffer_order = (0, 1, 2). Seed 3 and grads 7/8 are absent, so that test
cannot pass here. Next: the same script on a clean worktree of HEAD, to
tell whether the uncommitted lanes caused it or it predates them.

## 2026-10-03 Baseline fd68458d (pre-lane) for the small VJP

Worktree C:\Users\alber\AppData\Local\Temp\wtb at fd68458d (the commit before
de609156, which committed Fix 1 + Fix 2). The same left*right VJP script fails
earlier there: `node_identity_cell` raises "node 3 has no ingestion scope"
(backward graph unscoped, the gap Fix 2 closed). So at fd68458d the
backward lowering did not reach SSA at all; the dropped-Call shape is not
a regression of the Fix 1/Fix 2 lanes in any observable way. Next: find the last commit
where tests/test_llvm_training_runtime.py passed.

## 2026-10-03 Regression window for the small VJP

- bc5b347a (2026-08-20, the commit that added test_llvm_training_runtime):
  buffer_order (0, 1, 3, 2, 7, 8); grads 7/8 = [55, 91] / [22, 39]. CORRECT.
- 90fc38b0: RecursionError in apply_precision_pipeline (fixed in de609156).
- fd68458d: "node 3 has no ingestion scope" (fixed in de609156).
- de609156 + tree: buffer_order (0, 1, 2); seed and grads dropped.
So the dropped bw_* Call is a regression after bc5b347a, masked by the two
earlier raises. Bisecting next (bad = 7/8 absent; raises = skip).

## 2026-10-03 Bisect attempt 1 contaminated (shared scratchpad)

Another agent shares the session scratchpad and overwrote my bisect_vjp.sh
mid-run with its own (pytest of test_graph_reverse_is_a_compiled_parametric_vjp
in worktree ...\Temp\wtc). Only my first two steps are mine:
ce1b7965 GOOD (buffer_order (0,1,3,2,7,8)), 87b37d5d RecursionError (skip).
Everything after is that agent's verdict on ITS worktree; discarded.
Redoing with uniquely named files (kx806_*).

## 2026-10-03 Bisect (clean, kx806_*): first bad = 534a4941

Good->bad criterion: grads 7/8 present in buffer_order with [55,91]/[22,39].
First bad commit 534a4941 "Fix concordant numeric solve compilation":
buffer_order (0, 1, 3, 2) (seed kept, grads 7/8 dropped). Parent e39ff7cf is GOOD.
Note the current tree also drops the seed ((0, 1, 2)); possibly a second,
later change. Reading 534a4941 next.

## 2026-10-03 What 534a4941 broke (small VJP, lowering stage)

Same scratch lowering (kx806_lower.py) at parent e39ff7cf vs 534a4941:
- e39ff7cf: root `Call bw_mul -> 6`; `GEP/Load -> 7`, `GEP/Load -> 8`
  (graph ids of the two Indexed gradient nodes); Ret [2, 7, 8];
  semantic_output_ids (2, 7, 8).
- 534a4941: same Call -> 6, but the Loads mint FRESH ids 10 and 12;
  semantic_output_ids (2, 10, 12); `outputs` still says grad_0=7, grad_1=8,
  so the Ret filter in lower_training_motion_to_repository_ssa
  (public_output_ids) keeps only [2]. Grads are not aliased: they are
  re-keyed and then filtered out.
Next: compare the current tree at the same stage, then find the writer of 10/12.

## 2026-10-03 Current tree has a SECOND, later break (small VJP)

Current tree (de609156 + shared edits), same scratch: the root has NO
`Call bw_mul` at all; no bw_mul/unbroadcast functions in the module; args
[0, 1] (seed 3 gone); grads 7/8 in structural_output_shortfalls +
unresolved_required_source_values (Indexed of Call 6); Call 6 settled
`call-result-unavailable` / `absent_from_final_ssa`. This is the probe's
shape exactly (seeds and 12 Indexed grads gone). Bisecting 534a4941..de609156
on "root contains Call bw_mul" next.

## 2026-10-03 Bisect 2 abandoned

534a4941..de609156 on "root contains Call bw_mul": every probed commit
skipped (they raise: RecursionError in apply_precision_pipeline, or the
unscoped-backward ValueError), so bisect cannot localize the second break.
Switching to direct observation in the current tree: why the planner emits
no Call for motion node 6 (`Call bw_mul`, callee_ref -> registry graph).

## 2026-10-03 Suspect for the vanished Call (read, not yet observed)

glsl_deployment_strategy.py `plan_callsites` (~:24682-24694) skips any node
whose `expr_obj` is not an `ast.Call` before reading `callee_ref`. Added by
42a689a2 (2026-09-29, inside the skipped bisect window) to tell a resolved
method SELECTOR from its invocation. The adjoint builder creates every
backward `Call bw_*` with `expr_obj=None` (process_graph_autograd.py ~:1603),
so no callsite shell is planned for it -> no bw_* function -> Call result
unavailable -> Indexed grads unresolved, seeds unused. Observing next with a
scratch hook (no source edit).

## 2026-10-03 CONFIRMED: the expr_obj gate drops the backward Call

Scratch kx806_gate.py, current tree, small VJP: without hook -> FUNCS has no
bw_mul, ARGS [0, 1], RET [2]. With ONLY `G.nodes[6]["expr_obj"] = ast.Call(...)`
set in the scratch before lowering -> bw_mul + unbroadcast specialized
functions appear, ARGS [0, 1, 3] (seed back), RET [2, 7, 8] (grads back, ids
preserved; the 534a4941 10/12 re-key does not reproduce in the current tree).
Root cause: plan_callsites identifies an activation by the AST class of
expr_obj; a graph-native Call (adjoint builder, op "Call" + callee_ref,
expr_obj None) has no AST, so it is never planned.

## 2026-10-03 Fix chosen (plan_callsites gate), reasoning

The selector the gate excludes is an AST-built `ast.Attribute` accessor
(topological_reducer ~:11138, `accessor_kind="method"`); it always carries
its AST. An AST graph can also hold op "call" nodes that are NOT activations
(short-circuited boolop operands, fortran_c_shell ~:21046), so the AST rule
must stay for AST nodes. A graph-native node (expr_obj None: adjoint builder,
fused motion) has no AST; its declared op "Call" + callee_ref is its only
declaration of an activation. Rule: AST node -> ast.Call decides; node with
no authored AST -> declared op "call" decides.

## 2026-10-03 Fix applied: glsl_deployment_strategy.py plan_callsites gate

Edited the `expr_obj` gate (~:24691): expr_obj None -> admit iff declared op
casefolds to "call"; otherwise unchanged (`ast.Call` required). Comment above
names the program, wrong value and rule. Running the small VJP scratch next.

## 2026-10-03 Small VJP green after the fix

left*right VJP (scratch trace_at2.py, current tree + fix): buffer_order
(0, 1, 3, 2, 7, 8); grads 7 = [55, 91], 8 = [22, 39] (= seed*y, seed*x). Running
the repro next.

## 2026-10-03 Repro after the fix: next wall

`probe_collocation_jacobian.py jacobian --compile`: forward 188 nodes (the
shared tree's ingestion changed since the 154-node runs; not this lane),
7/7 differentiated, motion 996 nodes, seeds {87:188, ..., 187:194}. Lowering
now raises in lower_tensor_calls_to_repository_ssa
(tensor_ssa_lowering.py:5745): "repository SSA function collision for
'binary_value'". The bw_* callees now lower, so a second binary_value
reaches the module. Reading that site next.

## 2026-10-03 binary_value collision: reading

`_class_surface_ssa_program` deep-copies tensor_ssa_reference (c9607d25,
"own one copy for the entire source program") and already runs the late
`lower_tensor_calls_to_repository_ssa` on the affected slice with that copy
(fortran_c_shell.py ~:41373), linking the COPY's binary_value into the module.
`lower_training_motion_to_repository_ssa` (process_graph_autograd.py:2723-2727)
then runs `lower_tensor_calls_to_repository_ssa(module, reference)` again on
the whole module with the UNCOPIED reference; any linked root whose closure
reaches binary_value finds the copy's object -> "collision". Next: observe
which roots it links (hook dependency_closure).

## 2026-10-03 binary_value collision OBSERVED

Scratch kx806_collide.py (hooks on lower_tensor_calls_to_repository_ssa and
SSATensorCodeReference.dependency_closure), probe slice:
- every pass inside `_class_surface_ssa_program` (precompile_to_ssa:16887 x12,
  fortran_c_shell:41373) uses reference object A (the deepcopy).
- process_graph_autograd.py:2727 uses object B (`tensor_ssa_reference or
  c_backend_repository_ssa_reference()`, the uncopied cache). The module
  already holds A's binary_value/binary_double (`is` B's: False, False).
- remaining tensor work entering :2727: one `extent` Call in
  training_motion__unbroadcast__specialized_*; B's closure roots
  ('binary_double', 'binary_scalar_double') -> collision on binary_value.
One program, two reference identities. Checking other post-surface callers.

## 2026-10-03 Collision fixed by ANOTHER lane (not this one), concurrently

While I read, the shared tree gained (not my edit): fortran_c_shell.py ~:16590
records `module_metadata["tensor_ssa_reference"]` = the surface's own copy,
and process_graph_autograd.py ~:2723 links the remaining tensor calls from
`module.metadata["tensor_ssa_reference"]` (raises if absent). Its comment
cites pow-using graph-reverse motions. That is the same one-identity fix this
lane would make; left as is. This lane's only source edit remains the
plan_callsites gate in glsl_deployment_strategy.py. Rerunning the repro.

## 2026-10-03 Repro with both fixes: lowering complete, LLVM shortfalls

Repro: lowered (shortfalls (), outputs 26) in 340 s (was ~37 s when the
backward calls were silently dropped). LLVM emission then reports 4
shortfalls "Call: operation has no repository LLVM emission" in
training_motion__unbroadcast__specialized_*, bw_pow__specialized_* (x2),
bw_log__specialized_*. compile_artifact refuses. Building a seconds-long
sympy slice (pow + log) to read those Calls next.

## 2026-10-03 Seconds-long repro for the LLVM shortfalls

Scratch kx806_powlog.py: sympy `x**3*y + log(x)`, same pipeline as the probe
(ingest -> differentiate -> fuse(unit_loss_seed=False) -> lower -> emit LLVM),
9 s. Same 3 shortfall functions:
- bw_pow__specialized_*: `Call [] -> 3 {callee: '__plan_callsite_3__',
  plan_callsite_marker: True}` (an unlinked nested callsite marker).
- bw_log__specialized_*: `Call [] -> 2 {callee: '__plan_callsite_2__', marker}`.
- unbroadcast__specialized_*: `Call [64] -> int {tensor_operation: 'extent',
  extent_kind: 'dim', axis: 0, binding: 'iterable_extent'}` (unlowered extent).

## 2026-10-03 pow/log shortfalls are not a regression

kx806_powlog at e39ff7cf (wtb) fails earlier: prepare_graph_precompile raises
CompilationSubdivisionRequired (unbroadcast loop_node=66,
'unresolved-loop-bound'). So bw_pow/bw_log/unbroadcast never lowered on this
path; these are frontier, not regressions. Checking the other lanes'
continuation docs (one is fixing the pow motion concurrently) before editing.

Note for the orbital step-4 lane (CONTINUATION_orbital_step4_collocation.md,
which deferred its tiny-VJP KeyError 7 here): with this lane's plan_callsites
gate fix the same left*right VJP gives buffer_order (0, 1, 3, 2, 7, 8) and the
right gradients in the current tree. Its 534a4941 member-publication reading
is upstream of the gate; with the gate fixed the Indexed members are planned.

## 2026-10-03 eps() marker: the existing decomposition runs, the marker survives

bw_log graph: node 2 `Call` (ast.Call, callee_ref 8 -> `eps`, 1-node graph).
fortran_c_shell.py ~:30690-30750 already decomposes an unresolved `eps`
source-call record into `Const 1e-12 {structural_operation: backward_epsilon}`
(seen in the emitted bw_pow) with resolution "decomposed". But the
`Call [] -> 3 {callee: '__plan_callsite_3__', plan_callsite_marker}` that
also defines id 3 is still in the function. Reading how decomposed records
retire their marker next.

## 2026-10-03 Why the eps marker survives (read)

Markers are retired only for records with resolution "native_call"
(frame loop ~:35933 and post-frame cleanup ~:35995), or unrecorded+dead.
The eps and `_count` (tensor_numel) decompositions (~:30690, ~:30790) insert
a Const that DEFINES the record's caller result id and mark the record
"decomposed", but leave the `__plan_callsite_N__` marker that defines the
same id. Two definitions of one id; LLVM cannot emit the marker. Rule to
apply: a decomposition that inserts the replacing instruction IS the
executable occurrence, so it retires that callsite's marker at the same
point (exact plan_callsite_id), as the native_call path does.

## 2026-10-03 Fix applied: eps decomposition retires its marker

fortran_c_shell.py eps decomposition (~:30740): after the Const is inserted,
remove the `plan_callsite_marker` instruction whose plan_callsite_id equals
the record's callsite_id. tensor_numel (`_count`) has the same shape but was
not observed failing; left unchanged (open). Re-running kx806_powlog log next.

## 2026-10-03 eps fix verified; one shortfall left (unbroadcast extent)

kx806_powlog `log(x)*y`: lowered 5.6 s, LLVM shortfalls (). `x**3*y`: only
training_motion__unbroadcast__specialized_* remains:
`Call [64] -> int {tensor_operation: 'extent', extent_kind: 'dim', axis: 0,
binding: 'iterable_extent', source_value_id: 64}` (the `for ... in
enumerate(zip(g_shape, t_shape))` loop bound). Reading next.

## 2026-10-03 unbroadcast extent: observations so far

x**3*y motion: const exponent node 2 carries tensor {shape: (), dtype float64}
(after _annotate_numeric_metadata; the sympy forward graph itself has tensor {}).
bw_pow calls unbroadcast twice: gx -> unbroadcast__specialized_bcef747c8345
(folds, fine), gp = unbroadcast(..., p.shape) -> __specialized_bb51b1e18d0a,
whose target arg (bw_pow value 13) is dtype 'unknown' and whose shape loop
keeps the unlowerable iterable `extent`. Inferred, not yet observed: the
callsite specialized p to the literal 3, so `p.shape` has no shape fact.

## 2026-10-03 OBSERVED: p.shape loses its fact because p is specialized as a literal

Hook on glsl_deployment_strategy._callsite_specialized_shell_type (kx806_powmeta.py):
- motion node 13 -> bw_pow: planner_specializations {'p': 3} AND
  planner_tensor_descriptors p {shape (), float64, rank 0}.
- bw_pow node 11 (x.shape) -> unbroadcast specs {'target_shape': ()}  (folds).
- bw_pow node 20 (p.shape) -> unbroadcast specs None                  (no fold).
One value p, two records: literal 3 and scalar tensor (). The `.shape`
resolution reads one and misses. Finding the reader next.

## 2026-10-03 OBSERVED: the specialized Input p becomes a bare Constant 3

Hook at bw_pow's nested callsites (kx806_powmeta2.py): node 11's target arg
is Constant () (x.shape folded). Node 20's target arg is GetAttr 13 `.shape`
whose owner, node 1 (formerly Input p), is now `Constant 3` with NO tensor
descriptor (`_tensor_descriptor` -> None), although the shell's
planner_tensor_descriptors still say p = {shape (), float64, rank 0}. The
literal rewrite dropped the value's tensor record. Finding that rewrite.

## 2026-10-03 Where the descriptor is dropped (read)

glsl_deployment_strategy.py `replace()` inside the callsite literal fold
(~:21640-21705) rewrites a folded node to `Constant` and REPLACES its
attributes with the aggregate-ledger keys + value, dropping `binding_name`.
`_tensor_descriptor` finds a formal's descriptor through that binding_name
(`localized_formal`, planner_tensor_descriptors); a Constant with an int
literal and no `tensor.shape` gets none (`_tensor_descriptor_rule` ~:19323).
So `.shape` of the folded formal is unresolved. Fix: when the folded node is
an Input whose binding has a planner tensor descriptor, the Constant keeps
that descriptor as its `tensor` (one value: literal 3 AND scalar tensor ()).

## 2026-10-03 Fix applied: folded formal keeps its tensor descriptor

glsl_deployment_strategy.py `replace()` (callsite literal fold): an Input
folded to a Constant takes `planner_tensor_descriptors[binding_name]` as its
`tensor` when it has no shape of its own. Running kx806_powlog pow next.

## 2026-10-03 Coordinator relay (step-4 planner lane), with this lane's status

1. Relay: test_graph_reverse_is_a_compiled_parametric_vjp is red at clean HEAD
   (KeyError 7, unit seed too); first bad 534a4941; only its
   glsl_deployment_strategy.py breaks it (_propagate_callsite_tensor_specializations
   / _publish_callsite_return_members); the backward Call is published as an
   aggregate with the Indexed grads as members; publication used to be gated on
   "call has no tensor descriptor". Unverified link: that keeps grads out of
   region outputs.
   Status here: in the CURRENT tree the decisive drop is later and different:
   plan_callsites skipped the Call because expr_obj is not ast.Call (42a689a2),
   so no callee shell existed at all. With the gate fix (this lane) the same
   left*right VJP gives buffer_order (0,1,3,2,7,8) and grads [55,91]/[22,39]
   WITH the aggregate publication still in place. So publication is not what
   drops the grads now. The 8 s test itself is still to be run (below).
2. Relay: the binary_value collision fix (fortran_c_shell records
   `module_metadata["tensor_ssa_reference"]`; process_graph_autograd links
   from it) is the planner lane's. KEEP. This lane did not write it.
3. Relay: bw_pow `__plan_callsite_N__` marker and unbroadcast `extent`.
   - Marker: THIS LANE fixed it in fortran_c_shell.py (eps decomposition
     retires its marker). Verified: log(x)*y LLVM shortfalls ().
   - extent: this lane is on it. Root observed: the callsite literal fold
     turns bw_pow's formal p into `Constant 3` and drops its descriptor, so
     `p.shape` is unresolved. First fix (folded Constant keeps the formal's
     tensor descriptor) applied in glsl_deployment_strategy.py replace();
     the x**3*y slice STILL shows the extent -> investigating further.
4. Relay: a second compile in one process is refused with concordance
   disagreements (state leak between compiles). Not investigated here; open.

## 2026-10-03 replace() descriptor fix had no effect: REVERTED

Hook on _fold_literal_source_cells (kx806_powmeta4.py): bw_pow's formal p is
folded with planner_specializations {'p': 3} and planner_tensor_descriptors
None -- i.e. on the shared catalogue graph (literal agreed by
_propagate_callsite_planner_specializations), BEFORE any callsite copy
exists. The callsite copy (_callsite_specialized_shell_type ~:24092) is then
extracted from an already-folded original: p is `Constant 3`, no binding,
no tensor, and `_apply_callsite_tensor_descriptors` cannot reach it. The
replace() edit only acts when descriptors are already on the graph, so it
changed nothing here; reverted (glsl_deployment_strategy.py replace() is
back to its prior text). Open design question raised to the coordinator.

## 2026-10-03 Baseline check: test_linear_forward_loss_backward_is_one_parametric_graph_motion

Current tree (+ gate fix + eps-marker fix + other lanes' edits): 1 FAILED in
10 s: ConcordanceRefusal posting planner_specialization
(('lexical_reads:expand_reduction', 0), 'axis'): "REVISE without a changed
source". Baselining on wtb at clean de609156, then de609156 + gate hunk only.
Clean de609156 (wtb): the SAME ConcordanceRefusal, 1 failed. Pre-existing at
HEAD; not caused by this lane. (Related to the coordinator's item 4? both are
planner_specialization/book refusals; not investigated.)
Current tree: tests/test_orbital_transfer_compile.py 4 passed / 7 xfailed (30 s), baseline matches.
Current tree: tests/test_llvm_training_runtime.py::test_graph_reverse_is_a_compiled_parametric_vjp
1 PASSED (18 s); it is red (KeyError 7) at clean de609156 per the step-4 lane.

## 2026-10-03 STATUS (stopped on one design question)

Green: small explicit-seed VJP; log backward; orbital 4p/7xf; fast VJP test.
Red: probe repro still cannot print RESULT: every Pow in the slice reaches
unbroadcast(..., p.shape) with p folded to a bare Constant at catalogue level.
Pre-existing red: test_linear_forward_loss_backward_is_one_parametric_graph_motion
(ConcordanceRefusal at clean HEAD too).
Source edits by this lane: glsl_deployment_strategy.py plan_callsites gate;
fortran_c_shell.py eps decomposition retires its marker. Nothing else.
QUESTION: when _propagate_callsite_planner_specializations folds a formal to an
agreed literal on the SHARED catalogue graph (bw_pow's p = 3), where should
that formal's tensor fact (every caller passes a shape-() float64 tensor)
live: (a) the agreeing callers' descriptors are recorded with the literal and
the fold gives the Constant that `tensor`; or (b) a formal with a tensor
descriptor at its callsites is not literal-folded at catalogue level (it stays
a formal; the callsite copy keeps literal + descriptor)?

## 2026-10-03 Coordinator decision: (b)

Do not literal-fold a formal at catalogue level when its callers pass it with
a tensor descriptor; the per-callsite copy keeps literal + descriptor. A
per-callsite fold that morphs an identity posts its edge on the book (DERIVED
from the formal's cell and the caller's argument cell). Implementing in
_propagate_callsite_planner_specializations.

## 2026-10-03 Decision (b) applied

- concordance_declarations.py: reason SPECIALIZATION_TENSOR_ARGUMENT.
- glsl_deployment_strategy.py _propagate_callsite_planner_specializations:
  a static literal argument whose caller value has a `_tensor_descriptor`
  contributes `tensor_argument`; any such contribution posts
  Unresolved(SPECIALIZATION_TENSOR_ARGUMENT) and the catalogue formal is not
  folded. The callsite copy still specializes (literal + descriptor), and its
  fold posts PROVEN_LITERAL DERIVED from the fold's literal source cells (the
  existing `replace()` path). Running x**3*y next.
x**3*y slice: lowered 7.2 s, LLVM shortfalls (). Running the probe repro next.
Probe: lowering + LLVM + native build OK; 'native rows evaluated' at 334 s. Then the probe's own sympy REFERENCE (J_sym.evalf -> float) raises 'Cannot convert expression to float'. Inspecting the reference next.

## 2026-10-03 Probe fixture drift (not a compiler issue)

The reference Jacobian is non-numeric because the actuation law gained
attitude_{ab} and propellant_supply (engine_toy 861ccb4 step 7 + uncommitted
edits); slice_laws never bound them (forward inputs 29 vs 19 variables; the
native run fed them zeros). Probe edit: close() binds attitude = identity
(the library's `actuation_matrix(attitude=None)` default) and
propellant_supply = 1 (documented "1 when nothing ... is lit"/tank not
limiting). Rerunning the probe.

## 2026-10-03 Probe now runs end to end: RESULT max_rel 1.0

Forward 154 nodes / 19 inputs again; lowered (231 s), LLVM shortfalls (),
native built; forward native vs sympy max rel 2.39e-15. Jacobian nnz 55,
median rel 1.36e-16, max rel 1.0: worst row 0 col u0_0 native -0 vs ref -1e6.
The throttle columns (through the Min(Max(u,lo),hi) clamp) read 0. Making
a seconds-long clamp slice next.
Clamp slice kx806_clamp.py max: Max(0,u)*y at u=0.3,y=2: native grad_u 0.0 vs ref 2.0 (grad_y 0.3 correct). Reading bw_maximum.

## 2026-10-03 bw_maximum: where() result typed int64

kx806_clamp2.py (Max(0,u)*y): bw_maximum's planned regions compute
`where(x<y, 0, 0.5)` as where_double(...) then `Cast -> int64` (value 8), and
`where(x>y, 1, <that>)` -> Cast int64 (9); likewise 15/16 for the y branch;
region_5 then does `Mul [g(float64), 16(int64)]`. 0.5 truncates to 0 and the
float x int Mul is suspect. The where's result dtype is decided as int64 when
its arms are (int literal, float literal). Finding the dtype rule.

## 2026-10-03 Fix applied: where() result dtype = promotion of its value arms

glsl_deployment_strategy.py where descriptor (~:22381): dtype was "first
non-bool operand" (int64 for `where(c, 0, 0.5)`); now np.result_type of the
two value arms (same rule as the elementwise-binary descriptor). Rerunning
the clamp slice.
After where fix: Max slice grad_u = 1e-323 (int64 bits of 2 read as double), clamp slice 0.0. Dumping again.
Dump: where results now float64, but each where_double receives an int64
Const (0 or 1) as a value arm. tensor_ssa_lowering.py where (~:4955) passes an
operand whose shape equals the result's straight through ("conformed"); only
shape-mismatched scalar operands get the existing int->float64 constant
respelling. where_double reads the int64 1 as a double (4.9e-324), so
grad_u = 2 * 4.9e-324 = 1e-323. The kernel ABI is double-backed.

## 2026-10-03 Fix applied: where_double scalar literal arms are float64

tensor_ssa_lowering.py where lowering (~:4958): a shape-matched scalar
literal operand whose dtype is not float64 is respelled as its float64
constant (the rule the broadcast branch already applies to scalar literals).
Rerunning max/min/clamp slices.
max/min/clamp slices: grad_u 2.0, grad_y 0.3 = sympy reference. Running the probe.

## 2026-10-03 PROOF: RESULT jacobian max_rel 4.138e-16

Probe: lowered 235 s, LLVM shortfalls (), native built; forward max rel
2.39e-15; Jacobian (7,19) nnz 55, max rel 4.138e-16, median 0.0, structural
zeros 0. Worst: row 3 dt0 -137.35841359681334 vs -137.35841359681328.

## 2026-10-03 Checks after all fixes

- test_llvm_training_runtime.py::test_graph_reverse_is_a_compiled_parametric_vjp: 1 passed (13 s).
- tests/test_orbital_transfer_compile.py: 4 passed / 7 xfailed (24 s).
- left*right scratch: buffer_order (0,1,3,2,7,8), grads [55,91]/[22,39].
- tests/test_native_scalar_loss_adjoint.py (extra, not requested): did not
  finish in ~9 min; stopped. Checking current vs clean HEAD with a timeout.
Clean de609156: scalar-loss [True] FAILS in 10 s with the same expand_reduction 'axis' ConcordanceRefusal. Current tree: no result within 420 s (timeout).
Current tree: test_linear_forward_loss_backward_is_one_parametric_graph_motion
now fails DIFFERENTLY (12.7 s): ValueError "concorded source/SSA shape
disagreement for ('training_motion__unbroadcast__specialized_bc87d5b45e7b', 0):
ssa=(3, 2), concordance=(2, 2)". It gets past the HEAD ConcordanceRefusal.
Suspect: decision (b) as implemented is broader than p: _annotate_numeric_metadata
gives EVERY motion const a tensor {shape (), float64}, so structural literals
(axis, dim, ...) also lose their catalogue fold. Measuring the blast radius.
Blast radius (hook on _post_planner_specialization, linear test): rule (b)
blocked bw_mean{axis,g,keepdim}, expand_reduction{G,axis,keepdim,shape},
unbroadcast{target_shape}. "Has a _tensor_descriptor" is NOT a reliable
"caller passes a tensor": motion consts all get tensor {shape ()} from
_annotate_numeric_metadata, and numeric tuple literals get a descriptor from
the tensor_from_list literal rule. Need the declared distinction instead.

## 2026-10-03 STATUS + hunk map (for the main session's commit split)

GREEN: probe RESULT jacobian max_rel 4.138e-16 (median 0); fast VJP test
passed; orbital transfer 4p/7xf; left*right (0,1,3,2,7,8) [55,91]/[22,39].
REGRESSION RISK from decision (b) as implemented (`_tensor_descriptor(...) is
not None`): test_linear_forward_loss_backward... now fails with a shape
disagreement in an unbroadcast copy (was: ConcordanceRefusal at HEAD), and
test_native_scalar_loss_adjoint[True] no longer finishes in 420 s (HEAD: the
same refusal in 10 s). Observed: (b) blocks structural literals too (axis,
keepdim, shape, target_shape). The hang's cause is inferred, not traced.
Open question to the coordinator (below in the report).

Second-compile refusal: not investigated (not cheap). Closest repro I have:
clean de609156, `pytest tests/test_process_graph_autograd.py::test_linear_forward_loss_backward_is_one_parametric_graph_motion`
-> ConcordanceRefusal planner_specialization (('lexical_reads:expand_reduction', 0), 'axis')
"REVISE without a changed source" (10 s, single compile in the test).

Hunks owned by this lane:
- glsl_deployment_strategy.py: @@17070 (+import SPECIALIZATION_TENSOR_ARGUMENT),
  @@17078 (tensor_argument sentinel), @@17108 (tensor_argument contribution),
  @@17163 (Unresolved(SPECIALIZATION_TENSOR_ARGUMENT) post) = decision (b);
  @@22366/@@22369 = where dtype from value arms; @@24695 = plan_callsites gate.
- fortran_c_shell.py: @@30732 (eps decomposition retires its marker) ONLY.
  @@16592 (tensor_ssa_reference in module metadata) is the planner lane's.
- tensor_ssa_lowering.py: @@4959 (where_double scalar literal -> float64).
- concordance_declarations.py: @@371 (SPECIALIZATION_TENSOR_ARGUMENT).
- tools/compiler_probes/probe_collocation_jacobian.py: @@90, @@99 (attitude
  identity + propellant_supply 1 in slice_laws).
- process_graph_autograd.py @@2723: the planner lane's, not mine.
- precompile_to_ssa.py: not this lane.

## 2026-10-03 Coordinator decision: (i)

Committed by main session (all but precompile_to_ssa.py). Now: the adjoint
builder DECLARES per argument of each backward-rule Call its role (tensor
operand/gradient vs metadata constant); (b) reads that declaration, no
descriptor inference. Then: trace the per-callsite fold row's cells; rerun
probe, 13 s reverse test, orbital 4/7, linear motion test and
native_scalar_loss_adjoint[True] (with timeout).

## 2026-10-03 (i) applied

- process_graph_autograd.py registry_rule: Call attributes gain
  `argument_roles` (index 0 "gradient"; a bound forward source "operand";
  metadata/default constants "metadata"), from the builder's own
  argument_forward_sources.
- glsl_deployment_strategy.py (b): blocks the catalogue fold only when the
  call's `argument_roles[arg:i]` is gradient/operand; the descriptor test is
  gone. Undeclared (AST) calls and metadata args fold as at HEAD.
- concordance_declarations.py: reason comment updated.
Running the blast-radius hook on the linear test next.
Linear motion test with (i): only `bw_mean g` is blocked (g is a const
unit seed = gradient role); fails with the HEAD failure mode again
(ConcordanceRefusal expand_reduction 'axis', 8 s).
native_scalar_loss_adjoint[True] with (i): still no result in 300 s (timeout 124). Taking a stack dump at 150 s next.

## 2026-10-03 scalar-loss hang: attribution

Stack at 150 s (current tree): inside
_propagate_callsite_tensor_specializations (glsl ~:18069) ->
call_result_descriptor (~:17785) -> _fold_callsite_structural_values ->
_tensor_descriptor -> record_shape_transformation -> _post_or_unsourced.
Overlays on wtb at aa5f1aac (fast refusal there):
- + all non-glsl files of this lane: still the fast HEAD refusal.
- + current glsl with the plan_callsites gate reverted: HANG (>200 s).
- + also (b) reverted: back to the fast HEAD refusal (13 s).
So (b) is what moves this test off the refusal; the refusal was masking a
downstream non-termination (or extreme slowness) in the 534a4941 fixed point.
wtb restored to aa5f1aac clean. Sampling repeatedly to see if it progresses.

## 2026-10-03 scalar-loss "hang" = non-terminating fixed point (observed)

kx806_rounds.py (wraps _propagate_callsite_tensor_specializations' _progress):
rounds 3..184+ all report changed=True, return_members=2, every other
mutation 0, state_digest a24b0b2c18fbeee2 identical (repeats_round=3),
~2 s/round (book history scans grow). So the return-member publication
(534a4941's _publish_callsite_return_members) reports a change each round
without changing state. Reading it.
Repeating publication: bw_mul calls 29 and 30 (x*x: both operands one value) in scalar_loss_join_reverse; settled branch (incumbent leaves present, 2 descriptors) returns changed every round.
Observed per round for bw_mul call 29 (x*x): call_result_descriptor says
members are shape () (desc), while leaves 31/32 hold {shape (2,3)} again at
the start of each round; the publication rewrites them to () (changed=True)
and something rewrites them back to (2,3). Two writers disagree about one
value; trapping the (2,3) writer next.
Trap on leaf 31's `tensor` (kx806_trap.py): alternating writers every round:
- _fold_callsite_structural_values (~:22105) sets {shape (2,3)} (it re-derives
  an `indexed` node's descriptor when the stored shape is empty);
- _publish_callsite_return_members (~:17512, settled branch) sets {shape ()}
  from call_result_descriptor.
(2,3) is the true gradient shape of `left`; the () member descriptor from
call_result_descriptor is the wrong record. Tracing why next.
Descriptors handed to each bw_mul propagation copy are correct (g, x, y all
(2,3)); its published member descriptors still come out (). Inferred, not
traced: the copy's output descriptor is read from a row keyed by the
authored function name (shared by every bw_mul copy). Parking this; doing
the requested checks first.

## 2026-10-03 Per-callsite fold row: traced (x**3*y, kx806_foldrow.py)

For each bw_pow callsite copy (scope ('lexical_reads:bw_pow|fork', k)):
- `_post_copy_planner_specializations` posts planner_specialization
  ((fork k), 'p') = 3 DERIVED from the caller's argument cell
  Ref('ingestion_value', (('ingestion:training_motion', 0), 2), 0) (motion
  const 2, the exponent).
- the copy's fold of formal p posts PROVEN_LITERAL from
  `_fold_literal_source_cells` = (Ref('canonical_value', ((fork k), 1)) = the
  formal p's own cell, Ref('planner_specialization', ((fork k), 'p'))).
So the fold row derives from the formal's cell and, through the
planner_specialization row, the caller's argument cell. Confirmed by
observation; no change needed. (Each copy forks a new scope; 5+ forks for
one bw_pow callsite, i.e. the fixed point re-copies it per round.)
Probe with (i): RESULT jacobian max_rel 4.138e-16, median 0 (unchanged).
Reverse test 1 passed (12.8 s); orbital 4 passed / 7 xfailed. Back to the scalar-loss non-termination.

## 2026-10-03 STATUS after (i)

GREEN: probe max_rel 4.138e-16 (median 0); reverse test 1 passed (12.8 s);
orbital 4 passed / 7 xfailed; linear motion test back to its HEAD failure
mode (ConcordanceRefusal expand_reduction 'axis', 8 s); per-callsite fold row
traced (formal cell + planner_specialization DERIVED from caller arg cell).
RED (worse than HEAD): native_scalar_loss_adjoint[True] does not terminate.
(i) blocks only `bw_add g` (the unit seed, gradient role) at catalogue level;
the run then passes HEAD's refusal and enters a non-terminating
_propagate_callsite_tensor_specializations fixed point:
- bw_mul calls 29/30 (left*left, right*right) republish members every round;
- leaf 31's tensor alternates: _fold_callsite_structural_values (~:22105)
  writes (2,3) (true), _publish_callsite_return_members (~:17512) writes ()
  from call_result_descriptor;
- the bw_mul propagation copy gets correct inputs (g,x,y all (2,3)), but its
  outputs (nested `unbroadcast` Calls 5/8) have no descriptor after the
  copy's fold (desc None); no proven_shape_contract exists for (bw_mul,5);
  where the () member descriptor is finally produced is NOT yet traced.
Repro (stops itself): scratchpad kx806_rounds.py (200 s) / kx806_trap.py (60 s).
Second-compile refusal: not started.

## 2026-10-03 Coordinator: keep (i), fix the loop at its identity

(i) committed as a checkpoint. Task: trace the () member descriptor from
call_result_descriptor for the bw_mul copy; the return member must DERIVE
from the callee's return value cell; make _publish_callsite_return_members
post its write so a disagreement is a REVISE refusal, not a spin.
call_result_descriptor for bw_mul copies: early rounds outputs None; later rounds the copy's nested unbroadcast Call 5/8 carries tensor {shape ()} and that is published. Trapping the writer of Call 5's tensor in the copy.
Writer of the () on the bw_mul copy's Call 5 (trap): the round loop's
single-result branch (~:18158, `data["tensor"] = replacement`) running with
caller = the CATALOGUE bw_mul (it is in `graphs`, and receives merged
planner descriptors via ~:18430 `_apply_callsite_tensor_descriptors(callee,
additions)`), node 5 = its `unbroadcast(g*y, x.shape)` call. Each round's
bw_mul propagation copy is extracted from that catalogue graph and inherits
Call 5's {shape ()}. So the stale call-level () originates in
call_result_descriptor(catalogue bw_mul, 5, unbroadcast). Tracing that next.
Distinct unbroadcast propagation copies seen (kx806_ubin.py): besides the
exact ones, a copy with G (2,3), target_shape descriptor (2,) and NO literal
(specs {}) is made from catalogue bw_mul's Call 5 in some rounds; catalogue
bw_mul's Call 5 then gets {shape ()} at ~:18158. Catalogue bw_mul's x.shape
const is (2,3) when dumped. Checking what that non-literal copy returns.
Trap on CATALOGUE bw_mul Call 5 (kx806_cat5.py): round 1 sets (2,3) (correct);
~:18433 `_apply_callsite_tensor_descriptors(catalogue bw_mul, additions)`
invalidates it (pop); the next ~:18158 write sets () although the catalogue
state is then correct (g,x,y (2,3); Mul 3 (2,3); target const (2,3)). So
call_result_descriptor(catalogue bw_mul, 5, unbroadcast) answers () from
correct inputs. Checking the expanded unbroadcast copy's output answer.

## 2026-10-03 ROOT of the () member: shared (function, value) row beats the copy's own node

kx806_ub23.py: the expanded unbroadcast copy for bw_mul (target (2,3)) has
its reshape node 1 with tensor {shape (2,3)} (written by
_expand_specialized_unbroadcast_identity), but `_tensor_descriptor(copy, 1)`
answers {shape ()} (proof None): the concorded shape-transformation row
('unbroadcast', 1). Every expansion mints its reshape at id input_id+1 = 1, so
the bw_add copies (target ()) and the bw_mul copies (target (2,3)) share one
row keyed by the authored name; the earlier () wins. The gates that refuse the
shared row (formal_conflict / localized_formal / specialized_operator /
polymorphic_specialization) do not fire: reshape is not a listed operator, and
G=() is never published as a formal shape (only non-empty shapes are), so no
FORMAL_SHAPE_CONFLICT exists for unbroadcast's G.
Chain: shared row () -> copy output () -> call_result_descriptor ()
-> catalogue bw_mul Call 5 () (~:18158) -> bw_mul copies inherit () ->
member publication () vs fold re-derivation (2,3) -> endless rounds.

## 2026-10-03 Fix applied: rank-0 formal shapes are published

glsl_deployment_strategy.py: the two formal-shape publications (fixed point
~:18290, callsite copy ~:23952) gated on non-empty extents; they now gate on
`descriptor_states_a_shape` (a () with a known dtype is a stated shape). Then
unbroadcast's G () vs (2,3) records FORMAL_SHAPE_CONFLICT and copies stop
reading the shared row (polymorphic_specialization). Testing with timeout.
scalar-loss [True] now fails FAST (9.6 s) with the HEAD refusal (expand_reduction 'axis'); the fixed point terminates.
Reverse test 1 passed (8.9 s); orbital 4p/7xf. Now the expand_reduction 'axis' refusal (single compile, fresh process).

## 2026-10-03 The expand_reduction 'axis' refusal (single compile): observed

kx806_axis.py: catalogue row planner_specialization
(('lexical_reads:expand_reduction', 0), 'axis'):
1. top-level strategize: Unresolved(SPECIALIZATION_DYNAMIC_ARGUMENT) from
   cells (canonical_value(bw_sum, 0)) -- bw_sum's formal `axis` passed on.
2. nested strategize (~:28920), after bw_sum's own `axis` formal got its
   planner specialization (None): SpecializationFact(None, LITERAL) from the
   SAME cell set -> "REVISE without a changed source".
The cause of the change (bw_sum's planner_specialization row for `axis`) is
not among the argument cells: `_argument_identity_cells` adds the
PROVEN_LITERAL cell for a folded Constant but nothing for an Input the planner
specialized, whereas `_fold_literal_source_cells` does add that Input's
planner_specialization cell. Fix: `_argument_identity_cells` adds the
planner_specialization cell of a planner-fed Input (same lookup).
Observed (kx806_axis.py, AIC hook): in the second propagation bw_sum's
`axis` node is already a folded `Constant None` (structural_specialization),
and its argument cells are only canonical_value: replace() posts PROVEN_LITERAL
only for int/float/bool/str/tuple, never None, so the fold left no cause cell.
(The `_argument_identity_cells` Input-cell addition was not the observed path;
REVERTED.) Fix: replace() posts PROVEN_LITERAL for `None` as well.
PROVEN_LITERAL hook: no post for ('bw_sum', 0): replace()'s 'already recorded' test (literal_page.latest(row) != value) reads a missing row as None == None. Now also posts when the row has no cell.
Fix: replace() also posts when `book.latest_ref(PROVEN_LITERAL, row)` is None.
Result: native_scalar_loss_adjoint[True] gets past the refusal, compiles and
runs; FAILS FAST (24 s) on the numeric check: gradients stay NaN (the test's
poison fill), i.e. the gradient buffers are never written. New frontier.
Linear motion test now passes the refusal too and fails fast (12 s) further on: 'concorded source/SSA shape disagreement for (training_motion__unbroadcast__specialized_bc87d5b45e7b, 0): ssa=(3, 2), concordance=(2, 2)'.
Reverse test still passes but took 155 s (was 9-13 s). Investigating the slowdown.
Re-timed: reverse test 1 passed in 13.6 s (the 155 s run was transient; the rounds wrapper shows 3 rounds, 11 s). Running the probe.
Probe: RESULT jacobian max_rel 4.138e-16 (green). Orbital 4p/7xf (19 s). Next: _publish_callsite_return_members posting.

## 2026-10-03 Second compile in one process: precise repro

`pytest tests/test_native_scalar_loss_adjoint.py` (both params, one process,
15 s): [True] fails on the numeric check (NaN grads, as alone); [False] then
fails with ConcordanceRefusal planner_specialization
(('lexical_reads:unbroadcast', 0), 'target_shape') "REVISE without a changed
source". [False] ALONE fails only on the numeric check (12 s). So the refusal
is state carried from the first compile.

## 2026-10-03 STATUS (uncommitted, on top of 0519095d)

This lane's hunks, all in glsl_deployment_strategy.py:
- @@18291: fixed-point formal-shape publication gates on
  descriptor_states_a_shape (rank-0 shapes published) -> FORMAL_SHAPE_CONFLICT
  for unbroadcast G ()/(2,3); ends the non-terminating fixed point.
- @@23952/@@23954: same gate in _callsite_specialized_shell_type.
- @@21754/@@21768: replace() posts PROVEN_LITERAL for None, and treats a
  missing row as "not recorded" (was read as None) -> ends the
  expand_reduction 'axis' REVISE refusal.
Checks: probe max_rel 4.138e-16; reverse test 1 passed (13.6 s); orbital
4p/7xf; native_scalar_loss_adjoint[True] fails FAST (24 s) on NaN gradients
(new frontier: gradient buffers never written); linear motion fails fast
(12 s) on an unbroadcast copy shape disagreement ((3,2) vs (2,2)).
Not done: _publish_callsite_return_members still writes its raw page with
page.set (undeclared page "callsite_projection_specialization", no readers).
Posting it DERIVED from the call cell would refuse legitimate per-round
refinements (same cell, refined fact); a sound post needs the callee return
value's cell, which the function is not given. Question raised.
Second-compile refusal: the reverse-compile path never calls
begin_identity_book, so every compile in a process shares one detached book,
and the lru_cached BACKWARD_RULES graph keeps its scopes across compiles.
Repro above (whole test file, 15 s). Where a reverse compile's book begins
(and whether the cached rule graph re-posts into each book) is a design
decision; raised.

## 2026-10-03 Coordinator decisions D1, D2 (fixes committed by main session)

D1: pass the callee return value's cell into _publish_callsite_return_members;
member writes post DERIVED from it on a DECLARED page (no raw page.set); the
fold's re-derivation posts on the same row.
D2: a reverse compile's book begins at its public entry
(obtain_graph_reverse / compile_native_graph_reverse); the cached
backward-rule graph re-posts its rows into each new book, no scopes carried.
Then: NaN gradients (native_scalar_loss_adjoint), linear (3,2)/(2,2).

## 2026-10-03 D1 written (not yet run)

- concordance_declarations.py: page CALLSITE_RETURN_MEMBER
  (caller_scope SCOPE, call VALUE_ID, index INDEX, member VALUE_ID) -> object.
- glsl_deployment_strategy.py:
  * `_post_callsite_return_member` helper (row keyed by the caller COPY's
    lexical_read_scope, fact = descriptor receipt, `_post_if_changed`).
  * `call_result_descriptor` collects per slot the callee return value's
    cells (node_identity_cell + SHAPE_STATE latest_ref of the value actually
    described) into `return_cells_by_call`; the publication receives them.
  * `_publish_callsite_return_members(..., return_cells=)`: record() posts
    DERIVED from them (raw page "callsite_projection_specialization" gone).
  * structural fold (~:22105) re-derivation of an aggregate member posts on
    the same row DERIVED from the member's identity + shape-state cells.
D1 verified: reverse test 1 passed (10.3 s); scalar-loss [True] same NaN numeric failure (no refusal); left*right posts 2 callsite_return_member rows ((2,), float64).

## 2026-10-03 D2 written

- process_graph_autograd.py: `_compiled_backward_rule_process_graph` memo is
  per identity book (rebuilt, re-posting all rows under fresh scopes, when a
  new book is current; ~2.5 s); the lru_cache is gone.
  `reverse_compile_book()` context manager + `_owns_new_reverse_book()`:
  open a book unless an enclosing reverse compile owns the current one or a
  forward compile does (sympy Derivative folding). obtain_graph_reverse's
  AbstractTensor branch opens a standalone book (stays current for the
  caller's lowering) when it owns a new compile. A ProcessGraph source never
  opens one (its ingestion rows are already in the current book).
- llvm_training_runtime.py: compile_native_graph_reverse and
  compile_native_training_schedule run inside reverse_compile_book().
D2 verified on the repro: both params in one process now fail ONLY on the numeric check (no refusal), 30 s. Next: NaN gradients.
NaN anatomy (kx806_nan.py): grads 35/36 ARE in buffer_order and Ret. Root calls bw_add copy with NO args (g=1.0 folded) and loads its members 25/26 as shape (2,3); for scalar losses they must be (). bw_sum/expand chain then runs on (2,3). Checking the member rows.
The (2,3) member post on call 24 comes from the publication with a NEW source: shape_transformation_state ('bw_add', 4) revision 8 = (2,3), read in bw_add fork 5. Shared authored row again. Trapping its writer.
Writer found: catalogue bw_add's unbroadcast call (G (), target ()) is
answered (2,3) because `_structured_output_descriptor` overrides the copy's
own correct () descriptor with `proven_shape_contract_of('unbroadcast', 1)` =
(2,3): cemented by the bw_mul expansion copies (non-empty extents only are
cemented), and keyed by the authored name shared by every copy. Its comment
says conflicting specializations answer None, but nothing records the
()-vs-(2,3) conflict there.
Fix applied: _structured_output_descriptor skips the shared authored proof when the copy's authored function has a FORMAL_SHAPE_CONFLICT (same gate as _tensor_descriptor's polymorphic_specialization).
native_scalar_loss_adjoint[True]: 1 PASSED (15 s).
tests/test_native_scalar_loss_adjoint.py (both params, ONE process = two compiles): 2 passed (31 s).
Linear test still: unbroadcast__specialized_bc87d5b45e7b value 0 ssa (3,2) vs concordance (2,2). Tracing.
Root (read): fortran_c_shell `note_shape` keys value_shape rows by the
planned graph's function_name = the AUTHORED name shared by every callsite
copy; two unbroadcast copies (G (3,2) and (2,2)) wrote ('unbroadcast', 0),
last writer won, and the completed-module seam (~:41926) refused the other
copy. "polymorphic" was only set when ONE graph's formal got two shapes.
Fix: note_shape remembers which planned graph stated each row's shape; a
DIFFERENT graph stating a different shape marks the row polymorphic (sticky),
which the seam already skips.
Linear test now compiles; fails numerically: loss NaN at test line 376 (expected 0.576425).
Linear native run (kx806_lin.py): loss NaN, grad_1/grad_2 NaN, grad_3 = 0 (expected nonzero). ABI has no extents; forward regions look structurally right (matmul, broadcast b, add, sub, mul, sum/4).
Watch: 5 (matmul) correct; 6 = 5 + broadcast(b) is NaN -> the b broadcast_double in forward region_1.
Linear root: two hidden formals 2305..160/163 (shape (1,), not in buffer_order) feed the bw_mean copy: Call [160, 163] -> 21 bw_mean. (Watch reads final buffer contents, so 6=NaN may be a later overwrite; not conclusive.)
Linear: call-arity / frame-storage mismatch (observed, kx806_lin.py):
- bw_mean copy `training_motion__bw_mean__specialized_d19f9a5a7acd` has ONE
  formal (2305..154, linked_call_frame_storage for its expand_reduction
  callsite 11) and calls expand_reduction with NO args;
- the root calls it as `Call [2305..160, 2305..163] -> 21` (TWO hidden
  formals, linked_call_frame_storage for bw_mean callsite 21), and those two
  are root formals NOT in the artifact buffer_order.
A call passing 2 actuals to a 1-formal function and root formals outside the
ABI would corrupt pointers; consistent with NaN even in forward buffers.
Not traced further (frame linker). Running the green checks next.

## 2026-10-03 STATUS (uncommitted on top of ccba325c)

GREEN: probe max_rel 4.138e-16; reverse test 1 passed (8.9 s); orbital
4p/7xf; tests/test_native_scalar_loss_adjoint.py 2 passed in ONE process
(two reverse compiles, D2).
RED: linear motion test compiles but runs NaN (call-arity/frame-storage
mismatch above, not yet traced).
This lane's hunks:
- concordance_declarations.py @@558: CALLSITE_RETURN_MEMBER page (D1).
- glsl_deployment_strategy.py: @@17466 helper _post_callsite_return_member;
  @@17468..17492 publication posts DERIVED from return cells (D1);
  @@17710..18140 return_cells_by_call / output_cells in call_result_descriptor
  and passing them (D1); @@18950/@@18951 _structured_output_descriptor skips
  the shared authored proof for a polymorphic copy (NaN fix);
  @@22124 fold re-derivation posts on the member row (D1).
- fortran_c_shell.py @@16720/@@16722: note_shape cross-copy polymorphism.
- process_graph_autograd.py: per-book rule-graph memo, reverse_compile_book,
  obtain_graph_reverse standalone book, __all__ (D2).
- llvm_training_runtime.py: _one_reverse_compile_book on both entries (D2).
Not mine: precompile_to_ssa.py, examples/llvm_dt_system.py, other docs.

## 2026-10-03 Next: linear-motion NaN in the frame linker (51b4cebe committed)

Task: trace why the root passes bw_mean two linked_call_frame_storage actuals
while the copy declares one, why bw_mean calls expand_reduction with none,
and why the root's two are not in buffer_order. Join formals and actuals via
book rows (call-frame storage identity), never by position.
CORRECTION (kx806_arity.py, census after lowering AND after LLVM emit): there
is no arity mismatch. bw_mean declares TWO linked storage formals (154, 157)
and the root passes two (160, 163); the root's two are passed by the entry
wrapper as `root.frame.*` (wrapper-owned), not public buffers, which is why
they are absent from buffer_order. My earlier "one formal" read was wrong.
(The formals are dead: expand_reduction takes none.) So the NaN is elsewhere;
watching the backward values next.

## 2026-10-03 Linear NaN ROOT (observed): shape vectors retyped int32 -> int64

Watching every root value: 5 (matmul) correct, 6 = 5 + broadcast(b) NaN.
Forward region_1 calls broadcast_double(b, tmp, src_shape, src_ndim,
out_shape, out_ndim). The kernel reads shape arrays as i32
(`getelementptr inbounds i32, ptr %output_shape`), but the shape vectors are
Consts of dtype int64 (`alloca i64, i64 2`; store i64 2, i64 2). [2,2] as i64
read as i32 is [2,0]: total extent 0, tmp never written, garbage -> NaN.
tensor_ssa_lowering's `int_vector` mints these as dtype "int32"; something
retypes them to int64. Trapping the retype next.
Writer: tensor_ssa_lowering `int_vector` mints the shape vectors int32 (:2239,
for broadcast_double); ir_indexing.py `_propagate_scalar_dtypes` (:293, from
lower_indexing_to_ssa_addressing :144) then retypes them int64. Reading it.
Rule conflict in ir_indexing._propagate_scalar_dtypes: the Sep-26 rule says a
Const's declared scalar dtype is its identity, but the older (839a40d1,
Aug 22) normalization right after it widens every int/int32/i32 to int64,
declared or not. Fix: the widening skips a Const whose declared dtype was
kept. Testing.
Linear motion test: 1 PASSED (12.3 s). Running the green checks.
Reverse 1 passed (7.7 s); scalar-loss 2 passed (19.9 s); orbital 4p/7xf (20 s). Probe next.
Probe: RESULT jacobian max_rel 4.138e-16 (green).
Gate: tests/test_precompile_to_ssa.py 13 failed / 91 passed with the ir_indexing change, IDENTICAL failure set to clean 51b4cebe (wtb overlay of ir_indexing.py only). Cheap gate files next.
Gate files: test_abstract_tensor_indexing 2 passed; test_compiled_linalg 7
passed; test_ir_sequence_tables 3 failed / 36 passed, the SAME 3 failures at
clean 51b4cebe (wtb).

## 2026-10-03 STATUS: linear motion test green (uncommitted)

Only hunk of this step: src/compiler/ir_indexing.py
`_propagate_scalar_dtypes` -- a Const keeps its declared integer width; only
inferred integers widen to int64. The frame-linker suspicion was wrong (no
arity mismatch; root frame storage is wrapper-owned).
Green: linear motion 1 passed; reverse 1 passed; scalar-loss 2/2; orbital
4p/7xf; probe max_rel 4.138e-16. Gate unchanged vs clean HEAD.
