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
