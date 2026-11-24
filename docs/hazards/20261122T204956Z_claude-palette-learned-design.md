# Graph-native functions in the source compiler: splice, admission, training link (2026-11-22, case 9f272724)

Measured on checkout `claude/palette-learned-design` from main `8e3ecbf0`, uncommitted at the time of
writing. Baseline worktree: `C:\Users\alber\AppData\Local\Temp\wtb2` (detached `8e3ecbf0`).

## Update 2026-11-23: training compiles and runs natively

- Green: `tests/test_training_through_source_compiler.py` (linear MSE linked into `lower_ast_source_to_ssa`,
  0 concordance findings over 39 functions, LLVM O3 artifact equal to the hand-written prediction, loss and
  both gradients at every step of a 4-step descent, loss 9.5625 -> 0.0107825625);
  `tests/test_static_shape_parameter.py` (4); the splice test above.
- Regression gate (`test_precompile_to_ssa`, `test_symbolic_fluid_native_runtime`,
  `test_abstract_tensor_indexing`, `test_ssa_fusion_regions`, `test_region_kernel_dedup`,
  `test_compiled_linalg`, `test_ir_sequence_tables`): checkout 30 failed / 182 passed, clean `8e3ecbf0`
  31 / 181; same failures, plus `test_empty_dict_constructor_keeps_accepted_receiver_table_roles` red on main
  only. No new failure.
- The earlier diagnosis of the `unbroadcast` refusal (an uncalled function-table shell) was wrong: the
  emitter already lowers only the activation tree (`_walk_planned_shells(include_function_registry=False)`).
  The chain, each link observed:
  1. `_expand_specialized_unbroadcast_identity` matched the helper by module spelling; the rule graph
     compiles it from generated text (`('<string:10bcb0...>', 'unbroadcast')`), so it never fired (also on
     main). Under it, `ingest_exception_syntax` posted the definition span before `build_from_ast` stamped
     the `backward_rule_definition` cells, so the span was a root. Fixed: `stamp_generated_source_cells`
     (graph_express2) runs first; `_declared_backward_helper` identifies the helper by that declaration.
  2. `backward_registry.HELPER_STATIC_SHAPE_PARAMETERS` declares `unbroadcast.target_shape`
     (page `static_shape_parameter`, DERIVED from the declaration); `_require_static_shape_literals`
     refuses a callsite without the literal, for a copy and for a callsite left generic.
  3. The expansion, never run before: its nodes now derive from `G` and the copy's
     `planner_specialization` row (two Input cells gave two activity owners), and its identity table holds
     only `G` and the result (a stale `target_shape` committed a contract with no frame cell).
  4. A planned region's formal is static only through `value_type_abi`, an owner descriptor or a call
     shape at section time; the linked edge contract is settled onto the formal only after the sections
     (`linked_value_abi_settlement`), and section shortfalls are final. `value_type_abi` is written only for
     contracts with a `python_type`. Two gaps: the probe host was undeclared (fixed in the program: ProgramABI
     `values` with `python_type` AbstractTensor and shape), and graph-native values had no class.
     `admit_process_graph_function` now posts `source_value_class` AbstractTensor for every non-call node with
     a tensor descriptor, DERIVED from its identity cell.
  5. Audit: the singleton-span parameter seed check accepts the `ingestion_value` cause at the ABI row for a
     graph-native function (its read scope is its ingestion scope). The parameter memory contract check
     reads module membership from the book: a row is checked when the module holds the function, when
     `source_function_reachability` reaches it, or when a `lowered_source_function` address cell chains
     (through `function_address` predecessors) to its definition span. `source_pursuit_activation` is not
     membership: the registry build activates its whole generated module (demand = that module's span), so
     `NormalizedIndexAxis.advanced` reads True there although nothing here calls or lowers it. (A first
     version of this check read the Python `function_table` instead; replaced, owner 2026-11-23: "the
     concordance is the way to trace these things and nothing is valid that avoids it".)
  6. The training graph's tensor descriptors are on the book. The forward bridge already posted each
     recorded program value (`abstract_tensor_program_value`, NOVEL(INGEST_SOURCE)); the stamping pass
     `_annotate_numeric_metadata` (forward bridge, adjoint, fused graph) wrote its copied/inferred
     descriptors to node metadata only. It now posts `graph_tensor_descriptor` `(ingestion scope, value) ->
     (shape, dtype)`, DERIVED from the value's identity cell, the program value cells its lineage reaches
     (following `ingestion_value` edges and NOVEL `ADJOINT_OF` mint operands, `book.mint_of`) and its
     parents' rows. The graph-native class rows derive from that row. New audit
     `_graph_tensor_descriptor_findings` checks both. Measured: 65 descriptor rows, all reaching a program
     value; 0 findings.
- Requirement now visible: a host that links a training graph must declare its tensor inputs (ProgramABI
  `values` with `python_type` and shape). Undeclared, the same `sum` refusal returns.

## Update 2026-11-23 (final): the color game, and the three original open defects

- The color game (`examples/meowdoku/PALETTE_FEEDBACK.md`): random rounds of eight designs of distinct
  named colors with bevel and shadow, rankings into ordered pairs, a 0-1 slider score, Try again, Clear
  history, the comparison view, saved palettes; `palette_model.py` trains the native schedule continuously
  (shared group on everyone's responses, personal group on the owner's) and scores proposals with the same
  compiled function. SpeakToMe catalog: `palette_round`, `palette_clear`, slider `palette_feedback`
  (branch `claude/palette-learned-design` in `speaktome`). `tests/test_meowdoku_palette_feedback.py` (2,
  run with `C:/dev/Powershell/srv/python/python.exe` and `SPEAKTOME_REPO` set; the system Python lacks
  `cryptography`).
- All three original open defects are fixed and green:
  `test_spliced_graph_reverse_executes_through_the_source_compiler` (scalar Derivative),
  `test_spliced_outer_equation_executes_through_separate_linked_graph` when run alone (socket bindings are
  part of `symbolic_compilation_recipe`; `_make_program` keeps an authored program's `math_program_recipe`;
  `rebuild_symbolic_process_graph_in_current_book` rebuilds each body in the active book and binds it
  again), and training through the source compiler.
- `graph_tensor_descriptor` facts carry a basis: `program`, `lineage` (ancestor descriptor rows through
  `ingestion_value` edges and NOVEL mint operands) or the declaring compiler (`symbolic equation model`,
  posted by `resolve_graph_reverse_sockets` before a SymPy reverse is taken, citing `constant_role` rows).
  The audit checks the basis against the causes.
- `training_motion_function` refused nothing when an observed output was outside the loss graph: the id
  named an unrelated motion node. It now requires the node's lineage to reach that program value.
- `tests/test_graph_function_splicing.py` as a file failed 5 of 6 on clean `8e3ecbf0` ("node 4 ... has no
  identity row under (('ingestion:sympy', 0),)"): a `compile_sympy_equations` disk-cache hit returns a graph
  whose rows were posted in the book that filled the cache. Fixed: `bind_process_graph_function_sockets`
  first replays such an envelope into the active book in place
  (`rebuild_symbolic_process_graph_in_current_book`, its `G` and `roots`). Now 3 of 6 pass. The other three
  (finite sum, definite integral, zeta series; clean main never reached this stage, it stopped at
  `lexical_read_scope` / topology) stop at the final audit: "elementwise span input has no matching physical
  span result" in the region that compares and adds the runtime-length `arange` of the summation index
  (`Le(k, n)`, `Add(c, k)`). Not traced: the tensor lowering's physical facts for elementwise results over a
  runtime-length span.
- A first visit's two concurrent `/api/palette` requests (the page and `site-nav`'s theme) were issued two
  browser identities. Fixed: `palette-theme.js` `pageState()` is the page's one first request, shared by
  both modules (verified: one request, first ranking accepted). The server-wide race for other pages that
  fire concurrent first requests is unchanged.
- Gate after the final edits: 30 failed / 182 passed, the same set as main less one.

## Update 2026-11-23 (later): the palette scorer trains natively

- `examples/meowdoku/palette_native_training.py` follows `examples/pattern_project/native_image_training.py`
  (`compile_native_training_schedule`, groups `shared` and `personal`). Demo: step-0 native loss
  0.36677150580196 equals NumPy; loss 0.3668 -> 0.0412; held-out pair agreement with the planted preference
  0.949.
- Green: `test_training_through_source_compiler`, `test_llvm_training_runtime` (2; main fails
  `test_native_parameter_groups_step_only_the_selected_state` with `KeyError: 'rejected_role'`),
  `test_native_scalar_loss_adjoint` (2; main fails both with `TypeError: unhashable type: 'list'`),
  `test_static_shape_parameter` (4), the splice test. Gate: 30 failed / 182 passed, the same set as main less
  `test_empty_dict_constructor_keeps_accepted_receiver_table_roles`.
- Not measured: `test_graph_function_splicing.py`, `test_process_graph_function_linking.py` (a run exceeded an
  hour and was stopped).
- The chain fixed, each link observed:
  1. Values computed inside backward rule bodies had no `source_value_class`, so `unbroadcast` copies fed by
     them had no `value_type_abi` and their planned `sum` refused. `_abstract_tensor_operator_class`
     (operators with an AbstractTensor operand, DERIVED from the AbstractTensor dunder's
     `static_symbol_definition` cell) and `_abstract_tensor_method_result_class` (a `tensor_candidate`
     method whose declaration returns AbstractTensor; annotation read as an expression, since
     `from __future__ import annotations` keeps the quotes) extend `_propagate_builtin_scalar_operator_classes`.
     `detach`, `abs`, `sigmoid` now declare `-> "AbstractTensor"`; `sum`/`mean` do not (a full reduction is
     an `AbstractScalar`).
  2. That pass ran only when the registry was reduced, before any formal had a class; it now also runs in
     each specialization scope after `specialize_python_precision_widths` posts the copy's formal classes.
  3. Graph-native callers are never reduced: `_publish_call_result_class` gives a call result the class the
     specialized callee returns (from `publish_call_result_shape`).
  4. The `unbroadcast` expansion skips a reshape to the shape the chain already has, and leaves the authored
     body for a scalar gradient broadcast to a shaped target (`ones(t) * G` is no sum/reshape chain).
  5. `settle_call_result_types` typed a static view by its returned carrier: `expand_reduction`'s
     `G.reshape((1, 1))` of the literal seed returned `Ret 0` (scalar) for output 57. `static_view_proof`:
     when the semantic id is not the carrier and the callee's book states a structure for it with the same
     buffer layout and another shape, that structure types the result.

## Status (2026-11-22)

- Green on this checkout: `tests/test_graph_reverse_splice.py::test_derivative_sympy_cannot_take_becomes_a_spliced_graph_reverse`.
- Red, cause known, next in line:
  - `tests/test_graph_reverse_splice.py::test_spliced_graph_reverse_executes_through_the_source_compiler`
    stops at `physical-call-formal-unproven: ..unbroadcast__specialized_.. value 3: span_abi_disagreement`.
    For the scalar program the `target_shape` formal's SSA row is span rank 1 with shape `()`, while its
    VALUE_TYPE_ABI is span shape `(-1,)` with metadata_state `static`. The actual is the empty tuple `x.shape`.
  - Training graph linked through `lower_ast_source_to_ssa` (`training_function_recipe`, linear MSE, wrt
    weight/bias) now passes linking, reduction, call reachability and callsite planning. It stops in tensor
    lowering at `unbroadcast__specialized_0b170d1cec7c ... sum (dynamic reduction lacks a typed source span)`.
    That copy is the function-table `unbroadcast` shell (`strategize_shell_deployment`, function-shells loop),
    lowered with a runtime `target_shape` after `_propagate_callsite_tensor_specializations` published the
    callsites' agreed descriptors to it. Every callsite has its own folded copy (static `target_shape`).
- Pre-existing on clean `8e3ecbf0`, and the same defect class:
  - Any `lower_ast_source_to_ssa(..., linked_process_graphs={name: sympy_graph})` raised
    `KeyError: 'lexical_read_scope'` in `source_control_ir.ingest_function_control` (fixed here, see 3).
  - `tests/test_graph_function_splicing.py::test_spliced_outer_equation_executes_through_separate_linked_graph`
    raises "linked symbolic ProcessGraph topology changed after compilation": the socket splice edits an
    envelope built in another identity book, and link refuses to replay it. Not fixed yet. The splice has to
    run inside the compile's book (a recipe, as in 5).

## The defect class

A ProcessGraph built without source (a SymPy law, a spliced graph reverse, a training graph) never passes
through the reducer, so none of the rows the reducer posts for an AST function exist for it. Three
hand-written partial copies of those rows existed (the training graph's two builders and the symbolic
Derivative splice); everything else failed or refused.

## What changed

1. `process_graph_function_linking.splice_process_graph`: the one splice. It copies the body nodes that
   compute the requested outputs into the host, with `ingestion_value` NOVEL(GRAPH_FUNCTION_SPLICE,
   (context, body value)), operand rows through `_set_operands`, and `callsite_argument_role` rows. The
   functions the copied calls name move to the host table (`adopt_function_references`: declared from the
   body's `function_address` cell; calls rebound with `call_binding` revised by `_post_call_binding`).
   `_splice_call` (graph sockets) now uses it.
2. `process_graph_autograd.splice_graph_reverse`: `obtain_graph_reverse` of one host value, spliced back
   (saved-forward inputs are the host values, seed is the constant one or a host cotangent).
   `symbolic_process_graph`: a Derivative SymPy cannot take is a `graph_reverse` socket at ingestion and is
   resolved by `resolve_graph_reverse_sockets` after `compile_sympy_equations` / `MathProgram` stamp the
   descriptors. The old in-ingestion reverse refused "forward value 0 reaches the adjoint graph with no
   recorded tensor dtype", redeclared rule functions with no `function_address` source, and differentiated
   a placeholder Symbol for higher orders (zero). That splice had no test since c8d4952e (2026-09-22).
3. `process_graph_function_linking.admit_process_graph_function`: one writer for what reduction posts for an
   AST function: read scope (its ingestion scope), `authored_function_identity`
   (`("process_graph:<scope>", name)`), the activity graph, SEMANTIC_OUTPUT rows, version-zero NAME_BINDING
   rows. Called for every linked graph and by `lower_training_motion_to_repository_ssa` (replacing its
   hand-written rows). `fuse_forward_loss_backward`'s empty SOURCE_ACTIVITY_SCOPE post is removed.
4. `source_control_ingestion.ingest_graph_activity`: the activity graph of a graph body from its own operand
   edges (enter/exit per node keyed by its identity cell, flow edges, bindings). The audit
   (`identity_concordance._control_activity_findings`) is taught that an `ingestion_value` occurrence with
   no owners is a graph-native activity node. The ILP special case in `control_activity.ActivitySchedule`
   (old item 3, "empty activity scope") is reverted: the audit refused it and it is no longer needed.
5. `training_motion_function` / `training_function_recipe` (process_graph_autograd): a training graph declared
   as a linkable function. A recipe in `linked_process_graphs` is built by `link_process_graph_functions`
   inside the compile's book (cached per book for the second link pass). `link_process_graph_functions` also
   moves a linked graph's own function closure into the root table before adopting that table; replacing the
   table alone left `callee_ref` naming whatever the root table held at that address.

Earlier items kept (owner: "the fixes can stay"): operator-contract string guard, IfExp `rejected_role` and
rejected-arm pruning, loop-target Input activity binding, const descriptors from literals, static `arg:0` for
`broadcast_to`/`expand`, plus the Fortran-lane edits listed in the previous version of this entry.

## Not covered

- Loops and branches in the trained forward: `differentiate_process_program` needs the planner's
  ControlProgram and caller-supplied Loop/ConditionalAdjointContract; nothing derives them. Straight-line
  bodies (including `where`) only.
- The old deployment route (`lower_training_motion_to_repository_ssa`) has not been re-run since 3/4.
