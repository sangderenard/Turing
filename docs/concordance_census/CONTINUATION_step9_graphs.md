# Continuation: step 9 part A, the graphs as views of the book (lane G9)

Spec: `docs/concordance_census/100_plan_step9_graph_input_and_emission_output.md`
part A (sections 1-2).  Part B (emission) is NOT this lane's: it waits on
step 6's SSA value identities.  2026-10-01.  Function names only; no
compiler numberings; nothing committed.

## What changed (files / functions)

`src/compiler/concordance_declarations.py`, new "Step 9" section before
`__all__` only: stages `CONTROL_PROGRAM_BUILD`, `CONTROL_PROGRAM_REWRITE`;
the operand-rewrite cause transforms (`INGEST_EDGE`, `REDUCER_SYNTHESIS`,
`APPEND_OPERAND`, `REPLACE_INPUTS`, `REMOVE_NODE`, `REDIRECT_VALUE`,
`DISSOLVE_EXPR` / `_RETURN` / `_WRAPPER`, `PARAMETER_INPUT`,
`FUNCTION_SUBGRAPH_FILTER`, `CANONICAL_RELABEL_OPERANDS`, `PROJECTION_TO_LEAF`,
`AGGREGATE_MEMBER`, `AGGREGATE_FORMAL_MEMBERS`, `BOUND_RECEIVER`,
`SCALAR_INTRINSIC_RECEIVER`, `CALLSITE_FOLD_*`, `REGION_BOUNDARY_INPUT`,
`UNBROADCAST_CHAIN`, `LOOP_BODY_CLONE`, `LOOP_MATERIALIZER`,
`LOOP_EDGE_REBUILD`, `LOOP_PARENT_REPLACEMENT`,
`LOOP_CONTINUATION_REWIRE_OPERANDS`, `SYNTHESIZED_CONTROL`); reasons
`CONTROL_OWNER_UNKNOWN`, `REGION_CELL_UNROUTED`, `COLLAPSED_EMPTY_CONSTRUCT`,
`CONTROL_BLOCK_REVISION_UNCAUSED`, `CONTROL_FUNCTION_SCOPE_UNKNOWN`; facts
`ControlBlockKind`, `Arm`, `ROOT`, `ControlBlockFact`, `Placement`,
`ControlProgramFact`, `SHELL`; pages `CONTROL_BLOCK`,
`CONTROL_BLOCK_PLACEMENT`, `CONTROL_PROGRAM` (plan 100, 2.1).

`src/common/tensors/topological_reducer.py`

- `_operand_position_scope`: third fallback `ingestion_value_scope` (1.1).
- `_set_operands`: posts `OperandAppend(cause, operand)` DERIVED(operand
  node cell, consumer node cell, the position's previous cell when it had
  one) REVISE for every new position no move fills and no fork feeds;
  a graph with no scope posts one `Unsourced(NO_OPERAND_POSITION_SCOPE)`
  row per rewrite under `_UNSCOPED_OPERANDS`; new keywords `edge_payload`
  (extra edge attributes for a new networkx edge) and `materialize`
  (False only for the canonical relabel).  New `_materialize_operands`:
  the one writer of `children` and the networkx edge -- parents that left
  lose their entry and edge, every parent in the list has its edge (role =
  its first role) and exactly one `(consumer, role)` entry per role, stale
  role spellings replaced in place so sibling order is kept.
- `new_node(..., cause=REDUCER_SYNTHESIS)`: `add_node` with empty parents,
  post `ingestion_value`, then `_set_operands` (the consumer cell exists
  before the Append derives from it).  Its ~40 callers in
  `_normalize_lexical_values` were NOT edited to pass their transform; they
  record `reducer_synthesis`.
- Hand-written halves deleted (they call `_set_operands`): `_replace_inputs`
  (its pre-removal loop now skips operand parents and still drops non-operand
  edges such as a static reference's `callee` edge), `_redirect_value`
  (which read the old edge after `_set_operands` -- would have raised once
  the writer removed it), `_remove_node`, the Expr / Return dissolves, the
  slice / keyword / iterator wrapper removals (now
  `_set_operands(node, [], cause=DISSOLVE_WRAPPER)` before `remove_node`),
  the three first-class-function `_append_operand` sites and the static
  reference `callee` site.  The relabel's `_set_operands(..., same=mapping,
  materialize=False)`; the function-subgraph `children` filter stays (its
  excluded consumers are not in the subgraph).
- Canonical relabel tail: `identity_transition` rows under the build scope
  then the reduction's ingestion scope are re-posted under the read scope
  with mapped ids, DERIVED(the row continued), stage `CANONICAL_RELABEL`
  (plan 100 1.1 said this path existed; it did not).

`src/transmogrifier/graph/graph_express2.py`: `connect` and the Store-node
site of `build_from_ast` route the edge through `_set_operands(...,
cause=INGEST_EDGE, edge_payload={'extra': ...})`; the `Edge` record stays an
edge attribute.  `children` entries now carry the CONSUMER role (every
reducer writer's spelling), not `producer_role`.

`src/compiler/glsl_deployment_strategy.py`: the scalar-intrinsic receiver,
`_dispatch_subgraph`'s Store node (cause `DISPATCH_STORE`), the conditional
tuple Phi members and both Indexed member builders (`AGGREGATE_MEMBER`), the
fold's `replace` (`CALLSITE_FOLD_LITERAL`, replacing the raw `"parents": []`
update), `remove_node`, `replace_alias`, `_alias_projection_to_member`
(its unguarded `remove_edge` after `_set_operands` is gone), the aggregate
formal members, the unbroadcast chain (`UNBROADCAST_CHAIN`) and the bound
receiver: hand-written `children` / `add_edge` / `remove_edge` deleted,
string causes replaced by transforms.  `_ordinary_conditional_control_programs`
posts every program it returns (`post_control_program`, stage
`CONTROL_PROGRAM_BUILD`, label = the conditional's node cell).

`src/compiler/loop_composer.py`: `_rebuild_graph_edges`,
`_replace_parent_value`, `add_clone`, the materializer and the aggregate
consumer rewrites, `rewire_continuation` (both branches) go through
`_set_operands` with transforms.  `add_constant` / `add_port` write no
parents and were not touched.

`src/compiler/control_source.py`: `post_control_program(graph, program, *,
stage, cause, label, previous)` and `post_control_rewrite` (plan 100, 2.3),
with `_control_block_kind`, `_flatten_control_sequence`,
`_control_block_arms`, `_callsite_marker`.  Owner cells per 2.2 (node cell,
`deployment_region` cell under `planning_scope`, `call_binding` cell,
`field_state_cell`); a block whose owner cannot be named is keyed on the
program cell with `Unsourced(CONTROL_OWNER_UNKNOWN | REGION_CELL_UNROUTED)`,
and a second such block with a different fact is counted in
`metadata["control_program_unrouted"]`, not posted.  `ControlBlockFact`
carries the predicate / carried / site / callsite cells and the scalar
payload; region membership is placement, not identity, except for a marker
block.  A changed block fact or placement is a REVISE DERIVED from the
cells plus `cause`, checked against the api's rule at the writer (plan 80
R5) and `Unsourced(CONTROL_BLOCK_REVISION_UNCAUSED)` when the api would
admit no cause.  `previous=` withdraws a vanished block's placement with
`Unresolved(COLLAPSED_EMPTY_CONSTRUCT, read=(its placement cell,))`.
`compose_region_code(..., graph=None)` and `project_control_regions(...,
graph=None)` post when handed the graph.

`src/compiler/fortran_c_shell.py`, `_class_surface_ssa_program` only: one
`post_control_rewrite(graph, control)` line after each of the twelve
`control = ...` replacements, and `graph=graph` on its two
`project_control_regions` calls.  Nothing else in that file was touched
(lanes S6 / S7 edit it concurrently).

## Verified

- `probe_struct_intake`, `probe_branch_written_field`,
  `probe_scalar_write_only_arm`, `probe_record_in_tuple_return`,
  `probe_planner_specialization_chain`, `probe_control_binding_chain`: all
  checks ok.  `probe_scalar_native_correctness`: 14/14 equal to CPython.
- Audit, seven first lines unchanged: view 0, toplevel 1, energy 0,
  controller 1, controller_untyped 5, mapping 0, oscillator 0 finding(s);
  row and function counts identical.  Unsourced facts before -> after
  (the tree also holds lanes S6 / S7's uncommitted work): view 4101 ->
  4105, toplevel 2775 -> 2829, energy 3777 -> 3800, controller 4530 ->
  4536, controller_untyped 4422 -> 4427, mapping 195 -> 197, oscillator
  2866 -> 2951.  Unsourced identities: toplevel 35 -> 10, controller 62 ->
  39, mapping 1 -> 0, oscillator 2 -> 0 (not this lane's doing).  New
  listed groups: `control_block` / `control_program` /
  `control_block_placement` at the two control stages (owner unknown,
  revision uncaused) -- the worklist of 2.2.
- `measure_completeness.py`: mapping DERIVED 45.5% (257/565) -> 50.6%
  (332/656), tagged unsourced 35.9% -> 31.2%, MINTED with a mint record
  14/15 -> 15/15; controller DERIVED 71.4% (14603/20445) -> 77.5%
  (21758/28059), tagged 32.8% -> 24.4%.
- Viewer, mapping `identity_transition`: 9 -> 79 rows.  Controller:
  `control_block` 70, `control_block_placement` 35, `control_program` 12.
- Operand edges on the book vs the graph (scratch `g9_count_edges.py`,
  the resolved source ProcessGraph the compiler hands
  `resolved_process_graph_sink`, mapping case): `G.number_of_edges()` 25,
  `sum(len(parents))` 25, `sum(len(children))` 25; positions whose latest
  `identity_transition` fact under the graph's scope is Append / Move
  (target) / Fork: 23; graph-only 2, book-only 0.  The 2 are the `Assign`
  nodes' `type_comment` positions: their rows read Append then
  `Retire(function_subgraph_filter)` -- the function subgraph shares the
  build scope with the source graph, so the subgraph's filter retired the
  SOURCE graph's row.  That is the drift finding of plan 100 1.3, found by
  measurement: a subgraph needs its own scope before it rewrites.

## Not done (say what, do not substitute)

- The audit findings `operand-view-drift` and `control-view-drift` and the
  `control_program_view` reader (1.3, 2.4) live in
  `identity_concordance.py`, not this lane's files; not written.  The
  scratch counter above is the stand-in measurement only.
- `SSA_BLOCK` (2.6) needs the block being lowered as a cell and the
  function scope: Part B / step 5-6 territory; not declared here.
- Section 3 (`EXTRACTION_RECEIPT`, `LOOP_CONTROL_SITE`, the
  attribute-cache map and `attribute-cache-drift`): not started.
- Writers outside this lane still writing `parents` raw:
  `fortran_c_shell` optional-presence rewrite (`detach_inputs`,
  `replace_uses`: hand `children` / `add_edge` / an UNGUARDED
  `graph.remove_edge(old_id, child)` after `_set_operands` -- it will raise
  once that path runs, since the writer now removes the edge; one
  `has_edge` guard or deleting the hand half fixes it; not touched per the
  lane boundary), `bitops_process_graph`, `process_graph_autograd`,
  `process_graph_function_linking`, `symbolic_process_graph`,
  `shell_telemetry`, `cycle_unroller`.  The reducer's `ledger_reads` loop
  (loop node `carried_update` / `break_value` edges) writes networkx edges
  with no parents entry: not an operand position, left as is.
- `new_node` callers do not pass their transform (`REDUCER_SYNTHESIS`).
- Loop programs from `analyze_shader_loop_reductions`, the hierarchical
  `fix_aggregate_loop_bounds` rebuild, the `control_source` passes other
  than compose / project, `_install_lexical_sequence_mutations` and the
  `precompile_to_ssa` control-function assembly do not post; glsl's own
  `project_control_regions` calls pass no `graph`.  `LoopBlock` / `WhileBlock`
  with no source loop and the synthesized conditional use the program-cell
  fallback, not `SYNTHESIZED_CONTROL` mints.
- `_control_block_arms` treats `terminal_controls` and `cleanup` as arms by
  wrapping them in a `SequenceBlock` for flattening only.

## Exact next edit

Give the function subgraph its own operand-position scope before
`_set_operands(function_graph, member, ..., cause=FUNCTION_SUBGRAPH_FILTER)`
runs (the fork of the build scope's rows for the included members, as
`fork_read_scope` does for the read scope), so the source graph's rows stop
being retired by the subgraph; then the two graph-only positions of the
mapping measurement close and `operand-view-drift` can be written against a
graph whose scope is its own.

## Working-tree state

Uncommitted (this lane): the eight files above plus this note.  Other
lanes' uncommitted edits in the same tree: `fortran_c_shell.py` (S6 / S7,
large), `identity_concordance.py`, `ssa_record_return_state.py`,
`tensor_ssa_lowering.py`, `transformation_priority.py`, two probes, the
step 6 / 7 continuations.  Line endings kept (CRLF everywhere but
`control_source.py`, LF).  Scratch (session scratchpad only):
`g9_audit_before.txt`, `g9_audit_after.txt`, `g9_audit_final.txt`,
`g9_measure_before.txt`, `g9_measure_after.txt`, `g9_measure_final.txt`,
`g9_count_edges.py`.
