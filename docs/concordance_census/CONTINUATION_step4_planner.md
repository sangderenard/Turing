# Continuation: step 4, planner structure on the book (lane D, 2026-09-30)

Spec: `docs/concordance_census/80_plan_steps4_5_planner_and_control_builder.md`
part A, with design section 7 applied (every callee copy is its own variant;
the scope ladder is correct).  Nothing here is committed.  Function names
only; no compiler numberings.

## What changed (files / functions)

`src/compiler/glsl_deployment_strategy.py`

- `_ordinary_conditional_control_programs`: a reducer field-state merge Phi
  for THIS conditional (`source_conditional_id` == the control, with
  `field_state_arms`) is a retention reason, checked before the early
  `continue`.  New `_nest_region_less_conditionals` (called at the end):
  a retained conditional with no region of its own is placed inside the arm
  of the innermost retained conditional that lexically contains it (before
  the first region marker whose earliest source line follows it, before a
  trailing terminal control, else at the end) and its standalone program is
  dropped.  Without this the overlay appended the region-less child AFTER
  its parent conditional, the outer merge asked for the inner MERGED cell
  before the inner conditional was lowered, and the lowering raised
  `ssa_field_version disagreement`.  This is the fix for
  `probe_scalar_write_only_arm.py`.
- Specialization (A2.1): `_argument_identity_cells`, `_formal_identity_cells`,
  `_post_planner_specialization`, `_post_copy_planner_specializations`,
  `_materialize_planner_specializations`, `_post_if_changed` (the R5 guard:
  post a REVISE only when the fact or the cell set changed).
  `_propagate_callsite_planner_specializations` posts
  `SpecializationFact(value, LITERAL|DEFAULT)` or
  `Unresolved(SPECIALIZATION_DYNAMIC_ARGUMENT | _CALLSITES_DISAGREE |
  _NOT_SOURCE_STATIC)` per candidate, then materializes the dict.
  `_callsite_specialized_shell_type` and `call_result_descriptor` post the
  copy's rows under its forked scope from THIS callsite's argument cells
  (7.1); the shell-type cache key gains `id(current_identity_book())` (R3).
  `propagate_bound_planner_specializations` posts `BOUND` rows DERIVED from
  the call node cell and the caller's rows the argument read
  (`resolve(..., used)`); the entrypoint's own bindings are
  `Unsourced(EXTERNAL_DECLARATION)`.
- `_publish_formal_literal` / `_publish_formal_shape`: through `post`,
  conflict = `Unresolved(FORMAL_LITERAL_CONFLICT | FORMAL_SHAPE_CONFLICT)`
  reading both cells; the `try/except: pass` is gone.  Readers
  `_proven_formal_literal`, `_proven_formal_shape`, `_tensor_descriptor`
  (`formal_conflict`, `polymorphic_specialization`) read `Unresolved`.
- Classification (A2.3): `_is_dispatch_metadata_node_impl` is now a wrapper
  over `_dispatch_metadata_rule` (one `MetadataRule` per return / disjunct,
  same order); `_EXECUTABLE_RULES` names the EXECUTABLE ones.
  `_dispatch_metadata_node_classifier` mints a planning scope
  (`mint_scope("plan", PLANNER_SCOPE)`, stored at
  `graph.G.graph["planning_scope"]`) when the fingerprint changes and posts
  one `executable_node` row per classified node DERIVED from the node's
  identity cell (plus its `scalar_parameter` cell for an Input).
- Regions (A2.4): `_post_deployment_region` posts `deployment_region`
  (DERIVED from every member's `executable_node` cell, else its identity
  cell), `deployment_region_member` (INPUT / NODE / OUTPUT) and
  `dispatch_store` (NOVEL `DISPATCH_STORE` from the OUTPUT member);
  `strategize_shell_deployment` calls it per subgraph position after the
  `dispatch_subgraphs` tuple is built (the position IS the ordinal readers
  use).  `_dispatch_subgraph` accepts `region_index` but the caller does not
  pass it.
- Fold (A2.7 / E9): `replace` posts `proven_literal` DERIVED
  (`_fold_literal_source_cells`: the node's own cell, its operands' cells,
  the `planner_specialization` cell of a planner-fed Input, the node
  itself included); `_post_control_specialization` posts
  `source_control_specialization_concordance` DERIVED from the test's
  `proven_literal` cell and the control cell, `Unresolved(PREDICATE_NOT_KNOWN)`
  when the test is absent from `known` (the fold still selects `body` as
  before; A9.2 is held); the round record is row `(function, round)`
  DERIVED from the round's posts, else the previous round, else the
  function's `function_address` cell, else `Unsourced(RAW_PRIMITIVE)`.
  `_post_identity_table_mutation` posts `name_binding` REVISE
  (`Unresolved(BINDING_VERSION_REMOVED)` for a removed node; a
  `BindingFact` naming the alias source) and revises `return_site_slot`
  rows that named an aliased value; called from `remove_node`,
  `replace_alias`, the arm-removal tail, the unused-parameter removal,
  `_alias_projection_to_member`, and the loop composer's evaporation.
  `_post_pruned_return_sites` posts `Unresolved(RETURN_SITE_UNREACHABLE)` on
  the return slots of sites the fold pruned.
- Call binding (A2.8): `_post_call_binding` (rows DERIVED from the call
  node cell and the callee's `function_address` cell), called from
  `plan_callsites` (CALLSITE_SHELL / CONSTRUCTOR) and
  `process_graph_function_linking.link_process_graph_functions`
  (LINKED_PROCESS_GRAPH); `_post_callsite_activation` posts the activation
  record DERIVED from the binding cell.
- `_synthetic_device_scalar_shell`: scratch scope, three `canonical_value`
  rows NOVEL(`SYNTHETIC_DEVICE_SCALAR_PREDICATE`), one `name_binding` row.

`src/compiler/hierarchical_plan.py`: `assign_hierarchy_ids(root, previous,
*, shell=None)` mints a hierarchy scope and posts `hierarchy_value` (DERIVED
from the local value's identity cell in the closure's graph, found through
`shell.callsite_function_shells`; `Unsourced(SYNTHESIZED_NO_SOURCE)` for a
planner-minted value, R4) and `hierarchy_global_value`;
`HierarchyValueTable.scope`.  The five callers pass `shell=`.

`src/compiler/loop_composer.py`: `_post_planner_node_row` (a port's or
unrolled constant's `canonical_value` row, NOVEL `LOOP_RESULT_PORT` /
`LOOP_STATE_PORT` / `LOOP_COMPOSER_CONSTANT`); `add_port` posts it and
`loop_result_port_binding` DERIVED; `post_port_version` posts the port's
`name_binding` version; `rewire_continuation` rewrites operands through
`_set_operands(cause="loop_continuation_rewire", same=...)` and posts the
rewire page keyed by the READ SCOPE; `_post_loop_carried_binding`,
`_post_loop_region_membership` (fact = `deployment_region` Refs when the
planning scope has them, else ordinals).

`src/compiler/control_source.py`: `place_loop_carried_region_producers.owned_regions`
reads Ref or ordinal facts and posts `Unresolved(REGION_OWNERSHIP_UNKNOWN)`
before the marker fallback.

`src/compiler/identity_concordance.py`: `IdentityBook.mint_scope(label,
stage=None)` posts NOVEL(`MINT_SCOPE`) on `scope_registry` (N5); the module
`mint_scope` passes `stage`.

`src/common/tensors/topological_reducer.py` (`fork_read_scope` only): mints
with stage `READ_SCOPE_FORK`; does not copy `planner_specialization` /
`planner_tensor_descriptor` rows into the copy (7.1); a declared page whose
raw fact does not match its declared fact type is copied raw (this keeps
lowering alive while lane E migrates `_set_operands` to `OperandTransition`).

`src/compiler/concordance_declarations.py`, step 4 section only: 11 stages,
7 transforms, 11 reasons, facts (`SpecializationFact`, `NodeExecution`,
`MetadataRule`, `RegionFact`, `MemberRole`, `CallBinding`, ...), the 10 new
pages, `SOURCE_CALLSITE_ACTIVATION`, and the re-declared `formal_literal`,
`formal_shape`, `loop_carried_binding`, `loop_region_membership`,
`loop_result_port_binding`, `loop_continuation_rewire_concordance`
(constant `LOOP_CONTINUATION_REWIRE_PAGE`; the stage keeps the bare name),
`proven_literal`, `source_control_specialization_concordance`,
`structural_specialization_fixed_point`.

`tools/compiler_probes/probe_scalar_write_only_arm.py`: selects `step` by
name (the retained guard now owns a planned region listed first).
`tools/compiler_probes/probe_planner_specialization_chain.py`: new (A7).

## What is verified

- `probe_annotated_scalar_parameter`, `probe_struct_intake`,
  `probe_branch_written_field`, `probe_scalar_write_only_arm` pass;
  `probe_planner_specialization_chain` passes (every chain link 1-5 green;
  `unsourced` at `planner_specialization`, `planner_region_carve`,
  `planner_call_binding` is zero; `planner_structural_fold` shows 3
  round-1 fixed-point records with no cell to name).
- Audit (final rerun): `view` 0 finding(s) (before 0), `mapping` 0 (before
  0), unchanged.  `toplevel`, `energy`, `controller`, `controller_untyped`
  fail in lane E's in-progress `_ControlSSABuilder.fresh_value(transform=)`
  and `oscillator` in its `control_value_binding` REVISE; none of these
  reach step 4 code.  Two step-4 defects the first rerun exposed are fixed
  (the `LOOP_CONTINUATION_REWIRE` stage/page name collision; the fork
  copying raw `identity_transition` tuples onto lane E's declared page).
- Measurement (`measure_completeness.py mapping`): cells with an inbound
  DERIVED edge 99 (23.3%) -> 242 (43.0%); mint records 36 -> 82; pages
  declared 20 -> 66.  `controller` could not be measured (lane E state).
- Mapping worklist after: the groups `loop_carried_binding`,
  `loop_region_membership`, `loop_result_port_binding`, `scope_registry`
  are gone; `structural_specialization_fixed_point` shows 2 round-1
  records (`planner_structural_fold`, raw_primitive), `ingestion_value`
  4 `synthesized_no_source` (planner-minted nodes the seam listed).
- The unsourced FACT count rises on `view` and `mapping`: the new
  `hierarchy_value` rows for planner-minted values and the
  `ingestion_value` rows `node_identity_cell` posts for nodes with no row
  (both `SYNTHESIZED_NO_SOURCE`, the R4 residue the plan expected the audit
  to list), and the entrypoint's `BOUND` rows.  The unsourced GROUP count
  and the unsourced IDENTITY count fall.

## Not done from plan 80 part A (say what, do not substitute)

- E1 `record_proven_shape(..., source)`: its columns are causal LEVELS
  (`page.set(row, level, ...)`), `post` columns are revisions; re-expressing
  it needs a decision on the page's column semantics.  `proven_shape` stays
  raw; `publish_call_result_shape` and the fold's Input branch pass no
  source (census 20 sites 5, 6).
- E5 `planner_tensor_descriptor`: declared, never posted
  (`_propagate_callsite_tensor_specializations`,
  `_apply_callsite_tensor_descriptors`, `_publish_formal_shape`'s callers in
  the tensor fixed point pass no cells).
- E11: `consumer_operand`, `item_operand`, `call_argument_operand`,
  `operand_position_orphan`, `call_result_projection_concordance`,
  `callsite_return_specialization`, `callsite_tensor_result_specialization`,
  `tensor_shape_settlement_concordance`, `callsite_projection_*` (the
  `id(caller.G)` scope is still there), `source_precision_region_concordance`,
  `aggregate_ledger`, `operator_result_type_concordance` re-key: untouched.
- `plan_region_to_ssa_instrs.fresh_like` still mints raw (it has no
  hierarchy cell in scope).
- `_apply_callsite_aggregate_descriptors` leaf `name_binding` posts;
  `_retarget_all_cached_value_ids` `return_site_field_state` revise.
- The dicts are not yet pure read views: `planner_specializations` is
  materialized from the page over keys unmigrated writers set;
  `identity_table` is still rewritten by its callers (the posts run
  beside the rewrite); `structurally_specialized_conditional_node_ids`
  stays the shadow tuple (the page view would exclude
  `PREDICATE_NOT_KNOWN` rows while the fold still selects them; A9.2).
- Provenance deviations: a DEFAULT specialization derives from the formal
  Input's cell and its `scalar_parameter` row (the def statement is not on
  the graph, so the default's span cell is not reachable here);
  `formal_literal` derives from the caller's argument cells, not the copy's
  specialization cell (posted before the copy exists); the
  `executable_node` rule cells other than `scalar_parameter` are not named;
  `hierarchy_value` has no `call_argument_operand` edge.
- `_planning_alias_transition_findings` reads
  `planning_alias_transition_concordance`, not `identity_transition`; the
  "accepts the new cause" item does not apply as spelled.
- Unresolved calls (`unresolved_ast_calls`) post no `CALLEE_UNRESOLVED`.
- E18 (deleting the read-view rebuild sites) not started.
- A9.1 / A9.2 remain the user's.

## Exact next edit

E5: give `_apply_callsite_tensor_descriptors(graph, descriptors, cells=None)`
a per-parameter cell mapping and post `planner_tensor_descriptor` rows
`(graph read scope, parameter)` with `_post_if_changed` (DERIVED from the
argument node's identity cell; `Unsourced(RAW_PRIMITIVE)` when none);
collect the cells at its four callers (`_callsite_specialized_shell_type`
loop, `call_result_descriptor` loop, the `candidates` loop of
`_propagate_callsite_tensor_specializations` -- keep a parallel
`candidate_cells` dict keyed like `candidates` -- and
`propagate_bound_planner_specializations` with `Unsourced(EXTERNAL_DECLARATION)`).
Then `_publish_formal_shape` in the tensor fixed point passes
`(node_identity_cell(caller, parent),)`.

## Working-tree state the next session must know

- Uncommitted edits (this lane): the nine files above plus the new probe.
  Other lanes' uncommitted edits in the same tree: `precompile_to_ssa.py`
  (lane E, in progress -- `fresh_value(transform=...)`,
  `_post_carried_snapshots`), the step 5 section of
  `concordance_declarations.py` (lane E), `fortran_c_shell.py` (lane F).
- `probe_scalar_write_only_arm.py` and this lane's edits to the planner
  make a retained guard own a planned region, so `module.functions` lists
  `..._planned_region_0` before `..._step`.
- Scratch files (not in the repo): the session scratchpad holds
  `laneD_audit_before_lines.txt`, `laneD_audit_before_full.txt`,
  `laneD_audit_after_lines.txt`, `laneD_probe_chain.txt`,
  `laneD_show_args.py`, `measure_completeness.py` (the lead's).
- `TEST_BASELINE_AND_HAZARDS.md` has no line yet for
  `probe_planner_specialization_chain.py` (E17's second half).
