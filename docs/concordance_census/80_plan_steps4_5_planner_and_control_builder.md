# 80: Steps 4 and 5 edit plan -- planner structure and the control SSA builder

Read-only planning lane, 2026-09-30.  Everything below was read from the tree
with `git grep`/`sed`/`awk`; nothing was run.  Code is named by function,
never by line.  No compiler numberings appear.

Rests on `docs/CONCORDANCE_SINGLE_API_DESIGN_2026-09-30.md` sections 2, 4
and 6 (steps 4 and 5 are this file), census 20 (deployment and loops) and
census 30 (SSA lowering) for evidence, and plans 60 and 70 for the format.
Steps 1-3 are LANDED (`src/compiler/identity_concordance.py` holds
`IdentityBook.post`, the registry, the latch and the two audit findings;
`src/compiler/concordance_declarations.py` holds the step 2-3 vocabulary;
`node_identity_cell(graph, node_id)` in `topological_reducer.py` is the seam
every `Ref` to a graph node points at).  Where this plan says "the identity
cell of node N" it means that function's result.

Vocabulary: "cell" = one `Ref`; "row" = a page row; "post" = one
`IdentityBook.post` call.  Names in capitals are the names the registry
objects should carry, declared in `concordance_declarations.py` in a new
section per step; they are objects, never strings compared at runtime.

## 0. What the landed api gives these two steps, and one thing it lacks

Observed in `IdentityBook.post`: `Novel` with a `NEW` in a VALUE_ID field
mints through `GLOBAL_MONOTONIC_IDS.mint()`; `Novel` on a row with no
`NEW` writes only the origin edge (a root); `Transform.arity` is a fixed
int and `post` refuses an operand tuple of another length; `Mode.REVISE`
admits a revision only when a source cell is newer than the previous
revision or the source set differs; `Unsourced` is admitted while the latch
is OPEN and lands on the private unsourced page; `_source_stamp` refuses a
`Ref` to a cell that does not exist, so every cell a post names must be on
the book before the post.

Needed from step 1 (say so before step 5 starts, or step 5 adds it):

- **N4. Variadic transforms.**  A conditional Phi has two arm operands, a
  loop-exit Phi has one per break edge plus the header, a return merge has
  one per return edge.  One `Transform` object per name with a fixed arity
  cannot describe them.  `Transform.arity` admits a sentinel `VARIADIC`
  (any count, at least one) and `post` checks `>= 1` for it.  Used by
  step 5's `PHI_*` transforms only.
- **N5. `mint_scope` through `post`.**  Design section 2 says `mint_scope`
  becomes a NOVEL post on `scope_registry`; today it is a raw `concord`
  and the commit message for steps 2-3 lists "one scope mint per build" as
  residual unsourced.  Step 4 mints planning and hierarchy scopes, so it
  lands this: page `SCOPE_REGISTRY` row `(label LABEL, serial INDEX)`, fact
  `bool`, `Novel(MINT_SCOPE, ())`, stage = the caller's stage.  `fork_read_scope`
  passes `READ_SCOPE_FORK`; the planner passes `PLANNER_SCOPE`.

---------------------------------------------------------------------------

# PART A -- STEP 4: planner structure

## A0. What is being moved, observed in the code

| shadow ledger | writer(s) | what it decides | readers (representative) |
|---|---|---|---|
| `G.graph["planner_specializations"]` (formal name -> folded literal) | `_propagate_callsite_planner_specializations` (`setdefault(...)[param] = first` on the SHARED callee graph, after a per-callsite agreement fold over `_source_static_value` / `_source_static_literal` and the callee's `parameter_defaults`); `_propagate_callsite_tensor_specializations.call_result_descriptor` and `_callsite_specialized_shell_type` (fresh dict on the specialized copy); `propagate_bound_planner_specializations` (`setdefault().update(environment)` per reachable callee) | which lanes exist: literal folding in `_fold_callsite_structural_values` removes arms and dead calls | `_source_static_value`, `_source_static_literal`, `_fold_callsite_structural_values.evaluate`, `_record_field_descriptor`, `direct_return_value_ids`, `static_endpoint_value`, `evaluate_node`, `compile_discovery_program`, `_resolve_grounded_method_references`, `_resolve_grounded_tensor_operations`, `strategize_shell_deployment` |
| `formal_literal` page (book, PARTIAL) | `_publish_formal_literal` from `_callsite_specialized_shell_type`; all exceptions swallowed | the cross-copy literal `_proven_formal_literal` serves to `_source_static_value` when the copy's own dict is silent | `_proven_formal_literal` |
| `G.graph["planner_tensor_descriptors"]` (+ `callsite_tensor_descriptor_names`) | `_propagate_callsite_tensor_specializations` (`setdefault` + `update` on the shared callee, then `_apply_callsite_tensor_descriptors`), `_apply_callsite_tensor_descriptors` (replaces the dict wholesale), the two copy writers, `propagate_bound_planner_specializations` (mutable descriptors) | whether `_tensor_descriptor` trusts the book at all (`specialized_operator` / `localized_formal` gates); `_fold_callsite_structural_values` Input branch writes `proven_shape` from it | `_tensor_descriptor`, the fold, `specialization_state` digest |
| `formal_shape` page (PARTIAL) | `_publish_formal_shape` | the cross-copy shape | `_proven_formal_shape`, `_tensor_descriptor` |
| `G.graph["_dispatch_metadata_cache"]` | `_dispatch_metadata_node_classifier` (fingerprint = schema, node count, edge count; per-node verdict from `_is_dispatch_metadata_node_impl`) | the executable set: `strategize_shell_deployment` filters `_dependency_order` through it; the whole-program uncovered-node check reads it | those two callers |
| subgraph `G.graph["deployment_nodes"|"deployment_inputs"|"deployment_outputs"|"deployment_store_nodes"|"compartment_schedule"]` | `_dispatch_subgraph` (mints Store nodes from a local counter) | the region table | emitters, uncovered check, `_structural_region_program_from_subgraph`, `_subgraph_reduction_digest` |
| `HierarchyValueTable.correlations` (closure, local) -> global id | `assign_hierarchy_ids` (union-find over `PlanCall.argument_bindings` / `result_bindings`, dense `enumerate` of sorted class tokens); called from `ProcessGraphGLSLDeployment.__init__` per activation root and for the module shell, and from three later re-plan sites | the one global identity every hierarchical composition stage uses | `_build_hierarchical_glsl_artifact`, `hierarchical_effective_value_table` consumers in other shells |
| `call_result_projection_concordance` page (NO edge) | `_build_shell_hierarchy_plan` | call result path -> caller projection | linker |
| `operator_result_type_concordance` page keyed by `closure_id`; `fresh_like` mint | `plan_region_to_ssa_instrs` | a minted SSA value "like" a result, no edge | SSA consumers |
| `loop.carried_bindings` / `pending_plans`; `loop_carried_binding`, `loop_region_membership` pages (NO edge) | loop plan building; `analyze_shader_loop_reductions` copies the private plan onto the two pages | which Phis and ports exist; which regions a loop owns | `precompile_to_ssa` (`_enter_loop_state`, `_publish_loop_result_ports`), `control_source.place_loop_carried_region_producers` |
| `loop_result_port_binding` page (port minted by `next_process_value_id`) and `loop_continuation_rewire_concordance` (keyed by `function_name`, bypasses `_set_operands`) | `materialize_retained_loop_ports.add_port` / `.rewire_continuation` | the port's identity and every consumer's operand rewrite | `_publish_loop_result_ports`; nobody reads the rewire page |
| `G.graph["identity_table"]` mutated AFTER the reducer materialized it from `name_binding` | `_fold_callsite_structural_values.remove_node` (drops a version), `.replace_alias` (substitutes and dedups), the fold's `history[0]` projection-alias branch (the one that calls `retained_basic_index`) and its fixed-point tail, `_alias_projection_to_member`, `_apply_callsite_tensor_descriptors` / `_apply_callsite_aggregate_descriptors`, `loop_composer` (`add_constant`, `materialize_retained_loop_ports` post-loop rewrite), `_synthetic_device_scalar_shell` (a throwaway wrapper with its own table) | root and return selection; the builder's `value_name_histories` | ~110 readers of the dict (plan 60, 3.2) |
| `return_slot_values` / `return_value_nodes` / `return_record_field_states` rewritten in place | `replace_alias`, `_alias_projection_to_member`, `_retarget_all_cached_value_ids` | return identity | plan 70's readers |
| `structurally_specialized_conditional_node_ids`; `source_control_specialization_concordance` (PARTIAL; `known.get(test, unresolved)` sentinel is truthy) | `_fold_callsite_structural_values` (two sites) | which conditionals were folded | `fortran_c_shell` reads the shadow tuple |
| `proven_literal` page (NO edge) | `_fold_callsite_structural_values.replace` | the fold's literal, readable from other copies | `fortran_c_shell`, other copies |
| node `attributes["callee_ref"|"callee_resolution"]`; `source_callsite_activation_concordance` (bare fact) | `link_process_graph_functions`; `plan_callsites` | which shells exist (shell vs recursive backedge) | `plan_callsites`, `shell_reference_tables` |

Plan 60's top risk R1 is the `identity_table` row: the dict is a view of
`name_binding` until the planner writes it, then the two diverge silently.
Section A2.7 closes it.

## A1. Pages to declare (step 4 section of `concordance_declarations.py`)

Stages: `PLANNER_SPECIALIZATION` (`_propagate_callsite_planner_specializations`,
`propagate_bound_planner_specializations`, the two copy writers),
`PLANNER_TENSOR_SPECIALIZATION` (`_propagate_callsite_tensor_specializations`,
`_apply_callsite_tensor_descriptors`, `publish_call_result_shape`),
`PLANNER_STRUCTURAL_FOLD` (`_fold_callsite_structural_values` and its
closures), `PLANNER_PROJECTION_ALIAS` (`_alias_projection_to_member`,
`_apply_callsite_aggregate_descriptors`), `PLANNER_DISPATCH_CLASSIFICATION`,
`PLANNER_REGION_CARVE`, `PLANNER_HIERARCHY`, `PLANNER_CALL_BINDING`
(`link_process_graph_functions`, `plan_callsites`), `LOOP_COMPOSER`,
`LOOP_CONTINUATION_REWIRE`, `PLANNER_SCOPE`.

Transforms: `MINT_SCOPE` (0), `LOOP_RESULT_PORT` (1: the carried binding
cell it continues), `LOOP_STATE_PORT` (1), `LOOP_COMPOSER_CONSTANT` (1: the
loop construct cell), `SYNTHETIC_DEVICE_SCALAR_PREDICATE` (1: the predicate
node cell), `FRESH_LIKE` (1: the result's hierarchy row cell),
`DISPATCH_STORE` (1: the output member cell).

Reasons: `SPECIALIZATION_DYNAMIC_ARGUMENT`, `SPECIALIZATION_CALLSITES_DISAGREE`,
`SPECIALIZATION_NOT_SOURCE_STATIC`, `FORMAL_LITERAL_CONFLICT`,
`FORMAL_SHAPE_CONFLICT`, `DESCRIPTORS_DISAGREE`, `PREDICATE_NOT_KNOWN`
(the truthy `unresolved` sentinel), `BINDING_VERSION_REMOVED`,
`RETURN_SITE_UNREACHABLE`, `CALLEE_UNRESOLVED`, `REGION_OWNERSHIP_UNKNOWN`
(the marker-region fallback in `place_loop_carried_region_producers`).

### A1.1 New pages (10)

| page | row fields (kind) | fact type | mode | provenance |
|---|---|---|---|---|
| `planner_specialization` | callee scope (SCOPE: the callee graph's `lexical_read_scope`, forked per copy), parameter (NAME) | `SpecializationFact(value, source: LITERAL / DEFAULT / BOUND)` or `Unresolved` | REVISE | DERIVED(one argument-node identity cell per agreeing callsite plus that literal's `source_span` cell; for a default the `parameter_annotation`-side default span the reducer's `_parameter_default_expression` names; for BOUND the caller's `planner_specialization` cell) |
| `planner_tensor_descriptor` | callee scope (SCOPE), parameter (NAME) | descriptor `dict` (as today) or `Unresolved` | REVISE | DERIVED(the argument node's `proven_shape` cell or `shape_transformation_state` cell per agreeing callsite, the argument node identity cell) |
| `executable_node` | planning scope (SCOPE, minted per `strategize_shell_deployment` pass), node (VALUE_ID) | `NodeExecution(kind: EXECUTABLE / DISPATCH_METADATA, rule: MetadataRule)` | CONCORD | DERIVED(node identity cell; the cell the rule read when it is on the book: `callsite_projection_specialization` for `authored_call_result_projection`, `schema_node` for AST metadata, `scalar_parameter` for the scalar rule, `loop_carried_binding` for carried initials) |
| `deployment_region` | planning scope (SCOPE), region (INDEX) | `RegionFact(schedule_preference, output_count, store_count)` | CONCORD | DERIVED(every member's `executable_node` cell) -- a region is a function of its executable members and nothing else |
| `deployment_region_member` | planning scope (SCOPE), region (INDEX), node (VALUE_ID) | `MemberRole in {INPUT, NODE, OUTPUT}` | CONCORD | DERIVED(the node's `executable_node` cell, the `deployment_region` cell) |
| `dispatch_store` | planning scope (SCOPE), region (INDEX), output node (VALUE_ID) | store node id (int, subgraph-local) | CONCORD | NOVEL(`DISPATCH_STORE`, (the OUTPUT member cell,)) on a row with no `NEW`: the store id is `_dispatch_subgraph`'s local counter, not a MINTED id |
| `hierarchy_value` | hierarchy scope (SCOPE, minted per `assign_hierarchy_ids` call), closure (INDEX), local (VALUE_ID) | global id (int) | CONCORD | DERIVED(the local value's identity cell in its function's canonical scope; for a key reached through a binding also the `call_argument_operand` or `call_result_projection_concordance` cell that joined it) |
| `hierarchy_global_value` | hierarchy scope (SCOPE), global (VALUE_ID) | tuple of member `Ref`s | CONCORD | DERIVED(every member `hierarchy_value` cell).  Global ids are dense `enumerate` positions, consumed dense: DERIVED, not minted (plan 60's `canonical_value` argument applies verbatim) |
| `call_binding` | caller scope (SCOPE), call node (VALUE_ID) | `CallBinding(callee reference, resolution: LINKED_PROCESS_GRAPH / CONSTRUCTOR / EXTERNAL)` or `Unresolved(CALLEE_UNRESOLVED)` | CONCORD | DERIVED(call node identity cell, the callee's `function_address` cell) |
| `scope_registry` (N5) | label (LABEL), serial (INDEX) | bool | CONCORD | NOVEL(`MINT_SCOPE`, ()) |

### A1.2 Existing pages re-declared (17)

| page | row fields | fact | provenance after step 4 |
|---|---|---|---|
| `formal_literal` | authored function (NAME), parameter (NAME) | `("proven", value)` or `Unresolved(FORMAL_LITERAL_CONFLICT)` | DERIVED(the `planner_specialization` cell of the copy that proved it, the caller's `call_argument_operand` cell).  The `("conflicting", ...)` fact becomes the Unresolved; the `try/except: pass` goes |
| `formal_shape` | authored function (NAME), parameter (NAME) | `("proven", extents, dtype)` or `Unresolved(FORMAL_SHAPE_CONFLICT)` | DERIVED(`planner_tensor_descriptor` cell, `call_argument_operand` cell).  A third caller's shape is a REVISE with a changed source, no longer lost |
| `proven_literal` | function (NAME), value (VALUE_ID) | literal | DERIVED(the operand identity cells the fold's `known` held for this node, and the `planner_specialization` cell when an Input fed it) |
| `proven_shape` (planner writers only) | as today | as today | `record_proven_shape(..., source: Ref)` gains the source; `_fold_callsite_structural_values` Input branch passes the `planner_tensor_descriptor` cell; `publish_call_result_shape` passes the callee return's `proven_shape` cell (today it writes no edge at all -- census 20 site 6); `_tensor_descriptor` and `_propagate_callsite_tensor_specializations` pass the `shape_transformation_concordance` edge cell written in the same operation |
| `source_control_specialization_concordance` | function (NAME), retained control (VALUE_ID) | dict as today | DERIVED(the `proven_literal` cell of the test, the control node identity cell); when the test is absent from `known` the fact is `Unresolved(PREDICATE_NOT_KNOWN, read=(control cell,))` -- the truthy sentinel no longer records `selected_role = "body"` |
| `structural_specialization_fixed_point` | function (NAME), round (INDEX) | (digest, changed, mutations) | DERIVED(the `proven_literal` / `source_control_specialization_concordance` cells posted in the round); a round record with no posts derives from the previous round's cell |
| `callsite_return_specialization`, `callsite_tensor_result_specialization`, `tensor_shape_settlement_concordance` | as today | as today | DERIVED(`planner_tensor_descriptor` cells of the callee formals, the `proven_shape_contract_of` cell it restates) |
| `callsite_projection_specialization`, `callsite_projection_identity_concordance` | (caller scope, callsite VALUE_ID, index INDEX, leaf VALUE_ID) -- **`id(caller.G)` removed** from the scope | as today | DERIVED(the callee return `planner_tensor_descriptor` cell, the leaf identity cell) |
| `source_callsite_activation_concordance` | caller scope (SCOPE), callsite (VALUE_ID) | as today | DERIVED(`call_binding` cell, `recursion_table`'s row -- see A5 R6) |
| `operand_position_orphan`, `consumer_operand`, `item_operand`, `call_argument_operand` | as today | as today | DERIVED(the `lexical_read_binding` cell at the same position; for structure rows the consumer and operand identity cells) |
| `call_result_projection_concordance` | call node (VALUE_ID), projection (VALUE_ID) | path tuple | DERIVED(the `callsite_projection_specialization` cell that materialized the projection, the call node identity cell) |
| `source_precision_region_concordance` | as today | as today | DERIVED(the `source_value_class_concordance` and `source_precision_boundary_concordance` cells the members came from) |
| `aggregate_ledger` | as today, mode REVISE | as today | DERIVED(node identity cell); each lookup is a revision, history kept (today: silent `set(row, 0)` overwrite) |
| `operator_result_type_concordance` | hierarchy scope (SCOPE), `hierarchy_value` cell (PAGE_REF), region name (LABEL) | (opcode, "bool") | NOVEL-free: DERIVED(the output's `hierarchy_value` cell); the ephemeral `closure_id` leaves the key |
| `loop_carried_binding` | read scope (SCOPE), loop (VALUE_ID), updated (VALUE_ID), initial (VALUE_ID) | binding names | DERIVED(the `name_binding` cells of `initial` and `updated` for each name, the loop construct's identity cell) |
| `loop_region_membership` | read scope (SCOPE), loop (VALUE_ID) | tuple of `deployment_region` `Ref`s (today: ordinals) | DERIVED(those region cells, the loop construct cell) |
| `loop_result_port_binding` | read scope (SCOPE), port (VALUE_ID) | binding name | DERIVED(the port's `canonical_value` NOVEL row cell, the `loop_carried_binding` cell) |
| `loop_continuation_rewire_concordance` | read scope (SCOPE) -- **not `function_name`** -- consumer (VALUE_ID), role (LABEL), ordinal (INDEX) | (old, new, binding) | DERIVED(the consumer's `lexical_read_binding` cell at that position, the port's row cell, the `identity_transition` cell `_set_operands` posts for the rewrite) |

`shape_transformation_concordance` / `_dependents` / `_state` keep
`record_shape_transformation` (step 1 re-expressed it).  `transformation_*`
(`TransformationLedger`) is untouched: its `proof` becomes a `Ref` in step 8
with the book-backed tables.

## A2. Posts, writer by writer

### A2.1 `planner_specializations`

`_propagate_callsite_planner_specializations`: for every `(reference,
parameter)` in `candidates`, one post on `planner_specialization`, row
`(callee scope, parameter)`:

- all callsites agree: fact `SpecializationFact(first, LITERAL)` (or
  `DEFAULT` when every contribution was an omitted argument), DERIVED(for
  each contributing callsite: the argument node's identity cell and, for a
  Constant, the `source_span` cell `node_identity_cell` resolves to; for a
  default the default expression's span cell via `_post_source_span`);
- any `dynamic_argument`: `Unresolved(SPECIALIZATION_DYNAMIC_ARGUMENT,
  read=(the argument node cells,))`;
- inconsistent: `Unresolved(SPECIALIZATION_CALLSITES_DISAGREE, read=(...))`.

Today the two `continue`s record nothing; both become rows so a later
`_source_static_value` miss is visible.  `_source_static_value` and
`_source_static_literal` read `latest(row)` (the dict stays as a read view
materialized from `scope_rows(callee scope)`, `Unresolved` skipped).  A
`_source_static_literal` `ValueError` on a candidate becomes
`Unresolved(SPECIALIZATION_NOT_SOURCE_STATIC)` at that candidate's row.

Copy writers.  `_callsite_specialized_shell_type` and `call_result_descriptor`
build a fresh dict for the specialized copy: `extract_clean_process_subgraph`
already forks the read scope through `fork_read_scope` (stage
`READ_SCOPE_FORK`, DERIVED copies), so the copy's `planner_specialization`
rows are posted under the forked scope DERIVED(the argument node cells of
THIS callsite) -- they are callsite facts, not accumulators, exactly as the
comment above the `copy.deepcopy(specializations)` says.
`propagate_bound_planner_specializations` posts per reachable callee
`SpecializationFact(value, BOUND)` DERIVED(the caller's `planner_specialization`
cell for the binding it forwards, the call node's identity cell).

`_publish_formal_literal`: one CONCORD post, then on a differing value a
REVISE to `Unresolved(FORMAL_LITERAL_CONFLICT, read=(both specialization
cells,))`.  `_proven_formal_literal` returns `_FORMAL_LITERAL_CONFLICT` for
an `Unresolved`, unchanged to its callers.

### A2.2 `planner_tensor_descriptors`

`_propagate_callsite_tensor_specializations` (shared callee `setdefault` +
`update`) and `_apply_callsite_tensor_descriptors` (wholesale replace): each
`additions` entry is a REVISE post on `planner_tensor_descriptor`
DERIVED(per agreeing callsite the argument node's `proven_shape` cell when
one exists, else its `shape_transformation_state` cell, else the node
identity cell alone); disagreeing descriptors post
`Unresolved(DESCRIPTORS_DISAGREE)`.  The dict is materialized from the
page after every batch of posts (`_apply_callsite_tensor_descriptors` keeps
its `callsite_tensor_descriptor_names` tuple as the set of rows).
`_publish_formal_shape` mirrors `_publish_formal_literal`.

The `_fold_callsite_structural_values` Input branch that writes
`proven_shape` from `planner_tensor_descriptors[binding]` passes the
`planner_tensor_descriptor` cell to `record_proven_shape` (census 20 site
5).  `publish_call_result_shape` passes the callee's return `proven_shape`
cell (site 6).

### A2.3 `_dispatch_metadata_cache`

`strategize_shell_deployment` mints a planning scope (`mint_scope("plan")`
through N5) at the top of its selection phases and stores it as
`graph.G.graph["planning_scope"]`.  `_dispatch_metadata_node_classifier`
posts one `executable_node` row per classified node under that scope,
CONCORD, DERIVED(node identity cell plus the rule cell A1.1 lists).  The
`MetadataRule` enum names the branch of `_is_dispatch_metadata_node_impl`
that decided (`PRECISION_SELECTOR`, `CALL_RESULT_PROJECTION`, `AST_METADATA`,
`SCALAR_DEFERRAL`, ... one per `return` in that function), so the
2026-09-30 scalar-deferral decision is a row with a rule, not a cache
entry.  The dict stays as the read view for the current planning scope;
its fingerprint is replaced by the scope: a new pass mints a new scope and
classifies again (which is what "obtain a new classifier after changing the
graph's structure" already asks for).  The uncovered-node check in
whole-program compile reads `scope_rows(planning scope)`.

### A2.4 Regions

`_dispatch_subgraph`: after `node_ids`, `deployment_outputs` and
`store_nodes` are known, post `deployment_region` (region index = the
planner's ordinal for this subgraph, passed in by the caller that numbers
regions) DERIVED(every member's `executable_node` cell); one
`deployment_region_member` per node with role INPUT / NODE / OUTPUT
DERIVED(member's `executable_node` cell, the region cell); one
`dispatch_store` NOVEL per store node.  The five `subgraph.G.graph[...]`
tuples stay, materialized from the rows (byte-identical), so
`_structural_region_program_from_subgraph`, `_subgraph_reduction_digest`
and the emitters are unchanged.  `loop_composer.analyze_shader_loop_reductions`
posts `loop_region_membership` DERIVED(the `deployment_region` cells of
`region_indices`, the loop construct cell); `place_loop_carried_region_producers.owned_regions`
returns the marker-region fallback only after posting
`Unresolved(REGION_OWNERSHIP_UNKNOWN, read=(loop cell,))` on that row.

### A2.5 `HierarchyValueTable.correlations`

`assign_hierarchy_ids(root)` mints a hierarchy scope (N5) and, after the
union-find, posts one `hierarchy_value` row per `(closure_id, local_id)` key
DERIVED(the local value's identity cell -- the closure's function graph's
`canonical_value` row, reached through `PlanClosure.name` -> function shell
-> `lexical_read_scope` -- plus the `call_argument_operand` cell for a key
added by `argument_bindings` and the `call_result_projection_concordance`
cell for one added by `result_bindings`), and one `hierarchy_global_value`
row per equivalence class DERIVED(its members' cells).  `HierarchyValueTable`
is constructed from `scope_rows(hierarchy scope)` and gains the scope as a
field so `_build_hierarchical_glsl_artifact` can key
`operator_result_type_concordance` by the `hierarchy_value` cell.
`plan_region_to_ssa_instrs.fresh_like` becomes `post(SSA_VALUE, (control
scope, NEW), ..., Novel(FRESH_LIKE, (result hierarchy cell,)))` -- the
`SSA_VALUE` page is step 5's (B1); step 4 declares it if it lands first.

### A2.6 Loop plan records

`materialize_retained_loop_ports.add_port`: the port is a new graph node
after the canonical relabel, so `node_identity_cell` would post it
`Unsourced(SYNTHESIZED_NO_SOURCE)`.  Instead `add_port` posts the port's
`canonical_value` row `(read scope, port id)` NOVEL(`LOOP_RESULT_PORT` or
`LOOP_STATE_PORT`, (the `loop_carried_binding` cell of the binding it
continues,)) -- a row with no `NEW`, because the id is the graph's dense
`next_process_value_id`, not a MINTED id -- then `loop_result_port_binding`
DERIVED(that row, the carried cell).  `add_constant` in the loop composer
likewise posts its constant node's `canonical_value` NOVEL(`LOOP_COMPOSER_CONSTANT`,
(loop construct cell,)).  `loop.carried_bindings` come from node attribute
`loop_carried_bindings` written by the reducer; `analyze_shader_loop_reductions`
posts `loop_carried_binding` DERIVED(the `name_binding` cells of the initial
and the updated version for each name, the loop cell).

`rewire_continuation`: the operand rewrite goes through
`_set_operands(graph, consumer, parents, cause=LOOP_CONTINUATION_REWIRE,
same={old: new})` instead of assigning `data["parents"]` directly, so
`identity_transition` records the move (census 20 "bypasses `_set_operands`",
MASTER A3a "open"); `loop_continuation_rewire_concordance` is keyed by the
read scope and DERIVED(the consumer's `lexical_read_binding` cell, the
port's `canonical_value` cell, the transition cell).  The graph-root and
`pending_plans` retargets (`_retarget_plan_value_ids`,
`_retarget_cached_value_ids`) stay as they are; they move cached copies of
ids, which the transition cell now explains.

### A2.7 `identity_table` after the reducer (plan 60 R1)

The dict is `name_binding`'s read view.  Every post-reduction mutation
becomes a REVISE on the canonical `name_binding` row it changes, and the
view is re-materialized from the page after each:

| writer | today | post |
|---|---|---|
| `remove_node` | filters the id out of every history (positions shift) | for each `(name, version)` whose latest fact names the removed id: REVISE `Unresolved(BINDING_VERSION_REMOVED, read=(previous cell, the removed node's identity cell))`.  The view skips `Unresolved` rows, so the tuple is today's filtered tuple |
| `replace_alias` | substitutes and `dict.fromkeys` dedups | REVISE fact `BindingFact(source_id, ...)` DERIVED(previous cell, the `identity_transition` cell `_set_operands(cause=callsite_fold_replace_alias)` posted, the source node's identity cell).  The view dedups first-occurrence, as today |
| the fold's `history[0]` alias branch and its fixed-point tail | rewrite in place | same as `replace_alias`, cause `PROJECTION_TO_LEAF` |
| `_alias_projection_to_member` | rewrite in place; also `return_value_nodes` / `return_slot_values` | `name_binding` REVISE as above; `return_site_slot` REVISE DERIVED(old slot cell, the transition cell) per plan 70 section 4 |
| `_apply_callsite_tensor_descriptors` / `_apply_callsite_aggregate_descriptors` | read the dict to find the one Input, rewrite after materialization | read `name_binding`; post the materialized leaves' `name_binding` rows DERIVED(the aggregate descriptor's `planner_tensor_descriptor` cell) |
| `_retarget_all_cached_value_ids` | rewrites `return_record_field_states` | `return_site_field_state` REVISE DERIVED(old cell, transition cell) (plan 70, section 4, third row) |
| `loop_composer` `add_constant`; `materialize_retained_loop_ports` post-loop table rewrite | new names / retargeted versions | `name_binding` posts DERIVED(the NOVEL constant row / the port row) |
| `_synthetic_device_scalar_shell` | a throwaway wrapper graph with `{"result": (call_id,)}` and three nodes numbered `max(nodes)+1` | mint a scratch scope (N5); post the three nodes' `canonical_value` rows NOVEL(`SYNTHETIC_DEVICE_SCALAR_PREDICATE`, (predicate node cell,)) and one `name_binding` row DERIVED(the call node row); the wrapper's dict is the view of that scope.  The function's own docstring says it is UNVERIFIED; the posts make its verdict traceable if it ever fires |

The structural-fold return selection (the block that sets `graph.roots =
[selected_return_id]`) posts `Unresolved(RETURN_SITE_UNREACHABLE)` on each
pruned site's `return_site_slot` rows, as plan 70 section 4 specified.

### A2.8 Call binding

`link_process_graph_functions` posts `call_binding` per resolved call
DERIVED(call node identity cell, `function_address` cell of the callee);
an unresolved call posts `Unresolved(CALLEE_UNRESOLVED)` and stays in
`unresolved_ast_calls`.  `plan_callsites` posts
`source_callsite_activation_concordance` DERIVED(the `call_binding` cell and
the recursion unit's row -- see R6 for `recursion_table`).  Node attributes
`callee_ref` / `callee_resolution` stay as read views.

## A3. Readers and how they keep working

| reader | reads today | reads after step 4 |
|---|---|---|
| ~20 readers of `planner_specializations` (A0) | the dict | the dict, materialized from `planner_specialization.scope_rows(callee scope)` after every batch of posts; `_source_static_value` / `_source_static_literal` / `_fold_callsite_structural_values.evaluate` additionally read `latest(row)` directly where they already import the book |
| `_tensor_descriptor` gates, the fold, `specialization_state` digest | `planner_tensor_descriptors` | the dict, materialized likewise |
| `strategize_shell_deployment`, the uncovered-node check | `_dispatch_metadata_cache` | the classifier closure (unchanged signature) backed by `executable_node.scope_rows(planning scope)`; the dict stays as its memo |
| emitters, `_structural_region_program_from_subgraph`, `_subgraph_reduction_digest` | the five subgraph tuples | the same tuples, materialized from `deployment_region_member` / `dispatch_store` |
| every hierarchical composition stage; other shells' `hierarchical_effective_value_table` | `HierarchyValueTable.correlations` | the same tuple, built from `hierarchy_value.scope_rows(hierarchy scope)`; `global_id()` unchanged |
| `_enter_loop_state`, `_publish_loop_result_ports`, `_split_region_captures_by_binding`, `_loop_scope_rebinds` | `loop_carried_binding.latest`, `loop_result_port_binding.latest` | same rows, same facts |
| `place_loop_carried_region_producers.owned_regions` | `loop_region_membership` ordinals | the fact is a tuple of `Ref`s; the reader maps `ref.row[1]` to the ordinal.  **One reader changes** |
| `fortran_c_shell` readers of `structurally_specialized_conditional_node_ids` | the shadow tuple | the tuple, materialized from `source_control_specialization_concordance.scope_rows(function)` rows whose fact is not `Unresolved` |
| ~110 readers of `identity_table` | the dict | the dict, re-materialized from `name_binding` after each A2.7 post (skip `Unresolved`, dedup first occurrence) |
| audit `_source_callsite_activation_findings`, `_callable_identity_findings`, `_planning_alias_transition_findings` | their pages | unchanged rows; `_planning_alias_transition_findings` gains the `LOOP_CONTINUATION_REWIRE` cause |

## A4. Ordered edit list

E1  `identity_concordance.py`: `mint_scope` posts NOVEL on `scope_registry`
    (N5) with a `stage` argument defaulting to `RAW_STAGE` for callers not
    yet migrated; `record_proven_shape(..., source: Ref | None = None)`
    posts DERIVED when given, `Unsourced(RAW_PRIMITIVE)` otherwise (the
    latch lists the callers that still pass none: `tensor_ssa_lowering`'s
    four are step 6's).
E2  `concordance_declarations.py`: the step 4 section (A1).
E3  `_propagate_callsite_planner_specializations`: the three posts of A2.1;
    materialize the dict per callee.  `_source_static_literal`: the
    `ValueError` sites post `SPECIALIZATION_NOT_SOURCE_STATIC` when called
    from the propagate loop (pass the candidate row).
E4  `_callsite_specialized_shell_type`, `call_result_descriptor`,
    `propagate_bound_planner_specializations`: forked-scope posts for the
    copy; `_publish_formal_literal` / `_publish_formal_shape` through
    `post`, `try/except` removed.
E5  `_propagate_callsite_tensor_specializations`,
    `_apply_callsite_tensor_descriptors`: `planner_tensor_descriptor`
    posts; `publish_call_result_shape` and the fold's Input branch pass
    sources to `record_proven_shape`.
E6  `link_process_graph_functions`: `call_binding` posts.  `plan_callsites`:
    `source_callsite_activation_concordance` DERIVED.
E7  `strategize_shell_deployment`: mint the planning scope;
    `_dispatch_metadata_node_classifier` posts `executable_node`;
    the uncovered-node check reads the scope.
E8  `_dispatch_subgraph`: region, member and store posts; tuples
    materialized.  Its callers pass the region ordinal.
E9  `_fold_callsite_structural_values`: `replace` posts `proven_literal`
    DERIVED; the IfExp branch posts `source_control_specialization_concordance`
    DERIVED or `PREDICATE_NOT_KNOWN`; `structurally_specialized_conditional_node_ids`
    materialized; the fixed-point record DERIVED; `remove_node`,
    `replace_alias`, the projection-alias branch: `name_binding` REVISE
    posts (A2.7); the return selection posts `RETURN_SITE_UNREACHABLE`.
E10 `_alias_projection_to_member`, `_apply_callsite_aggregate_descriptors`,
    `_retarget_all_cached_value_ids`: A2.7 posts; the three in-place dict
    rewrites deleted in favour of the views.
E11 `_concord_consumer_operands`, `_concord_item_operands`,
    `_concord_call_argument_operands`, `_build_shell_hierarchy_plan`
    (`call_result_projection_concordance`), `_publish_callsite_return_members`,
    `_repair_missing_aggregate_leaf_projections`, `_precision_indivisible_node_groups`,
    `_record_aggregate_ledger_lookup`: DERIVED posts per A1.2; `id(caller.G)`
    leaves the projection scope.
E12 `assign_hierarchy_ids`: hierarchy scope, `hierarchy_value` /
    `hierarchy_global_value` posts; `HierarchyValueTable(correlations,
    scope)`; `plan_region_to_ssa_instrs`: `operator_result_type_concordance`
    re-keyed; `fresh_like` NOVEL.
E13 `loop_composer.py`: `add_port` and `add_constant` NOVEL `canonical_value`
    rows; `loop_result_port_binding`, `loop_carried_binding`,
    `loop_region_membership` DERIVED; `rewire_continuation` through
    `_set_operands` and the re-keyed rewire page; the post-loop
    `identity_table` rewrite as `name_binding` posts.
E14 `control_source.place_loop_carried_region_producers.owned_regions`:
    read `Ref` facts; post `REGION_OWNERSHIP_UNKNOWN` before the fallback.
E15 `_synthetic_device_scalar_shell`: scratch scope and NOVEL rows.
E16 `identity_concordance.py` audit: register the ten new pages with the
    generic findings (registration only); `_planning_alias_transition_findings`
    accepts the new cause.
E17 `tools/compiler_probes/probe_planner_specialization_chain.py` (A7) and
    its `TEST_BASELINE_AND_HAZARDS.md` line.
E18 Delete the read views' rebuild sites once `git grep` finds no reader of
    `planner_specializations`, `planner_tensor_descriptors`,
    `_dispatch_metadata_cache`, `structurally_specialized_conditional_node_ids`
    that bypasses the pages; `identity_table` stays a view (its readers are
    plan 60's).

Writer sites routed: 33 book sites in `glsl_deployment_strategy.py`, 5 in
`loop_composer.py`, 1 + 1 mint in `hierarchical_plan.py` (census 20's 42),
plus the no-book writers `_propagate_callsite_planner_specializations`,
`_dispatch_metadata_node_classifier`, `_dispatch_subgraph`,
`assign_hierarchy_ids`, `link_process_graph_functions`, `add_port`,
`add_constant`, `_synthetic_device_scalar_shell`, and the eight
`identity_table` mutators.  Pages: 10 new, 17 re-declared.

## A5. Risks

R1  **`fork_read_scope` copies every page row under the source scope.**
    `planner_specialization` and `planner_tensor_descriptor` rows are keyed
    by the callee's read scope, so a fork copies the SHARED callee's
    accumulated specializations into the copy -- exactly the
    `setdefault().update()` leak the comment in `_callsite_specialized_shell_type`
    describes.  The copy writers must therefore post their callsite facts
    as REVISE over the forked rows (a changed source: this callsite's
    argument cells) and the read view must take `latest`, never the union.
    Alternatively key the two pages by a specialization scope minted per
    copy (N5) instead of the read scope; the plan proposes the REVISE and
    asks (A9).
R2  **The planning scope is minted per pass.**  `_dispatch_metadata_node_classifier`
    is called from two places with independent lifetimes
    (`strategize_shell_deployment`, the whole-program uncovered check).
    Rows under different planning scopes for one graph are legal (the
    classification is per pass) but a reader that joins them by node id
    alone reintroduces a value-id match.  Every reader takes the scope
    from `graph.G.graph["planning_scope"]`.
R3  **`_subgraph_reduction_digest` and pickled graphs.**  The five subgraph
    tuples are materialized (plain tuples), so the `cloudpickle` digest is
    unchanged.  Nothing in A2 stores a `Ref` on `G.graph` or node data
    (plan 70 R1 discipline); `HierarchyValueTable` gains a scope tuple,
    not a `Ref`.  `_CALLSITE_SHELL_TYPE_CACHE` caches planned shells across
    calls of `_callsite_specialized_shell_type`: a cached shell's rows were
    posted in the compile that built it; a second compile reading the cache
    finds no rows.  Mitigation: the cache key already includes the callee
    identity; add the book identity (`id(current_identity_book())`) so a
    new compile re-plans.
R4  **`assign_hierarchy_ids` needs every local id's identity cell.**  A
    `PlanLine` input that is a planner-minted node (a port, a constant, a
    projection leaf materialized by `_apply_callsite_aggregate_descriptors`)
    has a `canonical_value` row only if A2.6 / A2.7 posted one.  A key with
    no cell posts `hierarchy_value` DERIVED(the closure's function span
    cell) plus `Unresolved(SYNTHESIZED_NO_SOURCE)` on `canonical_value`
    for that id, so the audit lists which planner writer still mints a node
    without a row.  Expect the `_publish_callsite_return_members` leaf
    projections and `_expand_specialized_unbroadcast_identity` bodies to be
    the residue.
R5  **REVISE refusals in fixed points.**  `_fold_callsite_structural_values`
    and `_propagate_callsite_tensor_specializations` iterate to a fixed
    point and re-derive the same fact from the same cells; `post` refuses a
    REVISE with no changed source.  Every fixed-point writer posts only when
    its fact or its source set changed (the fold already tests
    `same_structural_value(known[node], value)` before recording), and
    catches `ConcordanceRefusal` nowhere: a refusal is a writer that
    re-derived without cause and is fixed at the writer.
R6  **`recursion_table` is not on the book.**  `source_callsite_activation_concordance`
    wants the recursion unit as a cell.  Step 4 posts it DERIVED(`call_binding`
    cell) only and records the recursive-unit fact in the fact tuple as
    today; a `recursion_unit` page is step 8 territory with the book-backed
    tables.  Said here so the edge is known to be short.
R7  **Volume.**  One `executable_node` row per node per planning pass, one
    `hierarchy_value` row per scoped value per shell.  `scope_rows` is O(1)
    per read; `render_identity_book` grows.  Measure on the audit
    `controller` case before the Woodshop compile.

## A6. What step 4 needs from steps 2-3

1. `canonical_value` rows for every reducer node (landed) and the
   `node_identity_cell` seam accepting a node the planner just added
   (A2.6 posts the row first, so the seam finds it).
2. `name_binding` canonical rows (landed): A2.7's REVISE posts name them
   as previous cells.
3. `source_span` cells for literals and defaults (landed;
   `_post_source_span` and `_parameter_default_expression` exist in the
   reducer and are importable).
4. `lexical_read_binding` and `identity_transition` DECLARED (step 5's B1.3;
   step 4's E11 and E13 derive from `lexical_read_binding` cells and post
   `identity_transition` through `_set_operands`).  If step 4 lands first,
   E13's `_set_operands` call still writes raw (tagged) and the
   `loop_continuation_rewire_concordance` post names only the two cells
   that exist.
5. `return_site_slot` / `return_site_field_state` (landed) for A2.7's
   return rewrites.
6. `function_address` (landed) for `call_binding`.

## A7. The seconds-long proof

Existing: `python -u tools/audit_identity_concordance.py view` and
`controller` (the dt controller `root` lowered alone; the `rollback` fold
of `_source_static_value`'s comment is in `step_with_dt_control_used`, not
in this case -- the new probe below carries that shape);
`python -u tools/compiler_probes/probe_scalar_native_correctness.py` (the
scalar-deferral rule: every all-scalar program must now show an
`executable_node` row with rule `SCALAR_DEFERRAL` or `EXECUTABLE` for each
scalar expression); `probe_annotated_scalar_parameter.py` and
`probe_struct_intake.py` (unchanged pass).

New, seconds-long: `tools/compiler_probes/probe_planner_specialization_chain.py`,
same shape as `probe_branch_written_field.py` (`ExtractionContract(...)
.with_program_abi`, `lower_ast_source_to_ssa(source, "step", ...,
runtime_closure_only=True)`).  Source: a callee `advance(x: float,
rollback: bool = False) -> float` whose body is `saved = x * 2.0 if
rollback else 0.0; return x + saved`, and a root `step(x: float) -> float`
calling `advance(x, True)` once.  Reading only the book, it checks:

1. `planner_specialization` row `(advance scope, "rollback")` fact
   `SpecializationFact(True, LITERAL)` with an inbound edge to the `True`
   Constant's identity cell and through it to a `source_span` row whose
   kind is `Constant`;
2. `source_control_specialization_concordance` row for the `IfExp` DERIVED
   from the `proven_literal` cell of the test, which is DERIVED from (1);
   `structurally_specialized_conditional_node_ids` equals the page view;
3. `executable_node` rows under one planning scope for every node of
   `advance`; each `deployment_region` row DERIVED from its members; the
   `deployment_nodes` tuple equals the member view;
4. `hierarchy_value` rows for `advance`'s formal `x` and `step`'s argument
   `x` share one `hierarchy_global_value`, and the formal's row has an edge
   to the `call_argument_operand` cell;
5. `call_binding` row for the call node DERIVED from `advance`'s
   `function_address` cell;
6. zero `unsourced-fact` groups for stages `PLANNER_SPECIALIZATION`,
   `PLANNER_STRUCTURAL_FOLD`, `PLANNER_DISPATCH_CLASSIFICATION`,
   `PLANNER_REGION_CARVE`, `PLANNER_HIERARCHY`, `PLANNER_CALL_BINDING`;
   the probe prints the before/after `unsourced-fact` counts by stage so
   the drop is measured.

The dt-system `rollback` fold itself (the 52-leaf `copy_shallow` case) is
the user's run; the probe hands over the exact shape it will show.

## A8. Unsourced worklist groups step 4 retires

Group keys are `(page, stage, unit)` as `_unsourced_worklist` prints them.
Retired: `("formal_literal", raw_primitive, raw row)`, `("formal_shape",
...)`, `("proven_literal", ...)`, `("proven_shape", ...)` for the planner's
five writers, `("source_control_specialization_concordance", ...)`,
`("structural_specialization_fixed_point", ...)`, `("callsite_return_specialization",
...)`, `("callsite_tensor_result_specialization", ...)`,
`("tensor_shape_settlement_concordance", ...)`, `("callsite_projection_specialization",
...)`, `("callsite_projection_identity_concordance", ...)`,
`("source_callsite_activation_concordance", ...)`, `("operand_position_orphan",
...)`, `("consumer_operand", ...)`, `("item_operand", ...)`,
`("call_argument_operand", ...)`, `("call_result_projection_concordance",
...)`, `("source_precision_region_concordance", ...)`, `("aggregate_ledger",
...)`, `("operator_result_type_concordance", ...)`, `("loop_result_port_binding",
...)`, `("loop_continuation_rewire_concordance", ...)`, `("loop_region_membership",
...)`, `("loop_carried_binding", ...)`, `("scope_registry", raw_primitive,
raw row)` (N5), and the `("canonical_value", reduction, cell)`
`SYNTHESIZED_NO_SOURCE` entries for ports and loop-composer constants.
`unsourced-identity`: `plan_region_to_ssa_instrs.fresh_like`.

Left for later steps: `transformation_decision` / `_event` / `_rejection`
(step 8), `sequence_row_layout_concordance` (step 6), the four
`record_proven_shape` callers in `tensor_ssa_lowering` (step 6).

## A9. Held for the user

1. R1: key `planner_specialization` / `planner_tensor_descriptor` by the
   forked read scope with REVISE over the copied rows (proposed), or by a
   specialization scope minted per copy.
2. Whether `PREDICATE_NOT_KNOWN` should stop the fold (a folded IfExp with
   an unknown test is a structural decision from absence) or only record.
   The plan records and does not fold that IfExp (the `selected_role` is
   not chosen), which changes behaviour where the sentinel used to select
   `body`; A7's probe has a known test and does not exercise it.

---------------------------------------------------------------------------

# PART B -- STEP 5: the control SSA builder

## B0. What is being moved, observed in the code

`_ControlSSABuilder` (`precompile_to_ssa.py`) keeps the lowering
environment in private state and hands it to `Function.metadata` at
`finish`.  Observed writers:

| state | writers (functions) | count |
|---|---|---|
| `external_values` (graph id -> `SSAValue`) | `__init__` (uniforms, parameter seeds), `external_value` (field-effect incumbent, sequence length load, provisional argument via `_value_from_meta`, dtype refinement), `produced_value` (region result, versioned in-place write), `emit_region_call` and the call lowering (results), `lower_control_expression` (result; the `item` scalar merge), `_lower` for `ScalarFieldWriteBlock`, `SequenceQueryBlock`, mutation and row-base blocks, `lower_conditional` (three `update(values_before_*)` restores, `update(published)`, sequence merges), `_publish_loop_result_ports` (ports and equivalence groups), `lower_loop` / `lower_while` / `_complete_loop_latch_carried` (seeds, header Phis, latch updates, induction and target values, `pop` + previous restores, predicate) | the census's 54 assignment sites; `git grep` today lists about sixty write expressions counting `update` / `pop` |
| `value_aliases` / `concorded_value_aliases` | `__init__` (from `value_concordance` -- the `control_value_concordance` page's resolution), `_bind_loop_result_ports_inside_body` (spelling -> updated id while a body is lowered), `_restore_loop_result_port_aliases` | 3 |
| `value_name_histories` | `__init__` from `identity_table` | 1 (a copy) |
| `carried_snapshots` (local to `lower_conditional`) | `lower_conditional` (initial ids of `carried_aliases` and every nested conditional's) | 2 |
| `_carried_port_values`, `_carried_port_groups` | `_publish_loop_result_ports` | 3 |
| `declared_parameter_only_ids` | `__init__` add; `external_value` discard on first use | 2 |
| `region_signatures` | passed in from the two callers in `lower_control_sections_to_ssa` and the fused-program lowering that build `region_signatures[region_index] = (feeds, outputs)` | 3 sites |
| `field_version_values` (lane C, landed) | `_publish_field_version` | 1 |
| `control_identity_receipts` | `lower_control_expression` (`scalar_item_identity`) | 1 |
| `finish` | derives `parameter_names`, `value_names`, `named_outputs`, `carried_port_values`, `value_aliases`, `control_identity_receipts`, `validation_contracts`, `recursion_table`, `sequence_table` into `Function.metadata` | 1 site, 18 reader files (census 30, 3.1) |
| `fresh_value` | `GLOBAL_MONOTONIC_IDS.mint()` for every builder temporary | 168 call sites |

Book writes by the builder today (census 30, 1.1): 22 sites, of which
`identity_transition` (the `_region_feed` rank-0 item merge, a raw
`revise`), `loop_carried_entry`, `loop_entry_state`, `while_carried_test`,
`region_feed_consumer`, `region_capture_binding`, `callsite_argument`,
`loop_result_reconciliation`, `control_uniform_dtype`, `region_value_dtype`,
`control_value_concordance` (four `bind_alias` sites), `field_slot_storage_concordance`,
`sequence_contract_concordance`, `tensor_shape_concordance` (two sites),
`loop_scope` / `loop_scope_inner_transition` (through `declare_loop_scope` /
`rebind_loop_scope_inner`, wrapped in `try/except: pass`).  Lane C's
`ssa_field_version` posts go through `post` already.

Two pages the reducer writes raw and the builder reads are undeclared:
`lexical_read_binding` (`_concord_lexical_reads`; read by `_operand_bindings`,
`_split_region_captures_by_binding`, `_resolve_read`, `rewire_continuation`)
and `identity_transition` (`_set_operands`, whose docstring says its
revise writes "stay raw until it is" declared; `fork_read_scope` and
`_region_feed` also write it).  Observed row shapes on `identity_transition`:
`(scope, consumer, role, ordinal)` from `_set_operands`, `(forked scope,
"scope")` from `fork_read_scope`, `(control scope string, item id)` from
`_region_feed`.  Three shapes cannot share one declared page (B1.3).

## B1. Pages to declare (step 5 section of `concordance_declarations.py`)

Stages: `CONTROL_SSA` (landed), `CONTROL_SSA_ENTRY` (`__init__`),
`CONTROL_SSA_CONDITIONAL`, `CONTROL_SSA_LOOP`, `CONTROL_SSA_REGION`,
`CONTROL_SSA_FINISH`, `REDUCER_READ` (`_concord_lexical_reads`),
`OPERAND_POSITION` (landed).

Transforms (`fresh_value`; N4 marks the variadic ones): `CONTROL_CONST` (1:
the literal's identity cell or the `authored_constant_values` span),
`CONTROL_PREDICATE` (VARIADIC), `CONTROL_EXPRESSION` (VARIADIC: operand
value cells), `CONTROL_CAST` (1), `PHI_CONDITIONAL` (VARIADIC: arm cells
and the `carried_snapshot` cell), `PHI_LOOP_HEADER` (VARIADIC), `PHI_LOOP_EXIT`
(VARIADIC), `PHI_RETURN_MERGE` (VARIADIC), `LOAD` (1), `ADDRESS` (1),
`SEQUENCE_LENGTH_CELL` (1: the `sequence_contract_concordance` cell),
`DESCRIPTOR_CELL` (1), `REGION_CALL_RESULT` (1: the `region_signature`
cell), `VERSIONED_WRITE` (1: the previous binding cell), `LOOP_RESULT_VERSION`
(1), `REGION_FORMAL_SPLIT` (1), `FIELD_SLOT_ACCESS` (1), `TABLE_LOOKUP` (1),
`ROW_COLUMN_PROJECTION` (1).  Every `fresh_value` call passes one.

Reasons: `NO_PRODUCER_AT_USE` (a provisional argument minted because no
region has published the id yet), `NAME_ARM_VERSION_MISSING`,
`WHILE_TEST_NO_READ_EXPRESSION`, `REGION_FEED_NO_OPERAND_ROW`,
`CARRIED_ENTRY_NOT_ATTRIBUTED`, `RETURN_SLOT_UNRESOLVED_ON_EDGE`,
`BINDING_WITHDRAWN`, `NO_OPERAND_POSITION_SCOPE` (plan 70, section 3).

### B1.1 New pages (9)

| page | row fields (kind) | fact type | mode | provenance |
|---|---|---|---|---|
| `ssa_value` | function scope (SCOPE: the builder's `lexical_read_scope`, else `tensor_shape_concordance_scope`), ssa id (VALUE_ID) | `SSAValueFact(dtype, shape, origin: MINTED / ADOPTED_GRAPH_ID)` | REVISE (dtype refinement is a revision with the demanding cell as cause) | `fresh_value`: NOVEL(transform, operands) with `NEW`; `_value_from_meta` (an SSA value whose id IS the graph id): DERIVED(the graph id's `canonical_value` cell, the `region_signature` cell when the region declares it) |
| `control_value_binding` | function scope (SCOPE), graph id (VALUE_ID) | `ControlBinding(ssa_value: Ref, kind: BindingKind)` or `Unresolved` | REVISE (each rebinding is a version) | DERIVED(the graph id's `canonical_value` cell, the `ssa_value` cell, the writer's cause cell -- B2) |
| `control_value_alias` | function scope (SCOPE), alias (VALUE_ID) | `AliasFact(source id, kind: PLANNING / LOOP_BODY_SPELLING / RESTORED)` | REVISE | PLANNING: DERIVED(`control_value_concordance` cell); LOOP_BODY_SPELLING: DERIVED(`loop_result_port_binding` cell, the loop construct cell); RESTORED: DERIVED(the saved PLANNING cell, the loop cell) |
| `carried_snapshot` | function scope (SCOPE), conditional construct (PAGE_REF), initial id (VALUE_ID) | the `control_value_binding` cell current at branch entry (`Ref`) | CONCORD | DERIVED(that binding cell, the conditional construct's identity cell) |
| `carried_port_value` | function scope (SCOPE), port (VALUE_ID) | the exit Phi's `ssa_value` cell (`Ref`) | CONCORD | DERIVED(`loop_result_port_binding` cell, `loop_carried_entry` cell, the Phi's `ssa_value` cell) |
| `declared_parameter` | function scope (SCOPE), graph id (VALUE_ID) | `ParameterDeclaration in {CALL_ONLY, USED}` | REVISE | CALL_ONLY: DERIVED(`name_binding` version-0 cell of the parameter); USED: DERIVED(previous cell, the reading `lexical_read_binding` cell or the `callsite_argument` cell) |
| `region_signature` | function scope (SCOPE), region (INDEX) | `RegionSignature(feeds, outputs)` | CONCORD | DERIVED(step 4's `deployment_region` cell and its INPUT / OUTPUT member cells) |
| `function_parameter` | function scope (SCOPE), name (NAME) | ssa id (int) | CONCORD | DERIVED(`declared_parameter` cell, the `ssa_value` cell, the use cell that kept it) |
| `function_output` | function scope (SCOPE), slot (INDEX) | `(name or None, ssa id)` | CONCORD | DERIVED(the `return_site_slot` cells of every return edge (step 3), the merge Phi's `ssa_value` cell or the single edge's binding cell) |

### B1.2 Existing builder pages re-declared (15)

| page | row fields | provenance after step 5 |
|---|---|---|
| `control_value_concordance` | control name (SCOPE), alias (VALUE_ID) -> resident | DERIVED(alias node's `canonical_value` cell, resident's `canonical_value` cell, the `control.value_aliases` source: `identity_transition` or the planner's alias row).  `bind_alias` is replaced by a CONCORD post |
| `loop_carried_entry` | control scope (SCOPE), loop (VALUE_ID), binding (NAME) -> entry index | DERIVED(the `loop_carried_binding` cell it read) |
| `loop_entry_state` | control scope (SCOPE), loop (VALUE_ID), initial (VALUE_ID) | DERIVED(same cell, the pre-loop `control_value_binding` cell of the initial) |
| `while_carried_test` | control scope (SCOPE), loop (VALUE_ID) | DERIVED(the `lexical_read_binding` cells the predicate read); the "else" branch (a bare predicate id) becomes `Unresolved(WHILE_TEST_NO_READ_EXPRESSION, read=(predicate cell,))` |
| `region_feed_consumer` | as today | DERIVED(`consumer_operand` cells) |
| `region_capture_binding` | as today (already YES) | DERIVED(the `lexical_read_binding` cell, source `canonical_value` cell); the minted formal goes through `fresh_value(NOVEL REGION_FORMAL_SPLIT)` |
| `callsite_argument` | as today | DERIVED(`call_argument_operand` cell, the resolved `control_value_binding` cell); `try/except: pass` removed |
| `loop_result_reconciliation` | as today | DERIVED(the `carried_port_value` cell it read) |
| `control_uniform_dtype` | as today | DERIVED(the uniform's declaration: `scalar_parameter` or `parameter_annotation` cell) |
| `region_value_dtype` | as today, REVISE | DERIVED(the producing `region_signature` cell); a later region's different dtype is a REVISE with a cause, and the audit sees the retype |
| `field_slot_storage_concordance` | as today | DERIVED(`class_field_declaration` cell) |
| `sequence_contract_concordance` | as today | `commit_sequence_contract(..., source: Ref)` DERIVED(the sequence's `canonical_value` cell, its declaration span) |
| `tensor_shape_concordance` (the two builder sites) | as today | DERIVED(the lookup instruction's `ssa_value` cell / the IndexedStore's `canonical_value` cell; the silent max-rank merge over an incumbent becomes a REVISE with the incumbent as a source cell) |
| `loop_scope`, `loop_scope_inner_transition` | as today (YES edges) | through `post`; `try/except: pass` removed |
| `ssa_field_version` (landed) | as today | unchanged; `field_version_values` becomes a read view of `ssa_value` (B3) |

### B1.3 `lexical_read_binding` and `identity_transition`, declared

```
Page LEXICAL_READ_BINDING
  row_fields = (RowField(read_scope, SCOPE), RowField(consumer, LABEL),
                RowField(role, LABEL), RowField(ordinal, INDEX))
  fact_type = str                          # the authored binding name
  mode = CONCORD
```

`consumer` is LABEL, not VALUE_ID, because the reducer writes the
`"occurrence"` and `"return"` rows on the same page; `role` is LABEL
because roles are strings but written through `str(role)` at some sites
and raw at others.  `_concord_lexical_reads` posts the occurrence row
DERIVED(the Name occurrence node's identity cell) stage `REDUCER_READ`, and
each consumer-position row DERIVED(the occurrence row, the consumer node's
identity cell).  The `(scope, "return", "root", position)` rows in the
reducer's return handling derive from the return construct cell.  The
canonical relabel's re-concord (plan 60, 3.1 (e)) is already DERIVED.

```
Page IDENTITY_TRANSITION
  row_fields = (RowField(scope, SCOPE), RowField(consumer, LABEL),
                RowField(role, LABEL), RowField(ordinal, INDEX))
  fact_type = OperandTransition            # Move | Retire | Fork | Append (plan 70, section 3)
  mode = REVISE

Page SCOPE_ORIGIN                          # today's (forked, "scope") rows
  row_fields = (RowField(scope, SCOPE),)
  fact_type = ScopeFork(source_scope, cause: Transform)
  mode = CONCORD                           # NOVEL(FORK_READ_SCOPE, (source scope's SCOPE_ORIGIN cell or its scope_registry cell,))

Page SCALAR_ITEM_MERGE                     # today's _region_feed "merge" rows and lower_control_expression's receipt
  row_fields = (RowField(function_scope, SCOPE), RowField(item, VALUE_ID))
  fact_type = Ref                          # the merged source's ssa_value cell
  mode = CONCORD                           # DERIVED(item_operand cell, the operand's control_value_binding cell)
```

The split is forced by the three observed row shapes.  Readers:
`_planning_alias_transition_findings` reads the `"merge"` rows -- it reads
`SCALAR_ITEM_MERGE` instead; `fork_read_scope` skips `(forked, "scope")`
when copying -- it skips `SCOPE_ORIGIN` (a whole page, no special case);
`_set_operands`' `_OPERAND_POSITION_ROW_PAGES` follow-ups are unchanged.

With `IDENTITY_TRANSITION` declared, plan 70 section 3 executes: `cause`
becomes a `Transform` everywhere (`_set_operands` already accepts one and
records its name), the Append post lands, `_operand_position_scope(graph)
is None` posts `Unsourced(NO_OPERAND_POSITION_SCOPE)`.  The legacy
`cause: str` branch is deleted; the registry then refuses every caller
still passing a string, which is how the loop composer's and the planner's
remaining callers are found.

## B2. Posts, writer by writer

### B2.1 `fresh_value`

`fresh_value(*, dtype, shape, transform: Transform, operands: tuple[Ref,
...])`: `post(SSA_VALUE, (function scope, NEW), SSAValueFact(dtype, shape,
MINTED), stage=<caller's stage>, provenance=Novel(transform, operands),
mode=CONCORD)`; the `SSAValue` is built from `ref.row[1]`.  The builder
keeps `ssa_value_objects: dict[int, SSAValue]` -- ssa id -> the object --
because `SSAValue` objects are mutated in place after creation
(`value.dtype = ...` in `produced_value`, `port.shape = ...` in
`_publish_loop_result_ports`) and every emitted instruction holds the
object.  This is the one private structure step 5 keeps: an object table
keyed by the book's id, not a ledger (plan 70's cursor rule).  Each in-place
mutation is a REVISE on the value's `ssa_value` row DERIVED(previous cell,
the cell that demanded the change).

All 168 call sites pass a transform (B1's list is the closed set; the
registry refuses a site that invents one).  `constant_value`,
`expression_value`, `conditional_branch`'s predicates, the Phi emitters,
`_emit_table_lookup`'s descriptor cells, `_split_region_captures_by_binding`,
`_inject_field_slot_access.fresh`, `lower_class_navigation_to_ssa.Builder`
and `_lower_sequence_mutation_body`'s row-column projection are the
families.  `GLOBAL_MONOTONIC_IDS.mint()` disappears from `precompile_to_ssa.py`.

### B2.2 `external_values` (54 writers) -> `control_value_binding`

Every assignment `self.external_values[gid] = value` becomes
`self._bind(gid, value, kind, cause_cells)`, which posts
`control_value_binding` REVISE row `(function scope, gid)` fact
`ControlBinding(ssa_value cell of value, kind)` DERIVED(the gid's
`canonical_value` cell, the value's `ssa_value` cell, `*cause_cells`) and
then writes the dict (the read view).  Kinds and their cause cells, by
writer:

| writer | kind | cause cells |
|---|---|---|
| `__init__` uniforms | UNIFORM | `control_uniform_dtype` cell |
| `__init__` parameter seeds | PARAMETER_SEED | `name_binding` version-0 cell; posts `declared_parameter` CALL_ONLY |
| `external_value`, field-effect incumbent | FIELD_INCUMBENT | the `ssa_field_version` cell of the incumbent (lane C) or the `reducer_field_state` OBSERVED cell |
| `external_value`, sequence length load | SEQUENCE_LENGTH | the `sequence_contract_concordance` cell |
| `external_value`, no value yet (`_value_from_meta`, appended to `arguments`) | PROVISIONAL_ARGUMENT, fact `Unresolved(NO_PRODUCER_AT_USE, read=(canonical cell,))` | -- a value minted from absence is recorded as unresolved; `produced_value`'s later claim is the REVISE that resolves it |
| `external_value`, dtype refinement | REFINED | previous cell, the demanding read's `lexical_read_binding` cell; the `ssa_value` row is revised too |
| `produced_value` region result / claimed provisional | REGION_RESULT | `region_signature` cell (the region that publishes the id), the region call's `ssa_value` cell |
| `produced_value` versioned in-place write | VERSIONED_WRITE | previous binding cell (the `fresh_value` operand) |
| `emit_region_call`, call lowering results | CALL_RESULT | `region_signature` / `callsite_argument` cells |
| `lower_control_expression` result | CONTROL_EXPRESSION | the operand bindings' cells; the `item` merge posts `SCALAR_ITEM_MERGE` and the receipt tuple becomes a read view |
| `_lower` `ScalarFieldWriteBlock` | FIELD_WRITE | the `ssa_field_version` cell lane C posts (already the source) |
| sequence query / mutation / row-base blocks | SEQUENCE_RESULT | the sequence's `sequence_contract_concordance` cell, the block's node cell |
| `lower_conditional` `update(values_before_body / orelse)` | RESTORED | the `carried_snapshot` cell (B2.3) |
| `lower_conditional` `update(published)` | CONDITIONAL_MERGE | the Phi's `ssa_value` cell, for merged / initial / arm ids |
| `lower_conditional` sequence merges | CONDITIONAL_MERGE | the initial's binding cell |
| `_publish_loop_result_ports` ports and equivalence groups | LOOP_RESULT_PORT | the `carried_port_value` cell |
| `lower_loop` / `lower_while` seeds | LOOP_SEED | pre-loop binding cell, `loop_entry_state` cell |
| header Phis | LOOP_HEADER | the Phi's `ssa_value` cell, `loop_carried_entry` cell |
| `_complete_loop_latch_carried` and latch updates | LOOP_LATCH | the body's binding cell |
| induction / target / predicate | LOOP_CONTROL | the iterable's binding cell, the `while_carried_test` cell |
| `pop(...)` then `[...] = previous` | RESTORED | the saved cell; a `pop` with no previous posts `Unresolved(BINDING_WITHDRAWN, read=(previous cell,))` |

REVISE discipline (R5 of step 4 applies): a writer that re-binds the same
`SSAValue` from the same cells is a no-op at the dict and does not post; a
loop's latch rebinding always has a newer source (the body's cell), so it
is never refused.

### B2.3 `lower_conditional`: snapshots and the name-carried arm

At entry: one `carried_snapshot` row per initial id of `carried_aliases`
and every nested conditional's, fact = `latest_ref(control_value_binding,
(scope, initial))`, DERIVED(that cell, the conditional construct's identity
cell -- `ConditionalBlock.source_node_id` through `node_identity_cell`).

Field-carried arms: lane C's `_carried_field_arm` (landed) reads the
`ssa_field_version` row at the arm cell; unchanged, except that
`field_version_values` becomes a lookup into `ssa_value_objects` by the
row's fact.

Name-carried arms (no `carried_field_cells` entry): today
`external_values.get(true_id, carried_snapshots[initial])`.  Under the
page: `latest_ref(control_value_binding, (scope, true_id))`; if that cell
is stamped after the `carried_snapshot` cell the arm wrote and its value is
the arm; if the arm id has no binding newer than the snapshot AND the arm
id equals the initial id, the arm did not write and the snapshot is the
arm (the same rule as lane C's `arm_cell == other_arm_cell`); otherwise
post `control_value_binding` `Unresolved(NAME_ARM_VERSION_MISSING,
read=(snapshot cell,))` at the arm id and append
`SSALoweringShortfall("control", "carried-name-arm-missing", path, ...)`.
This is design step 5's "snapshot-as-arm refused" for names; the shortfall
kind mirrors lane C's `carried-field-arm-missing`.  The merge Phi is
`fresh_value(PHI_CONDITIONAL, operands=(true arm cell, false arm cell,
snapshot cell))` -- N4 -- and `update(published)` posts CONDITIONAL_MERGE
bindings DERIVED(the Phi cell).

### B2.4 Loops

`_enter_loop_state` posts `loop_carried_entry` / `loop_entry_state` DERIVED
(B1.2); a carried pair the book does not attribute posts `Unresolved(
CARRIED_ENTRY_NOT_ATTRIBUTED, read=(the pair's binding cells,))` instead of
silently keeping the header rebinding.  `_bind_loop_result_ports_inside_body`
posts `control_value_alias` LOOP_BODY_SPELLING per spelling DERIVED(the
port's `loop_result_port_binding` cell, the loop construct cell);
`_restore_loop_result_port_aliases` posts RESTORED DERIVED(the saved
PLANNING cell, the loop cell) -- the undo is an edge, never an erasure.
`_publish_loop_result_ports` posts `carried_port_value` per port
DERIVED(`loop_result_port_binding` cell, `loop_carried_entry` cell, the exit
Phi's `ssa_value` cell); `carried_entry_of`'s id-pair match when there is
no read scope stays only for a program the reducer never saw and posts
`Unresolved(CARRIED_ENTRY_NOT_ATTRIBUTED)`.  `_carried_port_values` and
`_carried_port_groups` become read views of the page.

### B2.5 `_region_feed` and the region signature

`lower_control_sections_to_ssa` (and the fused-program lowering) post
`region_signature` per region DERIVED(step 4's `deployment_region` cell and
its member cells) where they build `region_signatures[region_index]`;
the builder receives the dict as the read view.  `_region_feed`'s fallback
to `(consumer, None, 0)` when no `consumer_operand` row exists posts
`Unresolved(REGION_FEED_NO_OPERAND_ROW, read=(the feed's `region_feed_consumer`
cell,))` on `control_value_binding` for the feed before falling back; the
`_aliases_resolving_to` retry derives from the `control_value_alias` cells
it followed.

### B2.6 `finish`

`finish` posts, then materializes the metadata tuples from the posts:

- `function_parameter` per name in `parameter_value_names`: DERIVED(the
  `declared_parameter` cell (USED or CALL_ONLY), the argument's `ssa_value`
  cell, the use that kept it: a `lexical_read_binding` cell or the
  `function_output` cell for a returned parameter).  `parameter_names` =
  `[(name, fact) for rows in scope]`.
- `function_output` per slot: DERIVED(the `return_site_slot` cells of the
  return edges (step 3; `function_return_edges` carries the block, the
  site's cell is reached through the return construct), the merge Phi's
  `ssa_value` cell or the single edge's binding cell).  A slot whose value
  could not be resolved on an edge (today a `NoneValue` and a `"return"`
  shortfall) posts `Unresolved(RETURN_SLOT_UNRESOLVED_ON_EDGE)` -- the
  shortfall stays.  `named_outputs` = the rows with a name.
- `value_names` = for each `name_binding` name in scope the latest version
  whose `control_value_binding` is resolved and not `declared_parameter`
  CALL_ONLY; no new page (it is a join of two).
- `carried_port_values` = `carried_port_value.scope_rows` (port -> ssa id).
- `value_aliases` = `control_value_alias.scope_rows` latest (the loop-time
  spellings are restored by then, so the snapshot equals today's).
- `control_identity_receipts` = `SCALAR_ITEM_MERGE.scope_rows`.

The metadata dict keeps every key with the same plain-tuple values, so
the 18 reader files (backends included) change nothing.  `_canonicalize_non_dominating_loop_result_uses`
reads `carried_port_values` from the metadata as today and its `_note`
posts `loop_result_reconciliation` DERIVED(the `carried_port_value` cell).

## B3. Lane C's shortfall path and `external_values` as a read view

Landed (`precompile_to_ssa.py`): `_field_version_row`, `_publish_field_version`
(posts `ssa_field_version` CONCORD DERIVED), `_field_state_kind`,
`_carried_field_arm` (reads the arm's version at its cell; posts
`Unresolved(ARM_VERSION_MISSING)` in REVISE mode and appends the
`carried-field-arm-missing` shortfall; returns the snapshot only as the
emitted stand-in, never as a posted arm), the `ScalarFieldWriteBlock`
branch of `_lower` (publishes the version, then writes
`external_values[effect_node_id]` "as a read view during migration"), and
the merge's `_publish_field_version` from the two arm sources and the
merged cell.  `ConditionalBlock.carried_field_cells` and
`ScalarFieldWriteBlock.field_state_cell` (`control_source.py`) carry the
cells; `_ordinary_conditional_control_programs` fills them from the Phi
node's `field_state_arms` / `field_state_cell` attributes.

Step 5 generalizes exactly that shape to every binding: `external_values`
becomes the read view of `control_value_binding` -- `external_values[gid]`
is `ssa_value_objects[latest(row).ssa_value.row[1]]` -- kept as a real
`dict` during migration (the `_TracedExternals` debug subclass in
`__init__` keeps working because `_bind` writes through it) and deleted
once every `self.external_values[...]` read is `self._binding(gid)`.
`field_version_values` is deleted first: `_carried_field_arm` already
handles the "posted by another builder" case by building an `SSAValue`
from the row's fact; with `ssa_value_objects` the object is found by id.
The `SSAValue(int(fact), dtype=snapshot.dtype, ...)` reconstruction stays
for a version posted by a builder over another function scope.

## B4. Readers and how they keep working

| reader | reads today | reads after step 5 |
|---|---|---|
| every `self.external_values.get/[]` read inside the builder (`external_value`, `produced_value`, `existing_value` in `finish`, `_bind_loop_result_ports_inside_body`, ...) | the dict | the dict as read view; migrated read by read to `_binding(gid)` (returns the `SSAValue` or None) |
| `_resolve_read`, `_operand_bindings`, `_split_region_captures_by_binding`, `rewire_continuation.reads_binding`, `lexical_read_binding` (reducer helper) | `lexical_read_binding.latest` | unchanged rows, same facts |
| `_set_operands` follow-ups, `fork_read_scope` | `identity_transition` raw | `post` on `IDENTITY_TRANSITION` / `SCOPE_ORIGIN`; `fork_read_scope` skips `SCOPE_ORIGIN` |
| audit `_planning_alias_transition_findings` | `identity_transition` "merge" rows | `SCALAR_ITEM_MERGE` rows.  **One audit reader changes** |
| `fortran_c_shell`, `glsl_deployment_strategy`, `ir_identities`, `kernel_bank`, `native_package`, the C / Fortran / JavaScript backends, `ssa_self_check`, `ssa_storage_requirements`, autograd, `output_publication` | `Function.metadata["parameter_names" / "value_names" / "named_outputs" / "carried_port_values" / "value_aliases" / "control_identity_receipts"]` | the same keys, materialized from pages at `finish` (B2.6) |
| `_canonicalize_non_dominating_loop_result_uses` | `metadata["carried_port_values"]` | same |
| `tensor_ssa_lowering.call.resident_sequence`, `resolved_concordant_alias_bindings`, `fortran_c_shell` | `control_value_concordance` | same rows (posted CONCORD instead of `bind_alias`) |
| `_loop_scope_findings`, `loop_scope_declarations`, `concord_loop_scope_latch_residents` | `loop_scope` / `loop_scope_inner_transition` | same rows |
| `scalar_return_field_versions` (step 3, landed) | `ssa_field_version` | unchanged |

## B5. Ordered edit list

F1  `identity_concordance.py`: N4 (`VARIADIC` arity); `commit_sequence_contract`,
    `declare_loop_scope`, `rebind_loop_scope_inner` gain `source: Ref`
    and post; `_planning_alias_transition_findings` reads `SCALAR_ITEM_MERGE`.
F2  `concordance_declarations.py`: the step 5 section (B1) including
    `LEXICAL_READ_BINDING`, `IDENTITY_TRANSITION`, `SCOPE_ORIGIN`,
    `SCALAR_ITEM_MERGE`, `OperandTransition` facts.
F3  `topological_reducer.py`: `_concord_lexical_reads` posts DERIVED;
    `_set_operands` posts through `post` (plan 70 section 3 in full;
    `cause: Transform` only); `fork_read_scope` posts `SCOPE_ORIGIN`
    NOVEL and skips that page when copying; the return-row writer of
    `lexical_read_binding` posts DERIVED.  Update every `cause=` caller
    (reducer, `glsl_deployment_strategy`'s `_alias_projection_to_member`,
    `remove_node`, `replace_alias`, `loop_composer`'s callers incl. step
    4's E13).
F4  `_ControlSSABuilder.fresh_value(*, transform, operands)` posts NOVEL;
    `ssa_value_objects`; every one of the 168 call sites names its
    transform.  `_value_from_meta` posts the ADOPTED_GRAPH_ID row.
F5  `_ControlSSABuilder._bind(gid, value, kind, cells)`; the 54 assignment
    sites of B2.2 call it, in this order: `__init__`, `external_value`,
    `produced_value`, `lower_control_expression`, `_lower`, the sequence
    blocks, `emit_region_call`, `lower_conditional`, `_publish_loop_result_ports`,
    `lower_loop`, `lower_while`, `_complete_loop_latch_carried`.
F6  `lower_conditional`: `carried_snapshot` posts; the name-carried arm
    rule and `carried-name-arm-missing` (B2.3); `PHI_CONDITIONAL` operands.
F7  `_bind_loop_result_ports_inside_body`, `_restore_loop_result_port_aliases`:
    `control_value_alias` posts; `__init__`'s alias seeding posts PLANNING
    rows; `_enter_loop_state` DERIVED + `CARRIED_ENTRY_NOT_ATTRIBUTED`;
    `_publish_loop_result_ports`: `carried_port_value`; `lower_while`:
    `WHILE_TEST_NO_READ_EXPRESSION`.
F8  `_region_feed`: `SCALAR_ITEM_MERGE` post; `REGION_FEED_NO_OPERAND_ROW`;
    `lower_control_expression`'s `item` merge posts the same page and the
    receipts list becomes a view.
F9  `lower_control_sections_to_ssa` and the fused lowering: `region_signature`
    posts at the `region_signatures[...] =` sites; the four `bind_alias`
    sites become CONCORD posts; `region_value_dtype`, `control_uniform_dtype`,
    `field_slot_storage_concordance`, `tensor_shape_concordance` sites
    DERIVED; `_note_callsite_arguments` and `_note` without `try/except`.
F10 `finish`: B2.6 posts and materialization.
F11 `_inject_field_slot_access.fresh`, `lower_class_navigation_to_ssa.Builder`,
    `_lower_sequence_mutation_body`: through `fresh_value` with their
    transforms (same file, same counter).
F12 `identity_concordance.py` audit: register the nine new pages; add the
    per-page check `carried-name-arm-missing` beside
    `carried-field-arm-missing` (plan 70, E18).
F13 `tools/compiler_probes/probe_control_binding_chain.py` (B8) and its
    `TEST_BASELINE_AND_HAZARDS.md` line.
F14 Delete `field_version_values`; delete `external_values` once `git grep
    "self.external_values"` finds only `_bind` and `_binding`; delete
    `_carried_port_values`, `_carried_port_groups`, `control_identity_receipts`,
    `declared_parameter_only_ids` once their readers are on the pages.

Writer sites routed: 22 book sites (census 30, 1.1) + 54 `external_values`
writers + 168 `fresh_value` calls + 3 alias writers + 3 `_carried_port_values`
writers + 2 `declared_parameter_only_ids` writers + 3 `region_signatures`
builders + `finish` + `_concord_lexical_reads` + `_set_operands` +
`fork_read_scope`.  Pages: 9 new + 4 declared-from-raw, 15 re-declared.

## B6. Risks

R1  **Refusing the snapshot as a name-carried arm changes emitted SSA.**
    Any program that lowers today because `external_values.get(arm_id,
    snapshot)` silently took the snapshot will either get the arm's real
    binding in its Phi or stop with `carried-name-arm-missing`.  The
    census-50 path (the dt controller step, audit `controller` case) is
    where it shows first; B8's probe must be green before the user
    launches that lowering.  This is the top risk of step 5.
R2  **168 `fresh_value` sites and variadic arity.**  Without N4 the Phi
    families cannot be declared; with it, a site that passes the wrong
    operand count is refused at the call.  The migration is mechanical but
    wide; do it in F4 as one pass with the registry's refusals as the
    checklist, not incrementally.
R3  **REVISE refusals from repeated rebinding.**  `lower_conditional`
    publishes the merged value under merged / initial / both arm ids;
    nested conditionals then re-publish the parent's initial id from the
    same Phi cell.  `_bind` must skip a post when the latest fact already
    names the same `ssa_value` cell (a no-op at the dict too), or `post`
    refuses.  Every family in B2.2 has been checked for a fresh source
    except the RESTORED writes, which derive from the snapshot cell that
    was posted earlier than the arm's binding; they pass because the
    source SET differs from the arm's post.
R4  **`SSAValue` mutation after posting.**  `produced_value` sets
    `value.dtype`; `_publish_loop_result_ports` sets `port.shape` and
    `port.device`; `external_value` builds a `refined` copy and swaps it
    in `arguments`.  The `ssa_value` row's fact holds dtype and shape, so
    each mutation must be a REVISE (F4 adds `_revise_value(value, cells)`
    at those sites) or the row and the object drift; the audit cannot see
    an object.
R5  **`identity_transition` row-shape split.**  Rows written by the
    reducer before F3 lands (raw, tagged) and rows after live on pages with
    different names; `render_identity_book` output changes; the
    `_planning_alias_transition_findings` reader changes; any external
    script reading `"identity_transition"` `"merge"` rows breaks.  `git
    grep` finds only the audit reader; state it in the commit.
R6  **Volume.**  One post per binding write, one per minted value, two
    edge rows each: the book grows by several times the instruction count
    per function.  Measure on `controller` before the Woodshop compile
    (plan 70 R5's caveat, larger here).
R7  **Pickling.**  `Function.metadata` is pickled by `_HostSSACachePickler`;
    B2.6 materializes plain tuples and ints, never a `Ref`.
    `ssa_value_objects` and the pages stay on the builder and the book.
R8  **Functions the reducer never saw** (`lexical_read_scope is None`:
    unit-level entry points, synthetic functions).  There is no
    `canonical_value` cell to derive from.  `_bind` posts under the
    `tensor_shape_concordance_scope` DERIVED(the `ssa_value` cell only)
    with kind `UNSCOPED`, and `fresh_value` still mints NOVEL; the audit
    lists these functions by scope so the residue is visible, and no
    reader of `lexical_read_binding` is reached (they already guard on the
    scope).

## B7. What step 5 needs from steps 2-3 (and 4)

1. `canonical_value` cells for every graph id the builder binds (steps 2
   and 4: the planner's ports, constants and projection leaves must have
   rows, A2.6 / A2.7 / R4 of step 4).  Without them `_bind` derives from
   the `ssa_value` cell alone and the chain to the source stops at the
   builder.
2. `name_binding` canonical rows (landed) for parameter seeds and
   `finish`'s `value_names` join; the planner's REVISE posts (A2.7) so the
   builder's `value_name_histories` view is the page's.
3. `reducer_field_state`, `ssa_field_version` (landed).
4. `return_site_slot` cells (landed) for `function_output`.
5. `loop_carried_binding`, `loop_result_port_binding` DERIVED (step 4 E13)
   and the ports' `canonical_value` NOVEL rows.
6. `deployment_region` and member cells (step 4 E8) for `region_signature`.
   If step 5 lands before step 4, `region_signature` derives from the
   feeds' and outputs' `canonical_value` cells and is re-derived (REVISE
   with the region cell as new source) when E8 lands.
7. N4 and N5 from step 1.

## B8. The seconds-long proof

Existing: `python -u tools/compiler_probes/probe_branch_written_field.py`
(lane C's chain, must stay green and now show the Phi's `ssa_value` NOVEL
row with the two arm cells as operands), `probe_annotated_scalar_parameter.py`
(no conditionals, no loops: the pages cost nothing when empty),
`probe_struct_intake.py`, `python -u tools/audit_identity_concordance.py
controller` (the dt controller step lowered alone -- census 50's path;
the design's probe for this step).

New, seconds-long: `tools/compiler_probes/probe_control_binding_chain.py`,
same harness as `probe_branch_written_field.py`.  Source: one function
with a scalar parameter, a name rebound on one arm of a conditional, and a
`for` loop carrying one name with a `break`:

```
def step(k: int, flag: bool) -> int:
    total = k
    if flag:
        total = total + 1
    for i in range(4):
        total = total + i
        if total > 10:
            break
    return total
```

Reading only the book, it checks:

1. `control_value_binding` row for the arm's `total` id has a revision
   stamped inside the arm (after the `carried_snapshot` cell), kind
   CONTROL_EXPRESSION, DERIVED from the arm expression's `canonical_value`
   cell and its operand bindings;
2. the conditional Phi's `ssa_value` row is NOVEL(`PHI_CONDITIONAL`) with
   operands (the arm binding cell, the `carried_snapshot` cell); no
   `NAME_ARM_VERSION_MISSING`; the emitted `conditional_carried` Phi's two
   args differ;
3. the loop's header Phi is NOVEL(`PHI_LOOP_HEADER`); `loop_carried_entry`
   DERIVED from `loop_carried_binding`; `carried_port_value` for the
   `LoopResult` port DERIVED from its `loop_result_port_binding` cell and
   the exit Phi's `ssa_value` cell; the exit Phi's operands include the
   break edge's binding cell;
4. `function_output` slot 0 DERIVED from the `return_site_slot` cell and
   the port's `carried_port_value` cell; `metadata["named_outputs"]` and
   `metadata["parameter_names"]` equal the page-derived tuples;
   `function_parameter` for `k` DERIVED from `declared_parameter` USED;
5. `unsourced-identity` is empty for the function: every MINTED id in it
   has a mint edge;
6. the probe prints `unsourced-fact` by stage before and after; the
   groups for stages `CONTROL_SSA*`, `REDUCER_READ`, `OPERAND_POSITION`
   are zero.

## B9. Unsourced worklist groups step 5 retires

Raw rows: `("lexical_read_binding", reduction / canonical_relabel /
raw_primitive, raw row)` (every reducer and relabel writer, the largest
raw-row group in the audit today), `("identity_transition", ...)` (every
`_set_operands` caller, `fork_read_scope`, `_region_feed`),
`("control_value_concordance", ...)`, `("loop_carried_entry", ...)`,
`("loop_entry_state", ...)`, `("while_carried_test", ...)`,
`("region_feed_consumer", ...)`, `("region_capture_binding", ...)`,
`("callsite_argument", ...)`, `("loop_result_reconciliation", ...)`,
`("control_uniform_dtype", ...)`, `("region_value_dtype", ...)`,
`("field_slot_storage_concordance", ...)`, `("sequence_contract_concordance",
...)`, `("tensor_shape_concordance", ...)` for the two builder sites,
`("loop_scope", ...)`, `("loop_scope_inner_transition", ...)`.
`unsourced-identity`: every `_ControlSSABuilder.fresh_value` mint (168
sites), `_split_region_captures_by_binding`'s formal, `_inject_field_slot_access.fresh`,
`lower_class_navigation_to_ssa.Builder`, `_lower_sequence_mutation_body`'s
projection, and the `__init__` joined flat sequence id.

Left for later steps: `tensor_ssa_lowering.fresh` (28 sites), `ir_identities`
mints, `ssa_call_input_adapters`, `ssa_record_return_state`'s `freshen_redefined_ssa_objects`
and return-edge `materialize` (steps 6-7); `sequence_table` / `SSASequenceTable`
during lowering (step 8).

## B10. Held for the user

1. B2.3: `carried-name-arm-missing` stops the lowering with a shortfall
   (as lane C does for fields) or raises at the site.  The plan writes the
   `Unresolved` row and the shortfall so either policy has its record --
   the same question plan 70 held for `ARM_VERSION_MISSING`, answered once
   for both.
2. B2.2 PROVISIONAL_ARGUMENT: record the minted-from-absence formal as
   `Unresolved(NO_PRODUCER_AT_USE)` (proposed; it is the design's rule) or
   as a resolved fact kind.  With the Unresolved, `finish`'s
   `existing_value` treats it as absence and a genuinely producerless
   output is reported by the `return` shortfall instead of becoming an ABI
   input silently -- a behaviour change on programs that today compile by
   accident.

---------------------------------------------------------------------------

# Order between the two steps

Step 4 first, as the design orders them: step 5's `region_signature`,
`carried_port_value` and every binding of a planner-minted node derive
from step 4's cells (B7 items 1, 5, 6).  If the user wants step 5 first,
B7 names the fallbacks (derive from `canonical_value` and re-derive when
step 4 lands); nothing in step 5 is blocked, only shorter-rooted.  B1.3
(`LEXICAL_READ_BINDING`, `IDENTITY_TRANSITION`) is the one piece both steps
want first; it can land alone as the first commit of either step with
`probe_annotated_scalar_parameter` as its proof.
