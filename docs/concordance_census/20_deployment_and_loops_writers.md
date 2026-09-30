# Census 20: concordance writers in deployment planning and loop composition

Read-only scout, 2026-09-30. Scope: `src/compiler/glsl_deployment_strategy.py`,
`loop_composer.py`, `transformation_priority.py`, `control_source.py`,
`hierarchical_plan.py`, `shell_reference_tables.py`,
`process_graph_function_linking.py`.

This file complements `CONCORDANCE_MASTER_LIST.md` (referred to below as
MASTER). MASTER Part A gives page-level status; MASTER Part B (static scan of
2026-09-25) lists no-book and mixed functions. This census adds only what
MASTER lacks: a per-write-site EDGE classification, the fallback-as-fact
question, and the shadow ledgers that decide program structure. Where a
MASTER row is contradicted by the current code, section 5 says so.

Vocabulary used in the tables:

- EDGE?  YES = the row or fact names the exact source row(s) AND the stage it
  derives from.  PARTIAL = names a source identity (a value id, a caller
  name, a binding) but not its page/row/stage.  NO = a bare fact.
- fallback-as-fact = a value chosen from absence of information is written
  as if it were a fact (rather than as unresolved).
- write mode = `set(row, 0)` (silent overwrite at column zero; `IdentityPage.set`
  never refuses), `set(row, next)` (append at `len(history)`), `concord`
  (first statement owns the row, disagreement raises), `revise` (append),
  or a named helper (`record_proven_shape`, `record_shape_transformation`,
  `invalidate_proven_shape`, `commit_sequence_row_layout`).
- Function names are the reference; no line numbers, no compiler ids.

---

## 1. Write sites, grouped by page

### 1.1 The shape-transformation model (the edge-carrying pages)

Pages `shape_transformation_concordance` (edge), `shape_transformation_dependents`
(reverse index), `shape_transformation_state` (current projection), written
only through `record_shape_transformation`.

| Enclosing function | Row key (words) | Fact | Mode | EDGE? | Fallback-as-fact? | Readers |
|---|---|---|---|---|---|---|
| `_tensor_descriptor` (post-query block), one call per semantic parent | edge: (target scope, target value, source scope, source value, stage `graph_tensor_descriptor`, operation, role) | (source state, target state) | helper (edge appended at next column; state revised) | YES | YES: any non-None answer from `_tensor_descriptor_rule` is written, including an empty-extent / unknown-dtype answer (`descriptor_states_a_shape` documents that this answer means "recovery stopped"). The state page records it as `("resolved", ...)`, and `descriptor_from_shape_transformation_state` reconstitutes it unfiltered, so the next query returns it as the stage's decision. When no semantic parent exists a synthetic root `("descriptor_root", node)` is used as source: acceptable as a spontaneous assertion, but it is also used for nodes whose only parents carry filtered roles (`callee`, `func`, `function`, `definition`, `control`). | `_tensor_descriptor` (preamble), `withdraw_superseded_shape_derivations`, `concordant_shape_transformation_state` |
| `_propagate_callsite_tensor_specializations` (exact single result, observation branch) | edge from `("return", callee)` to caller call value, stage `callsite_return_observation` | (state, state) with source state == target state | helper | YES (edge) but the source identity `("return", callee)` has no state row of its own; the edge asserts the callee return has the caller's state, so withdrawal along it can only fire if someone else revises that synthetic identity | No | as above |
| `_propagate_callsite_tensor_specializations` (single Mapping result replaced, elif branch) | same edge shape, stage `callsite_return_specialization` | (replacement, replacement) | helper | YES, same caveat | No | as above |
| `_propagate_callsite_tensor_specializations.call_result_descriptor` (yield rows) | edge from `("yield", callee, column)` to `("yield_row", call value, column)`, stage `callsite_yield_observation` | (state, state) | helper | YES, same caveat (both endpoints synthetic, target is not a graph value) | No; columns not stated by every yield site are skipped | `_tensor_descriptor` via state lookups |

Companion invalidation:

| Enclosing function | Row key | Fact | Mode | EDGE? | Fallback-as-fact? | Readers |
|---|---|---|---|---|---|---|
| `_invalidate_tensor_descriptor_dependents` (called from `_publish_callsite_return_members`, `_propagate_callsite_tensor_specializations`, `_apply_callsite_tensor_descriptors`, and `loop_composer.rewire_continuation`) | `proven_shape` (function, value) | ("invalidated", first cause id, reason string) | helper `invalidate_proven_shape` (append) | PARTIAL: names one cause id and a free-text reason; not the edge row it withdraws | No, but the DEPENDENCY CLOSURE is computed from graph `parents` (a shadow), not from `shape_transformation_dependents`, and its role filter (`callee`, `func`, `definition`) differs from `_tensor_descriptor`'s edge filter (adds `function`, `control`): a value reached through a `control` parent is invalidated although no edge was ever recorded for it, and the node `tensor` slot is popped as a side effect | `_tensor_descriptor`, `proven_shape_contract_of` |

### 1.2 `proven_shape` (bare facts written beside, but not pointing at, the edges)

| Enclosing function | Row key | Fact | Mode | EDGE? | Fallback-as-fact? | Readers |
|---|---|---|---|---|---|---|
| `_tensor_descriptor` (post-query, extents non-empty and not dynamic) | (function, value) at dependency level | ("proven", extents, dtype) | `record_proven_shape` | NO on the row itself; an edge for the same target exists on `shape_transformation_concordance` from the same operation, but nothing in this row names it | No (empty extents refused by design) | `_tensor_descriptor`, `proven_shape_contract_of`, `site_bundle.py`, probes |
| `_propagate_callsite_tensor_specializations` (observation branch) | (caller, call value) at dependency level | same | helper | NO (edge written alongside, unreferenced) | No | same |
| `_propagate_callsite_tensor_specializations` (replacement branch) | same | same | helper | NO (same) | No | same |
| `_fold_callsite_structural_values` (Input formal branch, when `planner_tensor_descriptors[binding]` states a shape) | (function, Input value) | ("proven", extents, dtype) | helper | NO: the source is the shadow ledger `planner_tensor_descriptors` (section 2), no edge to the caller argument that supplied it | No | same |
| `_callsite_specialized_shell_type.publish_call_result_shape` | (caller, call value) | ("proven", extents, dtype) | helper | NO: unlike the propagate path, NO `record_shape_transformation` accompanies it; the callee return identity is not linked to the caller call value here | No | same |

### 1.3 Callsite specialization ledgers

| Page | Enclosing function | Row key | Fact | Mode | EDGE? | Fallback-as-fact? | Readers |
|---|---|---|---|---|---|---|---|
| `formal_literal` | `_publish_formal_literal` (called from `_callsite_specialized_shell_type`) | (function, parameter) | ("proven", value, caller) then ("conflicting", (previous, value), caller) | `set(row, 0)` first; `set(row, next)` on conflict | PARTIAL: names the caller function, not the callsite or the argument value row; the literal came from `_source_static_literal` reading `planner_specializations` or a Constant node | No; "conflicting" is honest. All exceptions swallowed | `_proven_formal_literal` (feeds `_source_static_value` / `_source_static_literal`), probes |
| `formal_shape` | `_publish_formal_shape` (called from `_propagate_callsite_tensor_specializations`, `propagate_bound_planner_specializations`, `_callsite_specialized_shell_type`) | (authored function, parameter) | ("proven", extents, dtype, caller) / ("conflicting", extents, dtype, caller) | as above | PARTIAL (caller name only) | No; but after "conflicting" nothing further is recorded, so a third caller's shape is lost | `_proven_formal_shape`, `_tensor_descriptor` (formal_conflict, polymorphic_specialization gates), tests, probes |
| `callsite_projection_specialization` | `_publish_callsite_return_members.record` | (caller, callsite, index, leaf) | (action in {enriched, reused, materialized}, descriptor receipt) | `set(row, next)` when changed | PARTIAL: names the leaf id; the descriptor's origin (`call_result_descriptor`'s round) is not named | No | tests only |
| `callsite_projection_identity_concordance` | `_publish_callsite_return_members` (stale leaves) | ((caller, callsite, `id(caller.G)`), index, stale id) | replacement id | `set(row, 0)` after `latest is None`; raises on disagreement | PARTIAL: names both ids; the scope embeds a Python object id (`id(caller.G)`) which is process-local, not a book-minted scope (MASTER working rule) | No | `_retarget_all_cached_value_ids` (consumed) |
| `callsite_projection_identity_concordance` | `_repair_missing_aggregate_leaf_projections` | same | same | same | PARTIAL, same | No | same |
| `callsite_return_specialization` | `_propagate_callsite_tensor_specializations.call_result_descriptor` | (caller, callee, callsite) | per-output (shape, dtype) or None | `set(row, next)` every round | NO (round record) | No | `IdentityPage.oscillating_rows` in the same function, probes |
| `callsite_tensor_result_specialization` | `_propagate_callsite_tensor_specializations` (replacement branch) | (caller, callsite) | (previous receipt, replacement receipt) | `set(row, next)` | PARTIAL: names before/after states of the node `tensor` slot, not the callee row | No | none found outside the writer |
| `tensor_shape_settlement_concordance` | `_propagate_callsite_tensor_specializations` (settlement pass) | (function, semantic value) | (extents, dtype) | `concord`; raises on disagreement | PARTIAL: restates `proven_shape_contract_of` at the same key; page and stage unnamed | No | none outside the writer (it is the fixed point's "seen" set on the book) |
| `source_callsite_activation_concordance` | `ProcessGraphGLSLDeployment.__init__.plan_callsites` | (caller identity, callsite) | (callee reference, mode in {callsite_shell, recursive_scc_backedge}, recursive unit) | `set(row, 0)` after `latest is None`; raises on disagreement | NO: decided from node attribute `callee_ref` and `recursive_unit_backedge` over `recursion_table` (shadows); neither is named | No | `identity_concordance` audit, tests |
| `source_control_specialization_concordance` | `_fold_callsite_structural_values` (IfExp `_StructuralValueAlias` branch) | (function, retained control id) | dict {graph control, source control, predicate value id, selected role, proof "structural_constant_predicate"} | `set(row, next)` | PARTIAL: names the predicate value id, not the `proven_literal` / `known` row that decided it | POSSIBLE: `predicate = known.get(test, unresolved)`; the `unresolved` sentinel object is truthy, so a test missing from `known` records `selected_role = "body"` | `fortran_c_shell.py` (via the shadow `structurally_specialized_conditional_node_ids`, not this page) |
| `source_control_specialization_concordance` | `_fold_callsite_structural_values` (dead-arm pruning, with `rejected_role`) | same | same plus rejected role | `set(row, next)` | PARTIAL, same | No (predicate id is concrete here) | same |
| `structural_specialization_fixed_point` | `_fold_callsite_structural_values` | (function) | (digest, changed, net tensor mutations) | `set(row, next)` | NO (round record) | No | none outside writer |
| `proven_literal` | `_fold_callsite_structural_values.replace` | (function, value) | the literal | `set(row, next)` when different | NO: the literal is the fold's `known[node]`; which operands / which `planner_specializations` entry produced it is not named | No | `fortran_c_shell.py`, and other graph copies of the same function (the docstring's stated purpose) |

### 1.4 Operand-position and hierarchy structure pages (planner)

| Page | Enclosing function | Row key | Fact | Mode | EDGE? | Fallback-as-fact? | Readers |
|---|---|---|---|---|---|---|---|
| `operand_position_orphan` | `_concord_consumer_operands` | (read scope, node, role, ordinal) | the `lexical_read_binding` fact at the same key | `concord` | PARTIAL: same key as the source row by convention; page and stage not named | No (this IS the unresolved record MASTER A3a describes) | `CorrelationTable` |
| `consumer_operand` | `_concord_consumer_operands` | (read scope, node, value) | ((role, ordinal), ...) | `concord` | PARTIAL: row plus fact reconstruct `lexical_read_binding` keys, but the positions are read from graph `parents` (shadow) | No | `topological_reducer._set_operands` (moves them), `precompile_to_ssa` |
| `item_operand` | `_concord_item_operands` | (read scope, item) | (operand, role, ordinal) | `concord` | PARTIAL (same) | No; skipped unless exactly one operand | `precompile_to_ssa` |
| `call_argument_operand` | `_concord_call_argument_operands` | (read scope, callsite, position) | (role, ordinal) | `concord` | PARTIAL (same); MASTER A3a notes these are not yet moved by `_set_operands` | No | `precompile_to_ssa` |
| `call_result_projection_concordance` | `_build_shell_hierarchy_plan` | (call node, caller projection) | authored result path | `concord` | NO: derived from a successor walk over `Indexed` nodes (shadow); the projection's own creation (`_publish_callsite_return_members`) is not named | No | `fortran_c_shell.py`, tests |
| `source_precision_region_concordance` | `_precision_indivisible_node_groups` | (numeric scope, terminal node) | (ordered members, promotes, collapses, widths) | `set(row, 0)` after `latest is None`; raises on disagreement | PARTIAL: members come from `source_value_class_concordance` and `source_precision_boundary_concordance` rows (book sources) but the fact names ids, not those rows; a mirror is also written to node attribute `source_precision_region` | No | `identity_concordance`, `tensor_ssa_lowering` |
| `aggregate_ledger` | `_record_aggregate_ledger_lookup` (from `_authored_aggregate_leaves`) | (owner, value, "ledger_lookup") | (state in {emptied, absent, resident, dangling}, counts, node type, attribute key set) | `set(row, 0)` EVERY call: silent overwrite, history lost; inside try/except | NO (diagnostic) | "absent" is recorded as a status, not a fact: acceptable; but each later lookup overwrites the earlier status | none |

### 1.5 `sequence_row_layout_concordance`

| Enclosing function | Row key | Fact | Mode | EDGE? | Fallback-as-fact? | Readers |
|---|---|---|---|---|---|---|
| `_tensor_descriptor` (answer carries `sequence_row_shape`) | (function, value) | row shapes, dtypes, source label "tensor descriptor transformation path" | `commit_sequence_row_layout` | PARTIAL: source is a string label | No | sequence lowering (outside slice) |

### 1.6 `loop_composer.py`

| Page | Enclosing function | Row key | Fact | Mode | EDGE? | Fallback-as-fact? | Readers |
|---|---|---|---|---|---|---|---|
| `loop_result_port_binding` | `materialize_retained_loop_ports.add_port` | (read scope, port) | binding name | `concord` | PARTIAL: the port id is minted here (a spontaneous identity) and the binding names it, but the carried entry / break value it continues is not named | No | `precompile_to_ssa` |
| `loop_continuation_rewire_concordance` | `materialize_retained_loop_ports.rewire_continuation` | (FUNCTION name scope, consumer, role, ordinal) | (old value, new value, binding) | `revise` | PARTIAL: names both ids and the binding; the key uses `function_name`, not the read scope that keys `lexical_read_binding`, so the two cannot be joined by key; the operand rewrite itself bypasses `_set_operands` / `identity_transition` (MASTER A3a "open") | No | none outside the writer (RECORD-ONLY) |
| `proven_shape` (invalidation) | `rewire_continuation` via `_invalidate_tensor_descriptor_dependents` | see 1.1 | ("invalidated", new value id, "loop-continuation-rewired") | helper | PARTIAL | No | see 1.1 |
| `loop_region_membership` | `analyze_shader_loop_reductions` | (read scope, loop node) | sorted region indices | `revise` | NO: region indices are planner ordinals (ephemeral); the regions' own identity rows are not named | No | `control_source.place_loop_carried_region_producers.owned_regions` |
| `loop_carried_binding` | `analyze_shader_loop_reductions` | (read scope, loop, updated, initial) | binding names | `concord` | NO: copied from the private plan `loop.carried_bindings`; the reducer's `carried_update` edge (MASTER A3d) is not named | No | `precompile_to_ssa` |

### 1.7 `transformation_priority.py` (`TransformationLedger`)

Scope is minted through `IdentityBook.mint_scope` (correct per MASTER).

| Page | Enclosing function | Row key | Fact | Mode | EDGE? | Fallback-as-fact? | Readers |
|---|---|---|---|---|---|---|---|
| `transformation_decision` | `propose` | (scope, identity) | (rule, proof, target) | `revise` | PARTIAL: `proof` is an opaque caller-supplied hashable, never a book row | No | `incumbent_target`, `_decision` |
| `transformation_event` | `_record_event` | (scope, serial) | event dict | `concord` | PARTIAL (same) | No | `_BookEventLog` |
| `transformation_rejection` | `propose` | (scope, identity, rule, proof, retained rule, retained proof) | event serial | `concord` | PARTIAL (same) | No | `propose` |

### 1.8 `hierarchical_plan.py`

| Page | Enclosing function | Row key | Fact | Mode | EDGE? | Fallback-as-fact? | Readers |
|---|---|---|---|---|---|---|---|
| `operator_result_type_concordance` | `plan_region_to_ssa_instrs` | (function scope, closure id, region name, output id) | (opcode, "bool") | `set(row, 0)` after `latest is None`; raises on disagreement | NO: a law-of-the-opcode claim (acceptable as spontaneous) but keyed by `closure_id`, a dense number from `assign_hierarchy_ids` that is valid for one plan | No | `identity_concordance`, tests |
| (none) | `plan_region_to_ssa_instrs.fresh_like` | -- | `GLOBAL_MONOTONIC_IDS.mint()` for a fresh SSA value | mint | NO: a minted identity with no edge to the result it is "like" | -- | SSA consumers |
| (none) | `assign_hierarchy_ids` | -- | (closure, local) -> global id correlation, `HierarchyValueTable.correlations`, then `shell.hierarchical_effective_value_table` (redirected copy) | private tuple | NO: NOT ON THE BOOK. Union-find over plan `argument_bindings` / `result_bindings`; the global id that every later stage uses as "the one semantic identity" (its own docstring) has no book row | No | every hierarchical composition stage in `_build_hierarchical_glsl_artifact` |

### 1.9 Files with no book writes

- `control_source.py`: one reader (`loop_region_membership` in
  `place_loop_carried_region_producers.owned_regions`). Returns None when no
  row exists and the caller falls back to marker regions: the fallback is
  not recorded anywhere.
- `shell_reference_tables.py`: readers only (`identity_table`, `map_ir`,
  `recursion_table`, node attributes `callee_ref` / `external_callee_ref` /
  `static_python_reference`). Produces `ShellReferenceTables.correlations`
  (shadow, section 2).
- `process_graph_function_linking.py` (`link_process_graph_functions`):
  writes `callee.G.graph` (`function_ref`, `function_name`), node
  `attributes` (`callee_ref`, `callee_resolution = "linked-process-graph"`),
  and prunes `unresolved_ast_calls`. A call-binding decision recorded only on
  the node. No book counterpart; `source_callsite_activation_concordance`
  later takes `callee_ref` as given.

---

## 2. Shadow ledgers (identity / provenance facts outside the book)

| Name | Fact held | Writer(s) in slice | Readers | Book counterpart? | Decides STRUCTURE? |
|---|---|---|---|---|---|
| `G.graph["planner_specializations"]` | formal name -> folded literal | `_propagate_callsite_planner_specializations` (`setdefault(...)[param] = value` on the SHARED callee graph: accumulates across callsites), `call_result_descriptor` and `_callsite_specialized_shell_type` (replace, on the copy), `propagate_bound_planner_specializations` (`setdefault().update`) | `_source_static_value`, `_source_static_literal`, `_fold_callsite_structural_values`, `_expand_specialized_unbroadcast_identity`, `_resolve_grounded_method_references`, `_resolve_grounded_tensor_operations`, `_resolve_bound_function_references`, capture/emit paths | `formal_literal` (PARTIAL, written only from `_callsite_specialized_shell_type`; the propagate writer records nothing) | YES: literal folding removes branches and dead calls, which decides which regions and lanes exist. The dt system's `rollback` was folded this way (comment in `_source_static_value`). |
| `G.graph["planner_tensor_descriptors"]` (+ `callsite_tensor_descriptor_names`) | formal name -> tensor descriptor | `_propagate_callsite_tensor_specializations` (`setdefault` + update on shared callee), `_apply_callsite_tensor_descriptors` (replace), the two copy writers | `_tensor_descriptor` (`specialized_operator`, `localized_formal` gates decide whether the BOOK answer is even consulted), `_fold_callsite_structural_values`, `specialization_state` digest | `formal_shape` (PARTIAL) | YES: gates which shape facts are trusted; `_expand_specialized_unbroadcast_identity` chooses a body from it |
| `G.graph["planner_parameter_classes"]`, `G.graph["parameter_record_abi"]` | formal -> (class, source) / record ABI | `_callsite_specialized_shell_type` | fold, linker | derived FROM `source_value_class_concordance` rows, written back as shadow | YES (record vs scalar lanes) |
| `G.graph["identity_table"]` | binding name -> value id history | `_fold_callsite_structural_values.remove_node`, `.replace_alias`, `_alias_projection_to_member`, `_apply_callsite_aggregate_descriptors` (rewrites) | MASTER B1 lists ~30 readers in this file alone | `lexical_read_binding` covers reads; there is no page for definitions | YES (root/return selection) |
| `G.graph["return_slot_values"]`, `return_value_nodes`, `return_record_field_states` | return identity per site | `replace_alias`, `_alias_projection_to_member`, `_follow_declared_value_source` | fold, `call_result_descriptor`, linker | none in slice | YES |
| `G.graph["structurally_specialized_conditional_node_ids"]` | which conditionals were folded | `_fold_callsite_structural_values` (two sites) | `fortran_c_shell.py` | `source_control_specialization_concordance` is written alongside, but the reader consumes the shadow | YES |
| `G.graph["_dispatch_metadata_cache"]` | node -> "is dispatch metadata" (excluded from executable set) | `_dispatch_metadata_node_classifier` | `strategize_shell_deployment` (executable node set) and the uncovered-node check in whole-program compile | none; the decision is recorded nowhere (confirms the 2026-09-30 scalar-deferral finding). Fingerprint = schema + node count + edge count only: an attribute change (e.g. `authored_call_result_projection` set later by `_publish_callsite_return_members`) does not invalidate it | YES: decides which nodes get regions |
| `G.graph["_dependency_order_cache"]`, `["_inert_routing_nodes_cache"]` | ordering / routing classification | `_dependency_order`, inert-routing helper | planner | none | indirectly (level numbers feed `proven_shape` columns) |
| subgraph `G.graph["deployment_nodes"|"deployment_inputs"|"deployment_outputs"|"deployment_store_nodes"|"compartment_schedule*"]` | region membership and IO | region carving | uncovered check, emitters | `loop_region_membership` names region ordinals only | YES (this IS the region table) |
| node `attributes["aggregate_leaf_value_ids"]`, `tensor_output_descriptors`, `authored_return_container`, `producer_kind`, `aggregate_kind`, `aggregate_leaf_republication`, `authored_call_result_projection` | which member leaves a call publishes | `_publish_callsite_return_members`, `_repair_missing_aggregate_leaf_projections` | `_authored_aggregate_leaves`, `_dispatch_metadata_node_classifier`, linker | `callsite_projection_specialization` (PARTIAL), `aggregate_ledger` (diagnostic, overwritten) | YES (leaves become formals / region IO) |
| node `attributes["structural_specialization"]` | node is a folded arm, not source data | fold | `_source_static_value` | `source_control_specialization_concordance` (PARTIAL) | YES |
| node `["tensor"]` slot | cached descriptor | many; popped by `_invalidate_tensor_descriptor_dependents` | `_tensor_descriptor_rule`, `specialization_state` digest | `proven_shape` / state page | indirectly |
| node `attributes["source_precision_region"]` | mirror of a book row | `_precision_indivisible_node_groups` | region reducer | exact mirror of `source_precision_region_concordance` | YES (region grouping) |
| node `attributes["callee_ref"|"callee_resolution"|"constructor_ref"]`, `G.graph["function_ref"]`, `unresolved_ast_calls` | call binding | `link_process_graph_functions`, `plan_callsites` (constructor_ref) | `plan_callsites`, `shell_reference_tables` | none | YES (which shells exist) |
| `G.graph["recursion_table"]` | recursive SCC units | outside slice | `recursive_unit_backedge`, `_dependency_order` | none | YES (backedge vs shell) |
| `HierarchyValueTable.correlations`, `shell.hierarchical_effective_value_table`, `shell.hierarchical_region_correlations`, `shell.hierarchical_value_aliases`, `shell.hierarchical_program_value_origins`, `shell.hierarchy_identity_collapses` | (closure, local) -> global id; region and alias correlations | `assign_hierarchy_ids`, `_build_hierarchical_glsl_artifact` | all hierarchical composition; other shells (`target.hierarchical_effective_value_table.correlations`) | none | YES: the global identity of every value in the composed program |
| `PlanClosure.argument_bindings` / `result_bindings`, `PlanCall` | caller <-> callee value pairing | `_build_shell_hierarchy_plan` | `assign_hierarchy_ids`, `_refresh_hierarchy_control_captures` | `call_argument_operand`, `call_result_projection_concordance` (PARTIAL/NO) | YES |
| `ShellReferenceTables.correlations`, `shell.reference_correlations`, `self.function_shell_types` | class/method identity -> function reference; reference -> shell type | `shell_reference_tables`, `plan_callsites` | deployment | `source_callsite_activation_concordance` (PARTIAL coverage) | YES |
| `shell.compiled_process_graph_aliases` | promote-boundary value -> authored producer | `ProcessGraphGLSLDeployment.__init__` | capture wiring | derived from `source_precision_boundary_concordance` (a copy of book rows, then consumed privately) | no (aliasing only) |
| `candidates` dicts in `_propagate_callsite_planner_specializations` and `_propagate_callsite_tensor_specializations`; `callsite_planning_visits`; `endpoint_identity` / `attribute_ids` in `compile_discovery_program`; `seen_round_states`, `seen_fixed_point_states` | per-pass agreement sets, endpoint aliasing | those functions | same functions | `callsite_return_specialization` / `structural_specialization_fixed_point` record digests only | `endpoint_identity`: YES (aliases planned region ids) |
| `loop.carried_bindings`, `pending_plans` (loop plan records) | carried (name, initial, updated) triples | `loop_composer` plan building; retargeted in `rewire_continuation` | `add_port`, `analyze_shader_loop_reductions` | `loop_carried_binding` (NO edge) | YES (which Phis exist) |

---

## 3. Verdict

Counts (book write sites in the slice): 42.
- `glsl_deployment_strategy.py`: 33 sites over 21 pages (plus the two
  companion shape pages written through the helper).
- `loop_composer.py`: 5 sites, 4 pages (+ `proven_shape` invalidation).
- `transformation_priority.py`: 3 sites, 3 pages, book-minted scope.
- `hierarchical_plan.py`: 1 site, 1 page; plus 1 unrecorded id mint.
- `control_source.py`, `shell_reference_tables.py`,
  `process_graph_function_linking.py`: 0 book writes.

EDGE classification: YES 4 (all four are `record_shape_transformation`
calls; three of them use a synthetic source identity that has no state row,
so withdrawal along them is inert), PARTIAL 24, NO 14.

Fallback-as-fact, confirmed: `shape_transformation_state` cements an
empty-extent / unknown-dtype descriptor answer as `resolved` and serves it
back on the next query (`_tensor_descriptor` preamble). Possible:
`source_control_specialization_concordance` records `selected_role = "body"`
from a truthy `unresolved` sentinel when the predicate is absent from
`known`. Structural: `_invalidate_tensor_descriptor_dependents` withdraws
along graph `parents` with a role filter different from the one that
recorded the edges, so the book's dependents page and the actual
invalidation closure are two different graphs.

Ten most load-bearing NO sites (book-NO, or no book at all):

1. `_propagate_callsite_planner_specializations` -> `planner_specializations`
   (no book write at all; decides folds; the `rollback` case).
2. `assign_hierarchy_ids` -> `HierarchyValueTable.correlations` (the global
   id of every value; not on the book).
3. `plan_callsites` -> `source_callsite_activation_concordance` (shell vs
   recursive backedge; bare fact from `callee_ref` + `recursion_table`).
4. `_fold_callsite_structural_values.replace` -> `proven_literal` (literal
   crosses graph copies with no edge to its operands).
5. `_fold_callsite_structural_values` Input branch -> `proven_shape` from
   `planner_tensor_descriptors` (shadow -> book fact, no edge).
6. `_callsite_specialized_shell_type.publish_call_result_shape` ->
   `proven_shape` (no callee-return edge, unlike the propagate path).
7. `_dispatch_metadata_node_classifier` -> `_dispatch_metadata_cache` (which
   nodes are executable; recorded nowhere).
8. `_build_shell_hierarchy_plan` -> `call_result_projection_concordance`
   (call result path -> projection, read by the linker; no edge to the
   projection's publication).
9. `analyze_shader_loop_reductions` -> `loop_carried_binding` and
   `loop_region_membership` (copied from private plan records; no edge to
   the reducer's `carried_update` claim or the regions' identities).
10. `plan_region_to_ssa_instrs.fresh_like` -> `GLOBAL_MONOTONIC_IDS.mint()`
    (a minted identity with no transform edge), and the same function's
    `operator_result_type_concordance` keyed by an ephemeral closure id.

Shadow ledgers that decide program STRUCTURE and therefore must be on the
book first, in dependency order:

1. `planner_specializations` (and its single writer that records nothing,
   `_propagate_callsite_planner_specializations`): every downstream fold,
   dead-call prune and conditional specialization descends from it.
2. `planner_tensor_descriptors`: gates whether `_tensor_descriptor` trusts
   the book at all.
3. `identity_table` / `return_slot_values` / `return_value_nodes`: binding
   definitions and return identity, rewritten by the fold.
4. Node aggregate ledgers (`aggregate_leaf_value_ids`,
   `tensor_output_descriptors`, `authored_call_result_projection`): which
   leaves exist, and they feed 5.
5. `_dispatch_metadata_cache` + subgraph `deployment_nodes`: the executable
   set and the region table.
6. Call binding (`callee_ref` from `link_process_graph_functions`,
   `recursion_table`, `function_shell_types`): which shells exist.
7. `HierarchyValueTable.correlations` (+ plan `argument_bindings` /
   `result_bindings`): the global identity used by everything after
   composition.
8. `loop.carried_bindings` / `pending_plans`: which Phis and ports exist.

---

## 4. What the book-NO writers would need to say (for the one-api design)

- `record_proven_shape` calls: name the `shape_transformation_concordance`
  edge row written in the same operation (the row exists in three of five
  sites; two sites have no edge to point at and must first record one).
- `formal_literal` / `formal_shape`: name the caller's `call_argument_operand`
  row and the argument value's `proven_literal` / `proven_shape` row.
- `source_callsite_activation_concordance`: name the `callee_ref` binding
  as a row (needs `link_process_graph_functions` to record the resolution)
  and the recursion unit row (needs `recursion_table` on the book).
- `loop_carried_binding`: name the reducer's `carried_update` operand row
  (`lexical_read_binding` at the loop node's operand position).
- `loop_region_membership`: needs a region identity row to point at.
- `operator_result_type_concordance`: key by the book-minted plan scope
  rather than the dense `closure_id`.
- `fresh_like`: record (minted id, "like", result id) as a transform edge.
- `assign_hierarchy_ids`: each `(closure, local) -> global` correlation as a
  `concord` row whose fact names the `argument_bindings` / `result_bindings`
  union it was joined through.

---

## 5. Where this reading differs from MASTER

- MASTER A1 marks `TransformationLedger` BOOK. Agreed on Source / Record /
  Consumed; but `proof` is opaque, so no row on these pages can be followed
  to a source row. BOOK by MASTER's definition, PARTIAL by this census's.
- MASTER A3 credits `loop_carried_binding` and `loop_result_port_binding` to
  `loop_composer` without status. Their SOURCE is the private plan
  (`loop.carried_bindings`, `attributes["binding_name"]`), so by MASTER's
  own three-part definition they are RECORD + Consumed with a private Source,
  not BOOK.
- MASTER B1 lists `_build_shell_hierarchy_plan` as no-book. It now writes
  `call_result_projection_concordance` and calls
  `_concord_call_argument_operands`: it is mixed.
- MASTER B1 lists `_source_static_value` and `_source_static_literal` as
  no-book. Both now read the book through `_proven_formal_literal`: mixed
  (readers). They still decide from `planner_specializations` first.
- MASTER B1 lists `_apply_callsite_tensor_descriptors` and
  `_callsite_specialized_shell_type.publish_call_result_shape` as no-book.
  Both write the book (`invalidate_proven_shape`, `record_proven_shape`).
- MASTER B does not cover `loop_composer.py`, `hierarchical_plan.py`,
  `control_source.py`, `shell_reference_tables.py`,
  `process_graph_function_linking.py` (not among its 12 scanned modules).
  Additions from this census: `assign_hierarchy_ids` / `HierarchyValueTable`
  (no-book, structural), `plan_region_to_ssa_instrs.fresh_like` (unrecorded
  mint), `link_process_graph_functions` (no-book call binding),
  `_dispatch_metadata_node_classifier` (no-book, structural),
  `_propagate_callsite_planner_specializations` (no-book, structural).
- MASTER B2 lists `_tensor_descriptor` as mixed. Agreed; this census adds
  that its state page cements fallback answers (section 1.1) and that its
  invalidation closure is computed from the graph, not the dependents page.
- MASTER working rule "scopes are minted by the compile's book": the two
  `callsite_projection_identity_concordance` writers embed `id(caller.G)` in
  the row scope, a process-local Python object id.
