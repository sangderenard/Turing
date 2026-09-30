# Concordance census 75: raw-only pages inventory

Read-only inventory taken 2026-09-30 on `main` by `git grep`, `sed`, an AST
walk over `src` (every `.page(<name>)` mention, with the enclosing function
and the methods called on the page object) and ONE run of a copy of the
lead's `measure_completeness.py` extended to print, per undeclared page, its
cell count and one sample row/fact shape with ids replaced by `<id>`.
No compiler ids or line numbers appear here; the sample shapes are shapes,
not values.

Relation to the other census files: 10/20/30/40 already carry per-write-site
tables (row key, fact, mode, readers) for most of these pages; this file is
the page-level roll-up those tables lacked, cross-checked against the live
book. Where this file and a census table disagree on a row shape, the live
book (the measurement) wins and the row here says so.

Declared vocabulary (34 pages): 4 private book pages + 3 shape-transformation
pages in `src/compiler/identity_concordance.py`, 27 pages in
`src/compiler/concordance_declarations.py` (steps 2-3). Everything else that
`IdentityBook.page(...)` is asked for by name is a raw-only page: created on
first mention, written through `set` / `concord` / `revise` /
`mapping(...)[k] =` / `bind_alias`, auto-tagged `Unsourced(RAW_PRIMITIVE)`.

## 0. Counts

| what | count |
|---|---|
| raw-only page names mentioned anywhere in `src` (static) | 148 |
| ... of which touched (created) in at least one of the three audit cases | 109 |
| ... of which hold at least one cell in the `controller` case | 52 |
| ... mentioned in `controller` but never written there (0 cells) | 49 |
| ... never touched by mapping/energy/controller at all | 39 |
| declared pages written in `controller` | 24 |

Measurement (controller case): 17834 cells on non-private pages, 9755 tagged
unsourced (54.7%); 125 pages written, 24 declared, 101 undeclared.
Energy: 9642 cells, 58.9% unsourced, 123/20/103. Mapping: 425 cells, 68.2%
unsourced, 100/20/80.

Ownership by design section 4 (`docs/CONCORDANCE_SINGLE_API_DESIGN_2026-09-30.md`):

| owner | pages | controller cells (sum) |
|---|---|---|
| step 4 planner structure | 28 | 1562 |
| step 5 control SSA builder | 18 (+ the pts branch of `identity_transition`, counted under steps 1-3) | 2569 |
| step 6 record materialization and return versions | 17 | 74 |
| step 7 frame linker | 27 | 259 |
| step 8 book-backed tables | 12 | 47 |
| section 4 names no owner (nearest step proposed per row) | 21 | 201 |
| steps 1-3 remainder (book internal, reducer source pages, source-stage declarations) | 25 | 2913 |
| total | 148 | 7625 (= every cell on an undeclared page in the controller case) |

The per-step tables are ordered largest cell count first (controller; where
controller is 0 the energy/mapping counts are shown as `0/e/m`).

Column legend: "row" is the observed row shape in words with a proposed
`RowFieldKind` per element (S = SCOPE, V = VALUE_ID, N = NAME, I = INDEX,
L = LABEL, P = PAGE_REF); "fact" is the observed fact shape; "mode" is the
raw primitive used; "cells" is controller/energy/mapping; "read by others" is
whether any function other than the writer reads the page (grep), naming it.

File abbreviations: fcs = `src/compiler/fortran_c_shell.py`,
gds = `src/compiler/glsl_deployment_strategy.py`,
pts = `src/compiler/precompile_to_ssa.py`,
tsl = `src/compiler/tensor_ssa_lowering.py`,
lc = `src/compiler/loop_composer.py`, cs = `src/compiler/control_source.py`,
ic = `src/compiler/identity_concordance.py`,
tr = `src/common/tensors/topological_reducer.py`,
ssa = `src/transmogrifier/ssa.py`,
tp = `src/compiler/transformation_priority.py`,
hp = `src/compiler/hierarchical_plan.py`, ii = `src/compiler/ir_identities.py`,
scia = `src/compiler/ssa_call_input_adapters.py`,
srrs = `src/compiler/ssa_record_return_state.py`,
ssc = `src/compiler/ssa_self_check.py`, sb = `src/compiler/site_bundle.py`.

## 1. Step 4: planner structure (28 pages, 1562 cells)

Design section 4 names `planner_specializations`, `planner_tensor_descriptors`,
`_dispatch_metadata_cache`, `HierarchyValueTable.correlations`. The callsite
specialization ledgers, the operand-position pages and the tensor-descriptor
pages below are those ledgers' book faces (census 20 sections 1.2-1.4, 1.8).

| page | writer(s) | row | fact | mode | cells | read by others |
|---|---|---|---|---|---|---|
| `transformation_event` | tp:`TransformationLedger._record_event` | (S minted ledger scope, I event serial) | dict {identity, rule, proof, priority} | concord | 540/549/25 | tp:`_BookEventLog` only (same ledger); ledger consumed by fcs and `ssa_result_type_resolution` through the object |
| `transformation_decision` | tp:`TransformationLedger.propose` | (S minted ledger scope, L identity tuple) | (rule name, proof, target descriptor) | revise | 531/538/25 | tp:`incumbent_target`, tp:`_decision` (same class) |
| `transformation_rejection` | tp:`TransformationLedger.propose` | (S minted ledger scope, L identity, N rule, L proof, N retained rule, L retained proof) | event serial int | concord | 9/11/0 | tp:`propose` only |
| `structural_specialization_fixed_point` | gds:`_fold_callsite_structural_values` | bare str function name (NOT a tuple; needs (S function,)) | (digest str, changed bool, net tensor mutations tuple) | set, next column | 187/52/20 | none |
| `proven_shape` | ic:`record_proven_shape`, ic:`invalidate_proven_shape` (called from gds:`_tensor_descriptor`, gds:`_propagate_callsite_tensor_specializations`, gds:`_fold_callsite_structural_values`, gds:`_callsite_specialized_shell_type.publish_call_result_shape`, fcs:`_class_surface_ssa_program.recover_structural_source_outputs`, lc:`rewire_continuation` via `_invalidate_tensor_descriptor_dependents`) | (S authored function, V value); column = dependency level | ("proven", extents, dtype) or ("invalidated", cause id, reason) | set at level column | 122/13/0 | ic:`proven_shape_contract_of`, ic:`shape_store_report`, gds:`_tensor_descriptor`, sb:`_concord_region_value_metadata`, tools/view, probes |
| `consumer_operand` | gds:`_concord_consumer_operands` (concord); tr:`_set_operands` (revise: moves/retires) | (S read scope, V consumer node, V operand value) | tuple of (role, ordinal) | concord + revise | 74/78/2 | pts:`_ControlSSABuilder._region_feed` |
| `aggregate_ledger` | gds:`_record_aggregate_ledger_lookup` | (S owner, V value, L "ledger_lookup") | (state str, leaf count, resident count, node type, op, key tuple) | set col 0 on EVERY call (silent overwrite) | 25/6/0 | none |
| `source_control_specialization_concordance` | gds:`_fold_callsite_structural_values` (two branches) | (S function, V retained control id) | dict {graph_control_id, source_control_id, predicate_value_id, selected_role[, rejected]} | set, next column | 18/0/0 | none on the book (fcs reads the shadow set `structurally_specialized_conditional_node_ids`) |
| `proven_literal` | gds:`_fold_callsite_structural_values.replace` | (S function, V value) | the literal (any) | set, next column when different | 14/2/1 | fcs:`_class_surface_ssa_program` (ABI settlement) |
| `callsite_return_specialization` | gds:`_propagate_callsite_tensor_specializations.call_result_descriptor` | (S caller, N callee, V callsite node) | tuple of (shape, dtype) or None per output | set, next column every round | 10/1/0 | same function (`oscillating_rows`), probes |
| `value_shape` | fcs:`_class_surface_ssa_program.note_shape` (shape book) | (S function, V value) | (state str, extents, dtype, storage) | set at history length | 10/0/0 | tsl:`_shape_polymorphic_function`, fcs SSA-shape seam, tests, probes |
| `call_argument_operand` | gds:`_concord_call_argument_operands` | (S read scope, V callsite, I position) | (role, ordinal) | concord | 9/2/0 | pts:`_ControlSSABuilder._callsite_argument` |
| `call_edge` | fcs:`_class_surface_ssa_program` (edge enumeration) | (S callee, V callee id, N caller, V caller id) | "argument" | set col 0 | 9/2/0 | probes only |
| `source_callsite_activation_concordance` | gds:`ProcessGraphGLSLDeployment.__init__.plan_callsites` | (S caller identity, V callsite node) | (callee reference, mode str, recursive unit) | set col 0 after latest is None; raise on differ | 4/1/0 | ic audit (census 20), tests |
| `item_operand` | gds:`_concord_item_operands` | (S read scope, V item node) | (operand id, role, ordinal) | concord | 0/4/0 | pts:`_ControlSSABuilder._region_feed`, pts:`lower_control_sections_to_ssa` |
| `shape.node` | fcs:`_class_surface_ssa_program` (block present twice; second copy re-sets col 0) | (S function, V value) | extents tuple | set col 0 | 0/4/0 | probes only |
| `shape.linked` | same | (S function, V value) | linked contract shape | set col 0 | 0/4/0 | probes only |
| `operator_result_type_concordance` | hp:`plan_region_to_ssa_instrs` | (S function scope, I closure id, N region name, V output) | (opcode, "bool") | set col 0 after latest is None; raise | 0/6/0 | ic audit (census 20), tests |
| `formal_literal` | gds:`_publish_formal_literal` | (S function, N parameter) | ("proven", value, caller) then ("conflicting", (prev, value), caller) | set col 0 / next | 0/0/0 (mentioned) | gds:`_proven_formal_literal` |
| `formal_shape` | gds:`_publish_formal_shape` | (S authored function, N parameter) | ("proven" or "conflicting", extents, dtype, caller) | set col 0 / next | 0/0/0 (mentioned) | gds:`_proven_formal_shape`, gds:`_tensor_descriptor` |
| `operand_position_orphan` | gds:`_concord_consumer_operands` | (S read scope, V node, N role, I ordinal) | the `lexical_read_binding` fact at the same key | concord | 0/0/0 (mentioned) | ic `CorrelationTable` |
| `call_result_projection_concordance` | gds:`_build_shell_hierarchy_plan` | (V call node, V caller projection id) -- NO scope element | tuple of ints (authored result path) | concord | 0/0/0 (mentioned) | fcs:`_class_surface_ssa_program` (link order block, via scope_rows on the call node) |
| `tensor_shape_settlement_concordance` | gds:`_propagate_callsite_tensor_specializations` (settlement pass) | (S function, V semantic value) | (extents, dtype) | concord; raise on differ | 0/0/0 (mentioned) | none (the fixed point's seen set) |
| `source_precision_region_concordance` | gds:`_precision_indivisible_node_groups` | (S numeric scope, V terminal node) | (ordered members, promotes, collapses, widths) | set col 0 after latest is None; raise | 0/0/0 | ic audit, tsl (census 20); mirrored to node attribute `source_precision_region` |
| `linked_value_abi_polymorphism` | fcs:`_class_surface_ssa_program` (polymorphic formal) | (S callee, V callee id) | (caller, caller id, existing shape, new shape) | set at history length | 0/0/0 (mentioned) | tsl:`_shape_polymorphic_function` |
| `callsite_projection_specialization` | gds:`_publish_callsite_return_members.record` | (S caller, V callsite node, I index, V leaf) | (action str, descriptor receipt) | set, next column when changed | untouched | tests only |
| `callsite_projection_identity_concordance` | gds:`_publish_callsite_return_members`, gds:`_repair_missing_aggregate_leaf_projections` | ((caller, callsite node, `id(caller.G)`) as S, I index, V stale id) | replacement id | set col 0 after latest is None; raise | untouched | gds:`_retarget_all_cached_value_ids` |
| `callsite_tensor_result_specialization` | gds:`_propagate_callsite_tensor_specializations` (replacement branch) | (S caller, V callsite) | (previous receipt, replacement receipt) | set, next column | untouched | none |

## 2. Step 5: control SSA builder (18 pages, 2569 cells)

Design section 4 names `external_values`, name histories, aliases and loop
rewrites, carried ports, `finish` deriving `parameter_names`. The loop
composer's port/carried pages are included because the builder is their only
reader (census 20 section 1.6, census 30 section 1.1).

| page | writer(s) | row | fact | mode | cells | read by others |
|---|---|---|---|---|---|---|
| `loop_result_reconciliation` | pts:`_canonicalize_non_dominating_loop_result_uses._note` | (S function, V argument) | (outcome str, "block#index" str, op str, detail tuple) | set at history length; inside try/except | 2490/2292/132 | probes only |
| `region_value_dtype` | pts:`lower_control_sections_to_ssa.region_argument` | (S control scope str, V value) | dtype str | revise when different | 36/35/1 | `typed_region_value` in the same function only |
| `region_feed_consumer` | pts:`_concord_region_feed_consumers` | (S control scope str, I region index, V feed) | tuple of consumer result ids | concord | 34/32/2 | pts:`_ControlSSABuilder._region_feed` |
| `callsite_argument` | pts:`_ControlSSABuilder._note_callsite_arguments` | (S function, V callsite, I position) | (graph id, resolved value id, inside-loop bool) | set at history length; whole body in try/except | 9/2/0 | probes only |
| `loop_scope` | ic:`declare_loop_scope` (from pts:`_ControlSSABuilder.lower_loop`, `lower_while`) | (S authored function, V loop node, L key: "boundary" or a rebind ordinal); fixed columns OUTER/CARRIED/INNER/... | (header, latch, exit) or generation tuples | set at fixed columns; call wrapped in try/except | 0/0/6 | ic:`loop_scope_declarations`, ic `_loop_scope_findings`, tests |
| `loop_result_port_binding` | lc:`materialize_retained_loop_ports.add_port` | (S read scope, V port) | binding name str | concord | 0/0/2 | pts:`_ControlSSABuilder._break_bound_initial`, pts:`_publish_loop_result_ports.carried_entry_of` |
| `loop_region_membership` | lc:`analyze_shader_loop_reductions` | (S read scope, V loop node) | sorted region indices tuple | revise | 0/0/1 | cs:`place_loop_carried_region_producers.owned_regions` |
| `loop_carried_binding` | lc:`analyze_shader_loop_reductions` | (S read scope, V loop, V updated, V initial) | tuple of binding names | concord | 0/0/1 | pts:`_enter_loop_state`, `_loop_scope_rebinds`, `_publish_loop_result_ports`, `_split_region_captures_by_binding` |
| `loop_entry_state` | pts:`_ControlSSABuilder._enter_loop_state` | (S control scope str, V loop node, V initial id) | (bindings tuple, initial id) | concord | 0/0/1 | pts:`_ControlSSABuilder._resolve_read` |
| `loop_carried_entry` | pts:`_ControlSSABuilder._enter_loop_state` | (S control scope str, V loop node, N binding) | entry index int | concord | 0/0/1 | pts:`_ControlSSABuilder._carried_entry_value` |
| `loop_scope_inner_transition` | ic:`rebind_loop_scope_inner` (from pts latch completion) | (S authored function, V loop node, V declared inner) | (resident inner id, reason str) | set, next column | 0/0/1 | ic:`loop_scope_declarations` |
| `control_value_concordance` | pts:`lower_control_sections_to_ssa` (four sites incl. `bind_resident`) | (S control owner, V alias) | resident id | bind_alias | 0/0/0 (mentioned) | tsl:`lower_tensor_calls_to_repository_ssa.resident_sequence` (`resolve_alias`), ic:`resolved_concordant_alias_bindings` |
| `control_uniform_dtype` | pts:`lower_control_sections_to_ssa` | (S control scope str, V uniform value) | dtype str | concord | 0/0/0 (mentioned) | region formal typing in the same function |
| `region_capture_binding` | pts:`_split_region_captures_by_binding` | (S control scope str, I region index, V value) | (source value id, binding) | concord | 0/0/0 (mentioned) | pts:`_ControlSSABuilder._region_feed` |
| `tensor_shape_concordance` | pts:`_ControlSSABuilder._emit_table_lookup`, pts:`_ControlSSABuilder.lower_loop`, pts:`lower_control_sections_to_ssa` | (S control scope str, V result) | contract dict (storage, rank, dynamic state, minted cell ids, keyed handle) | set at next page-wide column | 0/0/0 (mentioned) | pts:`lower_loop.concorded_resident`, tsl (`pages.get`) |
| `field_slot_storage_concordance` | pts:`lower_control_sections_to_ssa` (record table build) | (S control function name, N field name or I slot) | ("nested_record", identity) | concord | 0/0/0 (mentioned) | none |
| `while_carried_test` | pts:`_ControlSSABuilder.lower_while` | loop-state key: (S control scope str, V loop node) | sorted binding names tuple, or the predicate value id | concord | untouched | none |
| `loop_continuation_rewire_concordance` | lc:`materialize_retained_loop_ports.rewire_continuation` | (S function name, V consumer, N role, I ordinal) | (old value id, new value id, binding) | revise | untouched | none |
| `identity_transition` (pts branch) | pts:`_ControlSSABuilder._region_feed` (rank-0 item merge) | (S control scope str, V value) -- a 2-element row on the page whose reducer rows have 4 elements | ("merge", source id, "scalar_item") | revise | (counted under step 3 below) | see step 3 |

## 3. Step 6: record materialization and return versions (17 pages, 74 cells)

Design section 4 names `record_return_layouts`, `scalar_return_field_versions`
selections, minted Casts, per-field phis. Census 10 sections 1.5, 1.7, 1.8
and census 30 section 1.5.

| page | writer(s) | row | fact | mode | cells | read by others |
|---|---|---|---|---|---|---|
| `record_field_access_path` | fcs:`_class_surface_ssa_program.note_record_field_access`; forwarding fixed point in the same function | (S minted access scope, L access key ((symbol, parameter), field path, storage identity), N role) | witness tuple ("source-node", symbol, node) / ("numeric-record-layout", identity) / ("callsite", ...) | concord | 33/9/0 | same function only; tests |
| `record_field_access` | same | (S minted access scope, L access key) | frozenset of roles {read, write, layout, sequence} | mapping[]= | 31/9/0 | same function (`field_access_receipts`); tests |
| `record_field_layout_concordance` | fcs:`_class_surface_ssa_program` (call-linking reconciliation), fcs:`...materialize_record_phis` | (S symbol, V result, N storage identity) | (value ids, sequence id, record id, offset, storage kind) | set col 0; raise on differ | 10/4/0 | same function, two `latest` sites |
| `numeral_record_literal_concordance` | fcs:`_class_surface_ssa_program.materialize_program_abi_record_literals` | THREE shapes: (S symbol, V record, N field) ; (S symbol, V node, L "deferred") ; (S symbol, V node, L "completed") | field value id ; True ; tuple of (field, record id, value ids) | concord | untouched | fcs:`...complete_linked_literals` (scans rows for the "deferred" marker) |
| `record_field_decomposition` | fcs:`...materialize_parameter_record_abi.publish_keyed_decomposition` | (S table owner (minted), V record, N storage identity) | tuple of ((role path), member id) | concord | untouched | fcs:`_linked_caller_member`, fcs:`...resolve_keyed_mapping_iterables`, fcs record/sequence reconciliation |
| `record_return_phi_input_concordance` | fcs:`_concord_record_return_phi_inputs` | (S function, V phi result, N field, I position, N predecessor block) | (candidate id, chosen id, reason[, selection cell key]) | set at next page-wide column | untouched | tests only |
| `record_phi_temporal_fallback_concordance` | srrs:`repair_non_dominating_record_phi_uses` | (S function, V result, N block, I use index, I position) | (fallback id, target block, "initial_record_field_version") | set + raise | untouched | none |
| `loop_record_layout_concordance` | fcs:`...materialize_record_phis` (loop phi) | (S symbol, V source loop node, V result) | (old physical layout, new layout, "record_loop_phi") | set col 0 | untouched | none; metadata twin `loop_record_layout_transitions` |
| `loop_record_schema_concordance` | fcs:`...materialize_loop_record_phis` | (S symbol, V loop id, V result) | (initial signature, updated signature, projected names, discarded, "project_updated_to_initial") | set col 0 | untouched | none; metadata twin `loop_record_schema_projections` |
| `record_field_storage_concordance` | fcs:`_class_surface_ssa_program` (call-linking field refinement) | (S caller, V caller record, N storage identity) | dict {from, to, sequence_id, record_id} | set, next column | 0/0/0 (mentioned) | none |
| `record_field_demand_concordance` | fcs:`_class_surface_ssa_program` (numeric record ABI) | (S symbol, N parameter) | sorted leaf paths tuple | concord | untouched | same function |
| `record_field_sequence_view` | fcs:`_class_surface_ssa_program` (sequence view settlement) | (S minted access scope, N storage identity) | the one decisive receipt | concord; raise unless exactly one | 0/0/0 (mentioned) | same function (`latest` in keyed materialization) |
| `numeral_leaf_materialization_concordance` | fcs:`...materialize_parameter_record_abi.materialize_nested_record`, `...materialize_declared_row_columns` | (S symbol, N parameter, L leaf/nested path) | status str ("required_unread_leaf" / "declared_row_leaf_deferred:<storage>") | concord | untouched | none |
| `numeral_leaf_width_concordance` | fcs:`...materialize_nested_record`; fcs:`...allocate_result_storage` | TWO shapes: (S symbol, N parameter, L path) ; (S caller, V callsite, V old id) | declared precision limbs int | concord | untouched | none |
| `numeral_return_leaves_concordance` | fcs:`...complete_linked_literals` | (S symbol, V returned argument) | tuple of leaf ids | concord | untouched | none |
| `program_abi_keyed_row_record` | fcs:`...materialize_parameter_record_abi` (keyed field) | (S table owner, V record, N storage identity) | row record id (minted) | concord | untouched | none |
| `receiver_nested_record_field_concordance` | fcs:`_field_slot_ops` | (S sequence contract scope, N attribute) | nested record identity str | concord | untouched | same function |

## 4. Step 7: frame linker (27 pages, 259 cells)

Design section 4 names `result_storage_bindings`, `frame_ledgers`, argument
storage minted from absence, `record_storage_alias`, name fallbacks at the
forwarding root. Census 10 sections 1.1, 1.3, 1.4, 1.5 (forwarding), 1.6.

| page | writer(s) | row | fact | mode | cells | read by others |
|---|---|---|---|---|---|---|
| `formal_actual_concordance` | ic:`concord_compiler_frame_formals` | (S callee, V formal, N caller, N block, I instruction index, I position) | actual id | concord | 148/121/2 | same function only (`scope_rows`/`latest`) |
| `argument_binding` | fcs:`_class_surface_ssa_program` (per linked call) | (S callee symbol, V formal, L "binding"); column = callsite id | (kind str, source id or literal) | set | 31/10/0 | ic:`CorrelationTable._binding_kind_findings`, ic:`materializing_binding_kind`, fcs:`_complete_propagated_frame_tails` (`cells.get`), tsl (`pages.get`), tests |
| `argument_binding_resolution` | ic:`materializing_binding_kind` | (S callee, V formal, V source) | ("caller_storage", requested kind, "materializing_binding_kind") | set, next column | 20/10/0 | ic:`CorrelationTable._binding_kind_findings` |
| `planning_value_concordance` | fcs:`_publish_concordant_function_aliases`, fcs:`_class_surface_ssa_program` (aliases merge, record projection block), fcs:`_retire_concorded_record_identity_aliases`, fcs:`_retire_dead_planning_aliases` | (S function/symbol, V alias) | resident id (None tombstone on retirement) | bind_alias; del mapping[] | 12/2/0 | ic:`concordant_alias_bindings`, ic:`resolved_concordant_alias_bindings`, fcs:`_concordant_function_aliases` (many sites), pts, tests, probes |
| `kernel_by_value_formal_concordance` | fcs:`_class_surface_ssa_program` (kernel shape harmonization) | (S callee, V formal) | "by_value" | concord | 11/14/0 | same function |
| `scheduled_call_argument` | fcs:`_class_surface_ssa_program` | (S minted scope, L (caller, callsite, callee formal id)) | the resident SSAValue object | mapping[]= | 9/2/0 | same function (`.get`) |
| `record_parameter_value` | fcs:`_class_surface_ssa_program` (declared record parameters) | (S minted access scope, L (symbol, value id)) | (symbol, parameter name) | concord; read through mapping | 6/8/0 | same function (forwarding) |
| `pruned_callee_formal_concordance` | fcs:`_prune_unused_callee_formals_once` | (S callee, V formal) | "unused" | concord | 6/2/0 | same function |
| `call_link_order_concordance` | fcs:`_class_surface_ssa_program` (link order) | (S artifact name, N symbol) | rank int | concord | 4/2/0 | same function |
| `record_storage_alias` | fcs:`...materialize_parameter_record_abi` (three branches + resolve pass) | (S minted ("record_storage_alias", symbol) scope, V alias) | resident id | mapping[]= | 4/2/0 | same function (rewrites operands) |
| `output_identity_concordance` | fcs:`_publish_concorded_output_identities` | (S function, V alias) | result id | bind_alias | 3/4/2 | fcs:`_concordant_function_aliases`, fcs:`_retire_concorded_record_identity_aliases`, ic audit, tests |
| `record_forwarding_edge` | fcs:`_class_surface_ssa_program` (binding walk) | (S minted access scope, L (caller key, callee key)) | (callsite id,) | mapping[]= | 2/2/0 | same function |
| `alias_application_concordance` | fcs:`_apply_concorded_function_aliases` | (S function, N block, I instruction index, I position) | (argument id, resident id, selected id, reason, target block) | set, first write only; raise on differ | 1/0/0 | tests only; metadata twin `alias_application_receipts` |
| `record_field_resident_concordance` | fcs:`...coalesce_record_field_storage` | (S symbol, V min parameter id, N field) | resident id | concord | 1/2/0 | same function |
| `phi_edge_projection_placement_concordance` | fcs:`_class_surface_ssa_program` (phi-edge projection relocation) | (S function, V phi, I position, N predecessor, V value) | (moved instruction ids, placement block, reason) | concord | 1/0/0 | none; metadata twin `phi_edge_projection_repairs` |
| `formal_storage_resolution` | ic:`concord_compiler_frame_formals` | (S function, V formal) | ("compiler_frame_storage", source receipts tuple) | set, next column | 0/4/0 | none |
| `planning_alias_transition_concordance` | fcs:`_publish_concordant_function_aliases` | (S function, V alias) | (incumbent resident, new resident, provenance str) | set, next column | 0/0/0 (mentioned) | ic `_planning_alias_transition_findings`, tests |
| `record_parameter_row_handle` | fcs:`_class_surface_ssa_program` (keyed row handles) | (S minted access scope, L (symbol, value id)) | ((symbol, parameter), "field[]", row identity) | concord; read through mapping | 0/0/0 (mentioned) | same function (forwarding) |
| `record_forwarding_unresolved_actual` | fcs:`_class_surface_ssa_program` (binding walk, three branches) | (S minted access scope, L (caller, callsite, caller id, callee, callee id)) | (reason str, caller key or None, callee key or None) | concord | 0/0/0 (mentioned) | none |
| `member_formals` | fcs:`_class_surface_ssa_program` (member formal accounting) | (S function, V value, L "member_formal"); column = aggregate index | ("granted"/"refused", root name, binding kind, authored parameters) | set | 0/0/0 (mentioned) | ic audit (census 10) |
| `program_abi_frame_transition` | ic:`concord_program_abi_frame_transitions` | (S function, V formal) | (program abi record, field str, retired provisional keys tuple) | set col 0; raise on differ | 0/0/0 (mentioned) | none |
| `assignment_projection_source_concordance` | fcs:`_class_surface_ssa_program` (entry) | (S source scope, N aggregate name, I projection) | target name str | concord | 0/0/0 (mentioned) | same function at link time, tests |
| `assignment_projection_identity_concordance` | fcs:`_class_surface_ssa_program` (link order block) | (S caller, V callsite, I projection) | (aggregate name, target name, caller projection id, target history ids) | concord | 0/0/0 (mentioned) | same function (result binding), tests |
| `call_record_pair_concordance` | fcs:`_class_surface_ssa_program` (child record pairing) | (S caller, V callsite, V callee child record) | caller child record id | concord | 0/0/0 (mentioned) | fcs:`_linked_caller_member` |
| `frame_lease_link` | fcs:`_link_frame_lease` (three linker call sites) | (S caller symbol, V slot) | (callsite id, callee symbol, callee formal id) | concord | untouched | none (write-only) |
| `propagated_frame_tail_concordance` | fcs:`_complete_propagated_frame_tails` | (S owner, L callsite id (may be None), N callee, V formal) | (slot id, storage kind str) | set col 0; raise on differ | untouched | none |
| `entry_record_handle_concordance` | fcs:`_prune_dead_entry_field_aliases` | (S function, N parameter) | "fields_only" | concord | untouched | same function |

## 5. Step 8: book-backed tables (12 pages, 47 cells)

Design section 4 names `SSARecordTable.register`'s merge and `_mint_table_owner`
scopes. Census 40 section 4.

| page | writer(s) | row | fact | mode | cells | read by others |
|---|---|---|---|---|---|---|
| `record_member` | ssa:`_revise_member_claims` (from `SSARecordTable` on_change) | (S minted table owner, V member value) | tuple of (record id, field, storage identity, role) | revise | 22/8/0 | ssa:`SSARecordTable.member_claims`, fcs call linking (via `member_claims`) |
| `call_record` | ssa:`SSACallTable.__setitem__`, `__delitem__`, ssa:`_BookCallList._commit` | (S minted table owner, N caller) | tuple of call records (None on delete) | revise | 20/5/0 | `SSACallTable` readers (fcs call linking) |
| `record_descriptor` | ssa:`_BookRows.__setitem__` / `__delitem__` via `SSARecordTable.register` | (S minted table owner, V record id) | SSARecordDescriptor (None on delete) | revise | 5/4/0 | every `record_table.records` reader (fcs), ic audit |
| `sequence_descriptor` | ssa:`_BookRows` via `SSASequenceTable.register` | (S owner, V sequence id) | SSASequenceDescriptor (None on delete) | revise | 0/0/0 (mentioned) | sequence table readers |
| `sequence_member` | ssa:`_revise_member_claims` (from `SSASequenceTable`) | (S owner, V member) | tuple of (sequence id, role) | revise | 0/0/0 (mentioned) | ssa:`SSASequenceTable.member_claims`, fcs:`_class_surface_ssa_program` |
| `struct_descriptor` | ssa:`_SSALayoutTable._publish` via `_BookRows` (`SSAStructTable`) | (S owner, V struct id) | SSAStructDescriptor | revise | 0/0/0 (mentioned) | layout table readers |
| `union_descriptor` | same (`SSAUnionTable`) | (S owner, V union id) | SSAUnionDescriptor | revise | 0/0/0 (mentioned) | layout table readers |
| `struct_member` | ssa:`_revise_member_claims` (layout tables) | (S owner, V member) | claims tuple | revise | 0/0/0 (mentioned) | ssa:`withdraw_superseded_layout_derivations`, `member_claims` |
| `union_member` | same | (S owner, V member) | claims tuple | revise | 0/0/0 (mentioned) | same |
| `layout_state` | ssa:`_SSALayoutTable._publish`, `register`, `withdraw_superseded_layout_derivations` | (S owner, N kind, V row id) | ("resolved", descriptor, edge row) or ("invalidated", ...) | revise when changed | 0/0/0 (mentioned) | ic:`CorrelationTable._layout_table_findings`, ssa:`layout_state` |
| `layout_supersession` | ssa:`_SSALayoutTable._record_supersession` | (S owner, N kind, V target, V source, N stage) | (incumbent, replacement) | set at next page-wide column | 0/0/0 (mentioned) | ssa:`supersessions` |
| `sequence_column_claims` | ssa:`SSASequenceTable.register` | (S owner, V sequence, L "column_dtypes") | (column dtypes, key columns) | revise, every attempt | untouched | ssa:`offered_column_dtypes` |

## 6. Section 4 names no owner (21 pages, 201 cells)

These are written by files section 4 never mentions (tensor-call lowering,
physical call-input adaptation, precision passes, the self check, the
sequence-contract helpers, the fcs shape/ABI settlement seam). The "nearest"
column is a proposal for the two planning lanes to accept or move; nothing
here is decided.

| page | writer(s) | row | fact | mode | cells | read by others | nearest |
|---|---|---|---|---|---|---|---|
| `tensor_shape_enrichment` | tsl:`propagate_repository_ssa_call_metadata` (three sites) | (S function name (falls back to "?"), V value, N field); column is a TUPLE (round, step), not an int | (old, new, authoritative bool, source id) | set | 79/68/0 | `oscillating_rows` in the writer; probe `probe_keyed_tensor` | 5 |
| `exact_region_feed_dtype` | scia:`_concord_exact_region_feed_dtypes` | TWO shapes: (L "feed", S scope str, V value) and ("formal", callee, formal id) -- the literal comes FIRST, so the scope is not element 0 | (exact dtype,) | set col 0; raise on differ | 52/59/4 | ic audit, tests | 7 |
| `cross_function_references` | tsl:`propagate_repository_ssa_call_metadata` | (S owner scope str, V value) | tuple of owner names | set col 0 | 36/35/0 | none | 5 |
| `record_scalar_shape_concordance` | fcs:`_class_surface_ssa_program` (declared scalar record fields) | (S function, V value) | (prior shape, ()) | concord | 17/0/0 | none | 4 |
| `ssa_call_shape` | tsl:`_publish_exact_ssa_call_shapes` | (S caller, V call result, N callee, V callee value) | (shape tuple, dtype) | set, next column | 7/4/0 | none outside tsl | 5 |
| `ssa_call_shape_evidence` | same | same row | evidence str | set, next column | 7/4/0 | none outside tsl | 5 |
| `scalar_kernel_operand_concordance` | tsl:`lower_tensor_calls_to_repository_ssa` (three sites) | (S function, V result) | "scalar_cast" / "scalar_fill" / ("scalar_operand", position) | concord | 1/3/0 | same function | 5 |
| `physical_call_input_adaptation_fixed_point` | scia:`adapt_physical_call_inputs` | (L "whole-program", L tuple of every function name); column = round | changes per round int | set at round column | 2/4/2 | none | 7 |
| `call_input_conversion` | scia:`_adapt_physical_call_inputs_round` | (S caller, V actual, N callee, V formal) | (converted id, source dtype, target dtype, kind) | set, next column | 0/6/0 | tests, probes; metadata twin `call_input_conversions` | 7 |
| `operator_result_type_concordance` | -- listed under step 4 (hp is the planner) | | | | | | |
| `formal_storage_resolution` | -- listed under step 7 | | | | | | |
| `sequence_row_layout_concordance` | ic:`commit_sequence_row_layout`, ic:`invalidate_sequence_row_layout` (from fcs `_publish_concordant_function_aliases`, `_sequence_column_dtype_contracts`; gds `_tensor_descriptor`) | (S authored function, V sequence id) | (column shapes, column dtypes, source str) or ("invalidated", cause id, reason) | set, next column; revise on invalidate | 0/0/0 (mentioned) | ic:`committed_sequence_row_layout` (many callers) | 8 |
| `ssa_shape_materialization` | fcs:`_class_surface_ssa_program` (SSA shape seam) | (S function, V value) | (shape, dtype, storage, authored owner) | set col 0; raise on differ | 0/0/0 (mentioned) | none; metadata `ssa_shape_materializations` | 4 |
| `shape.ssa` | tsl:`_record_ssa_shape` | (S function, V value) | extents | set col 0; try/except | untouched | ic:`shape_store_report` (f"shape.{name}") | 4 |
| `kernel_input_conversion` | tsl:`lower_tensor_calls_to_repository_ssa` | (S function, N block, V result, I operand position) | (source id, converted id, dtypes, kind) | set, next column | untouched | scia:`_adapt_physical_call_inputs_round`, tests; metadata twin | 5 |
| `tensor_reduction_domain_concordance` | tsl:`lower_tensor_calls_to_repository_ssa` | (S function, V result) | domain fact tuple | set col 0; raise on differ | untouched | none | 5 |
| `call_input_storage_concordance` | scia:`_adapt_physical_call_inputs_round` | (S caller, V actual) | "sequence_arena_by_reference" | concord | untouched | none | 7 |
| `precision_channel_shape_concordance` | ii:`carry_precision_through_ssa.carry` | (S function, V value) | (logical shape, channel shape, limbs) | set col 0; raise on differ | untouched | ii:`lower_precision_operations`, ic audit, tests | 7 |
| `precision_declared_formal_width_concordance` | ii:`lower_precision_operations` | (S function, V formal) | limbs int | concord | untouched | none | 7 |
| `single_input_phi_descriptor_concordance` | ii:`settle_single_input_phi_descriptors` | (S function, V phi result) | (source id, dtype, shape) | set col 0; raise on differ | untouched | ic audit, tests | 7 |
| `formal_parity` | ssc:`check_formal_parity` | (S function, V formal, L "unaccounted_formal") | (accounting keys tuple, channel fill counts) | set col 0; try/except pass | untouched | none (diagnostic) | 7 |
| `sequence_contract_concordance` | ic:`commit_sequence_contract` (from fcs `_field_slot_ops`, `_class_surface_ssa_program`; pts `_ControlSSABuilder.__init__`) | (S scope, V sequence id) | (policy, column count, writable, source stage str) | set, next column | untouched | ic:`committed_sequence_contract` callers, pts, tests | 8 |
| `sequence_row_dtype_concordance` | ic:`concord_sequence_row_dtypes` | (S scope, V sequence id) | tuple of column dtypes | set, next column | untouched | ic:`committed_sequence_row_dtypes` | 8 |

(The two rows marked "listed under" are cross-references, not pages; the
group holds 21 pages.)

## 7. Steps 1-3 remainder (25 pages, 2913 cells)

Written by the book itself (step 1), by the reducer's operand-position
machinery that step 3 said `_set_operands` would source, or by source-stage
declaration code that step 2 did not migrate. They are inventoried so the
count of raw-only pages is complete; their owner is whichever lane reopens
steps 2-3, not steps 4-8.

| page | writer(s) | row | fact | mode | cells | read by others |
|---|---|---|---|---|---|---|
| `lexical_read_binding` | tr:`_concord_lexical_reads` (concord), tr:`_normalize_lexical_values` (return roots, canonical relabel; concord), tr:`_set_operands` (revise: move/retire/fork) | THREE shapes on one page: (S read scope, L "occurrence", V occurrence) ; (S read scope, L "return", L "root", I position) ; (S read scope, V consumer, N role, I ordinal) | binding name str (None = vacated position) | concord + revise | 2453/1170/24 | tr:`lexical_read_binding` accessor, gds:`_concord_consumer_operands`, pts:`_ControlSSABuilder._operand_bindings`, pts:`_split_region_captures_by_binding` |
| `identity_transition` | tr:`_set_operands` (revise), tr:`fork_read_scope` (concord), pts:`_ControlSSABuilder._region_feed` (revise) | THREE shapes: (S read scope, V node, N role, I ordinal) ; (S forked scope, L "scope") ; (S control scope str, V value) | ("retire"/"move"/"fork", ..., cause str) / ("fork", source scope, cause) / ("merge", source id, "scalar_item") | revise + concord | 293/255/10 | none (no audit finding reads it; census 40) |
| `scope_registry` | ic:`IdentityBook.mint_scope` (from ssa:`_mint_table_owner`, tp:`TransformationLedger.__init__`, tr:`_normalize_lexical_values`, fcs access/alias scopes, `_operand_position_scope`) | (S label str, I serial) | True | concord | 125/69/18 | ic:`mint_scope` (`scope_row_count`), tools/view_identity_concordance |
| `source_python_identity_concordance` | tr:`_normalize_lexical_values.resolve_expression` | (S value-class scope, V node) | (qualified python name, kind, ..., ()) | set col 0; raise on differ | 30/4/0 | none |
| `source_type_normalization_concordance` | tr:`_normalize_lexical_values.resolve_expression` | (S value-class scope, V node) | (ids..., name, class) | set col 0; raise | 7/0/0 | none |
| `source_numeric_specialization_concordance` | tr:`specialize_python_precision_widths` | (S authored scope, N digest) | (scope, receipt) | set col 0; raise | 5/2/1 | none |
| `source_function_reachability_concordance` | tr:`reduce_abstract_tensor_topology.propagate_call_formal_numeric_types` | (S caller scope, V call, N callee scope) | True | set col 0; raise | 0/1/0 | none |
| `source_numeric_record_abi_concordance` | fcs:`_lower_ast_source_to_ssa_impl.numeric_record_schema` | (S type name, I limbs) | (schema name, receipt) | set col 0; raise | 0/0/0 (mentioned) | fcs:`...materialize_parameter_record_abi`, fcs:`...allocate_result_storage` (rows scan) |
| `source_precision_boundary_concordance` | tr:`_concord_source_precision_boundary` | (S numeric scope, V value) | (boundary str, operand id, limbs) | set col 0; raise | 0/0/0 (mentioned) | fcs:`...recover_structural_source_outputs.ensure_structural_value`, gds:`ProcessGraphGLSLDeployment.__init__`, gds:`_precision_indivisible_node_groups`, ii:`lower_precision_operations` |
| `source_precision_operator_concordance` | tr:`reduce_abstract_tensor_topology.lower_python_precision.concord_operator`, tr:`specialize_python_precision_widths` | (S numeric scope, V node) | (operation, receiver id, class identity, limbs) | set col 0; raise | 0/0/0 (mentioned) | tests write it directly |
| `source_operator_dispatch_concordance` | tr:`reduce_abstract_tensor_topology.lower_class_operator_calls.commit` | (S numeric scope, V node) | (receiver id, class identity, method name, method reference, ...) | set col 0; raise | 0/0/0 (mentioned) | none |
| `source_receiver_constructor_concordance` | tr:`reduce_abstract_tensor_topology.resolve_concorded_receiver_constructor` | (S numeric scope, V node) | (receiver id, class identity, limbs) | set col 0; raise | 0/0/0 (mentioned) | none |
| `source_numeric_type_dependency_concordance` | fcs:`_lower_ast_source_to_ssa_impl` | (S type name, I limbs) | dependency receipts tuple | set col 0; raise | 0/0/0 (mentioned) | ic audit (census 10) |
| `source_numeric_method_dependency_concordance` | fcs:`_lower_ast_source_to_ssa_impl` | (S type name, I limbs) | same-type method roots tuple | set col 0; raise | 0/0/0 (mentioned) | none |
| `source_numeric_parameter_abi_concordance` | fcs:`_lower_ast_source_to_ssa_impl` | (S definition name, N argument name) | (schema, type, limbs) | set | 0/0/0 (mentioned) | none |
| `source_numeric_parameter_record_view_concordance` | fcs:`_lower_ast_source_to_ssa_impl.numeric_parameter_record_views` | (S scope, N parameter, V node) | (schema, type, limbs) | set | 0/0/0 (mentioned) | none |
| `source_precision_pack_concordance` | tr:`specialize_python_precision_widths` | (S numeric scope, V node) | (leaves, width, value id) | set col 0; raise | untouched | none |
| `source_numeric_component_concordance` | tr:`_concord_numeric_feature_projection` | (S scope, V result) | (receiver id, attribute, component path, receipt) | set col 0; raise | untouched | none |
| `source_numeric_intrinsic_concordance` | tr:`reduce_abstract_tensor_topology.propagate_numeric_field_projections` | (S scope, V node) | (receiver id, method, results) | set col 0; raise | untouched | none |
| `source_numeric_operator_specialization_concordance` | tr:`reduce_abstract_tensor_topology.specialize_concorded_same_type_numeric_operator` | (S scope, V node) | tuple | set col 0; raise | untouched | none |
| `source_parameter_identity_concordance` | tr:`_concorded_static_parameter_bindings` | (S scope, N receiver name) | descriptor | set col 0; raise | untouched | none |
| `source_sequence_mutation_concordance` | tr:`_normalize_lexical_values.reduce_statement` | (S value-class scope, V call) | (initial id, method, policy, argument ids, "mapping"/"sequence") | set col 0; raise | untouched | none |
| `source_call_result_identity_concordance` | tr:`reduce_abstract_tensor_topology.propagate_call_formal_numeric_types` | (S caller scope, V call) | (callee ref, returned class, limbs) | set col 0; raise | untouched | none |
| `static_record_piece_mapping_concordance` | fcs:`_lower_ast_source_to_ssa_impl` | (S class identity, N field, L key) | (binding, entry, artifact) | concord | untouched | none |
| `static_record_piece_sequence_concordance` | fcs:`_lower_ast_source_to_ssa_impl` | (S class identity, N field) | tuple of (binding, entry, artifact) | concord | untouched | none |

## 8. (a) Pages whose row embeds a process counter or object id

These cannot be declared as-is: the row would be valid for one process and
`_validate_row` would admit it (SCOPE admits any hashable), so the refusal
has to come from the declaration lane, not the api.

| page | the offending element | source of the id |
|---|---|---|
| `callsite_projection_identity_concordance` | scope tuple (caller, callsite node, `id(caller.G)`) / (function name, node, `id(graph.G)`) | gds:`_publish_callsite_return_members`, gds:`_repair_missing_aggregate_leaf_projections` build the scope with the CPython object id of the graph |
| `control_uniform_dtype` | scope str | pts:`lower_control_sections_to_ssa` mints `tensor_shape_concordance_scope` as `f"{control_name}@control:{id(control):x}"`; the measured sample rows are the 45- and 79-character strings this produces |
| `region_value_dtype` | scope str | same scope |
| `region_feed_consumer` | scope str | same scope |
| `region_capture_binding` | scope str | same scope |
| `loop_entry_state` | scope str (via `_ControlSSABuilder.tensor_shape_concordance_scope`) | same scope |
| `loop_carried_entry` | scope str | same scope |
| `while_carried_test` | scope str (the loop-state key) | same scope |
| `tensor_shape_concordance` | scope str | same scope |
| `identity_transition` (pts branch only) | scope str | same scope; the reducer rows on the same page use minted `(label, serial)` scopes and are fine |
| `exact_region_feed_dtype` (feed-side rows) | element 1 is the same control scope str; element 0 is the literal "feed" | scia:`_concord_exact_region_feed_dtypes` copies the feed's control scope |
| `cross_function_references` | scope str | tsl:`propagate_repository_ssa_call_metadata` keys owners by `scopes[id(function)]`, a str derived from function metadata (`source_region_integral`); the measured scope is 59 characters and matches the control-scope pattern -- treat as suspect until the writer is read in full |

Related but not disqualifying (declarable after a reshape, no process id):

- `structural_specialization_fixed_point`: row is a bare str, not a tuple.
- `call_result_projection_concordance`: element 0 is a call node id, not a
  scope; needs a function/read scope prepended.
- `physical_call_input_adaptation_fixed_point`: row is ("whole-program",
  tuple of every function name in the program) -- a LABEL that changes
  whenever the program does; the column is the round index.
- `tensor_shape_enrichment`: the COLUMN is a tuple (round, step); `post`
  chooses the column itself, so the (round, step) pair must move into the
  fact or become two rows.
- `exact_region_feed_dtype`, `lexical_read_binding`, `identity_transition`,
  `numeral_record_literal_concordance`, `numeral_leaf_width_concordance`:
  two or three row shapes share one page name; each shape needs its own
  declared page (or one page with a LABEL discriminator in a fixed
  position, which the "feed"-first rows already violate).
- `scheduled_call_argument`: the fact is a live `SSAValue` object; `record_descriptor`,
  `sequence_descriptor`, `struct_descriptor`, `union_descriptor`: the fact
  is a descriptor object with `None` as the removal fact. Declarable with
  `fact_type=object`; the tombstone convention is not expressible in
  `_validate_fact`.

## 9. (b) Pages written only by tests or probes

None. Every raw-only page name mentioned in `tests/`, `tools/` or `scripts/`
also has a writer in `src`. The test-side writers found (all also written in
`src`): `argument_binding`, `exact_region_feed_dtype`,
`output_identity_concordance`, `planning_value_concordance`,
`precision_channel_shape_concordance`, `single_input_phi_descriptor_concordance`,
`source_precision_operator_concordance`, `value_shape`
(tests/test_argument_binding_concordance.py, tests/test_native_call_input_receipts.py,
tests/test_sequence_contract_concordance.py, tests/test_process_graph_function_linking.py,
tests/test_callsite_formal_shape_concordance.py). No writer under `tools/`.

The nearest thing: pages whose ONLY readers are tests or probes, i.e. the
page is a receipt nobody in `src` consumes:
`loop_result_reconciliation` (2490 cells, probes), `callsite_argument`
(probes), `call_edge` (probes), `shape.node` / `shape.linked` (probes),
`record_return_phi_input_concordance` (tests), `alias_application_concordance`
(tests), `callsite_projection_specialization` (tests), `call_input_conversion`
(tests, probes), `tensor_shape_enrichment` (one probe), `callsite_return_specialization`
(probes plus its own writer).

Pages read by nothing at all (write-only receipts): `aggregate_ledger`,
`structural_specialization_fixed_point`, `source_control_specialization_concordance`
(on the book), `tensor_shape_settlement_concordance`,
`callsite_tensor_result_specialization`, `while_carried_test`,
`loop_continuation_rewire_concordance`, `field_slot_storage_concordance`,
`record_phi_temporal_fallback_concordance`, `loop_record_layout_concordance`,
`loop_record_schema_concordance`, `record_field_storage_concordance`,
`numeral_leaf_materialization_concordance`, `numeral_leaf_width_concordance`,
`numeral_return_leaves_concordance`, `program_abi_keyed_row_record`,
`formal_storage_resolution`, `program_abi_frame_transition`,
`record_forwarding_unresolved_actual`, `frame_lease_link`,
`propagated_frame_tail_concordance`, `phi_edge_projection_placement_concordance`,
`cross_function_references`, `ssa_call_shape`, `ssa_call_shape_evidence`,
`physical_call_input_adaptation_fixed_point`, `call_input_storage_concordance`,
`precision_declared_formal_width_concordance`, `formal_parity`,
`tensor_reduction_domain_concordance`, and every `source_*` page in section 7
except `source_precision_boundary_concordance`, `source_numeric_record_abi_concordance`
and `source_numeric_type_dependency_concordance`.

## 10. DRAFT declaration blocks (for the owning plans to adopt or amend)

Spelling follows `src/compiler/concordance_declarations.py`: `declare_page(
name, (RowField(...), ...), fact_type)` with `K = RowFieldKind`. Fact types
are `object`/`tuple`/`str`/`int`/`bool`/`dict` as observed; a plan that
names a structured fact replaces the type with a frozen dataclass. Pages in
section 8(a) are declared here with the process-id element REPLACED by the
scope the plan must mint (marked `# (a)`); pages with several row shapes are
split. Order inside each block is by controller cell count.

### Step 4 (DRAFT)

```python
# ============================================================================
# Step 4: planner structure (DRAFT, census 75)
# ============================================================================
TRANSFORMATION_EVENT = declare_page("transformation_event", (
    RowField("ledger_scope", K.SCOPE), RowField("serial", K.INDEX),
), dict)                                # mode CONCORD
TRANSFORMATION_DECISION = declare_page("transformation_decision", (
    RowField("ledger_scope", K.SCOPE), RowField("identity", K.LABEL),
), tuple)                               # mode REVISE
TRANSFORMATION_REJECTION = declare_page("transformation_rejection", (
    RowField("ledger_scope", K.SCOPE), RowField("identity", K.LABEL),
    RowField("rule", K.NAME), RowField("proof", K.LABEL),
    RowField("retained_rule", K.NAME), RowField("retained_proof", K.LABEL),
), int)                                 # mode CONCORD
STRUCTURAL_SPECIALIZATION_FIXED_POINT = declare_page("structural_specialization_fixed_point", (
    RowField("function", K.SCOPE),      # today a bare str: wrap in a 1-tuple
), tuple)                               # mode REVISE
PROVEN_SHAPE = declare_page("proven_shape", (
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)                               # mode REVISE; level column -> fact
CONSUMER_OPERAND = declare_page("consumer_operand", (
    RowField("read_scope", K.SCOPE), RowField("consumer", K.VALUE_ID),
    RowField("operand", K.VALUE_ID),
), tuple)                               # mode CONCORD; _set_operands REVISE
AGGREGATE_LEDGER = declare_page("aggregate_ledger", (
    RowField("owner", K.SCOPE), RowField("value_id", K.VALUE_ID),
    RowField("key", K.LABEL),
), tuple)                               # mode REVISE (today: overwrite)
SOURCE_CONTROL_SPECIALIZATION = declare_page("source_control_specialization_concordance", (
    RowField("function", K.SCOPE), RowField("control_id", K.VALUE_ID),
), dict)                                # mode REVISE
PROVEN_LITERAL = declare_page("proven_literal", (
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
), object)                              # mode REVISE
CALLSITE_RETURN_SPECIALIZATION = declare_page("callsite_return_specialization", (
    RowField("caller", K.SCOPE), RowField("callee", K.NAME),
    RowField("callsite", K.VALUE_ID),
), tuple)                               # mode REVISE
VALUE_SHAPE = declare_page("value_shape", (
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)                               # mode REVISE
CALL_ARGUMENT_OPERAND = declare_page("call_argument_operand", (
    RowField("read_scope", K.SCOPE), RowField("callsite", K.VALUE_ID),
    RowField("position", K.INDEX),
), tuple)                               # mode CONCORD
CALL_EDGE = declare_page("call_edge", (
    RowField("callee", K.SCOPE), RowField("callee_id", K.VALUE_ID),
    RowField("caller", K.NAME), RowField("caller_id", K.VALUE_ID),
), str)                                 # mode CONCORD
SOURCE_CALLSITE_ACTIVATION = declare_page("source_callsite_activation_concordance", (
    RowField("caller_identity", K.SCOPE), RowField("callsite", K.VALUE_ID),
), tuple)                               # mode CONCORD
ITEM_OPERAND = declare_page("item_operand", (
    RowField("read_scope", K.SCOPE), RowField("item", K.VALUE_ID),
), tuple)                               # mode CONCORD
SHAPE_NODE = declare_page("shape.node", (
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)                               # mode CONCORD (delete the duplicate block)
SHAPE_LINKED = declare_page("shape.linked", (
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)                               # mode CONCORD
OPERATOR_RESULT_TYPE = declare_page("operator_result_type_concordance", (
    RowField("function_scope", K.SCOPE), RowField("closure_id", K.INDEX),
    RowField("region", K.NAME), RowField("output_id", K.VALUE_ID),
), tuple)                               # mode CONCORD
FORMAL_LITERAL = declare_page("formal_literal", (
    RowField("function", K.SCOPE), RowField("parameter", K.NAME),
), tuple)                               # mode REVISE
FORMAL_SHAPE = declare_page("formal_shape", (
    RowField("function", K.SCOPE), RowField("parameter", K.NAME),
), tuple)                               # mode REVISE
OPERAND_POSITION_ORPHAN = declare_page("operand_position_orphan", (
    RowField("read_scope", K.SCOPE), RowField("node", K.VALUE_ID),
    RowField("role", K.NAME), RowField("ordinal", K.INDEX),
), object)                              # mode CONCORD
CALL_RESULT_PROJECTION = declare_page("call_result_projection_concordance", (
    RowField("caller", K.SCOPE),        # NEW element: today the row has no scope
    RowField("call_node", K.VALUE_ID), RowField("projection_id", K.VALUE_ID),
), tuple)                               # mode CONCORD
TENSOR_SHAPE_SETTLEMENT = declare_page("tensor_shape_settlement_concordance", (
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)                               # mode CONCORD
SOURCE_PRECISION_REGION = declare_page("source_precision_region_concordance", (
    RowField("numeric_scope", K.SCOPE), RowField("terminal", K.VALUE_ID),
), tuple)                               # mode CONCORD
LINKED_VALUE_ABI_POLYMORPHISM = declare_page("linked_value_abi_polymorphism", (
    RowField("callee", K.SCOPE), RowField("callee_id", K.VALUE_ID),
), tuple)                               # mode REVISE
CALLSITE_PROJECTION_SPECIALIZATION = declare_page("callsite_projection_specialization", (
    RowField("caller", K.SCOPE), RowField("callsite", K.VALUE_ID),
    RowField("index", K.INDEX), RowField("leaf", K.VALUE_ID),
), tuple)                               # mode REVISE
CALLSITE_PROJECTION_IDENTITY = declare_page("callsite_projection_identity_concordance", (
    RowField("caller", K.SCOPE),        # (a): today (caller, callsite, id(caller.G))
    RowField("callsite", K.VALUE_ID),   #      -> the graph copy must be a minted scope
    RowField("index", K.INDEX), RowField("stale_id", K.VALUE_ID),
), int)                                 # mode CONCORD
CALLSITE_TENSOR_RESULT_SPECIALIZATION = declare_page("callsite_tensor_result_specialization", (
    RowField("caller", K.SCOPE), RowField("callsite", K.VALUE_ID),
), tuple)                               # mode REVISE
```

### Step 5 (DRAFT)

```python
# ============================================================================
# Step 5: control SSA builder (DRAFT, census 75)
# ============================================================================
# (a): every "control_scope" below is today f"{name}@control:{id(control):x}";
# the plan mints it (book.mint_scope(("control", name))) before declaring.
LOOP_RESULT_RECONCILIATION = declare_page("loop_result_reconciliation", (
    RowField("function", K.SCOPE), RowField("argument", K.VALUE_ID),
), tuple)                               # mode REVISE
REGION_VALUE_DTYPE = declare_page("region_value_dtype", (
    RowField("control_scope", K.SCOPE), RowField("value_id", K.VALUE_ID),
), str)                                 # (a); mode REVISE
REGION_FEED_CONSUMER = declare_page("region_feed_consumer", (
    RowField("control_scope", K.SCOPE), RowField("region", K.INDEX),
    RowField("feed", K.VALUE_ID),
), tuple)                               # (a); mode CONCORD
CALLSITE_ARGUMENT = declare_page("callsite_argument", (
    RowField("function", K.SCOPE), RowField("callsite", K.VALUE_ID),
    RowField("position", K.INDEX),
), tuple)                               # mode REVISE
LOOP_SCOPE = declare_page("loop_scope", (
    RowField("function", K.SCOPE), RowField("loop_node", K.VALUE_ID),
    RowField("key", K.LABEL),
), tuple)                               # mode CONCORD; fixed columns -> LABEL key
LOOP_RESULT_PORT_BINDING = declare_page("loop_result_port_binding", (
    RowField("read_scope", K.SCOPE), RowField("port", K.VALUE_ID),
), str)                                 # mode CONCORD
LOOP_REGION_MEMBERSHIP = declare_page("loop_region_membership", (
    RowField("read_scope", K.SCOPE), RowField("loop_node", K.VALUE_ID),
), tuple)                               # mode REVISE
LOOP_CARRIED_BINDING = declare_page("loop_carried_binding", (
    RowField("read_scope", K.SCOPE), RowField("loop_node", K.VALUE_ID),
    RowField("updated", K.VALUE_ID), RowField("initial", K.VALUE_ID),
), tuple)                               # mode CONCORD
LOOP_ENTRY_STATE = declare_page("loop_entry_state", (
    RowField("control_scope", K.SCOPE), RowField("loop_node", K.VALUE_ID),
    RowField("initial", K.VALUE_ID),
), tuple)                               # (a); mode CONCORD
LOOP_CARRIED_ENTRY = declare_page("loop_carried_entry", (
    RowField("control_scope", K.SCOPE), RowField("loop_node", K.VALUE_ID),
    RowField("binding", K.NAME),
), int)                                 # (a); mode CONCORD
LOOP_SCOPE_INNER_TRANSITION = declare_page("loop_scope_inner_transition", (
    RowField("function", K.SCOPE), RowField("loop_node", K.VALUE_ID),
    RowField("declared_inner", K.VALUE_ID),
), tuple)                               # mode REVISE
CONTROL_VALUE_CONCORDANCE = declare_page("control_value_concordance", (
    RowField("control_owner", K.SCOPE), RowField("alias", K.VALUE_ID),
), int)                                 # mode REVISE (bind_alias)
CONTROL_UNIFORM_DTYPE = declare_page("control_uniform_dtype", (
    RowField("control_scope", K.SCOPE), RowField("value_id", K.VALUE_ID),
), str)                                 # (a); mode CONCORD
REGION_CAPTURE_BINDING = declare_page("region_capture_binding", (
    RowField("control_scope", K.SCOPE), RowField("region", K.INDEX),
    RowField("value_id", K.VALUE_ID),
), tuple)                               # (a); mode CONCORD
TENSOR_SHAPE_CONCORDANCE = declare_page("tensor_shape_concordance", (
    RowField("control_scope", K.SCOPE), RowField("value_id", K.VALUE_ID),
), dict)                                # (a); mode REVISE
FIELD_SLOT_STORAGE = declare_page("field_slot_storage_concordance", (
    RowField("function", K.SCOPE), RowField("slot", K.LABEL),
), tuple)                               # mode CONCORD
WHILE_CARRIED_TEST = declare_page("while_carried_test", (
    RowField("control_scope", K.SCOPE), RowField("loop_node", K.VALUE_ID),
), object)                              # (a); mode CONCORD; tuple | int fact
LOOP_CONTINUATION_REWIRE = declare_page("loop_continuation_rewire_concordance", (
    RowField("function", K.SCOPE), RowField("consumer", K.VALUE_ID),
    RowField("role", K.NAME), RowField("ordinal", K.INDEX),
), tuple)                               # mode REVISE
CONTROL_ITEM_MERGE = declare_page("control_item_merge", (   # split of identity_transition
    RowField("control_scope", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)                               # (a); mode REVISE
```

### Step 6 (DRAFT)

```python
# ============================================================================
# Step 6: record materialization and return versions (DRAFT, census 75)
# ============================================================================
RECORD_FIELD_ACCESS_PATH = declare_page("record_field_access_path", (
    RowField("access_scope", K.SCOPE), RowField("access_key", K.LABEL),
    RowField("role", K.NAME),
), tuple)                               # mode CONCORD
RECORD_FIELD_ACCESS = declare_page("record_field_access", (
    RowField("access_scope", K.SCOPE), RowField("access_key", K.LABEL),
), frozenset)                           # mode REVISE
RECORD_FIELD_LAYOUT = declare_page("record_field_layout_concordance", (
    RowField("symbol", K.SCOPE), RowField("result", K.VALUE_ID),
    RowField("storage_identity", K.NAME),
), tuple)                               # mode CONCORD
NUMERAL_RECORD_LITERAL_FIELD = declare_page("numeral_record_literal_field", (
    RowField("symbol", K.SCOPE), RowField("record", K.VALUE_ID),
    RowField("field", K.NAME),
), int)                                 # split 1 of numeral_record_literal_concordance
NUMERAL_RECORD_LITERAL_STATE = declare_page("numeral_record_literal_state", (
    RowField("symbol", K.SCOPE), RowField("node", K.VALUE_ID),
    RowField("state", K.LABEL),         # "deferred" | "completed"
), object)                              # split 2; mode CONCORD
RECORD_FIELD_DECOMPOSITION = declare_page("record_field_decomposition", (
    RowField("owner", K.SCOPE), RowField("record", K.VALUE_ID),
    RowField("storage_identity", K.NAME),
), tuple)                               # mode CONCORD
RECORD_RETURN_PHI_INPUT = declare_page("record_return_phi_input_concordance", (
    RowField("function", K.SCOPE), RowField("phi", K.VALUE_ID),
    RowField("field", K.NAME), RowField("position", K.INDEX),
    RowField("predecessor", K.NAME),
), tuple)                               # mode REVISE (Derived from RECORD_RETURN_FIELD_SELECTION)
RECORD_PHI_TEMPORAL_FALLBACK = declare_page("record_phi_temporal_fallback_concordance", (
    RowField("function", K.SCOPE), RowField("result", K.VALUE_ID),
    RowField("block", K.NAME), RowField("use_index", K.INDEX),
    RowField("position", K.INDEX),
), tuple)                               # mode CONCORD
LOOP_RECORD_LAYOUT = declare_page("loop_record_layout_concordance", (
    RowField("symbol", K.SCOPE), RowField("loop_node", K.VALUE_ID),
    RowField("result", K.VALUE_ID),
), tuple)                               # mode CONCORD
LOOP_RECORD_SCHEMA = declare_page("loop_record_schema_concordance", (
    RowField("symbol", K.SCOPE), RowField("loop_node", K.VALUE_ID),
    RowField("result", K.VALUE_ID),
), tuple)                               # mode CONCORD
RECORD_FIELD_STORAGE = declare_page("record_field_storage_concordance", (
    RowField("caller", K.SCOPE), RowField("record", K.VALUE_ID),
    RowField("storage_identity", K.NAME),
), dict)                                # mode REVISE
RECORD_FIELD_DEMAND = declare_page("record_field_demand_concordance", (
    RowField("symbol", K.SCOPE), RowField("parameter", K.NAME),
), tuple)                               # mode CONCORD
RECORD_FIELD_SEQUENCE_VIEW = declare_page("record_field_sequence_view", (
    RowField("access_scope", K.SCOPE), RowField("storage_identity", K.NAME),
), object)                              # mode CONCORD
NUMERAL_LEAF_MATERIALIZATION = declare_page("numeral_leaf_materialization_concordance", (
    RowField("symbol", K.SCOPE), RowField("parameter", K.NAME),
    RowField("path", K.LABEL),
), str)                                 # mode CONCORD
NUMERAL_LEAF_WIDTH = declare_page("numeral_leaf_width_concordance", (
    RowField("symbol", K.SCOPE), RowField("parameter", K.NAME),
    RowField("path", K.LABEL),
), int)                                 # split 1; mode CONCORD
NUMERAL_CALLSITE_LEAF_WIDTH = declare_page("numeral_callsite_leaf_width", (
    RowField("caller", K.SCOPE), RowField("callsite", K.VALUE_ID),
    RowField("old_id", K.VALUE_ID),
), int)                                 # split 2 of numeral_leaf_width_concordance
NUMERAL_RETURN_LEAVES = declare_page("numeral_return_leaves_concordance", (
    RowField("symbol", K.SCOPE), RowField("argument", K.VALUE_ID),
), tuple)                               # mode CONCORD
PROGRAM_ABI_KEYED_ROW_RECORD = declare_page("program_abi_keyed_row_record", (
    RowField("owner", K.SCOPE), RowField("record", K.VALUE_ID),
    RowField("storage_identity", K.NAME),
), int)                                 # mode CONCORD (Novel: the row record id is minted here)
RECEIVER_NESTED_RECORD_FIELD = declare_page("receiver_nested_record_field_concordance", (
    RowField("contract_scope", K.SCOPE), RowField("attribute", K.NAME),
), str)                                 # mode CONCORD
```

### Step 7 (DRAFT)

```python
# ============================================================================
# Step 7: frame linker (DRAFT, census 75)
# ============================================================================
FORMAL_ACTUAL = declare_page("formal_actual_concordance", (
    RowField("callee", K.SCOPE), RowField("formal", K.VALUE_ID),
    RowField("caller", K.NAME), RowField("block", K.NAME),
    RowField("instruction", K.INDEX), RowField("position", K.INDEX),
), int)                                 # mode CONCORD
ARGUMENT_BINDING = declare_page("argument_binding", (
    RowField("callee", K.SCOPE), RowField("formal", K.VALUE_ID),
    RowField("callsite", K.VALUE_ID),   # today the callsite is the COLUMN
), tuple)                               # mode REVISE
ARGUMENT_BINDING_RESOLUTION = declare_page("argument_binding_resolution", (
    RowField("callee", K.SCOPE), RowField("formal", K.VALUE_ID),
    RowField("source", K.VALUE_ID),
), tuple)                               # mode REVISE
PLANNING_VALUE = declare_page("planning_value_concordance", (
    RowField("function", K.SCOPE), RowField("alias", K.VALUE_ID),
), int)                                 # mode REVISE (bind_alias; None tombstone)
KERNEL_BY_VALUE_FORMAL = declare_page("kernel_by_value_formal_concordance", (
    RowField("callee", K.SCOPE), RowField("formal", K.VALUE_ID),
), str)                                 # mode CONCORD
SCHEDULED_CALL_ARGUMENT = declare_page("scheduled_call_argument", (
    RowField("scope", K.SCOPE), RowField("call_formal", K.LABEL),
), object)                              # mode REVISE; fact is an SSAValue
RECORD_PARAMETER_VALUE = declare_page("record_parameter_value", (
    RowField("access_scope", K.SCOPE), RowField("symbol_value", K.LABEL),
), tuple)                               # mode CONCORD
PRUNED_CALLEE_FORMAL = declare_page("pruned_callee_formal_concordance", (
    RowField("callee", K.SCOPE), RowField("formal", K.VALUE_ID),
), str)                                 # mode CONCORD
CALL_LINK_ORDER = declare_page("call_link_order_concordance", (
    RowField("artifact", K.SCOPE), RowField("symbol", K.NAME),
), int)                                 # mode CONCORD
RECORD_STORAGE_ALIAS = declare_page("record_storage_alias", (
    RowField("alias_scope", K.SCOPE), RowField("alias", K.VALUE_ID),
), int)                                 # mode REVISE
OUTPUT_IDENTITY = declare_page("output_identity_concordance", (
    RowField("function", K.SCOPE), RowField("alias", K.VALUE_ID),
), int)                                 # mode REVISE (bind_alias)
RECORD_FORWARDING_EDGE = declare_page("record_forwarding_edge", (
    RowField("access_scope", K.SCOPE), RowField("edge_key", K.LABEL),
), tuple)                               # mode REVISE
ALIAS_APPLICATION = declare_page("alias_application_concordance", (
    RowField("function", K.SCOPE), RowField("block", K.NAME),
    RowField("instruction", K.INDEX), RowField("position", K.INDEX),
), tuple)                               # mode CONCORD
RECORD_FIELD_RESIDENT = declare_page("record_field_resident_concordance", (
    RowField("symbol", K.SCOPE), RowField("parameter", K.VALUE_ID),
    RowField("field", K.NAME),
), int)                                 # mode CONCORD
PHI_EDGE_PROJECTION_PLACEMENT = declare_page("phi_edge_projection_placement_concordance", (
    RowField("function", K.SCOPE), RowField("phi", K.VALUE_ID),
    RowField("position", K.INDEX), RowField("predecessor", K.NAME),
    RowField("value_id", K.VALUE_ID),
), tuple)                               # mode CONCORD
FORMAL_STORAGE_RESOLUTION = declare_page("formal_storage_resolution", (
    RowField("function", K.SCOPE), RowField("formal", K.VALUE_ID),
), tuple)                               # mode REVISE
PLANNING_ALIAS_TRANSITION = declare_page("planning_alias_transition_concordance", (
    RowField("function", K.SCOPE), RowField("alias", K.VALUE_ID),
), tuple)                               # mode REVISE
RECORD_PARAMETER_ROW_HANDLE = declare_page("record_parameter_row_handle", (
    RowField("access_scope", K.SCOPE), RowField("symbol_value", K.LABEL),
), tuple)                               # mode CONCORD
RECORD_FORWARDING_UNRESOLVED_ACTUAL = declare_page("record_forwarding_unresolved_actual", (
    RowField("access_scope", K.SCOPE), RowField("binding", K.LABEL),
), tuple)                               # mode CONCORD (Unsourced with a reason)
MEMBER_FORMALS = declare_page("member_formals", (
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
    RowField("aggregate_index", K.INDEX),   # today the COLUMN
), tuple)                               # mode CONCORD
PROGRAM_ABI_FRAME_TRANSITION = declare_page("program_abi_frame_transition", (
    RowField("function", K.SCOPE), RowField("formal", K.VALUE_ID),
), tuple)                               # mode CONCORD
ASSIGNMENT_PROJECTION_SOURCE = declare_page("assignment_projection_source_concordance", (
    RowField("source_scope", K.SCOPE), RowField("aggregate", K.NAME),
    RowField("projection", K.INDEX),
), str)                                 # mode CONCORD
ASSIGNMENT_PROJECTION_IDENTITY = declare_page("assignment_projection_identity_concordance", (
    RowField("caller", K.SCOPE), RowField("callsite", K.VALUE_ID),
    RowField("projection", K.INDEX),
), tuple)                               # mode CONCORD
CALL_RECORD_PAIR = declare_page("call_record_pair_concordance", (
    RowField("caller", K.SCOPE), RowField("callsite", K.VALUE_ID),
    RowField("callee_record", K.VALUE_ID),
), int)                                 # mode CONCORD
FRAME_LEASE_LINK = declare_page("frame_lease_link", (
    RowField("caller", K.SCOPE), RowField("slot", K.VALUE_ID),
), tuple)                               # mode CONCORD
PROPAGATED_FRAME_TAIL = declare_page("propagated_frame_tail_concordance", (
    RowField("owner", K.SCOPE), RowField("callsite", K.LABEL),
    RowField("callee", K.NAME), RowField("formal", K.VALUE_ID),
), tuple)                               # mode CONCORD
ENTRY_RECORD_HANDLE = declare_page("entry_record_handle_concordance", (
    RowField("function", K.SCOPE), RowField("parameter", K.NAME),
), str)                                 # mode CONCORD
```

### Step 8 (DRAFT)

```python
# ============================================================================
# Step 8: book-backed tables (DRAFT, census 75)
# ============================================================================
RECORD_MEMBER = declare_page("record_member", (
    RowField("owner", K.SCOPE), RowField("member", K.VALUE_ID),
), tuple)                               # mode REVISE
CALL_RECORD = declare_page("call_record", (
    RowField("owner", K.SCOPE), RowField("caller", K.NAME),
), object)                              # mode REVISE; tuple | None
RECORD_DESCRIPTOR = declare_page("record_descriptor", (
    RowField("owner", K.SCOPE), RowField("record", K.VALUE_ID),
), object)                              # mode REVISE; SSARecordDescriptor | None
SEQUENCE_DESCRIPTOR = declare_page("sequence_descriptor", (
    RowField("owner", K.SCOPE), RowField("sequence", K.VALUE_ID),
), object)                              # mode REVISE
SEQUENCE_MEMBER = declare_page("sequence_member", (
    RowField("owner", K.SCOPE), RowField("member", K.VALUE_ID),
), tuple)                               # mode REVISE
STRUCT_DESCRIPTOR = declare_page("struct_descriptor", (
    RowField("owner", K.SCOPE), RowField("struct", K.VALUE_ID),
), object)                              # mode REVISE
UNION_DESCRIPTOR = declare_page("union_descriptor", (
    RowField("owner", K.SCOPE), RowField("union", K.VALUE_ID),
), object)                              # mode REVISE
STRUCT_MEMBER = declare_page("struct_member", (
    RowField("owner", K.SCOPE), RowField("member", K.VALUE_ID),
), tuple)                               # mode REVISE
UNION_MEMBER = declare_page("union_member", (
    RowField("owner", K.SCOPE), RowField("member", K.VALUE_ID),
), tuple)                               # mode REVISE
LAYOUT_STATE = declare_page("layout_state", (
    RowField("owner", K.SCOPE), RowField("kind", K.NAME),
    RowField("row_id", K.VALUE_ID),
), tuple)                               # mode REVISE
LAYOUT_SUPERSESSION = declare_page("layout_supersession", (
    RowField("owner", K.SCOPE), RowField("kind", K.NAME),
    RowField("target", K.VALUE_ID), RowField("source", K.VALUE_ID),
    RowField("stage", K.NAME),
), tuple)                               # mode CONCORD (one edge per row)
SEQUENCE_COLUMN_CLAIMS = declare_page("sequence_column_claims", (
    RowField("owner", K.SCOPE), RowField("sequence", K.VALUE_ID),
    RowField("key", K.LABEL),
), tuple)                               # mode REVISE
# nearest-8 from section 6:
SEQUENCE_CONTRACT = declare_page("sequence_contract_concordance", (
    RowField("scope", K.SCOPE), RowField("sequence", K.VALUE_ID),
), tuple)                               # mode REVISE
SEQUENCE_ROW_LAYOUT = declare_page("sequence_row_layout_concordance", (
    RowField("function", K.SCOPE), RowField("sequence", K.VALUE_ID),
), tuple)                               # mode REVISE
SEQUENCE_ROW_DTYPE = declare_page("sequence_row_dtype_concordance", (
    RowField("scope", K.SCOPE), RowField("sequence", K.VALUE_ID),
), tuple)                               # mode REVISE
```

### Unowned by section 4, remaining (DRAFT, nearest step in the comment)

```python
TENSOR_SHAPE_ENRICHMENT = declare_page("tensor_shape_enrichment", (      # nearest 5
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
    RowField("field", K.NAME),
), tuple)                               # mode REVISE; (round, step) moves into the fact
EXACT_REGION_FEED_DTYPE = declare_page("exact_region_feed_dtype", (     # nearest 7
    RowField("control_scope", K.SCOPE),   # (a); today ("feed", scope, id)
    RowField("value_id", K.VALUE_ID),
), tuple)                               # mode CONCORD; formal-side rows -> a second page
EXACT_FORMAL_DTYPE = declare_page("exact_formal_dtype", (               # split
    RowField("callee", K.SCOPE), RowField("formal", K.VALUE_ID),
), tuple)
CROSS_FUNCTION_REFERENCES = declare_page("cross_function_references", (  # nearest 5
    RowField("owner", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)                               # (a)? see section 8
RECORD_SCALAR_SHAPE = declare_page("record_scalar_shape_concordance", (  # nearest 4
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)
SSA_CALL_SHAPE = declare_page("ssa_call_shape", (                       # nearest 5
    RowField("caller", K.SCOPE), RowField("result", K.VALUE_ID),
    RowField("callee", K.NAME), RowField("callee_value", K.VALUE_ID),
), tuple)                               # mode REVISE
SSA_CALL_SHAPE_EVIDENCE = declare_page("ssa_call_shape_evidence", (
    RowField("caller", K.SCOPE), RowField("result", K.VALUE_ID),
    RowField("callee", K.NAME), RowField("callee_value", K.VALUE_ID),
), str)
SCALAR_KERNEL_OPERAND = declare_page("scalar_kernel_operand_concordance", (  # nearest 5
    RowField("function", K.SCOPE), RowField("result", K.VALUE_ID),
), object)                              # str | tuple; mode CONCORD
PHYSICAL_CALL_INPUT_ADAPTATION_ROUND = declare_page("physical_call_input_adaptation_fixed_point", (  # nearest 7
    RowField("program", K.SCOPE), RowField("round", K.INDEX),   # today the round is the column
), int)
CALL_INPUT_CONVERSION = declare_page("call_input_conversion", (         # nearest 7
    RowField("caller", K.SCOPE), RowField("actual", K.VALUE_ID),
    RowField("callee", K.NAME), RowField("formal", K.VALUE_ID),
), tuple)                               # mode REVISE
SSA_SHAPE_MATERIALIZATION = declare_page("ssa_shape_materialization", (  # nearest 4
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)
SHAPE_SSA = declare_page("shape.ssa", (                                 # nearest 4
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)
KERNEL_INPUT_CONVERSION = declare_page("kernel_input_conversion", (     # nearest 5
    RowField("function", K.SCOPE), RowField("block", K.NAME),
    RowField("result", K.VALUE_ID), RowField("position", K.INDEX),
), tuple)                               # mode REVISE
TENSOR_REDUCTION_DOMAIN = declare_page("tensor_reduction_domain_concordance", (  # nearest 5
    RowField("function", K.SCOPE), RowField("result", K.VALUE_ID),
), object)
CALL_INPUT_STORAGE = declare_page("call_input_storage_concordance", (   # nearest 7
    RowField("caller", K.SCOPE), RowField("actual", K.VALUE_ID),
), str)
PRECISION_CHANNEL_SHAPE = declare_page("precision_channel_shape_concordance", (  # nearest 7
    RowField("function", K.SCOPE), RowField("value_id", K.VALUE_ID),
), tuple)
PRECISION_DECLARED_FORMAL_WIDTH = declare_page("precision_declared_formal_width_concordance", (
    RowField("function", K.SCOPE), RowField("formal", K.VALUE_ID),
), int)
SINGLE_INPUT_PHI_DESCRIPTOR = declare_page("single_input_phi_descriptor_concordance", (
    RowField("function", K.SCOPE), RowField("phi", K.VALUE_ID),
), tuple)
FORMAL_PARITY = declare_page("formal_parity", (                         # nearest 7
    RowField("function", K.SCOPE), RowField("formal", K.VALUE_ID),
    RowField("key", K.LABEL),
), tuple)
```

### Steps 1-3 remainder (DRAFT, for whichever lane reopens steps 2-3)

```python
SCOPE_REGISTRY = declare_page("scope_registry", (
    RowField("label", K.SCOPE), RowField("serial", K.INDEX),
), bool, private=True)                  # the book's own; mint_scope writes it
LEXICAL_READ_OCCURRENCE = declare_page("lexical_read_occurrence", (   # split 1 of lexical_read_binding
    RowField("read_scope", K.SCOPE), RowField("occurrence", K.VALUE_ID),
), str)
LEXICAL_READ_RETURN_ROOT = declare_page("lexical_read_return_root", (  # split 2
    RowField("read_scope", K.SCOPE), RowField("position", K.INDEX),
), str)
LEXICAL_READ_BINDING = declare_page("lexical_read_binding", (          # split 3 (the operand rows)
    RowField("read_scope", K.SCOPE), RowField("consumer", K.VALUE_ID),
    RowField("role", K.NAME), RowField("ordinal", K.INDEX),
), object)                              # str | None; mode CONCORD then REVISE by _set_operands
IDENTITY_TRANSITION = declare_page("identity_transition", (            # reducer rows only
    RowField("read_scope", K.SCOPE), RowField("node", K.VALUE_ID),
    RowField("role", K.NAME), RowField("ordinal", K.INDEX),
), tuple)                               # mode REVISE
READ_SCOPE_FORK = declare_page("read_scope_fork", (                    # split of identity_transition
    RowField("forked_scope", K.SCOPE),
), tuple)                               # mode CONCORD
# one shape for the fourteen reducer/source-stage pages: (scope, node) -> tuple
for _name in (
    "source_python_identity_concordance", "source_type_normalization_concordance",
    "source_precision_boundary_concordance", "source_precision_operator_concordance",
    "source_operator_dispatch_concordance", "source_receiver_constructor_concordance",
    "source_precision_pack_concordance", "source_numeric_component_concordance",
    "source_numeric_intrinsic_concordance",
    "source_numeric_operator_specialization_concordance",
    "source_sequence_mutation_concordance", "source_call_result_identity_concordance",
):
    declare_page(_name, (
        RowField("scope", K.SCOPE), RowField("node", K.VALUE_ID),
    ), tuple)                           # mode CONCORD (set col 0 + raise today)
SOURCE_NUMERIC_SPECIALIZATION = declare_page("source_numeric_specialization_concordance", (
    RowField("scope", K.SCOPE), RowField("digest", K.NAME),
), tuple)
SOURCE_FUNCTION_REACHABILITY = declare_page("source_function_reachability_concordance", (
    RowField("caller_scope", K.SCOPE), RowField("call", K.VALUE_ID),
    RowField("callee_scope", K.NAME),
), bool)
SOURCE_PARAMETER_IDENTITY = declare_page("source_parameter_identity_concordance", (
    RowField("scope", K.SCOPE), RowField("receiver", K.NAME),
), object)
SOURCE_NUMERIC_RECORD_ABI = declare_page("source_numeric_record_abi_concordance", (
    RowField("type_name", K.SCOPE), RowField("limbs", K.INDEX),
), tuple)
SOURCE_NUMERIC_TYPE_DEPENDENCY = declare_page("source_numeric_type_dependency_concordance", (
    RowField("type_name", K.SCOPE), RowField("limbs", K.INDEX),
), tuple)
SOURCE_NUMERIC_METHOD_DEPENDENCY = declare_page("source_numeric_method_dependency_concordance", (
    RowField("type_name", K.SCOPE), RowField("limbs", K.INDEX),
), tuple)
SOURCE_NUMERIC_PARAMETER_ABI = declare_page("source_numeric_parameter_abi_concordance", (
    RowField("definition", K.SCOPE), RowField("argument", K.NAME),
), tuple)
SOURCE_NUMERIC_PARAMETER_RECORD_VIEW = declare_page("source_numeric_parameter_record_view_concordance", (
    RowField("scope", K.SCOPE), RowField("parameter", K.NAME),
    RowField("node", K.VALUE_ID),
), tuple)
STATIC_RECORD_PIECE_MAPPING = declare_page("static_record_piece_mapping_concordance", (
    RowField("class_identity", K.SCOPE), RowField("field", K.NAME),
    RowField("key", K.LABEL),
), tuple)
STATIC_RECORD_PIECE_SEQUENCE = declare_page("static_record_piece_sequence_concordance", (
    RowField("class_identity", K.SCOPE), RowField("field", K.NAME),
), tuple)
```

## 11. Method notes and caveats

- The AST walk resolves `.page(<str literal>)`, `.page(<module constant>)`
  and page objects held in a variable inside one function. Pages held on
  `self` (the transformation ledger, the SSA tables, `_BookRows`) and pages
  written inside a nested function through an outer variable
  (`note_shape`, `carry`, `commit`, `record`) were attributed by reading
  the source, and cross-checked against census 20/30/40.
- "cells" comes from ONE run of the measurement copy; the numbers are the
  live book's, not the census tables'. A page "mentioned, never written" was
  created by a read (`latest`/`scope_rows`) in that case.
- Three pages hold more than one row shape (`lexical_read_binding`,
  `identity_transition`, `numeral_record_literal_concordance`,
  `numeral_leaf_width_concordance`, `exact_region_feed_dtype`); their
  declared forms above are splits, so the writer count in the plan is larger
  than the page count here.
- `GLOBAL_MONOTONIC_IDS.mint()` appears beside several concords
  (`program_abi_keyed_row_record`, numeral leaves, `plan_region_to_ssa_instrs`);
  those are `Novel` posts in the api's terms and the plan that owns the page
  declares the transform.

## 12. Continuation

This file is the handoff for the raw-only pages; the session that produced
it ends here. Nothing in the repo was changed except this file.

Complete:

- Every page name `IdentityBook.page(...)` is asked for by name anywhere in
  `src` (148) is in exactly one of sections 1-7 with writer function(s),
  observed row shape with proposed `RowFieldKind`s, fact shape, raw
  primitive, controller/energy/mapping cell counts, and readers.
- Cell counts reconcile: the seven group sums equal the 7625 cells the
  measurement reports on undeclared pages in the controller case.
- Section 8(a) lists the 12 pages whose rows carry a process id (one via
  `id(caller.G)`, ten via the `@control:<hex id>` scope string minted in
  `lower_control_sections_to_ssa`, one suspect) and the five pages that
  carry two or three row shapes under one name.
- Section 9 establishes that no page is written only by tests or probes.
- Section 10 holds a DRAFT `declare_page(...)` block per owner in the
  spelling of `concordance_declarations.py`, with the process-id elements
  replaced and the multi-shape pages split.

Not classified (left to the owning plans):

- Ownership of the 21 pages in section 6: design section 4 names no lane for
  tensor-call lowering (`tensor_ssa_lowering.py`), physical call-input
  adaptation (`ssa_call_input_adapters.py`), the precision passes
  (`ir_identities.py`), the self check, the sequence-contract helpers or the
  fcs shape/ABI settlement seam. The "nearest" column is a proposal only.
- Which of the record pages in sections 3 and 4 the step 6 and step 7 lanes
  actually want on their side of the line: the split here follows census 10
  (materialization vs forwarding/linking), not a decision.
- Fact types: every block uses `tuple`/`dict`/`object`/`str`/`int`; the
  plan that adopts a page decides whether the fact becomes a frozen
  dataclass (design section 6, decision 3).
- `cross_function_references`: whether its scope string is process-bound
  (suspected from its length and the `id(function)` keying in the writer)
  was not confirmed by reading the whole of
  `propagate_repository_ssa_call_metadata`.
- Modes: the "mode" column records the raw primitive used today; the
  `Mode.CONCORD` / `Mode.REVISE` comment in each DRAFT line is the obvious
  translation, not a verified choice (e.g. `aggregate_ledger` overwrites
  column 0 today, which is neither).

Exact next step for whoever adopts a DRAFT block:

1. Read the owning plan's section for the page (plan 60/70 style) or write
   it; do not paste the block until the plan names the stage, transform and
   reason vocabulary the writers will post with.
2. Copy the block into `src/compiler/concordance_declarations.py` under a new
   `# Step N` section, keeping the `declare_page(name, (RowField(...), ...),
   fact_type)` spelling; declaring a name already in `Registry.pages` with a
   different shape is refused at import, so run
   `python -c "import src.compiler.concordance_declarations"` first.
3. For each page in section 8(a), mint the scope through
   `IdentityBook.mint_scope` (or reuse an existing minted scope) BEFORE
   routing the writer through `post`; a row that still carries the
   `@control:<hex>` string or `id(caller.G)` must be refused by the plan,
   not admitted by `RowField.admits` (SCOPE admits any hashable).
4. For the split pages (`lexical_read_binding`, `identity_transition`,
   `numeral_record_literal_concordance`, `numeral_leaf_width_concordance`,
   `exact_region_feed_dtype`) route each writer to its own declared page and
   move every reader named in the "read by others" column with it.
5. Re-run the lead's `measure_completeness.py` for `mapping energy
   controller`; the "undeclared (raw-only)" count must drop by exactly the
   number of pages declared, and the page must leave the "largest unsourced
   pages" line once its writers post with `Derived`/`Novel`.
