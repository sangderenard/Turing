# Concordance census 30: SSA lowering writers

Read-only census, 2026-09-30, by `git grep` and `sed` only, of every
identity-book write in:

- `src/compiler/precompile_to_ssa.py`
- `src/compiler/tensor_ssa_lowering.py`
- `src/compiler/ir_identities.py`
- `src/compiler/ssa_call_input_adapters.py`
- `src/compiler/ssa_record_return_state.py`
- `src/compiler/ssa_self_check.py`
- `src/compiler/ssa_c_backend.py`, `ssa_llvm_backend.py`,
  `ssa_fortran_backend.py`, `ssa_storage_requirements.py` (zero book
  writes; listed for completeness in section 5)

No compiler ids or numberings appear here.  Code is named by function.

Relation to `CONCORDANCE_MASTER_LIST.md` (the master list): Part A3 of the
master list already documents the read/loop/binding pages
(`lexical_read_binding`, `loop_carried_binding`, `loop_result_port_binding`,
`consumer_operand`, `call_argument_operand`, `region_feed_consumer`,
`loop_entry_state`, `item_operand`, `identity_transition`) and their
writers; Part B lists 11 no-book and 8 mixed functions in
`precompile_to_ssa.py`, 3+2 in `tensor_ssa_lowering.py`, 2+1 in
`ir_identities.py`, 3+2 in `ssa_call_input_adapters.py`, 4+4 in
`ssa_record_return_state.py`, and the backend functions.  This file does
not repeat those; it adds what the master list lacks: per-write-site EDGE?
classification, fallback-as-fact, minted ids without a transform edge, and
the builder-private state and metadata channels that decide identity.
Overlaps are cited as "master A3", "master B1", etc.  Section 6 lists the
places where this reading contradicts the master list.

Columns:

- **EDGE?** YES = the fact names the source identity AND the stage or
  transform that produced it (the row can be walked backwards).  PARTIAL =
  names a source identity or a stage, not both, or names the source only
  through the row key.  NO = a label, a count, or a measurement with no
  source recorded.
- **fallback?** YES when a value chosen from absence of information is
  recorded as a fact (or silently substituted with nothing recorded).
- **mode** is the page primitive used: `concord` (first writer owns, later
  disagreement raises), `revise` (append revision), `set` (raw column
  write; "set+raise" = a hand-rolled concord), `bind_alias`.

## 1. Census by page

### 1.1 `precompile_to_ssa.py` (22 write sites, 6 YES / 15 PARTIAL / 1 NO)

| Page | Enclosing function | Row key (words) | Fact | Mode | EDGE? | fallback? | Readers |
|---|---|---|---|---|---|---|---|
| `sequence_contract_concordance` | `_ControlSSABuilder.__init__` via `commit_sequence_contract` | (function, sequence id) | (policy, column count, writable, source stage string) | set-append (helper) | PARTIAL: stage named ("SSA sequence declaration"), no source row | no | `committed_sequence_contract`, `committed_sequence_row_layout` (identity_concordance), tests |
| `tensor_shape_concordance` | `_ControlSSABuilder._emit_table_lookup` | (control scope, authored result id) | contract dict: storage span, rank, dynamic state, the minted shape/rank/count cell ids, keyed handle id | set, next page-wide column | PARTIAL: names the minted descriptor cells (the transform's outputs) but not the lookup instruction or the sequence it came from | no | `lower_loop.concorded_resident` (read), `tensor_ssa_lowering` |
| `tensor_shape_concordance` | `lower_control_sections_to_ssa` (IndexedStore publish, after `final_resident`) | (control scope, each spelling: base, result, resident, every alias resolving to resident) | contract dict with `"source": "planned-region-indexed-store"`, rank = max over base/result/accounting | set, next page-wide column; incumbent merged with max rank, no disagreement raised | PARTIAL: stage named, source instruction not | PARTIAL: `tensor_metadata_state` "unknown" is honest; the silent max-rank merge over an incumbent is not | `lower_loop.concorded_resident`, `tensor_ssa_lowering` |
| `callsite_argument` | `_ControlSSABuilder._note_callsite_arguments` | (function, callsite, position) | (graph id, resolved value id, inside-loop flag) | set, column = history length; whole body in try/except pass | PARTIAL: both ends named, resolution path (`_callsite_argument` -> `_read`) not | no (but a swallowed exception loses the row silently) | `tools/compiler_probes` only |
| `loop_carried_entry` | `_ControlSSABuilder._enter_loop_state` | (control scope, loop node, binding) | carried entry index | concord | PARTIAL: derived from a `loop_carried_binding` row it does not name | no | `_carried_entry_value` |
| `loop_entry_state` | `_ControlSSABuilder._enter_loop_state` | (control scope, loop node, initial id) | (sorted bindings, initial id) | concord | PARTIAL: same source row unnamed | no | `_resolve_read` (master A3) |
| `identity_transition` | `_ControlSSABuilder._region_feed` (rank-0 item merge) | (control scope, item id) | ("merge", resolved source id, "scalar_item") | revise | YES | no | reducer `_set_operands` (writer of "move"/"fork"), audit `_planning_alias_transition_findings` (master A3c) |
| `loop_scope_inner_transition` | `_ControlSSABuilder._complete_loop_latch_carried` via `rebind_loop_scope_inner` | (function, loop node, declared inner id) | (resident id, reason) | set-append (helper) | YES | no | `loop_scope_declarations`, `concord_loop_scope_latch_residents`, probes.  Shadow twin: `completed.accounting["loop_scope_inner_transition"]` |
| `loop_scope` | `_ControlSSABuilder.lower_loop` via `declare_loop_scope` | (function, loop node, outer id or (outer id, bindings)) | columns OUTER / CARRIED / INNER / ("graph", outer, inner) / ("bindings", ...) | set, fixed columns; call wrapped in try/except pass | YES (the three generations of one quantity are the edge; bindings carried) | no, but a failed declaration vanishes silently | `_loop_scope_findings`, `loop_scope_declarations`, tests |
| `loop_scope` | `_ControlSSABuilder.lower_while` via `declare_loop_scope` | same | same | same | YES | same | same |
| `while_carried_test` | `_ControlSSABuilder.lower_while` | loop state key (control scope, loop node) | sorted binding names, OR the predicate value id when the predicate has no read expression | concord | PARTIAL: bindings derived from `lexical_read_binding` rows not named | YES: the "else" branch records a bare value id as the test identity | none outside (decision recorded, not re-read) |
| `loop_result_reconciliation` | `_canonicalize_non_dominating_loop_result_uses._note` | (function, argument id) | (outcome, block#index, op, detail) | set, column = history length; try/except pass | PARTIAL: names the candidate/argument in detail, not the ledger row it read (`carried_port_values`) | no | probes only |
| `region_feed_consumer` | `_concord_region_feed_consumers` | (control scope, region, feed id) | consumer result ids | concord | PARTIAL (structure; master A3) | no | `_region_feed` |
| `region_capture_binding` | `_split_region_captures_by_binding` (original capture) | (control scope, region, captured value id) | (source value id, first binding) | concord | YES | no | `_region_feed` |
| `region_capture_binding` | `_split_region_captures_by_binding` (per extra binding) | (control scope, region, minted formal id) | (source value id, binding) | concord | YES: mint + edge, the model for a minted id | no | `_region_feed` |
| `control_uniform_dtype` | `lower_control_sections_to_ssa` | (control scope, uniform value id) | dtype string | concord | PARTIAL: a source declaration (control.uniforms), stage unnamed | no | region formal typing in the same function |
| `region_value_dtype` | `lower_control_sections_to_ssa.region_argument` (per-region result loop) | (control scope, result id) | dtype string | revise whenever different | NO: no producer named; a later region silently retypes | no | `typed_region_value` |
| `control_value_concordance` | `lower_control_sections_to_ssa` (np.asarray view binding, on `value_concordance`) | (control name, alias id) | resident root | bind_alias | PARTIAL: alias -> resident is an edge, cause/stage absent | no | `resident_sequence` (tensor_ssa_lowering), `resolved_concordant_alias_bindings`, fortran_c_shell |
| `control_value_concordance` | `lower_control_sections_to_ssa` (control.value_aliases components) | (control name, alias id) | resident arena id | bind_alias when no incumbent | PARTIAL | no | same |
| `control_value_concordance` | `lower_control_sections_to_ssa` (publish resolved roots back to `value_concordance`) | (control name, alias id) | resident | bind_alias | PARTIAL | no | same |
| `control_value_concordance` | `lower_control_sections_to_ssa.bind_resident` | (control name, alias id) | resident root | bind_alias | PARTIAL | no | same |
| `field_slot_storage_concordance` | `lower_control_sections_to_ssa` (record table build) | (function, field name or slot) | ("nested_record", identity) | concord | PARTIAL | no | none outside |

Reads only in this file (not writes): `lexical_read_binding`
(`_operand_bindings`, `_split_region_captures_by_binding`),
`loop_carried_binding` (`_enter_loop_state`, `_loop_scope_rebinds`,
`_publish_loop_result_ports`, `_split_region_captures_by_binding`),
`call_argument_operand` (`_callsite_argument`), `loop_result_port_binding`
(`_break_bound_initial`, `_publish_loop_result_ports.carried_entry_of`),
`consumer_operand` and `item_operand` (`_region_feed`; planned-region
dependency signatures), `tensor_shape_concordance`
(`lower_loop.concorded_resident`).

Read-side fallbacks in this file (nothing recorded; master rule "a missing
row raises" not honoured):

- `_publish_loop_result_ports.carried_entry_of`: with no read scope,
  matches (updated id, initial id) pairs by id.
- `_region_feed`: a consumer with no `consumer_operand` row is retried
  through `_aliases_resolving_to` (the private `value_aliases` chain), then
  falls to `(consumer, None, 0)`; `_resolve_read` raises on the None
  binding only if the loop carries some of the other bindings.
- `lower_loop.concorded_resident`: when no `tensor_shape_concordance` row
  exists and Meta has a shape, a contract dict is synthesized locally from
  Meta ("static") and used as if concorded; not written.

### 1.2 `tensor_ssa_lowering.py` (18 write sites, 8 YES / 2 PARTIAL / 8 NO)

| Page | Enclosing function | Row key | Fact | Mode | EDGE? | fallback? | Readers |
|---|---|---|---|---|---|---|---|
| `ssa_call_shape` | `_publish_exact_ssa_call_shapes` | (callee, formal id, caller, actual id) | (extents, dtype) | set when changed, column = history length | YES via row key (actual -> formal); stage implicit | no | `_settle_exact_ssa_call_shapes`, probes |
| `ssa_call_shape_evidence` | same | same | evidence kind (native result / explicit view / resident descriptor / ssa occurrence) | set when changed | PARTIAL | no | probes |
| `proven_shape` | `_publish_exact_ssa_call_shapes` via `record_proven_shape` | (authored callee, formal id) | ("proven", extents, dtype) at level 0 | helper set | NO: the actual it was derived from is known here and discarded by the helper | no | `proven_shape_of`, `glsl_deployment_strategy`, `site_bundle`, 11 probes |
| `proven_shape` | `settle_shape_preserving_value_metadata` via `record_proven_shape` (level None, "unanimous") | (function, result id) | same | helper set; try/except pass | NO: sources known, discarded | no | same |
| `proven_shape` | `lower_tensor_calls_to_repository_ssa.call` (cast branch) | (function, result id) | same | try/except pass | NO | no | same |
| `proven_shape` | `lower_tensor_calls_to_repository_ssa.call` (shape-preserving branch) | (function, result id) | same | try/except pass | NO | no | same |
| `shape.ssa` | `_record_ssa_shape` (called twice from the binary-operand path of `call`) | (mangled function name, value id) | extents | set column 0 (overwrites the same cell); try/except pass | NO | YES: scope name recovered by splitting on `__specialized_` / `__planned_region` / last `__` | probes only |
| `tensor_shape_enrichment` | `propagate_repository_ssa_call_metadata.enrich._log` | (function, value id, field) | (old, new, authoritative flag, source id) with column (round, step) | set | YES | no | oscillation detection in the same fixed point (`oscillating_rows`), probes |
| `tensor_shape_enrichment` | `propagate_repository_ssa_call_metadata` (descriptor.shape of a callee formal) | (callee, formal id, "descriptor.shape") | same shape | set | YES | no | same |
| `tensor_shape_enrichment` | same (semantic actual -> actual shape) | (function, actual id, "shape") | same shape | set | YES | no | same |
| `tensor_shape_enrichment` | same (semantic actual -> actual dtype) | (function, actual id, "dtype") | same shape | set | YES | no | same |
| `shape_transformation_concordance` (+ `_dependents`, `_state`) | `enrich` via `record_shape_transformation` | (target scope, target id, source scope, source id, stage, "ssa_metadata_transport", role) | (source state, target state) | helper (set-append edge, revise state) | YES (the model; census 00) | no | `concordant_shape_transformation_state`, `withdraw_superseded_shape_derivations` |
| `cross_function_references` | `propagate_repository_ssa_call_metadata` | (scope, value id) | owner names tuple | set column 0 | PARTIAL (structure) | no | none outside |
| `scalar_kernel_operand_concordance` | `call` (cast) | (function, result id) | "scalar_cast" | concord | NO | YES: decided from empty shape + zero rank + "not unknown", i.e. absence of shape information | none outside |
| `scalar_kernel_operand_concordance` | `call` (broadcast/expand) | (function, result id) | "scalar_fill" | concord | NO | YES (same test) | none outside |
| `scalar_kernel_operand_concordance` | `call` (binary kernel) | (function, result id) | ("scalar_operand", position) | concord | NO (position, not identity) | PARTIAL | none outside |
| `tensor_reduction_domain_concordance` | `call` (reduction) | (function, result id) | (source id, op, source shape, count) | set+raise | YES | no | audit `_tensor_reduction_domain_findings`, tests |
| `kernel_input_conversion` | `call` (broadcast source conversion) | (function, block, result id, operand position) | (operand id, broadcast source id, dtypes...) | set-append | YES | no | tests; shadow twin `function.metadata["kernel_input_conversions"]` |

Reads only: `linked_value_abi_polymorphism`, `value_shape`
(`_shape_polymorphic_function`, by authored-name match over all rows),
`control_value_concordance` (`call.resident_sequence`).

### 1.3 `ir_identities.py` (3 write sites, 1 YES / 0 PARTIAL / 2 NO)

| Page | Enclosing function | Row key | Fact | Mode | EDGE? | fallback? | Readers |
|---|---|---|---|---|---|---|---|
| `precision_channel_shape_concordance` | `carry_precision_through_ssa.carry` | (function, value id) | (logical shape, channel shape, limbs) | set+raise column 0 | NO: limbs come from `value.accounting["precision_limbs"]`, no source row | no | `lower_precision_operations` (read), audit, tests |
| `precision_declared_formal_width_concordance` | `lower_precision_operations` | (function, formal id) | limbs | concord | NO (from accounting) | no | none outside |
| `single_input_phi_descriptor_concordance` | `settle_single_input_phi_descriptors` | (function, phi result id) | (source id, dtype, shape) | set+raise column 0 | YES | no | audit, tests |

Reads only: `source_precision_boundary_concordance`
(`lower_precision_operations`, scans all rows by function/qualified-name
match).

### 1.4 `ssa_call_input_adapters.py` (5 write sites, 2 YES / 1 PARTIAL / 2 NO)

| Page | Enclosing function | Row key | Fact | Mode | EDGE? | fallback? | Readers |
|---|---|---|---|---|---|---|---|
| `exact_region_feed_dtype` | `_concord_exact_region_feed_dtypes` | ("formal", callee, formal id) and the feed-side row | (exact dtype,) | set+raise column 0 | PARTIAL: feed and formal are separate rows, the pairing is not a fact | no | audit, tests |
| `kernel_input_conversion` | `_adapt_physical_call_inputs_round` (broadcast_double scalar) | (caller, block, consumer id, 0) | (source id, converted id, dtypes, callee) | set-append | YES: mint + edge | no | tests; shadow twin `caller.metadata["kernel_input_conversions"]` |
| `call_input_storage_concordance` | `_adapt_physical_call_inputs_round` | (caller, actual id) | "sequence_arena_by_reference" | concord | NO (from accounting flag) | no | none outside |
| `call_input_conversion` | `_adapt_physical_call_inputs_round` | (caller, actual id, callee, formal id) | (converted id, source dtype, target dtype, kind) | set-append | YES: mint + edge | no | tests, probes; shadow twin `caller.metadata["call_input_conversions"]` |
| `physical_call_input_adaptation_fixed_point` | `adapt_physical_call_inputs` | ("whole-program", function names) | changes per round | set, column = round | NO (a count) | no | none |

### 1.5 `ssa_record_return_state.py` (1 write site, YES)

| Page | Enclosing function | Row key | Fact | Mode | EDGE? | fallback? | Readers |
|---|---|---|---|---|---|---|---|
| `record_phi_temporal_fallback_concordance` | `repair_non_dominating_record_phi_uses` | (function, result id, block, use index, position) | (fallback id, target block, "initial_record_field_version") | set+raise | YES for the substitution; the fallback id itself comes from a Phi attribute (`initial_value_id`), a shadow channel with no row | the substitution IS a fallback but is labelled as such | none outside |

The rest of this module's identity decisions are off-book; see section 3.

### 1.6 `ssa_self_check.py` (1 write site, NO)

| Page | Enclosing function | Row key | Fact | Mode | EDGE? | fallback? | Readers |
|---|---|---|---|---|---|---|---|
| `formal_parity` | `check_formal_parity` | (function, formal id, "unaccounted_formal") | channel fill counts + accounting keys | set column 0; try/except pass | NO (diagnostic) | no | none |

## 2. Minted ids

Every `GLOBAL_MONOTONIC_IDS.mint()` in the slice (21 sites), and whether a
book row records the transform that produced the new value.

| File | Site (function) | What is minted | Book edge? | Shadow record |
|---|---|---|---|---|
| precompile_to_ssa | `_ControlSSABuilder.fresh_value` | every builder temporary (168 call sites: phis, casts, addresses, loads, sequence length cells, descriptor cells, carried ports) | NO: the SSA instruction is the only edge | none |
| precompile_to_ssa | `_ControlSSABuilder.__init__` (joined flat sequence view) | flat sequence id for a joined sequence | NO: source sequence -> flat id lives in private `joined_flat_sequence_ids`; the descriptor reaches `SSASequenceTable` only at `finish` | none |
| precompile_to_ssa | `_lower_sequence_mutation_body` (`indexed_load` of a record row column) | projected row-column value | NO: `source_sequence_id`/`column_index` only as instruction attributes | instruction attributes |
| precompile_to_ssa | `_split_region_captures_by_binding` | region formal per extra binding | YES: `region_capture_binding` (source id, binding) | none needed |
| precompile_to_ssa | `lower_class_navigation_to_ssa.Builder.__init__` / `.emit` | LUT helper formals and results | NO | none |
| precompile_to_ssa | `_inject_field_slot_access.fresh` | Const / address ids | NO | none |
| tensor_ssa_lowering | `lower_tensor_calls_to_repository_ssa.fresh` (28 call sites) | kernel call temporaries, constants, int vectors | NO except where `kernel_input_conversion` or `record_proven_shape` (no source) follows | none |
| ir_identities | pow-rewrite `fresh` | mul/reciprocal/sqrt temporaries | NO | none |
| ir_identities | `reduce_precision_operations.fresh(like)` | per-limb temporaries (copies `like.accounting`) | NO | copied accounting |
| ir_identities | `contract_multiply_add_to_fma` (negated addend) | Neg result | NO | `lowered_from` attribute |
| ir_identities | `lower_precision_operations` (extra formal limbs) | limb formals | NO | accounting `precision_formal_source_id`, `precision_limb_index` |
| ir_identities | `lower_precision_operations.fresh` (extent-shaped value) | expansion value | NO | accounting `precision_extent_source_id` |
| ir_identities | `lower_precision_operations...scalar_constant` | Const | NO | none |
| ir_identities | `lower_precision_operations...constant` (int) | Const | NO | none |
| ssa_call_input_adapters | `_adapt_physical_call_inputs_round` (broadcast_double) | converted scalar | YES: `kernel_input_conversion` | accounting `kernel_input_conversion` |
| ssa_call_input_adapters | `_adapt_physical_call_inputs_round` (call input) | converted actual | YES: `call_input_conversion` | accounting `call_input_conversion` |
| ssa_call_input_adapters | `_adapt_physical_call_inputs_round` (call output) | `temporary` result re-typed to backend dtype | NO: accounting `call_output_conversion` only; `output_ids` attribute rewritten | accounting |
| ssa_record_return_state | `freshen_redefined_ssa_objects` | fresh id for a redefined SSA object (identity SPLIT) | NO: accounting `source_value_id`; `module.metadata["redefined_ssa_object_receipts"]` | accounting + module metadata |
| ssa_record_return_state | return-edge `materialize` (clone of a producer chain) | recomputed value (identity DUPLICATE) | NO: accounting `return_edge_recomputed_from`; `function.metadata["return_edge_recomputations"]` | accounting + metadata |
| ssa_record_return_state | `publish_scalar_record_return_fields` (Cast at a return edge) | converted field value | NO: attribute `source_field_value_id` only | instruction attribute |

Not mints: the `first_value_id=GLOBAL_MONOTONIC_IDS.peek()` arguments
passed to the sequence helper lowerings.  `ir_sequence_tables._Builder`
ignores `first_value_id` and mints through the shared counter itself, and
`_register_sequence_lowering` only collects the helper's ids; the helper's
minted values have no book edge either (out of this slice).

Tally: 21 mint sites, 3 with an edge on the book, 18 without.  Two of the
18 (`freshen_redefined_ssa_objects`, return-edge `materialize`) split or
duplicate an existing identity, which is exactly the case the target API's
clause (a) exists for.

## 3. Shadow ledgers

### 3.1 `function.metadata` / `module.metadata` channels written in the slice

| Channel | Fact held | Writer | Readers (files) | Book counterpart? |
|---|---|---|---|---|
| `parameter_names` | (name, formal id) pairs = the ABI naming of formals | `_ControlSSABuilder.finish` (from `value_name_histories` + `external_values` + `declared_parameter_only_ids` + use scan); `lower_fused_program_to_ssa` | 18 files: fortran_c_shell, glsl, ir_identities, kernel_bank, native_package, c/fortran/javascript backends, self_check, ... | NONE |
| `value_names` | (name, latest value id) per authored name | `finish` (same derivation) | fortran_c_shell, kernel_bank, javascript backend, reference evaluator, native runtime | NONE |
| `named_outputs` | (name, merged return value id) | `finish` (after return merges); `lower_fused_program_to_ssa` | 17 files incl. ir_identities (4 reads), fortran backend, storage_requirements, autograd, output_publication | NONE |
| `value_aliases` | snapshot of the builder's live alias map | `finish` | fortran_c_shell, glsl, ir_identities, identity_concordance | `control_value_concordance` holds the planning aliases; the loop-time rewrites (below) are not on it |
| `carried_port_values` | port id -> carried Phi VALUE standing at the port | `_publish_loop_result_ports` into `_carried_port_values`, snapshotted in `finish` | fortran_c_shell, ir_identities (2), python materializer, self_check, `_canonicalize_non_dominating_loop_result_uses` | `loop_result_port_binding` is port -> binding; port -> phi value has no page |
| `recursion_table` | recursion bookkeeping | `finish` (dict of `ssa_recursion_table`), module merges | fortran_c_shell, glsl, loop_composer, shell_reference_tables, webgpu | NONE |
| `validation_contracts` | predicate id, error code, expression | `lower_control_block` (ValidationBlock) -> `finish` | fortran backend (module report), project_compilation_product | NONE |
| `control_identity_receipts` | (source graph id, ssa id, "scalar_item_identity") = a MERGE decided in `lower_control_expression` | `lower_control_expression` -> `finish` | fortran_c_shell | `identity_transition` "merge" exists for the `_region_feed` path only; this path is off-book |
| `table_lookup_ownership_receipts` | (result id, ownership kind, owner ids) | `lower_control_block` (lookup) -> `finish` | none outside | NONE |
| `loop_result_use_rebindings` | receipts of argument -> port-value substitutions | `finish` and `lower_control_sections_to_ssa` via `_canonicalize_non_dominating_loop_result_uses` | fortran_c_shell | `loop_result_reconciliation` records outcomes (PARTIAL) |
| `authored_constant_values` | authored literal values | `_materialize_control_constants` | fortran_c_shell | NONE |
| `sequence_table` (SSASequenceTable), `sequence_static_capacity_bounds`, `sequence_helper_functions`, `sequence_array_argument_ids` | descriptors and helpers | `finish` from private `sequence_descriptors`, `static_sequence_capacities`, `sequence_helper_functions` | many | master A1 says `SSASequenceTable` is BOOK; during lowering the authority is the private dict (see section 6) |
| `source_output_value_ids`, `source_region_integral` | callee output identity / region marker | `tensor_ssa_lowering` | ir_identities (4), adapters | NONE |
| `call_metadata_fixed_point_*` (module) | rounds, receipts, oscillation | `propagate_repository_ssa_call_metadata` | none | derived FROM `tensor_shape_enrichment` (book is source; fine) |
| `kernel_input_conversions`, `call_input_conversions` | duplicates of the two conversion pages | tensor_ssa_lowering, adapters | tests | twins of `kernel_input_conversion` / `call_input_conversion` |
| `precision_lowered_values`, `parameter_member_formals`, `discarded_pure_call_sites`, `discarded_outputless_source_regions`, `host_linear_region_inlining` | precision lowering identity (which formals are members of which parameter) | ir_identities | fortran_c_shell and later precision passes | NONE (`parameter_member_formals` decides formal identity) |
| `identity_book` (module) | a detached book for a standalone precision transaction | `apply_numeric_feature_pipeline` | `identity_book(module)` | is the book itself, parked in metadata |
| `record_return_layouts` | layout of scalar record returns | `publish_inout_scalar_return_snapshots` | fortran_c_shell, project_compilation_product | NONE |
| `record_return_state_receipts` | (span, slot ids, (receiver, field, value) states) copied from the source graph | `scalar_return_field_versions` (as a side effect of being called) | `publish_scalar_record_return_fields` rebuilds a fake source graph from it and calls the lookup again | NONE; see 3.3 |
| `return_edge_recomputations`, `retained_loop_update_receipts`; module `redefined_ssa_object_receipts`, `inout_scalar_return_snapshot_receipts`, `identity_cast_result_receipts`, `forwarded_record_result_receipts`, `conditional_phi_continuation_receipts` | receipts of identity splits, casts, forwards | ssa_record_return_state | none (receipts) | NONE |

### 3.2 `_ControlSSABuilder` private state that decides identity

| State | Fact held | Writers | Readers | Book counterpart? |
|---|---|---|---|---|
| `external_values` (graph id -> current SSAValue) | the lowering environment: which SSA value a graph identity denotes NOW | 54 assignment sites (`external_value` itself, `_complete_loop_latch_carried`, `lower_control_expression`, loop entry/latch, field effects, sequence lengths, ...) | every read not routed through `_resolve_read`; `finish` naming | none (master A6.2 names this as the line to move) |
| `value_aliases` / `concorded_value_aliases` | alias -> source; the live map is rewritten per loop body by `_publish_loop_result_ports` and undone by `_restore_loop_result_port_aliases` | constructor (from `value_concordance` resolution), the two loop-time writers | `external_value` (follows chains), `_aliases_resolving_to`, `finish` snapshot | `control_value_concordance` holds the planning aliases; the loop-time spellings -> updated id rewrites are never on the book |
| `value_name_histories` (name -> authored id history) | reducer's identity table, passed in | constructor | parameter seeding in `__init__`, `finish` naming (`value_names`, `parameter_names`) | reducer pages record reads per position; the name -> versions history has no page in this slice |
| `declared_parameter_only_ids` | formals kept only because a plan call needs them | `__init__` add; `external_value` discard on first use | `finish` naming | NONE |
| `region_signatures` (region -> (feeds, outputs)) | which ids a numerical region consumes/publishes | constructor | `emit_region_call`, `lower_loop`, `lower_while`, formal reservation | `region_feed_consumer` covers consumers, not signatures |
| `loop_frames`, `enclosing_loop_states` | the builder's position and each loop's carried tuples (updated id, initial id, initial, updated, current) | `_enter_loop_state`, latch completion | `_resolve_read`, `_carried_entry_value` | `loop_entry_state` records bindings + initial id; the VALUES live here only |
| `sequence_descriptors`, `sequence_storage_values`, `sequence_length_values`, `joined_flat_sequence_ids`, `sequence_record_identities` | sequence identity and its physical cells | `_sequence_descriptor`, `__init__` | everywhere sequences are lowered | `SSASequenceTable` at `finish` only |
| `scalar_field_effect_destinations` | SetAttr version id -> (field value id, dtype) | constructor | `external_value` (pre-store version denotes the incumbent field) | NONE |
| `control_identity_receipts`, `table_lookup_ownership_receipts`, `validation_contracts`, `_carried_port_values`, `_carried_port_groups`, `ssa_recursion_table` | see 3.1 | see 3.1 | `finish` | see 3.1 |

### 3.3 `ssa_record_return_state.scalar_return_field_versions` and the source-graph ledgers

What the ledgers are: the reducer (`topological_reducer`, outside this
slice) writes on the source graph, keyed by the return statement's source
span, `return_slot_values` (the slot value ids returned at that site) and
`return_record_field_states` ((receiver, field, value) triples: the value
each record field held at that return).  A later reducer pass renumbers
both into SSA definition numbering.  Neither is on the book.

What `scalar_return_field_versions(function, source_graph, functions)`
reads: `return_record_field_states` (returns a constant `fallback` lambda
if empty), `return_slot_values`, the function's blocks, each block's
terminal `Br`/`CondBr` targets (to build a CFG and immediate dominators),
each predecessor terminal's `return_source_value_ids` attribute, every
definition's op and `binding` attribute, formal ids, `accounting` flags
`program_abi_field_written` and `ssa_storage_alias`, and callee bodies
(recursively, `argument_readonly`).

Side effect on entry: writes `function.metadata["record_return_state_receipts"]`
(a copy of both ledgers) before deciding anything.

What the returned `lookup(receiver, field, predecessor, fallback,
alias_receivers)` decides: whether to REPLACE the record field's SSA value
at one return edge (the `fallback`, which is the physical field's current
value) with a different value: the one the source ledger says the field
held at that return site, provided it is a `Const` or a
`conditional_carried` Phi with one definition, not a formal, rank-0, of
the same dtype (or a boolean Phi tree), whose definition block dominates
the predecessor, with no intervening store, alias write, or non-read-only
call between definition and return edge (a tail-CFG walk with the
definition's in-edges removed).  Twelve distinct conditions return
`fallback` silently.

What it records: NOTHING on the book.  Neither the substitution nor any
of the twelve abstentions is written anywhere.

Callers and what happens next:

- `fortran_c_shell` (the record-return Phi rebuild): `select_return_arguments`
  calls the lookup per predecessor; if the selected value's dtype differs
  it MINTS a `Cast` (accounting-free, attributes `record_return_field_conversion`,
  `source_field_value_id`) and inserts it before the edge terminal.  The
  result list is then passed to `_concord_record_return_phi_inputs`, which
  writes `record_return_phi_input_concordance` (function, phi result,
  field, position, predecessor) -> (candidate id, chosen id, reason).  The
  reason for a candidate that is not the Phi's own result is
  "incoming_record_field" -- so a candidate that was in fact a
  ledger-selected conditional Phi or a freshly minted Cast is recorded on
  the book with a provenance label that says it came from the incoming
  record.  On 2026-09-30, in the native dt-system lowering, the concord
  raised: a later linking round proposed a different chosen id for a row
  whose prior fact was the substituted one.  The raise was correct; the
  row it protected was a fallback-as-fact.
- `publish_scalar_record_return_fields` (this module): rebuilds a
  throwaway graph from `record_return_state_receipts`, calls the lookup
  again, mints its own `Cast` the same way, rewrites the Phi's args and
  its `initial_value_id` attribute.  No book row.

Verdict for this ledger: the value chosen (a conditional Phi or Cast in
place of a record field's physical value) is an identity decision made
from a graph-side ledger with no book row, recorded only as a metadata
copy of its own input, and then mislabelled on the one page that does see
it.  Under the target API this write is admissible only as clause (b)
naming the `return_record_field_states` row, the dominance proof, and the
stage; every `return fallback` exit must become an UNRESOLVED row.

## 4. Fallback-as-fact sites (all files)

| Site | What is chosen from absence | Recorded as |
|---|---|---|
| `lower_while` -> `while_carried_test` | a bare predicate value id when the predicate has no read expression | a concorded fact |
| `call` -> `scalar_kernel_operand_concordance` ("scalar_cast", "scalar_fill") | "this operand is a scalar" from empty shape + zero rank | a concorded label |
| `_record_ssa_shape` -> `shape.ssa` | the authored scope recovered by string-splitting the function name | the row key |
| `lower_control_sections_to_ssa` IndexedStore publish -> `tensor_shape_concordance` | rank = max(base, result, accountings, incumbent) | a new column, no disagreement |
| `region_value_dtype` revise | the dtype of whichever region wrote last | latest revision |
| `scalar_return_field_versions` (12 exits) | keep the physical field value | nothing; then labelled "incoming_record_field" by `_concord_record_return_phi_inputs` |
| `repair_non_dominating_record_phi_uses` | the Phi's `initial_value_id` attribute | a YES row (honestly labelled a fallback) |
| Silent non-writes wrapped in `try/except Exception: pass`: `declare_loop_scope` (x2), `_note_callsite_arguments`, `_note`, `_record_ssa_shape`, three `record_proven_shape` calls, `formal_parity` | a failed write | nothing; the absence later reads as "no row" |

## 5. Backends and storage requirements

`ssa_c_backend.py`, `ssa_llvm_backend.py`, `ssa_fortran_backend.py`,
`ssa_storage_requirements.py`: zero book reads or writes (the LLVM backend
mentions `GLOBAL_MONOTONIC_IDS` in a comment only).  They decide from the
shadow channels: `parameter_names` (c, fortran), `named_outputs` (fortran,
storage_requirements), `validation_contracts` (fortran writes a module
report from it), `linked_call_frame_storage` and `program_abi_storage`
accounting (master B1/B2 rows).  Whatever the book says about a formal's
name or an output's identity never reaches emission; only the `finish`
snapshot does.

## 6. Where this census differs from the master list

1. Master A3 gives `loop_entry_state`'s fact as "(carried bindings,
   pre-loop value)".  The code records (bindings, initial id): the authored
   state identity, not the SSA value (the comment in `_enter_loop_state`
   explains why).  The pre-loop VALUE lives only in `loop_frames`.
2. Master A1 marks `SSASequenceTable` BOOK.  Inside the builder the
   authority is the private `sequence_descriptors` dict plus
   `joined_flat_sequence_ids`, `sequence_storage_values`,
   `sequence_length_values`; the table is built from it at `finish`.
   Status during lowering: PRIVATE, then BOOK at the snapshot.  The joined
   flat view's relation to its source sequence never reaches the book.
3. Master B2 counts `ssa_record_return_state.py` as 4 no-book + 4 mixed
   and lists none of its functions as book writers.
   `repair_non_dominating_record_phi_uses` writes
   `record_phi_temporal_fallback_concordance`; and
   `scalar_return_field_versions`, the module's identity-substituting
   engine (also driving `fortran_c_shell`), is absent from Part B.
4. Master working rule "a missing row raises; no value-id fallback where
   the book is expected" is not honoured by
   `_publish_loop_result_ports.carried_entry_of` (id-pair match when no
   read scope), `_region_feed` (alias chain retry, then a None binding),
   or `scalar_return_field_versions` (twelve silent exits).
5. Master A3c says `identity_transition` "merge" is recorded for the
   item merge.  True for the `_region_feed` path only; the same merge in
   `lower_control_expression` is recorded as a `control_identity_receipts`
   metadata tuple and nothing on the book.
6. Master B1 lists `_ControlSSABuilder.finish` as mixed on
   `value_aliases`/`parameter_names`.  Agreed, and stronger: `finish` is
   where every naming channel (`parameter_names`, `value_names`,
   `named_outputs`, `carried_port_values`, `value_aliases`) is born, from
   private state, with no page, and it is the only source the backends
   read.

## 7. Verdict

Counts: 50 book write sites in the slice (precompile 22, tensor lowering
18, ir_identities 3, adapters 5, record return state 1, self check 1;
backends 0).  EDGE: 18 YES, 18 PARTIAL, 14 NO.  Fallback-as-fact: 7 sites
plus 9 silent non-writes.  Minted ids: 21 sites, 3 with an edge.  Shadow
channels deciding identity: 12 metadata channels, 9 builder-private
structures, 2 source-graph ledgers.

Ten most load-bearing NO sites (widest readers, identity-deciding):

1. `_ControlSSABuilder.finish` naming snapshot: `parameter_names`,
   `value_names`, `named_outputs`, `carried_port_values`, `value_aliases`.
   No page; 18 reader files; the backends read nothing else.
2. `scalar_return_field_versions` + `select_return_arguments` /
   `publish_scalar_record_return_fields`: field-version substitution and
   a minted Cast, nothing recorded, then mislabelled by
   `_concord_record_return_phi_inputs` (the 2026-09-30 raise).
3. `external_values` (54 writers): the lowering environment itself.
4. Loop-time `value_aliases` rewrites (`_publish_loop_result_ports`,
   `_restore_loop_result_port_aliases`) that `external_value` follows.
5. `record_proven_shape` x4 in `tensor_ssa_lowering`: the source is in
   hand at every call and discarded; 13 readers.
6. `control_identity_receipts` "scalar_item_identity" merge in
   `lower_control_expression`: a merge with no `identity_transition` row.
7. `freshen_redefined_ssa_objects` and return-edge `materialize`:
   identity split / duplicate with a fresh id and accounting only.
8. `scalar_kernel_operand_concordance` x3: a label decided from absence
   of shape, concorded as fact.
9. `precision_channel_shape_concordance` / `precision_declared_formal_width_concordance`:
   `value.shape` rewritten from an accounting field with no source row;
   `parameter_member_formals` metadata carries the resulting formal
   identity off-book.
10. `region_value_dtype` revise and `_record_ssa_shape` (`shape.ssa`):
    retyping and rescoping with no producer named.

Must be on the book FIRST (they decide value identity: naming, aliasing,
versions of a field), in this order: (1) `external_values` as a view of
binding state at each causal point (master A6.2); (2) the alias map
including its loop-time rewrites, unified with `control_value_concordance`;
(3) `value_name_histories` -> `value_names` / `parameter_names` /
`named_outputs` as pages the backends read; (4) `carried_port_values` as a
port -> phi-value page beside `loop_result_port_binding`; (5) the reducer's
`return_slot_values` / `return_record_field_states` and every
`scalar_return_field_versions` decision, with `return fallback` recorded
as UNRESOLVED; (6) `record_proven_shape` gaining a source-row argument so
its four callers here can pass what they already hold.
