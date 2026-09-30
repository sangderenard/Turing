# Concordance census 10: `fortran_c_shell.py` writers

Read-only census of every site in `src/compiler/fortran_c_shell.py` that
writes the identity book (`src/compiler/identity_concordance.py`), taken
2026-09-30 with `git grep` and `sed` only.  No compiler ids or numberings
appear here; code is referred to by function name (dotted for nested
functions).  `_class_surface_ssa_program` is one ~25k-line function; the
nested-function names below are the closest enclosing `def` that the
indentation shows, so a site attributed to the outer function may sit in a
nested block that has no `def` of its own.

Relation to `CONCORDANCE_MASTER_LIST.md` (the master list): its A2 table
lists the linker structures moved onto the book, and B1/B2 list this
file's no-book and mixed functions under the Source/Record/Consumed test.
This file adds what that list does not carry: one row per WRITE SITE with
an EDGE? classification, a fallback-as-fact column, the minted ids that
have no transform edge, and the shadow ledgers.  Master-list rows are cited
as "master list A2 row N" (row numbers count the table rows in order).
Where this reading contradicts a master-list status, section 5 says so.
The book's own primitives and helpers are census 00
(`00_book_primitives_and_helpers.md`); helper internals are cited from
there, not repeated.

Column vocabulary (same as census 00):

- EDGE? YES = the written fact or row names the exact source row(s) AND
  the stage it derives from.  PARTIAL = names a source id or a stage label
  but not a source page-row.  NO = a bare fact.
- fallback-as-fact? = yes when the written fact can be a default or a
  decision made from ABSENCE of information (no binding, no reference, no
  candidate) recorded indistinguishably from a proven fact.
- readers = who consumes the page, from `git grep` of the page name across
  `src`, `tools`, `tests`.  "self" = only the writing function reads it
  back (through the `concord` return value or a `latest` in the same
  function).  "none" = write-only receipt.

Method.  The grep the task specified returned 257 lines; of those, 78
are page writes (80 physical sites: the `shape.node` / `shape.linked`
block is present twice, verbatim), 66 are `GLOBAL_MONOTONIC_IDS.mint()` /
`mint_compiler_value_id()` calls, 1 is `GLOBAL_MONOTONIC_IDS.mint` passed
as a callable, and the rest are page handles, imports, `latest` /
`cells` / `rows` / `scope_rows` / `history` reads, and comments.  Reads
that are hazards in their own right are listed in section 4.

---

## 1. Write sites, grouped by page

### 1.1 Planning residency and its transitions

| # | enclosing function | page | row key | fact | mode | EDGE? | fallback-as-fact? | readers |
|---|---|---|---|---|---|---|---|---|
| 1 | `_publish_concordant_function_aliases` | `planning_value_concordance` | (function name, alias value id) | resident value id | `bind_alias` | PARTIAL: resident id only; the evidence that bound it lives in `provenance` (a string) and reaches the book only via #5 (when a prior alias existed) and #6 (when a shape state exists) | no | `_concordant_function_aliases` (12+ call sites here, each snapshotting into a private dict), `identity_concordance.resolved_concordant_alias_bindings` + audit, `precompile_to_ssa`, 2 probes, 3 test files |
| 2 | `_class_surface_ssa_program` (shell control handoff, the `planning_value_aliases` merge) | `planning_value_concordance` | (symbol, alias id) | resident id | `bind_alias` | NO: the merged sources are the private dicts `conditional_sequence_aliases`, `sequence_concat_aliases`, `singleton_name_aliases`; none is named | yes: `singleton_name_aliases` is built by AST name matching (`ast.Name` Load over `identity_table` histories) and single-`elts` tuples | as #1 |
| 3 | `_retire_concorded_record_identity_aliases` | `planning_value_concordance` | (symbol, alias id) | `None` (tombstone revision) | `del mapping(scope)[key]` | PARTIAL: the two `output_identity_concordance` rows that justified retirement are in the metadata receipt `retired_record_identity_aliases`, not on the page | no | as #1 |
| 4 | `_retire_dead_planning_aliases` | `planning_value_concordance` | (symbol, alias id) | `None` | `del mapping(scope)[key]` | NO | yes: retired because the ids are "absent from final SSA and descriptors"; the reason goes to metadata `retired_dead_planning_aliases`, the page gets a bare `None` | as #1 |
| 5 | `_publish_concordant_function_aliases` | `planning_alias_transition_concordance` | (function, alias id) | (incumbent resident, new resident, provenance string) | `set` (next column) | PARTIAL: prior and new resident ids and a stage label; no source row | no | `identity_concordance._planning_alias_transition_findings`, tests |
| 6 | `_publish_concordant_function_aliases` | `shape_transformation_concordance` / `_dependents` / `_state` | see census 00 | (source state, target state), stage `planning_value_concordance`, op `alias_residency`, role = provenance | helper `record_shape_transformation` | YES (census 00: the model writer) | no | census 00 |
| 7 | `_publish_concordant_function_aliases` | `sequence_row_layout_concordance` | (authored scope, resident id) | (column shapes, dtypes, source string "planning concordance transformation a->b") | helper `commit_sequence_row_layout` | PARTIAL: source is a string naming two ids, not a row | via helper (census 00: `unknown` dtypes written as facts) | census 00 |
| 14 | `_publish_concorded_output_identities` | `output_identity_concordance` | (function, alias id) | result id | `bind_alias` | NO: which merge / Phi / return produced the output identity is not recorded; the write is guarded only against the metadata twin `output_identity_aliases` | no | `_concordant_function_aliases(include_output_identities=True)`, `_retire_concorded_record_identity_aliases`, audit, 2 test files |

`_concord_unbound_variant_rows` writes through #1 with provenance
`unbound_variant_row`; `_class_surface_ssa_program`'s record-projection
alias block writes through #1 with provenance
`linked_call_record_projection` and mirrors the decision in metadata
`record_projection_alias_receipts`.

### 1.2 Sequence contracts (helpers)

| # | enclosing function | page | row key | fact | mode | EDGE? | fallback-as-fact? | readers |
|---|---|---|---|---|---|---|---|---|
| 8 | `_sequence_column_dtype_contracts` | `sequence_row_layout_concordance` | (function name, sequence id) | (shapes, dtypes, "ProgramABI sequence row record") | helper | PARTIAL (string) | via helper | census 00 |
| 9 | `_field_slot_ops` (authored parameter annotation) | `sequence_contract_concordance` | (scope, first history id of the parameter) | (policy, columns, writable, "authored parameter ...") | helper `commit_sequence_contract` | PARTIAL (string) | no | `committed_sequence_contract` callers here, `precompile_to_ssa`, tests |
| 10 | `_field_slot_ops` (runtime aggregate) | same | (scope, sequence id) | (..., "runtime aggregate kind.name viewed as ...") | helper | PARTIAL | `writable` derived from `aggregate_kind not in {tuple, bytes}` when no committed contract | as #9 |
| 11 | `_field_slot_ops` (shell sequence declaration) | same | (scope, sequence id) | (..., "shell sequence declaration") | helper | PARTIAL | yes: every referenced field table not otherwise declared is written as `("unique", 2, writable)` -- a fixed default | as #9 |
| 12 | `_class_surface_ssa_program` (control sequence mutation) | same | (symbol, sequence id) | (..., "control sequence mutation node:op") | helper | PARTIAL | yes: `column_count` falls back to `max(1, len(argument_value_ids))` when neither an explicit row width nor an incumbent exists | as #9 |
| 13 | `_class_surface_ssa_program` (shell control handoff) | same | (symbol, sequence id) | (..., "shell control handoff") | helper | PARTIAL | inherits #11/#12 | as #9 |

### 1.3 Alias application and record-return Phi inputs

| # | enclosing function | page | row key | fact | mode | EDGE? | fallback-as-fact? | readers |
|---|---|---|---|---|---|---|---|---|
| 15 | `_concord_record_return_phi_inputs` | `record_return_phi_input_concordance` | (function, Phi result id, field name, position, predecessor block) | (candidate id, chosen id, reason) | `set` (next column) | PARTIAL: names candidate and chosen ids and a reason; the `materialize_record_phis` selection that proposed the candidate is not a row | the `merged_descriptor_self_candidate_rejected` branch keeps the incumbent because the candidate was the Phi itself -- recorded as a decision with a reason (good form) but not as unresolved | tests only (master list B2: mixed) |
| 16 | `_apply_concorded_function_aliases` | `alias_application_concordance` | (function, block, instruction index, operand position) | (argument id, resident id, selected id, reason, target block) | `set`, first write only | PARTIAL: resident id named; the `planning_value_concordance` row it resolved through is implied by scope, not written | `resident_not_available_at_use` keeps the original operand from lack of dominance -- recorded with reason, not as unresolved | tests; metadata twin `alias_application_receipts` |

### 1.4 Frame linking: bindings, leases, tails, pruning

| # | enclosing function | page | row key | fact | mode | EDGE? | fallback-as-fact? | readers |
|---|---|---|---|---|---|---|---|---|
| 22 | `_class_surface_ssa_program` (per linked call, after `frame_bindings` is built) | `argument_binding` | (callee symbol, callee value id, "binding"); column = callsite id | (kind, source): kind in caller_value / caller_alias / caller_storage / default_literal / caller_literal ..., source a caller value id or literal | `set` | PARTIAL: names the caller value id and the callsite (column) but not how it was found (`identity_aliases`, `default_literals`, `discovery_linked_member`, `allocate_result_storage` are all private) | YES, load-bearing: the final `else` ("every remaining callee argument ... give the caller a distinct storage slot") mints storage from absence and records it as `caller_storage`, indistinguishable from a proven binding | `identity_concordance._binding_kind_findings`, `materializing_binding_kind`, `tensor_ssa_lowering`, `_complete_propagated_frame_tails` (reads `.cells` directly), tests |
| 21 | `_link_frame_lease` (called from three linker sites: caller-storage clone, distinct-result replacement, per-owner slot replacement) | `frame_lease_link` | (caller symbol, slot id) | (callsite id, callee symbol, callee formal id) | `concord` | PARTIAL: names the source identity (callsite + callee formal) but no page/row/stage | no | none (write-only; `_lease_source` docstring refers to it) |
| 23 | `_complete_propagated_frame_tails` | `propagated_frame_tail_concordance` | (owner, callsite id, callee, callee formal id) | (slot id, storage kind string) | `set` col 0, raise on differ | PARTIAL: callee formal named; the `argument_binding` cell consulted is named only inside the kind string (`argument_binding:restored_caller_storage`) | yes: when no concordance binding exists a fresh slot is minted and recorded as `linked_call_frame_storage` / `compiler_frame_storage` / `returned_record_storage` | none |
| 24 | `_prune_unused_callee_formals_once` | `pruned_callee_formal_concordance` | (callee, formal id) | "unused" | `concord` | NO | yes: absence of use recorded as a fact | self |
| 17 | `_prune_dead_entry_field_aliases` | `entry_record_handle_concordance` | (function, parameter name) | "fields_only" | `concord` | NO | yes: written when the aggregate id is unreferenced and owns no storage accounting | self |
| 18 | `_class_surface_ssa_program` (child record pairing in call linking) | `call_record_pair_concordance` | (caller, callsite id, callee child record id) | caller child record id | `concord` | NO: `caller_children` is keyed by `storage_identity` string (master list Part B known matchers: "bound_record_pairs child pairing by storage_identity") | no (raises on disagreement), but the source is a name match | `_linked_caller_member` (`latest`) |
| 28 | `_class_surface_ssa_program` (link order) | `call_link_order_concordance` | (artifact name, symbol) | rank | `concord` | NO | no | self |
| 73 | `_class_surface_ssa_program` (scheduled call operands, mapping under a minted scope) | `scheduled_call_argument` | (caller, callsite, callee formal id) | the resident SSAValue | mapping-set | NO: the loop lowering that resolved the operand is not named | no | self (`.get` in the same function); master list A2 row 7 BOOK |
| 74 | `_class_surface_ssa_program` (shape harmonization of kernel formals) | `kernel_by_value_formal_concordance` | (callee, formal id) | "by_value" | `concord` | NO (decided from `llvm_argument_names` metadata + dtype) | no | self |
| 75 | `_class_surface_ssa_program` (Phi-edge projection relocation) | `phi_edge_projection_placement_concordance` | (function, Phi id, position, predecessor, value id) | (moved instruction ids, placement block, reason) | `concord` | PARTIAL: names moved ids and reason, no stage/row | no | none; metadata twin `phi_edge_projection_repairs` |

### 1.5 Record ABI materialization (the `record_field_access` family)

All rows in this group live under one minted scope
`mint_scope("record_field_access")` (master list A1 `scope_registry`).

| # | enclosing function | page | row key | fact | mode | EDGE? | fallback-as-fact? | readers |
|---|---|---|---|---|---|---|---|---|
| 35 | `_class_surface_ssa_program` (declared record parameters loop) | `record_parameter_value` | (scope, (symbol, value id)) | (symbol, parameter name) | `concord`; also read through a `PageMapping` | NO: derived from `identity_table` name -> ids; the graph row is not named | no | self (forwarding); master list A2 row 3 |
| 43 | same | `record_parameter_row_handle` | (scope, (symbol, value id)) | ((symbol, parameter), "field[]", row identity) | `concord`; also a `PageMapping` | NO: derived from a `setattr`/keyed-field graph node, not named | yes: row identity falls back to the `value_record` schema NAME when the ABI record has no `identity` | self (forwarding) |
| 42 | same | `record_field_demand_concordance` | (symbol, parameter) | sorted leaf paths | `concord` | NO: the ABI record walk is not named | no | self |
| 36 | `_class_surface_ssa_program.note_record_field_access` | `record_field_access` | (scope, ((symbol, parameter), field path, storage identity)) | frozenset of roles {read, write, layout, sequence} | mapping-set | YES by companion: every role added here is accompanied by #39 | no | self (transitive loop, `field_access_receipts`), tests; master list A2 row 1 (under a different page name, see section 5) |
| 39 | `_class_surface_ssa_program.note_record_field_access` | `record_field_access_path` | (scope, access key, role) | witness tuple: ("source-node", symbol, node) / ("numeric-record-layout", record identity) / ... | `concord` | YES: names the exact source node or declaration | no | self, tests |
| 37 | `_class_surface_ssa_program` (forwarding fixed point, caller-ward) | `record_field_access` | caller access key | incumbent roles | propagated | YES by companion #40 | no | self |
| 40 | same | `record_field_access_path` | (scope, caller access key, role) | (("callsite", caller, callsite, callee), *callee path) | `concord` | YES: names callsite and copies the callee's path -- this is the causal edge the goal asks for | no (but see #47: only `callsites[0]` is named) | self |
| 38 | same (callee-ward "sequence" role) | `record_field_access` | callee access key | incumbent roles + {"sequence"} | mapping-set | YES by companion #41 | no | self |
| 41 | same | `record_field_access_path` | (scope, callee access key, "sequence") | (("callsite-sequence-view", ...), *caller path) | `concord` | YES | no | self |
| 44-46 | `_class_surface_ssa_program` (binding walk over `pending_call_records`) | `record_forwarding_unresolved_actual` | (scope, (caller, callsite, caller id, callee, callee id)) | (reason string, caller key or None, callee key or None) | `concord` x3 branches | YES: names both ends and the callsite; this is the one place in the file where an absence is recorded AS unresolved, as the goal requires | no (it records the absence honestly) | none (write-only; nothing consumes the unresolved rows yet) |
| 47 | same | `record_forwarding_edge` | (caller key, callee key) | (callsite id,) | mapping-set | PARTIAL: names one callsite; a second callsite for the same pair overwrites the fact (a revision), and the propagation witness uses `callsites[0]` only | no | self; master list A2 row 4 |
| 48 | `_class_surface_ssa_program` (sequence view settlement) | `record_field_sequence_view` | (scope, storage identity) | the one decisive receipt | `concord`, raise if not exactly one | NO | no | self (`latest` in keyed materialization) |

### 1.6 Record storage residency (per-function scope)

| # | enclosing function | page | row key | fact | mode | EDGE? | fallback-as-fact? | readers |
|---|---|---|---|---|---|---|---|---|
| 49-51 | `_class_surface_ssa_program.materialize_parameter_record_abi.coalesce_record_field_storage` | `record_storage_alias` (mapping under `mint_scope(("record_storage_alias", symbol))`) | value id | resident id | mapping-set (3 branches: read-only field, getters, write sources) | NO | YES: resident is the first argument matching a read id, else `read_ids[0]` / `candidates[0]` -- first-in-list from absence of an argument match | self (rewrites every instruction operand in the function); master list A2 row 6 BOOK |
| 52 | `...materialize_parameter_record_abi` (resolve pass) | `record_storage_alias` | alias id | terminal resident | mapping-set | NO | inherits | self |
| 53 | `...coalesce_record_field_storage` | `record_field_resident_concordance` | (symbol, min(parameter ids), field name) | resident id | `concord` | NO | yes (same choice as #49); also `min(parameter_ids)` uses id ORDER as the row identity | self |

### 1.7 Record ABI materialization: decompositions, numerals, literals

| # | enclosing function | page | row key | fact | mode | EDGE? | fallback-as-fact? | readers |
|---|---|---|---|---|---|---|---|---|
| 19 | `...materialize_parameter_record_abi.publish_keyed_decomposition` | `record_field_decomposition` | (table owner, record id, storage identity) | ((role path, member id), ...) | `concord` | NO: member ids are freshly minted here; no edge from the declared ABI field that produced them | no | `_linked_caller_member`, `_class_surface_ssa_program.resolve_keyed_mapping_iterables`, tests |
| 20 | `...materialize_parameter_record_abi` (keyed field with value record) | `program_abi_keyed_row_record` | (owner, record id, storage identity) | row record id (minted in `materialize_declared_row_columns`) | `concord` | NO | no | none |
| 54 | `...materialize_parameter_record_abi.materialize_nested_record` | `numeral_leaf_materialization_concordance` | (symbol, parameter, nested path) | "required_unread_leaf" | `concord` | NO | yes: an unread leaf is minted right after; the status string is a receipt, the mint has no edge | none |
| 55 | `...materialize_parameter_record_abi.materialize_declared_row_columns` | same | (symbol, parameter, leaf path) | "declared_row_leaf_deferred:<storage>" | `concord` | NO | yes (deferral recorded as a status) | none |
| 56 | `...materialize_nested_record` | `numeral_leaf_width_concordance` | (symbol, parameter, nested path) | declared `precision_limbs` | `concord` | NO (source is the ABI record, unnamed) | no | none |
| 57 | `_class_surface_ssa_program.allocate_result_storage` | `numeral_leaf_width_concordance` | (caller, callsite, callee old id) | the one width found | `concord` | PARTIAL, and the source rows are found by `storage_identity.rpartition(".")` string matching over `source_numeric_record_abi_concordance` rows (master list Part B known matcher "numeral result limbs") | no (skips when widths != 1) | none |
| 58 | `_class_surface_ssa_program.materialize_program_abi_record_literals` | `numeral_record_literal_concordance` | (symbol, record id, field name) | field value id | `concord` | NO | YES: when the literal omits a field with a `default`, `value_id = mint()` and that minted default is recorded as the field's value identity with no marker | `complete_linked_literals` (scans rows for the "deferred" marker) |
| 59 | same | same | (symbol, node id, "deferred") | True | `concord` | NO | receipt of incompleteness (acceptable form) | `complete_linked_literals` |
| 60 | same | same | (symbol, node id, "completed") | ((field, record id, value ids), ...) | `concord` | NO | no | none |
| 61 | `_class_surface_ssa_program.complete_linked_literals` | `numeral_return_leaves_concordance` | (symbol, returned argument id) | leaf ids | `concord` | NO | no | none |
| 25 | `_field_slot_ops` (receiver nested record field) | `receiver_nested_record_field_concordance` | (sequence contract scope, attribute) | nested record identity string | `concord` | NO | yes: `nested_receipt.get("identity") or nested_schema` -- the schema NAME stands in when the ABI receipt lacks an identity | self |

### 1.8 Record Phi and loop-record layouts

| # | enclosing function | page | row key | fact | mode | EDGE? | fallback-as-fact? | readers |
|---|---|---|---|---|---|---|---|---|
| 69 | `_class_surface_ssa_program.materialize_record_phis` | `record_field_layout_concordance` | (symbol, Phi result id, storage identity) | (value ids, sequence id, record id, offset, storage kind) | `set` col 0, raise on differ | NO: the field value ids were minted for the merged descriptor (see section 2) and the Phi inputs they merge are not named | no | `_class_surface_ssa_program` call-linking record reconciliation (`latest`, two sites) |
| 70 | `_class_surface_ssa_program` (call-linking field refinement) | `record_field_storage_concordance` | (caller, caller record id, storage identity) | {"from": storage, "to": storage, "sequence_id", "record_id"} | `set` (next column) | PARTIAL: before/after storage and the incoming descriptor's ids; the callee row it came from is not named | no | none |
| 71 | `_class_surface_ssa_program.materialize_record_phis` (loop Phi) | `loop_record_layout_concordance` | (symbol, source loop node, result id) | (old physical layout, new layout, "record_loop_phi") | `set` col 0 | PARTIAL: old and new id tuples; no descriptor row | no | none; metadata twin `loop_record_layout_transitions` |
| 72 | `_class_surface_ssa_program.materialize_loop_record_phis` | `loop_record_schema_concordance` | (symbol, loop id, result id) | (initial signature, updated signature, projected names, discarded names, "project_updated_to_initial") | `set` col 0 | PARTIAL; the `projected_updated_id` minted next is NOT in the fact | no | none; metadata twin `loop_record_schema_projections` |

### 1.9 Shape pages (whole-program ABI settlement, in `_class_surface_ssa_program`)

| # | enclosing function | page | row key | fact | mode | EDGE? | fallback-as-fact? | readers |
|---|---|---|---|---|---|---|---|---|
| 29 | `_class_surface_ssa_program.note_shape` | `value_shape` | (function, value id) | (state, extents, dtype, storage); state "declared"/"propagated"/"polymorphic"/... | `set` at column = history length, skipped when payload unchanged | NO: which caller edge delivered the shape is not in the fact (the edge exists on #30 but is not referenced) | no | `tensor_ssa_lowering`, the `ssa_shape_materialization` seam (#76), 2 tests, 4 probes |
| 30 | same function (edge enumeration) | `call_edge` | (callee, callee id, caller, caller id) | "argument" | `set` col 0 | YES: the row is the edge (fact is constant) | no | 3 probes only |
| 31 | same function (polymorphic formal) | `linked_value_abi_polymorphism` | (callee, callee id) | (caller, caller id, existing shape, new shape) | `set` at history length | YES: names the disagreeing caller value | no | `tensor_ssa_lowering`, 1 probe |
| 32, 33 | same function, block present TWICE verbatim | `shape.node`, `shape.linked` | (function, value id) | node tensor extents / linked contract shape | `set` col 0 | NO | no | 3 probes.  The second copy re-`set`s column 0 of the same rows: `IdentityPage.set` has no guard, so the cell is overwritten with the clock advanced -- one fact, two stamps |
| 76 | `_class_surface_ssa_program` (SSA shape materialization seam) | `ssa_shape_materialization` | (function, value id) | (shape, dtype, storage, authored owner) | `set` col 0, raise on differ | PARTIAL: derived from the `value_shape` row of the same key (implied, not written) | no | none; metadata `ssa_shape_materializations` |
| 77 | `_class_surface_ssa_program` (declared scalar record fields) | `record_scalar_shape_concordance` | (function, value id) | (prior shape, ()) | `concord` | NO | no | none |
| 78 | `_class_surface_ssa_program.recover_structural_source_outputs.ensure_structural_value` | `proven_shape` | (authored function, value id) | ("proven", extents, dtype) | helper `record_proven_shape` | NO (census 00: no source, no stage) | YES: when operand shapes are broadcast-incompatible the code keeps "the ranked operand" and records it as proven | 11 files (census 00) |

### 1.10 Source-stage declarations (`_lower_ast_source_to_ssa_impl` and entry of `_class_surface_ssa_program`)

| # | enclosing function | page | row key | fact | mode | EDGE? | fallback-as-fact? | readers |
|---|---|---|---|---|---|---|---|---|
| 26 | `_class_surface_ssa_program` (entry) | `assignment_projection_source_concordance` | (source scope, aggregate name, projection index) | target name | `concord` | NO (from the normalizer's `assignment_projections`, names only) | no | same function at link time, tests |
| 27 | `_class_surface_ssa_program` (link order block) | `assignment_projection_identity_concordance` | (caller, callsite, projection index) | (aggregate name, target name, caller projection id, target history ids) | `concord` | PARTIAL: derived from #26 and from `call_result_projection_concordance` (written by `glsl_deployment_strategy`), neither named; the scope match is `str.endswith("." + candidate)` | no | same function (result binding), tests |
| 34 | `_class_surface_ssa_program` (member formal accounting, inside a `try`) | `member_formals` | (function, value id, "member_formal"); column = aggregate index | ("granted"/"refused", root name, binding kind, authored parameters) | `set` | PARTIAL: names the root binding name | "refused" is recorded with its cause (good form) | `identity_concordance` audit |
| 62 | `_lower_ast_source_to_ssa_impl.numeric_record_schema` | `source_numeric_record_abi_concordance` | (type name, limbs) | (schema name, receipt) | `set` col 0, raise on differ | NO (declaration) | no | `materialize_parameter_record_abi`, `allocate_result_storage` (#57, by string) |
| 63 | `_lower_ast_source_to_ssa_impl` | `source_numeric_parameter_abi_concordance` | (definition name, argument name) | (schema, type, limbs) | `set` | NO | no | none |
| 64 | `_lower_ast_source_to_ssa_impl.numeric_parameter_record_views` | `source_numeric_parameter_record_view_concordance` | (scope, parameter, node id) | (schema, type, limbs) | `set` | NO | no | none |
| 65 | `_lower_ast_source_to_ssa_impl` | `source_numeric_type_dependency_concordance` | (type name, limbs) | dependency receipts | `set` | NO | no | `identity_concordance` audit |
| 66 | `_lower_ast_source_to_ssa_impl` | `source_numeric_method_dependency_concordance` | (type name, limbs) | same-type method roots | `set` | NO | no | none |
| 67 | `_lower_ast_source_to_ssa_impl` | `static_record_piece_mapping_concordance` | (class identity, field, key) | (binding, entry, artifact) | `concord` | NO | no | none |
| 68 | `_lower_ast_source_to_ssa_impl` | `static_record_piece_sequence_concordance` | (class identity, field) | ((binding, entry, artifact), ...) | `concord` | NO | no | none |

Not writes, listed for completeness: `_loop_scope_rebound_outer_ids`
(reads `loop_scope_declarations`), `_concordant_function_aliases`,
`_linked_caller_member`, `resolve_keyed_mapping_iterables`,
`complete_linked_literals` (rows scan), the `source_precision_boundary`
lookup in `ensure_structural_value`, `boundary_page`, `decomposition_page`,
`literal_page`, `abi_page`.

---

## 2. Minted ids: is the mint accompanied by an edge?

66 mint sites (`GLOBAL_MONOTONIC_IDS.mint()` directly or through the local
`mint_compiler_value_id`, which is a bare passthrough) plus one site that
hands `GLOBAL_MONOTONIC_IDS.mint` to `CTypesInterception` so layout tables
mint outside this file.  The `MINTED` flag (census 00 section 4) marks each
result; nothing at mint time records the transform.

| where (function) | what is minted | book row at or after the mint | edge? |
|---|---|---|---|
| `_complete_propagated_frame_tails` | a caller slot for an unbound callee formal | `propagated_frame_tail_concordance` (#23): (owner, callsite, callee, formal) -> (slot, kind) | PARTIAL: callee formal named; chosen from absence of an `argument_binding` cell |
| `_propagate_record_field_demand.linked_member.grow`; `_propagate_record_field_demand` (propagated caller field) | grown / propagated ProgramABI field formals | none (master list B1: no-book) | NONE -- bare |
| `_class_surface_ssa_program` (emit outputs) | index and address for output stores | none | bare scaffolding |
| `...recover_structural_source_outputs.structural_boolop_value` | chain intermediates of a Boolean fold | none | bare |
| `...structural_membership_value` | negated membership call result | none | bare |
| `...ensure_structural_value` | index / address for a one-element load | none | bare |
| `...materialize_parameter_record_abi.materialize_nested_record` (about 17 sites) | column arenas, lengths, stride, row offset, pointers, length pointer, pooled scalar columns, token constants, comparisons, merged predicate, `part_id`, and `candidates = (mint(),)` after "required_unread_leaf" | parts reach `record_field_decomposition` (#19) as members; the leaf mint follows #54; the rest none | NONE: the decomposition names the parts but not the declared field they came from; the "required_unread_leaf" mint is a fallback |
| `...materialize_declared_row_columns` | pooled column, `row_record_id` | `program_abi_keyed_row_record` (#20) records the row record id | NO edge |
| `...materialize_parameter_record_abi` (top level) | `candidate_ids` fallback, extra column ids, live flags, status, length, capacity, `part_id`, presence ids, `became_present` | members registered on `SSASequenceTable` / `SSARecordTable` (master list A1 BOOK pages `sequence_member`, `record_member`) | membership, not transform: the pages say WHICH descriptor owns the id, not what produced it |
| `...materialize_program_abi_record_literals` | default field value (`"default" in field`), presence, inactive payload | `numeral_record_literal_concordance` (#58) records the default as the field value | fallback-as-fact, no edge |
| `...materialize_record_phis.select_return_arguments` | the `Cast` converting a Boolean-leaf Phi to the field dtype | none; the instruction carries `source_field_value_id` in its attributes | known bare mint; the edge exists only on the instruction |
| `...materialize_record_phis` | merged field values of a record Phi | `record_field_layout_concordance` (#69) records the ids as the layout | NO edge from the Phi's inputs |
| `...materialize_loop_record_phis` | `projected_updated_id`, `header_record_id` | `loop_record_schema_concordance` (#72) names loop and result, not the new ids; metadata `loop_record_schema_projections` does | bare on the book |
| `_class_surface_ssa_program` (constructor frame remap, three sites) | `remap[old_id]` for referenced ids, constructor args, pool ids | none; `remap` is private | bare |
| same (instance pool strides) | `row_stride_id`, `scalar_stride_id` | none | bare |
| `_class_surface_ssa_program.allocate_result_storage` | caller storage for a callee result | `result_storage_bindings` (private), accounting `returned_record_storage` + `callsite_id`; later `argument_binding` (#22) as `("caller_storage", new id)` | PARTIAL, one hop later, via a bare id |
| same (result record bindings) | caller record ids for nested returned records | none | bare |
| `_class_surface_ssa_program.resolve_call_feed` | fold intermediates | none | bare |
| `_class_surface_ssa_program` (linker: caller storage clone, `record_id_map`, `allocate_late_result_storage`, mapped field ids) | linked frame storage | `frame_lease_link` (#21) for the clone; accounting `linked_call_frame_storage` | PARTIAL |
| same (`replacement_id` for distinct results; per-owner slot replacement) | replacement slots | `frame_ledger.propose(identity, rule, proof, target)` -- `TransformationLedger` pages `transformation_decision` / `transformation_event` (master list A1 BOOK) | the best-formed mints in the file: rule + proof + target on the book; still no `stage` |
| same (missing aggregate output position) | `node_id` when `projected_by_index` lacks the index | none | bare fallback |
| same (instance-pool pointers/offsets, derived length, returned length, default/caller literal values x3, fresh `caller_result_id` on a collision, output index/address, projected element addresses, child pool columns/lengths x2) | frame scaffolding | none (accounting strings only) | bare |
| `_class_surface_ssa_program` (freshen redefined synthetic ids) | new `result.id` | metadata `freshened_synthetic_value_ids` | bare on the book |
| `_lower_ast_source_to_ssa_impl` | `GLOBAL_MONOTONIC_IDS.mint` handed to `CTypesInterception` | none | bare, outside this file |

Tally: 0 mints with a full edge; 4 groups PARTIAL (frame tails, leases,
result storage one hop later, ledger-proposed replacements); every other
mint (about 50 sites) is bare.  The two 2026-09-29 additions the task
names: `record_parameter_row_handle` (#43) is a bare fact with a
name fallback; `record_forwarding_unresolved_actual` (#44-46) is the one
writer in the file already in the target form (absence recorded as
unresolved, both ends named).

---

## 3. Shadow ledgers in this file

Identity or provenance facts kept outside the book.  "twin" = a book page
holds the same fact; "none" = the book has no counterpart.

### 3.1 `function.metadata[...]` channels

| channel | identity fact it holds | writers (this file) | readers | book counterpart |
|---|---|---|---|---|
| `parameter_names` | (authored name, value id) per formal; the name->id root of every ABI decision | `_rebind_recorded_scalar_identities`, `materialize_program_abi_record_literals` (+ upstream writers outside this file) | 24 read sites here (`_drop_unused_*`, `_recover_late_*`, `_prune_*`, `_linked_authored_parameter_aliases`, `_report_unmaterialised_record_parameters`, ...) and ~60 files across `src`/`tools` | none (`member_formals` covers aggregate members only) |
| `record_return_layouts` | (record id, physical layout ids) per function; the returned-record ABI | 6 sites: `recover_structural_source_outputs`, `materialize_program_abi_record_literals`, `materialize_record_phis`, `complete_linked_literals`, two call-linking blocks | 8 read sites here plus `ssa_record_return_state` (also writes it), `project_compilation_product` | partial: `record_field_layout_concordance` (#69) per field of Phi results only |
| `value_aliases` | the durable planning-alias snapshot | `_publish_concordant_function_aliases`, both `_retire_*` | `_concordant_function_aliases` (12+ snapshots into private dicts), 20 other files | twin of `planning_value_concordance`; master list A6.3 |
| `output_identity_aliases` | output identity snapshot | `_publish_concorded_output_identities` | `_concordant_function_aliases`, audit, 2 probes | twin of `output_identity_concordance` (checked for equality on every read) |
| `storage_formals`, `closure_formals`, `parameter_member_formals` | which formals are compiler storage / closure captures / aggregate members | `_class_surface_ssa_program`, `_record_module_closure_formals`, control lowering | audit, `ssa_self_check`, `ssa_python_materializer`, probes | `formal_storage_resolution` (census 00) for storage; `member_formals` (#34) for members; none for closures |
| `carried_port_values` | loop-carried port -> resident value | written outside this file (`precompile_to_ssa`) | 5 read sites here (record return expansion, loop record Phis) | none |
| `singleton_call_result_concordance` | callee, callsite, temporary id, projected result id, reason | `_class_surface_ssa_program` (linker) | `_reconcile_singleton_destructured_call_results` | none -- a "concordance" that exists only in metadata |
| `nonreturn_result_bindings`, `destructured_call_result_concordance`, `aggregate_return_layouts` | leftover physical bindings vs. the callee Ret ABI; destructuring receipts | linker | linker | none |
| `returned_record_*` (6 channels: slot / projection incumbent / duplicate position / descriptor / concordant surface / argument reconciliations), `late_returned_record_materializations`, `post_aggregate_record_result_receipts`, `forwarded_record_result_reconciliations` | how returned records were rebound at call frames | linker, `_reconcile_post_aggregate_record_results` | diagnostics | none |
| `record_projection_alias_receipts`, `retired_record_identity_aliases`, `retired_dead_planning_aliases`, `alias_application_receipts`, `loop_record_layout_transitions`, `loop_record_schema_projections`, `pruned_dead_entry_field_aliases`, `phi_edge_projection_repairs` | receipts restating a page write, sometimes with MORE cause than the page (#3, #4, #72) | the writers of #1, #3, #4, #16, #17, #71, #72, #75 | diagnostics | twins whose page half is the poorer copy |
| `frame_transformation_provenance` | the ledger's events | linker (`frame_ledger.events`) | diagnostics | live view of `transformation_event` (BOOK) |
| `structural_output_shortfalls`, `unresolved_record_sequence_rows`, `record_phi_temporal_fallbacks`, `settled_nonlive_structural_shortfalls`, `unresolved_required_source_values`, `unresolved_call_diagnostics` | decisions taken from absence or fallbacks | several | diagnostics, `lower_ast_source_to_ssa` reporting | none -- these are exactly the "unresolved" facts the goal wants on the book |
| `freshened_synthetic_value_ids`, `redefined_ssa_object_freshenings`, `identity_cast_result_reconciliations`, `structural_identity_rebindings`, `loop_result_use_rebindings`, `control_identity_receipts` | id rewrites | several | diagnostics | none |

### 3.2 Private dicts (inside `_class_surface_ssa_program` unless noted)

| name | identity fact | writers | readers | book counterpart |
|---|---|---|---|---|
| `result_storage_bindings` (per linked call) and `result_storage_bindings_by_call` | callee value id -> caller storage id for returned records | `allocate_result_storage`, `allocate_late_result_storage`, field mapping | 35 read sites in the linker | reaches the book only as the bare source id inside `argument_binding` (#22); master list A6.3 names only the `_by_call` map |
| `identity_aliases`, `default_literals` (per call) | the inputs of `frame_bindings` | binding walk | #22 | none; #22 records their OUTPUT |
| `caller_children` | callee child record -> caller child record, keyed by `storage_identity` string | pairing block | #18 | `call_record_pair_concordance` records the result of the string match |
| `constructor_anchors`, `constructor_instance_pools`, `call_anchor_value_ids` | constructor result anchors and pooled instances | literal materialization, linker | linker | none (master list A6.3) |
| `frame_ledgers` | caller -> `TransformationLedger` handle | linker | linker | the ledger IS on the book; the dict is a handle cache (master list A6.3 overstates this one) |
| `caller_value_aliases`, `caller_record_aliases`, `record_identity_aliases`, `caller_aliases`, `sequence_aliases`, `aggregate_result_aliases` | snapshots of `_concordant_function_aliases` taken at eight points in the linker, each then chased privately (`physical_caller_storage`, `resident_*` helpers) | linker | linker | views of `planning_value_concordance`; stale within a pass once #1 advances |
| `singleton_name_aliases`, `conditional_sequence_aliases`, `sequence_concat_aliases`, `planning_value_aliases` | the alias sources merged into #2 | shell handoff | #2 | none for the source decisions |
| `resident_by_value`, `kind_by_value` (in `_sequence_concat_ops`, `_promote_conditional_sequence_aliases`, `_sequence_augassign_ops`, `_sequence_prepend_*`, `_sequence_inplace_bit_pack_call_ops`, `_sequence_row_operations`) | sequence residency and kind per value; 21 write sites | those functions | `_class_surface_ssa_program` via `conditional_sequence_aliases` | none until #2 (bare) |
| `aliases_by_value`, `sequence_record_by_value`, `const_sources`, `direct_field_by_value`, `remap`, `record_id_map`, `indexed_aliases` / `indexed_storage_aliases` (`_loop_carried_storage_aliases(graph)`) | more residency maps decided from graph structure | `_nested_row_projection_ops`, `_record_sequence_projection_bindings`, `_field_slot_ops`, `_sequence_column_dtype_contracts`, record ABI, constructor remap | local | none |
| `output_identity_aliases` (dict in `recover_structural_source_outputs`) | final output identity per structural value | structural recovery | #14 via `_publish_concorded_output_identities` | twin |
| `scheduled_call_sources` | now a `PageMapping` (#73) | | | on the book (state, no edge) |

### 3.3 `graph.graph[...]`

`parameter_record_abi`, `parameter_value_abi`, `sequence_record_abi`,
`parameter_sequence_record_abi`, `program_abi`, `linked_value_abi`,
`optional_record_presence_receipts`, `optional_record_presence_lowered`,
`layout_type_tables`, plus `identity_table` (read 54 times here, written
upstream).  These are the DECLARED sources most 1.5-1.7 rows derive from
and are the "source row" the facts fail to name.  `linked_value_abi` is
the private dict `linked_value_abi_by_graph` copied onto the graph after
the shape settlement; its book twins are `value_shape` / `shape.linked`.

---

## 4. Read-side hazards seen while tracing writers

- `_complete_propagated_frame_tails` reads `binding_page.cells.get((row, column))` directly, bypassing `latest` / `history`.
- `ensure_structural_value`'s `source_precision_boundary_concordance` lookup, when the exact row is absent, scans every row of the page for `int(row[1]) == value_id and candidate[1] in parent_ids`: a value-id join outside a row (master list working rule 1).
- `complete_linked_literals`, `allocate_result_storage` (#57) and `materialize_program_abi_record_literals` scan `page.rows()` and match `row[0]`/`row[2]` by string instead of asking a row.
- Eight linker sites snapshot `_concordant_function_aliases` into private dicts and chase them with their own `while current in aliases` loops (`physical_caller_storage`, `resident_caller_record_id`, `resident_record_id`, `caller_resident`, `resident_argument_record_id`, `resident_result_slot`, `resolve_indexed_storage`, `resolve_record_storage`) instead of `resolve_alias`.

---

## 5. Where this census differs from the master list

1. A2 rows 1 and 2 name pages `record_field_demand` and
   `record_field_write` "(via PageMapping)".  Neither string exists in
   `src`, `tools` or `tests`.  The live pages are `record_field_access`
   (one frozenset of roles read/write/layout/sequence per field row, via
   `PageMapping`), `record_field_access_path` (the witness edge) and
   `record_field_demand_concordance` (leaf paths).  Status BOOK holds for
   the access page under the three-part test; the row names are stale.
2. A2 row 6 `record_storage_alias` BOOK: the resident is chosen by
   `read_ids[0]` / `candidates[0]` when no formal matches and recorded as a
   fact (#49-53).  BOOK by the three-part test; fallback-as-fact by the
   edge rule.
3. A2 row 7 `scheduled_call_argument` BOOK and row 3
   `record_parameter_value` BOOK: agreed as state; neither fact names what
   produced it (NO edge).
4. B1 lists `_class_surface_ssa_program.materialize_parameter_record_abi`
   and `.resolve_keyed_mapping_iterables` as no-book.  The first now writes
   `record_field_decomposition`, `program_abi_keyed_row_record`,
   `numeral_leaf_*`, `record_storage_alias`,
   `record_field_resident_concordance` (through nested defs); the second
   reads `record_field_decomposition`.  Part B's scan predates these; the
   functions are mixed, and by the edge rule every one of those writes is
   NO.
5. A6.3 names `frame_ledgers` as a remaining private map.  The ledger
   pages are on the book (A1); only the handle cache and the metadata
   mirror `frame_transformation_provenance` are private.  A6.3 omits the
   per-call `result_storage_bindings` (35 reads), the larger gap.
6. Part B "known matchers" lists the `storage_identity` child pairing; add
   that the result of that string match is then CONCORDED (#18), so the
   page certifies a name match as an identity fact.
7. `_concordant_function_aliases` and `_publish_concordant_function_aliases`
   are B2 mixed; agreed.  Under the edge rule the publish is PARTIAL (#1)
   and only its shape half (#6) carries an edge.
8. The `shape.node` / `shape.linked` writer exists twice, verbatim, in
   `_class_surface_ssa_program`; the master list has no row for these
   pages.

---

## 6. Verdict

Counts: 78 write sites on 56 pages (80 physical; 45 pages are written
only from this file).  EDGE? YES 12 (all in two places: `call_edge` /
`linked_value_abi_polymorphism`, and the `record_field_access` family with
its `_path` witnesses and `record_forwarding_unresolved_actual`); PARTIAL
23; NO 43.  Fallback-as-fact: 18 sites.  Mints: 66 sites, 0 with a full
edge, 4 groups partial, about 50 bare.  Pages consumed by nothing but
their writer or by nothing at all: 31 of 56.

The ten most load-bearing bare or fallback writers, by reader count and by
what depends on them:

1. `argument_binding` (#22), the `else` branch: caller storage minted from
   absence and recorded as `caller_storage`.  Read by the audit,
   `materializing_binding_kind`, `tensor_ssa_lowering`, the frame-tail
   pass.  Everything the frame linker later believes about a callee
   formal rests on this row.
2. `planning_value_concordance` at the shell handoff (#2): the merged
   private residency dicts, including AST-name-derived
   `singleton_name_aliases`, become planning facts with no source.  Read
   through `_concordant_function_aliases` at 12+ linker sites.
3. `output_identity_concordance` (#14): the output identity of every
   merged value, no producing merge named.
4. `record_field_layout_concordance` from `materialize_record_phis` (#69):
   minted field ids with no edge from the Phi inputs; the linker's record
   reconciliation reads it.
5. `record_field_decomposition` (#19): minted member ids per declared
   keyed field; `_linked_caller_member` and keyed-iterable resolution bind
   through it.
6. `record_storage_alias` + `record_field_resident_concordance` (#49-53):
   first-in-list residency rewriting every operand in the function.
7. `record_parameter_value` / `record_parameter_row_handle` (#35, #43):
   the roots of the forwarding graph; a name fallback at the root
   propagates as YES-edges downstream.
8. `call_record_pair_concordance` (#18): a string match certified as
   identity, consumed by member linking.
9. `value_shape` (#29): the shape fact does not cite the `call_edge` row
   that delivered it; `tensor_ssa_lowering` and the SSA seam consume it.
10. `proven_shape` from `ensure_structural_value` (#78): a
    broadcast-incompatible operand recorded as proven; 11 reader files.
    Close behind: `numeral_record_literal_concordance` (#58) recording a
    minted default as the field value, and `scheduled_call_argument` (#73).

Shadow ledgers that must move onto the book FIRST for the frame linker and
record materialization to have causal edges, in order:

1. `result_storage_bindings` (per call) with `identity_aliases` and
   `default_literals`: these are the inputs of #22; once they are rows,
   #22 can name them and the `else` mint can be recorded as unresolved.
2. `record_return_layouts` (metadata, 6 writers, read by two other
   modules): the returned-record ABI that #69, #70 and `allocate_result_storage`
   derive from without naming.
3. `caller_children` and the returned-record reconciliation channels
   (`singleton_call_result_concordance`, `nonreturn_result_bindings`,
   `returned_record_*`): the pairing decisions behind #18 and the linker's
   result rebinding.
4. `resident_by_value` / `kind_by_value` and the other sequence residency
   maps merged into #2, so planning residency is born with its source.
5. `parameter_names`: the name->id root every ABI pass re-reads (60+
   files); until it is a page, no record-ABI row can name its source.
6. `constructor_anchors` / `call_anchor_value_ids` /
   `constructor_instance_pools` and the constructor `remap`.

Not classified with confidence: the exact nested-block attribution of
some `_class_surface_ssa_program` sites (indentation-derived); whether the
eight private alias snapshots ever diverge from the page within one pass
(they are copies taken at different points of the fixed point, which the
static read cannot prove either way); and which of the 31 unread pages are
consumed by tests only versus by no one (tests were grepped by page name,
not read).
