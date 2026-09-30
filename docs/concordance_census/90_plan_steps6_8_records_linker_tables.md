# 90: Steps 6-8 edit plan -- record materialization, the frame linker, the book-backed tables

Read-only planning lane, 2026-09-30.  Companion to
`docs/CONCORDANCE_SINGLE_API_DESIGN_2026-09-30.md` (sections 2, 4 and 6 are
the api; steps 6, 7 and 8 of section 4 are this file), census files 10
(fortran_c_shell writers), 40 (reducer / graph_express2 / ssa.py tables) and
50 (stages E and F of the worked example), and the two executed plans 60
(step 2) and 70 (step 3), whose format this file follows.  Everything below
was read from the tree with `git grep`/`sed`/`awk`; nothing was run.  Code
is named by function.  No compiler numberings appear.

Files in scope: `src/compiler/fortran_c_shell.py` (`_class_surface_ssa_program`
and its nested definitions, `_publish_concorded_output_identities`,
`_concord_record_return_phi_inputs`, `_linked_caller_member`,
`_link_frame_lease`, `_complete_propagated_frame_tails`, the sequence
residency helpers), `src/compiler/ssa_record_return_state.py`,
`src/compiler/precompile_to_ssa.py` (one edit: the return-merge Phi in
`_ControlSSABuilder.finish`), `src/compiler/transformation_priority.py`,
`src/transmogrifier/ssa.py`, `src/compiler/identity_concordance.py` (the
api, `mint_scope`, the audit), `src/compiler/concordance_declarations.py`,
`tools/audit_identity_concordance.py`.

Vocabulary as in plan 70: "cell" = one `Ref`; "post" = one
`IdentityBook.post`; Page / Stage / Transform / Reason names in capitals are
registry OBJECTS declared in `concordance_declarations.py`, never strings.

## 0. What is on the tree at the start of step 6 (observed)

Landed (commit "Concordance steps 2-3"): `IdentityBook.post` with `Derived`
/ `Novel` / `Unsourced`, `Mode.CONCORD` / `Mode.REVISE`, the four private
pages (edge, dependents, mint, unsourced), the OPEN latch, raw-primitive
tagging in `IdentityPage.set` (every `revise`, `concord`, `bind_alias`,
`PageMapping` write bottoms out there), `Registry` with `declare_page` /
`declare_stage` / `declare_transform` / `declare_reason`, `Ref(page, row,
column)`, `RowFieldKind` with SCOPE / VALUE_ID / NAME / INDEX / LABEL /
PAGE_REF, `NEW` admissible only in a VALUE_ID field of a `Novel` post, the
two generic findings `unsourced-fact` / `unsourced-identity`, the step 2 and
step 3 pages, `node_identity_cell` in the reducer, `return_site_cell` on
return nodes, `SSA_FIELD_VERSION` posts in `_ControlSSABuilder`, and
`scalar_return_field_versions.decide`, which posts every exit on
`RECORD_RETURN_FIELD_SELECTION`.

Two dormant items lane C left for step 6, both read from the tree:

- **D1.** `materialize_record_phis` reads `phi_cell = attributes.get("identity_cell")`
  on the return-merge Phi and `scalar_return_field_versions.decide` returns
  without posting when `phi_cell` is not a `Ref`.  Nothing on the tree
  writes `identity_cell`: `git grep identity_cell` finds the two readers in
  `fortran_c_shell.py` and the probe.  The twelve return-version Reasons are
  declared and reachable, and every one of them is silent today because the
  row cannot be keyed.
- **D2.** The record-return `Cast` is still `SSAValue(GLOBAL_MONOTONIC_IDS.mint(), ...)`
  in both `select_return_arguments` and `publish_scalar_record_return_fields`,
  with a comment saying it should be NOVEL `RECORD_RETURN_FIELD_CONVERSION`
  and that "no declared page carries a VALUE_ID for an SSA Cast".  The
  transform is declared (arity 1); the page is not.

Also observed, load-bearing for all three steps:

- `_ControlSSABuilder.fresh_value` draws every SSA id from
  `GLOBAL_MONOTONIC_IDS.mint()`, so every control-builder value, including
  the return-merge Phi, carries the `MINTED` flag and is listed by
  `unsourced-identity` until a `Novel` post owns it.  Direct mint sites in
  `fortran_c_shell.py` today: 95 (`GLOBAL_MONOTONIC_IDS.mint()` plus
  `mint_compiler_value_id()`, which is a bare passthrough), one more in
  `deployment_ssa_binding.bind_deployment_dataflow`, three in
  `ssa_record_return_state` (`freshen_redefined_ssa_objects`,
  `repair_non_dominating_return_phi_inputs.cloneable`,
  `publish_scalar_record_return_fields`), and `precompile_to_ssa` (step 5).
- `IdentityBook.mint_scope` still writes `scope_registry` through
  `page.concord`, so every scope (`_normalize_lexical_values`, `fork_read_scope`,
  `ensure_node`'s ingestion scope, `mint_scope("record_field_access")`,
  `("record_storage_alias", symbol)`, `"scheduled_call_argument"`,
  `TransformationLedger`, `_mint_table_owner`) is a raw row on an
  undeclared page.  Design section 2 assigned this to step 1; it was not
  done.  It is E6.0 below because steps 6 and 7 need scope cells as sources.
- `IdentityBook._source_stamp` refuses a `Ref` whose page is not in the
  registry.  The book-backed table pages (`record_descriptor`,
  `record_member`, `sequence_descriptor`, `sequence_member`, `call_record`,
  `struct_*`, `union_*`, `layout_state`, `layout_supersession`,
  `sequence_column_claims`) and the ledger pages (`transformation_decision`,
  `transformation_event`, `transformation_rejection`) are undeclared.  Steps
  6 and 7 derive from their cells, so their DECLARATION is E6.0 too; routing
  their writers stays step 8.
- Nothing sets `IdentityBook.active_stage` and nothing sets
  `Latch.CLOSED`; the audit's worklist therefore lists every raw write under
  stage `raw_primitive`, unit `raw row` (unregistered page) or `cell`
  (registered page).
- The audit tool's `CASES` today: `view`, `toplevel`, `energy`,
  `controller`, `controller_untyped`, `mapping` (the six the design names)
  plus `oscillator`, added since.  `main` fails only on the first line's
  finding count; the `unsourced:` line is informational.

### 0.1 What steps 6-8 need from steps 1-5 (say so before step 6 starts)

- **N1. A function scope for every SSA function.**  Step 3 pages key SSA
  facts by the source graph's `lexical_read_scope` (see `selection_scope` in
  `materialize_record_phis` and `scope` in `scalar_return_field_versions`).
  A function with no source graph (a specialized copy, a control function
  built in `precompile_to_ssa`, a helper table) has none, and today those
  sites "decide as they always did and the book records nothing".  Step 6
  declares `FUNCTION_SCOPE` (section 1.2) so every function has exactly one
  scope row; a function whose graph has a `lexical_read_scope` derives the
  row from that scope's `scope_registry` cell (E6.0), a function without one
  gets a NOVEL scope minted through `post`.
- **N2. Variadic transforms.**  `Transform.arity` is checked exactly by
  `post`.  A return-merge Phi is made from N return edges; a record Phi's
  field from N incoming fields.  Either `Transform` admits `arity=None`
  (any count >= 1, a one-line change in the arity check) or the variable
  set becomes one DERIVED "set row" cell first (page `CELL_SET`, section
  1.2) and the transform takes arity 1.  Held for the user (section 12.1);
  the plan is written for the set row, which needs no api change.
- **N3. From step 5:** `ReturnBlock` (control_source) does not carry the
  return construct's `return_site_cell`; `scalar_return_field_versions.return_site_cell`
  recovers it by scanning graph nodes for a matching span.  Step 6 keeps
  using that helper until step 5 puts the cell on `ReturnBlock` and on the
  `Br`'s attributes beside `return_source_value_ids`.  Not a blocker.
- **N4. From step 4:** `record_return_layouts`' semantic-position writer in
  the linking loop derives from `source_output_value_ids` (planner
  metadata).  Until step 4 posts it, that derivation is `Unsourced(PLANNER_OUTPUT_UNROUTED)`.
- **N5. From step 2/3:** `name_binding` rows (the `identity_table` page)
  are the sources of `record_parameter_value` (parameter name -> ids) and
  of `output_identity_concordance` (output name histories).  Both exist.

## 1. The identity cell for compiler-minted SSA values (steps 6 and 7)

### 1.1 The rule

One page holds the identity of every SSA value the compiler mints after
reduction: `SSA_VALUE_IDENTITY`, row `(function_scope, value_id)`, fact
`SSAValueFact(op, kind)` where `op` is the defining instruction's op
(`"Phi"`, `"Cast"`, `"Const"`, `""` for a formal) and `kind` a declared
`SSAValueKind` (RETURN_MERGE, RECORD_FIELD_PHI, RECORD_RETURN_FIELD_CONVERSION,
PROGRAM_ABI_DEFAULT, OPTIONAL_PRESENCE, OPTIONAL_INACTIVE_PAYLOAD,
LOOP_RECORD_HEADER, LOOP_RECORD_PROJECTION, NESTED_RECORD_PART,
DECLARED_ROW_COLUMN, RESULT_STORAGE, FRAME_TAIL_SLOT, FRAME_STORAGE_CLONE,
REPLACEMENT_SLOT, CONSTRUCTOR_REMAP, FRAME_SCAFFOLD, STRUCTURAL_FOLD,
FRESHENED).  The row IS the key: the value id, in the function's scope.

- A value the book mints: `post(SSA_VALUE_IDENTITY, (scope, NEW), fact,
  provenance=Novel(transform, operands), mode=CONCORD)`; the minted id is
  `ref.row[1]`; the caller builds its `SSAValue(ref.row[1], ...)` from it.
  Every `GLOBAL_MONOTONIC_IDS.mint()` in scope is replaced by exactly this.
- A value minted elsewhere (the return-merge Phi from `fresh_value`, step
  5's builder): the same row posted DERIVED from the cells it was made
  from; no NEW.  It remains listed by `unsourced-identity` until step 5
  routes `fresh_value` through `post`; the row exists so step 6's selection
  rows can be keyed today.
- The cell is stamped `instruction.attributes["identity_cell"] = ref` AND
  `value.accounting["identity_cell"] = ref` at the mint.  Neither is the
  key; both are caches.  Readers resolve through one helper,
  `ssa_value_identity_cell(function, value_id) -> Ref | None` (new,
  `identity_concordance.py`): `book.latest_ref(SSA_VALUE_IDENTITY,
  (function_scope_of(function), value_id))`, falling back to the caches only
  when the row is absent.  `materialize_record_phis`' `phi_cell =
  attributes.get("identity_cell")` becomes that call, which is what wakes
  D1: later frame rounds rebuild instructions and drop attributes (the
  comment on `record_field_phi` says so), but the row survives.
- `Ref` pickles by value; `Page` is a frozen dataclass compared by equality
  in `post`, so a cell that travelled through `metadata` pickling still
  validates against the registry.  A cell naming a book from another
  process fails `_source_stamp` ("source cell does not exist"): that is a
  refusal at the site, never a silent miss (risk R6.5).

### 1.2 Pages declared for the three steps

New declarations (`concordance_declarations.py`, a new "Steps 6-8" section):

| page | row fields (kind) | fact type | mode | step |
|---|---|---|---|---|
| `SCOPE_REGISTRY` (`scope_registry`, re-declared) | label (LABEL), serial (INDEX) | `ScopeFact(kind)` | CONCORD; NOVEL(MINT_SCOPE, ()) -- a root row, no NEW | 6 (E6.0) |
| `FUNCTION_SCOPE` | function (NAME) | tuple (the scope) | CONCORD; DERIVED(scope_registry cell) or NOVEL(MINT_SCOPE, ()) | 6 |
| `CELL_SET` | function_scope (SCOPE), label (LABEL), serial (INDEX) | tuple of Ref | CONCORD; DERIVED(every member cell) | 6 (N2) |
| `SSA_VALUE_IDENTITY` | function_scope (SCOPE), value_id (VALUE_ID) | `SSAValueFact` | CONCORD; NOVEL(transform, operands) or DERIVED | 6, 7 |
| `RECORD_RETURN_LAYOUT` | function_scope (SCOPE), record_id (VALUE_ID) | tuple of int (physical layout) | REVISE; DERIVED(record_descriptor cell, each layout value's identity cell) | 6 |
| `RECORD_PHI_EXPANSION` | function_scope (SCOPE), record_phi (VALUE_ID), field (NAME), slot (INDEX) | Ref (the field Phi's identity cell) | CONCORD; DERIVED(record Phi identity cell, each incoming record_member cell) | 6 |
| `RECORD_DESCRIPTOR_MERGE` | owner (SCOPE), record_id (VALUE_ID), revision (INDEX) | `RecordMergeFact(incumbent, incoming, merged, widened_fields, adopted_pool)` | CONCORD; DERIVED(incumbent record_descriptor cell, incoming source cell) | 6 (page), 8 (writer) |
| `RECORD_FIELD_LAYOUT` (`record_field_layout_concordance`, re-declared) | function_scope (SCOPE), record_id (VALUE_ID), storage_identity (LABEL) | layout tuple | CONCORD; DERIVED | 6 |
| `LOOP_RECORD_LAYOUT` (re-declared) | function_scope (SCOPE), loop_node (VALUE_ID), result (VALUE_ID) | (old layout, new layout) | CONCORD; DERIVED | 6 |
| `LOOP_RECORD_SCHEMA` (re-declared) | function_scope (SCOPE), loop_node (VALUE_ID), result (VALUE_ID) | `LoopSchemaFact(initial, updated, projected, discarded, projected_record_cell)` | CONCORD; DERIVED | 6 |
| `RECORD_FIELD_DECOMPOSITION` (re-declared) | owner (SCOPE), record_id (VALUE_ID), storage_identity (LABEL) | tuple of (role path, member id) | CONCORD; DERIVED(class_field_declaration or program-abi field cell, each part's identity cell) | 6 |
| `PROGRAM_ABI_KEYED_ROW_RECORD` (re-declared) | owner (SCOPE), record_id (VALUE_ID), storage_identity (LABEL) | int | CONCORD; DERIVED | 6 |
| `NUMERAL_RECORD_LITERAL` (re-declared) | function_scope (SCOPE), node_or_record (VALUE_ID), field_or_status (LABEL) | object | CONCORD; DERIVED / Unresolved | 6 |
| `NUMERAL_LEAF` (folds `numeral_leaf_materialization_concordance`, `numeral_leaf_width_concordance`, `numeral_return_leaves_concordance`) | function_scope (SCOPE), owner (LABEL), path (LABEL) | `NumeralLeafFact(status, width, leaves)` | REVISE; DERIVED(source_numeric_record_abi cell) | 6 |
| `RECORD_FIELD_RESIDENT` (`record_field_resident_concordance`, re-declared) | function_scope (SCOPE), parameter_root (VALUE_ID), field (NAME) | int or Unresolved | CONCORD; DERIVED(formal identity cell, each read's identity cell) | 6 (page), 7 (fallback) |
| `OUTPUT_IDENTITY` (`output_identity_concordance`, re-declared) | function (NAME), alias (VALUE_ID) | int | REVISE; DERIVED(name_binding history cells, result identity cell) | 6 |
| `RECORD_RETURN_PHI_INPUT` (re-declared) | function_scope (SCOPE), phi (PAGE_REF identity cell), field (NAME), position (INDEX), predecessor (LABEL) | (candidate, chosen, reason) | REVISE; DERIVED(selection cell) | 6 |
| `RECORD_FIELD_STORAGE` (re-declared) | function_scope (SCOPE), record_id (VALUE_ID), storage_identity (LABEL) | `StorageRefinement(from, to, sequence_id, record_id)` | REVISE; DERIVED(incumbent layout cell, callee descriptor cell) | 6 |
| `RESULT_STORAGE_BINDING` | caller_scope (SCOPE), callsite (LABEL), callee_value (VALUE_ID) | Ref (caller storage identity cell) | CONCORD; DERIVED | 7 |
| `CALL_BINDING_INPUT` | caller_scope (SCOPE), callsite (LABEL), callee_value (VALUE_ID), source (LABEL: identity_alias / default_literal / discovery) | object | CONCORD; DERIVED | 7 |
| `ARGUMENT_BINDING` (re-declared) | callee (NAME), callee_value (VALUE_ID), callsite (LABEL) | `BindingFact(kind, source)` or Unresolved | CONCORD; DERIVED | 7 |
| `FRAME_LEASE` (`frame_lease_link`, re-declared) | caller (NAME), slot (VALUE_ID) | (callsite, callee, formal) | CONCORD; DERIVED(slot identity cell, callee formal cell) | 7 |
| `FRAME_TAIL` (`propagated_frame_tail_concordance`, re-declared) | owner (NAME), callsite (LABEL), callee (NAME), formal (VALUE_ID) | (slot, kind) | CONCORD; DERIVED(argument_binding cell) or Unresolved | 7 |
| `RECORD_STORAGE_ALIAS` (re-declared) | function_scope (SCOPE), value (VALUE_ID) | int | REVISE; DERIVED(record_field_resident cell) | 7 |
| `RECORD_PARAMETER_VALUE` (re-declared) | scope (SCOPE), (symbol, value) (LABEL) | (symbol, parameter) | CONCORD; DERIVED(name_binding cell, parameter_annotation / abi record cell) | 7 |
| `RECORD_PARAMETER_ROW_HANDLE` (re-declared) | scope (SCOPE), (symbol, value) (LABEL) | (key, path prefix, row identity) or Unresolved | CONCORD; DERIVED | 7 |
| `CALL_RECORD_PAIR` (re-declared) | caller (NAME), callsite (LABEL), callee_child (VALUE_ID) | int | CONCORD; DERIVED(bound pair cell, both fields' record_member cells) | 7 |
| `LINKED_CALLER_MEMBER` | caller (NAME), callsite (LABEL), callee (NAME), formal (VALUE_ID) | int or Unresolved | CONCORD; DERIVED | 7 |
| `SEQUENCE_RESIDENCY` | function (NAME), value (VALUE_ID) | `ResidencyFact(resident, kind, helper)` | REVISE; DERIVED | 7 |
| `PLANNING_VALUE` (`planning_value_concordance`, re-declared) | function (NAME), alias (VALUE_ID) | int or None | REVISE; DERIVED | 7 |
| `PLANNING_ALIAS_TRANSITION`, `ALIAS_APPLICATION`, `SCHEDULED_CALL_ARGUMENT`, `CALL_LINK_ORDER`, `KERNEL_BY_VALUE_FORMAL`, `PHI_EDGE_PROJECTION_PLACEMENT`, `PRUNED_CALLEE_FORMAL`, `ENTRY_RECORD_HANDLE`, `MEMBER_FORMALS` (all re-declared with their present row shapes) | as today | as today | as today's mode | 7 |
| `TRANSFORMATION_DECISION` / `_EVENT` / `_REJECTION` (re-declared) | as today | as today | REVISE / CONCORD | 6 (declare), 7 (route) |
| `RECORD_DESCRIPTOR`, `RECORD_MEMBER`, `SEQUENCE_DESCRIPTOR`, `SEQUENCE_MEMBER`, `STRUCT_DESCRIPTOR`, `STRUCT_MEMBER`, `UNION_DESCRIPTOR`, `UNION_MEMBER`, `LAYOUT_STATE`, `LAYOUT_SUPERSESSION`, `SEQUENCE_COLUMN_CLAIMS`, `CALL_RECORD` (re-declared) | as today | as today | REVISE | 6 (declare), 8 (route) |
| `TABLE_OWNER` | owner (SCOPE) | `TableOwnerFact(kind, function)` | CONCORD; DERIVED(function_scope cell) | 8 |

Stages: `RECORD_LITERAL_MATERIALIZATION`, `RECORD_PHI_EXPANSION_STAGE`,
`RECORD_RETURN_LAYOUT_STAGE`, `RECORD_ABI_MATERIALIZATION`,
`STRUCTURAL_RECOVERY`, `OUTPUT_IDENTITY_STAGE` (step 6);
`FRAME_BINDING`, `FRAME_LINK`, `FRAME_TAIL_STAGE`, `RECORD_FORWARDING`,
`PLANNING_RESIDENCY`, `SHELL_HANDOFF` (step 7); `TABLE_REGISTRATION`,
`SCOPE_MINT` (step 8; `SCOPE_MINT` is used by E6.0).

Transforms (arity): `MINT_SCOPE` (0), `RECORD_FIELD_PHI` (1: the
`RECORD_PHI_EXPANSION` row cell, which names the record Phi and the
incoming members), `RETURN_MERGE` (1: a `CELL_SET` of the return-edge slot
cells), `RECORD_RETURN_FIELD_CONVERSION` (1, declared), `PROGRAM_ABI_DEFAULT`
(1: the abi record field's declaration cell), `OPTIONAL_PRESENCE` (1),
`OPTIONAL_INACTIVE_PAYLOAD` (1), `LOOP_RECORD_HEADER` (1: the
`LOOP_RECORD_SCHEMA` or `LOOP_RECORD_LAYOUT` row), `LOOP_RECORD_PROJECTION`
(1), `NESTED_RECORD_PART` (1: the declared field cell), `DECLARED_ROW_COLUMN`
(1), `RESULT_STORAGE_LEASE` (1: the callee value's identity cell),
`FRAME_TAIL_SLOT` (1: the callee formal's identity cell),
`FRAME_STORAGE_CLONE` (1), `REPLACEMENT_SLOT` (1: the ledger decision cell),
`CONSTRUCTOR_REMAP` (1), `FRAME_SCAFFOLD` (1: the value it addresses or
sizes), `STRUCTURAL_FOLD` (1: a `CELL_SET` of the fold operands),
`FRESHEN` (1: the redefined value's identity cell), `TABLE_OWNER_SCOPE` (1:
the function_scope cell).

Reasons: `PLANNER_OUTPUT_UNROUTED`, `LAYOUT_MEMBER_NOT_YET_DEFINED`,
`LITERAL_FIELD_DEFERRED`, `RESIDENT_CHOSEN_BY_ORDER`,
`STORAGE_MINTED_FROM_ABSENCE`, `ROW_IDENTITY_FROM_SCHEMA_NAME`,
`FORMAL_UNBOUND_AT_TAIL`, `MEMBER_NOT_BOUND_AT_CALL`,
`AGGREGATE_POSITION_MISSING`, `RESIDENCY_FROM_NAME_MATCH`,
`NO_FUNCTION_SCOPE`, `SEQUENCE_CLAIM_WITHOUT_PROPOSER`.

## 2. Step 6: record materialization and return versions

### 2.1 What is being moved, observed

| writer (function) | what it writes today | mode today | edge today |
|---|---|---|---|
| `_ControlSSABuilder.finish` | the return-merge `Phi` (`binding == "return_merge"`, `return_slot_index`, `incoming_blocks`), result from `fresh_value` | SSA instruction | none; no identity cell (D1) |
| `materialize_record_phis` | per-field `Phi` per record Phi slot: `SSAValue(GLOBAL_MONOTONIC_IDS.mint(), accounting={record_phi, record_field, record_field_slot})`, attributes `record_field_phi`, `initial_value_id`, `record_return_scalar`, `record_return_receivers` | mint + instruction | none (S18 absent) |
| same | `record_field_layout_concordance` `(symbol, result id, storage identity) -> layout` | `set` col 0, raise on differ | none |
| same (loop Phi branch) | `loop_record_layout_concordance` `(symbol, loop node, result) -> (old, new, "record_loop_phi")`; metadata `loop_record_layout_transitions` | `set` col 0 | none |
| same | `table.records[result_id] = merged_descriptor` (loop) or `table.register(merged_descriptor)`; `function.metadata["record_return_layouts"]` | `_BookRows` revise; metadata | none |
| `select_return_arguments` (inner) | the `Cast` (`GLOBAL_MONOTONIC_IDS.mint()`, attributes `record_return_field_conversion`, `source_field_value_id`); `Unsourced(RECORD_DESCRIPTORS_DIFFER)` on `RECORD_RETURN_FIELD_SELECTION` when the descriptors differ | mint; post | D2 |
| `scalar_return_field_versions.decide` | `RECORD_RETURN_FIELD_SELECTION` row `(scope, phi_cell, field, position, predecessor)`; `Unsourced(reason)` when `read` is empty | post | keyed only when `phi_cell` is a Ref (never today) |
| `_concord_record_return_phi_inputs` | `record_return_phi_input_concordance` `(function, result id, field, position, predecessor) -> (candidate, chosen, reason[, selection key])` | `set` next column; raises unless the selection cell is newer | PARTIAL (the key is appended to the fact, not an edge) |
| `publish_scalar_record_return_fields` | the same `Cast` mint; rewrites `operation.args` and `initial_value_id` | mint; in place | none |
| `materialize_loop_record_phis` | `loop_record_schema_concordance` `(symbol, loop, result) -> (initial sig, updated sig, projected, discarded, "project_updated_to_initial")`; `projected_updated_id = mint()`; `header_record_id = mint()`; metadata `loop_record_schema_projections`, `loop_record_phi_materializations` | `set` col 0; mints | none |
| `materialize_program_abi_record_literals` | default field value `value_id = GLOBAL_MONOTONIC_IDS.mint()` with `program_abi_default` accounting; presence / inactive payload mints; `numeral_record_literal_concordance` `(symbol, node, "deferred"/"completed")`; `table.register`; `record_return_layouts` (append) | mints; `concord`; metadata | none; the default is recorded as the field value |
| `materialize_parameter_record_abi.materialize_nested_record` (about 17 mints), `materialize_declared_row_columns` (about 20), `publish_keyed_decomposition`, the keyed value-record branch | column arenas, lengths, strides, pointers, pooled columns, `part_id`, `candidates = (mint(),)` after "required_unread_leaf"; `record_field_decomposition` `(owner, record, storage identity) -> roles`; `program_abi_keyed_row_record`; `numeral_leaf_materialization_concordance`, `numeral_leaf_width_concordance` | mints; `concord` | none |
| `coalesce_record_field_storage` | `record_field_resident_concordance` `(symbol, min(parameter ids), field) -> resident`, resident = the formal among the reads else `read_ids[0]` | `concord` | none; first-in-list from absence |
| `recover_structural_source_outputs` | `record_return_layouts` (`returned_record_layouts`); `output_identity_aliases` -> `_publish_concorded_output_identities` | metadata; `bind_alias` | none |
| `_publish_concorded_output_identities` | `output_identity_concordance` `(function, alias) -> result`; metadata twin `output_identity_aliases` | `bind_alias` when no incumbent; raises on differ | none |
| `complete_linked_literals` | re-runs the literal materialization for deferred numerals; `numeral_return_leaves_concordance`; `record_return_layouts` (leaves) | `concord`; metadata | none |
| linking loop, record alias block (nested in the `while changed` frame loop, after `insert_at_loop_anchor`) | `record_return_layouts[record] = layout`, and under `positional_semantics` also `layouts[semantic_record_id]` + `register` of a twin descriptor | metadata; register | none |
| post-frame pass over `all_record_tables` (after `final_frame_value`) | `record_return_layouts` from Ret expansion | metadata | none |
| `publish_inout_scalar_return_snapshots` (`ssa_record_return_state`) | `record_return_layouts` re-sliced from old/new returns; `table.records[...] = replace(...)` | metadata; revise | none |
| linking loop `resident_result_slot` region | `record_field_layout_concordance` from the caller's incumbent field; `record_field_storage_concordance` `{"from","to","sequence_id","record_id"}` | `set` col 0; `set` next column | none |
| `SSARecordTable.register` | complementary-view merge: `writable = resident or incoming`, new fields appended, `instance_pool = existing or incoming`; assigned through `_BookRows` | revise | none; census 40 section 4.2 |

Seven `record_return_layouts` writers (the census's six plus
`publish_inout_scalar_return_snapshots`), eleven readers here plus
`project_compilation_product` and `symbolic_fluid_native_runtime`.

### 2.2 The identity cell for the return-merge Phi (D1)

`_ControlSSABuilder.finish`, after `merged = self.fresh_value(dtype=dtype)`
and before `self.emit(Handler.Phi, ...)`:

1. For each `(block, values)` in `edges` at this slot, the source cell is the
   `RETURN_SITE_SLOT` row `(reduction scope, return_site_cell, slot)`.  The
   site cell comes from the `ReturnBlock` (N3) or, until step 5 lands it,
   from the graph node whose `return_site_cell` attribute the reducer
   stamped (the same lookup `scalar_return_field_versions.return_site_cell`
   performs).  A slot whose site row is `Unresolved(RETURN_SLOT_NOT_A_VALUE)`
   contributes that cell; the Phi's row is then posted DERIVED all the same
   (the edge names the unresolved slot).
2. `edge_set = post(CELL_SET, (scope, "return_merge_edges", NEW-less serial),
   tuple(site cells), Derived(site cells), stage CONTROL_SSA, CONCORD)`.
3. `post(SSA_VALUE_IDENTITY, (scope, int(merged.id)), SSAValueFact("Phi",
   RETURN_MERGE), Derived((edge_set,)), stage CONTROL_SSA, CONCORD)`; stamp
   `attributes["identity_cell"]` and `merged.accounting["identity_cell"]`.
   (When step 5 routes `fresh_value` through `post`, this becomes
   `Novel(RETURN_MERGE, (edge_set,))` with NEW; nothing else changes.)
4. `scope` is `self.lexical_read_scope` (already held by the builder for
   `SSA_FIELD_VERSION`); when None, `FUNCTION_SCOPE` (N1) supplies it.

Effect: `materialize_record_phis` reaches `phi_cell` through
`ssa_value_identity_cell`, `selection_row` is non-None, and every exit of
`lookup` posts its Reason on `RECORD_RETURN_FIELD_SELECTION`.  The twelve
Reasons wake without a change to `decide`.

### 2.3 Posts, writer by writer

| writer | post | provenance |
|---|---|---|
| `materialize_record_phis`, per-field Phi (S18) | (a) `RECORD_PHI_EXPANSION` row `(scope, record Phi id, field, slot)`, fact = the field Phi cell (posted after b, so a and b are one transaction: post b with a provisional `CELL_SET` of the incoming members first, then a); (b) `SSA_VALUE_IDENTITY` `(scope, NEW)` fact `SSAValueFact("Phi", RECORD_FIELD_PHI)` | (b) NOVEL(RECORD_FIELD_PHI, (the `CELL_SET` of: the record Phi's identity cell, each incoming record's `record_member` cell for `(candidate.value_ids[slot])`)); (a) DERIVED(same cells).  The instruction's `initial_value_id`, `record_return_receivers` stay as caches |
| same, merged descriptor | `RECORD_FIELD_LAYOUT` row `(scope, result id, storage identity)` fact = layout | DERIVED(each merged id's identity cell, the incoming fields' `record_member` cells).  `Mode.CONCORD` keeps today's raise on a differing layout |
| same, loop Phi branch | `LOOP_RECORD_LAYOUT` fact `(old layout, new layout)` | DERIVED(the incumbent `record_descriptor` cell, the merged ids' identity cells); metadata `loop_record_layout_transitions` becomes a read view |
| same, `table.register(merged_descriptor)` / `table.records[result_id] = ...` | unchanged calls; `register(descriptor, source=<RECORD_PHI_EXPANSION cells>)` once step 8 adds the keyword (section 4.3) | -- |
| same, `record_return_layouts` | `RECORD_RETURN_LAYOUT` row `(scope, result id)` fact = the layout tuple, REVISE | DERIVED(the `record_descriptor` cell of the result, each layout id's identity cell) stage RECORD_PHI_EXPANSION_STAGE |
| `select_return_arguments`, the Cast (D2, S20) | `SSA_VALUE_IDENTITY` `(scope, NEW)` fact `SSAValueFact("Cast", RECORD_RETURN_FIELD_CONVERSION)`; the `SSAValue` is built from `ref.row[1]` | NOVEL(RECORD_RETURN_FIELD_CONVERSION, (the selection cell for this `(phi_cell, field, position, predecessor)`,)).  When `selection_cell(...)` is None (no scope) the Cast is not minted and the argument stands; the absence is already a `NO_FUNCTION_SCOPE` unresolved on `FUNCTION_SCOPE` (N1) |
| same, descriptors differ | the existing `Unresolved(RECORD_DESCRIPTORS_DIFFER, read=())` becomes `read=(source record_descriptor cell, physical record_descriptor cell)` | DERIVED(those two cells) instead of `Unsourced` |
| `scalar_return_field_versions.decide`, `read` empty (`NO_RETURN_FIELD_RECEIPTS`, `PREDECESSOR_BLOCK_EMPTY`, `PREDECESSOR_NOT_A_RETURN_EDGE`) | `Unresolved(reason, read=(phi_cell,))` | DERIVED((phi_cell,)); the Phi's identity cell is what was read.  `Unsourced` leaves this function |
| `_concord_record_return_phi_inputs` (S21) | `RECORD_RETURN_PHI_INPUT` row `(scope, phi_cell, field, position, predecessor)` fact `(candidate, chosen, reason)`, REVISE | DERIVED(the selection cell; the chosen value's identity cell).  The api's REVISE check replaces the hand-written stamp comparison: a changed choice with no changed source is refused by `post`, so the `ValueError` branch becomes `except ConcordanceRefusal: raise ValueError(...)` with the same message |
| `publish_scalar_record_return_fields` | the same Cast post; the `lookup` call passes `phi_cell` and `position` so its exits post; `operation.args` rewrite unchanged | as above |
| `materialize_loop_record_phis` | `LOOP_RECORD_SCHEMA` fact gains the projected record's identity cell; `projected_updated_id` and `header_record_id` become `SSA_VALUE_IDENTITY` NOVEL posts (`LOOP_RECORD_PROJECTION` operand = the schema row cell; `LOOP_RECORD_HEADER` operand = the `LOOP_RECORD_LAYOUT` or schema cell); `loop_record_schema_projections` and `loop_record_phi_materializations` metadata become read views | DERIVED(initial and updated `record_descriptor` cells, the `output_identity` cells that paired them) |
| `materialize_program_abi_record_literals`, default field | `SSA_VALUE_IDENTITY` `(scope, NEW)` fact `SSAValueFact("Const", PROGRAM_ABI_DEFAULT)` | NOVEL(PROGRAM_ABI_DEFAULT, (the abi record field's declaration cell: `class_field_declaration` when the record is authored, else the `contract_demand` PARAMETER_RECORD row from step 2,)).  The `program_abi_default` accounting stays as the cache |
| same, presence / inactive payload | two NOVEL posts, `OPTIONAL_PRESENCE` / `OPTIONAL_INACTIVE_PAYLOAD`, operand = the payload's identity cell | -- |
| same, "deferred" / "completed" | `NUMERAL_RECORD_LITERAL` `(scope, node, "deferred")` fact `Unresolved(LITERAL_FIELD_DEFERRED, read=(the declaration cells of the missing fields,))`; `"completed"` fact = the field tuple DERIVED(each field value's identity cell) | the deferral is an Unresolved, never `True` |
| same, `record_return_layouts` | `RECORD_RETURN_LAYOUT` REVISE | DERIVED(the new `record_descriptor` cell, each layout id's identity cell) stage RECORD_LITERAL_MATERIALIZATION |
| `materialize_nested_record`, `materialize_declared_row_columns` | every mint -> `SSA_VALUE_IDENTITY` NOVEL (`NESTED_RECORD_PART` / `DECLARED_ROW_COLUMN`, operand = the declared field's cell); `candidates = (mint(),)` after "required_unread_leaf" -> the same NOVEL plus `NUMERAL_LEAF` fact `NumeralLeafFact("required_unread_leaf", width, ())` REVISE DERIVED(the `source_numeric_record_abi_concordance` cell); `"declared_row_leaf_deferred:<storage>"` -> `Unresolved(LITERAL_FIELD_DEFERRED, ...)` | -- |
| `publish_keyed_decomposition` | `RECORD_FIELD_DECOMPOSITION` fact = roles | DERIVED(the declared field cell, the identity cells of `keys`, `values`, `length`) |
| keyed value-record branch | `PROGRAM_ABI_KEYED_ROW_RECORD` fact = row record id | DERIVED(the row record's `record_descriptor` cell, the declared field cell) |
| `coalesce_record_field_storage` | `RECORD_FIELD_RESIDENT` row `(scope, min(parameter ids), field)`: when the resident is a formal, fact = its id DERIVED(the formal's identity cell, each read's identity cell); when it is `read_ids[0]` / `candidates[0]`, fact `Unresolved(RESIDENT_CHOSEN_BY_ORDER, read=(each read's identity cell,))` and the code keeps using `read_ids[0]` as the working resident | the row key `min(parameter_ids)` (id ORDER as identity) stays in step 6; step 7 replaces it with the parameter's `record_parameter_value` cell |
| `recover_structural_source_outputs` | `RECORD_RETURN_LAYOUT` REVISE for `returned_record_layouts` | DERIVED(the returned record's `record_descriptor` cell, the layout ids' identity cells) stage STRUCTURAL_RECOVERY |
| `_publish_concorded_output_identities` | `OUTPUT_IDENTITY` row `(function, alias)` fact = result id, REVISE | DERIVED(the `name_binding` cells of `identities[output_name]` (step 2's page) for the alias, the result's identity cell).  The incumbent / metadata disagreement raises stay; the metadata twin becomes a read view |
| `complete_linked_literals` | `NUMERAL_LEAF` `(scope, returned argument, "return_leaves")` fact leaves DERIVED(each leaf's identity cell, the numeric record abi cell); `RECORD_RETURN_LAYOUT` REVISE for the leaves | -- |
| linking loop, record alias block | `RECORD_RETURN_LAYOUT` REVISE DERIVED(record_descriptor cell, layout ids); the `positional_semantics` twin row DERIVED(`source_output_value_ids`' cell when step 4 has posted it, else `Unsourced(PLANNER_OUTPUT_UNROUTED)`) | N4 |
| post-frame `all_record_tables` pass; `publish_inout_scalar_return_snapshots` | `RECORD_RETURN_LAYOUT` REVISE | DERIVED(record_descriptor cell, the new layout ids' identity cells); the in-place `table.records[...] = replace(...)` goes through `register(..., source=...)` |
| linking `resident_result_slot` region | `RECORD_FIELD_LAYOUT` CONCORD DERIVED(the caller's incumbent `record_descriptor` cell, the caller field's identity cells); `RECORD_FIELD_STORAGE` REVISE DERIVED(the incumbent layout cell, the callee `record_descriptor` cell) | the "one-way type refinement" is now an edge from the callee row that proved it |
| `SSARecordTable.register`, complementary merge | `RECORD_DESCRIPTOR_MERGE` row `(owner, record id, revision)` fact `RecordMergeFact(...)` | DERIVED(the incumbent `record_descriptor` cell, `source` when the caller passed one, else the incoming descriptor's fields' `record_member` cells).  Written in step 8 (section 4.3); the page is declared here so step 6's `register` callers can pass `source=` |

`function.metadata["record_return_layouts"]` becomes a read view built from
`RECORD_RETURN_LAYOUT.scope_rows(function scope)` (latest per record), so
the eleven readers here and the two other modules do not change in step 6.
Same for `output_identity_aliases`, `loop_record_layout_transitions`,
`loop_record_schema_projections`, `loop_record_phi_materializations`,
`record_return_state_receipts`.  The read views are read-only mappings
(plan 70, risk 6): a stray write raises at its site.

### 2.4 Readers and read views

| reader | reads today | reads after step 6 |
|---|---|---|
| `_reconcile_post_aggregate_record_results`, `emit_outputs`, `mapping_slots`, `insert_at_loop_anchor` region, `final_frame_value` region, `project_compilation_product`, `symbolic_fluid_native_runtime` | `metadata["record_return_layouts"]` | the read view (unchanged code) |
| `materialize_record_phis` return-scalar branch | `attributes.get("identity_cell")` | `ssa_value_identity_cell(function, result id)` |
| `_concord_record_return_phi_inputs` | `selection_cells` passed by the caller; `page.stamps` | `RECORD_RETURN_FIELD_SELECTION` latest per position (unchanged), REVISE through `post` |
| linking loop record reconciliation (two `latest` sites) | `record_field_layout_concordance` | same page, same rows (re-declared, not re-keyed: the function name stays the SCOPE element for these rows until step 7's `FUNCTION_SCOPE` swap, which is one search-and-replace of the row builder) |
| `_linked_caller_member`, `resolve_keyed_mapping_iterables` | `record_field_decomposition` | same rows |
| `complete_linked_literals` | `numeral_record_literal_concordance` `rows()` scan by `row[0]`/`row[2]` | `scope_rows(function scope)` filtered by the status element; the `rows()` scan is the read-side hazard census 10 section 4 names |
| `_retire_concorded_record_identity_aliases`, `_concordant_function_aliases(include_output_identities=True)`, `materialize_loop_record_phis` | `output_identity_concordance` | same rows |

### 2.5 Ordered edit list (step 6)

E6.0  `identity_concordance.py`: `IdentityBook.mint_scope` posts
      `SCOPE_REGISTRY` `(label, serial)` NOVEL(MINT_SCOPE, ()) at stage
      `SCOPE_MINT` (a root row, mints nothing; `scope_row_count` still
      numbers it).  `concordance_declarations.py`: declare every page of
      section 1.2 (all three steps), the stages, transforms, reasons and
      fact types.  Declaring a page changes no writer; raw writes on a
      declared page are still admitted under the OPEN latch and appear as
      `cell` units instead of `raw row` units in the worklist.
E6.1  `identity_concordance.py`: `function_scope_of(function, graph=None)`
      (posts `FUNCTION_SCOPE` once per function: DERIVED from the graph's
      `lexical_read_scope` registry cell, else NOVEL(MINT_SCOPE)),
      `ssa_value_identity_cell(function, value_id)`, `post_cell_set(scope,
      label, cells)`.
E6.2  `precompile_to_ssa._ControlSSABuilder.finish`: section 2.2 (the
      return-merge Phi's `CELL_SET` and `SSA_VALUE_IDENTITY` row, stamped
      as `identity_cell`).
E6.3  `materialize_record_phis`: `phi_cell` via `ssa_value_identity_cell`;
      per-field Phi NOVEL + `RECORD_PHI_EXPANSION`; `RECORD_FIELD_LAYOUT`,
      `LOOP_RECORD_LAYOUT`, `RECORD_RETURN_LAYOUT` posts; `select_return_arguments`
      Cast NOVEL and the DERIVED `RECORD_DESCRIPTORS_DIFFER`.
E6.4  `ssa_record_return_state.scalar_return_field_versions.decide`: the
      three empty-`read` exits derive from `(phi_cell,)`.
      `publish_scalar_record_return_fields`: pass `phi_cell` / `position`
      to `lookup`; Cast NOVEL.  `publish_inout_scalar_return_snapshots`:
      `RECORD_RETURN_LAYOUT` posts; `register(..., source=)`.
E6.5  `_concord_record_return_phi_inputs`: `RECORD_RETURN_PHI_INPUT`
      REVISE posts DERIVED(selection cell, chosen identity cell); delete the
      hand-written stamp check.
E6.6  `materialize_loop_record_phis`: `LOOP_RECORD_SCHEMA` post; the two
      mints NOVEL; metadata read views.
E6.7  `materialize_program_abi_record_literals`: default / presence /
      payload NOVEL; `NUMERAL_RECORD_LITERAL` posts (deferred as
      Unresolved); `RECORD_RETURN_LAYOUT`.
E6.8  `materialize_parameter_record_abi`: `materialize_nested_record`,
      `materialize_declared_row_columns`, `publish_keyed_decomposition`, the
      keyed value-record branch, `coalesce_record_field_storage`'s
      `RECORD_FIELD_RESIDENT` (Unresolved on order); `NUMERAL_LEAF` folds the
      three numeral pages.
E6.9  `recover_structural_source_outputs` and `_publish_concorded_output_identities`:
      `RECORD_RETURN_LAYOUT`, `OUTPUT_IDENTITY` DERIVED(name_binding cells).
E6.10 `complete_linked_literals`: `scope_rows` reads; `NUMERAL_LEAF`
      return leaves; `RECORD_RETURN_LAYOUT`.
E6.11 The two linking-loop layout writers and the post-frame pass:
      `RECORD_RETURN_LAYOUT`; the `resident_result_slot` region's
      `RECORD_FIELD_LAYOUT` / `RECORD_FIELD_STORAGE` posts.
E6.12 `metadata["record_return_layouts"]` and the five sibling channels
      become read views (installed where each function is first seen in
      `_class_surface_ssa_program`; `project_compilation_product` and
      `symbolic_fluid_native_runtime` read the view unchanged).
E6.13 `tools/compiler_probes/probe_branch_written_field.py`: link 6 asserts
      instead of printing "none" (section 2.8).

Writer sites routed: 31 (E6.0 1, E6.2 1, E6.3 6, E6.4 4, E6.5 1, E6.6 3,
E6.7 5, E6.8 8, E6.9 2, E6.10 2, E6.11 4; counting each distinct
`set`/`concord`/`bind_alias`/mint/metadata write replaced).  Mint sites
routed: about 50 (`materialize_nested_record` 16-17, `materialize_declared_row_columns`
about 20, `abi_record_for_call` region 2, `select_return_arguments` 2,
`materialize_record_phis` 1, `materialize_loop_record_phis` 2,
`publish_scalar_record_return_fields` 1, plus the default / presence /
payload mints; counts by enclosing def, read from the tree today).

### 2.6 Risks (step 6)

R6.1  **The Cast becomes bookkept and the selection rows wake at once.**
      With `phi_cell` present every one of `lookup`'s exits posts; the
      REVISE admission check is now real.  A selection that changes between
      rounds because the receipt cell changed is admitted; one that changes
      with the same sources is refused by `post` and surfaces as the
      `ValueError` the 2026-09-30 raise produced -- correctly, because that
      case is a missing edge.  The probe (2.8) must be green before the
      user launches the dt-system lowering.  This is the top risk.
R6.2  **Order within `materialize_record_phis`**: the per-field Phi's
      NOVEL operands are `record_member` cells of the incoming records; a
      member "defined by a call this round" that is not yet in `values`
      already breaks with `members_pending`.  The post must sit after that
      check, never before, or a refusal ("source cell does not exist")
      replaces today's deferral.
R6.3  **`register(source=)` from step 8 is not there yet**: step 6 calls
      `table.register(descriptor)` as today; the merge edge is step 8's.
      Until then the `record_descriptor` cells step 6 derives from are
      raw-tagged cells on a declared page -- valid sources, listed as
      unsourced themselves.  Intended: the chain has one unsourced hop that
      step 8 closes.
R6.4  **`min(parameter_ids)` as a row element** on `RECORD_FIELD_RESIDENT`
      is id order as identity; kept in step 6 so the readers in the same
      function do not move twice, replaced in step 7 (E7.5).
R6.5  **Pickled Refs**: `Ref` and `Page` pickle by value; a Ref that
      travels in `metadata` (the read views hand out plain tuples, so none
      should) and is later handed to `post` in another process is refused
      as a nonexistent cell.  The read views must materialize ints, never
      Refs, into `graph.G.graph` / `metadata` (plan 70 risk 1 again).
R6.6  **Volume**: about 20 `SSA_VALUE_IDENTITY` rows per keyed field per
      function (row columns) plus one per default.  Measure with the audit
      `controller` case before the Woodshop compile.
R6.7  **`FUNCTION_SCOPE` for a function with no graph** mints a scope the
      step-3 pages never used; any later post that keys a step-3 page by it
      finds no rows.  That is the truth (no reduction happened for that
      function), not a defect; say so in the page docstring.

### 2.7 What step 6 needs from steps 2-5

`name_binding` rows (E6.9), `class_field_declaration` / `contract_demand`
rows (E6.7, E6.8), `return_site_slot` rows and the `return_site_cell`
attribute (E6.2), `SSA_FIELD_VERSION` rows and `lexical_read_scope` on the
builder (E6.2, E6.4), `record_member` / `record_descriptor` cells (raw, but
declared by E6.0).  From step 4: `source_output_value_ids` as a cell (N4).
From step 5: `ReturnBlock.return_site_cell` (N3) and `fresh_value` through
`post` (turns E6.2's DERIVED row into a NOVEL one).

### 2.8 Proof (step 6)

`python -u tools/compiler_probes/probe_branch_written_field.py` (seconds).
Link 6 today prints "none: the return-merge Phi carries no identity cell".
After E6.2-E6.5 the probe asserts: exactly one `SSA_VALUE_IDENTITY` row
with kind RETURN_MERGE for `step`'s record slot, one inbound edge to a
`CELL_SET` whose members are the two `RETURN_SITE_SLOT` cells; two
`RECORD_RETURN_FIELD_SELECTION` rows keyed by that cell, the tail site's
fact = the MERGED `SSA_FIELD_VERSION` cell (DERIVED), the terminal arm's
fact `Unresolved(VERSION_NOT_CONST_OR_CARRIED_PHI)` or a version cell
(whichever the tree produces; the probe prints which); zero
`Unsourced` rows at stage `RECORD_RETURN_VERSION`; `unsourced-identity`
lists no id whose defining instruction is the record-return Cast or a
`record_field_phi`.  The probe prints the `unsourced-fact` groups before and
after by page and stage so the drop is measured.

`python -u tools/compiler_probes/probe_row_handle_record_parameter.py`
(seconds) must still print the same formals: it exercises E6.8's keyed
row columns and `RECORD_FIELD_DECOMPOSITION`.

The trigger case is the native dt-system lowering (census 50; the
`controller` audit case is the seconds-long slice of it: run
`python -u tools/audit_identity_concordance.py controller` first).  The
full lowering is launched by the user, not by this lane; the expected
outcome is that `_concord_record_return_phi_inputs` sees the later round's
Cast as a REVISE with a changed selection cell and admits it.

Unsourced groups retired (worklist keys `(page, raw_primitive, raw row)`
today): `scope_registry`, `record_field_layout_concordance`,
`loop_record_layout_concordance`, `loop_record_schema_concordance`,
`record_field_decomposition`, `program_abi_keyed_row_record`,
`numeral_record_literal_concordance`, `numeral_leaf_materialization_concordance`,
`numeral_leaf_width_concordance`, `numeral_return_leaves_concordance`,
`record_field_resident_concordance`, `output_identity_concordance`,
`record_return_phi_input_concordance`, `record_field_storage_concordance`;
the `Unsourced` rows at stage `record_return_version`; and the
`unsourced-identity` entries for the ids the E6.3-E6.8 functions mint.

## 3. Step 7: the frame linker

### 3.1 What is being moved, observed

All inside `_class_surface_ssa_program` unless named.

| structure / writer | what it holds or writes | edge today |
|---|---|---|
| `identity_aliases`, `default_literals` (per linked call, built before `frame_bindings`) | callee formal -> caller formal (from `parameter_names` and the callee identity table); callee Input -> Python default (from the child graph's Constant nodes and `inspect.signature`) | none; they are the INPUTS of `argument_binding` |
| `result_storage_bindings` (per call) and `result_storage_bindings_by_call` | callee value -> caller storage minted by `allocate_result_storage` (`GLOBAL_MONOTONIC_IDS.mint()`, accounting `returned_record_storage`, `callsite_id`, `record_field_storage_identity`; `numeral_leaf_width_concordance` when the leaf has a width) and `allocate_late_result_storage` (`mint_compiler_value_id()`, accounting adds `late_record_surface`, `compiler_frame_storage`) | none; 35 read sites |
| `frame_bindings` -> `argument_binding_page.set((callee, value, "binding"), callsite, (kind, source))` | kinds `caller_value`, `caller_literal`, `caller_storage` (from `storage_bindings`, `identity_aliases`, `discovery_linked_member`, `record_instance`, or the final `else` which calls `allocate_result_storage`), `default_literal` | none; the `else` mints storage from absence and records `caller_storage` (census 10 #22, the file's most load-bearing fallback) |
| `_complete_propagated_frame_tails` | reads `binding_page.cells.get(...)` directly; restores a slot from a `caller_storage` binding or mints `GLOBAL_MONOTONIC_IDS.mint()` with `linked_call_frame_storage`; writes `propagated_frame_tail_concordance` `(owner, callsite, callee, formal) -> (slot, kind string)` | PARTIAL |
| `_link_frame_lease` (three callers: caller-storage clone, distinct-result replacement, per-owner slot replacement) | `frame_lease_link` `(caller, slot) -> (callsite, callee, formal)` | PARTIAL |
| `frame_ledgers` / `frame_transformation_ledger` / `TransformationLedger.propose` | `transformation_decision` `(scope, identity) -> (rule, proof, target)` REVISE; `transformation_event` `(scope, serial)` CONCORD; `transformation_rejection` keyed by the full rejection; rules `linked_record_member` < `distinct_owner` < `exact_argument_binding` < `distinct_result`; four `propose` sites in the frame loop | the best-formed writes in the file: rule + proof + before/after, no source cell |
| `coalesce_record_field_storage` -> `record_storage_alias` `PageMapping` under `mint_scope(("record_storage_alias", symbol))`; the resolve pass; `_publish_concordant_function_aliases(provenance="parameter_record_storage")` | value -> resident (three branches), then every operand and `function.args` rewritten | none; `read_ids[0]` / `candidates[0]` |
| `record_parameter_value` (concord under `mint_scope("record_field_access")`), `record_parameter_row_handle` (concord; `row_identity` falls back to the `value_record` schema NAME when the abi record has no `identity`) | the roots of the forwarding graph | none; name fallback at the root |
| child record pairing -> `call_record_pair_concordance` | `caller_children = {field.storage_identity: field.record_id}` over the bound caller record; `concord((caller, callsite, callee child), caller child)` | none; the match is by declared `storage_identity` |
| `_linked_caller_member` | reads `record_member`, `sequence_member`, `call_record_pair_concordance`, `record_field_decomposition`; returns the caller value or None; raises on disagreement; records nothing | none (the decision is unrecorded; master list A2) |
| `resident_by_value` / `kind_by_value` in `_sequence_concat_ops` (32 sites), `_promote_conditional_sequence_aliases` (21 + `promote` 17), `_sequence_augassign_ops` (9), `_sequence_row_operations` (9), `_sequence_prepend_*` (3 + 3), `_sequence_inplace_bit_pack_call_ops` (3) -- 104 sites | sequence residency and kind per value | none |
| shell control handoff -> `planning_value_concordance.bind_alias` | the merge of `compiled_process_graph_aliases`, `conditional_sequence_aliases`, `sequence_concat_aliases`, `singleton_name_aliases` (AST-name derived) | none |
| `_publish_concordant_function_aliases` | `planning_alias_transition_concordance` `set`; `planning_value_concordance.bind_alias`; `record_shape_transformation` (edge-carrying); `commit_sequence_row_layout` | PARTIAL |
| `scheduled_call_argument` mapping, `call_link_order_concordance`, `kernel_by_value_formal_concordance`, `phi_edge_projection_placement_concordance`, `_prune_unused_callee_formals_once` -> `pruned_callee_formal_concordance`, `_prune_dead_entry_field_aliases` -> `entry_record_handle_concordance`, `member_formals`, `_apply_concorded_function_aliases` -> `alias_application_concordance` | decisions per callsite / formal / instruction | NO or PARTIAL (census 10, 1.1-1.4) |
| mint sites in the linker (by enclosing def, tree today): `scheduled_source` region 12, `resident_record_id` region 12, `call_operands_available` 6, `nested_record_closure` 5, `mint_compiler_value_id` callers (`allocate_late_result_storage` and the `insert_at_loop_anchor` regions), `allocate_result_storage` 2, `physical_caller_storage` 1, `final_frame_value` 1, `cleaned_frame_value` 1, `destroys_value` 1, `calls_into` (frame tails) 1, `grow` 2, `ensure_structural_value` 2, `structural_membership_value` 1, `resolve_call_feed` 1, `visit_constructor_context` (three copies) | frame storage, replacement slots, constructor remaps, pool strides, output index / address constants, fold intermediates | census 10 section 2: 0 full edges |

### 3.2 Posts, writer by writer

| writer | post | provenance |
|---|---|---|
| binding walk, `identity_aliases` | `CALL_BINDING_INPUT` row `(caller scope, callsite, callee formal, "identity_alias")` fact = caller formal id | DERIVED(the callee formal's `name_binding` cell for `parameter_names`, the caller formal's identity cell) stage FRAME_BINDING |
| binding walk, `default_literals` | `CALL_BINDING_INPUT` `(..., "default_literal")` fact = the literal | DERIVED(the child graph's Constant node cell (`node_identity_cell`) or the `parameter_annotation` / `source_span` cell of the `ast.arg` default) |
| `discovery_linked_member` | `CALL_BINDING_INPUT` `(..., "discovery")` fact = caller member id | DERIVED(the `LINKED_CALLER_MEMBER` cell below) |
| `allocate_result_storage`, `allocate_late_result_storage` | (a) `SSA_VALUE_IDENTITY` `(caller scope, NEW)` fact `SSAValueFact("", RESULT_STORAGE)` NOVEL(RESULT_STORAGE_LEASE, (the callee value's identity cell,)); (b) `RESULT_STORAGE_BINDING` row `(caller scope, callsite, callee value)` fact = the new cell, CONCORD DERIVED(callee value cell, the callee `record_member` cell when `field` is given) | `result_storage_bindings` and `_by_call` become read views over (b); the `numeral_leaf_width_concordance` width becomes a `NUMERAL_LEAF` REVISE DERIVED(the `source_numeric_record_abi_concordance` cell found by row, replacing the `rpartition(".")` scan) |
| `frame_bindings` -> `argument_binding_page.set` | `ARGUMENT_BINDING` row `(callee, formal, callsite)` (the callsite moves from the column into the row: one row per binding, history per revisit) fact `BindingFact(kind, source)` | DERIVED(the `CALL_BINDING_INPUT` or `RESULT_STORAGE_BINDING` or `storage_bindings` cell it came from, the source value's identity cell).  The final `else`: fact `Unresolved(STORAGE_MINTED_FROM_ABSENCE, read=(the callee formal's identity cell,))` posted BESIDE the `RESULT_STORAGE_BINDING` row the mint makes -- the storage exists and is DERIVED from the formal (NOVEL), but the BINDING is not a proven fact.  Readers (`_binding_kind_findings`, `materializing_binding_kind`, `tensor_ssa_lowering`, `_complete_propagated_frame_tails`) treat the Unresolved as absence and read the storage through `RESULT_STORAGE_BINDING` |
| `_complete_propagated_frame_tails` | `FRAME_TAIL` fact `(slot, kind)` CONCORD | DERIVED(the `ARGUMENT_BINDING` cell read through `latest_ref`, never `cells.get`; the restored / minted slot's identity cell).  The minted slot is `SSA_VALUE_IDENTITY` NOVEL(FRAME_TAIL_SLOT, (callee formal cell,)); when no binding cell exists the tail row is `Unresolved(FORMAL_UNBOUND_AT_TAIL, read=(formal cell,))` and the slot is still leased |
| `_link_frame_lease` | `FRAME_LEASE` CONCORD | DERIVED(the slot's identity cell, the callee formal's identity cell, the ledger decision cell when the caller is a `propose` site) |
| `TransformationLedger.propose` | `TRANSFORMATION_DECISION` REVISE fact `(rule, proof, target)`; `TRANSFORMATION_EVENT` CONCORD; `TRANSFORMATION_REJECTION` CONCORD | `propose(identity, rule, proof, *, before, after, sources: tuple[Ref, ...])`: DERIVED(sources).  The four frame-loop callers pass the `LINKED_CALLER_MEMBER` cell (`linked_record_member`), the `RESULT_STORAGE_BINDING` cell (`distinct_result`), the `ARGUMENT_BINDING` cell (`exact_argument_binding`), the owner's `record_member` cell (`distinct_owner`).  The replacement slot the accepted proposal mints is `SSA_VALUE_IDENTITY` NOVEL(REPLACEMENT_SLOT, (decision cell,)).  `TransformationLedger.__init__`'s `mint_scope` is E6.0's |
| `coalesce_record_field_storage` aliases; the resolve pass | `RECORD_STORAGE_ALIAS` row `(function scope, value)` fact = resident, REVISE | DERIVED(the `RECORD_FIELD_RESIDENT` cell of step 6 -- when that cell is `Unresolved(RESIDENT_CHOSEN_BY_ORDER)` the alias row is `Unresolved` too and the rewrite still uses the working resident); the resolve pass posts each terminal DERIVED(the chain's cells).  The `PageMapping` becomes a read view; `_publish_concordant_function_aliases` receives the alias cells (below) |
| `record_parameter_value` | `RECORD_PARAMETER_VALUE` CONCORD | DERIVED(the `name_binding` cells of `identity_table[parameter]`, the `contract_demand` PARAMETER_RECORD cell or `class_declaration` cell of the record) stage RECORD_FORWARDING |
| `record_parameter_row_handle` | `RECORD_PARAMETER_ROW_HANDLE` CONCORD fact `(key, path prefix, row identity)` | DERIVED(the Indexed node's identity cell, the keyed field's `class_field_declaration` / abi field cell, the value record's declaration cell).  When the abi record has no `identity` and the schema NAME stands in: `Unresolved(ROW_IDENTITY_FROM_SCHEMA_NAME, read=(...))`; the forwarding writer then records the edge as `record_forwarding_unresolved_actual` does today (it is the one writer already in the target form) |
| child record pairing | `CALL_RECORD_PAIR` CONCORD | DERIVED(the bound record pair's cell -- the `argument_bindings` entry as a `RECORD_PARAMETER_VALUE` / `ARGUMENT_BINDING` cell -- and both fields' `record_member` cells).  The join is through the declared field identity both sides derive from one abi declaration; that is an edge, not a name match, once the cells are named |
| `_linked_caller_member` | `LINKED_CALLER_MEMBER` row `(caller, callsite, callee, formal)` fact = the caller value, CONCORD; `Unresolved(MEMBER_NOT_BOUND_AT_CALL, read=(...))` when it returns None today | DERIVED(the callee formal's `record_member` / `sequence_member` cell, the `CALL_RECORD_PAIR` or `ARGUMENT_BINDING` cell, the caller field's `record_member` cell, the `RECORD_FIELD_DECOMPOSITION` cell when it followed one).  The raises stay: two chains naming different values are a disagreement `post` CONCORD would also refuse |
| `resident_by_value` / `kind_by_value` (104 sites) | `SEQUENCE_RESIDENCY` row `(function, value)` fact `ResidencyFact(resident, kind, helper)` REVISE | DERIVED(the value's identity cell (step 2 `canonical_value` for authored values, `SSA_VALUE_IDENTITY` for minted), the operation's construct cell).  The dicts become read views over the page's scope; `conditional_sequence_aliases` / `sequence_concat_aliases` are built from them |
| shell handoff `planning_value_concordance.bind_alias` | `PLANNING_VALUE` REVISE per alias | DERIVED(the `SEQUENCE_RESIDENCY` cell for the sequence aliases; the shell's `compiled_process_graph_aliases` cell when step 4/5 posts it, else `Unsourced(PLANNER_OUTPUT_UNROUTED)`; for `singleton_name_aliases`: `Unresolved(RESIDENCY_FROM_NAME_MATCH, read=(the `name_binding` cells matched,))` -- an AST-name match is not a proven residency) stage SHELL_HANDOFF |
| `_publish_concordant_function_aliases(function, bindings, *, provenance, sources)` | `PLANNING_ALIAS_TRANSITION` REVISE; `PLANNING_VALUE` REVISE; the shape half unchanged (it already has the edge) | `provenance` stays as the `role` string of the shape transformation; `sources: Mapping[int, tuple[Ref, ...]]` per alias gives the DERIVED cells.  Callers: `coalesce_record_field_storage` (the alias cells), the record-projection block (`record_table` proof = the caller's `record_member` cell), `_concord_unbound_variant_rows` (the `unbound_variant_source_id` cell), `_retire_*` (the two `OUTPUT_IDENTITY` cells that justified retirement; `del mapping[...]` becomes `Unresolved(superseded)`-style `None` REVISE DERIVED from them) |
| `_apply_concorded_function_aliases` | `ALIAS_APPLICATION` CONCORD | DERIVED(the `PLANNING_VALUE` cell resolved through, the dominance decision's block cell is not a page: the fact keeps `target_block` as data) |
| `scheduled_call_argument` mapping | `SCHEDULED_CALL_ARGUMENT` CONCORD | DERIVED(the loop lowering's resident cell -- step 5's carried-port page; until then `Unsourced(PLANNER_OUTPUT_UNROUTED)`) |
| `call_link_order_concordance`, `kernel_by_value_formal_concordance`, `phi_edge_projection_placement_concordance`, `pruned_callee_formal_concordance`, `entry_record_handle_concordance`, `member_formals` | same rows, CONCORD / REVISE as today | DERIVED from the cells each decision read (link rank: the `call_record` cell; by-value: the formal's identity cell and its `llvm_argument_names` metadata is not a page -> Unsourced(PLANNER_OUTPUT_UNROUTED) for that element; projection placement: the moved instructions' identity cells; pruned formal: `Unresolved(FORMAL_UNBOUND_AT_TAIL)`-style Unresolved, since "unused" is absence; fields_only: the same; member formals: the `name_binding` cell of the root) |
| the remaining mint families | every mint -> `SSA_VALUE_IDENTITY` NOVEL with the family's transform: `FRAME_STORAGE_CLONE` (caller storage clone, `record_id_map`, mapped field ids), `CONSTRUCTOR_REMAP` (`remap[old_id]`, constructor args, pool ids, strides), `FRAME_SCAFFOLD` (output index / address, derived and returned lengths, literal values, projected element addresses, child pool columns), `STRUCTURAL_FOLD` (`structural_boolop_value`, `structural_membership_value`, `ensure_structural_value`'s index / address, `resolve_call_feed` intermediates; operand = a `CELL_SET` of the operand cells), `FRESHEN` (freshened `result.id`; operand = the redefined value's cell), `AGGREGATE_POSITION_MISSING`: the `node_id` minted when `projected_by_index` lacks the index is NOVEL(FRAME_SCAFFOLD, (the call's cell,)) plus an `Unresolved(AGGREGATE_POSITION_MISSING)` on the output binding row | -- |

### 3.3 Readers and read views

`result_storage_bindings` (35 reads), `identity_aliases`, `default_literals`,
`caller_children`, `frame_ledgers` (a handle cache; the ledger is on the
book), `resident_by_value` / `kind_by_value`, `scheduled_call_sources` keep
their names as read views over the pages until each reader is migrated;
`_complete_propagated_frame_tails` reads `latest_ref` instead of
`cells.get`; the eight private alias snapshots (`caller_value_aliases`,
`caller_record_aliases`, ...; census 10 section 4) are NOT migrated in step
7 -- they are copies of `PLANNING_VALUE`, and the fix is `resolve_alias`
at each site, a separate cleanup this step does not need.

### 3.4 Ordered edit list (step 7)

E7.1  `identity_concordance.py`: `ssa_value_identity_cell` handles formals
      (the row for a formal is posted by whoever appended it to
      `function.args`: `allocate_result_storage`, the tail pass, the
      replacement sites).
E7.2  `TransformationLedger.propose(..., sources)`; `_record_event` and
      the rejection row through `post`; `_BookEventLog` unchanged as a read
      view.
E7.3  `allocate_result_storage`, `allocate_late_result_storage`:
      `RESULT_STORAGE_BINDING` + NOVEL; `result_storage_bindings` read view.
E7.4  The binding walk: `CALL_BINDING_INPUT` posts; `frame_bindings` ->
      `ARGUMENT_BINDING` posts, the `else` as Unresolved beside the lease.
E7.5  `record_parameter_value`, `record_parameter_row_handle` posts;
      `RECORD_FIELD_RESIDENT`'s row swaps `min(parameter_ids)` for the
      `RECORD_PARAMETER_VALUE` cell of the parameter (R6.4); `coalesce_record_field_storage`
      aliases as `RECORD_STORAGE_ALIAS` posts.
E7.6  Child record pairing: `CALL_RECORD_PAIR` DERIVED; `_linked_caller_member`
      posts `LINKED_CALLER_MEMBER`; `discovery_linked_member` derives from
      it.
E7.7  `_complete_propagated_frame_tails`: `latest_ref`; `FRAME_TAIL`
      posts; tail slot NOVEL.  `_link_frame_lease(..., sources)`.
E7.8  The four `propose` sites pass sources; replacement slots NOVEL.
E7.9  `_sequence_concat_ops`, `_promote_conditional_sequence_aliases`,
      `_sequence_augassign_ops`, `_sequence_row_operations`, the three
      `_sequence_prepend_*` / bit-pack helpers: `SEQUENCE_RESIDENCY` posts
      (104 sites, one helper `post_residency(function, value, resident,
      kind, sources)` replaces the paired dict writes).
E7.10 Shell handoff: `PLANNING_VALUE` posts with the residency cells; the
      `singleton_name_aliases` Unresolved.  `_publish_concordant_function_aliases(...,
      sources)`; its five callers.
E7.11 `_apply_concorded_function_aliases`, `scheduled_call_argument`,
      link order, by-value formals, Phi-edge projection, the two prune
      pages, `member_formals`.
E7.12 The remaining mint families (E7.3 and E7.7-E7.8 covered the storage
      ones): `FRAME_STORAGE_CLONE`, `CONSTRUCTOR_REMAP`, `FRAME_SCAFFOLD`,
      `STRUCTURAL_FOLD`, `FRESHEN`; `mint_compiler_value_id` is deleted
      (its two definitions were bare passthroughs).
E7.13 `probe_row_handle_record_parameter.py` gains the book assertions of
      section 3.7.

Writer sites routed: about 40 page writes (census 10's #1-#5, #14-#18,
#21-#24, #28, #34, #35, #43, #49-#53, #57, #73-#75 and the shell handoff)
plus 104 residency dict writes and about 45 mint sites.

### 3.5 Risks (step 7)

R7.1  **The `argument_binding` `else` becomes visible.**  Every callee
      formal bound from absence is now an `Unresolved` row.  Readers that
      today take `("caller_storage", id)` as proof -- `materializing_binding_kind`,
      `tensor_ssa_lowering`, `_complete_propagated_frame_tails`'s restore
      path -- must read the storage from `RESULT_STORAGE_BINDING` and the
      kind from `ARGUMENT_BINDING` separately.  Nothing is lost (the slot is
      still leased) but the audit's `_binding_kind_findings` will list every
      such formal on the Woodshop compile.  That list is the finding the
      design wants; it is also the top risk, because the count is unknown
      until the probe and the `controller` case run.
R7.2  **Row shape change on `argument_binding`** (callsite moves from
      column to row).  `_complete_propagated_frame_tails` and the audit read
      by `(row, column)`; both change in E7.7 / the audit registration.
      Tests that read `history` on this page change with it (one test file,
      per census 10's reader list; enumerate before editing).
R7.3  **Fixed-point order**: `RESULT_STORAGE_BINDING` is CONCORD; a second
      round that re-allocates for the same callee value with
      `distinct_slot=True` is a NEW row today (`setdefault` keeps the
      first).  The post must key `distinct_slot` allocations by
      `(callsite, callee value, serial)` or the second lease is a CONCORD
      disagreement.  Read `allocate_result_storage`'s `distinct_slot`
      callers before E7.3.
R7.4  **`SEQUENCE_RESIDENCY` volume and the `promote` inner function**
      (17 writes): the helper must post once per (value, resident, kind)
      change, not per dict assignment, or `REVISE` refuses same-source
      rewrites.
R7.5  **Ledger sources**: `propose` is called with `before`/`after` ids
      that may have no identity row yet (a callee value from a function not
      yet materialized).  A missing cell is a refusal; the caller must
      post the callee value's identity first (E7.1) or pass the
      `LINKED_CALLER_MEMBER` Unresolved cell as the source.

### 3.6 What step 7 needs from steps 2-6

Step 6's `SSA_VALUE_IDENTITY`, `FUNCTION_SCOPE`, `RECORD_FIELD_RESIDENT`,
`RECORD_FIELD_DECOMPOSITION`, `OUTPUT_IDENTITY` cells; step 2's
`name_binding`, `class_field_declaration`, `contract_demand`,
`canonical_value` cells; step 5's carried-port page for
`SCHEDULED_CALL_ARGUMENT` and the shell's `compiled_process_graph_aliases`
as a cell (else `Unsourced(PLANNER_OUTPUT_UNROUTED)`, which the latch
lists until step 5 lands).

### 3.7 Proof (step 7)

`python -u tools/compiler_probes/probe_row_handle_record_parameter.py`
(seconds; `World.items` keyed field, row handle passed to `center`).  Today
it prints the lowered functions' formals.  After step 7 it asserts, reading
only the book: one `RECORD_PARAMETER_ROW_HANDLE` row for the Indexed value
in `sync` with edges to the `items` field declaration and the `Item`
declaration cells; for every formal of `center`, one `ARGUMENT_BINDING` row
whose fact is not `Unresolved` OR whose `Unresolved` reason is printed;
one `LINKED_CALLER_MEMBER` row per `Item` leaf `center` reads
(`orientation`, `mass`) with an edge to `sync`'s `record_member` cell for
that leaf's column; zero `RESULT_STORAGE_BINDING` rows for `center`'s
record formals (they are bound, not leased); `unsourced-identity` lists no
id defined in the two functions.  Then `probe_branch_written_field` must
stay green (its `step` has one parameter record and no calls: proves the
frame pages cost nothing when empty).  The whole-program Woodshop compile
is the user's to launch.

Unsourced groups retired: `argument_binding`, `frame_lease_link`,
`propagated_frame_tail_concordance`, `call_record_pair_concordance`,
`record_parameter_value`, `record_parameter_row_handle`,
`record_storage_alias`, `planning_value_concordance`,
`planning_alias_transition_concordance`, `alias_application_concordance`,
`scheduled_call_argument`, `call_link_order_concordance`,
`kernel_by_value_formal_concordance`, `phi_edge_projection_placement_concordance`,
`pruned_callee_formal_concordance`, `entry_record_handle_concordance`,
`member_formals`, `transformation_decision`, `transformation_event`,
`transformation_rejection`; the `unsourced-identity` entries for every id
the linker mints.

## 4. Step 8: the book-backed tables

### 4.1 What is being moved, observed

| writer | what it writes | edge today |
|---|---|---|
| `_mint_table_owner(book, label)` (callers: `new_layout_tables`, `SSARecordTable.__init__`, `_SSALayoutTable.__init__` and `__deepcopy__`, `SSASequenceTable.__init__`, `SSACallTable.__init__`, `IRModule._layout_owner`) | `scope_registry` `(label or "table" or "module", serial)`; `label` is the function name string (`SSARecordTable(owner=symbol)` at about nine sites in `fortran_c_shell`, `owner="call_records"` for the linker's call table, `SSASequenceTable()` with no owner at four sites and in tests) | none; the join table -> function is a label string (census 40, 4.3) |
| `_BookRows.__setitem__` / `__delitem__` | `<kind>_descriptor` `(owner, id)` REVISE; `on_change` -> `_revise_member_claims` -> `<kind>_member` `(owner, member)` REVISE | none; the descriptor revision and the member revisions do not name each other |
| `SSARecordTable.register` | the complementary-view merge (section 2.1, last row) | none |
| `_SSALayoutTable.register` -> `_publish`, `_record_supersession`, `withdraw_superseded_layout_derivations` | `layout_state` `("resolved", descriptor, edge row or None)`; `layout_supersession` `(owner, kind, target, source, stage)` -> `(incumbent, replacement)` at the next free column; `("superseded", ...)`, `("invalidated", ...)` | YES for a re-declaration (the model writer); NO for a first declaration; `stage` defaults to the string `"declaration"` |
| `SSASequenceTable.register` | `sequence_column_claims` `(owner, sequence, "column_dtypes")` REVISE on every attempt; `sequence_descriptor` via `_BookRows` | none; the proposing site is not named |
| `SSACallTable.__setitem__` / `__delitem__` / `_BookCallList._commit` | `call_record` `(owner, caller name)` REVISE to the whole tuple; nine in-place list edits in `fortran_c_shell` (`setdefault(...).append`, whole-list assignments of `rebuilt` / `refreshed_records` at six sites, `final_call_records[...] = [...]`) | none; which record changed is a column diff |
| `__reduce__` on every table | drops the serial; a pickled table is rebuilt under a new scope on load | the census-40 gap |

### 4.2 `TABLE_OWNER`: scopes joined to functions by a row

`_mint_table_owner(book, label, *, function_scope: Ref | None = None)`:
`SCOPE_REGISTRY` row NOVEL(MINT_SCOPE) as E6.0 (unchanged), then
`TABLE_OWNER` row `(owner scope,)` fact `TableOwnerFact(kind, function)`
DERIVED(the function's `FUNCTION_SCOPE` cell) when given, else
`Unresolved(NO_FUNCTION_SCOPE, read=())` for `"module"`, `"table"`,
`"call_records"` and the test constructors.  Every `SSARecordTable(owner=symbol)`
/ `SSASequenceTable(owner=symbol)` in `fortran_c_shell` passes
`function_scope=function_scope_of(all_functions[symbol])` (one helper
`table_for(symbol)` replaces the nine `setdefault(..., SSARecordTable(owner=...))`
spellings).  `__deepcopy__` derives the copy's owner row from the
original's `TABLE_OWNER` cell (stage TABLE_REGISTRATION), so a copied table
is joined to the same function.  `__reduce__` keeps dropping the serial (a
pickle carries no book) but `__setstate__` posts the rebuilt owner
`Unresolved(NO_FUNCTION_SCOPE)` so a table rebuilt on another book is
visible as such rather than silently renumbered.

With this row a `record_member` claim on `(owner, value)` reaches the
function's `FUNCTION_SCOPE` cell, and through it the `lexical_read_scope`
whose `canonical_value` rows name the same value: the cross-table join
census 40 says cannot be written today.

### 4.3 `_BookRows` revisions as DERIVED from the writer's cells

`_BookRows.__setitem__(key, value, *, sources: tuple[Ref, ...] = ())` and
`__delitem__(key, *, sources=())`; `MutableMapping` syntax cannot carry
sources, so the tables gain an explicit method `assign(id, descriptor,
sources)` and `register(descriptor, *, sources=(), stage=...)` and the bare
`table.records[id] = descriptor` / `del` spellings become
`Unsourced(RAW_PRIMITIVE)`-tagged as they are today until every caller
passes sources (the latch lists them).  Posts:

| change | post | DERIVED from |
|---|---|---|
| descriptor set | `RECORD_DESCRIPTOR` `(owner, id)` REVISE fact = descriptor | `sources` (the caller's cells: for `materialize_record_phis` the `RECORD_PHI_EXPANSION` cells; for `materialize_program_abi_record_literals` the field value identity cells and the abi field declaration cells; for the linker's propagated descriptors the callee `record_descriptor` cell and the `frame_map` bindings' `ARGUMENT_BINDING` cells; for `publish_inout_scalar_return_snapshots` the return snapshot receipt cells) |
| descriptor removal | REVISE fact None | DERIVED(the cell that superseded it) |
| member claim change (`_revise_member_claims`) | `RECORD_MEMBER` `(owner, member)` REVISE fact = surviving claims | DERIVED(the descriptor cell just posted) -- the member row names the descriptor revision that changed it |
| `SSARecordTable.register` complementary merge | `RECORD_DESCRIPTOR_MERGE` row `(owner, record id, revision)` fact `RecordMergeFact(incumbent, incoming, merged, widened_fields, adopted_pool)`; then the descriptor REVISE DERIVED(the merge cell) | DERIVED(the incumbent `RECORD_DESCRIPTOR` cell, `sources`).  This is `_record_supersession`'s shape for records: the merge is an edge from incumbent and incoming to merged, `writable` widening and `instance_pool` adoption are named fields of the fact instead of silent ORs |
| `_SSALayoutTable._publish` first declaration | `LAYOUT_STATE` fact `("resolved", descriptor, edge row)` | today `edge_row` is None for a first declaration: post DERIVED(the `<kind>_descriptor` cell just written); `stage` becomes a `Stage` object (`CTYPES_INTERCEPTION`, `DECLARATION`) -- the string default goes |
| `_record_supersession` | `LAYOUT_SUPERSESSION` at the next free column | already the full edge; re-expressed as `post(..., Derived((incumbent cell, replacement cell)), stage)` so its cells are sourced too |
| `withdraw_superseded_layout_derivations` | `LAYOUT_STATE` `("invalidated", ...)` | DERIVED(the changed nested row's state cell) |
| `SSASequenceTable.register` | `SEQUENCE_COLUMN_CLAIMS` REVISE per attempt | `register(descriptor, *, sources)`: DERIVED(sources); a caller that passes none posts `Unsourced(SEQUENCE_CLAIM_WITHOUT_PROPOSER)`, so the latch lists every anonymous proposer (the sequence conflict report then names sites, which is what the census asked for) |
| `SSACallTable` / `_BookCallList` | `CALL_RECORD` `(owner, caller)` REVISE | `_BookCallList` gains `append(record, *, sources)`, `replace(index, record, *, sources)`, `remove_at(index, *, sources)`; `_commit` posts DERIVED(sources).  The nine `fortran_c_shell` edits pass the `ARGUMENT_BINDING` / `RESULT_STORAGE_BINDING` / `LINKED_CALLER_MEMBER` cells the rebuilt record was built from (whole-list assignments become one post per changed record: `_commit` diffs by `callsite_id`, which is the record's identity, and posts each changed record's cells) |

### 4.4 The latch CLOSE criteria and the audit tool

CLOSE when, on every audit case (`view`, `toplevel`, `energy`,
`controller`, `controller_untyped`, `mapping`, and `oscillator`):

1. `unsourced-fact` is zero for every unit -- no `raw row` (every page
   written is declared) and no `cell` without an edge or mint row;
2. `unsourced-identity` is zero -- every `MINTED` id in every function has
   a mint row, which requires step 5's `fresh_value` and
   `deployment_ssa_binding.bind_deployment_dataflow`'s one mint to be
   routed too (out of these steps; listed);
3. `git grep` finds no call of `IdentityPage.set`, `revise`, `concord`,
   `bind_alias`, `PageMapping.__setitem__` / `__delitem__` / `setdefault`
   outside `identity_concordance.py` itself, and no `IdentityBook.page(<str>)`
   creation of an undeclared page (`book.page` accepts a `Page` or a
   registered name only);
4. `GLOBAL_MONOTONIC_IDS.mint` has one caller: `IdentityBook.post`.

Then: `IdentityBook.__init__` takes `latch: Latch = DEFAULT_LATCH` with
`DEFAULT_LATCH = Latch.CLOSED` (a module constant flipped once; tests that
need an OPEN book pass it explicitly), `begin_identity_book` forwards it,
`_admit_raw_write` and the `Unsourced` branch of `post` refuse as already
written.  The raw primitives are then made non-public (`_set`, ...) with
`PageMapping` surviving as a read view (`__setitem__` raises).

`tools/audit_identity_concordance.py` `main`: parse the report's
`unsourced:` line; when the book's latch is CLOSED and either count is
non-zero, `failures += 1` (the closed latch makes an unsourced row a
defect, not a worklist entry); while OPEN the line stays informational.
One more flag, `--require-sourced`, applies the same rule under an OPEN
latch for the transition period so CI can gate a case that has reached
zero without closing the latch globally.  `concordance_report` is
unchanged: the first line still gates the per-page findings.

### 4.5 Ordered edit list (step 8)

E8.1  `_mint_table_owner(..., function_scope)` + `TABLE_OWNER`; `table_for(symbol)`
      in `fortran_c_shell`; `__deepcopy__` / `__setstate__` owner rows;
      `new_layout_tables` and `IRModule._layout_owner` pass the module's
      scope (a `FUNCTION_SCOPE`-like row for the module: name `"<module>"`).
E8.2  `_BookRows` `assign` / `remove` with sources; `_revise_member_claims`
      DERIVED(descriptor cell); `SSARecordTable.register(..., sources)` and
      `RECORD_DESCRIPTOR_MERGE`; `SSASequenceTable.register(..., sources)`;
      `_SSALayoutTable._publish` first-declaration edge, `Stage` objects.
E8.3  `_BookCallList` sourced mutators; `SSACallTable.__setitem__(...,
      sources)`; the nine `fortran_c_shell` call-record edits.
E8.4  Every `register` / `records[...] =` caller in `fortran_c_shell`,
      `precompile_to_ssa` (`SSASequenceTable({...})` helper tables,
      `SSARecordTable({...})` control-function tables), `ctypes_layout`
      (`struct_table.register(..., stage=CTYPES_INTERCEPTION)`) passes
      sources.  Tests constructing bare tables pass none and are tagged.
E8.5  Audit: register the table pages with `unsourced-fact` (generic);
      `_table_member_findings` / `_layout_table_findings` unchanged; add
      `record-merge-widened` listing `RECORD_DESCRIPTOR_MERGE` rows whose
      `widened_fields` is non-empty (a field made writable by one
      projection is now a listed decision).
E8.6  `tools/audit_identity_concordance.py` `main` (section 4.4);
      `DEFAULT_LATCH`; the primitives made non-public; `book.page(str)`
      refuses undeclared names.

### 4.6 Risks (step 8)

R8.1  **Row-shape validation on the re-declared table pages.**  `post`
      validates rows; today's rows are `(owner, int)` for descriptors and
      members, `(owner, str)` for `call_record`, `(owner, kind, int)` for
      `layout_state`, a 5-tuple for `layout_supersession`, and the ledger's
      `(scope, identity-tuple)`.  Any writer whose row deviates (a `None`
      caller name, a non-int member) is refused at the first post; run the
      seven audit cases after E6.0 (declaration alone) to find them before
      E8.2 routes writers.  Top risk for this step.
R8.2  **Pickled tables**: `__reduce__` rebuilds under a new scope; after
      E8.1 the rebuilt owner is an `Unresolved` row, so a module unpickled
      into a compile (host SSA cache, `_HostSSACachePickler`) shows every
      table row as descended from an unresolved owner.  True, and the latch
      cannot close over such a compile; the cache must store the owner
      label and re-derive from `FUNCTION_SCOPE` on load (a follow-up, named
      here, not done here).
R8.3  **`sources` plumbing breadth**: `register` has about 40 callers
      across `fortran_c_shell`, `precompile_to_ssa`, `ctypes_layout`,
      tests.  A caller left unsourced is tagged, not broken; the latch
      cannot close until the last one is routed.  That is the intended
      pressure.
R8.4  **`_commit` diff by `callsite_id`**: two records at one callsite
      (a decomposed plan call re-emitted) would be one identity; read
      `decomposed_plan_call` handling before E8.3.
R8.5  **Closing the latch is global**: one compile path that still writes
      raw (a tool, a test helper, `deployment_ssa_binding`) raises for the
      whole program.  The `--require-sourced` flag and the per-case zero
      counts are the staged path; `DEFAULT_LATCH` flips last.

### 4.7 What step 8 needs from steps 2-7

Step 6's `FUNCTION_SCOPE`, `SSA_VALUE_IDENTITY`, `RECORD_PHI_EXPANSION`
cells (sources for `register`); step 7's `ARGUMENT_BINDING`,
`RESULT_STORAGE_BINDING`, `LINKED_CALLER_MEMBER` cells (sources for
`call_record`); step 5's `fresh_value` through `post` and step 4/5's
planner cells (criterion 2 cannot reach zero without them); step 2's
`canonical_value` rows (the far end of the `TABLE_OWNER` join).

### 4.8 Proof (step 8)

`python -u tools/audit_identity_concordance.py view` and `mapping`
(seconds): every `record_member` row has one inbound edge to a
`record_descriptor` cell; every `record_descriptor` cell has an inbound
edge or a `RECORD_DESCRIPTOR_MERGE` cell; every `TABLE_OWNER` row derives
from a `FUNCTION_SCOPE` cell except the module layout owner; `layout_state`
first declarations have edges.  `tests/test_ir_sequence_tables.py` and
`tests/test_native_record_return_state.py` construct bare tables: they
must still pass (tagged, not refused, while OPEN).  Then all seven cases
with `--require-sourced`; when all seven report zero, flip `DEFAULT_LATCH`
and run them once more: same output, latch CLOSED.

Unsourced groups retired: `record_descriptor`, `record_member`,
`sequence_descriptor`, `sequence_member`, `struct_descriptor`,
`struct_member`, `union_descriptor`, `union_member`, `layout_state`,
`layout_supersession`, `sequence_column_claims`, `call_record`, and the
`TABLE_OWNER` / `scope_registry` rows minted with a label only.

## 5. Not in these steps (named so the latch's list is read correctly)

`value_shape`, `call_edge`, `linked_value_abi_polymorphism`, `shape.node`
/ `shape.linked`, `ssa_shape_materialization`, `record_scalar_shape_concordance`,
`proven_shape` (whole-program shape settlement); the sequence contract
helpers (`sequence_contract_concordance`, `sequence_row_layout_concordance`);
the source-stage declarations of `_lower_ast_source_to_ssa_impl`
(`source_numeric_*`, `static_record_piece_*`, `assignment_projection_*`);
the `record_field_access` family (already edge-carrying);
`record_phi_temporal_fallback_concordance`; `deployment_ssa_binding`'s
mint; `precompile_to_ssa`'s `fresh_value` and `first_value_id=GLOBAL_MONOTONIC_IDS.peek()`
sites (step 5).  Each stays listed by the audit until its own step.

## 6. Order across the three steps

E6.0 first (declarations and `mint_scope`), then E6.1-E6.2 (scopes and the
return-merge cell), then the rest of step 6, then step 7, then step 8, with
one exception: E8.1 (`TABLE_OWNER`) can land any time after E6.1 and
should land before E7.6, so `LINKED_CALLER_MEMBER`'s `record_member`
sources are already joined to a function.  Every step is one commit and one
probe run; the audit's seven cases run after each.

## 7. Held for the user

1. **N2, variadic transforms.**  `Transform(arity=None)` admitting any
   operand count (one line in `post`'s arity check) versus the `CELL_SET`
   page (no api change, one extra hop in every variadic chain).  The plan
   is written for `CELL_SET`; the alternative removes the page and E6.1's
   `post_cell_set`.
2. **The `argument_binding` `else` (R7.1).**  Unresolved binding beside a
   NOVEL lease (as planned), or refuse the compile at the site once the
   latch closes.  The plan records; the repo's stated rule may want the
   raise.  Both need the same edit; only the fact differs.
3. **`RESIDENT_CHOSEN_BY_ORDER` (E6.8).**  Unresolved as planned, or
   choose no resident and leave the reads distinct (a behaviour change in
   the operand rewrite).  The plan keeps today's behaviour and records it.
4. **When `DEFAULT_LATCH` flips.**  After the seven audit cases only, or
   after the Woodshop whole-program compile also reports zero.  The plan
   proposes the seven cases plus `--require-sourced` on the Woodshop
   compile before the flip.
