# Continuation: step 7 (the frame linker)

Lane S7, 2026-10-01.  Plan `90_plan_steps6_8_records_linker_tables.md`
section 3; census 10 sections 1.1-1.6 and 2; census 75 section 4 and the
step-7 DRAFT of section 10.  Nothing committed.  Searched with `git grep` /
`sed` only; no pytest, no Woodshop or dt-system lowering.

Decisions in force (lead, user approved): argument storage minted from
absence is RECORDED (Unresolved, or NOVEL where an id is minted), never
raised; RESIDENT_CHOSEN_BY_ORDER is its own Reason; many-source posts use
step 5's `cell_set` page; SSA value identities go on step 5's `ssa_value`
page (lane S6 adopted the same page).

## What changed

### `src/compiler/concordance_declarations.py` (one "Step 7" sub-block at the end of the Steps 6-8 section, after S6's step-6 block)

Stages `FRAME_BINDING`, `FRAME_LINK`, `FRAME_TAIL_STAGE`,
`RECORD_FORWARDING`, `PLANNING_RESIDENCY`, `SHELL_HANDOFF`.
Transforms (arity 1) `RESULT_STORAGE_LEASE`, `FRAME_TAIL_SLOT`,
`FRAME_STORAGE_CLONE`, `REPLACEMENT_SLOT`, `CONSTRUCTOR_REMAP`,
`FRAME_SCAFFOLD`, `STRUCTURAL_FOLD`, `NESTED_RECORD_PART`,
`DECLARED_ROW_COLUMN`.  Reasons `STORAGE_MINTED_FROM_ABSENCE`,
`ROW_IDENTITY_FROM_SCHEMA_NAME`, `FORMAL_UNBOUND_AT_TAIL`,
`MEMBER_NOT_BOUND_AT_CALL`, `AGGREGATE_POSITION_MISSING`,
`RESIDENCY_FROM_NAME_MATCH`, `FRAME_SOURCE_CELL_ABSENT` (new: a linker
statement whose source value has no `ssa_value` / `canonical_value` cell).
Facts `ArgumentBindingFact` (a 2-tuple subclass `(kind, source)`, so every
reader of `(kind, source)` pairs keeps reading), `ResidencyFact`,
`CallBindingSource` (IDENTITY_ALIAS / DEFAULT_LITERAL / DISCOVERY).
Reused from S6's block: `RECORD_FIELD_RESIDENT`, `RECORD_FIELD_DECOMPOSITION`,
`RESIDENT_CHOSEN_BY_ORDER`, `PLANNER_OUTPUT_UNROUTED`,
`RECORD_ABI_MATERIALIZATION`, `FRESHEN`; from step 5: `SSA_VALUE`,
`CELL_SET`, `GRAPH_ID_WITHOUT_CANONICAL_CELL`, `CONTROL_FUNCTION_ROOT`.

Pages, new: `result_storage_binding` (caller SCOPE, callsite LABEL,
callee_value VALUE_ID, serial INDEX) -> Ref of the storage's `ssa_value`
cell (the serial keys `distinct_slot` leases, plan R7.3);
`call_binding_input` (caller, callsite, callee_value, source LABEL) ->
object; `linked_caller_member` (caller NAME, callsite LABEL, callee NAME,
formal VALUE_ID) -> int, REVISE (a later round may bind a record this round
could not; the comment in the file says CONCORD -- the code posts REVISE);
`sequence_residency` (function NAME, value VALUE_ID) -> `ResidencyFact`
(declared only, see "Not done").  Re-declared with present shapes:
`argument_binding` (callee NAME, formal VALUE_ID, callsite LABEL) ->
`ArgumentBindingFact` -- the callsite moved from the column into the row;
`argument_binding_resolution`, `frame_lease_link`,
`propagated_frame_tail_concordance`, `record_storage_alias`,
`record_parameter_value`, `record_parameter_row_handle`,
`record_forwarding_edge`, `record_forwarding_unresolved_actual`,
`call_record_pair_concordance`, `planning_value_concordance` (fact
`object`: int, None on retirement), `planning_alias_transition_concordance`,
`alias_application_concordance`, `scheduled_call_argument`,
`call_link_order_concordance`, `kernel_by_value_formal_concordance`,
`phi_edge_projection_placement_concordance`,
`pruned_callee_formal_concordance`, `entry_record_handle_concordance`,
`member_formals`, `formal_actual_concordance`, `transformation_decision`
(scope, identity LABEL) -> tuple, `transformation_event` (scope, serial) ->
dict, `transformation_rejection` (six-element row) -> int.

### `src/compiler/fortran_c_shell.py`

Module-level helpers before `_linked_caller_member`: `_frame_book_scope`
(the function's `tensor_shape_concordance_scope`, else its name -- step 5's
`ssa_value` row scope), `_frame_cells`, `_frame_value_cell` (`ssa_value`
cell of a value in a function), `_frame_graph_cell` (`canonical_value` cell
of a graph id under the graph's `lexical_read_scope`), `_frame_table_cell`
(a book-backed table's member / descriptor cell), `_frame_post` (DERIVED
from the cells named, else `Unsourced(reason)`; CONCORD keeps `concord`'s
disagreement rule; REVISE with an unchanged fact posts nothing; a REVISE the
api refuses for want of a changed source is recorded Unsourced), `_frame_mint`
(a NOVEL `ssa_value` row through step 5's `_mint_ssa_id`; several operand
cells become one `cell_set` row; none names the function root exactly as
`fresh_value` does), `_RecordAbiMinter`, `_result_storage_lease_cell`,
`_post_result_storage_lease`, `_argument_binding_fact` (reads the step-7
row first, then the pre-step-7 `(callee, formal, "binding")` row at column
callsite), `_linked_caller_member_cell`.

Routed writers:

- `_linked_caller_member` posts `linked_caller_member` at every return:
  the caller value DERIVED(the callee formal's `record_member` /
  `sequence_member` cell, the `call_record_pair` cell when used, both
  records' `record_descriptor` cells, the `record_field_decomposition` cell
  when followed, the caller member's cells); None becomes
  `Unresolved(MEMBER_NOT_BOUND_AT_CALL, read=cells)`.  The raises stay.
- `_link_frame_lease(..., caller_function, callee_function, sources)`:
  `frame_lease_link` DERIVED(slot cell, formal cell, sources).  The three
  callers pass the ledger decision cell (`TransformationLedger.decision_cell`)
  or the `linked_caller_member` cell.
- `_complete_propagated_frame_tails`: reads the binding through
  `latest_ref` (never `cells.get`); an `Unresolved` binding reads its
  storage from `result_storage_binding` and restores it as caller storage;
  the tail slot is NOVEL(FRAME_TAIL_SLOT, callee formal cell); the tail row
  derives from the binding cell, the slot cell and the formal cell, or is
  posted `Unsourced(FORMAL_UNBOUND_AT_TAIL)` with the `(slot, kind)` fact
  kept (deviation from the plan's Unresolved fact: the disagreement check
  and the slot id read the fact back; the reason is on the unsourced page).
- `allocate_result_storage` / `allocate_late_result_storage`: storage
  NOVEL(RESULT_STORAGE_LEASE, callee value cell + its `record_member` cell
  when `field` is given); `result_storage_binding` row posted beside it;
  `result_storage_bindings` dicts stay as the read view.  The nested
  returned-record ids (`result_record_bindings`, `record_id_map`) are
  NOVEL(FRAME_STORAGE_CLONE, callee descriptor cell).
- The binding walk: `call_binding_input` rows for `identity_aliases`
  (never populated today -- the dict is read-only in the tree),
  `default_literals` (DERIVED(child graph Constant / Input cell, callee
  formal cell)), `discovery_linked_member` (DERIVED(the
  `linked_caller_member` cell)); `argument_binding` rows `(callee, formal,
  callsite)` posted REVISE DERIVED(formal cell, caller value cell, lease
  cell, the input cell); the final `else` posts
  `Unresolved(STORAGE_MINTED_FROM_ABSENCE, read=cells)` beside the lease.
- Child record pairing: `call_record_pair_concordance` CONCORD
  DERIVED(the bound pair's and both children's `record_descriptor` cells).
- `record_parameter_value`: DERIVED(the parameter's `name_binding` cell
  per version, the value's `canonical_value` cell, the record's
  `contract_demand` PARAMETER_RECORD cell).  `record_parameter_row_handle`:
  DERIVED(the Indexed node's cell, its base nodes' cells, both records'
  `contract_demand` cells); when the value record declares no `identity`
  the fact is kept and posted `Unsourced(ROW_IDENTITY_FROM_SCHEMA_NAME)`
  (deviation: an Unresolved fact would break `forwarding_caller_side`,
  which unpacks the tuple).
- `coalesce_record_field_storage`: `record_field_resident_concordance`
  posts the resident DERIVED(the parameter root's and every read's graph
  cells), or `Unresolved(RESIDENT_CHOSEN_BY_ORDER)` when `read_ids[0]` was
  taken (the rewrite still uses the working resident); every
  `record_storage_alias` write goes through `post_storage_alias` (REVISE,
  DERIVED(the aliased value's cell, the resident cell), reason
  RESIDENT_CHOSEN_BY_ORDER for `candidates[0]`); the resolve pass derives
  each terminal from the chain's alias cells; `_publish_concordant_function_aliases`
  receives `sources`.
- `_publish_concordant_function_aliases(..., sources=None)`:
  `planning_alias_transition_concordance` and `planning_value_concordance`
  through `_frame_post` REVISE DERIVED(sources, the alias's and resident's
  `ssa_value` cells); callers passing nothing are recorded Unsourced.
- Shell handoff: `planning_value_concordance` per alias family --
  loop-carried storage and identity returns DERIVED(`canonical_value`
  cells); `compiled_process_graph_aliases` and the sequence residency
  families `Unsourced(PLANNER_OUTPUT_UNROUTED)`; `singleton_name_aliases`
  `Unsourced(RESIDENCY_FROM_NAME_MATCH)` (deviation: the int fact is kept
  so `resolve_alias` / `alias_bindings` keep working; the reason is on the
  unsourced page).
- `materialize_parameter_record_abi`: the 35 `GLOBAL_MONOTONIC_IDS.mint()`
  sites inside it (and its nested materializers) route through
  `_RecordAbiMinter`, bound to that name inside the function: each is
  NOVEL(NESTED_RECORD_PART, the declared parameter record's
  `contract_demand` cell, set per record by `declare`).  The declared row
  columns and row record are NOVEL(DECLARED_ROW_COLUMN, ...); the
  `required_unread_leaf` candidate is NOVEL(NESTED_RECORD_PART, ...).
- `mint_compiler_value_id(function=None, transform=None, operands=())`:
  with a function, a NOVEL `ssa_value` row (default FRAME_SCAFFOLD); the
  frame-loop sites for the caller storage clone (FRAME_STORAGE_CLONE), the
  late result lease, the nested record ids and the two replacement slots
  (REPLACEMENT_SLOT from the proposal's cells) pass `caller`.  Callers that
  name no function still draw a bare id (see "Not done").

### `src/compiler/transformation_priority.py`

`TransformationLedger.propose(..., sources=())`: with sources, the decision
(REVISE), the event and the rejection (CONCORD) post DERIVED(sources) under
`FRAME_LINK`; a refused revision is recorded `Unsourced(PLANNER_OUTPUT_UNROUTED)`;
without sources the raw writes are unchanged.  `decision_cell(identity)`
returns the retained decision's Ref.  The four frame-loop callers pass:
the `linked_caller_member` cell (`linked_record_member`); the
`result_storage_binding` cell and the callee formal cell (`distinct_result`);
the `argument_binding` cell and the caller value cell
(`exact_argument_binding`); the owner's `record_member` cell, the
`linked_caller_member` cell and the slot cell (`distinct_owner`).

### `src/compiler/identity_concordance.py`, `src/compiler/tensor_ssa_lowering.py`

`argument_binding_history(page, callee, formal)` reads both row shapes;
`_binding_kind_findings` groups per (callee, formal) across rows,
`materializing_binding_kind` and the fixed-point report use it.  An
`Unresolved` fact is skipped by both readers (absence).

### `tools/compiler_probes/probe_row_handle_record_parameter.py`

Plan 3.7's book assertions (`book_assertions`): one row handle for the
Indexed value in `sync` with 2 inbound edges; `center`'s two formals bound
`caller_storage` at the one callsite (no Unresolved); two
`linked_caller_member` rows with 4 inbound edges each; zero
`result_storage_binding` rows for `center`'s record formals; zero MINTED
ids without a mint record in `sync` and `center` (7 before the record-abi
minter).  Exits 1 on any failure.

## Verified

- `py_compile` on every edited file; CRLF kept (fortran_c_shell,
  identity_concordance, concordance_declarations, tensor_ssa_lowering, the
  probe); `transformation_priority.py` was and is LF; ASCII only (the
  probe's two pre-existing non-ASCII bytes are in its old docstring).
- Gate probes all green: `probe_row_handle_record_parameter` (`failures: []`),
  `probe_struct_intake`, `probe_branch_written_field`,
  `probe_scalar_write_only_arm`, `probe_record_in_tuple_return`,
  `probe_planner_specialization_chain`, `probe_control_binding_chain`;
  `probe_scalar_native_correctness` 14/14 (`failures: 0`).
- Audit, seven first lines unchanged: view 493/24/0, toplevel 322/31/1,
  energy 490/22/0, controller 528/27/1, controller_untyped 531/27/5,
  mapping 24/2/0, oscillator 331/22/0.  `unsourced:` fact / identity totals
  before -> after: view 4101/48 -> 4088/48; toplevel 2775/35 -> 2818/10;
  energy 3777/55 -> 3794/46; controller 4530/62 -> 4530/39;
  controller_untyped 4422/62 -> 4421/39; mapping 195/1 -> 195/0;
  oscillator 2866/2 -> 2932/0.  Controller's `argument_binding` group went
  from 28 raw rows (raw_primitive) to 1 cell (frame_binding, an Unsourced
  post); the fact totals rise where a newly declared page is counted per
  cell instead of per raw row (as the step-8 lane observed).
- `measure_completeness.py mapping controller`, before -> after: mapping
  565 cells, DERIVED 257 (45.5%), mint 112, unsourced 203, pages 114
  (declared 68 / undeclared 46), MINTED 15 with record 14 -> 618 cells,
  DERIVED 296 (47.9%), mint 126, unsourced 203, pages 117 (declared 87 /
  undeclared 30), MINTED 15 with record 15; controller 20445 cells, DERIVED
  14603 (71.4%), mint 1125, unsourced 6714, pages 142 (75 / 67), MINTED 141
  with record 79 -> 22284 cells, DERIVED 15988 (71.7%), mint 1498,
  unsourced 6836, pages 147 (107 / 40), MINTED 141 with record 102.
- Caveat: the before run was the clean tree at ad235ce3; the after run is
  the shared tree, which also carries lanes S6 / step-5 in-progress edits
  (`topological_reducer.py`, `control_source.py`, `glsl_deployment_strategy.py`,
  `loop_composer.py`, `ssa_record_return_state.py`, `graph_express2.py`,
  `probe_branch_written_field.py`).  The unchanged first lines and the
  green probes were taken with those edits present.

## Not done (plan 90 section 3.4, by item)

- E7.9 `SEQUENCE_RESIDENCY`: page declared; none of the 104
  `resident_by_value` / `kind_by_value` writes routed (they run in the
  sequence helpers before the function exists, keyed by graph ids; the
  helper `post_residency(function, value, resident, kind, sources)` is the
  next edit, one helper at a time, `_sequence_concat_ops` first).
- E7.11 `alias_application_concordance`, `scheduled_call_argument`,
  `call_link_order_concordance`, `kernel_by_value_formal_concordance`,
  `phi_edge_projection_placement_concordance`, the two prune pages,
  `member_formals`: declared, writers still raw.
- E7.12 remaining mints: 43 textual `GLOBAL_MONOTONIC_IDS.mint()` /
  `mint_compiler_value_id()` sites outside `materialize_parameter_record_abi`
  and the routed frame-loop sites (output index / address scaffolding,
  structural folds, constructor remaps, instance-pool strides, projected
  element addresses, child pool columns, the `node_id` for a missing
  aggregate position -- `AGGREGATE_POSITION_MISSING` is declared, not yet
  posted -- and the `freshen` site).  `mint_compiler_value_id` is not
  deleted: it routes when handed a function.  S6's three functions
  (`materialize_program_abi_record_literals`, `materialize_record_phis`,
  `materialize_loop_record_phis`) were not touched.
- `numeral_leaf_width_concordance` in `allocate_result_storage` stays a
  raw `concord` (S6 declared no `NUMERAL_LEAF` fold page).
- `_publish_concordant_function_aliases`: only `coalesce_record_field_storage`
  passes `sources`; the record-projection block,
  `_concord_unbound_variant_rows`, the two `_retire_*` and the other
  callers still post Unsourced(FRAME_SOURCE_CELL_ABSENT) per alias.
- `record_parameter_value`'s `RECORD_FIELD_RESIDENT` row keeps
  `min(parameter_ids)` as its VALUE_ID (S6 declared the field as VALUE_ID,
  so plan R6.4's cell swap does not fit the declared shape).
- Observed, not fixed: in the probe, `linked_caller_member` for `center`'s
  `orientation` formal records one `items[].orientation.column` value while
  `argument_binding` at the same callsite binds a later-minted one; both are
  on the book now (REVISE history), the pass that re-mints the column after
  the link is the next thing to read.

## Exact next edit

`_sequence_concat_ops` (fortran_c_shell.py, `resident_by_value` /
`kind_by_value` writes): add `post_residency(function_name, value_id,
resident_id, kind, cells)` posting `sequence_residency` REVISE
DERIVED(the value's `canonical_value` cell under the graph's
`lexical_read_scope`, the operation node's cell), replacing each paired
dict write with one call; keep the dicts as read views built from the
page's scope rows.
