# Continuation: step 6, record materialization and return versions

Lane S6, 2026-10-01.  Plan `90_plan_steps6_8_records_linker_tables.md`
sections 1-2; design sections 2, 6, 7.  Functions are named, never line
numbers; no compiler numberings appear.  Nothing committed.  Lane S7 edited
the linking code of `fortran_c_shell.py` concurrently; its hunks are in the
same working tree and are not described here.

## Decisions in force (from the lead)

- Plan 90's SSA_VALUE_IDENTITY IS step 5's `ssa_value` page: every value a
  record pass mints is a NOVEL row on `ssa_value` under the function's
  control scope (`metadata["tensor_shape_concordance_scope"]`, the scope the
  control builder minted; `function.name` for a function no control lowering
  built).  The step-5 `SSAValueFact(dtype, shape, origin)` is the fact; the
  transform on the mint page says what kind of value it is.
- Several sources become one `cell_set` row (step 5's page, through
  `precompile_to_ssa._mint_ssa_id`); no variadic transforms.
- `RESIDENT_CHOSEN_BY_ORDER` is declared as its own Reason (the writer,
  `coalesce_record_field_storage`, is S7's; not routed here).

## What changed

### `src/compiler/concordance_declarations.py` (Steps 6-8 section, after the step-8 block)

Stages `RECORD_LITERAL_MATERIALIZATION`, `RECORD_PHI_EXPANSION_STAGE`,
`RECORD_RETURN_LAYOUT_STAGE`, `OUTPUT_IDENTITY_STAGE`,
`RECORD_ABI_MATERIALIZATION`, `STRUCTURAL_RECOVERY`, `RECORD_RETURN_REPAIR`.
Transforms (arity 1) `RECORD_FIELD_PHI`, `PROGRAM_ABI_DEFAULT`,
`OPTIONAL_PRESENCE`, `OPTIONAL_INACTIVE_PAYLOAD`, `LOOP_RECORD_HEADER`,
`LOOP_RECORD_PROJECTION`, `FRESHEN`.  Reasons `PLANNER_OUTPUT_UNROUTED`,
`LAYOUT_MEMBER_NOT_YET_DEFINED`, `LITERAL_FIELD_DEFERRED`,
`RESIDENT_CHOSEN_BY_ORDER`.  New pages `record_return_layout`
`(function_scope, record) -> tuple` (REVISE) and `record_phi_expansion`
`(function_scope, record_phi, field, slot) -> Ref` (CONCORD).  Re-declared
with the shapes written today: `record_field_layout_concordance`,
`loop_record_layout_concordance`, `loop_record_schema_concordance`,
`numeral_record_literal_concordance` (one page, third element a LABEL: a
field name or a status), `record_field_decomposition`,
`program_abi_keyed_row_record`, `record_field_resident_concordance` (fact
`object`: int or Unresolved), `record_field_storage_concordance`,
`numeral_leaf_materialization_concordance`, `numeral_return_leaves_concordance`,
`output_identity_concordance`, `record_return_phi_input_concordance`
(census-75 draft shape: the phi element is the result VALUE_ID, so a row is
keyable without a phi cell), `record_phi_temporal_fallback_concordance`.

Not declared: `FUNCTION_SCOPE` (the `ssa_value` rows are keyed by the scope
string directly; a `function_scope` page would re-derive it from the
`scope_registry` row the builder minted -- step 8's `table_owner` can derive
from that row instead), `numeral_leaf_width_concordance` (two row shapes on
one name today; the second writer is S7's `allocate_result_storage`).

### `src/compiler/ssa_record_return_state.py`

- Helpers (module level): `function_scope_of`, `ssa_value_identity_cell`,
  `identity_cells`, `mint_ssa_value` (NOVEL `ssa_value` through
  `precompile_to_ssa._mint_ssa_id`; no operand cell -> the function root),
  `record_descriptor_cell`, `record_member_cell`, `post_record_return_layout`
  (REVISE DERIVED(descriptor cell, each layout id's cell); unchanged layout
  posts nothing; a changed layout with no cell to name falls back to
  `Unsourced(LAYOUT_MEMBER_NOT_YET_DEFINED)` through `_post_or_unsourced`),
  `assign_record_descriptor` (`records.assign(..., sources=)`, raw on
  refusal).
- `scalar_return_field_versions.decide`: the three empty-read exits derive
  from `(phi_cell,)`; `Unsourced` has left the function.  The reduction
  scope is left in `metadata["record_return_state_scope"]` so
  `publish_scalar_record_return_fields` keys the same selection rows over
  the receipt view.
- `publish_scalar_record_return_fields`: `lookup(..., phi_cell=, position=)`
  with the `record_phi`'s identity cell; the Cast is NOVEL
  (`RECORD_RETURN_FIELD_CONVERSION`) from the selection cell (else the
  operand's cell).
- `publish_inout_scalar_return_snapshots`: the re-sliced descriptor goes
  through `assign_record_descriptor`; `record_return_layout` posted.
- `freshen_redefined_ssa_objects`: the clone is NOVEL(`FRESHEN`) from the
  redefined value's cell.

### `src/compiler/fortran_c_shell.py` (this lane's functions only)

- `_publish_concorded_output_identities`: `output_identity_concordance`
  REVISE DERIVED(the alias's and the result's `ssa_value` cells), raw when
  neither has a row; the disagreement raises stay.
- `_concord_record_return_phi_inputs(..., function=)`: a row posts REVISE
  DERIVED(selection cell, chosen value's cell); the same decision again
  writes nothing; the hand-written changed-source check stays for the raw
  path (a call without `function`, as the unit test makes); a post the api
  refuses (candidate or reason changed with no changed source) is written
  raw, so the audit lists it.
- `materialize_record_phis`: `phi_cell` is `ssa_value_identity_cell` of the
  record Phi (`record_phi`, read from the attribute, else the result's
  accounting) -- never an instruction attribute; the first expansion pass
  passes the record Phi's cell too.  Per-field Phi NOVEL(`RECORD_FIELD_PHI`)
  from the `cell_set` of the record Phi's cell and each incoming member's
  `record_member` cell (after the members-pending check);
  `record_phi_expansion` CONCORD DERIVED(the same cells);
  `record_field_layout_concordance` CONCORD DERIVED(field Phi cells, member
  cells) with the incumbent-disagreement raise kept;
  `loop_record_layout_concordance` CONCORD DERIVED(incumbent descriptor
  cell, merged ids); `table.register(..., sources=)` /
  `assign_record_descriptor`; `record_return_layout` per merged record.
  `select_return_arguments`: the Cast NOVEL from the selection cell;
  `RECORD_DESCRIPTORS_DIFFER` reads the two descriptor cells (DERIVED).
- `materialize_loop_record_phis`: `loop_record_schema_concordance` CONCORD
  DERIVED(initial and updated descriptor cells); projection id
  NOVEL(`LOOP_RECORD_PROJECTION`, schema cell); header id
  NOVEL(`LOOP_RECORD_HEADER`, the two descriptors' cells);
  `register(..., sources=)`.
- `materialize_program_abi_record_literals`: default Const
  NOVEL(`PROGRAM_ABI_DEFAULT`) from the field's `class_field_declaration`
  cell (class identity = abi record identity) else the record's
  `contract_demand` PARAMETER_RECORD row else the function root; presence /
  inactive payload NOVEL from the payload's cell; the coefficient-field row
  DERIVED(child descriptor cell); `"deferred"` is
  `Unresolved(LITERAL_FIELD_DEFERRED, read=<declaration cells of the
  missing fields>)` (`complete_linked_literals` reads the row's presence,
  not its fact, so it is unchanged); `"completed"` DERIVED(descriptor cell,
  field value cells); `register(..., sources=)`; `record_return_layout`.

### `tools/compiler_probes/probe_branch_written_field.py`

Link 6 counts the return-merge Phis in `step` and asserts selection rows
only when one exists.  Observed: this program's two `return m` fold into
one exit; the finished `step` holds NO Phi at all, so no selection row can
exist and the probe says so.  Two variants tried in the scratchpad (a
constructed record in the terminal arm; three return sites with two
constructed records) both lower to `conditional_result` record Phis: the
per-field Phis, `record_phi_expansion`, `record_field_layout`,
`record_return_layout` and `output_identity` rows all post with edges, but
no `return_merge` of records arises, so `record_return_field_selection`
stays empty.  The `controller` audit case has one `return_merge` Phi
(`_restore_type`, a scalar).  The twelve Reasons are wired (every exit of
`lookup` now has a keyable row when the record return-merge Phi exists)
but no seconds-long case on this tree exercises them; the trigger remains
the full dt-system lowering, which the user launches.

## Verified

- `py_compile` of the three files; CRLF only, ASCII only.
- Probes: `probe_struct_intake`, `probe_branch_written_field`,
  `probe_scalar_write_only_arm`, `probe_record_in_tuple_return`,
  `probe_planner_specialization_chain`, `probe_control_binding_chain` all
  ok; `probe_scalar_native_correctness` failures: 0.
- Fresh-book checks: `_concord_record_return_phi_inputs` raw path matches
  the unit test's fact; with `function=` the row has one inbound edge from
  the chosen value's `ssa_value` cell; `mint_ssa_value(FRESHEN)` records
  its mint from the source cell; zero unsourced rows.
- Audit, seven cases, first lines identical to the baseline (view 0,
  toplevel 1, energy 0, controller 1, controller_untyped 5, mapping 0,
  oscillator 0 findings; same row and function counts).  `unsourced:`
  before -> after: view 4101/48 -> 4088/48, toplevel 2775/35 -> 2816/22,
  energy 3777/55 -> 3794/49, controller 4530/62 -> 4518/48,
  controller_untyped 4422/62 -> 4409/48, mapping 195/1 -> 195/0,
  oscillator 2866/2 -> 2932/0 (facts/identities; the after numbers include
  S7's concurrent edits and the unit change of newly declared pages).
- Measurement (`measure_completeness.py mapping controller`), before ->
  after: mapping DERIVED 45.5% -> 47.4%, tagged unsourced 35.9% -> 33.2%,
  MINTED ids with a mint record 14/15 -> 15/15, declared pages 68 -> 84;
  controller DERIVED 71.4% -> 71.6%, unsourced 32.8% -> 30.8%, MINTED ids
  with a mint record 79/141 -> 93/141, declared pages 75 -> 104.

## Not done (plan 90, step 6)

- E6.8 (`materialize_parameter_record_abi`, `coalesce_record_field_storage`
  and its `RECORD_FIELD_RESIDENT` Unresolved), E6.9's
  `recover_structural_source_outputs`, E6.10 (`complete_linked_literals`),
  E6.11 (the linking-loop layout writers): S7's territory or outside this
  lane's list; the pages are declared.
- E6.12 read views: `metadata["record_return_layouts"]` and the sibling
  channels are still written beside the pages.
- `repair_non_dominating_return_phi_inputs.cloneable`'s clone mint is still
  raw (no transform named for it in plan 90).
- `FUNCTION_SCOPE`, `NUMERAL_LEAF` (fold of the three numeral pages) and
  the `numeral_leaf_width_concordance` split are not declared.

## Exact next edit

Find or build a seconds-long source whose record return merge survives as
a `return_merge` Phi (census 50's `step_with_dt_control_used` shape with a
record and a scalar in one tuple; the probe docstring records why the
`(m, dt)` form was rejected by the full-native contract), lower it, and
read `record_return_field_selection`: every row must have an inbound edge
and `_concord_record_return_phi_inputs` must show the selection cell as a
source.  Then E6.12: `record_return_layouts` as a read view of
`record_return_layout.scope_rows(function scope)`.
