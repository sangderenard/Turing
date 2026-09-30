# Plan, step 3: the reducer's field state goes on the book

Read-only planning lane, 2026-09-30.  Everything below was established by
`git grep`/`sed`/`awk` on the current tree; nothing was run.  Code is named
by function, never by line.  No compiler numberings appear here.

Rests on: `docs/CONCORDANCE_SINGLE_API_DESIGN_2026-09-30.md` sections 2 and 6
(the one call `Concordance.post(page, row, fact, *, stage, provenance,
mode)`, `Ref = (page, row, column)`, `Unresolved(reason, read)` as the only
way to record "could not decide", `Unsourced(reason)` behind the latch), and
`docs/concordance_census/50_causal_chain_worked_example.md` (stage B; section
2 "why the branch write did not reach the phi"; section 3 posts S3 to S9 and
S11 to S15).  Step 1 (api, registry, latch) and step 2 (ingestion roots) are
other lanes; this file assumes their pages exist and names what it needs
from them in section 8.

Vocabulary used below: "cell" = one `Ref`; "row" = a page row; "post" = one
`Concordance.post` call.  Page, Stage, Transform, Reason names are
registry OBJECTS (design 6, decision 3); the identifiers written here in
capitals are the names those objects should carry, not strings to compare.

## 0. What is being moved, observed in the code

`_normalize_lexical_values` (topological_reducer.py) opens four private
dicts beside the read scope it mints:

- `identity_bindings: name -> [node id, ...]`  (published as
  `graph["identity_table"]` at the canonical relabel; step 2 owns the name
  histories; step 3 owns only the two FIELD appends, see 2.4)
- `static_attribute_values: (id(python object), attr) -> node id`
- `attribute_effect_nodes: (receiver node id, attr) -> node id`
- `attribute_value_nodes: (receiver node id, attr) -> node id`

Writers of the two attribute dicts, all inside that function's closures:

| writer | what it writes | value it stores |
|---|---|---|
| `resolve_expression`, `ast.Attribute` branch, resolved receiver | `attribute_value_nodes.setdefault(key, GetAttr node)` ("Observation does not assign a new field version") | the read node, only if no state yet |
| `resolve_expression`, same branch | reads `attribute_effect_nodes[key]` to add the `after_write` ordering operand through `_replace_inputs` | (read) |
| `resolve_expression`, static receiver | reads `static_attribute_values[(id(obj), attr)]`, redirects the read to it | (read) |
| `bind_target`, `ast.Attribute` target | `attribute_effect_nodes[key] = SetAttr node`; `attribute_value_nodes[key] = RHS node`; `static_attribute_values[(id(obj), attr)] = RHS node` when the receiver is static; `identity_bindings["<name>.<attr>"].append(SetAttr node)`; `record_ingestion_definition(...)` | effect = SetAttr, value = RHS |
| `bind_target`, `ast.Subscript` target whose base is `obj.field` | both dicts `[ (outer receiver, field) ] = IndexedStore node` | effect = value = the store |
| `reduce_statement`, `ast.If` | snapshot both dicts before; run body; snapshot; restore; run else; snapshot; restore; then EITHER copy one arm's dicts back (a terminal other arm) OR merge per field key (below) | -- |
| `reduce_statement`, `ast.If` merge loop | for a key first seen inside an arm: mint a GetAttr `initial` (`initial_record_field_state`), seed `before` and the arm that has no effect; for every key in `before`: equal arms -> `attribute_value_nodes[key] = body value`; non-int test -> `continue` (state stays at the pre-branch value, the arm write is dropped); else mint `Phi` (test, body, orelse) with `initial_value_id`, `source_conditional_id`, `record_field_state`, and set BOTH dicts to the Phi; `identity_bindings[binding_name].append(Phi)` | value = effect = Phi |
| `reduce_statement`, `ast.Return` | `graph["return_slot_values"][span] = slot ids`; `graph["return_record_field_states"][span] = ((receiver, field, value) for every dict entry whose receiver is a slot)`; `graph["return_container_kinds"][span]` | copies of dict values |
| the canonical relabel (tail of `_normalize_lexical_values`) | rewrites `return_slot_values` and `return_record_field_states` through `mapping` in place; publishes `identity_table` from `identity_bindings`; re-concords `lexical_read_binding` rows under the canonical scope; copies `source_value_class_concordance` rows | -- |

Observed, not in the task statement: the `ast.For` / `ast.While` branch of
`reduce_statement` snapshots and merges the NAME environment
(`before_loop`, `loop_carried_bindings`, break/continue site bindings) but
never touches `attribute_effect_nodes` / `attribute_value_nodes`.  A field
written in a loop body leaves the dicts holding the body's last write as the
post-loop state; there is no field Phi at the loop, and a break site's
bindings hold names only.  In census 50 that is why X3 (after `while True`)
carries B11 (the body merge) as its receipt.  Section 2.3 says what step 3
posts there.

Consumers of the four dicts' contents, outside the reducer:

| file | function | what it reads today |
|---|---|---|
| glsl_deployment_strategy.py | `_ordinary_conditional_control_programs` | every Phi whose `source_conditional_id` is this conditional: copies `(body, orelse, initial_value_id, merged)` into `ConditionalBlock.carried_aliases`; `_record_field_state_keys` on the merged id; `arm_return_control` reads `return_slot_values[span]` |
| glsl_deployment_strategy.py | the structural-fold return selection (the block that sets `graph.roots = [selected_return_id]`) | prunes `return_slot_values` to the selected sites |
| glsl_deployment_strategy.py | `_fold_callsite_structural_values`, `_alias_projection_to_member`, `_retarget_all_cached_value_ids` | rewrite ids inside `return_slot_values` / `return_record_field_states` in place |
| fortran_c_shell.py | `_class_surface_ssa_program`, the local-record loop after `declared_parameter_records` ("A record produced inside this function ... has no parameter storage cell") | admits each SetAttr as `ScalarFieldWriteBlock(None, ControlExpression, dtype, effect_node_id=SetAttr)`; `continue`s silently when no dtype is provable |
| fortran_c_shell.py | `_class_surface_ssa_program`, `direct_return_ids` | `return_slot_values` values |
| fortran_c_shell.py | `_identity_return_aliases` | `return_slot_values` values |
| fortran_c_shell.py | `_class_surface_ssa_program.materialize_record_phis` / `select_return_arguments` | calls `scalar_return_field_versions`; mints the `Cast` |
| precompile_to_ssa.py | `_ControlSSABuilder.__init__` (`index_scalar_field_effects`) | `ScalarFieldWriteBlock.effect_node_id` -> destination |
| precompile_to_ssa.py | `_ControlSSABuilder._lower`, `ScalarFieldWriteBlock` branch | emits the value, `external_values[effect_node_id] = value` |
| precompile_to_ssa.py | `_ControlSSABuilder.lower_conditional` | `carried_snapshots[initial]`; `true_carried[initial] = external_values.get(true_id, snapshot)`; same for false; emits the `conditional_carried` Phi |
| ssa_record_return_state.py | `scalar_return_field_versions` | `return_record_field_states`, `return_slot_values`; writes `function.metadata["record_return_state_receipts"]`; the `lookup` closure |
| ssa_record_return_state.py | `publish_scalar_record_return_fields` | rebuilds a graph from the receipts and calls `scalar_return_field_versions` again |
| loop_composer.py | the `ast.Return` handling in the loop describer | `return_slot_values[span]` |

Counts: 14 dict-writing sites in the reducer (plus 8 snapshot/restore
blocks in the `ast.If` branch that become scope handling, not posts), 3
return-receipt writes, 2 relabel rewrites; 19 reader sites in 5 consumer
files.

## 1. The field-state page

### 1.1 Page `REDUCER_FIELD_STATE`

```
Page REDUCER_FIELD_STATE
  row_fields = (
    RowField(reduction_scope, SCOPE),      # the read scope minted by _normalize_lexical_values:
                                           #   (read_scope, "ingestion") while reducing,
                                           #   read_scope after the canonical relabel
    RowField(receiver, VALUE_ID),          # the receiver node's canonical or ingestion id
    RowField(field, NAME),                 # the attribute name
  )
  fact_type = FieldState
  mode = REVISE                            # each write is the next version; history is the chain

FieldState = (kind: FieldStateKind, value: Ref, effect: Ref)
  FieldStateKind in {OBSERVED, WRITTEN, ELEMENT_WRITTEN, MERGED, ARM_SELECTED, LOOP_EXIT}
  value  = the cell that identifies the node holding the field's current value
  effect = the cell that identifies the node whose execution last touched the field
```

`value` and `effect` are Refs to NODE IDENTITY cells, never ids.  A node's
identity cell is the step-2 row for that node: the span root row for an
authored construct, or the NOVEL row `new_node` posts for a reducer-minted
node (section 8).  The dicts held ids; the page holds cells, so the same
row can be followed backwards to the AST span that produced the value.

One row per `(scope, receiver, field)`; every write is a REVISE with an
edge.  The column of a revision is the version; "the arm's state" below
always means a specific `(REDUCER_FIELD_STATE, row, column)` cell, which is
what the merge and every later consumer name.

Reading the current state is `latest(row)`; reading the state as of a
snapshot is the cell captured at snapshot time.  `attribute_value_nodes`
and `attribute_effect_nodes` survive as read views over
`latest(row).value` / `latest(row).effect` until every reader is migrated,
then are deleted.

### 1.2 Page `STATIC_ATTRIBUTE_STATE`

`static_attribute_values` is keyed by a Python object address (a
compile-time constant fold over a static parameter binding), not by a graph
value.  It gets its own small page:

```
Page STATIC_ATTRIBUTE_STATE
  row_fields = (RowField(reduction_scope, SCOPE), RowField(receiver_path, LABEL), RowField(field, NAME))
  fact_type = Ref                          # the RHS node identity cell
  mode = REVISE
```

Written by `bind_target` (static receiver) DERIVED(the receiver's
`source_parameter_identity_concordance` row cell when it exists, else the
static binding root row step 2 posts for `_python_bindings`; the RHS node
cell).  Read by `resolve_expression` (static receiver branch) in place of
the dict.  `receiver_path` is `_StaticPythonReference.path`, not `id()`.

### 1.3 Posts, writer by writer (the S-numbers are census 50 section 3)

Stages registered for this step: `REDUCER_ATTRIBUTE_READ`,
`REDUCER_BIND_TARGET`, `REDUCER_INDEXED_STORE`,
`REDUCER_CONDITIONAL_MERGE`, `REDUCER_TERMINAL_ARM`, `REDUCER_LOOP_EXIT`,
`REDUCER_RETURN`, `REDUCER_CANONICAL_RELABEL`.

| writer | post | provenance |
|---|---|---|
| S3 `resolve_expression`, resolved receiver, no state yet | REVISE row `(scope, receiver, attr)` fact `(OBSERVED, value=GetAttr cell, effect=GetAttr cell)` | DERIVED(receiver node cell, field schema cell, GetAttr node cell) with stage `REDUCER_ATTRIBUTE_READ`.  When state exists: NO post (observation is not a version); the `after_write` operand append is recorded by `_set_operands` (section 3), derived from `latest(row).effect` |
| S4 `bind_target`, `ast.Attribute` target | REVISE fact `(WRITTEN, value=RHS node cell, effect=SetAttr node cell)` | DERIVED(receiver node cell, field schema cell, RHS node cell, SetAttr node cell) stage `REDUCER_BIND_TARGET` |
| `bind_target`, `obj.field[i] = v` | REVISE fact `(ELEMENT_WRITTEN, value=IndexedStore cell, effect=IndexedStore cell)` | DERIVED(outer receiver cell, field schema cell, IndexedStore cell) stage `REDUCER_INDEXED_STORE` |
| S5/S6/S7 `reduce_statement` `ast.If` merge, arms differ, runtime test | (a) NOVEL node row for the Phi, `Transform.CONDITIONAL_FIELD_MERGE` arity 3, operands (body state cell, else state cell, test node cell); (b) REVISE field row fact `(MERGED, value=Phi cell, effect=Phi cell)` | (b) DERIVED(body arm state cell, else arm state cell, test node cell, the Phi's NOVEL cell) stage `REDUCER_CONDITIONAL_MERGE` |
| merge, key first seen in an arm | the synthesized `initial` GetAttr is NOVEL `Transform.FIELD_STATE_SEED` operands (receiver cell, field schema cell); the seeded pre-branch state is REVISE fact `(OBSERVED, initial cell, initial cell)` | DERIVED(receiver cell, field schema cell, initial NOVEL cell) |
| merge, `body_value == else_value` | no post: both arms' cells are the same cell; the row's latest already is that cell.  (Today it re-assigns the same id.) |
| merge, `not isinstance(test_value, int)` (today: `continue`, arm write dropped) | REVISE fact `Unresolved(FIELD_MERGE_STATIC_TEST, read=(body state cell, else state cell, test cell))` | the Unresolved carries what was read; readers see absence, the audit sees the drop |
| S8 `ast.If` with one terminal arm (`body_terminal and not else_terminal` or mirror) | for every key whose live-arm cell differs from the pre-branch cell: REVISE fact `(ARM_SELECTED, value/effect = live arm's cells)` | DERIVED(live arm state cell, the conditional's construct row cell) stage `REDUCER_TERMINAL_ARM`.  The terminal arm's last state is NOT lost: it is the cell the terminal arm's own `ast.Return` row derived from (S9), so the "state ends at X2" fact is the return-site row, exactly as census 50 S8 says |
| loop exit (`ast.For`/`ast.While`), observed gap in section 0 | for every key whose body-exit cell differs from the pre-loop cell: REVISE fact `Unresolved(LOOP_EXIT_FIELD_STATE_UNMERGED, read=(pre-loop cell, body-exit cell, loop construct cell))` stage `REDUCER_LOOP_EXIT` | see the decision in section 9: this records what the reducer does today (no merge) as unresolved instead of letting the body's last write stand as the post-loop fact |
| S9 `ast.Return` | see section 2 |
| S10 canonical relabel | for every row of `REDUCER_FIELD_STATE` under `(read_scope, "ingestion")`: post the row under `read_scope` with receiver = canonical id, fact with value/effect Refs re-pointed to the canonical node cells | DERIVED(the ingestion row's latest cell, the step-2 `ssa_identity_tokens` cells for receiver, value node and effect node) mode CONCORD stage `REDUCER_CANONICAL_RELABEL`.  Same shape as the existing `lexical_read_binding` re-concord in that function; nothing is rewritten in place |

Snapshot / restore in the `ast.If` branch: the eight `clear()`/`update()`
blocks stop copying dicts.  "Snapshot before" becomes: for every field row
in scope, remember `latest(row)`'s cell.  "Restore" becomes nothing on the
book (the book is append-only); the arm's posts stay on the row as history
and the merge names the arm cells by their captured Refs.  The read view
`attribute_value_nodes` for "current state" during the else arm is then
NOT `latest(row)` (that would be the body's last version) -- it is the
pre-branch cell.  So the closure keeps one small structure: `current_cell:
dict[row, Ref]`, the reducer's cursor into each row, set on every post and
reset to the snapshot at arm boundaries.  It holds Refs, never ids, and is
never read by a consumer outside the reducer.  This is the one piece of
private state this step keeps, and it is a cursor, not a ledger.

## 2. Return-site pages

### 2.1 Page `RETURN_SITE_SLOT`

```
Page RETURN_SITE_SLOT
  row_fields = (RowField(reduction_scope, SCOPE), RowField(return_site, PAGE_REF), RowField(slot, INDEX))
  fact_type = Ref | Unresolved            # the slot value's node identity cell
  mode = CONCORD
```

`return_site` is the Ref of the step-2 span root row for the returned
expression (the `ast.Return`'s value; the same span the dict key was).
Posted by `reduce_statement`, `ast.Return`, one row per slot, DERIVED(the
slot expression's node cell, the return construct cell) stage
`REDUCER_RETURN`.  A slot whose expression resolves to a non-value (today
`None`) posts `Unresolved(RETURN_SLOT_NOT_A_VALUE, read=(return construct
cell,))`.

### 2.2 Page `RETURN_SITE_CONTAINER`

Row `(SCOPE, PAGE_REF return_site)`, fact `ContainerKind in {TUPLE, LIST,
VALUE}`, CONCORD, DERIVED(return construct cell).  Replaces
`return_container_kinds`.

### 2.3 Page `RETURN_SITE_FIELD_STATE`

```
Page RETURN_SITE_FIELD_STATE
  row_fields = (RowField(reduction_scope, SCOPE), RowField(return_site, PAGE_REF),
                RowField(receiver, VALUE_ID), RowField(field, NAME))
  fact_type = Ref                          # the REDUCER_FIELD_STATE cell current at this return
  mode = CONCORD
```

Posted by `reduce_statement`, `ast.Return`, for every `REDUCER_FIELD_STATE`
row in scope whose receiver has a `RETURN_SITE_SLOT` row at this site (the
filter `receiver in slot_values` becomes the derivation): fact = the
reducer's cursor cell for that row; DERIVED(that field-state cell, the
`RETURN_SITE_SLOT` cell that carries the receiver) stage `REDUCER_RETURN`.
If the cursor cell holds an `Unresolved` (a loop exit, a static-test merge)
the return-site row is `Unresolved(FIELD_STATE_UNRESOLVED_AT_RETURN,
read=(that cell,))` -- the return site says it looked and found no version.

S9 in census 50: X1 derives from S3's cell, X2 from the W3 WRITTEN cell,
X3 (after the loop) from whatever section 2.3's loop-exit decision leaves.

Canonical relabel: both pages re-posted under `read_scope` DERIVED(ingestion
row cell, `ssa_identity_tokens` cells), CONCORD; no in-place rewrite.

`graph["return_slot_values"]`, `graph["return_record_field_states"]`,
`graph["return_container_kinds"]` become read views built from the three
pages (span -> ids) so the 19 reader sites keep working until each is
migrated (section 4); then they are deleted.

## 3. `_set_operands`: every operand change posts its cause

Today `identity_transition` receives `("move", consumer, role, ordinal,
cause)`, `("retire", None, None, None, cause)` and `("fork", source, cause)`
with `cause` a free string naming the caller; an APPEND (a new position with
no move source) records nothing; and when `_operand_position_scope(graph)`
is None the parents are written and nothing at all is recorded.

Change: `cause: str` becomes `cause: Transform` (registered objects, arity
0): `APPEND_OPERAND`, `REPLACE_INPUTS`, `REMOVE_NODE`, `REDIRECT_VALUE`,
`CANONICAL_RELABEL`, `DISSOLVE_EXPR`, `DISSOLVE_RETURN`, `PARAMETER_INPUT`,
`FUNCTION_SUBGRAPH` (reducer callers), `PROJECTION_TO_LEAF`
(`_alias_projection_to_member`), plus one per caller in `loop_composer.py`
and elsewhere (the registry refuses an unregistered cause, which is how the
remaining callers are found).  The `fork_read_scope` cause likewise.

Posts, all on `IDENTITY_TRANSITION` (the existing page, declared in step 1
with row `(SCOPE, VALUE_ID consumer, NAME role, INDEX ordinal)`), stage =
the Transform's stage object, mode REVISE:

| change | fact | DERIVED from |
|---|---|---|
| move `(role, ordinal) -> (role', ordinal')` | `Move(target position)` | (the source position's `IDENTITY_TRANSITION` cell or, for a position never recorded, its `LEXICAL_READ_BINDING` cell; the consumer node cell; the operand node cell) |
| retire | `Retire()` | (the vacated position's cell; the consumer node cell) |
| fork | `Fork(source position)` | (the source position's cell at the other consumer; the consumer node cell) |
| **append** (new position, no move source, no fork) | `Append(operand)` where `operand` is the operand node's identity cell | (the operand node cell; the consumer node cell).  For the `after_write` append in `resolve_expression` the operand cell IS `latest(field row).effect`, so the ordering edge is joined to the field version it orders after |
| position-keyed page follow-ups (`_OPERAND_POSITION_ROW_PAGES`, `_OPERAND_POSITION_FACT_PAGES`) | as today (moved fact / `()` / None) | DERIVED(the `IDENTITY_TRANSITION` cell just posted) -- the follow-up names the transition that caused it |
| `_operand_position_scope(graph) is None` | `Unsourced(NO_OPERAND_POSITION_SCOPE)` on the unsourced page, one per rewrite, with the caller's stage | latched: while OPEN the audit lists every graph that is rewritten before it has a scope; closing the latch proves none remain |

The `same=mapping` path in the canonical relabel posts its moves DERIVED from
the `ssa_identity_tokens` cell of each renamed operand, which is the S10
"morph graph" edge the design asks for.

## 4. Readers: what each consumer reads instead

| consumer | reads today | reads instead |
|---|---|---|
| `_ordinary_conditional_control_programs` (planner) | Phi parents copied to `carried_aliases = (body id, orelse id, initial id, merged id)` | For every Phi under this `source_conditional_id` whose node has a `REDUCER_FIELD_STATE` MERGED revision (found by the reverse index from the Phi's NOVEL cell): post on a new page `CONTROL_CARRIED_FIELD` row `(planning scope, PAGE_REF conditional construct, PAGE_REF field row)` fact `CarriedField(body_cell, else_cell, initial_cell, merged_cell)` -- the four Refs the merge's edge names -- DERIVED(the MERGED cell) stage `PLANNER_CONDITIONAL_CONTROL`, CONCORD.  `ConditionalBlock.carried_aliases` keeps its id tuple (control-program dataclasses are not pages) but each tuple is BUILT from that row (`row_value_id` of each cell), and the block carries the row Ref in a new field `carried_field_rows` so the lowering can get back to the cells.  This is S11: alias -> S5, no id copy.  Name-carried Phis (no field row) are step 5's business and are untouched. |
| `_ordinary_conditional_control_programs.arm_return_control`, `loop_composer` return handling, `_identity_return_aliases`, `direct_return_ids` in `_class_surface_ssa_program` | `return_slot_values[span]` | `RETURN_SITE_SLOT` rows for the site (via the read view first; then `latest` on the page keyed by the site's span root Ref) |
| the structural-fold return selection in glsl | prunes `return_slot_values` | posts `Unresolved(RETURN_SITE_UNREACHABLE, read=(site row cell, the terminal arm's construct cell))` on each pruned site's `RETURN_SITE_SLOT` rows in REVISE mode (an invalidation is an edge, never an erasure) stage `PLANNER_RETURN_SELECTION` |
| `_fold_callsite_structural_values`, `_alias_projection_to_member`, `_retarget_all_cached_value_ids` | rewrite ids in the two dicts | post the new slot / field-state facts under the same rows in REVISE mode DERIVED(the old cell, the alias's `IDENTITY_TRANSITION` cell) stage `PLANNER_PROJECTION_ALIAS`.  The read views then show the new ids; the history shows the move |
| `_class_surface_ssa_program` local-record loop (D1) | scans SetAttr nodes; `continue` when no dtype | for every `REDUCER_FIELD_STATE` WRITTEN/ELEMENT_WRITTEN cell in the function's scope: post `SCALAR_FIELD_WRITE_ADMISSION` row `(function scope, PAGE_REF field-state cell)` fact `Admitted(dtype, field_value_id or None)` DERIVED(the WRITTEN cell, the dtype's source cell: `parameter_value_dtypes` / `constant_values` root row from step 2 or 4, or the node's tensor descriptor cell) stage `LINKER_SCALAR_WRITE_ADMISSION`; when no dtype is provable: `Unresolved(SCALAR_WRITE_NO_DTYPE, read=(the WRITTEN cell,))` instead of `continue` (S12).  `ScalarFieldWriteBlock` gains `field_state_cell: Ref` beside `effect_node_id` |
| `_ControlSSABuilder._lower`, `ScalarFieldWriteBlock` (D2) | `external_values[effect_node_id] = value` | (a) the Const from `lower_control_expression` is NOVEL `Transform.CONTROL_CONST` operands (the literal's node cell) (S13; step 5 generalizes); (b) post `SSA_FIELD_VERSION` row `(function scope, PAGE_REF field-state cell)` fact = the SSA value id DERIVED(the Const's NOVEL cell, the admission cell) stage `CONTROL_SSA_SCALAR_WRITE`, CONCORD (S14).  `external_values[effect_node_id]` stays as a read view during migration |
| `_ControlSSABuilder.lower_conditional` (D3/D4) | `external_values.get(true_id, carried_snapshots[initial])` | section 5 below |
| `scalar_return_field_versions` (E5/E6) | `return_record_field_states` / `return_slot_values`; `record_return_state_receipts` metadata | reads `RETURN_SITE_SLOT` and `RETURN_SITE_FIELD_STATE` for the function's canonical scope; matches the `Br`'s `return_source_value_ids` to a `RETURN_SITE_SLOT` row (the slot ids are `row_value_id` of the slot cells); the receipt's field cell is a `REDUCER_FIELD_STATE` cell; the SSA version is `SSA_FIELD_VERSION` at that cell (or, for a MERGED cell, the `conditional_carried` Phi's `SSA_FIELD_VERSION` row that `lower_conditional` posts).  Its selection is posted, section 6.  `record_return_state_receipts` metadata becomes a read view of the two pages |
| `publish_scalar_record_return_fields` | rebuilds a graph from the receipts | reads the same pages; no rebuilt graph |

## 5. The one bug: the cell the phi-arm lookup must read

Observed chain (census 50 section 2): `bind_target` stores the RHS node as
the field's value and the SetAttr node as its effect; the reducer Phi's
`body` parent is the RHS node; the planner copies that id into
`carried_aliases`; `_ControlSSABuilder._lower` publishes the emitted Const
under `external_values[SetAttr id]`; `lower_conditional` looks up
`external_values.get(RHS id, snapshot)`, misses, and emits the snapshot as
the arm.

Under the pages there is no id to mismatch, because both ends name the same
cell:

- `bind_target`'s WRITTEN revision is ONE cell
  `W = (REDUCER_FIELD_STATE, (scope, receiver, field), column_of_the_write)`
  whose fact holds both the RHS cell and the SetAttr cell.
- The merge's edge rows name `W` as the body arm (section 1.3, S5).
- The planner's `CONTROL_CARRIED_FIELD` row copies `W` as `body_cell`.
- `_ControlSSABuilder._lower` posts `SSA_FIELD_VERSION` at row
  `(function scope, W)`.

So the arm lookup in `lower_conditional` is:

```
for each carried field row R = (body_cell, else_cell, initial_cell, merged_cell):
    for arm_cell in (body_cell, else_cell):
        if arm_cell == initial_cell:
            arm value = carried_snapshots[initial]          # the edge PROVES the arm did not write
        else:
            version = latest(SSA_FIELD_VERSION, (function scope, arm_cell))
            if version is None or isinstance(version, Unresolved):
                post SSA_FIELD_VERSION (function scope, arm_cell) =
                    Unresolved(ARM_VERSION_MISSING, read=(arm_cell, R's cell))
                append SSALoweringShortfall("control", "carried-field-arm-missing", path,
                    naming R's row and arm_cell) and STOP lowering this function
            arm value = the SSA value the version row names
```

The exact cell: `(SSA_FIELD_VERSION, (function scope, <the REDUCER_FIELD_STATE
cell that the merge's DERIVED edge names as that arm>))`.  Never
`external_values[<any graph id>]`, never a snapshot unless the arm cell IS
the initial cell.  The emitted `conditional_carried` Phi then posts its own
`SSA_FIELD_VERSION` row at `(function scope, merged_cell)` fact = the Phi's
SSA value, DERIVED(body version cell, else version cell, the
`CONTROL_CARRIED_FIELD` cell) stage `CONTROL_SSA_CONDITIONAL` (S15), and
`external_values.update(published)` for merged/initial/arm ids stays as the
read view.

In the census-50 program this changes the emitted SSA: the inner
conditional's Phi gets the arm's Const on the true side instead of the
snapshot; the false side is the snapshot because its arm cell equals the
initial cell.  That is the lost `False` write reaching the phi.

## 6. `scalar_return_field_versions`: twelve fallbacks become twelve reasons

Observed: the function returns a lambda yielding `fallback` when there are
no receipts, and the `lookup` closure has ten `return fallback` exits;
`select_return_arguments` (in `_class_surface_ssa_program.materialize_record_phis`)
has one more silent fallback (`selected_arguments.append(argument);
continue` when the source and physical records differ).  That is twelve
sites that choose the descriptor's field value from absence and record
nothing.  (The design says "twelve silent `return fallback` exits"; eleven
of them spell `fallback`, the twelfth is the `continue`.)

Each becomes a post on page `RECORD_RETURN_FIELD_SELECTION`, row
`(function scope, PAGE_REF return-merge Phi identity cell, NAME field, INDEX
position, LABEL predecessor block)`, mode REVISE (a later linking round may
legitimately change the selection when a cell it derives from changed; the
api checks the stamps, which is what turns the observed raise into a legal
revision), stage `RECORD_RETURN_VERSION`:

- success: fact = the selected `SSA_FIELD_VERSION` cell, DERIVED(the
  `RETURN_SITE_FIELD_STATE` cell for this predecessor's site, that version
  cell) (S19).
- otherwise: fact = `Unresolved(reason, read=(...))` with the cells that
  were read.  The caller then keeps the descriptor field value as the
  ARGUMENT, but the page says why, and `_concord_record_return_phi_inputs`
  (S21) records its input DERIVED(this selection cell), so a later round's
  different choice is a revision with a cause instead of a disagreement.

The twelve Reason objects, in the order the exits occur:

| Reason | exit today |
|---|---|
| `NO_RETURN_FIELD_RECEIPTS` | no `return_record_field_states` at all: the lambda |
| `PREDECESSOR_BLOCK_EMPTY` | `block is None or not block.instrs` |
| `PREDECESSOR_NOT_A_RETURN_EDGE` | terminator has no `return_source_value_ids` |
| `SITE_WITHOUT_FIELD_STATE` | no site matches the slots, or some matching site has no state for `(receiver, field)` |
| `SITES_DISAGREE_ON_VERSION` | `len(candidates) != 1` over sites |
| `VERSION_NOT_UNIQUELY_DEFINED` | the version id has zero or several SSA definitions |
| `VERSION_NOT_CONST_OR_CARRIED_PHI` | definition op is neither `Const` nor a `conditional_carried` Phi |
| `VERSION_IS_FORMAL_SHAPED_OR_MISTYPED` | formal id, non-scalar shape, or dtype differs without a Boolean-leaf Phi tree |
| `INTERVENING_CALL_NOT_READONLY` | a call on the owner-to-predecessor path touches an alias and cannot be proven read-only |
| `INTERVENING_STORE` | a store/atomic on that path targets the slot or an alias, or an instruction redefines the slot |
| `VERSION_DOES_NOT_DOMINATE_RETURN` | the dominator walk from the predecessor never reaches the owner block |
| `RECORD_DESCRIPTORS_DIFFER` | `select_return_arguments`: source and physical record identity or fields differ |

The minted `Cast` (S20) is NOVEL `Transform.RECORD_RETURN_FIELD_CONVERSION`
operands (the selection cell,), replacing `GLOBAL_MONOTONIC_IDS.mint()` in
both `select_return_arguments` and `publish_scalar_record_return_fields`
(step 6 finishes the rest of that family; these two sites are on this
step's path).

## 7. The seconds-long probe

`tools/compiler_probes/probe_branch_field_return.py` (new; same shape as
`probe_annotated_scalar_parameter.py`: `ExtractionContract(...).with_program_abi`,
`lower_ast_source_to_ssa(source, "step", ..., runtime_closure_only=True)`).
Source, modelled on `Metrics.hard_failure` in
`dt_controller.step_with_dt_control_used` but with no loop, so the loop-exit
decision (section 9) does not enter:

```
from dataclasses import dataclass

@dataclass
class Metrics:
    hard_failure: bool = False
    value: float = 0.0

def step(m: Metrics, rejected: bool, dt: float) -> tuple[Metrics, float]:
    if bool(m.hard_failure):
        return m, dt * 0.5
    if rejected:
        m.hard_failure = False
        dt = dt * 0.25
    return m, dt
```

One field, one read before every write (R1), one write on one arm (W1),
two return sites (X1 inside a terminal arm, X2 at the tail after a
non-terminal conditional).  `program_abi.records` declares `Metrics` with
`hard_failure` scalar bool mutable.

What the book must show afterwards (read with `render_identity_book` and
the audit tool's `unsourced-fact` finding):

1. `REDUCER_FIELD_STATE` row `(scope, m, "hard_failure")` with three
   revisions: OBSERVED (R1), WRITTEN (W1), MERGED (the `if rejected` Phi),
   each with inbound edges; the MERGED edge names the WRITTEN cell as body
   and the OBSERVED cell as else.
2. `RETURN_SITE_FIELD_STATE` rows for X1 -> the OBSERVED cell and X2 -> the
   MERGED cell, each DERIVED from its `RETURN_SITE_SLOT` row for slot 0.
3. A rooted chain, followed backwards with the reverse index from X2's row:
   MERGED -> WRITTEN -> RHS literal span root and the field schema cell
   (step 2's `Metrics.hard_failure` schema row, NOVEL from the `AnnAssign`
   span) and the receiver's parameter root.  No hop ends in a bare id.
4. `CONTROL_CARRIED_FIELD` one row DERIVED(MERGED); `SSA_FIELD_VERSION` at
   the WRITTEN cell (the arm Const) and at the MERGED cell (the
   `conditional_carried` Phi) with the Phi's edges naming the WRITTEN
   version and the initial; NO `ARM_VERSION_MISSING`.
5. `RECORD_RETURN_FIELD_SELECTION` two rows: X1 either a version cell or
   `Unresolved(VERSION_NOT_CONST_OR_CARRIED_PHI)` (the OBSERVED cell is a
   GetAttr, not a Const/Phi -- that reason is the honest record of today's
   behaviour); X2 -> the MERGED version cell, DERIVED.  Zero raises from
   `_concord_record_return_phi_inputs`.
6. The emitted SSA's `conditional_carried` Phi for the field has arguments
   (Const False, snapshot), not (snapshot, snapshot).  Assert this in the
   probe by reading the function's instructions: exactly one `Phi` with
   `binding == "conditional_carried"` whose two args differ.

`unsourced-fact` delta on this probe, counted per fact the census
attributes to this step: B4, B7, B9 (one merge here), B13 (two sites), B14
(the relabel of those), C3, C4, D1, D2, D3, E5, E6, E7's mint and the
descriptor-value-as-chosen at E8 stop being unsourced: 14 facts, plus one
`identity_transition` append per `_set_operands` append in the function
(the R1 `after_write` operand at least; the exact append count is read from
the probe's own book, not predicted here).  Facts that stay unsourced on
this probe belong to steps 4 to 7 (D5, D7, D8, E1 to E4, F1 to F3).  The
probe prints the before/after `unsourced-fact` count by stage so the drop
is measured, not asserted.

Also required green after the step: `probe_annotated_scalar_parameter.py`
(no fields, no conditionals: proves the pages cost nothing when empty) and
`probe_struct_intake` per design 6.4.

## 8. What step 3 needs from step 2

Cells that step 3's DERIVED posts name and that must therefore exist as
rows before this step's first post:

1. A NODE IDENTITY row per graph node, on the page step 2 declares for
   `ssa_identity_tokens` / span roots: for an authored node the span root
   row (NOVEL per source construct); for a reducer-minted node
   (`new_node`, which today advances `value_id_watermark`) a NOVEL row whose
   transform and operands the minting site supplies -- step 3 supplies them
   for the field Phi and the seeded initial GetAttr; `new_node` must accept
   a `Novel(...)` and post it, and the returned Ref is what the field-state
   facts store.  Without this there is no `value` / `effect` Ref to write.
2. The `ssa_identity_tokens` NOVEL rows (ingestion id -> canonical id):
   the relabel re-posts of sections 1.3 and 2 derive from them.
3. Field schema rows (`map_ir` `objects` / `class_definitions` /
   `program_abi.records` field entries, census 50 A1) keyed by (class
   identity, field name): every WRITTEN/OBSERVED post names the schema cell.
   If step 2 cannot resolve the receiver's class (the census-50 receiver is
   a call result whose class comes from `source_value_class_concordance`),
   the field-state post derives from the receiver cell and the field NAME
   only and posts an additional `Unresolved(FIELD_SCHEMA_UNKNOWN,
   read=(receiver class cell,))` on the schema page, so the missing root is
   visible rather than assumed.
4. Return-site span root rows (one per `ast.Return` value expression):
   `RETURN_SITE_*` rows use them as `PAGE_REF`.
5. Conditional and loop construct rows (span roots for `ast.If`,
   `ast.For`, `ast.While`): the merge, terminal-arm and loop-exit posts name
   them; the planner's `CONTROL_CARRIED_FIELD` row is keyed by one.
6. `identity_table`'s page: the two field appends in `bind_target`
   (`"<name>.<attr>" -> SetAttr`) and the merge (`binding_name -> Phi`) post
   onto step 2's name-history page DERIVED(the WRITTEN / MERGED cell); the
   `record_ingestion_definition` call for the field identity derives from
   the same cell.  Step 3 does not declare a name-history page.
7. Static binding roots for `_python_bindings` (the receiver of
   `STATIC_ATTRIBUTE_STATE`).

## 9. Ordered edit list

Each item names the function edited and what changes in it.  Order is
chosen so every post's sources exist before it is made and every reader has
a read view until it is migrated.

1. `identity_concordance.py` registry (step 1 owns the file; step 3 adds
   entries): declare pages `REDUCER_FIELD_STATE`, `STATIC_ATTRIBUTE_STATE`,
   `RETURN_SITE_SLOT`, `RETURN_SITE_CONTAINER`, `RETURN_SITE_FIELD_STATE`,
   `CONTROL_CARRIED_FIELD`, `SCALAR_FIELD_WRITE_ADMISSION`,
   `SSA_FIELD_VERSION`, `RECORD_RETURN_FIELD_SELECTION`; the fact types
   `FieldState`, `CarriedField`, `Admitted`, `ContainerKind`; the stages of
   sections 1.3, 3, 4, 6; the transforms of sections 1.3, 3, 4, 6; the
   reasons of sections 1.3, 2, 5, 6, 8.  Declare `IDENTITY_TRANSITION`'s
   fact as `Move | Retire | Fork | Append` objects.
2. `topological_reducer.py`, `_normalize_lexical_values` body: replace the
   four dict declarations with the `current_cell` cursor plus read-view
   accessors that consult the pages; keep the names
   `attribute_value_nodes` / `attribute_effect_nodes` /
   `static_attribute_values` as those read views so the closures compile
   unchanged until each is edited.
3. `new_node` (closure): accept `provenance: Novel` and post the node row;
   return the Ref alongside the id (or store it on the node data under one
   key, `identity_cell`, that every later `NodeRef(node_id)` lookup reads).
   All existing `new_node` callers in this function pass a Novel with the
   transform they perform; the ones in this step are the field Phi and the
   seeded initial GetAttr.
4. `bind_target`, `ast.Attribute` target: post WRITTEN (S4) and, for a
   static receiver, `STATIC_ATTRIBUTE_STATE`; post the two name-history
   appends DERIVED(WRITTEN cell).  `ast.Subscript` target on `obj.field`:
   post ELEMENT_WRITTEN.
5. `resolve_expression`, `ast.Attribute`: static receiver reads
   `STATIC_ATTRIBUTE_STATE`; resolved receiver posts OBSERVED when no row
   exists; the `after_write` operand is appended from `latest(row).effect`
   and goes through `_set_operands` with cause `REPLACE_INPUTS` (unchanged
   call) so section 3's Append post records it.
6. `_set_operands`: `cause: Transform`; post Append for new positions;
   post follow-ups DERIVED(the transition cell); `Unsourced(
   NO_OPERAND_POSITION_SCOPE)` when scope is None.  Update every caller's
   `cause=` argument (`_append_operand`, `_replace_inputs`, `_remove_node`,
   `_redirect_value`, the canonical relabel, the Expr/Return dissolve sites,
   the function-subgraph filter and parameter-input sites in
   `topological_reducer.py`; `_alias_projection_to_member` in
   `glsl_deployment_strategy.py`; the `loop_composer.py` callers;
   `fork_read_scope`).  The registry refuses any missed caller.
7. `reduce_statement`, `ast.If`: replace the eight snapshot/restore blocks
   with cursor capture/reset; merge loop posts per section 1.3 (Phi NOVEL +
   MERGED REVISE; seeded initial; static-test Unresolved; equal arms no
   post); terminal-arm branch posts ARM_SELECTED.
8. `reduce_statement`, `ast.For` / `ast.While`: at loop exit post the
   LOOP_EXIT Unresolved for every field row whose cursor differs from the
   pre-loop capture (the decision below must be taken first).
9. `reduce_statement`, `ast.Return`: post `RETURN_SITE_SLOT`,
   `RETURN_SITE_CONTAINER`, `RETURN_SITE_FIELD_STATE`; stop writing the
   three graph dicts directly; install the three read views on
   `graph.G.graph` (a mapping proxy built from the pages at access time).
10. The canonical relabel tail: re-post the field-state and return-site rows
    under `read_scope` DERIVED(ingestion cells, `ssa_identity_tokens` cells)
    beside the existing `lexical_read_binding` re-concord; delete the two
    in-place dict comprehensions.
11. `glsl_deployment_strategy.py`, `_ordinary_conditional_control_programs`:
    build the field entries of `carried_aliases` from `CONTROL_CARRIED_FIELD`
    posts; add `carried_field_rows` to `ConditionalBlock`
    (`control_source.py`); `arm_return_control` reads `RETURN_SITE_SLOT`.
12. `glsl_deployment_strategy.py`, structural-fold return selection,
    `_fold_callsite_structural_values`, `_alias_projection_to_member`,
    `_retarget_all_cached_value_ids`: post REVISE / Unresolved instead of
    rewriting the dicts; the read views reflect the posts.
13. `fortran_c_shell.py`, `_class_surface_ssa_program` local-record loop:
    iterate WRITTEN/ELEMENT_WRITTEN cells; post admission or
    `SCALAR_WRITE_NO_DTYPE`; `ScalarFieldWriteBlock.field_state_cell`
    (`control_source.py`).  `_identity_return_aliases` and
    `direct_return_ids` read `RETURN_SITE_SLOT`.
14. `precompile_to_ssa.py`, `_ControlSSABuilder._lower`
    (`ScalarFieldWriteBlock`): Const NOVEL; `SSA_FIELD_VERSION` post; keep
    `external_values[effect_node_id]` as read view.
15. `precompile_to_ssa.py`, `_ControlSSABuilder.lower_conditional`: the
    arm lookup of section 5 for field-carried aliases; the merged Phi's
    `SSA_FIELD_VERSION` post; the shortfall on `ARM_VERSION_MISSING`.
    Name-carried aliases keep today's path (step 5).
16. `ssa_record_return_state.py`, `scalar_return_field_versions` and
    `publish_scalar_record_return_fields`; `fortran_c_shell.py`
    `select_return_arguments`: read the pages, post selections and the
    twelve reasons, NOVEL Casts.  `record_return_state_receipts` metadata
    becomes a read view.
17. `fortran_c_shell.py`, `_concord_record_return_phi_inputs`: its row
    derives from the `RECORD_RETURN_FIELD_SELECTION` cell (S21); the raise
    stays for a change with no changed source cell.
18. `identity_concordance.py` audit: teach `unsourced-fact` the nine pages
    (generic, so this is registration, not a new check); add one per-page
    check `carried-field-arm-missing` that lists `ARM_VERSION_MISSING`
    rows, since a lowering that stopped must be visible after the fact.
19. `tools/compiler_probes/probe_branch_field_return.py` (section 7) and
    the `TEST_BASELINE_AND_HAZARDS.md` line for it.
20. Delete the read views (items 2, 9, 14, 16) once `git grep` finds no
    reader of `attribute_value_nodes`, `attribute_effect_nodes`,
    `static_attribute_values`, `return_slot_values`,
    `return_record_field_states`, `return_container_kinds`,
    `record_return_state_receipts` outside the pages.

## 10. Risks

1. **The canonical relabel and pickled/digested graphs.**  Today the
   relabel rewrites ids inside the dicts in place and the dicts travel with
   `graph.G.graph`.  `_subgraph_reduction_digest` (glsl) hashes a dispatch
   subgraph with `cloudpickle.dumps`; `host_code_modules` pickles compiler
   IR (`_HostSSACachePickler`); `fortran_c_shell` pickles a payload for a
   digest.  If a `Ref` (which holds a registry `Page` object) is stored on
   `graph.G.graph` or on node data (item 3's `identity_cell`), (a) the
   digest changes for every unchanged region, invalidating the incremental
   backup once (acceptable, must be stated), (b) a pickled graph carries
   Refs into a process whose registry objects are different instances --
   `Page` must pickle by name and re-resolve through the registry on load,
   and the book itself is never pickled (the census-40 `_mint_table_owner`
   `__reduce__` problem in another form).  Mitigation: store on the graph
   only the read-view mapping proxies built lazily, and make `Ref`
   `__reduce__` to `(page name, row, column)`.
2. **Readers that iterate the dicts.**  `reduce_statement`'s return
   handling iterates `attribute_value_nodes.items()`; the `ast.If` merge
   iterates `before_attribute_values.items()` and the union of arm keys;
   `scalar_return_field_versions` iterates `receipts.items()`;
   `_fold_callsite_structural_values` iterates `return_slot_values.items()`.
   The read views must be full `Mapping`s over the page's `scope_rows`, in
   recorded order, or iteration order changes and with it Phi creation
   order and `binding_name` selection (`next(iter(authored_field_names))`).
3. **Behaviour change at section 5.**  The arm Const now reaches the
   `conditional_carried` Phi.  Any program that today compiles because the
   snapshot was silently taken will either compile with a different (now
   correct) Phi or stop with `ARM_VERSION_MISSING` naming the row.  Per the
   repo's rule the stop is the right outcome, but the native dt-system
   lowering is the first place it will show; the probe in section 7 must be
   green before that lowering is launched (by the user).
4. **Loop-exit field state (section 1.3, decision).**  Posting the
   post-loop field state as `Unresolved(LOOP_EXIT_FIELD_STATE_UNMERGED)`
   records what the reducer does (no merge) truthfully, but changes census
   50's X3: its receipt becomes Unresolved, so `scalar_return_field_versions`
   posts `SITE_WITHOUT_FIELD_STATE` for X3 and the descriptor field value is
   the argument on every round -- no Cast, no disagreement, and no branch
   write reaching the X3 return.  The alternative (post the body-exit cell
   as a DERIVED fact) records a fallback as fact, which the design forbids.
   The choice is the user's: the plan proposes Unresolved and asks.
5. **`_set_operands` volume.**  Appends are the most frequent operand change
   (every `new_node` with parents, every `_replace_inputs`).  Each becomes a
   post with two edge rows.  The book grows by roughly the graph's edge
   count per reduction; `scope_rows` stays O(1) per read, but
   `render_identity_book` output and the audit's page scans grow with it.
   Measure on the probe before the Woodshop compile.
6. **`Unsourced` auto-tagging (step 1) will list this step's read views**
   if any consumer writes through them by mistake; the read views must be
   read-only mappings (no `__setitem__`), so a stray write raises at the
   site instead of being silently tagged.
7. **Two receivers for one field (`returned_parameter_aliases`).**  The
   admission loop widens `receivers` with `_identity_return_aliases`
   (callee returns its argument).  The field-state row is keyed by the
   receiver node; a `coerce_metrics(metrics)` result is a different node
   from its argument, so R1 before the call and W1 after it can be rows of
   two receivers.  Today the dicts have the same split, so no regression,
   but the DERIVED chain from W1 to R1 passes through the call's
   `source_call_result_identity_concordance` / alias row, which step 4 or 7
   owns.  Until then the probe avoids the alias by writing and reading
   through one parameter.

## 11. Held for the user

1. Section 1.3 / risk 4: loop-exit field state as `Unresolved` (proposed)
   or as a DERIVED fact with a LOOP_EXIT kind (records a fallback as fact).
2. Whether `ARM_VERSION_MISSING` stops the lowering with a shortfall (the
   design's "stops the lowering") or raises at the site (the repo's "a
   missing row raises").  The plan writes the shortfall AND the Unresolved
   row so either policy has its record.
