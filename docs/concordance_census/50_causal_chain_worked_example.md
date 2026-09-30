# Causal chain worked example: `Metrics.hard_failure` through `step_with_dt_control_used`

Read-only census, 2026-09-30.  Everything below was established by reading
source (`git grep`/`sed`), not by running a lowering.  Where a claim depends
on a runtime condition the text says so.  No compiler numberings appear
here; values are named by the function that mints them and the source
construct they stand for.

Companion documents:

- `CONCORDANCE_MASTER_LIST.md` (Part A: what is on the book; Part B: the
  138 no-book and 35 mixed functions).  This file does not repeat it; it
  cites its rows and says where a reading here disagrees with its status.
- `docs/CONTINUATION_2026-09-05_RECORD_RETURN_IDENTITY.md` (the record-return
  identity work this field passes through: receiver reuse at result
  publication, the return-merge Phi over three record references,
  `materialize_record_phis` re-run inside the link fixed point, the
  `conditional_carried` repair of non-dominating call arguments).

## 0. The field and its journey in source

`Metrics` (`src/common/dt_system/dt_scaler.py`) is a dataclass; the field is
declared at class level as `hard_failure: bool = False`.

`step_with_dt_control_used` (`src/common/dt_system/dt_controller.py`) touches
it, inside a `while True:` body whose receiver `metrics` is rebound each
iteration by `ok, metrics = advance(state, dt_for_advance)` and then
`metrics = coerce_metrics(metrics)` (an identity return: `coerce_metrics`
returns its argument):

| Site | Source construct | Control position |
|---|---|---|
| R1 | `if bool(metrics.hard_failure):` (read) | straight-line in the loop body, before every write |
| W1 | `metrics.hard_failure = False` | inside `if rejected and ctrl.dt_min is not None:` inside `if float(dt_for_advance) <= dt_floor * (...)`, last statement of that inner arm, no else |
| W2 | `metrics.hard_failure = True` | inside `if rejected and retries_exhausted and allow_unresolved:`; arm ends with `rejected = False` (falls through) |
| W3 | `metrics.hard_failure = True` | inside `if rejected:` then `if retries_exhausted:`; arm ends with `return metrics, ...` |
| X1 | `return metrics, dt_next, dt_tensor` | inside `if not rollback:` (first pass) |
| X2 | `return metrics, dt_tensor * 0.5, 0.0` | the W3 arm |
| X3 | `return metrics, dt_next, dt_tensor` | function tail |

The field is returned three times as a member of the `metrics` record, so
the compiler has to merge three record references at the function exit and
then decide, per field, which version each return edge carries.

## 1. Every fact written about the field, in order

Legend for the **Edge** column: YES = a row on the book joins this fact to
the source fact it derives from; PARTIAL = the fact carries an id or a name
pointing at its source, but no row, or a row that names ids rather than rows;
NO = neither.  Graph parent edges are structure; they are counted as PARTIAL
only when the same fact is also on the book.

### Stage A: ingestion (`src/transmogrifier/graph/graph_express2.py`)

| # | Fact | Page / ledger | Row or shape | Writer | Derived from | Edge |
|---|---|---|---|---|---|---|
| A1 | `Metrics` has attribute `hard_failure`, identity `Metrics.hard_failure`, storage `class`, annotation `bool` | map IR `objects` tuple (graph metadata; later `class_table` / `program_abi.records`) | dict `{name, identity, storage, annotation, permissions}` | `_class_schema_from_ast` via `_map_ir_from_ast` | the `ast.AnnAssign` in the class body | NO |
| A2 | `hard_failure` has no instance slot | (not written) | `instance_attribute_slot` returns `None` for `storage == "class"` | `instance_attribute_slot` | A1 | NO (see note) |
| A3 | the field's annotation resolves to the class `bool` | page `source_field_identity_concordance` | row `(owner identity, "hard_failure")` -> resolved reference, column 0, concord-style | `_resolve_class_body_field` (only when some `self.<field>` navigation asks) | A1 + import bindings | NO: the fact is the class object; nothing joins it to A1 |

Note on A2: every dataclass field is a class-level `AnnAssign`, so
`_class_schema_from_ast` records `storage == "class"` and
`instance_attribute_slot` returns `None` for all of them.  The `attribute_slot`
attribute the reducer mirrors on GetAttr/SetAttr nodes is therefore never
written for any `Metrics` field, independently of the second condition below
(receiver must be an annotated parameter).  For dataclass records the slot
pathway is inert; the working identity is the `program_abi.records` field
schema.

### Stage B: reduction (`src/common/tensors/topological_reducer.py`, the `ProcessGraph` build closures)

| # | Fact | Page / ledger | Row or shape | Writer | Derived from | Edge |
|---|---|---|---|---|---|---|
| B1 | R1 is a `GetAttr` node, `attributes.attribute = "hard_failure"`, parent `(receiver, "value")`, optional `(last write, "after_write")` ordering parent | graph node | node attributes + `parents` | `resolve_expression`, `ast.Attribute` branch | receiver value node; `attribute_effect_nodes` for `after_write` | PARTIAL (graph parents only) |
| B2 | the receiver's class, read for B1 | page `source_value_class_concordance` (read) | row `(scope, receiver id)` -> `(class, limbs, source)` | read by the `ast.Attribute` branch; written earlier for the `coerce_metrics` call result by the call-result class binding | callee return annotation | (read) |
| B3 | R1 is *not* stamped `record_field`, `producer_kind`, or `attribute_slot` | graph node (absent) | -- | same branch: `record_field` is written only inside `if field_kind is not None` (aggregate fields); `attribute_slot` only when `parameter_class_names` knows the receiver, and `metrics` is a call result | -- | NO (absence is not recorded) |
| B4 | the current value of `(metrics, "hard_failure")` is the R1 read node | private dict `attribute_value_nodes` | `(receiver id, "hard_failure") -> node id`, `setdefault` | same branch, "Observation does not assign a new field version" | B1 | NO |
| B5 | the Name `metrics` read at R1 is a bound read | page `lexical_read_binding` | `(read scope, attr node, "value", 0) -> binding` | reducer Name-load commit (`CONCORDANCE_MASTER_LIST.md` A3) | the receiver binding | YES (for the receiver, not the field) |
| B6 | W1 is a `SetAttr` node, `op = "setattr"`, `attributes.attribute`, parents `(receiver, "object")`, `(Const False, "value")` | graph node | node attributes + `parents` via `_replace_inputs` | `bind_target`, `ast.Attribute` target branch | AST `Assign`; the RHS Const node | PARTIAL (graph parents only) |
| B7 | last effect on the field is W1; current value is W1's RHS | private dicts `attribute_effect_nodes` (effect = SetAttr node), `attribute_value_nodes` (value = the RHS Const node) | keyed `(receiver id, "hard_failure")` | `bind_target` | B6 | NO |
| B8 | `metrics.hard_failure` is a binding whose history now includes W1 | private `identity_bindings[...]` list, `ingestion_definitions` (via `record_ingestion_definition`), later `ingestion_identity_table` / `identity_table` graph metadata | name-keyed list of ids with span/context tokens | `bind_target` | B6 + AST span | NO (name-keyed history) |
| B9 | inner conditional merge of W1: a `Phi` node named `metrics.hard_failure`, parents `(test)`, `(body = the RHS Const of W1)`, `(orelse = the R1 read node)`, attributes `source_conditional_id`, `initial_value_id` (= R1 node), `record_field_state = (receiver, "hard_failure")` | graph node; also `identity_bindings` append; `attribute_value_nodes`/`attribute_effect_nodes` := this Phi | `new_node("Phi", ...)` | `reduce_statement`, `ast.If` branch, the `for field_key, initial_value in before_attribute_values.items()` loop | B4 (initial), B7 (arm value) | PARTIAL (graph parents; `initial_value_id` is an id) |
| B10 | outer conditional merge (`if rejected and ctrl.dt_min is not None`): `Phi` with body = B9, orelse = R1 read | same shape as B9 | same writer | B9, B4 | PARTIAL |
| B11 | merge of W2 (`if rejected and retries_exhausted and allow_unresolved`): `Phi` body = W2's RHS Const, orelse = B10 | same shape | same writer | B10 | PARTIAL |
| B12 | W3's arm is terminal, so the fall-through ledger keeps the *else* state (B11) and W3 is not merged | private ledgers restored from the else snapshot | the `body_terminal and not else_terminal` rule | same writer | -- | NO (the decision "not merged, arm returns" leaves no row) |
| B13 | at X2 the returned `metrics` carries `hard_failure = W3's RHS Const`; at X1 it carries the R1 read; at X3 it carries B11 | graph metadata `return_record_field_states[span] = ((receiver, "hard_failure", value id), ...)`, filtered to receivers in the slots; beside `return_slot_values[span]` and `return_container_kinds[span]` | keyed by the source span of the returned expression | `reduce_statement`, `ast.Return` branch | `attribute_value_nodes` at the return | NO (a triple of ids) |
| B14 | canonical renumbering rewrites `return_slot_values`, `return_record_field_states`, and node attributes `record_field_state`, `initial_value_id`, `source_conditional_id` through the pre-canonical -> canonical `mapping`, in place | same graph metadata and node attributes, overwritten | dict comprehension over `mapping` | the canonical relabel (the block that sets `lexical_read_scope` / `operand_position_scope`) | B9 to B13 | NO for the field facts (in-place rewrite; contrast: `lexical_read_binding` rows are re-concorded under the new read scope, YES) |

Two silent losses are visible in this stage's code, both relevant later:

- In the merge loop, `if not isinstance(test_value, int): continue` skips
  the Phi when the conditional's test reduced to a non-runtime value and
  leaves `attribute_value_nodes` at the *initial* (the arm's write is
  dropped, unrecorded).  Not the observed case, but the same shape.
- The Phi's body arm is the RHS value (`attribute_value_nodes` stores the
  RHS, not the SetAttr node).  Stage D keys the write under the SetAttr
  node.  This mismatch is the observed loss; see section 2.

### Stage C: planning (`src/compiler/glsl_deployment_strategy.py`)

| # | Fact | Page / ledger | Row or shape | Writer | Derived from | Edge |
|---|---|---|---|---|---|---|
| C1 | per-node dispatch classification of the field's nodes (structural vs numeric), plus the set of carried initials | `G.graph["_dispatch_metadata_cache"]` | `{__fingerprint__, __carried_initials__, node id -> bool}` | the dispatch classifier factory (the function whose docstring says "Obtain a new classifier after changing the graph's structure") | node attributes | NO |
| C2 | the R1 read feeds the predicate region for `bool(...)`; its operand position | pages `consumer_operand`, `region_feed_consumer` | `(read scope, node, value) -> ((role, ordinal), ...)`; `(control scope, region, feed) -> consumers` | `_concord_consumer_operands`, `lower_control_sections_to_ssa` (per `CONCORDANCE_MASTER_LIST.md` A3) | B5 | YES (read position) |
| C3 | the conditional carries `(body id, orelse id, initial id, merged id)` for each reducer Phi with `initial_value_id` under this `source_conditional_id`; and `record_field_state` keys of the merge | `ConditionalBlock.carried_aliases` (control program dataclass), `direct_phi_record_fields` local set via `_record_field_state_keys` | tuples of ids copied from the Phi's `parents` | `_ordinary_conditional_control_programs` | B9, B10, B11 | NO (ids copied, no row) |
| C4 | each return site's slot values become `ReturnBlock(return_value_ids = slots)` | control program | from `return_slot_values[span]` | `arm_return_control` inside `_ordinary_conditional_control_programs` | B13 | NO |
| C5 | when a proven terminal top-level arm makes later return sites unreachable, `return_slot_values` is replaced by the selected sites; and `_alias_projection_to_member` / `_retarget_all_cached_value_ids` rewrite ids inside `return_slot_values` and `return_record_field_states` in place | graph metadata, overwritten | dict rewrites | the structural-fold return selection; `_alias_projection_to_member` | B13/B14 | NO |

### Stage D: control-program completion and control SSA

D1 is in `src/compiler/fortran_c_shell.py` (it prepares the control program
the builder lowers); D2 onward is `precompile_to_ssa._ControlSSABuilder`.

| # | Fact | Page / ledger | Row or shape | Writer | Derived from | Edge |
|---|---|---|---|---|---|---|
| D1 | W1 (and W2, W3) becomes `ScalarFieldWriteBlock(field_value_id=None, value_expression=ControlExpression("const", value_id=RHS Const node, literal=False), dtype, effect_node_id=SetAttr node)`, placed in its authored arm by `_install_lexical_sequence_mutations` via `_branch_compartments` | control program list `scalar_writes`; private set `scalar_write_effect_ids` | dataclass | the local-record loop after `declared_parameter_records` in the frame linker's control preparation ("A record produced inside this function ... has no parameter storage cell") | B6; `constant_values` / `parameter_value_dtypes` for the dtype; `_graph_control_expression` for the leaf | NO.  If the dtype is not provable the loop `continue`s and the write has no block at all (silent) |
| D2 | the arm emits `Const` with a **fresh** SSA id and `attributes.value = False`; then `external_values[SetAttr node id] = that value`; no `Store` (no destination) | SSA instruction; private `external_values` | `lower_control_expression`, `"const"` branch: `result = result_override or self.fresh_value(...)` | `_ControlSSABuilder.lower`, `ScalarFieldWriteBlock` branch | D1 | NO.  The Const carries no accounting naming the RHS graph node or the SetAttr node; `source_effect_node_id` is only written on the `Store`, which is skipped |
| D3 | the inner conditional's carried merge: `Phi(binding = "conditional_carried", initial_value_id, incoming_blocks)` with arguments `true_carried[initial]` and `false_carried[initial]`, where each is `external_values.get(arm id, carried_snapshots[initial])` | SSA instruction; `external_values` updated for merged, initial, and both arm ids | `emit(Handler.Phi, [true_value, false_value], merged, ...)` | `_ControlSSABuilder.lower_conditional` (the `for ... in conditional.carried_aliases` loop) | C3 (ids), D2 (by id), the snapshot | NO.  When the arm id is absent the snapshot is taken **and emitted as the argument**, with no shortfall and no row |
| D4 | the outer conditional's carried merge, same shape, arms looked up the same way | SSA instruction | same | same | C3, D3 | NO |
| D5 | `value_name_histories` (the `identity_table`) is consulted for parameter seeding and reported in `finish` as `value_names` / `parameter_names` | `function.metadata` | name -> id history | `_ControlSSABuilder.__init__`, `finish` | B8 | NO |
| D6 | each return edge: `Br` to the exit with `attributes.return_source_value_ids = ReturnBlock.return_value_ids`, `source_control = "return"`; `function_return_edges.append((block, edge_values))` | SSA instruction attributes; private list | tuple of reducer slot ids | `_ControlSSABuilder.lower`, `ReturnBlock` branch | C4 | PARTIAL (the attribute names reducer ids, not rows) |
| D7 | the exit merges the `metrics` slot: `Phi(binding = "return_merge", return_slot_index, incoming_blocks, output_name)` over the three record-valued slot values | SSA instruction | `emit(Handler.Phi, incoming_values, merged, ...)` | `_ControlSSABuilder.finish` | D6 | PARTIAL (`incoming_blocks`) |
| D8 | `authored_constant_values` metadata; consolidation of `Const` instructions whose result id is in `constant_values` | `function.metadata`; SSA | -- | `_materialize_control_constants` | planner `constant_values` | NO.  D2's Const has a fresh id, so it is not in `constant_values` and is untouched |
| D9 | per-argument reconciliation outcome of loop-result uses | page `loop_result_reconciliation` | `(function, argument id) -> (outcome, "block#index", op, detail)`, one column per visit | `_canonicalize_non_dominating_loop_result_uses` | `carried_port_values` metadata | PARTIAL (a decision row keyed by id; RECORD-ONLY in the master list's vocabulary) |

### Stage E: record materialization (`fortran_c_shell` link fixed point, `src/transmogrifier/ssa.py`, `src/compiler/ssa_record_return_state.py`)

| # | Fact | Page / ledger | Row or shape | Writer | Derived from | Edge |
|---|---|---|---|---|---|---|
| E1 | the `metrics` value is a `Metrics` record; `hard_failure` is one scalar field with one value id | pages `record_descriptor` `(owner, record id) -> SSARecordDescriptor`, `record_member` `(owner, value id) -> ((record id, field, storage identity, role), ...)` | written through `_BookRows` and `_revise_member_claims` | `SSARecordTable.register`; for a call result the descriptor is published by the result-publication path that reuses the callee formal's descriptor when the callee returns its own argument (`_identity_return_aliases`; `docs/CONTINUATION_2026-09-05_RECORD_RETURN_IDENTITY.md` finding 4).  That path was not read line by line here | the callee (`coerce_metrics`, `advance`) descriptors | PARTIAL: `record_member` is a membership (value is field F of record R), BOOK; but nothing joins this descriptor to the callee descriptor it was copied from |
| E2 | record id -> physical field layout | `function.metadata["record_return_layouts"]` (six writer sites) | tuple of `(record id, layout)` | `materialize_program_abi_record_literals` and five others | E1 | NO |
| E3 | the return-merge Phi over records is expanded to one `Phi` per field: result minted by `GLOBAL_MONOTONIC_IDS.mint()`, `accounting = {record_phi, record_field, record_field_slot}`, attributes `record_field_phi`, `record_field`, `initial_value_id`, `record_return_scalar`, `record_return_receivers`, `return_slot_index`, `incoming_blocks`; arguments = each incoming record's field value (E1) | SSA instruction; `values` map | rebuilt instruction | `materialize_record_phis` (inner function of `_class_surface_ssa_program`) | D7, E1 | PARTIAL: `record_return_receivers` and `incoming_blocks` are ids/labels; the mint has no NOVEL row |
| E4 | the merged descriptor's field layout | page `record_field_layout_concordance` `(symbol, result id, storage identity)` -> layout | | `materialize_record_phis` | E3 | PARTIAL (keyed by the minted id; does not name E1's rows) |
| E5 | `record_return_state_receipts = ((span, slots, states), ...)` copied from the graph | `function.metadata` | tuple | `scalar_return_field_versions` | B13/B14 (via `source_graphs_by_symbol`) | NO |
| E6 | for return edge P and field F: the version reaching P is the `Const` or `conditional_carried Phi` whose id the receipt names, if it dominates P with no intervening store/whole-record call; else the descriptor field value ("fallback") | closure `lookup` result (no ledger) | matches `return_source_value_ids` (D6) against `return_slot_values` by tuple equality; then `(receiver, field) -> id`; then `definitions[id]` | `scalar_return_field_versions.lookup` | D6, E5, D3/D4 by id | NO.  Neither "substituted" nor "fell back" is recorded anywhere |
| E7 | when E6's version has another dtype and is a Boolean-leaf Phi tree: `Cast` minted by `GLOBAL_MONOTONIC_IDS.mint()`, attributes `record_return_field_conversion = F`, `source_field_value_id = the version's id`, inserted before P's terminator; cached in the private `conversions` map | SSA instruction | | `select_return_arguments` (inner function of `materialize_record_phis`) and again `publish_scalar_record_return_fields` | E6 | PARTIAL (`source_field_value_id` is an id pointer) |
| E8 | the chosen argument for E3 position i | page `record_return_phi_input_concordance` `(function, phi result, F, position, predecessor block) -> (candidate id, chosen id, reason)`, a new column per revisit; raises when `chosen` differs from the prior column's `chosen` | | `_concord_record_return_phi_inputs` | E3's incumbent args, E7's output | NO: the row records two ids and a reason; it does not name E1's field row, E6's receipt, or E7's Cast as its source, so when the chosen id changes between rounds the page can only see two different ids |
| E9 | the same substitution repeated after signatures settle, rewriting `operation.args` and `attrs["initial_value_id"]` in place | SSA instruction | | `publish_scalar_record_return_fields` | E5 to E7 | NO |

### Stage F: frame linking (`fortran_c_shell`)

| # | Fact | Page / ledger | Row or shape | Writer | Derived from | Edge |
|---|---|---|---|---|---|---|
| F1 | which caller value a callee record member is | (returned, not recorded) | reads `record_member`, `sequence_member`, `call_record_pair_concordance` | `_linked_caller_member` | E1 + the call's `argument_bindings` | NO (`CONCORDANCE_MASTER_LIST.md` A2: "BOOK source; decision still unrecorded") |
| F2 | old result id -> new storage id for the call's result record | private `result_storage_bindings` per call, `result_storage_bindings_by_call` | dict | `allocate_result_storage` and its callers | E1, E2 | NO |
| F3 | each callee field value is bound to the caller's canonical field slot | `storage_bindings` -> `frame_bindings` entries `(value id, "caller_storage", caller storage id)` on the call record (`SSACallTable`, page `call_record`) | tuple on a BOOK row | the frame-tail completion that writes `caller_storage = int(caller_field.value_ids[0])` | E1 for both sides | PARTIAL (id pairs on a book row; the `record_member` rows they came from are not named) |

### Count

42 facts (A1 to A3, B1 and B3 to B14, C1 to C5, D1 to D9, E1 to E9, F1 to
F3; B2 is a read and is not counted; B5 and C2 are about the receiver's
read position, not the field's value).

- YES: 2 (B5, C2), both `lexical_read_binding`-family rows about the
  receiver read, not about the field's value.  Zero edges join a field
  version to the version it derives from.
- PARTIAL: 13 (B1, B6, B9, B10, B11, D6, D7, D9, E1, E3, E4, E7, F3), all
  either graph parent edges or id-valued attributes on an SSA instruction
  or a book row.
- NO: 27.

The first missing edge is B4/B7: the reducer's `resolve_expression`
(`ast.Attribute`) and `bind_target` (`ast.Attribute` target) record the
field's current value in `attribute_value_nodes` and its last effect in
`attribute_effect_nodes`, two private dicts keyed by `(receiver id, field
name)`, with no row on the book.  Everything after that (B9 to B13, C3, C4,
E5, E6) derives from those two dicts by copying ids.

## 2. Why the branch write did not reach the phi (from the code)

Observed 2026-09-30 in the native dt-system lowering: the return-merge arm
from one return site first took the record descriptor's field value; on a
later fixed-point round `scalar_return_field_versions.lookup` substituted a
`conditional_carried` Phi whose two incoming arms were both the initial
value, cast to the field dtype; `record_return_phi_input_concordance` raised
a disagreement because no edge joined the two.

The chain, function by function:

1. `bind_target` (reducer) records W1 as `attribute_value_nodes[key] =
   <RHS Const node>` and `attribute_effect_nodes[key] = <SetAttr node>`.
   The comment at the merge site says it explicitly: "SetAttr contributes
   its RHS rather than its effect node."
2. The `ast.If` merge in `reduce_statement` builds the reducer Phi with
   `body = <RHS Const node>`, `orelse = <R1 read node>` (the initial, since
   R1 seeded `before_attribute_values` earlier in the same loop body),
   `initial_value_id = <R1 read node>`.
3. `_ordinary_conditional_control_programs` (planner) copies the Phi's
   parents into `carried_aliases = (body id, orelse id, initial id, merged
   id)`.  The body id is the **RHS Const node**.
4. The frame linker's local-record loop admits W1 as
   `ScalarFieldWriteBlock(None, ControlExpression("const", value_id = RHS
   Const node, literal = False), dtype, effect_node_id = SetAttr node)`.
5. `_ControlSSABuilder.lower` (`ScalarFieldWriteBlock` branch) calls
   `lower_control_expression`, whose `"const"` branch emits a `Const` under
   `self.fresh_value(...)`, a **new** id, and does not register it under
   `expression.value_id`.  It then publishes `external_values[SetAttr node
   id] = value`.  So the written value is reachable under the SetAttr id,
   never under the RHS Const id.
6. `lower_conditional` computes `true_carried[initial] =
   self.external_values.get(int(true_id), carried_snapshots[int(initial_id)])`
   with `true_id` = the RHS Const id from step 3.  The key is absent; the
   `.get` default is the pre-branch snapshot.  The false arm has no write
   and also resolves to the snapshot.  The `conditional_carried` Phi is
   emitted with both arguments equal to the initial.  No shortfall is
   appended; nothing is written to any page.  Compare `true_results`, three
   lines below, which uses `self.external_value(int(true_id))` and would
   have materialized or declared the value rather than defaulting.
7. `scalar_return_field_versions.lookup` (E6) finds the receipt for the X3
   span naming the merged id, finds its definition is a `conditional_carried`
   Phi (permitted), passes the Boolean-leaf and dominance tests, and returns
   it; `select_return_arguments` mints the `Cast` because the Phi inherited
   the snapshot's dtype rather than the field's.  On the earlier round the
   same lookup had returned `fallback` (the descriptor field value) because
   the receipts or the Phi definition were not yet visible in that round's
   function, so `_concord_record_return_phi_inputs` had recorded the
   descriptor value as `chosen`.  On the later round `chosen` is the Cast;
   the page has no row saying "Cast derives from the carried Phi derives
   from the receipt derives from the descriptor field", so it raises.

Two facts make this a fallback recorded as truth rather than an unresolved
mark: step 6's `.get(..., snapshot)` and step 7's `return fallback`.  Both
choose from absence of information and neither writes that it did.

What is *not* established here: whether the RHS Const id could have reached
`external_values` by another route in the real lowering (for example if the
same literal were also a region output or a planner constant materialized
under its graph id by `_materialize_control_constants`).  The code path read
above gives it no such route for a constant that is only the RHS of a
SetAttr in a structural arm.

Cross-check against the master list: `CONCORDANCE_MASTER_LIST.md` A3d says
merges "are the reducer's Phis, which it creates for every runtime
conditional whose arms differ"; that holds, and it is exactly why the arm
identity is the RHS node.  Part B does not list `scalar_return_field_versions`
at all (its scan keys are ABI labels and private structures; the receipts
`return_record_field_states` / `return_slot_values` are not among them), so
the function that made the observed substitution is invisible to Part B.
`_concord_record_return_phi_inputs` is listed in B2 as mixed; this reading
agrees and adds that the row it writes cannot express a derivation, which is
why it can only raise.

## 3. What the single api would have recorded

Every post below goes through one call.  Mode NOVEL(minted id, transform)
mints an identity and records the transform edge that produced it; mode
DERIVED(source rows, stage) names the exact row(s) and stage it derives
from.  A post that cannot name a source is UNRESOLVED and is a query, not a
fact.  The same journey:

| # | Post | Mode | Edge recorded |
|---|---|---|---|
| S1 | field schema `Metrics.hard_failure`: storage class, dtype bool, default False | NOVEL(field row, `_class_schema_from_ast` from `ast.AnnAssign` span) | schema row -> source span |
| S2 | receiver value of `coerce_metrics(...)` is a `Metrics` | DERIVED(`source_value_class_concordance` row for the callee return; S1's class row; stage reducer) | receiver class -> callee return annotation row |
| S3 | R1 read: value of field S1 on receiver S2 | DERIVED(S2, S1; stage reducer `resolve_expression`) | read -> (receiver, field) |
| S4 | W1 write: field S1 on receiver S2 takes the literal False | NOVEL(literal, `bind_target` from `ast.Constant` span) then DERIVED(S2, S1, the literal; stage reducer `bind_target`) | write -> receiver, field, literal |
| S5 | field state after the inner arm: merge(test, W1 value, S3) | DERIVED(S4, S3, the test's row; stage reducer conditional merge) | merge -> both arms and the test |
| S6 | field state after the outer arm: merge(test, S5, S3) | DERIVED(S5, S3, test row) | same |
| S7 | W2 write and its merge with S6 | as S4 then DERIVED(W2, S6) | same |
| S8 | W3 write; the arm returns, so the state is not merged | DERIVED(W3 write row; stage reducer) with the explicit fact "terminal arm, state ends at X2" | write -> X2 return row |
| S9 | return sites X1, X2, X3 each carry field state (S3, S8, S7 respectively) for slot 0 | DERIVED(S3 / S8 / S7, the return row; stage reducer `ast.Return`) | return-site field state -> the state row it copies |
| S10 | canonical relabel of every row above | DERIVED(each pre-canonical row; stage canonical relabel) via `identity_transition` "move" rows | new id -> old id per row (the morph graph) |
| S11 | the planner's carried alias for the inner conditional | DERIVED(S5; stage planner) | alias -> S5 (no id copy) |
| S12 | `ScalarFieldWriteBlock` for W1 | DERIVED(S4; stage control preparation).  If the dtype cannot be proven: UNRESOLVED(S4, "no dtype"), never `continue` | block -> S4 |
| S13 | the arm's Const SSA value | NOVEL(fresh id, `lower_control_expression` const) with DERIVED(S12's literal) | SSA Const -> S4's literal row |
| S14 | the published field version in the arm | DERIVED(S13, S12; stage control SSA) under the **field-state row**, not under a graph id | version -> S13 |
| S15 | the `conditional_carried` Phi arms | DERIVED(S14 for the true arm, S3's SSA value for the false arm; stage `lower_conditional`).  An arm with no posted version is UNRESOLVED(S11, arm) and stops the lowering; a snapshot is never posted as an arm | Phi arg i -> version row |
| S16 | the return-merge Phi over records | DERIVED(S9 x3, D6's edges; stage `finish`) | Phi arg i -> return-site row |
| S17 | the record descriptor for the receiver | DERIVED(callee descriptor rows; stage result publication) | descriptor -> callee descriptor |
| S18 | the per-field Phi minted by `materialize_record_phis` | NOVEL(minted id, record-Phi expansion) with DERIVED(S16, S17 field rows) | field Phi -> record Phi, descriptor field |
| S19 | the version selected for return edge P | DERIVED(S9 for P, S15 or S13 as named by S9; stage `scalar_return_field_versions`).  "fallback" is posted as UNRESOLVED(S9 for P, reason), never as the descriptor value | selection -> receipt row -> version row |
| S20 | the Cast | NOVEL(minted id, `record_return_field_conversion`) with DERIVED(S19) | Cast -> selection |
| S21 | the Phi input at position i | DERIVED(S20 or S19; stage phi-input concord).  A later round posts DERIVED(S20, superseding S17 field) and the page can see one path, so a change is a revision with a cause, not a disagreement | input -> its derivation |
| S22 | caller storage for the field | DERIVED(S17 both sides, the call's `argument_bindings` row; stage frame linking) | binding -> member rows |

The observed failure disappears at S15: the arm cannot default, and at
S19: a fallback cannot masquerade as a version.  The disagreement at S21
becomes a revision whose cause is the S19 row that changed.

## 4. Shadow ledgers this one field passed through

Private, non-book structures that held a fact about `Metrics.hard_failure`
between its source and the emitted program, in order of first touch:

1. `graph_express2`: the map IR `objects` tuple; `class_table` /
   `class_definitions` / `program_abi.records` graph metadata.
2. `topological_reducer` closures: `attribute_effect_nodes`,
   `attribute_value_nodes`, `static_attribute_values`, `identity_bindings`,
   `ingestion_definitions`; graph metadata `return_slot_values`,
   `return_record_field_states`, `return_container_kinds`,
   `ingestion_identity_table`, `identity_table`; node attributes
   `record_field_state`, `initial_value_id`, `source_conditional_id`,
   `binding_name`; the canonical `mapping`.
3. `glsl_deployment_strategy`: `G.graph["_dispatch_metadata_cache"]`,
   `ConditionalBlock.carried_aliases` / `result_aliases`,
   `ReturnBlock.return_value_ids`, the pruned `return_slot_values` copy,
   the rewritten `identity_table`.
4. `fortran_c_shell` control preparation: `scalar_writes`,
   `scalar_write_effect_ids`, `scalar_write_sources`, `constant_values`,
   `parameter_value_dtypes` / `parameter_value_shapes`, `resident_value_ids`.
5. `precompile_to_ssa._ControlSSABuilder`: `external_values`,
   `carried_snapshots`, `scalar_field_effect_destinations`,
   `value_name_histories`, `_carried_port_values`, `function_return_edges`,
   `value_aliases`; metadata `carried_port_values`, `value_names`,
   `parameter_names`, `authored_constant_values`.
6. record materialization: `function_values` map, `layouts`,
   `record_return_layouts`, `record_return_state_receipts`, `conversions`,
   `output_identity_aliases`.
7. frame linking: `result_storage_bindings` / `result_storage_bindings_by_call`,
   `storage_bindings`, `frame_bindings`, `frame_ledgers`.

Book pages touched along the way, for contrast: `source_field_identity_concordance`
(conditionally), `source_value_class_concordance` (the receiver),
`lexical_read_binding`, `consumer_operand`, `region_feed_consumer`,
`identity_transition` (operand moves only), `loop_result_reconciliation`,
`record_descriptor`, `record_member`, `record_field_layout_concordance`,
`record_return_phi_input_concordance`, `call_record`,
`call_record_pair_concordance`.  None of them carries an edge from a field
version to the version it was derived from.
