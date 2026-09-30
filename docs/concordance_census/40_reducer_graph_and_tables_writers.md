# Census 40: reducer, graph_express2 and book-backed table writers

Read-only scout, 2026-09-30.  Slice: `src/common/tensors/topological_reducer.py`,
`src/transmogrifier/graph/graph_express2.py`, `src/transmogrifier/ssa.py`,
`src/transmogrifier/ctypes_layout.py`,
`src/transmogrifier/graph/python_identity_programs.py`,
`src/transmogrifier/function_table.py`,
`src/common/tensors/accelerator_backends/aot_compile.py`.

This census does not repeat `CONCORDANCE_MASTER_LIST.md`.  It cites that
file's rows (A1, A3, A3a, A3c, B1, B2) where a site is already listed there and
records only what it lacks: for every write, whether the write carries an EDGE
to the row(s) it derives from; whether a fallback is recorded as a fact; whether
a re-registration records an edge from the incumbent; and which ingestion facts
never reach the book.  Where this reading contradicts the master list's status,
the contradiction is stated in section 7.

Code is named by function, never by line.  No compiler numberings appear here.

## 1. Legend

- **row key shape** -- what the row tuple is made of, in words.
- **fact shape** -- what the recorded fact is made of.
- **mode** -- `set0` (write column 0 only if no incumbent, else compare and
  raise; the hand-rolled equivalent of `concord`), `set!` (unconditional
  `set`, no incumbent check), `concord`, `revise` (append next column),
  `mapping` (`PageMapping` assignment, a `revise`), `copy` (columns copied
  verbatim to a new row), `mint` (`IdentityBook.mint_scope`, a `concord` on
  `scope_registry`).
- **EDGE** -- YES: the fact or row names the concordance row(s) and stage it
  derives from, or the transform that produced a minted identity.  PARTIAL:
  the fact names a value id, scope or free-text cause in the same scope but no
  page/stage, or an edge exists only at scope granularity.  NO: nothing in the
  write says where the fact came from.
- **fallback** -- a value chosen from the absence of information that the
  write records as if it were known.
- **readers** -- by function name where known; "audit" means a
  `CorrelationTable` finding method in `identity_concordance.py`; "none
  outside writer" means `git grep` finds the page name only in the writing
  file.

## 2. topological_reducer.py

### 2.1 Page `source_value_class_concordance`

The mechanism is one helper, `_concord_source_value_class`: row `(numeric
scope string, value id)`, fact `(class identity, limbs, source string)`, mode
`set0`.  The comparison excludes the `source` string, so two callers with
different causes agree if class and limbs agree; the first cause wins the
record.  `limbs` is `max(precision_limbs or 1, 1)`: an absent width becomes
the fact "width one".  Readers: `_resolved_source_value_class`,
`_concorded_numeric_descriptor`, receiver-fact lookups in
`resolve_expression` and `reduce_statement`, `propagate_call_formal_numeric_types`,
`resolve_concorded_receiver_constructor`, `glsl_deployment_strategy`, tests.

| enclosing function | source string written | EDGE | fallback-as-fact |
|---|---|---|---|
| `specialize_python_precision_widths` (specialization call argument) | "specialization call argument <name>" | PARTIAL (names the parameter, not the caller's actual row) | width from `parameter_classes`; else width one |
| `specialize_python_precision_widths` (callsite-specialized Precision result) | fixed string | PARTIAL | `precision_limbs or 1` on the node |
| `specialize_python_precision_widths` (producer flow) | fixed string | NO (no operand named) | no |
| `specialize_python_precision_widths` (Precision operator) | fixed string | NO; the operator fact is on another page (2.3) | no |
| `_normalize_lexical_values.input_value` | "parameter annotation" family | NO (annotation is graph metadata, not a row) | limbs one when descriptor absent |
| `_normalize_lexical_values.resolve_expression` (numeric-component-projection) | fixed string | PARTIAL via companion `_concord_numeric_feature_projection` | no |
| `resolve_expression` (resolved call result) | fixed string | NO (callee not named) | `precision_limbs` may be None; skipped, not recorded |
| `resolve_expression` (keyed value-record edge) | fixed string | NO | **yes**: `precision_limbs=1` hard-coded |
| `reduce_abstract_tensor_topology.specialize_concorded_same_type_numeric_operator` | fixed string | PARTIAL via companion page (2.3) | no |
| `reduce_abstract_tensor_topology.resolve_concorded_receiver_constructor` | fixed string | PARTIAL via companion page | no |
| `reduce_abstract_tensor_topology.propagate_numeric_field_projections` (intrinsic result) | fixed string | PARTIAL via companion | no |
| `propagate_numeric_field_projections` (post-call component projection) | fixed string | PARTIAL via companion | no |
| `reduce_abstract_tensor_topology.lower_class_operator_calls` (returned class) | fixed string | PARTIAL: limbs = max of operand limbs, operands named on `source_operator_dispatch_concordance` | `other_id` absent -> limbs one |
| `reduce_abstract_tensor_topology.lower_python_precision` (Precision.of boundary) | fixed string | PARTIAL via `_concord_source_precision_boundary` | no |
| `lower_python_precision` (Precision BinOp/UnaryOp) | "Precision <op class>" | PARTIAL via `concord_operator` | no |
| `lower_python_precision` (Precision method) | "Precision <method>" | PARTIAL via `concord_operator` | no |
| `reduce_abstract_tensor_topology.propagate_call_formal_numeric_types` (call result) | "call result <callee scope>" | PARTIAL: names the callee scope as text; the callee output row is not named | no |
| `propagate_call_formal_numeric_types` (call actual to formal) | "call actual <caller scope>:<actual id> via <prior source>" | PARTIAL: the causal chain is carried as **free text** in the fact; it is not a row reference and the audit cannot follow it | when the caller row is absent, `_resolved_source_value_class` is consulted and its width-one default flows in |

Two sites copy rows of this page without the helper:

| enclosing function | what | mode | EDGE |
|---|---|---|---|
| `specialize_python_precision_widths` | every authored-scope row whose value id exists in the specialized graph is copied to the specialization scope, column by column | `copy` | NO on this page; PARTIAL overall because `source_numeric_specialization_concordance` row `(authored scope, digest)` -> `(specialization scope, receipt)` joins the two scopes |
| `_normalize_lexical_values` (canonical relabel tail) | every row at an ingestion id is copied to the canonical id | `copy` | **NO**: the ingestion-id to canonical-id mapping is held only on the graph (`ssa_identity_tokens`, `identity_table`); no page row says "canonical row X came from ingestion row Y" |

### 2.2 Operand positions: `lexical_read_binding`, `identity_transition`, `consumer_operand`, `scope_registry`

Master list A3, A3a, A3c describe the design.  Per-site EDGE reading:

| enclosing function | page | row key shape | fact shape | mode | EDGE | fallback | readers |
|---|---|---|---|---|---|---|---|
| `_concord_lexical_reads` | `lexical_read_binding` | `(ingestion read scope, "occurrence", occurrence id)` | binding name | concord | NO -- this is a root fact (the authored read); acceptable as a spontaneous identity but nothing records the AST occurrence it came from except the id itself | no | `_set_operands` (fork source) |
| `_concord_lexical_reads` | `lexical_read_binding` | `(scope, consumer, role, ordinal)` | binding name | concord | PARTIAL: derived from the occurrence row, but the row does not say so | no | `lexical_read_binding` accessor, planner `_concord_consumer_operands`, control `_resolve_read` |
| `_normalize_lexical_values` (return roots) | `lexical_read_binding` | `(ingestion scope, "return", "root", position)` | binding name | concord | NO | only written when exactly one binding returned the root; the multi-binding case is silently not recorded (absence, not an unresolved mark) | control lowering |
| `_normalize_lexical_values` (canonical relabel) | `lexical_read_binding` | `(canonical read scope, canonical consumer, role, ordinal)` | copied fact | concord | PARTIAL: the ingestion scope is literally `(canonical scope, "ingestion")`, so the join is structural in the key; no transition row | rows whose latest is None are dropped silently | same |
| `_normalize_lexical_values` | `scope_registry` | `("lexical_reads:<numeric scope>", serial)` | True | mint | PARTIAL (label carries the numeric scope name as text) | no | `mint_scope` |
| `_set_operands` | `identity_transition` | `(scope, consumer, role, ordinal)` | `("move", consumer, new role, new ordinal, cause)`, `("retire", None, None, None, cause)`, `("fork", source node, source role, source ordinal, cause)` | revise | YES for move/retire/fork: names the position it acts on and where it goes; `cause` is a free string (a writer name), not a stage row | a rewrite with `_operand_position_scope` None writes the graph's `parents` and records **nothing**; an append creates a new position with no origin fact | `fork_read_scope` (skips scope rows), `precompile_to_ssa`; **no audit finding reads this page** |
| `_set_operands` | `lexical_read_binding` | position rows | moved fact or None | revise | PARTIAL (the move is on `identity_transition`; this page just receives the new value) | vacated position -> None revision (correct: a removal, not a fact) | as above |
| `_set_operands` | `consumer_operand` | `(scope, consumer, value)` | tuple of positions | revise | PARTIAL | `()` revision when the value id was renamed away | planner |
| `_replace_inputs`, `_remove_node`, `_redirect_value`, `_append_operand`, canonical relabel, Expr/Return dissolve, function-subgraph filter, parameter inputs | via `_set_operands` | -- | -- | -- | inherit above; each passes a `cause` string | `_replace_inputs` substitutes an `UNTRANSLATED_NODE_TYPE` operand for an absent one: that placeholder is a graded fact on the node, not a book row | -- |
| `fork_read_scope` | `scope_registry` + every page | `(forked scope, ...)` copies of all source-scope rows at **column 0**; `identity_transition` `(forked scope, "scope")` -> `("fork", source scope, cause)` | mint, `set`, concord | PARTIAL: one edge at scope granularity; every copied row loses its history (columns flattened to 0) and carries no per-row edge | no | copies' own rewrites |

### 2.3 Numeric and precision pages (one row per value in a numeric scope)

All use the `set0` pattern; all are keyed `(numeric scope string, value id)`
unless noted.  The numeric scope comes from `_source_numeric_scope`, whose
fallback `"<module>"` becomes a row key when a graph has no `function_name`:
a fallback recorded as a fact.

| enclosing function | page | fact shape | EDGE | fallback-as-fact | readers |
|---|---|---|---|---|---|
| `specialize_python_precision_widths` | `source_numeric_specialization_concordance` (row `(authored scope, digest)`) | `(specialization scope, receipt of specialized inputs)` | PARTIAL: joins two scopes; the digest is a hash of the receipt, not a row | no | tests only |
| `specialize_python_precision_widths` | `source_precision_pack_concordance` | `(leaf ids, width, packed value id)` | PARTIAL (operand ids, same scope) | no | none outside writer |
| `specialize_python_precision_widths` | `source_precision_operator_concordance` | `(operation, first operand id, class, width)` | PARTIAL | no | `fortran_c_shell`, `ir_identities`, audit `_source_precision_operator_findings` |
| `_concord_numeric_feature_projection` (helper; callers in `resolve_expression`, `propagate_numeric_field_projections`) | `source_numeric_component_concordance` | `(receiver id, attribute, component path, descriptor receipt or None)` | PARTIAL | `None` descriptor is recorded as the fact when unknown -- the row asserts "no descriptor" as a settled fact | audit `_source_numeric_component_findings`, tests |
| `_concord_source_precision_boundary` (helper; callers in `specialize_python_precision_widths`, `lower_python_precision`) | `source_precision_boundary_concordance` | `(boundary kind, operand id, limbs)` | PARTIAL | `max(limbs or 1, 1)` | `fortran_c_shell`, `glsl_deployment_strategy`, `ir_identities`, audit |
| `_concorded_static_parameter_bindings` | `source_parameter_identity_concordance` (row `(source identity tuple, receiver name)`) | `("static_class", module, qualname)` | NO: the binding came from `_python_bindings` on the AST; nothing names that | no | audit `_source_parameter_identity_findings` |
| `resolve_expression` | `source_type_normalization_concordance` | `(subject id, normalized id, ensured type, guarded type)` | PARTIAL | no | none outside writer |
| `resolve_expression` | `source_python_identity_concordance` | `(identity string, program kind, direct operator, direct attributes)` | NO: names no operand or source row; the same mapping is also stamped on the node as `python_identity_program` | no | none outside writer |
| `reduce_statement` | `source_sequence_mutation_concordance` | `(initial id, method, policy, argument ids, "mapping" or "sequence")` | PARTIAL | `sequence_writable` defaults True when absent, deciding whether the row is written at all | audit `_source_sequence_mutation_findings` |
| `specialize_concorded_same_type_numeric_operator` | `source_numeric_operator_specialization_concordance` | `(receiver id, other id, operation id, descriptor receipt, method address)` | PARTIAL; the method address is a `FunctionTable` private counter | no | audit, tests |
| `resolve_concorded_receiver_constructor` | `source_receiver_constructor_concordance` | `(receiver id, class, limbs)` | PARTIAL: derived from the receiver's `source_value_class_concordance` row, which is not named | no | none outside writer |
| `propagate_numeric_field_projections` | `source_numeric_intrinsic_concordance` | `(receiver id, method, results)` | PARTIAL | no | audit `_source_numeric_intrinsic_findings` |
| `lower_class_operator_calls.commit` | `source_operator_dispatch_concordance` | `(receiver id, class, method name, method address, reflected)` | PARTIAL | no | none outside writer |
| `lower_python_precision.concord_operator` | `source_precision_operator_concordance` | `(operation, receiver id, class, limbs)` | PARTIAL | `max(limbs or 1, 1)` | as above |
| `propagate_call_formal_numeric_types` | `source_function_reachability_concordance` (row `(caller scope, call id, callee scope)`) | True | **YES** structurally: the row itself is the edge caller-call -> callee scope | no | none outside writer |
| `propagate_call_formal_numeric_types` | `source_call_result_identity_concordance` | `(callee address, class, limbs)` | PARTIAL: names the callee by address, not the callee output value's row; the class was read from `identity_table` (a shadow ledger) plus the callee's class rows | only written when exactly one distinct returned class; a disagreement among returns is silently not recorded | none outside writer |

### 2.4 Page `callable_identity_concordance`

| enclosing function | row key shape | fact | mode | EDGE | fallback | readers |
|---|---|---|---|---|---|---|
| `_normalize_lexical_values.first_class_function_node` | `(numeric scope, new node id)` | function address | **`set!`** (no incumbent check; the only unconditional column-0 write in the slice) | NO: address is a `FunctionTable` counter, minted outside the book | `_source_numeric_scope` module fallback | audit `_callable_identity_findings`, tests |
| `resolve_expression` (attribute read of a field holding a function ref) | `(function_name or "<module>", field value id)` | function address | `set0` **then `set(row, 1, ...)` unconditionally on every visit** (column 1 is overwritten, not revised; same fact, so silent) | NO | the row scope is `function_name or "<module>"`, a different scope spelling from `_source_numeric_scope` (no method owner prefix): the same value can have two rows under two scope spellings | same |

## 3. graph_express2.py

| enclosing function | page | row key shape | fact shape | mode | EDGE | fallback | readers |
|---|---|---|---|---|---|---|---|
| `_resolve_class_body_field` | `source_field_identity_concordance` | `(owner key string, attribute)` | an AST reference object (compared by `_same_ast_reference`) | `set0` | NO: derived from the class body and import bindings, none named | owner key falls back from qualified source identity to `definition.name` -- a bare class name as the row scope | audit `_source_field_identity_findings`, tests |
| `_source_class_field_reference` | `source_method_identity_concordance` | `(source identity tuple..., attribute)` | the `ast.FunctionDef` object | `set0` | NO | source identity falls back to `(definition.name,)` | none outside writer |
| `_mark_source_pursuit_active` | `source_pursuit_activation_concordance` | `(identity tuple..., id(definition))` | True | `set0` | NO; the row key includes a **Python object address**, a process-ephemeral identity on the book | identity falls back to definition name | none outside writer (audit `_source_callsite_activation_findings` reads a different page) |
| `ProcessGraph.build_from_ast.ingest_external_class` | `source_record_class_concordance` | `(qualified class identity,)` | the `ast.ClassDef` object | `set0` (identity comparison `is not`) | NO: the class was demanded by a program-ABI parameter record or the retained list; neither is named | no | `build_from_ast` itself (parameter record lookup), tests |

`_apply_boundary_resolution` writes `receipt.mapping()` into node data and
`boundary_namespace_receipts` on the graph: a shadow ledger, not a page (the
`.mapping()` matched by the grep is a receipt serializer, not `PageMapping`).

Shadow ledgers written in `build_from_ast` (section 5): `map_ir`,
`function_parameter_annotations`, `class_definitions`,
`selected_class_identities`, `state_machine_controls`.

## 4. ssa.py: the book-backed tables

### 4.1 How `_BookRows` turns a table into rows

`_BookRows` is a `MutableMapping` over page `<kind>_descriptor` rows
`(owner scope, id)`.  `__setitem__` reads the incumbent, `revise`s the row to
the new descriptor, then calls `on_change(old, new)`; `__delitem__` revises to
None.  `on_change` is `_revise_member_claims`, which for every member id whose
claim set changed revises page `<kind>_member` row `(owner, member id)` to the
union of surviving claims.  So a table write is two or more revisions: the
descriptor row and each member row.  Neither revision names the other; the
join is by owner scope plus the record id embedded in each claim tuple
(`(record id, field name, storage identity, role)`).  EDGE: PARTIAL.

| enclosing function | page | row key shape | fact shape | mode | EDGE | fallback |
|---|---|---|---|---|---|---|
| `_BookRows.__setitem__` (via `SSARecordTable.register`, `SSARecordTable.__init__`) | `record_descriptor` | `(owner, record id)` | `SSARecordDescriptor` | revise | NO: the previous descriptor is only in the row's history; the write does not say what changed it | -- |
| `_revise_member_claims` | `record_member`, `struct_member`, `union_member`, `sequence_member` | `(owner, member value id)` | sorted tuple of claim tuples | revise | PARTIAL (claim tuples embed the container id) | -- |
| `_SSALayoutTable._publish` | `struct_descriptor` / `union_descriptor` | `(owner, row id)` | descriptor | revise via `_BookRows` | NO on this page | -- |
| `_SSALayoutTable._publish` | `layout_state` | `(owner, kind, row id)` | `("resolved", descriptor, edge row or None)` | revise (only if changed) | **YES** when re-declared: names the `layout_supersession` row; NO for a first declaration (edge row None -- the origin of a first layout is unrecorded) | `stage` defaults to the string "declaration" |
| `_SSALayoutTable._record_supersession` | `layout_supersession` | `(owner, kind, replacement id, incumbent id, stage)` | `(incumbent descriptor, replacement descriptor)` | `set` at next free column (page-monotonic) | **YES**: the only write in the slice that is a full edge (from row, to row, stage, both facts) | -- |
| `_SSALayoutTable.register` (identity moved to a new id) | `layout_state` | `(owner, kind, old id)` | `("superseded", (kind, new id), stage)` | revise | YES | -- |
| `_SSALayoutTable.withdraw_superseded_layout_derivations` | `layout_state` | `(owner, container kind, container id)` | `("invalidated", (kind, nested id), reason)` | revise | YES (names the row that changed underneath; not the edge row) | -- |
| `SSASequenceTable.register` | `sequence_column_claims` | `(owner, sequence id, "column_dtypes")` | `(column dtypes, key columns)` | revise, every attempt | NO: the proposing site is not named; the page records that a proposal happened, not from where | -- |
| `SSASequenceTable.register` | `sequence_descriptor` | `(owner, sequence id)` | descriptor | revise via `_BookRows` | NO | -- |
| `SSACallTable.__setitem__`, `__delitem__`, `_BookCallList._commit` | `call_record` | `(owner, caller name string)` | whole tuple of records | revise | NO: a per-record insert/sort/delete is a whole-tuple revision; which record changed is recoverable only by diffing columns | -- |
| `_mint_table_owner` (called from `new_layout_tables`, `SSARecordTable.__init__`, `_SSALayoutTable.__init__`, `_SSALayoutTable.__deepcopy__`, `SSASequenceTable.__init__`, `SSACallTable.__init__`, `IRModule._layout_owner`) | `scope_registry` | `(label, serial)` | True | mint | NO: the label is the function name or "module"; nothing joins a table owner to that function's lexical read scope, numeric scope or graph | label defaults to "table" or "module" when None |

Readers by page name outside ssa.py: `record_descriptor`, `record_member`,
`struct_member`, `layout_state` are read by the audit (`_table_member_findings`,
`_layout_table_findings`, which also calls `supersessions()`).
`layout_supersession`, `sequence_column_claims`, `call_record`,
`sequence_descriptor`, `union_member` have no reader by page name outside
ssa.py; they are consumed through the table interfaces.

### 4.2 Does `SSARecordTable.register` record the widening?

No.  When the incoming descriptor has the same identity, a compatible field
overlap and a compatible instance pool, `register` builds a merged descriptor
(resident fields keep their physical facts; `writable` becomes resident OR
incoming; new incoming fields are appended; `instance_pool` is resident OR
incoming) and assigns it through `_BookRows.__setitem__`.  The book sees one
more revision of `record_descriptor` and the member deltas.  It does not see:

- that a merge (not a replacement) happened;
- which callee's projection supplied the new fields;
- that `writable` was widened, or `instance_pool` adopted from the newcomer.

The decision that the two views were "complementary" is made from the
descriptors alone and is unrecorded.  Compare `_SSALayoutTable.register`,
which for the same situation writes a `layout_supersession` edge and a
`layout_state` fact naming it.  The record table has no equivalent.  The
choice of `writable = resident or incoming` is a fallback-as-fact: a field
becomes writable on the strength of any one projection.

`SSASequenceTable.register` raises on any difference (no merge) but records
every attempt on `sequence_column_claims` without naming the attempting site.

### 4.3 What `_mint_table_owner` scopes mean for causal joins

Every table instance is its own scope `(label, serial)`; a deepcopy mints a
new one (shared between a module's struct and union table via the memo).  The
label is a function name string.  Consequences:

- Rows of one function's record table, its sequence table, its call table and
  its lexical read scope live under four unrelated scopes whose only common
  element is a name string.  A causal edge from a `record_member` claim to the
  `lexical_read_binding` row of the value it claims cannot be written today
  because no page records "owner scope S is the record table of graph G whose
  read scope is R".
- `__reduce__` drops the serial: a pickled table is rebuilt under a new scope
  on load, so facts recorded before pickling are not the rows read after.
- The struct/union pair shares a scope by construction (`new_layout_tables`,
  `IRModule._layout_owner`); that is the one cross-table join the design
  provides, and it is by identity of the scope tuple, not by a recorded edge.

## 5. Ingestion identity: book row or shadow ledger

| fact | writer | book row? | shadow ledger | readers |
|---|---|---|---|---|
| Class declarations, objects, schema node ids | `ProcessGraph.build_from_ast` -> `_map_ir_from_ast`; renumbered by `_normalize_lexical_values` canonical relabel | no | `graph["map_ir"]` | `build_class_navigation_table`, `reduce_abstract_tensor_topology`, `resolve_expression` |
| Parameter annotation spellings | `build_from_ast` | no | `graph["function_parameter_annotations"]`; normalized into `function_parameter_numeric_descriptors` / `local_numeric_descriptors` by `publish_numeric_annotation_descriptors` (reducer) | `input_value` (via `_concord_source_value_class`), linker |
| Locally defined class names | `build_from_ast` | no | `graph["class_definitions"]` | `resolve_expression` |
| Source field provenance | `_resolve_class_body_field` | yes: `source_field_identity_concordance` (fact is an AST object) | none | pursuit, reducer |
| Source method identity | `_source_class_field_reference` | yes: `source_method_identity_concordance` | none | pursuit |
| Retained record class | `ingest_external_class` | yes: `source_record_class_concordance` (fact is the `ClassDef` object) | `definition._python_record_identity_keys`, `_python_bindings` on AST nodes | `build_from_ast`, pursuit |
| Pursuit activation | `_mark_source_pursuit_active` | yes, keyed by object address | `definition._source_pursuit_active` | pursuit |
| Callable identity (first-class function ref) | `first_class_function_node`, `resolve_expression` | yes: `callable_identity_concordance` (fact is a `FunctionTable` address) | node attributes `function_ref`, `first_class_function_ref` | audit, linker |
| Function addresses themselves | `FunctionTable.declare` (`_next_address` counter) | **no** | `FunctionTable._entries`, `_qualified` | everything that holds a `function_ref`, `method_ref`, `callee_ref` |
| Name -> value id history | `_normalize_lexical_values` (`identity_bindings`, published at canonical relabel) | no | `graph["identity_table"]`, `graph["ingestion_identity_table"]`; rewritten again in `aot_compile` through `hierarchical_root_value_ids` | 20+ linker/deployment functions (master list B1/B2), `propagate_call_formal_numeric_types`, `specialize_python_precision_widths` |
| Ingestion id -> canonical id | canonical relabel | no | `graph["ssa_identity_tokens"]` | `precompile_to_ssa`, `glsl_deployment_strategy`, `aot_compile._feed_origins_from_ssa_identity`, `site_bundle` |
| Class table (methods by address, fields, defaults) | `reduce_abstract_tensor_topology` | no | `graph["class_table"]` | `resolve_concorded_receiver_constructor`, `lower_class_operator_calls`, `specialize_concorded_same_type_numeric_operator`, linker |
| Attribute slot (class identity, slot) on GetAttr/SetAttr | `resolve_expression`, `bind_target` | no | node attribute `attribute_slot` | `reduce_statement`, linker |
| Scalar value kind on Inputs | `reduce_abstract_tensor_topology` (function-subgraph filter) | no | node attribute `value_kind` | lowering |
| Operand positions (`parents`) | `_set_operands` | movement only (`identity_transition`); the list itself is node data | node `parents` / `children` | everything |
| Per-node numeric facts | every `_concord_source_value_class` caller also stamps the node | duplicated: book row AND node attributes `result_class_ref`, `precision_limbs`, `numeric_feature_descriptor`, `numeric_identity_source` | node attributes | linker reads the attributes, not the page (master list A5) |
| Python identity program | `resolve_expression` | duplicated: `source_python_identity_concordance` AND node attribute `python_identity_program` (`identity_program.mapping()`) | node attribute | lowering |
| Return-site slot values | `_normalize_lexical_values.reduce_statement` (keyed by source span); remapped by the canonical relabel; rewritten by `glsl_deployment_strategy`; consumed and re-emitted by `ssa_record_return_state` | **no** | `graph["return_slot_values"]` | `ssa_record_return_state`, `fortran_c_shell`, `glsl_deployment_strategy`, `loop_composer` |
| Return-site record field states | `reduce_statement` (from `attribute_value_nodes`, filtered to receivers in the slot values) | **no** | `graph["return_record_field_states"]` | `ssa_record_return_state` (reads both receipts by span), `glsl_deployment_strategy` |
| Return container kinds | `reduce_statement` | no | `graph["return_container_kinds"]` | lowering |
| Boundary namespace receipts | `_apply_boundary_resolution` | no | node data `boundary_receipt` and `graph["boundary_namespace_receipts"]` | pursuit |
| ctypes struct/union row ids | `CTypesInterception._struct_row`, `_union_row`, `_scalar_member_struct` (ids from the injected `mint` callable) | rows land on the book through `struct_table.register` (stage "ctypes_interception") | `_struct_ids`, `_union_ids` memo (type object -> row id) | `intercept`, `member_path_layout` |
| Lexical read scope, operand position scope, numeric scope | `_normalize_lexical_values`, `fork_read_scope`, `specialize_python_precision_widths` | the scope exists on `scope_registry`; **which graph owns it is only on the graph** | `graph["lexical_read_scope"]`, `graph["operand_position_scope"]`, `graph["source_numeric_scope"]` | every reader of the pages |

## 6. Files in the slice with no book writes

- `ctypes_layout.py`: writes reach the book only through
  `_SSALayoutTable.register`; the type-object memo is private.  The ids come
  from a caller-supplied `mint` callable, not from the book.
- `python_identity_programs.py`: `mapping()` is a receipt serializer; no
  book access.  Its programs become node attributes and one `set0` row
  (`source_python_identity_concordance`) in `resolve_expression`.
- `function_table.py`: mints function addresses from a private counter and
  keeps `_entries`, `_bindings`, `_qualified` privately.  Those addresses are
  then recorded as facts on four pages (2.3, 2.4).  This is the master list's
  "never a process-wide counter" rule violated at the root of every callable
  identity.
- `aot_compile.py`: no book access.  Rebuilds `identity_table` by
  substituting `hierarchical_root_value_ids`, and recovers feed origins from
  `ssa_identity_tokens` (`_feed_origins_from_ssa_identity`): two shadow
  ledgers joined by value id.
- `GLOBAL_MONOTONIC_IDS`: not referenced in any file of this slice.  Its
  users are `fortran_c_shell.py` and `deployment_ssa_binding.py` (outside the
  slice); ids it mints reach this slice's tables as record, sequence and
  struct ids with no minting edge on the book.

## 7. Verdict

### Counts

- topological_reducer.py: 51 write sites (18 through
  `_concord_source_value_class`, 2 row copies of that page, 13 `set0` sites
  on 13 numeric/precision pages, 2 `callable_identity_concordance` sites, 4
  `lexical_read_binding` sites, `_set_operands` writing 3 pages, 2 scope
  mints, `fork_read_scope`, and 8 `_set_operands` callers).  EDGE: YES 2
  (`identity_transition` move/retire/fork; `source_function_reachability_concordance`),
  PARTIAL 30, NO 19.
- graph_express2.py: 4 book writes (all NO) plus 6 shadow-ledger writes.
- ssa.py: 7 scope mints (NO), `_BookRows` revisions on 5 descriptor pages
  (NO), `_revise_member_claims` on 4 member pages (PARTIAL), `layout_state`
  3 fact kinds (YES for re-declaration and withdrawal, NO for first
  declaration), `layout_supersession` (YES), `sequence_column_claims` (NO),
  `call_record` 3 writers (NO).  One full edge in the whole slice:
  `_record_supersession`.
- ctypes_layout.py, python_identity_programs.py, function_table.py,
  aot_compile.py: 0 book writes; 3 private identity sources.

Fallback-as-fact, by kind: width one when limbs are unknown (5 helper paths
plus one hard-coded `precision_limbs=1`); `"<module>"` and bare
`definition.name` as row scopes; `"declaration"` as a default stage;
`writable`/`instance_pool` OR-merge in `SSARecordTable.register`; a `None`
descriptor recorded on `source_numeric_component_concordance`;
`sequence_writable` defaulting True to decide whether a mutation row exists.
Silent non-recording (absence, not an unresolved mark): return roots with
several bindings; call results with several returned classes;
`_set_operands` with no position scope.

### The ten most load-bearing NO sites

1. `SSARecordTable.register` complementary-view merge -> `_BookRows.__setitem__`:
   the widened descriptor replaces the incumbent as a bare revision; no edge,
   no callee named, no "merge" fact (4.2).
2. `_normalize_lexical_values` canonical relabel copying
   `source_value_class_concordance` rows from ingestion ids to canonical ids:
   the id mapping exists only on the graph, so every numeric class fact after
   this point has no recorded origin.
3. `_mint_table_owner` scopes joined to functions by label string only (4.3):
   no edge can cross from any table row to any read-scope or numeric-scope
   row.
4. `specialize_python_precision_widths` copying authored-scope class rows into
   the specialization scope: the per-row lineage is lost; only a
   scope-level join survives on `source_numeric_specialization_concordance`.
5. `first_class_function_node` unconditional `set` on
   `callable_identity_concordance`, with a fact minted by `FunctionTable`'s
   private counter.
6. `_concord_source_value_class`'s `source` free text, especially
   `propagate_call_formal_numeric_types`' "call actual ... via ..." chain:
   the causal chain is written where the audit cannot read it.
7. `SSACallTable.__setitem__` / `_BookCallList._commit`: whole-tuple
   revisions keyed by caller name string.
8. `SSASequenceTable.register` -> `sequence_column_claims`: proposals without
   proposers.
9. `ingest_external_class` -> `source_record_class_concordance`: an AST
   object as fact, no edge to the ABI parameter record or retained list that
   demanded it.
10. `_mark_source_pursuit_active`: a row keyed by `id(definition)`, an
    address that means nothing after the process ends and can collide across
    compiles in one process.

### Ingestion facts not on the book (roots that cannot anchor an edge)

The first identity anything in this pipeline has is one of: an AST node
(ingestion id), a name binding, a function address, a class declaration, a
parameter annotation, a return-site span.  Of these, only field provenance,
method identity, retained class and pursuit activation reach a page, and each
as an object fact with no edge.  Not on the book at all: `map_ir`,
`function_parameter_annotations`, `class_definitions`, `identity_table`,
`ingestion_identity_table`, `ssa_identity_tokens`, `class_table`,
`attribute_slot`, `value_kind`, `return_slot_values`,
`return_record_field_states`, `return_container_kinds`,
`boundary_namespace_receipts`, function addresses, the graph-to-scope
ownership (`lexical_read_scope`, `operand_position_scope`,
`source_numeric_scope`), and `GLOBAL_MONOTONIC_IDS` mints.  Because these are
the sources, every PARTIAL row above that names "a value id in this scope"
bottoms out in a graph attribute, not a row: the book today has leaves, not
roots.

### Where this reading contradicts CONCORDANCE_MASTER_LIST.md

- A1 marks `SSARecordTable`, `SSASequenceTable`, `SSACallTable` BOOK.  As
  storage, yes.  As causal record, the merge in `SSARecordTable.register` is
  unrecorded, `sequence_column_claims` and `call_record` have no page-name
  reader outside `ssa.py`, and every table scope is joined to its function by
  a string.  Status for the "Record" leg of the merge event: PRIVATE.
- A1 / working rules: "scopes are minted by the compile's book, never by
  process counters".  Function addresses (`FunctionTable._next_address`) and
  `id(definition)` are recorded as facts and row keys on five pages.
- A3a says `_append_operand` appends "through the writer".  It does, but an
  append records no fact about the new position's origin; only moves, retires
  and forks are recorded.
- B1 lists `reduce_abstract_tensor_topology` as a no-book function reading
  `binding_name`.  Its nested functions hold about twenty book writes and its
  own body calls `_set_operands`; the static scan attributed nested functions
  separately, so the row is true of the outer body only if `_set_operands`
  is not counted as a book touch.
- B1 lists `graph_express2._expand_unresolved_ast_parents` as the module's
  only identity-reading function.  `_resolve_class_body_field`,
  `_source_class_field_reference`, `_mark_source_pursuit_active` and
  `build_from_ast.ingest_external_class` all write pages.
- A3a "Audit: `operand_position_orphan`" -- present.  But no audit finding
  reads `identity_transition`; the `cause` strings `_set_operands` records
  are written and never read by the audit.
