# A laid-out (ctypes) type through the class frontend -- path map

Date: 2026-09-29. Status: **read-only trace; nothing implemented.**
Companion to `docs/UNION_TYPE_DESIGN_2026-09-29.md` (read that first) and
to the struct/union rows landed in `src/transmogrifier/ssa.py`
(`SSAStructDescriptor`, `SSAUnionDescriptor`, `IRModule.struct_table`,
`IRModule.union_table`) and `src/transmogrifier/ctypes_layout.py`
(`CTypesInterception`, `member_path_layout`).

Method: `git grep` and reading only. Locations are `file:function`; line
numbers drift and are omitted. Compiler ids and numberings are ephemeral
and are never written here. Where a claim is about what a pass records,
the page name or the private structure is named so it can be checked with
`identity_book` logs rather than by re-running anything.

## 0. The question this answers

A Python class annotated as a laid-out type (a `ctypes.Structure` /
`ctypes.Union` subclass) must travel through the frontend the way an
ordinary class does today and land on the module's `struct_table` /
`union_table`, with attribute access lowering to address + load/store in
source order. Today an ordinary class travels through nine hops. This
document states, per hop: the fact written, where it lives (identity
concordance page or private structure), who consumes it, and whether a
laid-out type passes through unchanged, needs a new row, or replaces the
hop's mechanism.

The one-sentence finding: **every hop below records a class as a set of
independently stored leaves (decomposed columns); no hop records a type as
a byte layout, and the only place the compiler has ever held "field ->
position" is an ordinal slot (`attribute_slot`, `SSAClassField.slot`), not
an offset.** The struct/union rows are the first layout facts in the
system, and nothing yet reads or writes them on the class path.

## 1. The class path today, hop by hop

### Hop A -- source -> `map_ir`

- Where: `src/transmogrifier/graph/graph_express2.py:ProcessGraph.build_from_ast`
  calls `_map_ir_from_ast`, which calls `_class_schema_from_ast` for every
  `ast.ClassDef` in the tree.
- Fact written (private structure `graph.G.graph["map_ir"]`):
  `objects` -- one record per class: `class_name`, `class_identity`
  (dotted `_python_source_identity` when the class was pursued from another
  module, else the bare name), `attributes` (each `name`, `identity`
  `Class.attr`, `storage` = `"class"` for a class-body `Assign`/`AnnAssign`
  or `"instance"` for a `self.attr` assignment found inside a method,
  `annotation` text), `methods` (name, `graph_identity`, AST node id,
  parameter names). Also `schema` (module/class/function annotations),
  `state_machines` (classes whose `bases` name `AbstractTensorStateMachine`
  -- the one place a base class is read at ingestion), `graphs`.
- Also written on the graph: `class_definitions` (frozenset of locally
  defined class names, "published once so no later pass rediscovers it"),
  `function_parameter_annotations` (per function, parameter -> annotation
  text), `selected_class_identities` for retained external classes.
- External classes: `build_from_ast(..., source_record_classes=...)`
  (`ingest_external_class`) turns a live class into a `ClassDef` via
  `_source_ast_definition` (`inspect.getsource`), attaches its methods
  (`_attach_external_methods`), appends it to the tree, and writes page
  **`source_record_class_concordance`** row `(qualified identity,)` fact =
  the definition node. `_filter_discovered_definition` rebuilds a pursued
  class with `bases=[]` and only the methods that are called -- the base
  class fact is dropped here for discovered definitions.
- The sanctioned entry (`src/compiler/fortran_c_shell.py:lower_ast_source_to_ssa`
  -> `_lower_ast_source_to_ssa_impl`) selects which bound classes become
  `source_record_classes`: those `python_bindings` values that are classes
  and whose `module.qualname` equals a `program_abi.records[*].identity`
  from the extraction contract.
- Field identity: `graph_express2` writes page
  **`source_field_identity_concordance`** row `(owner identity, attribute)`
  fact = the constructor-provenance AST reference of a class field.
- Consumers: Hop B (`build_class_navigation_table` reads `map_ir.objects`),
  the reducer (`class_definitions`), `oop_schema.class_schema_from_map_ir_object`
  (interchange layer, reads `map_ir.objects`).

### Hop B -- the navigation table (and the reducer's `class_table`)

- Where: `src/compiler/shell_reference_tables.py:build_class_navigation_table`.
  Memoized on the graph as `graph.G.graph["_class_navigation_table"]` by
  `src/common/tensors/topological_reducer.py` (once per graph). Also built
  whole-compilation by `fortran_c_shell` (passed as
  `compilation.class_navigation`), `process_graph_autograd`, `aot_compile`,
  `site_bundle`.
- Fact written (private structure `ClassNavigationTable`): per class a
  `ClassNavigationRecord(identity, permissions, members,
  instantiation_functions)`; per member `ClassNavigationMember(name,
  identity, kind, storage, function_reference, permissions, slot)`. `slot`
  is computed by `graph_express2.instance_attribute_slot`: the ordinal of
  the attribute among `storage == "instance"` attributes in declaration
  order; `None` for class-level attributes and methods. Methods get
  `function_reference` = the function table address of `Class.method`.
- A second, independent catalogue: the reducer writes
  `graph.G.graph["class_table"]` (private dict) per class: `methods` (name
  -> function-table address), `fields` (class-body `AnnAssign` names plus
  every `self.x` attribute target walked from method bodies, plus aggregate
  field kinds), `field_defaults` (literal class-body values), inherited
  from bases by name. This dict, not the navigation table, is what SSA
  lowering reads for slot order (Hop E).
- Consumers: Hop C (`resolve_dot`), Hop D (`lower_class_navigation_to_ssa`),
  `glsl_deployment_strategy` (planner: `_CompiledStructuralClass` from
  `class_table`), `fortran_c_shell._field_slot_ops` (`class_table[owner]["fields"]`).

### Hop C -- `attribute_slot` stamps on GetAttr/SetAttr nodes

- Where: `topological_reducer.py` inside the per-function reduction:
  `parameter_class_names[name] = annotation.id` when the annotation is a
  bare `ast.Name` naming an identity in the navigation table (or a
  numeric-wrapper descriptor's type name). `_resolve_instance_attribute_slot`
  calls `navigation_table.resolve_dot(class, attr, permissive,
  receiver_kind="instance")` and returns `member.slot` for attribute members.
- Fact written (private structure: graph node `attributes`):
  - `resolve_expression` creates the `GetAttr` node
    (`attributes["attribute"]`, `source_type`) and, when the receiver is a
    parameter with a known class, stamps `attributes["attribute_slot"] =
    (class_identity, slot)`.
  - `bind_target` creates/marks the `SetAttr` node, records the field
    spelling `obj.field` into `identity_bindings` (`record_ingestion_definition`),
    and stamps the same `attribute_slot`.
  - The parameter `Input` node gets `attributes["result_class_ref"] =
    class_identity` (graph view of the class fact).
- Page written: **`source_value_class_concordance`** row `(scope, value id)`
  fact `(class identity, precision limbs, source)` via
  `_concord_source_value_class`; re-keyed to canonical ids when the reducer
  publishes `graph.G.graph["identity_table"]` (name -> value ids, including
  `obj.field` spellings).
- Consumers of `attribute_slot`: only the reducer itself (recovering a
  receiver class from attribute-effect nodes when the receiver's class is
  unknown) and `glsl_deployment_strategy` (copied wholesale in `attrs`,
  per its own comment). **No SSA lowering pass reads `attribute_slot`.**
  Hop E derives its slot independently from field order.

### Hop D -- `SSAClassTable`

- Where: `src/compiler/precompile_to_ssa.py:lower_class_navigation_to_ssa`.
  On the sanctioned path `fortran_c_shell._class_surface_ssa_program`
  calls it with `compilation.class_navigation`, then rewrites each
  `SSAClassMethod.function_name` from the lowered `function_symbols`, and
  hands the table to `IRModule(..., class_table=...)`. Modules are merged
  by `precompile_to_ssa` (repository merge) with a conflict raise per identity.
- Fact written (private structure on the module): `SSAClassDefinition(identity,
  fields=(SSAClassField(name, slot),...), methods=(SSAClassMethod(name,
  function_reference, function_name),...))` -- the navigation table's
  ordinal slots, verbatim. The four LUT functions (`lookup`, `instantiate`,
  `resolve`, `permission`) carry `class_navigation_lut` as an attribute.
- Consumers: `ssa_fortran_backend` publication only (`"class_table_schema":
  "turing.repository-ssa-class-table.v1"`); `oop_schema` for interchange.
  **Neither `ssa_c_backend` nor `ssa_llvm_backend` reads `class_table`.**

### Hop E -- instance field access -> slot arena + GEP/Load/Store

This is the only place attribute access becomes "address + load/store in
order" today, and it exists only for a method receiver (`self`).

- Where: `fortran_c_shell._field_slot_ops` builds `slot_of = {field:
  index}` from, in priority order, `parameter_record_abi["self"].fields`
  (the contract) else `class_table[owner]["fields"]` (the reducer dict);
  walks graph nodes in node-id (source) order and emits `field_ops =
  (("read"|"write", value id, slot), ...)` for every `getattr`/`setattr`
  on a slot-known attribute. `precompile_to_ssa` (method lowering) then
  calls `_inject_field_slot_access`.
- Fact written (SSA instructions, private): `self` becomes one or more
  typed column arenas (`separate_field_storage` when the fields have
  declared contracts); each read becomes `Const offset` +
  `GetElementPtr(column, index)` + `Load` placed before the first consumer;
  each write becomes `Const` + `GetElementPtr` + `Store` placed after the
  source producer. Formal accounting: `program_abi_parameter: "self"`,
  `program_abi_field`, `program_abi_storage: scalar|reference`,
  `program_abi_rank: 0`, `program_abi_mutable`, `program_abi_field_written`.
  The parameter list is rebuilt `(*receiver_columns, *non_self_params)`.
- Indexing semantics: GEP index is an **element** index into a column
  whose element dtype is the column's dtype. There is no byte offset
  anywhere on this hop.
- Consumers: backends (Hop I) render the GEP/Load/Store directly.

### Hop F -- parameter record ABI materialization (the contract path)

For a parameter whose class is declared in the extraction contract
(`program_abi.records` + `program_abi.bindings`), the class is not lowered
via Hops D/E at all; it is decomposed into per-field formals.

- Where: the planning stage in `fortran_c_shell` selects, per function by
  `fnmatchcase(function, binding.function)`, the declared records into
  `graph.graph["parameter_record_abi"]` (also copied into
  `function.metadata["parameter_record_abi"]`, which the C backend reads).
  `_lower_optional_record_presence_graph` rewrites `is None` tests into
  `<param>.<field>.__present` Inputs first. Then
  `fortran_c_shell.materialize_parameter_record_abi(symbol, graph)`.
- Fact written:
  - `coalesce_record_field_storage`: every `getattr` of a declared field on
    the parameter, plus write sources of `setattr`, are aliased to one
    resident value. Page **`record_storage_alias`** (scope minted via
    `mint_scope(("record_storage_alias", symbol))`), page
    **`record_field_resident_concordance`** row `(symbol, parameter id,
    field)`; the alias application is published through
    `_publish_concordant_function_aliases` onto pages
    **`planning_value_concordance`** / **`planning_alias_transition_concordance`**.
  - The `GetAttr` result value ids **become the formals**: for each
    candidate (`record_field_candidates`: `getattr` nodes and constant-key
    `.get` calls on the parameter) a formal is created or re-accounted with
    `program_abi_record`, `program_abi_parameter`, `program_abi_field`,
    `program_abi_storage` (scalar/span/reference/keyed/sequence/table),
    `program_abi_rank`, `program_abi_mutable`, `program_abi_field_written`,
    `program_abi_fixed_length`, `program_abi_optional_payload`; optional
    fields add a `bool` presence formal with `program_abi_optional_presence`.
  - Keyed fields become `length`/`keys`/`values` part formals plus the
    mapping occurrence, and page **`record_field_decomposition`** row
    `(table owner, record id, storage identity)` fact = role -> member id
    tuple. Sequence fields register `SSASequenceDescriptor` rows.
  - Nested and row records (`materialize_nested_record`): a keyed field's
    `value_record` rows are materialized **per indexed occurrence** with a
    `row_handle_id`; a `table` leaf becomes column arenas plus
    `program_abi_child_table_row` (`Mul` row handle x stride) and
    `program_abi_child_table_column` (`GetElementPtr(arena, row_offset)`)
    -- the one place a row is addressed through a handle rather than
    decomposed. Leaves are minted **only for fields the function reads**
    (`record_field_candidates`), or when an access receipt demands them
    (`field_access_receipts`; page
    **`numeral_leaf_materialization_concordance`** records
    `required_unread_leaf`).
  - Access facts: `note_record_field_access` writes page
    **`record_field_access`** (path page) row `(access scope, (parameter
    key, field path, storage identity), role)` fact = witness; page
    **`record_parameter_value`** row `(access scope, (symbol, value id))`
    fact = `(symbol, parameter)`.
  - Finally `table.register(SSARecordDescriptor(record id, identity,
    fields))` on `all_record_tables[symbol]` (`SSARecordTable`, whose rows
    live on pages **`record_descriptor`** `(owner, record id)` and
    **`record_member`** `(owner, value id)` -> claims `(record id, field
    name, storage identity, role)`; "nothing is kept beside the book").
- Consumers: Hop G (linking), `ssa_self_check`, backends (Hop I),
  `ssa_fortran_backend` publication (`record_table_schema`).

### Hop G -- call linking of record members

- Where: `fortran_c_shell._propagate_record_field_demand` (grows caller
  formals to a fixed point, then `_harmonize_call_argument_shapes`) and
  several later frame-completion sites all call
  `fortran_c_shell._linked_caller_member`.
- Facts read: callee `record_member` claims for the formal; the call's
  `argument_bindings` (pairs of caller record id / callee record id; page
  **`argument_binding`** row `(callee, formal id, "binding")` per callsite
  column, kinds `caller_storage`/`caller_alias`/`caller_value`; page
  **`argument_binding_resolution`**); for nested records page
  **`call_record_pair_concordance`** row `(caller, callsite id, callee child
  record id)` fact = caller child record id, written by the call-planning
  pass that pairs nested `RECORD` fields by `storage_identity`; the
  caller's `record_descriptor` row; `record_field_decomposition` for
  sequence-role members.
- Rule: the caller field is found **by `storage_identity` equality** on the
  bound caller record. Scalar fields resolve to the one caller input slot;
  `value` members by position; sequence members by role. If the bound
  caller record has no field of that storage identity and no
  `grow_caller_field` hook supplies one, it raises "...bound to caller
  record ..., which has no such field".
- Fact written: `grow` (only inside `_propagate_record_field_demand`)
  appends a caller formal copying the callee field's accounting with
  `program_abi_field_written: False` and re-registers the caller
  descriptor with the grown field. Page **`program_abi_frame_transition`**
  row `(function, formal id)` records when a provisional frame lease is
  superseded by a ProgramABI field (`identity_concordance`).

### Hop H -- self check

- Where: `src/compiler/ssa_self_check.py:check_formal_parity`.
- Rule: a root formal is admitted if it is in `metadata.parameter_names`,
  `storage_formals`, `closure_formals`, `parameter_member_formals`, or
  carries `accounting["program_abi_parameter"]`. Failures are written to
  page **`formal_parity`** row `(function, value id, "unaccounted_formal")`.
- `check_optional_merges` is the precedent gate for a typed alternative.

### Hop I -- backends

- C (`src/compiler/ssa_c_backend.py`): `record_parameter_ids` = root
  formals whose `parameter_names` entry is a key of
  `metadata["parameter_record_abi"]`. A record parameter's arena is a
  **cell table: one 8-byte cell per declared field in declaration order**;
  the relocation prologue copies SCALAR values and SPAN pointers into those
  cells from the bound buffers (reading `module.record_tables[function]`).
  `GetElementPtr` emission picks the element type from the base:
  `_pointer_value_depth` (`ptr` -> depth 1 -> element `double`;
  `ptrptr_float64` -> `double *`; else `buffer_type(base)`); the index is an
  element index. Buffer dtypes come from `solved_buffer_type` (physical
  dtype union-find, else `_value_llvm_type` mapped by
  `dtype_layout.c_lane_storage_for_llvm_type`).
- LLVM (`src/compiler/ssa_llvm_backend.py`): `_value_llvm_type` from
  `physical_dtype`/`dtype` via `dtype_layout.llvm_type_for_dtype`;
  `_align` is the natural alignment of the LLVM scalar, capped by the
  type table (never above 8); `_frame_slot` allocas `align 8` under 16 KiB
  else `malloc`. GEP is resolved through `aggregate_members` (compile-time
  projection maps for `ssa.aggregate` returns) or as a typed pointer step.
  Reads `program_abi_rank` from accounting; reads neither `class_table` nor
  `record_tables`.
- Fortran (`src/compiler/ssa_fortran_backend.py`): validates that every
  `record_tables` row names live values of its function; publishes
  `class_table` and `record_tables` into artifact metadata with schema
  strings.
- Host side: the wrapper passes one `void *` per `buffer_order` entry; a
  root record parameter is `void **`-table slots per leaf, never one span.

## 2. Fact ledger (what each hop writes and who reads it)

| Hop | Fact | Where it lives | Reader |
|---|---|---|---|
| A | class attributes with `storage` instance/class, methods, annotations | private `graph.G.graph["map_ir"].objects`, `class_definitions`, `function_parameter_annotations` | B, reducer, oop_schema |
| A | external class definition | page `source_record_class_concordance` | ingestion only |
| A | field constructor provenance | page `source_field_identity_concordance` | reducer |
| B | member -> ordinal slot, method -> function reference | private `ClassNavigationTable` (memo `_class_navigation_table`) | C, D, planner |
| B | class fields/methods/defaults | private `graph.G.graph["class_table"]` | E (`_field_slot_ops`), planner |
| C | receiver class of a parameter | node attr `result_class_ref`; page `source_value_class_concordance` | reducer, planner, region planning |
| C | field position on an access node | node attr `attribute_slot = (class, slot)` | reducer only (glsl copies) |
| C | name -> value ids incl. `obj.field` | private `graph.G.graph["identity_table"]` | F (`identities[parameter]`) |
| D | class layout as ordinal slots + methods | private `IRModule.class_table` | Fortran publication, oop_schema |
| E | field read/write as GEP/Load/Store on typed columns | SSA instructions; formal accounting `program_abi_*` | I |
| F | one formal per read leaf with `program_abi_*` accounting | SSA formals | G, H, I |
| F | field storage aliasing | pages `record_storage_alias`, `record_field_resident_concordance`, `planning_value_concordance` | F itself, alias resolution |
| F | record and member claims | pages `record_descriptor`, `record_member` (via `SSARecordTable`) | G, I (C prologue), Fortran |
| F | keyed decomposition roles | page `record_field_decomposition` | G |
| F | access demand | pages `record_field_access`, `record_parameter_value`, `numeral_leaf_materialization_concordance` | F (nested materialization) |
| G | caller/callee pairing | pages `argument_binding`, `argument_binding_resolution`, `call_record_pair_concordance`, `program_abi_frame_transition` | G, frame completion |
| H | unaccounted formals | page `formal_parity` | diagnostics |
| I | cell table layout of a record parameter | C source only (8-byte cells, declaration order) | runtime |

Nowhere in this ledger is there a byte size, an alignment, or an offset.
`dtype_layout.py` now declares per-dtype size and alignment, and the
struct/union rows declare per-type layout, but no hop writes or reads them.

## 3. Where a ctypes type joins this path

Legend: **unchanged** = the mechanism carries the laid-out type as is;
**new row** = the mechanism stays, a new fact/page/attribute is added;
**replaced** = the decomposed-columns mechanism cannot carry a laid-out
span and a different mechanism takes over at this hop.

### Hop A -- new row (and one fact that is currently destroyed)

- `_class_schema_from_ast` on a `ctypes.Structure` subclass records **one
  class-storage attribute named `_fields_` and zero instance attributes**
  (there are no `self.x = ...` assignments). The class therefore has no
  slots at Hop B and no stamps at Hop C. `_filter_discovered_definition`
  strips `bases`, so for a pursued class even the fact "this derives from
  `ctypes.Structure`" is gone before `map_ir` sees it.
- Precedent for keeping a base-class fact: `_state_machine_schema_from_ast`
  reads `definition.bases` for the `AbstractTensorStateMachine` marker. The
  same read gives `map_ir.objects[i]["layout_kind"] = "struct"|"union"`
  when a base is `ctypes.Structure`/`ctypes.Union` (spelled `ctypes.X` or
  imported `X`; both are `_ast_qualified_name` cases).
- The numbers must come from the **live class**, never from parsing
  `_fields_` (the module docstring of `ctypes_layout.py` is explicit). The
  live class reaches the compiler only through `python_bindings`; the
  sanctioned entry already surveys `python_bindings` for classes
  (`source_record_candidates`). `CTypesInterception.is_layout_type(value)`
  belongs in that survey; `intercept` writes the row into the module-wide
  `struct_table`/`union_table` (physical layout record, like
  `tensor_tables`), and the fact is written to a new page
  **`layout_type_concordance`** row `(type identity,)` fact `(kind,
  byte_size, alignment, ((member, offset, size, leaf dtype or nested
  identity, count), ...))`, so a second interception of the same identity
  concords or raises exactly as `SSAStructTable.register` does.
- Identity spelling must concord: `ctypes_layout.type_identity` is
  `module.qualname`; `map_ir` uses `class_identity` (dotted source identity
  or bare name); `ingest_external_class` records `_python_record_identity_keys
  = (qualified, qualname, name)`. The `layout_type_concordance` row must be
  reachable from all three spellings the way record identities are matched
  today (`record_identity == row_identity or record_identity.rsplit(".",1)[-1]
  == row_identity` in the access-receipt pass).

### Hop B -- new row

- `instance_attribute_slot` returns `None` for every ctypes member (none
  is `storage == "instance"`). A laid-out member's position is a **byte
  offset**, not an ordinal; the ordinal is meaningless for a union (every
  member is at offset zero).
- Smallest change: `build_class_navigation_table` emits, for a class whose
  `map_ir` object has `layout_kind`, one `ClassNavigationMember` per
  `_fields_` entry with `kind="attribute"`, `storage="layout"`, `slot=None`,
  and a new optional field `layout=(type identity, member name)` so
  `resolve_dot` still answers "is this a member" and Hop C can ask the
  struct row for the offset. The reducer's `class_table["fields"]` dict
  gets the member names too (it is what `_field_slot_ops` consults), but
  with `slot_of` **not** built from it for a laid-out class (Hop E).

### Hop C -- new row beside `attribute_slot`

- `parameter_class_names` already binds a parameter to its class when the
  annotation is a bare `Name` in the navigation table; a ctypes class
  defined in the source (or ingested via `python_bindings`) is in that set.
  **Unchanged.** `source_value_class_concordance` records the receiver's
  class identity -- **unchanged**, and it is the hook by which Hop E finds
  the layout row (identity -> `layout_type_concordance`).
- New stamp on the `GetAttr`/`SetAttr` node, written in the same two
  places as `attribute_slot` (`resolve_expression`, `bind_target`):
  `attributes["attribute_layout"] = (type identity, member path, byte
  offset, leaf dtype or None, count)` resolved by
  `ctypes_layout.member_path_layout`. A chained access `cell.pair.x` is
  one path: the inner `GetAttr` on a nested struct member yields an
  aggregate `(kind, id)` and the outer one a leaf; the path accumulates
  offsets exactly as `member_path_layout` does.
- Page: **`layout_member_access`** row `(scope, value id)` fact `(type
  identity, member path, byte offset, leaf dtype, count, "read"|"write")`,
  re-keyed to canonical ids when `identity_table` is published, exactly as
  `source_value_class_concordance` rows are today.

### Hop D -- new row

- `SSAClassField(name, slot)` cannot express a layout. Either
  `SSAClassDefinition` gains `struct_id`/`union_id` pointing at the module
  row, or the row's `identity` equals the class identity and the class
  table stays untouched. The second needs no schema change and keeps the
  Fortran publication (`class_table_schema` v1) stable; the cross-reference
  is then the `layout_type_concordance` page. **Recommend the second.**
- `struct_table`/`union_table` need the same publication the other tables
  have: `ssa_fortran_backend` snapshot with schema strings
  (`turing.repository-ssa-struct-table.v1`, `...-union-table.v1`) and the
  repository merge (`precompile_to_ssa`) raising on identity conflict via
  `SSAStructTable.register`.

### Hop E -- replaced

- `_field_slot_ops` / `_inject_field_slot_access` decompose the receiver
  into typed columns indexed by element. A laid-out value is **one base
  address**; members are byte offsets with member dtypes and member
  alignment; a union's members alias the same bytes. Decomposed columns
  cannot represent aliasing at all (the design doc's "side-by-side"
  fallback is exactly the optional's two cells and does not type-pun).
- Replacement lowering for a laid-out parameter (receiver or ordinary
  parameter alike): one formal of dtype `ptr` with accounting
  `ssa_layout_kind`, `ssa_layout_identity` (the names
  `tools/compiler_probes/probe_union_torture.py` already expects),
  `program_abi_parameter` (so Hop H admits it), `program_abi_storage:
  "layout"`. Each read: `Const byte_offset` + `GetElementPtr(base, offset)`
  carrying attributes `layout_type`, `layout_member`, `byte_offset`,
  `leaf_dtype`, `alignment` + `Load` with the leaf dtype; each write the
  same address + `Store`. Placement: the same producer/first-consumer
  schedule `_inject_field_slot_access` uses, so store-then-load order is
  source order and type punning through the union reads back the bytes the
  other member wrote.
- GEP semantics are the wall: both backends treat the index as an element
  count of the base's element type, and a `ptr` base is a `double` payload
  in C. A byte-offset GEP needs either (a) a byte-typed base (`uint8`
  span) with a typed pointer cast at the leaf, or (b) a GEP attribute the
  backends honour (`byte_offset`) so C spells `(leaf_type *)((char *)base +
  off)` and LLVM `getelementptr i8, ptr %base, i64 off` + `load <leaf>,
  ptr, align <member alignment>`. (b) keeps the SSA shape the design doc
  chose for LLVM ("bytes at the row's alignment with typed access at the
  offsets") and avoids inventing a `Cast` op path through
  `_pointer_value_depth`.
- Concordance: every member address is an edge
  `record_shape_transformation(base, leaf, stage="layout_member_projection",
  operation="GetElementPtr", role=member path)`; a write is the reverse
  role. The design doc's `union_injection`/`union_projection` stages are the
  union-specific names of the same two edges.

### Hop F -- replaced for the laid-out parameter; unchanged for everything else

- `materialize_parameter_record_abi` turns every `getattr` result into a
  formal. For a laid-out parameter this is wrong twice: the reads are
  `Load` results (instructions, not inputs), and the parameter itself is
  physical storage, whereas the record path treats the container as
  "correlation only" (`conceptual_record_formal`: a record container with
  no field of its own never crosses the ABI). The laid-out base **must**
  cross the ABI as one `ptr`.
- The join point is the contract: `extraction_contract.ProgramABIField`
  admits `scalar|span|record|reference|keyed|table`. A laid-out type needs
  either a binding kind at the parameter level (`bindings[].layout:
  <identity>` beside `record`) or a new field storage `layout` with `type:
  <identity>` so a laid-out member inside an ordinary record is one
  span-like leaf. The probe binds classes only via `python_bindings` with
  an empty `program_abi`; the parameter's type is then known only from the
  annotation (Hop C). Which of the two is authoritative is question 1.
- Writes: today `coalesce_record_field_storage` aliases every read of a
  field to one resident and marks `program_abi_field_written`; for a
  laid-out member there is nothing to coalesce -- the address is the
  identity -- but the C prologue's "copy SCALAR values into cells" must not
  run for it (the host hands the base address; `run_pointer_table` in the
  probe already does exactly that).

### Hop G -- unchanged

- A laid-out base is one ordinary argument; `argument_binding` pairs it
  like any scalar or span formal. `_linked_caller_member` returns `None`
  (no `record_member` claim) and the value passes positionally. No
  per-leaf caller column has to exist for the callee to read a member,
  because the member fact is on the type row, not on a per-function
  descriptor. This is precisely why the frame-linker failure in Section 4
  does not arise on this path.

### Hop H -- new row

- `check_formal_parity` admits the base through
  `accounting["program_abi_parameter"]`. A laid-out formal minted from a
  bare annotation (no contract binding) has no such key; either mint it
  with `program_abi_parameter = <parameter name>` (truthful: the parameter
  is authored) or add a `layout_formals` metadata channel. A new
  `check_layout_access` beside `check_optional_merges`: every
  `GetElementPtr` with `layout_type` names a row in `struct_table` /
  `union_table`, its `byte_offset` + leaf size fits the row, and a `Load`
  dtype equals the member's leaf dtype; wired into
  `_full_native_link_failures`.

### Hop I -- new row (C, LLVM, Fortran all need a spelling)

- C: `typedef struct { ... } _Alignas(N) S_<identity>;` /
  `typedef union {...}` from the rows (the row is the layout, so the
  typedef is documentary and the access is by byte offset); a `ptr` base
  formal is `void *`; GEP-by-byte per Hop E; `buffer_dtypes` gets `ptr`
  for the base slot (the probe maps `ptr` to `np.uintp`).
- LLVM: `getelementptr i8` + typed load/store with `align <member
  alignment>` (`_align` today caps at 8, which is also ctypes' maximum for
  the declared dtypes, so the helper's promise stays true); a frame-resident
  laid-out temporary is `alloca [byte_size x i8], align <row alignment>`
  (today every alloca is `align 8`).
- Fortran: `bind(C)` derived type per row; publication of both tables.
- Host: nothing new -- one `void *` slot per base in `buffer_order`.

## 4. The frame-linker failure the row-handle probe reproduces

`tools/compiler_probes/probe_row_handle_record_parameter.py`: `World.items`
is a keyed field whose rows are record `Item`; `sync` indexes the mapping,
reads `item.mass`, and passes the row to `center`, whose parameter is bound
to `Item` and reads `item.orientation` (a span the caller never touched).
The raise is `_linked_caller_member`'s "callee formal is member ... of
declared field 'Item.orientation' ... bound to caller record ..., which has
no such field".

Which hop's fact is missing: **Hop F's caller-side leaf.** Two facts exist
and are correct -- the callee's `record_member` claim (callee side,
declaration-driven: `materialize_parameter_record_abi` expands the bound
`Item` into one formal per declared field) and the record pairing
(`argument_bindings` / `call_record_pair_concordance`). The caller's row
record, minted by `materialize_nested_record` with a `row_handle_id`, holds
only the leaves the caller itself read (`record_field_candidates` over the
caller's own `getattr` nodes; `mass` and its `items[].mass.column`), because
caller-side materialization is **demand-driven** while callee-side
expansion is **declaration-driven**. The `record_descriptor` row for the
caller's row record therefore has no field whose `storage_identity` is
`Item.orientation`, and there is no `record_field_decomposition` receipt
for a span leaf (that page only records keyed roles). `grow_caller_field`
is offered only from `_propagate_record_field_demand`, and a grown flat
formal would be the wrong shape anyway: the caller's leaf is a
row-indexed column, not a per-row scalar/span input. The linker is then
asked to choose between two representations of one identity and correctly
refuses.

In the terms of this document: the fact "field `orientation` of type `Item`
is at position P with dtype D" is only ever written **per function, per
read**, as a physical leaf. There is no type-level row a callee could
address into. The parked rule (`docs/PARKED_2026-09-29.md` section 2, "a
record parameter whose type is a keyed field's `value_record` is a row
handle") and the ctypes path are the same statement from two sides: give
the callee the base (handle) plus the type row, and it computes the
address itself. For a ctypes-declared `Item`, that row exists
(`struct_table`), the base is one `ptr`, and Hop G needs no per-leaf caller
column.

## 5. Smallest frontend changes, in order

Each names the process that records its layer on the concordance. Each
step is independently checkable by reading the identity book log.

1. **Intercept at binding intake.** In the sanctioned entry's survey of
   `python_bindings` (where `source_record_candidates` is built), run
   `CTypesInterception.intercept` on every value for which
   `is_layout_type` holds, using the module's `struct_table`/`union_table`
   and the global id issuer. Recorder: the interception writes page
   `layout_type_concordance` row `(type identity,)` (Hop A). Also keep the
   `layout_kind` base-class fact in `_class_schema_from_ast` (do not
   strip it in `_filter_discovered_definition`).
2. **Navigation members for `_fields_`.** `build_class_navigation_table`
   emits `storage="layout"` members with `slot=None` and a `layout`
   reference (Hop B). No page: the navigation table is a private LUT
   today and stays one; its facts are the `map_ir` object plus the row.
3. **`attribute_layout` stamp.** In `resolve_expression`/`bind_target`,
   beside `attribute_slot`, stamp `(type identity, member path, byte
   offset, leaf dtype, count)` from `member_path_layout`; write page
   `layout_member_access` `(scope, value id)`; re-key with
   `identity_table` (Hop C). Recorder: the reducer.
4. **Layout parameter lowering.** A new sibling of `_field_slot_ops` /
   `_inject_field_slot_access` for a parameter whose class has a layout
   row: one `ptr` formal (accounting `ssa_layout_kind`,
   `ssa_layout_identity`, `program_abi_parameter`, `program_abi_storage:
   "layout"`), `Const` + `GetElementPtr(byte_offset, layout_*)` + typed
   `Load`/`Store` in producer/first-consumer order (Hop E). Recorder:
   `record_shape_transformation(stage="layout_member_projection")` per
   access; `materialize_parameter_record_abi` must skip such a parameter
   (Hop F).
5. **Gate.** `ssa_self_check.check_layout_access` and admission of the
   base formal in `check_formal_parity` (Hop H); wired into
   `_full_native_link_failures`. Recorder: page `formal_parity` on
   failure, as today.
6. **Publication and merge.** `struct_table`/`union_table` in the
   Fortran snapshot with schema strings and in the repository module
   merge (Hop D). Recorder: `SSAStructTable.register` conflict raise.
7. **Backend spellings.** C byte-offset GEP + `_Alignas` typedefs; LLVM
   `getelementptr i8` + aligned typed load/store + aligned alloca; Fortran
   `bind(C)` derived types (Hop I). The probe's judge (payload bytes +
   return value against eager ctypes) is the acceptance test; it fails at
   lowering today by the HEAD commit message.

Step 4 is the load-bearing one: it is the first time the compiler emits an
address from a **type** rather than from a per-function column, and it is
the mechanism the parked row-handle rule also needs.

## 6. Open questions for the user

1. **Authority for "this parameter is laid out":** the annotation naming a
   `ctypes` class present in `python_bindings` (what the torture probe
   does, `program_abi` empty), or a contract declaration (`bindings[].layout`
   / a `layout` field storage), or both with the contract winning on
   disagreement?
2. **GEP form for byte offsets:** a `byte_offset` attribute on
   `GetElementPtr` honoured by every backend (recommended above), or a
   byte-typed base span (`uint8`) with an explicit pointer `Cast` at each
   leaf?
3. **Class table cross-reference:** leave `SSAClassTable` untouched and
   join by identity through `layout_type_concordance`, or add
   `struct_id`/`union_id` to `SSAClassDefinition` (schema bump for the
   Fortran `class_table_schema`)?
4. **Row-handle unification:** should the parked rule (a `value_record`
   row parameter is a handle) be implemented as a laid-out type -- i.e. an
   ordinary contract record that is a keyed field's `value_record` gets a
   struct row derived from its declared fields (dtype sizes from
   `dtype_layout`, offsets by natural alignment) -- so the same Hop E
   lowering serves both, or must ctypes remain the only source of rows?
5. **Nested layouts inside decomposed records:** may a laid-out type
   appear as a field of an ordinary contract record (a `layout` storage
   kind at the field level, crossing the ABI as one span), or only as a
   whole parameter for now?
