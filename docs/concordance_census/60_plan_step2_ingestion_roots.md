# 60: Step 2 edit plan -- the ingestion roots go on the book

Read-only planning lane, 2026-09-30.  Companion to
`docs/CONCORDANCE_SINGLE_API_DESIGN_2026-09-30.md` (sections 2 and 6 are the
api; 6.4 is this step) and census files 40 (ingestion section, shadow
ledgers) and 50 (stages A and B).  Everything below was read from source with
`git grep`/`sed`; nothing was run.  Code is named by function.  No compiler
numberings appear.

Files in scope: `src/transmogrifier/graph/graph_express2.py`,
`src/common/tensors/topological_reducer.py` (ingestion and canonical relabel
only), `src/transmogrifier/function_table.py` (one signature).  Reducer
field state (`attribute_value_nodes` / `attribute_effect_nodes`) is step 3
and is not touched here.

## 0. What this plan assumes from step 1, and three things it needs from it

Assumed (design 6.1 / 6.2): `Page(name, row_fields, fact_type)`,
`RowField(name, kind)` with kinds `SCOPE, VALUE_ID, NAME, INDEX, LABEL,
PAGE_REF`, `Stage`, `Transform`, `Ref(page, row, column)`, provenance
`Derived(cells)` / `Novel(transform, operands)` / `Unsourced(reason)`,
`Mode.CONCORD` / `Mode.REVISE`, `Concordance.post(...) -> Ref`, the latch
OPEN, raw primitives auto-tagged `Unsourced(RAW_PRIMITIVE)`.  The working
tree today holds only the shared clock and per-cell stamps on
`IdentityPage`; the names above are the design's, and this plan adapts to
step 1's final spelling without changing anything else.

Needed from step 1 (say so before step 2 starts, or step 2 adds them):

- **N1. A row-keyed NOVEL post.**  Design 6.2 says a `Novel` row carries the
  `NEW` sentinel and `post` mints the id.  A source construct has no minted
  id: its identity IS its row (module, qualname, node path).  `post` must
  admit `Novel(transform, operands=())` on a row with no `NEW`, mint nothing,
  and still write the mint edge `(row) -> (transform, ())`.  Two pages need
  it: `source_span` and `contract_demand`.
- **N2. Registered objects** for the stages, transforms and reasons named in
  section 2.3.  They are registry entries, not strings.
- **N3. `fact_type` may be `ast.AST`** for three existing pages
  (`source_field_identity_concordance`, `source_method_identity_concordance`,
  `source_record_class_concordance`).  Step 2 gives their facts edges; it
  does not change the fact's type (see 3.6 and risk R4 for why not yet).

One wording conflict in the design is resolved here.  Design 6.4 says
`ssa_identity_tokens` is "posted NOVEL per canonical id".  Design 3 (row
S10) and this task's brief say the relabel is DERIVED, new id <- old id.
DERIVED is right: a canonical id is `enumerate(ordered)` -- dense, chosen
by position, consumed as dense by `ordered_graph.add_node(value_id, ...)`
over `range(len(mapping))` and by every later pass -- so it cannot be a
`compose(serial, MINTED)` mint, and it is not a new identity; it is the
ingestion identity renumbered.  Section 3.1 posts it DERIVED.

## 1. The root row: page `source_span`

One NOVEL root per source construct.  Row = `(module, qualname, path)`.
Never `id(node)`, never a line/column pair as key.

### 1.1 `module` (RowField kind SCOPE)

- For a definition discovered by pursuit: `definition._python_source_identity[0]`.
  Set in `_expand_unresolved_ast_parents` (the discovered-definition branch
  and `admit_occurrence_class`, including their member methods) and, for
  retained/record classes, `ingest_external_class`.  It is the live
  `__module__` string.
- For the submitted program's own definitions (no `_python_source_identity`):
  the `filename` argument `build_from_ast` already passes to `ast.parse`,
  with `ast.parse`'s own spelling `"<string>"` when there is none.  This is
  not a fallback fact: it is the name Python itself gives the program.
  `build_from_ast` stores it once as `tree._turing_source_module` before
  `_expand_unresolved_ast_parents` runs, so every helper reads one value.

### 1.2 `qualname` (NAME)

- Pursued definition: `_python_source_identity[1]`.
- Otherwise: `".".join(node._turing_source_scope)` -- the lexical scope
  tuple the `OwnerVisitor` in graph_express2 already stamps on every node
  (`_turing_source_scope`; class/def nodes carry their own name as last
  element).  For a node inside a definition the qualname is the enclosing
  definition's; the path (1.3) locates the node within it.

### 1.3 `path` (LABEL)

The AST field path from the owning definition to the node: a tuple of
`(field_name, index)` steps over `ast.iter_fields`, index `-1` for a
non-list field.  The definition itself has path `()`.  Deterministic,
picklable, unique within one definition body, and stable across the
process.  Positions (`lineno`, `col_offset`, ...) are NOT in the key: pursuit
copies nodes (`copy.deepcopy(generator.iter)`, the comprehension and
closure rewrites) and calls `ast.fix_missing_locations`, so two nodes can
share a span; a path cannot.

Helper (new, module level in graph_express2, next to `_class_schema_from_ast`):

    source_span_row(owner_definition, node) -> (module, qualname, path)
    post_source_span(owner_definition, node) -> Ref     # idempotent

`post_source_span` builds, per owner definition, a transient
`{id(child): path}` index by one walk of `ast.iter_fields` (an in-process
index only, discarded with the walk; `id()` never reaches a row -- the same
discipline `loop_target_bindings_by_ast` uses), then posts
`Novel(INGEST_SOURCE, ())`, `Mode.CONCORD`, stage `INGESTION`.  A second
post for the same row is the design's "same fact, records its edge" no-op.

### 1.4 The fact

`SpanFact(kind, lineno, col_offset, end_lineno, end_col_offset, dump_sha256)`:
node class name, the four positions (what `node_description` and
`record_ingestion_definition` record today), and the sha256 of
`ast.dump(node, include_attributes=False)`.  The positions become a checkable
fact instead of an identity.

### 1.5 When it is posted

After pursuit has finished rewriting the tree, i.e. never before
`_expand_unresolved_ast_parents` returns inside `build_from_ast`.  Every
consumer below posts the span of the node it is about to describe at the
moment it describes it (`ensure_node`, `_class_schema_from_ast`,
`bind_target`, ...).  `_map_ir_from_ast` runs before pursuit today (it is
called at the top of the ingestion block in `build_from_ast`); its posts are
therefore made from a second pass (see edit E4) after pursuit, or
`_map_ir_from_ast` is moved after `_expand_unresolved_ast_parents`.  The
plan moves the call: the map IR describes the tree that is actually
ingested, which is the rewritten one.

## 2. Pages to declare

### 2.1 New pages (14)

| page | row fields (kind) | fact type | provenance | stage |
|---|---|---|---|---|
| `source_span` | module (SCOPE), qualname (NAME), path (LABEL) | `SpanFact` | NOVEL(INGEST_SOURCE, ()) | INGESTION |
| `contract_demand` | kind (LABEL: RETAIN / PARAMETER_RECORD / PURSUIT_ROOT), identity (NAME) | `DemandFact(record or None)` | NOVEL(CONTRACT_DEMAND, ()) | INGESTION |
| `ingestion_value` | ingestion scope (SCOPE), ingestion id (VALUE_ID) | `NodeFact(type, op, label)` | DERIVED(source_span of `expr_obj` or `source`) else Unsourced(SYNTHESIZED_NO_SOURCE) | INGESTION / REDUCTION |
| `canonical_value` | canonical read scope (SCOPE), canonical id (VALUE_ID) | token chain (tuple of str) | DERIVED(ingestion_value cell) | CANONICAL_RELABEL |
| `name_binding` | read scope (SCOPE), name (NAME), version (INDEX) | `BindingFact(value_id, authored, span_positions, context_sha256)` | DERIVED(ingestion_value / canonical_value row of the value; source_span of the target when authored) | REDUCTION / CANONICAL_RELABEL |
| `class_declaration` | module (SCOPE), class identity (NAME) | `ClassFact(class_name, permissions)` | DERIVED(source_span of ClassDef) | INGESTION |
| `class_field_declaration` | module (SCOPE), class identity (NAME), field (NAME) | `FieldFact(storage, annotation text, permissions)` | DERIVED(source_span of every declaring statement) | INGESTION |
| `class_method_declaration` | module (SCOPE), class identity (NAME), method (NAME) | `MethodFact(graph_identity, parameters)` | DERIVED(source_span of FunctionDef) | INGESTION |
| `annotation_declaration` | module (SCOPE), owner qualname (NAME), name (NAME) | `AnnotationFact(annotation text, value text)` | DERIVED(source_span of AnnAssign) | INGESTION |
| `state_machine_declaration` | module (SCOPE), class name (NAME) | `StateMachineFact(marker, bases, transition_identity)` | DERIVED(source_span of ClassDef) | INGESTION |
| `schema_node` | scope (SCOPE), id (VALUE_ID) | True | DERIVED(source_span of the AnnAssign); relabel row DERIVED(ingestion row) | INGESTION / CANONICAL_RELABEL |
| `parameter_annotation` | module (SCOPE), function identity (NAME), parameter (NAME) | annotation text | DERIVED(source_span of the `ast.arg`) | INGESTION |
| `scalar_parameter` | scope (SCOPE), input id (VALUE_ID) | `ValueKind.SCALAR` | DERIVED(parameter_annotation row or source_span of the default literal; the Input's ingestion/canonical row) | FUNCTION_SUBGRAPH / CANONICAL_RELABEL |
| `function_address` | qualified name (NAME) | address (int) | DERIVED(source_span of the definition) else Unsourced(EXTERNAL_DECLARATION) | FUNCTION_TABLE |

### 2.2 Existing pages re-declared (6)

| page | row fields | fact type | provenance after step 2 |
|---|---|---|---|
| `source_field_identity_concordance` | owner key (SCOPE), attribute (NAME) | `ast.AST` or class object (as today) | DERIVED(source_span of each class-body statement that assigned the field) |
| `source_method_identity_concordance` | source identity parts (SCOPE, NAME...), attribute (NAME) | `ast.FunctionDef` | DERIVED(source_span of the method) |
| `source_record_class_concordance` | qualified class identity (SCOPE) | `ast.ClassDef` | DERIVED(contract_demand row, source_span of the ClassDef) |
| `source_pursuit_activation_concordance` | module (SCOPE), qualname (NAME) -- **`id(definition)` removed** | True | DERIVED(source_span of the definition; the demand: caller call-site span or contract_demand PURSUIT_ROOT) |
| `callable_identity_concordance` | numeric scope (SCOPE), node id (VALUE_ID) | address (int) | DERIVED(function_address row, ingestion_value row of the node) |
| `source_value_class_concordance` | numeric scope (SCOPE), value id (VALUE_ID) | `(class, limbs, source)` | ingestion-side writers DERIVED (3.9); other callers Unsourced under the latch |

### 2.3 Registry entries

Stages: `INGESTION` (`build_from_ast`, `ensure_node`, `_map_ir_from_ast`),
`PURSUIT` (`_expand_unresolved_ast_parents` and helpers),
`REDUCTION` (`_normalize_lexical_values` closures),
`CANONICAL_RELABEL` (the relabel tail of `_normalize_lexical_values`),
`FUNCTION_SUBGRAPH` (the function-subgraph filter in
`reduce_abstract_tensor_topology`), `FUNCTION_TABLE` (`FunctionTable.declare`).
Transforms: `INGEST_SOURCE` (arity 0), `CONTRACT_DEMAND` (arity 0).
Reasons: `SYNTHESIZED_NO_SOURCE`, `EXTERNAL_DECLARATION`,
`HELPER_CALLER_UNROUTED`.

## 3. Each structure

### 3.1 `ssa_identity_tokens` and the canonical relabel

(a) Written in the relabel tail of `_normalize_lexical_values`:
`ordered_graph.graph["ssa_identity_tokens"] = {mapping[node_id]:
node_token_chains[node_id]}`.  Holds canonical id -> structural token chain
(ordering prefix + `structural_context_tokens` of `node_description` +
`version:n`).  Readers: `glsl_deployment_strategy`
(`_structural_region_program_from_subgraph`, `remap_program` -> program
extras), `precompile_to_ssa` (`lower_fused_integral_to_repository_ssa`,
`tensor_algorithm` -> function metadata), `aot_compile._feed_origins_from_ssa_identity`,
`site_bundle` (`program_owner`, `union`), `project_compilation_product.compile_function_shell`,
`training_data_store.put_reduced_graph_view`.  All read the graph dict or
a copy of it in `extras`.

(b) Pages `ingestion_value` and `canonical_value` (2.1).  The ingestion
scope is `ingestion_read_scope = (read_scope, "ingestion")`, already minted
in `_normalize_lexical_values`; the canonical scope is `read_scope`.

(c) `ingestion_value` rows: DERIVED.  Two writers.  `ProcessGraph.ensure_node`
(every AST node ingested; the node's `expr_obj` is the AST node, its owner is
the enclosing definition from `_turing_source_scope`) posts
DERIVED(`post_source_span(owner, node)`).  `new_node` in
`_normalize_lexical_values` (every reducer-synthesized node) posts
DERIVED(source_span of `source`) when `source` is an `ast.AST`, else
`Unsourced(SYNTHESIZED_NO_SOURCE)`.  The latch then lists every `new_node`
caller that passes no `source` -- that list is real work for step 3 (Phis,
Inputs for captured parameters, static constants) and is exactly what the
latch is for.  `canonical_value` rows: one DERIVED post per entry of
`mapping`, cells = `(ingestion_value, (ingestion scope, old id), 0)`, stage
`CANONICAL_RELABEL`, fact = the token chain.  This is the row S10 asked for:
new id <- old id, one per relabelled node.

(d) The dict stays, as a read view materialized once from the page at the
end of the relabel: `graph.G.graph["ssa_identity_tokens"] = {row[1]: fact
for row in canonical_value.scope_rows(read_scope)}`.  Zero reader changes.
(`remap_program` filters a copy into `extras`; unchanged.)

(e) The relabel of the other pages in the same tail, each as DERIVED rows
new <- old with stage `CANONICAL_RELABEL`: `lexical_read_binding` (today
`read_page.concord((read_scope, mapping[row[1]], ...), latest)` -- becomes
DERIVED(ingestion position row)); `source_value_class_concordance` (today a
column-by-column `class_page.set` copy -- becomes one DERIVED post per row,
cells = every column of the ingestion row); `schema_node`,
`scalar_parameter`, `name_binding` (3.2).  The `return` rows of
`lexical_read_binding` (`(scope, "return", "root", position)`) are DERIVED
from their ingestion row too.  `identity_transition` itself is unchanged;
`_set_operands` with `cause="canonical_relabel"` keeps writing it.

(f) Probe: `tools/compiler_probes/probe_annotated_scalar_parameter.py`
(seconds; three one-function programs).  Expected: every `canonical_value`
row has one inbound edge; `unsourced-fact` for stage `CANONICAL_RELABEL`
is zero; the raw-primitive entries from `class_page.set` and
`read_page.concord` in the relabel tail disappear (today: one per copied
row).

### 3.2 `identity_table` / `ingestion_identity_table`

(a) Both are built in the relabel tail from two closure dicts of
`_normalize_lexical_values`: `identity_bindings` (name -> list of value ids;
appended in `input_value`, `bind_target` [Name target; named-receiver
Attribute target], `bind_loop_target`, and `reduce_statement` at four sites:
positional return slots, the conditional name merge, the record-field
merge, loop-carried updates) and `ingestion_definitions` (name -> list of
(value id, source dict); appended by `input_value` with `{}` and by
`record_ingestion_definition`, called from the two `bind_target` sites).
`identity_table` = `{name: tuple(mapping[v] ...)}`; `ingestion_identity_table`
= the authored subset with version, span and structural context tokens.
Readers of `identity_table`: about 60 functions in `fortran_c_shell`
(`parameter_of`, `order_field_effects`, `_sequence_*`, `numeral_leaves`,
`frame_fixed_point_digest`, ...), `glsl_deployment_strategy` (~45),
`loop_composer`, `precompile_to_ssa` (`value_name_histories`),
`process_graph_autograd`, `symbolic_process_graph`, `shell_reference_tables`,
`site_bundle`, `identity_concordance.publish_program_abi_graph_identities`,
reducer `call_return_identity` / `ordered_actuals`.  Readers of
`ingestion_identity_table`: `training_data_store` and one test.

(b) Page `name_binding` (2.1).  Rows are written in the ingestion scope
during reduction and re-posted in the canonical scope at the relabel.
`version` is the position in the list, exactly today's `enumerate`.

(c) DERIVED everywhere; no NOVEL here (the root is the span).  Per writer:
`input_value` -> cells (ingestion_value row of the Input, source_span of
the `ast.arg`), `authored=True`; `bind_target` Name -> (ingestion_value row
of `value`, source_span of `target`), `authored=True`, fact carries the
positions and `context_sha256` that `record_ingestion_definition` computes
(the function folds into the post; its `context_tokens` stay in the fact);
`bind_target` named-receiver Attribute -> same with the SetAttr node;
`bind_loop_target` -> (ingestion_value row of the loop Input, source_span
of the target Name); `reduce_statement` return-slot positional name ->
(ingestion_value row of the returned value, source_span of the Return);
`reduce_statement` conditional name merge and record-field merge ->
(ingestion_value row of the Phi); loop-carried update -> (ingestion_value
row of `updated`).  Canonical rows: DERIVED(ingestion `name_binding` row,
`canonical_value` row of `mapping[value]`).

(d) Both dicts stay as read views materialized from the canonical scope of
the page at the end of the relabel, byte-identical to today's shapes, so
none of the ~110 readers change in step 2.  Note what this does NOT cover:
`glsl_deployment_strategy` (`_alias_projection_to_member`,
`_synthetic_device_scalar_shell`, `remove_node`, `replace_alias`,
`retained_basic_index`), `loop_composer` (`expand_generator`,
`reads_binding`), `process_graph_autograd.lower_training_motion_to_repository_ssa`
and `aot_compile` (`hierarchical_root_value_ids` substitution) MUTATE the
dict later.  Those mutations are planner/loop-composer work (design steps 4
and 5) and stay off the page in step 2; the page is authoritative for the
reduction's own history only.  Say this in the page's docstring.

(e) Covered by (c): the canonical `name_binding` row is the transition row,
one per (name, version).

(f) Probe: `probe_annotated_scalar_parameter` (`k` is one authored
binding; `k + 1` is one return slot) and the reducer test that reads
`executable.graph["ingestion_identity_table"]["signal"]` in
`tests/test_abstract_tensor_topological_reducer.py` (unchanged assertions).
Expected: `unsourced-fact` zero on `name_binding`.

### 3.3 `map_ir` objects, `class_definitions`, `schema_node_ids`, `selected_class_identities`

(a) `build_from_ast` sets `self.G.graph["map_ir"] = _map_ir_from_ast(tree)`
(objects via `_class_schema_from_ast`, state machines via
`_state_machine_schema_from_ast`, `schema` annotations, `schema_roots`,
`schema_node_ids` -- the last three are `id(node)` tuples -- and `graphs`);
then `map_ir["selected_class_identities"]` from `retained_identities`; then
`class_definitions = frozenset(class_name for objects)`.  The relabel tail
renumbers `schema_node_ids` / `schema_roots`.  Readers: `build_class_navigation_table`
and `build_map_dependency_regions` (`shell_reference_tables`),
`oop_schema.class_schemas_from_process_graph`, reducer `_sequence_annotation_dtypes`
and the class-table build in `reduce_abstract_tensor_topology`,
`glsl_deployment_strategy._is_ast_metadata_node` (`schema_node_ids`),
`boundary_namespace.graph_input`, `dual_ir_shell.compose_dual_ir_shell`,
`site_bundle.build_source_inspection_page`; `class_definitions` is read by
`resolve_expression` (local class construction) and `boundary_namespace.graph_input`.
Two later writers replace `map_ir` wholesale (`fortran_c_shell` at the
selection report; `aot_compile` after navigation SSA) -- they add
`runtime/mapped/retained/bindings` blocks; they keep working on the view.
`ast_node_id` / `class_node_id` have no reader outside graph_express2.

(b) Pages `class_declaration`, `class_field_declaration`,
`class_method_declaration`, `annotation_declaration`,
`state_machine_declaration`, `schema_node`, `contract_demand` (2.1).

(c) `_class_schema_from_ast`: per class DERIVED(source_span of the
ClassDef); per field one post with cells = source_span of EVERY declaring
statement `add_attribute` saw (the `seen_attributes` dedup today keeps the
first declaration and drops the rest; the post keeps the first as fact and
names all as cells); per method DERIVED(source_span of the FunctionDef).
`_state_machine_schema_from_ast`: DERIVED(ClassDef span).  `_map_ir_from_ast`
annotations: DERIVED(AnnAssign span) per module / class member / function
local entry.  `schema_node`: for every node under a schema statement,
DERIVED(source_span of that statement) in the ingestion scope -- posted
where `ensure_node` posts the node's `ingestion_value` row (it knows the
statement through `_turing_source_scope` and the path), not by a second
`ast.walk`.  `contract_demand`: NOVEL(CONTRACT_DEMAND) once per retained
class (`retain=`), per parameter record (`source_parameter_records`) and
per pursuit root (`pursuit_roots`), posted at the top of `build_from_ast`;
`selected_class_identities` is then DERIVED(contract_demand RETAIN row).

(d) `map_ir` stays a dict assembled by `_map_ir_from_ast` exactly as today
minus the three `id()` fields, which become the span row key
`(module, qualname, path)` under the same names (`ast_node_id`,
`class_node_id`) -- no reader consumes them.  `schema_node_ids` /
`schema_roots` stay tuples, materialized from `schema_node`'s scope at
ingestion and at the relabel (replacing the two `mapping[...]`
comprehensions).  `class_definitions` stays a frozenset built from
`class_declaration.scope_rows(module)`.  No reader changes.

(e) `schema_node` canonical rows DERIVED(ingestion row), stage
`CANONICAL_RELABEL`.  Class/field/method/annotation rows are keyed by
source identity and do not relabel.

(f) Probe: `probe_annotated_scalar_parameter` has no class; use the audit
tool's `view` case (`Cell` with two fields and three methods) -- seconds --
and assert two `class_field_declaration` rows and three
`class_method_declaration` rows, each with one inbound edge to a
`source_span` row whose `kind` is `Assign`/`FunctionDef`.

### 3.4 `function_parameter_annotations`

(a) Built in `build_from_ast` from `tree.body` (top-level functions and
class members): `{function identity: {parameter: ast.unparse(annotation)}}`.
Readers: reducer `_parameter_declaration`, `_record_numeric_annotation_descriptors`
(-> `publish_numeric_annotation_descriptors`), `fortran_c_shell`
`_current_authored_parameter_annotations` and `scalar_fact_dtype`.

(b) Page `parameter_annotation` (2.1).

(c) DERIVED(source_span of the `ast.arg` node) per annotated parameter,
stage `INGESTION`.  Nested functions (below top level) are not in the dict
today and are not posted; the page mirrors the dict.

(d) The dict is materialized from the page's module scope right after the
posts; readers unchanged.  `_current_authored_parameter_annotations` keeps
its flat-string compatibility branch.

(e) None (keyed by source identity).

(f) `probe_annotated_scalar_parameter` (`bump(k: int)`): one row, one
edge to the `arg` span.

### 3.5 `value_kind` scalar marks

(a) Written twice in the function-subgraph filter of
`reduce_abstract_tensor_topology`, on Input nodes whose `binding_name` is in
`scalar_parameter_names`: once before `_normalize_lexical_values` (ingestion
ids) and once after (canonical ids).  `scalar_parameter_names` comes from
(i) a parameter annotation that is a bare `Name` in `{bool, bytes, complex,
float, int, str}` and (ii) a literal default of a scalar Python type.
Readers: `glsl_deployment_strategy` (the control-expression builder that
admits an Input only when `value_kind == "scalar"`, and one later site).
The tuple `scalar_parameters` is also stored on the graph.

(b) Page `scalar_parameter` (2.1).

(c) DERIVED.  For (i): cells = (`parameter_annotation` row for
(function identity, parameter), the Input's `ingestion_value` row).  For
(ii): cells = (source_span of the default expression, the Input's
`ingestion_value` row).  The second write, after normalization, is the
relabel's job: the post-normalize loop is deleted and the relabel tail
posts the canonical row DERIVED(ingestion row) like every other page
(section 3.1 (e)).  The node attribute `value_kind` is still written
(read view on the node) by the same helper.

(d) Readers unchanged (attribute stays).

(e) Canonical row per Input, stage `CANONICAL_RELABEL`.

(f) `probe_annotated_scalar_parameter`: the annotated forms produce one
row each, edge to the `parameter_annotation` row; the plain form produces
none.  This is also the probe that distinguishes the annotated and plain
programs -- their books must differ by exactly these rows.

### 3.6 `source_field_identity_concordance`, `source_method_identity_concordance`, `source_record_class_concordance`

(a) `_resolve_class_body_field` writes `(owner key, attribute) -> resolved
reference` (`set0`); `_source_class_field_reference` writes
`(*source identity, attribute) -> ast.FunctionDef`; `ingest_external_class`
writes `(qualified identity,) -> ast.ClassDef`.  Readers: `_resolve_class_body_field`
itself (returns the incumbent to pursuit), `build_from_ast`'s parameter
record loop (`record_class_page.latest(...)` then walks `owner.body`),
audit `_source_field_identity_findings`, tests in
`test_process_graph_function_linking`, `test_abstract_tensor_topological_reducer`,
`test_ast_parent_ingestion`.

(b) Re-declared (2.2).  Facts stay `ast.AST` / class objects (N3).

(c) DERIVED.  `_resolve_class_body_field`: cells = source_span of each
class-body statement whose value it resolved (the function iterates
`definition.body`; collect the statements it took `values` from).
`_source_class_field_reference`: cells = source_span of `method`.
`ingest_external_class`: cells = (the `contract_demand` row that demanded
the class -- RETAIN for `retained`, PARAMETER_RECORD for `record_classes` --
and source_span of the ClassDef with module = the class's `__module__`,
qualname = `retained_qualname`).

(d) Readers unchanged: `latest(row)` still returns the object.

(e) None.

(f) Audit `view` case (two `self.<field>` navigations) and
`tests/test_ast_parent_ingestion.py`'s record-class test (assertions
unchanged).  `unsourced-fact` on the three pages: zero.

### 3.7 `source_pursuit_activation_concordance`

(a) `_mark_source_pursuit_active(definition)` writes row `(*identity,
id(definition)) -> True` and sets `definition._source_pursuit_active`.
Callers: the seed loop in `_expand_unresolved_ast_parents`
(`active_seed_definitions`), `requeue_definition` (seven call sites inside
pursuit), and `build_from_ast`'s parameter-record selection.  Reader: the
function-subgraph filter reads the attribute into
`graph["source_pursuit_active"]`; no page reader.

(b) Row becomes `(module, qualname)` = the definition's own span row key
without the path.  `id(definition)` is removed.

(c) DERIVED.  `_mark_source_pursuit_active(definition, *, demand)` gains a
required `demand: Ref`.  Seed loop: `contract_demand` PURSUIT_ROOT row (or,
when `pursuit_roots` is empty and every definition is seeded, the module's
`source_span` row with path `()`).  `requeue_definition`: every one of its
callers holds the call node that resolved to the definition (`node`,
`call`, `entry_call`); pass `post_source_span(call owner, call)`; the two
callers that requeue a member/constructor definition on class admission
pass the class's span.  `build_from_ast` selection: the PARAMETER_RECORD
demand row.  Cells = (demand, source_span of the definition).

(d) The attribute `_source_pursuit_active` is still set; the reader is
unchanged.

(e) None.

(f) Audit `view` case; the count of rows equals the number of activated
definitions and no row key contains an address.

### 3.8 `callable_identity_concordance` and `FunctionTable._next_address`

(a) `FunctionTable.declare` mints `FunctionReference(self._next_address)`
and increments; no page.  Callers: reducer function-subgraph filter (two
`function_table.declare` sites) and `external_function_table.declare`;
`aot_compile`, `process_graph_autograd`, `process_graph_function_linking`,
`symbolic_process_graph` (one or two each).  `first_class_function_node`
writes `callable_identity_concordance` `(numeric scope, node) -> address`
with an unconditional `set`; the attribute-read site in `resolve_expression`
writes `(function_name or "<module>", field value) -> address` with `set0`
then an unconditional column-1 `set` on every visit.  Readers: audit
`_callable_identity_findings`, linker (`function_ref` attributes), tests.

(b) Page `function_address` (2.1); `callable_identity_concordance`
re-declared (2.2).  Addresses stay the dense counter: they are dict keys and
`function_ref` ints throughout; they are not MINTED ids and are not made
so here (see section 6).

(c) `FunctionTable.declare(..., source: Ref | None = None)`: on a new
declaration posts `function_address` `(qualified,) -> address`
DERIVED(source) when given, `Unsourced(EXTERNAL_DECLARATION)` when
`external=True` and no source.  The two reducer callers pass
`post_source_span(owner, statement)`; the external-table caller passes
none; the four out-of-slice callers are left to the latch (they declare
synthetic or linked functions and belong to later steps).
`first_class_function_node`: `Mode.CONCORD`, DERIVED(`function_address`
row for the reference's qualified name, `ingestion_value` row of the new
node).  `resolve_expression` site: scope spelling unified to
`_source_numeric_scope(graph)` (today's `function_name or "<module>"`
drops the method-owner prefix and can give one value two rows), one
`Mode.CONCORD` post DERIVED(`function_address` row, `ingestion_value` row
of `field_value`); the column-1 overwrite is deleted (same fact every
visit; CONCORD records the edge).

(d) `FunctionTable._entries` / `_qualified` unchanged; readers of
`function_ref` unchanged.  Audit `_callable_identity_findings` reads
history: with CONCORD there is one column; the existing test that asserts
`history == (0, 0)` on this page (two columns, same fact) in
`test_process_graph_function_linking` must become `(0,)` -- **one test
assertion changes**.

(e) `callable_identity_concordance` rows are at ingestion ids and are
relabelled today by nobody (the census did not list them; the row is
written before the relabel and never moved).  Add them to the relabel
tail: canonical row DERIVED(ingestion row).  This is a latent bug today
(the audit reads rows the linker can no longer join); step 2 fixes it as a
by-product and the probe in 3.8 (f) checks it.

(f) `test_process_graph_function_linking`'s callable record-field case
(seconds): one `function_address` row per declared function with an edge
to its `FunctionDef` span; one `callable_identity_concordance` row in the
canonical scope, one column.

### 3.9 `source_value_class_concordance` -- ingestion side only

(a) 18 callers of `_concord_source_value_class` plus two row copies
(census 40, 2.1).  In scope here: `input_value` (parameter annotation ->
class, limbs) and the relabel copy.

(b) Re-declared (2.2).  Helper signature gains `provenance`.

(c) `input_value`: DERIVED(`parameter_annotation` row, the Input's
`ingestion_value` row; when the class came from `class_field_classes` or
`static_parameter_bindings`, the `class_declaration` row or the
`source_parameter_identity_concordance` row).  Relabel copy: 3.1 (e).  The
other 16 callers pass no provenance and the helper posts
`Unsourced(HELPER_CALLER_UNROUTED)`; the latch lists them by enclosing
function as the worklist for steps 3 and 4.  The `source` free string stays
in the fact for now (it is the reason the audit cannot follow; the edge
now carries the causal link).

(d) Readers (`_resolved_source_value_class`, `_concorded_numeric_descriptor`,
...) unchanged: same page, same rows.

(e) 3.1 (e).

(f) `probe_annotated_scalar_parameter`: `k: int` gives one row with an
edge; the plain form gives none.

## 4. Ordered edit list (function by function)

E1  `graph_express2`: add `source_span_row`, `post_source_span` (section 1);
    `build_from_ast` sets `tree._turing_source_module` before pursuit and
    posts `contract_demand` rows for `retain`, `source_parameter_records`,
    `pursuit_roots`.
E2  `ProcessGraph.ensure_node`: post `ingestion_value` DERIVED(span);
    post `schema_node` when the node lies under a schema `AnnAssign`.
E3  `_mark_source_pursuit_active(definition, *, demand)`: new row key, one
    DERIVED post; update its three caller sites (`_expand_unresolved_ast_parents`
    seed loop, `requeue_definition` and its seven callers, `build_from_ast`
    selection).
E4  `build_from_ast`: move `_map_ir_from_ast(tree)` after
    `_expand_unresolved_ast_parents`; `_class_schema_from_ast`,
    `_state_machine_schema_from_ast`, `_map_ir_from_ast` post their pages;
    `ast_node_id` / `class_node_id` carry span row keys; `schema_node_ids` /
    `schema_roots` / `class_definitions` / `selected_class_identities` /
    `function_parameter_annotations` become views materialized from pages.
E5  `_resolve_class_body_field`, `_source_class_field_reference`,
    `ingest_external_class`: replace `page.set` with one DERIVED post each.
E6  `FunctionTable.declare(..., source=None)`: post `function_address`.
E7  reducer function-subgraph filter in `reduce_abstract_tensor_topology`:
    pass `source` to the two `declare` calls; replace the pre-normalize
    `value_kind` loop with `scalar_parameter` posts; delete the
    post-normalize `value_kind` loop.
E8  `_normalize_lexical_values.new_node`: post `ingestion_value`.
E9  `input_value`, `bind_target` (both branches), `bind_loop_target`,
    `record_ingestion_definition`, the four `reduce_statement`
    `identity_bindings` sites: post `name_binding`; `identity_bindings` /
    `ingestion_definitions` are deleted once the relabel reads the page.
E10 `first_class_function_node` and the `resolve_expression` callable site:
    CONCORD posts with provenance; unified scope; delete the column-1 write.
E11 `_concord_source_value_class(..., provenance=None)`; `input_value`
    passes cells; other callers get `Unsourced(HELPER_CALLER_UNROUTED)`.
E12 relabel tail of `_normalize_lexical_values`: `canonical_value` posts;
    `name_binding`, `lexical_read_binding`, `source_value_class_concordance`,
    `schema_node`, `scalar_parameter`, `callable_identity_concordance`
    canonical rows DERIVED(ingestion row); materialize `ssa_identity_tokens`,
    `identity_table`, `ingestion_identity_table`, `map_ir["schema_node_ids"]`
    / `["schema_roots"]` from the pages.
E13 `tests/test_process_graph_function_linking.py`: the `(0, 0)` history
    assertion on `callable_identity_concordance` becomes `(0,)`.
E14 new `tools/compiler_probes/probe_ingestion_roots.py` (section 7).

Writer sites routed: 27 (E1 3, E2 2, E3 1+3 callers, E4 8, E5 3, E6 1,
E7 4, E8 1, E9 8, E10 2, E11 1, E12 8 posts; counting each distinct
`set`/`concord`/dict write replaced).  Pages declared: 20 (14 new, 6 re-declared).

## 5. Risks

R1  **`identity_table` is mutated downstream** (3.2 (d)): `glsl_deployment_strategy`,
    `loop_composer`, `process_graph_autograd`, `aot_compile` write the dict.
    With the dict a materialized view, those writes diverge from the page
    silently.  Mitigation in step 2: none beyond the docstring and the
    per-page audit (`name_binding` rows vs dict entries differ = a finding
    `name-binding-view-drift`, cheap to add to `_function_findings`'s
    sibling set).  Fix is steps 4/5.  This is the top risk.
R2  **Pickling.**  `ProcessGraph.__getstate__` pickles `G.graph` whole:
    the materialized views are plain dicts/tuples/frozensets, unchanged.
    `PageMapping.__reduce__` returns a dict, so nothing here may hand a
    `PageMapping` into `G.graph` (E12 materializes real dicts, not
    mappings).  The book travels with `module.metadata["identity_book"]`,
    not with the graph: a graph unpickled into a new compile has canonical
    ids with no `canonical_value` rows in the new book (the same class of
    gap census 40, 4.3 records for `_mint_table_owner`).  AST facts on the
    three re-declared pages already pickle today (with their attached
    `_python_bindings`); step 2 does not add to that.
R3  **Readers keyed by dict identity.**  `first_class_function_nodes`,
    `loop_target_bindings_by_ast`, `static_reference_nodes` stay as
    closure memos (in-process indexes, not facts).  No reader holds a
    reference to `identity_bindings` or `ingestion_definitions` outside the
    closure; deleting them is safe.
R4  **AST objects as facts** (N3): `latest(row)` returning the node is
    relied on by pursuit and `build_from_ast`.  Converting the fact to a
    `Ref` needs a span -> node resolver on the book; deferred, listed.
R5  **`_map_ir_from_ast` moves after pursuit** (E4): `_expand_unresolved_ast_parents`
    appends discovered definitions to `module.body`, so the moved call sees
    more classes than today's.  Readers of `objects` (`build_class_navigation_table`,
    class-table build) already receive pursued classes through
    `ingest_external_class` / `selected_class_identities`; the observable
    difference must be measured on the audit `view` case before and after
    (row counts of `class_declaration`).  If the count changes, the move is
    reverted and E4 posts from a second pass instead.
R6  **Span path stability across `rewrite_body`**: pursuit rewrites
    definition bodies before reduction; posting after pursuit (1.5) makes
    paths refer to the final tree.  A post made before a rewrite would be
    stale; E2/E8/E9 all run after pursuit by construction.
R7  **Performance**: one `ingestion_value` row per AST node per function
    plus one `canonical_value` row per node.  `lexical_read_binding`
    already writes per read; the book is dicts.  Measure on the `controller`
    audit case (the largest seconds-long lowering) before merging.

## 6. What cannot be made DERIVED in step 2

- **Reducer-synthesized nodes with no `source`** (`new_node` callers that
  pass none): Phi arms, captured-parameter Inputs, static constants,
  materialized attribute nodes.  Their `ingestion_value` rows are
  `Unsourced(SYNTHESIZED_NO_SOURCE)` until step 3 gives Phis DERIVED(arms,
  test) and field state its page.  The `name_binding` rows for merges are
  DERIVED from the Phi's `ingestion_value` row and therefore reach a root
  only once step 3 lands.
- **`source_value_class_concordance` non-ingestion callers** (16): steps 3
  and 4 (reducer semantics and planner specialization).
- **`identity_table` mutations by planner / loop composer / autograd / aot**:
  steps 4 and 5.
- **Function addresses as MINTED identities**: addresses are dense counter
  values consumed as ints everywhere; making them `compose(serial, MINTED)`
  mints is a design decision for the user (design step 8 territory, with
  `_mint_table_owner`), not step 2.  Step 2 records them on a DERIVED row.
- **The `source` free string in `_concord_source_value_class` facts**: kept
  until every caller has an edge; then it is dropped (a fact-shape change).

## 7. Proof

Existing: `python -u tools/compiler_probes/probe_annotated_scalar_parameter.py`
(3.1, 3.2, 3.4, 3.5, 3.9), `python -u tools/compiler_probes/probe_struct_intake.py`
(unchanged pass), `python -u tools/audit_identity_concordance.py view`
(3.3, 3.6, 3.7) and `controller` (R7), `tests/test_process_graph_function_linking.py`
callable record-field case (3.8).

New, seconds-long: `tools/compiler_probes/probe_ingestion_roots.py` lowers,
with `runtime_closure_only=True` under the same contract as
`probe_annotated_scalar_parameter`, one source holding a dataclass with a
`bool` field, a function with an annotated scalar parameter that constructs
the local class, writes the field in a conditional and returns the record.
It then, reading only the book:
1. for the field: `class_field_declaration` row -> inbound edge ->
   `source_span` row whose kind is `AnnAssign`;
2. for the parameter: the canonical Input's `canonical_value` row ->
   `ingestion_value` row -> `source_span` row whose kind is `arg`; its
   `scalar_parameter` row -> `parameter_annotation` row -> the same span;
3. for every row on the 14 new pages: one inbound edge or one mint edge,
   else print the row and its stage;
4. prints the `unsourced-fact` worklist grouped by page and stage.

Expected after step 2: (3) prints nothing for stages `INGESTION`,
`CANONICAL_RELABEL`, `FUNCTION_SUBGRAPH`, `FUNCTION_TABLE`; the remaining
`unsourced-fact` entries are exactly the `SYNTHESIZED_NO_SOURCE` rows from
`new_node` (the Phi and any static constant in the probe source) and the
`HELPER_CALLER_UNROUTED` rows from `resolve_expression` -- the step 3
worklist.  The drop in `unsourced-fact` relative to step 1's baseline on
the same probe equals the count of raw writes removed by E5, E10, E11 (one
each), E12 (one per relabelled row on each of the six relabelled pages) and
the raw-primitive tags step 1 attaches to today's `class_page.set` /
`read_page.concord` copies; the probe prints both numbers so the difference
is measured, not asserted in advance.
