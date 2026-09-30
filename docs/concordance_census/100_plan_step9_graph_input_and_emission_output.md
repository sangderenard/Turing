# 100: Step 9 edit plan -- the graphs as views of the book, emission as its last layer

Read-only planning lane, 2026-09-30.  Everything below was read from the tree
with `git grep`/`sed`/`awk`; nothing was run.  Code is named by function,
never by line.  No compiler numberings appear.

Rests on `docs/CONCORDANCE_SINGLE_API_DESIGN_2026-09-30.md` (sections 2 and
6: the one call `IdentityBook.post(page, row, fact, *, stage, provenance,
mode)`, `Ref = (page, row, column)`, `Unresolved` for "looked and could not
decide", `Unsourced` behind the latch), the executed plans 60 (step 2) and 70
(step 3), whose format this file follows, and plans 80 (steps 4-5) and 90
(steps 6-8), which this step assumes and names where it depends on them
(section 6).

The question this step answers (user, 2026-09-30): the concordance is the
entire description of the compilation; its input is the IR graphs and its
output is a language pattern, and today neither end is drawn.  After step 9
a viewer opens a C token and diffuses backwards through the book to the
source span it came from, with no hop outside the book.

Files in scope.  Part A: `src/common/tensors/topological_reducer.py`
(`_set_operands`, `_operand_position_scope`, `new_node`, `_append_operand`,
`_replace_inputs`, `_remove_node`, `_redirect_value`),
`src/transmogrifier/graph/graph_express2.py` (`ensure_node` and the AST-walk
edge writer), `src/compiler/glsl_deployment_strategy.py`
(`_ordinary_conditional_control_programs`, `_dispatch_subgraph`, the
projection-leaf materializers, `_synthetic_device_scalar_shell`, the
`fix_aggregate_loop_bounds` rebuild), `src/compiler/loop_composer.py`
(`analyze_shader_loop_reductions` and its loop-block construction,
`rewire_continuation`, `add_port`, `add_constant`),
`src/compiler/control_source.py` (the block dataclasses, `ControlProgram`,
`order_control_region_dependencies`, `place_loop_carried_region_producers`,
`enrich_represented_conditionals`, `compose_region_code`,
`project_control_regions`, `overlay_scheduled_control`,
`place_validations_after_region_producers`), `src/compiler/fortran_c_shell.py`
(`_class_surface_ssa_program`'s block rewriters and
`_install_lexical_sequence_mutations`), `src/compiler/precompile_to_ssa.py`
(`_ControlSSABuilder._lower`, the control-function assembly).  Part B:
`src/compiler/ssa_c_backend.py`, `src/compiler/ssa_llvm_backend.py`,
`src/compiler/ssa_fortran_backend.py`, `src/compiler/fused_program_wasm_backend.py`,
`src/compiler/ssa_javascript_backend.py`, `src/compiler/identity_concordance.py`
(`identity_book`, the audit), `src/compiler/concordance_declarations.py`,
`tools/view_identity_concordance.py`, `tools/compiler_probes/probe_scalar_native_correctness.py`.

Vocabulary as in plan 70: "cell" = one `Ref`; "post" = one
`IdentityBook.post`; Page / Stage / Transform / Reason names in capitals are
registry OBJECTS declared in `concordance_declarations.py`, never strings.
"Node cell" = `node_identity_cell(graph, node_id)` (landed): the node's
`canonical_value` cell after the relabel, its `ingestion_value` cell before.

## 0. What is on the tree, observed

### 0.1 The api, as step 9 will use it

`IdentityBook.post` validates the row against `page.row_fields`
(`RowField.admits`: SCOPE any hashable, VALUE_ID int, NAME str, INDEX int,
LABEL any hashable, PAGE_REF a `Ref`), validates the fact with
`isinstance(fact, page.fact_type)` (an `Unresolved` always passes), refuses a
`Derived` with no cells and a `Ref` to a cell that does not exist
(`_source_stamp`), writes one edge row per source on the private edge page
plus the reverse index, ticks the clock once.  `edges_into(ref)` and
`edges_out_of(ref)` are one `scope_rows` read each; `mint_of(ref)` returns a
Novel row's (transform, operands).  `Registry.declare_page` refuses a second
declaration with a different shape.  These are the facts the emission pages
are shaped around; nothing in this plan needs a change to `post`.

### 0.2 Where the book lives when a backend runs

`lower_ast_source_to_ssa` (the wrapper in `fortran_c_shell.py`) calls
`begin_identity_book()`, runs `_lower_ast_source_to_ssa_impl`, and in its
`finally` calls `end_identity_book(token)` and `_dump_identity_book_log`
(the dense `render_identity_book` text under `artifacts/identity_logs/`).
`_lower_ast_source_to_ssa_impl` stores the same book as
`module_metadata["identity_book"]`.  `identity_book(module)` returns the
attached book first and only falls back to `current_identity_book()`, whose
docstring says why: after the `finally`, the ambient book is a fresh
detached one.

Every backend entry (`emit_ssa_module_to_c`, `emit_ssa_function_to_c`,
`emit_ssa_function_to_llvm`, `_emit_repository_call_module`,
`ssa_fortran_backend.emit_module`, `emit_ssa_module_to_javascript`) is
called by its harness AFTER `lower_ast_source_to_ssa` returned
(`native_package.piece_from_law`, `native_package` C packaging,
`probe_scalar_native_correctness.native_result`, `kernel_bank`,
`electrical_llvm`, `repository_ssa_dispatch`, ...).  None of the five
backend files imports `identity_concordance` or reads the book today
(`git grep identity_book src/compiler/ssa_*backend.py
fused_program_wasm_backend.py` finds nothing).  So at emission time:

- the book exists and is reachable through `identity_book(module)` for the
  C, LLVM, Fortran (when given an `IRModule`) and JavaScript lanes;
- `ssa_fortran_backend.emit_module` also accepts a bare
  `Mapping[str, Function]` -- no module, no book;
- `emit_wasm_module` receives a `FusedProgram`, never a module; its callers
  (`wasm_class_modules`, `site_bundle`, `machine_targets`, the
  `abstract_ui_*` modules) hold the module or the shell that owns it;
- the compile-end log has already been written, so emission rows posted on
  the attached book are NOT in that log unless something writes it again
  (risk R9.6);
- `host_code_modules` pickles compiler IR with `_HostSSACachePickler`;
  whether `metadata["identity_book"]` survives or is dropped is not read
  here (risk R9.7).

### 0.3 Process-graph structure: three copies of every edge, many writers

An operand edge exists in three places: `data["parents"]` (a list of
`(parent id, role)`), `data["children"]` on the parent, and the networkx
edge `graph.G.add_edge(parent, node, role=role)`.

`_set_operands` is documented as "the one writer of a node's operand list"
and writes ONLY `data["parents"]`; it records moves, retires and forks on
`identity_transition` (rows `(scope, consumer, role, ordinal)`, raw
`revise`), returns silently when `_operand_position_scope(graph)` is None
(no `operand_position_scope` and no `lexical_read_scope`), and records
nothing for an APPEND.  Plan 70 section 3 specified the Append post and the
`Unsourced(NO_OPERAND_POSITION_SCOPE)` tag; neither has landed
(`_set_operands`' docstring says the revise writes "stay raw until"
`identity_transition` is declared; plan 80 B1.3 declares it in step 5).

Writers of `children` and of the networkx edge beside `_set_operands`, by
function (each is a place where the three copies can disagree):

| file | writer | what it writes by hand |
|---|---|---|
| graph_express2 | `ensure_node` | `parents=[]`, `children=[]` on the new node |
| graph_express2 | the AST-walk edge writer in `build_from_ast` (`self.G.add_edge(src_id, tgt_id, extra=set())`, then `children` / `parents` lists) | every authored operand edge, before any read scope exists |
| topological_reducer | `new_node` | `add_edge(parent, node, role)` and `children.append` for the `parents=` it is given, BEFORE it posts the node's `ingestion_value` row |
| topological_reducer | `_append_operand`, `_replace_inputs`, `_remove_node`, `_redirect_value`, the function-subgraph filter, the `Expr`/`Return` dissolve sites, the class-table member materializer | call `_set_operands` for `parents`, then rewrite `children` / `add_edge` themselves |
| glsl_deployment_strategy | `_alias_projection_to_member` region, the projection-leaf materializers (`graph.G.add_edge(parent, member, role)`, `children.append((node_id, "elts"))`), the `base`/`index` member builders, `_dispatch_subgraph` (`subgraph.G.add_edge(output_id, store_id)`), `_synthetic_device_scalar_shell` (`input_data["parents"] = []`), the bound-receiver site | same split: `_set_operands` for parents, hand-written children and edges |
| loop_composer | `add_constant` / `add_port` region (`children = []`, `add_edge`, `data["parents"] = normalized`), `rewire_continuation` (`data["parents"] = rewritten`, no `_set_operands`; plan 80 A2.6 routes it), the node clone (`cloned["parents"] = list(parents)`, `cloned["children"] = []`), the materializer and consumer `parents` rewrites | direct list writes |
| fortran_c_shell | the optional-presence graph rewrite (`graph.nodes[parent]["children"] = ...`, `graph.add_edge(new_id, child, **edge)`, `graph.add_edge(positive, compare_id, role="operand")`) | `_set_operands` for parents, hand-written children and edges |

Position-keyed pages that already describe operand structure as facts:
`lexical_read_binding` (the authored binding name an operand position read;
rows `(scope, consumer, role, ordinal)`; declared by step 5 B1.3),
`consumer_operand` (`(scope, consumer, operand) -> positions`, planner),
`item_operand`, `call_argument_operand` (step 4 A1.2 re-declares them
DERIVED).  None of them is keyed by the EDGE as a relation of two node
cells; the Append row of plan 70 section 3 -- fact `Append(operand cell)`
DERIVED(operand cell, consumer cell) at row `(scope, consumer, role,
ordinal)` -- is exactly that row, and it is the row this step completes.

### 0.4 Control graph: frozen blocks rebuilt by fifteen rewriters, no row

`control_source.py` declares the block dataclasses (`StatementBlock`,
`SequenceBlock`, `ConditionalBlock`, `LoopBlock`, `WhileBlock`,
`LoopControlBlock`, `StateMachineTick`, `ParallelDeployment`, `CallBlock`,
`DispatchBlock`, `ResourceScopeBlock`, `ExternalReferenceCallBlock`,
`ValidationBlock`, `SequenceMutationBlock`, `SequenceQueryBlock`,
`ScalarFieldWriteBlock`, `StreamPublishBlock`) and `ControlProgram(root,
region_indices, uniforms, value_aliases, ..., recursion_regions,
deployment_regions, ..., specialized_conditional_node_ids, anchor_region)`.
`_ControlSSABuilder._lower` dispatches on all seventeen.

Identity the blocks already carry (observed fields): `ConditionalBlock.
source_node_id`, `carried_aliases` (ids), `carried_field_cells` (Refs, lane
C), `predicate_value_id`, `predicate_expression.read` (a
`lexical_read_binding` row key), `body_callsite_ids` / `orelse_callsite_ids`,
`predicate_region_indices`; `LoopBlock` / `WhileBlock`: `source_loop_node_id`,
`recursion_region_id`, `control_site_ids`, `result_ports`, `carried_aliases`,
`carried_seeds`; `LoopControlBlock`: `site_node_id`, `site_values`,
`return_value_ids`; `CallBlock`, `DispatchBlock`, `ExternalReferenceCallBlock`:
`callsite_id`; `ScalarFieldWriteBlock`: `effect_node_id`, `field_state_cell`
(a Ref, lane C); `SequenceQueryBlock`: `result_value_id`, `source_call_node_id`,
`producer_loop_node_id`; `ResourceScopeBlock`: `source_scope_id`;
`ValidationBlock`: `extraction_identity`; `StatementBlock`: the marker
`__scheduled_region_N__` (a region ordinal) or `__plan_callsite_N__` (a
callsite).  `SequenceBlock` carries nothing: it is a container.

Builders and rewriters (each returns a NEW tree; nothing records that the
new tree is the old one moved):

| function | builds / rewrites |
|---|---|
| `glsl_deployment_strategy._ordinary_conditional_control_programs` | one `ControlProgram` per authored `if` / `IfExp`: `SequenceBlock` of region markers + `ConditionalBlock(predicate, body, orelse, carried_aliases, carried_field_cells, source_node_id, ...)` + `LoopControlBlock` return controls; `anchor_region` |
| `loop_composer.analyze_shader_loop_reductions` (its loop-block construction, `planned_root`) | `LoopBlock` / `WhileBlock` with `carried_aliases`, `result_ports`, `carried_seeds`, `terminal_controls`, `control_site_ids`, `source_loop_node_id`; a `ControlProgram` per loop |
| `glsl` hierarchical `fix_aggregate_loop_bounds` | rebuilds `LoopBlock` / `WhileBlock` / `StateMachineTick` / `ParallelDeployment` / `CallBlock` with one field changed |
| `control_source.order_control_region_dependencies`, `place_loop_carried_region_producers`, `enrich_represented_conditionals`, `overlay_scheduled_control`, `project_control_regions`, `compose_region_code`, `place_validations_after_region_producers` | insert markers, drop markers of resolved regions, substitute region bodies (`RegionCode`), collapse empty constructs, wrap prelude |
| `fortran_c_shell._class_surface_ssa_program` inner rewriters: `strip`, `insert_ordered` (four copies), `insert_in_conditional` (four copies), `insert_into_loop` (three copies), `relocate`, `wrap_scope`, `insert_after_producer`, `insert_at_query_region`, `remove_replaced_regions`, `rewrite`, and `_install_lexical_sequence_mutations` | add `ScalarFieldWriteBlock`, `SequenceQueryBlock`, `SequenceMutationBlock`, `DispatchBlock` / `ExternalReferenceCallBlock`, `ResourceScopeBlock`, `ValidationBlock`, one synthesized `ConditionalBlock` (the `synthesized = ConditionalBlock(...)` site); reorder by authored position |
| `precompile_to_ssa` control-function assembly | `SequenceBlock((*queries, *body_blocks))` for helper control functions |

Step 4 (plan 80) puts the REGIONS and LOOP RECORDS on the book
(`deployment_region`, `deployment_region_member`, `loop_carried_binding`,
`loop_region_membership`, `loop_result_port_binding`, `call_binding`,
`source_control_specialization_concordance`); step 3 put the field-carried
merge there (`control_carried_field`); step 5 puts the LOWERING of a block
there (`carried_snapshot`, `region_signature`, `control_value_binding`,
`function_output`).  No step puts the BLOCK there: nothing says "a
ConditionalBlock for this `if` exists, owns these two arms and these
regions, and sits at this position of this function's program".  That is
what section 2 declares.

### 0.5 Node payloads

`ProcessGraph.ensure_node` stores on every authored node: `label`, `type`,
`op`, `expr_obj` (the AST node itself), `source_span` (positions dict),
`source_scope`, `source_class`, `extra_args`, `attributes` (extraction
receipts), `constant`, `extraction_contract`, `domain_node`, `store_id`,
`parents`, `children`.  `new_node` stores `label`, `type`, `op`,
`expr_obj=None`, `extra_args={}`, `domain_node=None`, `store_id=None`,
`parents`, `children`, `attributes`, optional `source_span`, `constant`.
Attribute keys written by the reducer and the planner (counted with `git
grep` over `attributes["..."] =`) are classified in section 3.

### 0.6 Emission: where a unit of text is spelled from an SSA value

Common seam: `SSAValue.id` (`SSAValue.name()` spells `%t{id}`),
`Instr.op` / `args` / `res` / `attributes` / `source_span`,
`Function.name` / `args` / `blocks` / `metadata`.  Every backend spells its
units from these and nothing else; the function is identified by
`Function.name` everywhere.

| backend | entry | unit spelled | where | artifact and its build |
|---|---|---|---|---|
| C scalar lane | `emit_ssa_function_to_c` (one `entry` block) | `Const` -> hex literal inlined (no line); every other result -> `lines.append("const double t{id} = {rendered};")`; `Ret` -> `out[i] = ...` stores; header `TURING_EXPORT void {name}(const double *in, double *out)`; `_turing_sin_table` | the `for instruction in function.blocks["entry"].instrs` loop | `CFunctionArtifact(name, source, input_names, output_names, shortfalls, publications, surface plan)`; `CFunctionArtifact.compile` |
| C module lane | `emit_ssa_module_to_c` | per reachable `fn`: prototype + definition `static {ret} {_c_symbol(fn)}(params)`, formals `v{id}`, results `t{id}`, block labels `_c_label(block)`, one `body.append` per instruction in the `op` dispatch (Phi hoisted, Const `t{id} = literal`, Call `_c_symbol(callee)(...)`, outlined region `{_c_symbol(fn)}_r{region}`), `phi_edge_assignments`, `output_publications`; entry wrapper `TURING_EXPORT void {name}(void **buffers, long long *extents)`; `source = "\n".join(...)` | the per-block `for position, instruction in enumerate(block.instrs)` loop; `emission_context["block"/"instruction"]` already names the unit being spelled | `CModuleArtifact(name, source, buffer_order, buffer_dtypes, shortfalls, buffer_shapes, extent_order, precision_sections, pool_required, pooled_regions, linked_llvm, linked_libraries)`; `compile(directory, optimization, link)` writes `{name}.c` and each piece `{symbol}.ll`, runs `python -m ziglang cc -shared ... -o {name}.dll`, sets `library_path`; `compile_standalone` |
| LLVM scalar lane | `emit_ssa_function_to_llvm` (single block, closure of one) | `scalars[value id] = (rendering, type)`, `lines`, `globals_out`, `buffer(value_id)` loads, `define void @{name}(ptr %buffers, ptr %extents)` | the instruction loop | `LLVMFunctionArtifact(name, llvm_ir, buffer_order, buffer_shapes, extent_order, shortfalls, buffer_dtypes, needs_text_sink, output_publications, output_surfaces, watched, watch_shortfalls)`; `compile_artifact` writes `{name}.ll`, zig cc -> `{name}.dll` |
| LLVM module lane | `_emit_repository_call_module` | per reachable function `emitted_functions.append("define internal void @{internal_symbols[name]}(...) {...}")` with `pointer`, `load_as`, `literal`, `emit_return_values`, `capture_block_history`; kernel `definitions[symbol]` / `declarations[symbol]` from `extract_llvm_function`; wrapper `define void @{entry_name}`; `llvm_ir = "\n\n".join(...)`; `_annotate_noalias` | the per-function body loop | same artifact; `with_native_sgd_loop` / `with_native_adam_loop` wrap it and mint values with a local `fresh_id()` (values no SSA function owns) |
| Fortran | `emit_module` -> `emit_subroutines` -> `emit_function` -> `_FunctionEmitter.emit` | `_name(value) -> "t{id}"`; `_emit_block` appends `! block {name}`, labels `{n} continue`, `goto`, `if (...) then`, `_statements(instr)`, or inlines `_expression(instr)` into its consumer when `_may_inline` (a value spelled with NO statement of its own) | `_emit_block` | `FortranSubroutine(name, source, shortfalls, extent_names, ..., argument_dtypes, output_dtypes)` -> `FortranModule(name, source, subroutines, api, precision_sections)`; `write` -> `{name}.f90` + `{name}.api.yaml`; `compile_module` -> `{name}.dll` / `.so` |
| WebAssembly | `emit_wasm_module` (and `_emit_pure_matmul_module`, the container store / load / name-hash modules) | input is a `FusedProgram` (`feeds`, `steps: OpStep(step_id, op_name, input_ids, result_id, attrs)`, `outputs`, `meta`, `extras`), NOT SSA; `evaluate_steps` appends WAT lines and `local.set {names[step.result_id]}` per live step; `_assemble` mirrors the same steps into the binary through `CodeBuilder` | `evaluate_steps` / `emit_step`; `_assemble.emit_step` | `WasmModule(name, source, shortfalls, parameters, value_type, api, binary)`; `write` -> `.wat`, `.wasm`, `.api.yaml`; `compile_wat` (wat2wasm) |
| JavaScript | `emit_ssa_module_to_javascript` | per reachable function `function_symbol = "impl_" + _symbol(label)`, `definitions.append(...)`; values are spelled by `_source_names(function)` -- the authored name from `metadata["argument_names" / "parameter_names" / "value_names"]` when one exists -- else by id; class wrappers `class_symbol = _symbol(definition.identity)` | the per-function definition loop | `JavaScriptModuleArtifact(name, source, entry, buffer_order, pointer_formals, shortfalls, api)`; no compile step |

Artifact-level records that already exist beside the text:
`function_output_publications` / `publication_surface_plan`
(`output_publication.py`, read into every artifact), `CompiledProgramAPI`
(`api` on Fortran / WASM / JS artifacts, written as `.api.yaml`),
`buffer_order` / `extent_order` (C, LLVM, JS).  None is on the book.

### 0.7 The viewer

`tools/view_identity_concordance.py` `extract_graph` makes one node per
`(page name, row)` of every non-private page, one id node per integer atom
found in a row or its latest fact (`_atoms`), and takes causal edges from
`_api_causal`: `book.edges_into(ref)` per cell (DERIVED, tagged with the
stage), `book.mint_of(ref)` (MINT, tagged with the transform),
`book.unsourced_rows()` (UNSOURCED nodes), `book.latch`.  `diffuse_heat` /
`diffuse_effects` diffuse from seed nodes along `causal_edges`;
`resolve_focus(graph, spec)` picks seeds.  A new page's rows and edges are
drawn without a viewer change; the token-to-span walk of section 5.4 is
the one addition.

## 1. Part A (1): every operand edge is a row; `G.edges` is a view

### 1.1 Pages

`IDENTITY_TRANSITION` as plan 80 B1.3 declares it (row `(scope SCOPE,
consumer LABEL, role LABEL, ordinal INDEX)`, fact `OperandTransition = Move
| Retire | Fork | Append`, mode REVISE).  Step 9 declares nothing new here;
it completes plan 70 section 3 and adds one rule and one reader:

- **The edge IS the row's latest fact.**  For every `(scope, consumer, role,
  ordinal)` whose latest fact is `Append(operand cell)`, `Move(...)` into
  that position or `Fork(...)`, the operand list of `consumer` holds that
  operand at `(role, ordinal)`; a latest `Retire()` means no operand there.
  `data["parents"]`, `data["children"]` and the networkx edge are three
  materializations of that fact, and `_set_operands` writes all three.
- **`_operand_position_scope` gains the ingestion scope.**  Today it reads
  `operand_position_scope` then `lexical_read_scope`; step 9 adds
  `ingestion_value_scope` (the scope `ensure_node` posts under) as the third
  fallback, so an edge written by `build_from_ast` before any reduction has
  a scope and posts.  The relabel's `same=mapping` path already moves the
  rows to the canonical scope (plan 70 section 3, "the S10 morph graph
  edge"); the ingestion-scope rows are the ones it moves.

### 1.2 Posts, writer by writer

| writer | today | post (all on `IDENTITY_TRANSITION`, stage = the caller's, mode REVISE) |
|---|---|---|
| `_set_operands`, new position with no move source and no fork | records nothing | `Append(operand cell)` DERIVED(operand node cell, consumer node cell) -- plan 70 section 3 verbatim |
| `_set_operands`, scope None | returns after writing `parents` | `Unsourced(NO_OPERAND_POSITION_SCOPE)` on the unsourced page, one per rewrite (plan 70), which after 1.1's third fallback happens only for a graph that was neither built by `build_from_ast` nor entered normalization -- the audit lists it |
| `_set_operands`, every call | writes `data["parents"]` only | ALSO: for every parent that left the list, remove `(consumer, role)` from that parent's `children` and `graph.G.remove_edge` when no other position of `consumer` still names it; for every parent that arrived, append to `children` and `add_edge(parent, consumer, role=role)`.  One writer for three copies.  The callers listed in 0.3 that rewrite `children` / `add_edge` by hand are edited to stop (they call `_set_operands` already) |
| graph_express2 AST-walk edge writer in `build_from_ast` | `self.G.add_edge(src, tgt, extra=set())`, list writes | `_set_operands(self, tgt_id, parents_with_roles, cause=INGEST_EDGE)` once per node after its parents are known (the `extra=set()` payload stays a networkx edge attribute set by `_set_operands` from a new keyword `edge_payload`).  Stage `INGESTION`.  Scope = `ingestion_value_scope` via 1.1 |
| `new_node` | `add_edge` + `children.append` per parent, then posts `ingestion_value` | reorder: `add_node` -> post `ingestion_value` (the consumer cell must exist before the Append derives from it) -> `_set_operands(graph, node_id, parents, cause=<the caller's transform>)`.  `new_node` gains `cause: Transform` with the caller's transform; its ~40 callers in `_normalize_lexical_values` pass the transform they already name for the node's NOVEL row where plan 70 gave them one, else `REDUCER_SYNTHESIS` |
| `_append_operand`, `_replace_inputs`, `_remove_node`, `_redirect_value`, function-subgraph filter, dissolve sites, member materializer | `_set_operands` + hand-written children / edges | delete the hand-written half; `cause` becomes a `Transform` (plan 70 section 3's list) |
| glsl projection-leaf materializers, `base` / `index` member builders, `_dispatch_subgraph` store edge, `_synthetic_device_scalar_shell`, bound-receiver site | hand-written children / edges (some also `_set_operands`) | through `_set_operands` with causes `PROJECTION_TO_LEAF`, `AGGREGATE_MEMBER`, `DISPATCH_STORE` (plan 80's transform), `SYNTHETIC_DEVICE_SCALAR_PREDICATE`, `BOUND_RECEIVER` |
| loop_composer `add_constant` / `add_port` region, node clone, materializer and consumer `parents` rewrites | direct list writes | `_set_operands` with `LOOP_COMPOSER_CONSTANT`, `LOOP_RESULT_PORT`, `LOOP_BODY_CLONE`, `LOOP_MATERIALIZER`; `rewire_continuation` per plan 80 A2.6 (`LOOP_CONTINUATION_REWIRE`) |
| fortran_c_shell optional-presence rewrite | `_set_operands` + hand-written children / edges (`cause="optional_presence_detach"` etc. as strings) | delete the hand-written half; causes become the transforms `OPTIONAL_PRESENCE_DETACH`, `OPTIONAL_PRESENCE_INPUT`, `OPTIONAL_PRESENCE_COMPARE` |

The Append's operand cell for a reducer node is `node_identity_cell(graph,
parent)`; for an authored node at ingestion it is the `ingestion_value` cell
`ensure_node` posted (`_post_ingestion_value`), which exists because
`build_from_ast` posts nodes as it visits them and edges after.

### 1.3 Readers and the view

`G.edges`, `data["parents"]`, `data["children"]` keep every reader (several
hundred sites) unchanged: they are the materialized view.  One new reader:
audit finding `operand-view-drift` (`CorrelationTable._operand_view_drift_findings`):
for every graph the audit case reaches (the root graph and each dispatch
subgraph via `deployment_nodes`), for every node, the tuple of `(role,
ordinal, parent)` from `data["parents"]` must equal the tuple of positions
whose latest `IDENTITY_TRANSITION` fact under `_operand_position_scope(graph)`
is not `Retire()`, with the Append / Move / Fork operand cell resolving to
`parent`; and `children` / `G.edges` must agree with `parents`.  A
disagreement names the node, the position and both records.  This is the
proof that the graph is a view and not a fourth copy.

### 1.4 What step 5 covers and what remains after 1.2

Covered by step 5 F3 (plan 80): declaring `IDENTITY_TRANSITION`, the Append
post itself, `cause: Transform`, `fork_read_scope`'s `SCOPE_ORIGIN`.  Step 9
adds: the ingestion-scope fallback (1.1), the ingestion edge writer and
`new_node` reorder, `_set_operands` owning `children` and the networkx edge,
the twenty-odd hand-written edge writers of 0.3, the drift finding.  If step
5 has not landed when step 9 starts, F3's Append post is done here first
(it is one `post` call in `_set_operands`).

## 2. Part A (2): every control block is a row; the ControlProgram is a view

### 2.1 Pages

```
Page CONTROL_BLOCK
  row_fields = (RowField(function_scope, SCOPE),      # the graph's lexical_read_scope (step 6's FUNCTION_SCOPE row when there is none)
                RowField(kind, LABEL),                # ControlBlockKind (one member per block class except SequenceBlock)
                RowField(owner, PAGE_REF))            # the cell of the construct the block represents (2.2)
  fact_type = ControlBlockFact                        # frozen: the block's identity fields (below), never the tree
  mode = CONCORD                                      # a block exists once; its fields are facts about its owner
```

`ControlBlockFact(kind, predicate: Ref | None, carried: tuple[Ref, ...],
sites: tuple[Ref, ...], regions: tuple[Ref, ...], callsite: Ref | None,
extra: tuple)` holds the block's identity-bearing fields as CELLS -- the
predicate value's node cell, `carried_aliases` as `name_binding` /
`control_carried_field` cells, `control_site_ids` as node cells,
`predicate_region_indices` / marker regions as `deployment_region` cells,
`callsite_id` as the `call_binding` cell -- and `extra` the non-identity
payload (`expect_true`, `comparison`, `schedule_preference`, `dtype`, the
`induction` name, `error_code`, ...).

```
Page CONTROL_BLOCK_PLACEMENT
  row_fields = (RowField(function_scope, SCOPE), RowField(block, PAGE_REF))   # the CONTROL_BLOCK cell
  fact_type = Placement                              # (parent: Ref | ROOT, arm: Arm, ordinal: int) or Unresolved
  mode = REVISE                                      # every rewriter that moves the block posts the next placement with its cause
```

`Arm` is a small enum: `ROOT_SEQUENCE`, `BODY`, `ORELSE`, `CONDITION`,
`CALLEE`, `CLEANUP`, `CASE(n)`, `DEFAULT`, `LANE(n)`, `TERMINAL`.  The
ordinal is the block's index among its siblings in that arm after
`SequenceBlock` flattening (a `SequenceBlock` is not a block; nesting of
sequences is flattened for the ordinal, which is what every rewriter's
`insert_ordered` already does).

```
Page CONTROL_PROGRAM
  row_fields = (RowField(function_scope, SCOPE), RowField(program, LABEL))   # SHELL, or the conditional / loop construct's cell for a per-construct program
  fact_type = ControlProgramFact                     # (region_indices as deployment_region cells, uniforms, value_aliases as cells, anchor_region cell, specialized conditionals as cells)
  mode = REVISE                                      # the projected / composed program is a revision of the planned one
```

### 2.2 The owner cell of each block kind

| block | owner cell | when there is none |
|---|---|---|
| `ConditionalBlock` | node cell of `source_node_id` (the `ast.If` / `IfExp` node) | the synthesized conditional in `_class_surface_ssa_program`: NOVEL row, `Transform.SYNTHESIZED_CONTROL` arity 1, operand = the mutation's node cell; the owner is then that NOVEL row's cell |
| `LoopBlock`, `WhileBlock` | node cell of `source_loop_node_id` | a loop with none (an unrolled or planner-made loop): NOVEL `SYNTHESIZED_CONTROL` from the `recursion_region_id`'s `RecursionRegion` -- itself a row on `CONTROL_PROGRAM.recursion_regions` -- else `Unsourced(CONTROL_OWNER_UNKNOWN)` |
| `LoopControlBlock` | node cell of `site_node_id`; for `action == "return"` the return construct cell (`return_site_cell` attribute, step 3) | `Unsourced(CONTROL_OWNER_UNKNOWN)` |
| `StatementBlock` region marker | the `deployment_region` cell (step 4) for the ordinal | before step 4: the region ordinal as a LABEL in `extra` and `Unsourced(REGION_CELL_UNROUTED)` |
| `StatementBlock` `__plan_callsite_N__` | the `call_binding` cell (step 4) | as above |
| `CallBlock`, `DispatchBlock`, `ExternalReferenceCallBlock` | `call_binding` cell of `callsite_id` | as above |
| `ScalarFieldWriteBlock` | `field_state_cell` (the WRITTEN revision, lane C) | the SetAttr node cell of `effect_node_id` |
| `SequenceQueryBlock` | node cell of `result_value_id` (the query's own node) | `source_call_node_id`'s cell |
| `SequenceMutationBlock` | node cell of `mutation.effect_node_id` | -- |
| `ResourceScopeBlock` | node cell of `source_scope_id` | -- |
| `ValidationBlock` | node cell of `predicate_value_id` | -- |
| `StreamPublishBlock` | node cell of `value_id` | -- |
| `StateMachineTick`, `ParallelDeployment` | the `CONTROL_PROGRAM` row's own cell (they are program-level structure) | -- |

Every owner is a cell that steps 2-4 already post; a block whose owner
cannot be named is recorded `Unsourced` under the latch and listed, which is
the worklist.

### 2.3 Posts, writer by writer

One helper does the work so that fifteen rewriters make one call each:

```
post_control_program(function_scope, program, *, stage, cause: tuple[Ref, ...], label=SHELL) -> None
```

walks `program.root`, flattening `SequenceBlock`s; for each block posts
`CONTROL_BLOCK` CONCORD DERIVED(owner cell, the cells named in its
`ControlBlockFact`) -- a block already on the book is the design's "same
fact, records its edge" no-op -- then posts `CONTROL_BLOCK_PLACEMENT` REVISE
DERIVED(the block's `CONTROL_BLOCK` cell, the parent block's cell or the
`CONTROL_PROGRAM` cell for the root, `*cause`) ONLY when the placement
differs from the row's latest fact; then posts `CONTROL_PROGRAM` REVISE
DERIVED(every root block cell, `*cause`) when its fact changed.  A rewriter
that changes nothing posts nothing (REVISE would refuse a same-source
re-post; the helper compares first, as plan 80 R5 requires).

| writer | stage | cause cells passed |
|---|---|---|
| `_ordinary_conditional_control_programs` | `CONTROL_PROGRAM_BUILD` | the conditional's node cell |
| `analyze_shader_loop_reductions` loop-block construction | `CONTROL_PROGRAM_BUILD` | the loop's node cell, its `loop_carried_binding` cells |
| `fix_aggregate_loop_bounds` (glsl hierarchical) | `CONTROL_PROGRAM_REWRITE` | the aggregate binding's `planner_tensor_descriptor` cell (step 4) |
| `order_control_region_dependencies`, `place_loop_carried_region_producers`, `enrich_represented_conditionals`, `overlay_scheduled_control`, `place_validations_after_region_producers` | `CONTROL_PROGRAM_REWRITE` | the `deployment_region` cells inserted or moved; for `place_loop_carried_region_producers` the `loop_region_membership` cell; for the marker-region fallback plan 80's `Unresolved(REGION_OWNERSHIP_UNKNOWN)` cell |
| `project_control_regions` | `CONTROL_PROGRAM_REWRITE` | the retained `deployment_region` cells; a block that collapses posts `Placement` = `Unresolved(COLLAPSED_EMPTY_CONSTRUCT, read=(its previous placement cell,))` -- a removal is an edge, never an erasure |
| `compose_region_code` | `CONTROL_PROGRAM_REWRITE` | each `RegionCode`'s `deployment_region` cell; a substituted interior's blocks are posted with the marker block's cell as parent cause |
| `_class_surface_ssa_program` rewriters (`strip`, the `insert_*` families, `relocate`, `wrap_scope`, `insert_after_producer`, `insert_at_query_region`, `remove_replaced_regions`, `rewrite`) and `_install_lexical_sequence_mutations` | `CONTROL_PROGRAM_REWRITE` (stage object per family is not needed: the cause cells say what moved) | the inserted block's owner cell (the `field_state_cell`, the query node cell, the mutation node cell, the callsite's `call_binding` cell, the scope node cell) |
| `precompile_to_ssa` control-function assembly | `CONTROL_PROGRAM_BUILD` | the helper function's `FUNCTION_SCOPE` cell (step 6) |
| `_ControlSSABuilder._lower` | reads | when lowering a block, `self._block_cell = latest_ref(CONTROL_BLOCK, ...)`; step 5's posts that name "the conditional construct's identity cell" (`carried_snapshot`, `SSA_BLOCK` below) derive from the `CONTROL_BLOCK` cell instead, so the lowering points at the block row, and the block row points at the construct |

### 2.4 The view

`control_program_view(function_scope, label=SHELL) -> ControlProgram`
rebuilds the tree from `CONTROL_BLOCK_PLACEMENT.scope_rows(function_scope)`
(latest per block, skipping `Unresolved`) and the `CONTROL_BLOCK` facts,
re-materializing each dataclass with its identity fields resolved from
cells back to ids (`row_value_id`) and `extra` restored.  It is used by the
audit finding `control-view-drift`: the tree `_class_surface_ssa_program`
hands to `lower_control_sections_to_ssa` must equal the view (dataclass
equality, which the frozen blocks already define), else the finding names
the first differing block and its two placements.  Until every rewriter
posts, the finding lists the drift; when it is zero the passed tree can be
replaced by the view and the rewriters return nothing but their cause
cells.  Step 9 does not make that replacement; it makes it possible and
measures the distance.

### 2.5 What steps 4-5 cover and what remains

Step 4 covers region and loop-record identity (the owner cells of markers,
loops, ports); step 5 covers what the builder DOES with a block.  Remaining
and done here: the block rows, the placement history, the program rows,
`ReturnBlock`'s missing cell (plan 90 N3: the `LoopControlBlock` with
`action == "return"` carries `return_site_cell` in its `ControlBlockFact`),
and `SSA_BLOCK` (2.6).

### 2.6 One small page the emission layer needs from the control layer

Backends spell basic-block labels (`_c_label(block_name)`, Fortran
`{n} continue`, LLVM labels).  No page names a basic block.

```
Page SSA_BLOCK
  row_fields = (RowField(function_scope, SCOPE), RowField(block, NAME))
  fact_type = BlockFact(successors: tuple[str, ...])
  mode = CONCORD
```

Posted by `_ControlSSABuilder` where it creates a block (its block-naming
helper, in `lower_conditional`, `lower_loop`, `lower_while`, the
loop-control edges, `finish`'s exit block) DERIVED(the `CONTROL_BLOCK` cell
being lowered; for `entry` and the exit block the `CONTROL_PROGRAM` cell).
This is a step-5 writer; if step 5 has landed without it, step 9 adds it
(one post at the block-creation helper).  Until then emission's
`BLOCK_LABEL` units derive from the `FUNCTION_SCOPE` cell alone and the
latch lists `Unsourced(BLOCK_ORIGIN_UNROUTED)` per label.

## 3. Part A (3): node payload versus identity fact

Rule: an attribute is an IDENTITY FACT when its value names another node,
value, cell, scope, region or a decision about the node's identity (which
version, which merge, which callee, which region, which class); it belongs
on a page keyed or sourced by the node cell, and the attribute survives as
a cache that the audit checks (plan 90 1.1: "neither is the key; both are
caches").  It is PAYLOAD when it describes the node's own content and
nothing else identifies through it; it stays on the node.

| attribute (writer) | class | page (existing / step) |
|---|---|---|
| `expr_obj`, `label`, `type`, `op`, `constant`, `extra_args`, `domain_node`, `store_id` | payload | none; `source_span`'s positions are already `SpanFact` (step 2), the dict on the node is its view |
| `source_scope`, `source_class` | identity | `source_span` row key (module, qualname) / `class_declaration` (step 2) |
| `extraction_contract`, `extraction_action`, `extraction_rule`, `extraction_identity`, `extraction_classification`, `extraction_occurrences` (`ensure_node`) | identity (a contract decision about this node) | `contract_demand` (step 2) has kinds RETAIN / PARAMETER_RECORD / PURSUIT_ROOT only; step 9 declares `EXTRACTION_RECEIPT` row `(ingestion scope, node VALUE_ID)` fact = the receipt, DERIVED(node cell, the `contract_demand` row of the rule) -- the receipt is the one node-level contract fact with no page |
| `parents`, `children` | identity | `IDENTITY_TRANSITION` view (section 1) |
| `value_id`, `identity_cell`, `field_state_cell`, `field_state_arms`, `return_site_cell` | cache of a cell | `canonical_value`, `reducer_field_state`, `return_site_slot` (steps 2-3) |
| `value_kind`, `binding_name` | identity | `scalar_parameter`, `name_binding` (step 2) |
| `initial_value_id`, `source_conditional_id`, `record_field_state`, `conditional_member_bindings` (reducer Phi) | identity | `reducer_field_state` MERGED (step 3), `control_carried_field` (plan 70 section 4); name-carried Phis: step 5's `PHI_CONDITIONAL` operands.  The `source_conditional_id` join is the `CONTROL_BLOCK` owner cell after section 2 |
| `loop_carried_bindings`, `loop_target_bindings`, `loop_target_initials`, `loop_state_effects`, `loop_iteration_outputs`, `loop_carried_updated_ids`, `loop_result_ports`, `loop_ports_materialized`, `loop_aggregate_axis` | identity | `loop_carried_binding`, `loop_result_port_binding` (step 4 A1.2) |
| `loop_break_sites`, `loop_break_bindings`, `terminal_return_values` | identity | NO page in plans 60-90.  Step 9 declares `LOOP_CONTROL_SITE` row `(read scope, loop VALUE_ID, site VALUE_ID)` fact `LoopSiteFact(action, bindings as name_binding cells)` DERIVED(loop cell, site node cell, each binding cell), stage `REDUCTION`; `LoopControlBlock.site_values` is its view |
| `callee_ref`, `callee_resolution`, `method_ref`, `constructor_ref`, `function_ref`, `class_ref`, `result_class_ref`, `receiver_class_ref`, `method_resolution`, `operator_reference_node`, `static_python_reference`, `static_call_arguments` | identity | `call_binding` (step 4), `callable_identity_concordance` / `function_address` (step 2), `source_value_class_concordance` (step 2 / 3.9); `static_call_arguments` -> `planner_specialization` (step 4) |
| `source_type`, `precision_limbs`, `precision_element`, `python_precision_boundary`, `python_precision_operator`, `precision_pack_source_id`, `precision_pack_limb_ids`, `numeric_feature_descriptor`, `numeric_component_results`, `numeric_composite_pending` | identity | the existing `source_precision_*` and `source_numeric_*` pages (declared raw today; plan 90 section 5 leaves them "not in these steps") -- step 9 lists them as the residue that the attribute-cache finding will show |
| `producer_kind`, `iterator_kind`, `constant_folded`, `materialization_axis`, `materialization_kind`, `tensor`, `tensor_resolution`, `tensor_output_descriptors`, `basic_index_axes`, `basic_index_source_shape`, `optional_presence`, `control_ir_owned`, `unrolled_from` | payload with an identity edge | the node cell is the row; a descriptor's shape half is `proven_shape` / `planner_tensor_descriptor` (step 4).  `unrolled_from` names a node: identity -> `IDENTITY_TRANSITION` Fork |
| `aggregate_leaf_value_ids`, `materialized_source_value_ids`, `materialized_value_ids`, `materializer_node_id`, `value_source_id`, `structural_identity_actual_value_id`, `recovered_initial_value_id`, `aggregate_leaf_republication`, `deployment_memberships`, `recursion_region_id`, `collection_iterable_value_id` | identity | `callsite_projection_*`, `call_result_projection_concordance`, `deployment_region_member`, `loop_region_membership` (step 4); `value_source_id` / `structural_identity_actual_value_id` / `recovered_initial_value_id` -> `name_binding` REVISE (plan 80 A2.7) |

Audit finding `attribute-cache-drift` (generic, table-driven): a declared
map attribute key -> (page, row builder, fact projection); for every node
of every graph the case reaches, the attribute's value must equal the
projection of the page's latest fact, else the finding names node,
attribute and both values.  The map is declared beside the pages in
`concordance_declarations.py` so a new identity attribute cannot be added
without saying which page owns it.

## 4. Part B: emission as the last layer

### 4.1 What emission needs from `ssa_value_identity`

Plan 80 B1.1 declares `ssa_value` (`(function scope, ssa id)`, NOVEL through
`fresh_value`) for builder-minted values; plan 90 section 1 declares
`SSA_VALUE_IDENTITY` (`(function_scope, value_id)`, NOVEL or DERIVED) for
values minted after reduction, and the resolver
`ssa_value_identity_cell(function, value_id)`.  Two pages, one identity
space.  Emission needs exactly four things from that seam, stated here so
steps 5 and 6 land them in the shape emission reads:

1. **One resolver over both pages.**  `ssa_value_identity_cell(function,
   value_id) -> Ref | None` (plan 90 E6.1) looks up `SSA_VALUE_IDENTITY`,
   then `ssa_value`, then -- for an SSA id that IS a graph id
   (`_value_from_meta`, ADOPTED_GRAPH_ID) -- the `control_value_binding` row
   whose fact's `ssa_value` cell it is, then `function_parameter` for a
   formal.  Emission calls this and nothing else.
2. **Total coverage.**  Every `SSAValue` a backend touches -- `function.args`,
   every `instr.res`, every element of `instr.args` -- resolves.  The values
   plan 80 B9 and plan 90 section 5 leave for later (`tensor_ssa_lowering.fresh`,
   `ir_identities` mints, `ssa_call_input_adapters`,
   `deployment_ssa_binding.bind_deployment_dataflow`'s mint,
   `with_native_sgd_loop` / `with_native_adam_loop`'s `fresh_id()`) are the
   ones emission will find first; each is posted `Unsourced(VALUE_WITHOUT_IDENTITY_CELL)`
   on its unit row under the latch (4.3), so the list is the worklist, and
   `with_native_*_loop`'s wrapper values get their own transform here
   (`NATIVE_LOOP_WRAPPER_VALUE`, arity 1, operand = the wrapped root's
   `FUNCTION_SCOPE` cell) because no other step owns them.
3. **The row is keyed by the function's scope, and backends hold only a
   `Function`.**  `function_scope_of(function)` (plan 90 N1 / E6.1) must be
   readable from the finished module after the compile closed: it reads
   `FUNCTION_SCOPE` on the ATTACHED book (`identity_book(module)`), never
   `current_identity_book()`.
4. **Refs must resolve on the attached book after `end_identity_book`.**
   `_source_stamp` checks `self.pages`; the attached book holds them.  A
   module whose metadata has no book (pickled through the host cache, or a
   bare `Mapping` handed to Fortran's `emit_module`) has no cells to derive
   from; emission then posts nothing and records `Unsourced(NO_BOOK_AT_EMISSION)`
   once per artifact on the detached ambient book, which is visible in the
   audit and can never close the latch (risk R9.7).

### 4.2 Pages

```
Page EMISSION_UNIT
  row_fields = (RowField(function_scope, SCOPE),
                RowField(backend, LABEL),            # Backend enum: C_SCALAR, C_MODULE, LLVM_SCALAR, LLVM_MODULE, FORTRAN, WASM_WAT, WASM_BINARY, JAVASCRIPT
                RowField(unit, INDEX))               # emission ordinal within (function, backend), in the order the text is produced
  fact_type = EmittedUnit(kind: UnitKind, text: str, spelling: str)
      UnitKind: FUNCTION_HEADER, FORMAL, BLOCK_LABEL, STATEMENT, INLINED_EXPRESSION, LITERAL,
                PHI_EDGE_ASSIGNMENT, BRANCH, RETURN, CALL, DECLARATION, OUTPUT_STORE, TABLE, PROTOTYPE
      text     = the exact line(s) appended to the artifact (or, for INLINED_EXPRESSION and LITERAL, the expression text that was inlined -- no line of its own exists)
      spelling = the token the unit binds the result value to (`t{id}`, `v{id}`, `%...`, the WASM local index, the JS name); "" for a unit with no result
  mode = CONCORD
  provenance = DERIVED(the identity cell of instr.res when there is one, the identity cell of every instr.args element,
                       the SSA_BLOCK cell of the block being emitted, the EMISSION_FUNCTION-HEADER cell (4.3) of the enclosing function;
                       JavaScript additionally the name_binding cell whose name it printed;
                       a Phi edge assignment additionally the SSA_BLOCK cells of source and target blocks)
```

The ordinal is the row key because two units can have the same text (two
`goto` lines) and a unit can spell no value (a label).  The cells it derives
from are what make it findable from the value side: `edges_out_of(value
cell)` lists every unit that spelled the value, in every backend.

```
Page EMISSION_FUNCTION
  row_fields = (RowField(function_scope, SCOPE), RowField(backend, LABEL))
  fact_type = FunctionEmission(symbol: str, unit_count: int, text_sha256: str)
  mode = CONCORD
  provenance = DERIVED(the FUNCTION_SCOPE cell; the function's FUNCTION_HEADER unit cell)
```

Posted FIRST (with `unit_count` and hash as `Unresolved(FUNCTION_TEXT_PENDING)`)
so the units can derive from it, then REVISED once at the end of the
function's emission with the final count and hash DERIVED(every unit cell of
the function) -- so the page is REVISE, not CONCORD, and the second post's
source set differs from the first (admitted).  `unit_count` is the check
that no `append` bypassed the post.

```
Page EMISSION_ARTIFACT
  row_fields = (RowField(artifact, NAME),            # the artifact / entry name the harness asked for
                RowField(backend, LABEL),
                RowField(part, LABEL))               # ArtifactPart: MODULE_TEXT, SOURCE_FILE, PIECE_FILE(symbol), COMPILE_COMMAND, LIBRARY, BINARY, API_CONTRACT, BUFFER_ORDER
  fact_type = ArtifactFact(sha256: str, byte_length: int, location: tuple)   # location = path parts or the command tuple; never the text
  mode = REVISE
  provenance:
    MODULE_TEXT     DERIVED(every reachable function's EMISSION_FUNCTION cell, the entry wrapper's unit cells)
    SOURCE_FILE     DERIVED(MODULE_TEXT cell)                                     -- written by compile()/write()
    PIECE_FILE      DERIVED(the LLVM piece's own EMISSION_ARTIFACT MODULE_TEXT cell) -- linked_llvm; a piece never emitted on this book is Unsourced(PIECE_ARTIFACT_UNROUTED)
    COMPILE_COMMAND DERIVED(SOURCE_FILE cell, every PIECE_FILE cell)               -- the flags (work contract) are payload in `location`, not a source: the contract is not on the book (listed in section 8)
    LIBRARY         DERIVED(COMPILE_COMMAND cell)
    BINARY (wasm)   DERIVED(MODULE_TEXT cell)                                     -- `_assemble` mirrors the WAT step for step
    API_CONTRACT    DERIVED(every function's FUNCTION_HEADER unit cell, the `function_output` cells (step 5) the publications name)
    BUFFER_ORDER    DERIVED(the identity cell of every value in buffer_order / extent_order)
```

Stages: `EMISSION_C`, `EMISSION_LLVM`, `EMISSION_FORTRAN`, `EMISSION_WASM`,
`EMISSION_JAVASCRIPT`, `ARTIFACT_BUILD`.  Reasons: `VALUE_WITHOUT_IDENTITY_CELL`,
`NO_BOOK_AT_EMISSION`, `BLOCK_ORIGIN_UNROUTED`, `FUNCTION_TEXT_PENDING`,
`PIECE_ARTIFACT_UNROUTED`, `NO_FUNCTION_SCOPE` (plan 90), `WASM_REGION_UNROUTED`.
Transform: `NATIVE_LOOP_WRAPPER_VALUE` (1).

### 4.3 How each backend posts without changing its text

Each emitter gains one local closure, made by a shared helper in a new small
module `src/compiler/emission_concordance.py` (the backends must not import
the reducer):

```
recorder = emission_recorder(book, function, backend)      # None when book is None: every method is a no-op that counts
recorder.header(text, spelling)                            # posts EMISSION_FUNCTION (pending) then the FUNCTION_HEADER unit
recorder.unit(kind, text, *, result=None, args=(), block=None, spelling="", extra_cells=())
recorder.finish(text_of_function)                          # REVISE EMISSION_FUNCTION with count and hash
```

`recorder.unit` resolves each `SSAValue` through `ssa_value_identity_cell`,
posts the unit DERIVED from the cells found, and when some value has no cell
posts the SAME row `Unsourced(VALUE_WITHOUT_IDENTITY_CELL)` instead (the
latch admits it; the audit lists it with the backend stage and the value's
`%t` spelling in the fact).  The text is what was appended: the emitter
appends to its list exactly as today and hands the same string to the
recorder, so the artifact is byte-identical (4.6 checks this).

| backend | book | `header` | `unit` sites | `finish` / artifact |
|---|---|---|---|---|
| `emit_ssa_function_to_c` | `identity_book(module)` | the `TURING_EXPORT void {name}(const double *in, double *out)` line | each `lines.append("const double t{id} = ...")` -> STATEMENT with `result=instruction.res`, `args=instruction.args`; each Const / Pi -> LITERAL with the hex text; each `stores.append` -> OUTPUT_STORE `args=(value,)`; `emitted_tables` -> TABLE | `finish(source)`; `CFunctionArtifact.compile` posts SOURCE_FILE, COMPILE_COMMAND, LIBRARY |
| `emit_ssa_module_to_c` | `identity_book(module)` | per `fn`: the `static {ret} {_c_symbol(fn)}(...)` definition line (PROTOTYPE for the prototype list) | in the per-block loop, after each `body.append` for the current `emission_context["instruction"]` (the context already names block and instruction: the recorder reads it) -> STATEMENT / CALL / RETURN / BRANCH by `op`; `body.append(f"{_c_label(block_name)}: (void)0;")` -> BLOCK_LABEL with `block=`; `phi_edge_assignments` -> PHI_EDGE_ASSIGNMENT; the hoisted Phi declarations -> DECLARATION; `output_publications` -> OUTPUT_STORE; trace lines (`trace_lines`, `value_trace_lines`) are NOT units (diagnostics that change no value; recorded as TRACE only when `trace=True`, so the default artifact's units are its program); the entry wrapper -> units under the root function's scope | `finish` per `fn`; `CModuleArtifact` construction posts MODULE_TEXT and BUFFER_ORDER; `compile` posts SOURCE_FILE, PIECE_FILE per `linked_llvm`, COMPILE_COMMAND, LIBRARY; `compile_standalone` the same parts under `part` labels suffixed `STANDALONE` |
| `emit_ssa_function_to_llvm` | `identity_book(module)` | the `define void @{name}(ptr %buffers, ptr %extents)` line | each `lines.append` in the instruction loop -> STATEMENT with the `scalars[...]` rendering as spelling; `globals_out` -> DECLARATION; `buffer(value_id)` loads -> STATEMENT `args=(value,)` | `finish`; `LLVMFunctionArtifact` posts MODULE_TEXT, BUFFER_ORDER; `compile_artifact` posts SOURCE_FILE, COMPILE_COMMAND, LIBRARY |
| `_emit_repository_call_module` | `identity_book(module)` | per reachable function the `define internal void @{internal_symbols[name]}(...)` line; the wrapper `define void @{entry_name}` under the root's scope | every `body.append` in the per-function loop (the helpers `pointer`, `load_as`, `literal`, `emit_return_values` return text; the loop that appends it is the site); kernel `definitions[symbol]` from `extract_llvm_function` -> DECLARATION units under the root's scope DERIVED(the call instruction's cell that demanded the kernel); `capture_block_history` -> TRACE (diagnostic) | as above; `with_native_sgd_loop` / `with_native_adam_loop` post their wrapper's units under a NOVEL `NATIVE_LOOP_WRAPPER_VALUE` row per `fresh_id()` |
| `_FunctionEmitter` (Fortran) | `identity_book(module)` when `emit_module` received an `IRModule`; else None -> `Unsourced(NO_BOOK_AT_EMISSION)` once per `FortranSubroutine` on the detached book | the `subroutine {native_symbol}(...) bind(C)` line `emit()` produces | in `_emit_block`: `! block` / `{n} continue` -> BLOCK_LABEL; `goto` / `if (...) then ... goto` -> BRANCH; `return` -> RETURN; `_statements(instr)` groups -> STATEMENT (one unit per instruction, `text` = the joined group); an instruction `_may_inline` folded into its consumer -> INLINED_EXPRESSION with `text=_expression(instr)` and `spelling=""` posted where `_operand` inlines it, so a value that has NO statement of its own is still a unit; declarations `emit()` builds from `_locals` -> DECLARATION | `emit()` -> `finish`; `emit_module` posts MODULE_TEXT; `FortranModule.write` posts SOURCE_FILE and API_CONTRACT; `compile_module` posts COMPILE_COMMAND and LIBRARY |
| `emit_wasm_module`, `_assemble`, the matmul / container / name-hash variants | passed in: `emit_wasm_module(program, ..., book=None, region_cell=None)`; callers (`wasm_class_modules`, `site_bundle`, `machine_targets`, the `abstract_ui_*` modules) pass `identity_book(module)` and the `deployment_region` cell (step 4) of the region the `FusedProgram` was carved from | the `(func ${function_name} ...)` line | `evaluate_steps` / `emit_step`: each appended instruction group + `local.set` -> STATEMENT with `result` the `canonical_value` cell of `step.result_id` and `args` the cells of `step.input_ids`, resolved through the region's `deployment_region_member` cells (the value ids of a `FusedProgram` are the region's graph ids; `program.extras` carries the `ssa_identity_tokens` copy `remap_program` filtered -- observed in plan 60 3.1 (a); the read scope must travel with it, section 6 item 7); `_assemble.emit_step` -> the same units under `WASM_BINARY` DERIVED(the `WASM_WAT` unit cell of the same step) | `WasmModule` posts MODULE_TEXT and BINARY; `write` posts SOURCE_FILE, API_CONTRACT; `compile_wat` posts COMPILE_COMMAND, LIBRARY |
| `emit_ssa_module_to_javascript` | `identity_book(module)` | the `function impl_{symbol}(...)` line; class wrappers as FUNCTION_HEADER units with the `class_declaration` cell as extra source | each `definitions.append` line inside the per-function loop -> STATEMENT / CALL / RETURN; the value's printed name -> `extra_cells=(name_binding cell,)` from `_source_names` | `finish`; `JavaScriptModuleArtifact` posts MODULE_TEXT, API_CONTRACT, BUFFER_ORDER |

The recorder never raises for a missing cell (it posts `Unsourced`); it DOES
raise `ConcordanceRefusal` for a malformed row or an undeclared stage, which
is a programming error at the emitter, found at the first probe.

### 4.4 Readers and the view

No backend reads the pages; the emitted text is still assembled from the
emitter's own lists.  Readers are the audit and the viewer:

- audit finding `emitted-unit-unsourced` (generic `unsourced-fact` over the
  three pages: registration only) and `emitted-function-count`: for every
  `EMISSION_FUNCTION` row, `unit_count` equals the number of `EMISSION_UNIT`
  rows under `(function_scope, backend)` -- an `append` that bypassed the
  recorder shows here;
- `spelled-values`: for every `Function` reaching a backend, every value id
  that `ssa_value_identity_cell` resolves has at least one `EMISSION_UNIT`
  edge out of its cell, or is named by a declared elision (Fortran
  INLINED_EXPRESSION counts as spelled; a Phi in C is spelled by its
  hoisted DECLARATION; an aggregate projection skipped by
  `aggregate_projection_instruction_ids` is `Unresolved(UNIT_ELIDED,
  read=(the binding instruction's unit cell,))` on an `EMISSION_UNIT` row
  of its own, so the elision is a row);
- the viewer (4.5).

### 4.5 What the viewer then draws

`extract_graph` already makes a node per row and takes DERIVED / MINT edges
from `edges_into` / `mint_of`; `EMISSION_UNIT` rows appear as nodes whose
fact text is a string atom (skipped for id nodes) and whose inbound edges
go to value cells.  One addition: `resolve_focus(graph, spec)` accepts
`token:<backend>:<function>:<spelling>` and seeds every `EMISSION_UNIT` row
whose fact's `spelling` equals it; `diffuse_heat` then runs along
`causal_edges` backwards (source direction), which is the chain the user
asked for, page by page:

```
EMISSION_UNIT (C token, e.g. t{id})
  -> SSA_VALUE_IDENTITY / ssa_value        (the value the token spells)
     -> [mint operands] control_value_binding / CELL_SET / region_signature / ssa_field_version
        -> canonical_value                 (the graph node the SSA value was made from)
           -> [Move/Append rows of IDENTITY_TRANSITION: the operand edges]
           -> ingestion_value              (the same node before the relabel)
              -> source_span               (the AST construct; a root: mint_of returns (INGEST_SOURCE, ()))
  -> SSA_BLOCK -> CONTROL_BLOCK -> canonical_value (the if / loop node) -> source_span
  -> EMISSION_FUNCTION -> FUNCTION_SCOPE -> scope_registry
```

with the planning hops in between where the value went through a region:
`ssa_value (REGION_CALL_RESULT) -> region_signature -> deployment_region ->
deployment_region_member -> executable_node -> canonical_value`, and the
record hops where it went through a field: `ssa_field_version ->
reducer_field_state -> canonical_value (receiver, RHS) -> source_span`.
Every hop is an existing or planned page; the viewer draws them because
the edges exist, not because it knows the pages.

### 4.6 Byte-identity check

`probe_scalar_native_correctness` already compiles and runs every program;
the emission posts must not change `artifact.source`.  The probe (section 7)
emits each program twice, once with `identity_book(module)` replaced by a
detached book (recorder no-ops) and once normally, and asserts the two
`source` strings are equal.  That is the proof that the recorder is beside
the text, not in it.

## 5. Ordered edit list

E9.1  `concordance_declarations.py`, a "Step 9" section: `ControlBlockKind`,
      `Arm`, `Backend`, `UnitKind`, `ArtifactPart` enums; fact types
      `ControlBlockFact`, `Placement`, `ControlProgramFact`, `BlockFact`,
      `EmittedUnit`, `FunctionEmission`, `ArtifactFact`, `LoopSiteFact`;
      pages `CONTROL_BLOCK`, `CONTROL_BLOCK_PLACEMENT`, `CONTROL_PROGRAM`,
      `SSA_BLOCK`, `EMISSION_UNIT`, `EMISSION_FUNCTION`, `EMISSION_ARTIFACT`,
      `LOOP_CONTROL_SITE`, `EXTRACTION_RECEIPT`; stages `CONTROL_PROGRAM_BUILD`,
      `CONTROL_PROGRAM_REWRITE`, `EMISSION_*`, `ARTIFACT_BUILD`; transforms
      `SYNTHESIZED_CONTROL`, `NATIVE_LOOP_WRAPPER_VALUE`, `INGEST_EDGE`,
      `REDUCER_SYNTHESIS`, `AGGREGATE_MEMBER`, `BOUND_RECEIVER`,
      `LOOP_BODY_CLONE`, `LOOP_MATERIALIZER`, `OPTIONAL_PRESENCE_*`; reasons
      of 4.2 plus `CONTROL_OWNER_UNKNOWN`, `REGION_CELL_UNROUTED`,
      `COLLAPSED_EMPTY_CONSTRUCT`, `UNIT_ELIDED`; the attribute-cache map of
      section 3.  If `IDENTITY_TRANSITION` (plan 80 B1.3) is not yet
      declared, declare it here with plan 80's shape.
E9.2  `topological_reducer._operand_position_scope`: the `ingestion_value_scope`
      fallback.  `_set_operands`: the Append post and
      `Unsourced(NO_OPERAND_POSITION_SCOPE)` (plan 70 section 3) if step 5
      has not landed them; ownership of `children` and the networkx edge
      (with `edge_payload`); delete the legacy `cause: str` branch once E9.4
      is done.
E9.3  `new_node`: post `ingestion_value` before `_set_operands`; `cause:
      Transform` parameter; its callers pass their transform.
E9.4  Every hand-written `children` / `add_edge` / `parents` writer of 0.3
      (reducer helpers, glsl materializers and `_dispatch_subgraph`,
      loop_composer `add_constant` / `add_port` / clone / materializer /
      `rewire_continuation`, fortran_c_shell optional-presence rewrite):
      through `_set_operands` with a `Transform`.  `graph_express2`'s
      AST-walk edge writer: `_set_operands(..., cause=INGEST_EDGE)`.
E9.5  audit: `operand-view-drift` (1.3).
E9.6  `control_source.py`: `post_control_program`, `control_program_view`;
      `LoopControlBlock` return carries `return_site_cell` in its fact.
E9.7  Builders post: `_ordinary_conditional_control_programs`,
      `analyze_shader_loop_reductions`' loop-block construction, the
      `precompile_to_ssa` control-function assembly (`CONTROL_PROGRAM_BUILD`).
E9.8  Rewriters post (`CONTROL_PROGRAM_REWRITE`): the seven `control_source`
      passes, `fix_aggregate_loop_bounds`, `_class_surface_ssa_program`'s
      inner rewriters and `_install_lexical_sequence_mutations` -- one
      `post_control_program` call at each return, with the cause cells of
      2.3; the synthesized `ConditionalBlock` NOVEL.
E9.9  `_ControlSSABuilder`: `SSA_BLOCK` posts at block creation; `_lower`
      reads the `CONTROL_BLOCK` cell of the block it lowers and step 5's
      construct-cell derivations point at it.  audit: `control-view-drift`.
E9.10 `LOOP_CONTROL_SITE` posts in `reduce_statement`'s loop branch where
      `loop_break_sites` / `loop_break_bindings` are written;
      `EXTRACTION_RECEIPT` post in `ensure_node` beside `_post_ingestion_value`;
      audit `attribute-cache-drift`.
E9.11 `src/compiler/emission_concordance.py`: `emission_recorder`, the
      artifact posting helpers (`post_artifact_part`), `NATIVE_LOOP_WRAPPER_VALUE`
      posting for `with_native_*_loop`.  `identity_concordance.ssa_value_identity_cell`
      covers the four lookups of 4.1 (with step 6's helper if landed).
E9.12 `ssa_c_backend`: `emit_ssa_function_to_c`, `emit_ssa_module_to_c`
      (recorder beside every `lines.append` / `body.append`; the
      `emission_context` feeds it), `CFunctionArtifact.compile`,
      `CModuleArtifact.compile` / `compile_standalone`, the artifact
      constructors.
E9.13 `ssa_llvm_backend`: `emit_ssa_function_to_llvm`, `_emit_repository_call_module`,
      `compile_artifact`, `with_native_sgd_loop`, `with_native_adam_loop`.
E9.14 `ssa_fortran_backend`: `_FunctionEmitter.__init__` takes `recorder`;
      `_emit_block`, `_operand` (inlined units), `emit`; `emit_module`,
      `emit_subroutines` (pass the recorder factory), `FortranModule.write`,
      `compile_module`; the `Mapping` input path posts `NO_BOOK_AT_EMISSION`.
E9.15 `fused_program_wasm_backend`: `emit_wasm_module(..., book, region_cell)`
      and the four variant emitters, `_assemble`, `WasmModule.write`,
      `compile_wat`; callers pass the book and region cell.
E9.16 `ssa_javascript_backend.emit_ssa_module_to_javascript`.
E9.17 `tools/view_identity_concordance.py`: `resolve_focus` token spec.
E9.18 `fortran_c_shell._dump_identity_book_log(..., phase)`: a second dump
      `{label}.{stamp}.emitted.log` called by `post_artifact_part` when it
      posts LIBRARY (decision, section 9.1).
E9.19 `tools/compiler_probes/probe_scalar_native_correctness.py`: section 7;
      `TEST_BASELINE_AND_HAZARDS.md` line.
E9.20 audit: `emitted-function-count`, `spelled-values`, the completeness
      report (section 8) and `--completeness` in `tools/audit_identity_concordance.py`.

Writer sites routed: about 24 hand-written edge writers (E9.4) + 2
(E9.2-E9.3); 3 builders + about 22 rewriter returns (E9.7-E9.8); block
creation in the builder (E9.9, one helper); 2 reducer / ingestion posts
(E9.10); emission: C 2 emitters + 3 build methods, LLVM 2 + 1 + 2 wrappers,
Fortran 1 emitter class + 3 module functions, WASM 5 emitters + `_assemble`
+ 2 build functions, JS 1 -- every `append` inside them goes beside one
recorder call (counted at edit time: the recorder's `unit_count` check is
what makes a missed `append` visible).  Pages: 9 new, 1 declared-from-raw
if step 5 has not.

## 6. What step 9 needs from steps 4-8

1. **Step 5 F3** (`IDENTITY_TRANSITION` declared, Append posted, `cause:
   Transform`).  Fallback: E9.2 does it; the two lanes must not both edit
   `_set_operands` -- whichever lands first owns it, the other rebases.
2. **Step 4 E7 / E8** (`executable_node`, `deployment_region`,
   `deployment_region_member`): the owner cells of region markers and the
   WASM units.  Fallback: `Unsourced(REGION_CELL_UNROUTED)` / `WASM_REGION_UNROUTED`
   until they land; the blocks and units still post.
3. **Step 4 E6 / E13** (`call_binding`, `loop_carried_binding`,
   `loop_result_port_binding`, `rewire_continuation` through `_set_operands`).
4. **Step 5 B2.1 / B2.6** (`ssa_value` NOVEL rows, `function_parameter`,
   `function_output`) and **step 6 E6.0 / E6.1** (`FUNCTION_SCOPE`,
   `SSA_VALUE_IDENTITY`, `ssa_value_identity_cell`, `mint_scope` through
   `post`): without them every unit is `Unsourced(VALUE_WITHOUT_IDENTITY_CELL)`
   and the chain from a token stops at the unit.  Step 9's emission part
   is therefore ordered AFTER steps 5 and 6; its Part A can land any time
   after step 3.
5. **Step 7 E7.3 / E7.12** (linker-minted values on `SSA_VALUE_IDENTITY`):
   the C module lane's frame storage formals (`v{id}`) resolve only then.
6. **Plan 90 N3** (`return_site_cell` on `ReturnBlock`): E9.6 carries it.
7. **From the planner (step 4 or here):** `FusedProgram.extras` must carry
   the read scope (or the `deployment_region` cell) beside the
   `ssa_identity_tokens` copy `remap_program` places there, or
   `emit_wasm_module` cannot build a `canonical_value` Ref from
   `step.result_id`.  One line in `remap_program`; listed as E9.15's
   precondition.

## 7. The seconds-long proof

`python -u tools/compiler_probes/probe_scalar_native_correctness.py`
(seconds; seven programs, each annotated and plain, lowered, emitted to C,
compiled, run against CPython).  Extended, reading only the book
`identity_book(module)` after `emit_ssa_module_to_c` and `artifact.compile`:

1. **Byte identity** (4.6): emit each program twice (detached book / real
   book); `artifact.source` equal.
2. **One token's chain.**  For program `bump` (`k + 1`), take the value the
   root's `Ret` returns (`outputs[root.name][0].id`); find its identity cell
   through `ssa_value_identity_cell`; take `edges_out_of` that cell
   restricted to `EMISSION_UNIT` rows with `backend == C_MODULE` -- the C
   statement `t{id} = ...` (or the output store).  Print the token.  Then
   walk `edges_into` from the unit breadth-first, printing each hop as
   `page row -> page row (stage)`, stopping at rows whose `mint_of` has no
   operands (roots) or at depth 40.  Assert the walk reaches a `source_span`
   row whose fact `kind` is `BinOp`, passing through `canonical_value`,
   `ingestion_value`, and for the operand `k` through `scalar_parameter`
   (annotated form) or `name_binding` (plain form) -- the two forms must
   differ by exactly those rows, as plan 60 3.5 (f) already requires of the
   same probe pair.  Print the hop count.
3. **Every unit derived.**  Count `EMISSION_UNIT` rows for the root function
   under `C_MODULE`; count those listed by `unsourced_rows()`; print both.
   Expected after E9.12 with steps 5-6 landed: unsourced 0 for `bump`,
   `chain`, `twice` (straight-line scalar); for `cond` and `loop` the
   `BLOCK_LABEL` units are sourced iff E9.9 landed (printed either way);
   for `scale` and `shared` the tensor region's units resolve through
   `region_signature` iff step 4 E8 landed.
4. **Artifact chain.**  The `EMISSION_ARTIFACT` LIBRARY row has an edge to
   COMPILE_COMMAND, which has one to SOURCE_FILE, which has one to
   MODULE_TEXT, which has one per reachable function's `EMISSION_FUNCTION`
   cell; `unit_count` on each equals the unit rows.
5. **The view drift findings** (`operand-view-drift`, `control-view-drift`,
   `attribute-cache-drift`) over the probe's module: printed counts, expected
   zero for the straight-line programs, and for `cond` / `loop` the count is
   the measure of rewriters not yet routed (E9.8).
6. `unsourced-fact` by stage before and after, as every plan's probe prints
   it, so the drop is measured.

Also green after the step: `probe_annotated_scalar_parameter.py`,
`probe_struct_intake.py`, `probe_branch_written_field.py` (unchanged
assertions), `python -u tools/audit_identity_concordance.py controller`
(the audit cases lower but do not emit; the Part A findings run on them, the
Part B pages are empty there, which proves emission costs nothing when no
backend ran).

## 8. The completeness measure

`concordance_completeness(book, module, artifacts) -> CompletenessReport`
(`identity_concordance.py`; `tools/audit_identity_concordance.py --completeness`
prints it), five ratios and the rows behind each:

1. **cells sourced**: cells on declared non-private pages with an inbound
   edge or a mint row / all such cells (the existing `unsourced-fact` as a
   ratio; 1.0 is plan 90's latch criterion 1).
2. **identities minted**: MINTED ids with a mint row / all MINTED ids
   (`unsourced-identity`; latch criterion 2).
3. **edges viewed**: process-graph operand edges whose latest
   `IDENTITY_TRANSITION` row agrees / all edges of every graph reached
   (section 1.3), and control blocks with a `CONTROL_BLOCK` row and a
   resolved placement / all blocks in the tree handed to the builder
   (section 2.4).
4. **units derived**: `EMISSION_UNIT` rows with an inbound edge whose
   breadth-first source walk reaches a root (a Novel row with an arity-0
   transform: `source_span`, `contract_demand`, `scope_registry`) / all
   units; printed per backend.
5. **values spelled**: SSA values reaching a backend with at least one unit
   edge out of their cell or a declared elision row / all such values.

"Every cell edged or minted; every emitted unit derived" is ratios 1, 2 and
4 at 1.0.  The report names the first ten offenders of each ratio by page,
stage and unit so the number is a worklist, not a score.  Not on the book
and therefore outside the ratios, named so the report is read correctly:
the work contract (compiler flags, `active_contract()`), the toolchain
version, the `CompiledProgramAPI` schema itself, the trace helpers.

## 9. Risks

R9.1  **Volume and clock cost.**  One unit row and two to five edge rows
      per emitted line, per backend lane emitted; `EMISSION_FUNCTION`'s
      final REVISE adds one edge per unit again.  A whole-program C module
      (the Woodshop compile) is tens of thousands of lines; the book grows
      by that times roughly six dict writes, and `render_identity_book`
      grows linearly.  `post` itself is dict operations and one clock tick
      -- no scan -- so the cost is memory and log size, not time complexity.
      Mitigations in the plan: TRACE units only under `trace=True`; the
      final REVISE derives from a `CELL_SET` of the unit cells (plan 90 N2)
      when the user chooses the set-row answer, halving the edge count.
      Measure on `probe_scalar_native_correctness` (small) and the `controller`
      case lowered then emitted once (seconds) before the Woodshop compile.
R9.2  **The book at emission time -- the top risk.**  Backends run after
      `end_identity_book`; they must use `identity_book(module)` and never
      `current_identity_book()` (which would mint a detached book and lose
      every post silently).  Three inputs have no attached book: Fortran's
      `Mapping` path, `FusedProgram` (no module at all), and a module
      unpickled from the host cache if `_HostSSACachePickler` drops the
      metadata book.  Each posts `Unsourced(NO_BOOK_AT_EMISSION)`; none
      can close the latch.  The `FusedProgram` case is fixed by E9.15
      (callers pass the book); the `Mapping` path stays listed; the host
      cache needs `_HostSSACachePickler` read before E9.12 (R9.7).
R9.3  **REVISE refusals in `post_control_program`.**  Fifteen rewriters run
      in sequence; a rewriter that re-derives a placement from the same
      cells is refused.  The helper compares with the latest fact before
      posting and passes cause cells that differ per rewriter (2.3); a
      refusal that still occurs names a rewriter that moved nothing and
      claimed a cause -- fixed at the rewriter, never caught (plan 80 R5).
R9.4  **`_set_operands` owning `children` and the networkx edge.**  Some
      graphs carry edges with payload (`extra=set()` from graph_express2,
      `**edge` copied in the optional-presence rewrite); `_set_operands`
      must preserve the payload through `edge_payload`, and `remove_edge`
      must not drop an edge another position of the same consumer still
      names.  The `operand-view-drift` finding on the seven audit cases is
      the check; run it after E9.2 and before E9.4.
R9.5  **Block owner cells missing.**  A `StatementBlock` marker before step
      4, a synthesized conditional, a planner-made loop: each is `Unsourced`
      or NOVEL by 2.2, never refused.  The count of `CONTROL_OWNER_UNKNOWN`
      on the `controller` case is the honest residue.
R9.6  **The compile-end log predates emission.**  `_dump_identity_book_log`
      runs in `lower_ast_source_to_ssa`'s `finally`; emission rows are not
      in that file.  E9.18 writes a second log when a LIBRARY part is
      posted; the alternative is that harnesses render the book themselves.
      Held (9.1 below).
R9.7  **Pickled modules.**  Whether `metadata["identity_book"]` survives
      `_HostSSACachePickler` is not read here.  If it does, the cached
      module carries a whole book (size); if it does not, emission from a
      cached module is `NO_BOOK_AT_EMISSION`.  Read
      `host_code_modules._HostSSACachePickler` before E9.12 and state which.
R9.8  **`emission_context` is C-module-only.**  The other emitters have no
      "current instruction" slot; the recorder must be handed the
      instruction at each `append`.  Missing one is caught by
      `emitted-function-count`, not by a refusal.
R9.9  **JavaScript spells by name.**  `_source_names` prefers the authored
      name; two values with one name (versions) print the same token.  The
      unit's `spelling` is that token; the value cell disambiguates, and
      `token:` focus in the viewer seeds every unit with that spelling --
      correct, since the source had one name.

## 10. Held for the user

1. **R9.6:** second book log at LIBRARY time (E9.18, proposed) or no second
   log (probes and the viewer render `identity_book(module)` themselves).
2. **Always-on emission recording** (proposed: every backend posts whenever
   a book is attached) or a per-call switch.  Always-on is the design's
   rule (the book is the whole description); the cost is R9.1.
3. **`EMISSION_FUNCTION`'s final REVISE**: derive from every unit cell
   (proposed; N edges) or from one `CELL_SET` (plan 90 N2's answer decides).
4. **The view replacement** (2.4): step 9 measures drift and does not
   replace the passed tree by the view; whether to make the replacement in
   this step once drift is zero on the seven cases, or in a step 10.

## 11. Continuation

Decided by this plan:

- Part A (1): the operand edge is the latest `IDENTITY_TRANSITION` fact;
  `_set_operands` is the one writer of `parents`, `children` and the
  networkx edge; `_operand_position_scope` falls back to
  `ingestion_value_scope`; every hand-written edge writer of 0.3 routes
  through it with a `Transform`; `operand-view-drift` measures the view.
- Part A (2): pages `CONTROL_BLOCK` (row keyed by the construct's cell,
  fact = identity fields as cells), `CONTROL_BLOCK_PLACEMENT` (REVISE
  history of where the block sits), `CONTROL_PROGRAM`, `SSA_BLOCK`; one
  helper `post_control_program` called at every builder and rewriter
  return; `control_program_view` and `control-view-drift`.
- Part A (3): the identity-fact / payload rule of section 3, the
  attribute -> page map declared beside the pages, two new pages
  (`LOOP_CONTROL_SITE`, `EXTRACTION_RECEIPT`), `attribute-cache-drift`.
- Part B: pages `EMISSION_UNIT` (function scope, backend, ordinal; fact =
  kind, text, spelling; DERIVED from the value cells it spells and the
  block and function cells), `EMISSION_FUNCTION`, `EMISSION_ARTIFACT`
  (MODULE_TEXT -> SOURCE_FILE -> COMPILE_COMMAND -> LIBRARY, plus PIECE_FILE,
  BINARY, API_CONTRACT, BUFFER_ORDER); one `emission_recorder` beside every
  `append` in the five backends, reading `identity_book(module)`, never
  `current_identity_book()`; byte-identical text; a value without a cell is
  `Unsourced(VALUE_WITHOUT_IDENTITY_CELL)` on its unit row, a backend
  without a book is `Unsourced(NO_BOOK_AT_EMISSION)`; the viewer's `token:`
  focus; the five completeness ratios.
- Order: Part A after step 3 (any time); Part B after steps 5 and 6
  (`ssa_value`, `SSA_VALUE_IDENTITY`, `FUNCTION_SCOPE`,
  `ssa_value_identity_cell`), before which every unit would be unsourced.

Open for the user: the four items of section 10, and whether step 9's Part
A should wait for step 5 F3 or land the Append post itself (section 6 item
1; the plan proposes: whichever lane is first owns `_set_operands`).

The exact first edit: E9.1 -- add the "Step 9" section to
`src/compiler/concordance_declarations.py` declaring the pages, fact types,
stages, transforms and reasons of sections 2.1, 2.6, 3 and 4.2 exactly as
spelled there (and `IDENTITY_TRANSITION` with plan 80 B1.3's shape if it is
not yet declared).  Declaring changes no writer and no output; run
`python -u tools/audit_identity_concordance.py view` afterwards to confirm
the registry accepts the shapes (`Registry.declare_page` refuses a
conflicting redeclaration at import).  Then E9.2 (`_operand_position_scope`
fallback and the Append post in `_set_operands`), proved by
`probe_annotated_scalar_parameter.py` plus the new `operand-view-drift`
count on the audit `view` case.
