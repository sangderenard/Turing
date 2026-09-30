# Concordance census 00: the book itself (primitives and helpers)

Read-only census of `src/compiler/identity_concordance.py`,
`src/compiler/id_space.py`, `src/compiler/monotonic_ids.py`, taken
2026-09-30 by `git grep` and `sed` only.  No compiler ids or numberings
appear here: they are valid for one run and never documented.

Relation to `CONCORDANCE_MASTER_LIST.md` (the master list): that document
classifies STRUCTURES and PIPELINE FUNCTIONS as BOOK / RECORD-ONLY /
PRIVATE by its three-part test (Source / Record / Consumed).  It has no
rows for the book's own primitives, for the helper functions defined in
`identity_concordance.py`, or for value-id minting (`monotonic_ids`,
`id_space`).  This file supplies only those, and adds three columns the
master list does not carry: per-write-site EDGE? classification,
fallback-as-fact, and the shadow ledgers each helper writes beside the
book.  Where a master-list row overlaps it is cited as "master list A1" etc.
Where this reading contradicts a master-list status or rule, section 6 says
so explicitly.  Writer files (`fortran_c_shell.py`, `precompile_to_ssa.py`,
`glsl_deployment_strategy.py`, `topological_reducer.py`, ...) are the
subject of master list Part B and of the companion censuses, not of this
file.

Goal being served: one API for writing the concordance that admits only
(a) a minted novel identity with the edge for its transform, or (b) a fact
that names the exact source row(s) + stage it derives from.  A default
chosen from absence of information must be recorded as UNRESOLVED, not as a
fact.

Vocabulary (from the module's own comments, ~line 2570, and the
`IdentityPage` dataclass):

- book: one `IdentityBook` per top-level compile (contextvar); a dict of
  pages sharing ONE construction clock.
- page: one pipeline stage's table, named by a free string.
- row: one identity, "whatever a page decides makes two facts about the
  same thing" -- any hashable; by convention a tuple whose first element is
  the scope.  Nothing enforces a row shape.
- column: one round -- "whatever round means on that page" (fixed-point
  iteration, phase index, loop generation, callsite id, revision serial).
  An int, nothing more.
- cell / fact: `cells[(row, column)]`; the fact is ANY object.  `None` is
  the removal fact by `PageMapping` convention only.
- scope: the first element of a row; `page.scopes[scope]` indexes rows by
  it.  `mint_scope` produces `(label, serial)` scopes (master list A1,
  "Table owners / scopes").
- clock / stamp: `clock[0]` is ticked by every `set`; `stamps[(row,column)]`
  records the reading.  Order across pages is recorded, not inferred.
  A page built outside a book (`IdentityPage("name")`) has its own clock.

## 1. Primitives

Every write in the whole system bottoms out in ONE method:
`IdentityPage.set(row, column, fact)`.  Everything else is sugar over it.

| primitive | signature | semantics (docstring quoted where present) | source reference in the fact? |
|---|---|---|---|
| `IdentityPage.set` | `(row, column, fact)` | Writes the cell, appends `column` to `columns` if new, stamps the clock, ticks it, indexes `row[0]` as scope. No docstring. Overwrites a cell silently if `(row, column)` already exists (the stamp slot is re-stamped; the prior fact is lost). | NONE. Row, column and fact are opaque. |
| `IdentityPage.revise` | `(row, fact) -> fact` | "Append fact as row's next revision ... For a row whose fact legitimately grows ... Identity facts that must never change use concord." Column = last column + 1 (per row). | NONE. |
| `IdentityPage.concord` | `(row, fact) -> fact` | "The first statement owns the row; a later stage may repeat it but may not replace it, so a different proposal is a disagreement, never a silent overwrite." Writes column 0 once; raises `ValueError` on a differing proposal. | NONE. First-writer-wins; the writer is not recorded. |
| `IdentityPage.mapping` -> `PageMapping.__setitem__` | `m[key] = value` | Row `(scope, key)`; revises only when the latest fact differs. `None` is refused ("None is the removal fact; delete the row"). This is how master list A2's `record_field_demand` is written. | NONE. |
| `PageMapping.__delitem__` | `del m[key]` | Revises the row to `None`. KeyError if already absent. | NONE (a deletion has no cause recorded). |
| `PageMapping.setdefault` | `(key, default)` | `if key not in self: self[key] = default`. **Writes the default as a fact.** This is the primitive that turns absence of information into a recorded decision. | NONE. |
| `IdentityPage.bind_alias` | `(scope, alias, resident)` | "Concord one planning value occurrence with its resident identity. Rebinding a row appends a new column so the page remains both the live planning authority and the history." Row `(scope, int(alias))`, fact `int(resident)`. Despite the docstring word "concord", it never raises: it REVISES. | PARTIAL: the fact IS a source id (the resident) but names no page/row/stage. |
| `IdentityBook.page` | `(name) -> IdentityPage` | Creates the page on first mention with the book's shared clock. Any string mints a new page; no registry of page names or row shapes. | n/a |
| `IdentityBook.mint_scope` / module `mint_scope` | `(label) -> (str, int)` | "A fresh scope, numbered by this book in causal order. Page scope_registry row (label, serial) records every scope minted under label." Fact is `True`. (master list A1 last row, status BOOK.) | NONE. The registry records THAT a scope exists, not what produced it. |
| `IdentityBook.latest_by_page` | `(row) -> {page: fact}` | Read: final fact per page that ever saw this exact row key. | read |
| `IdentityBook.disagreements` | `(row) -> dict or None` | Read: `latest_by_page` if the `repr`s differ. Only meaningful when two pages share one row-key shape; nothing guarantees that. | read |
| `begin_identity_book` | `() -> (book, token)` | New book, set as current (contextvar). `lower_ast_source_to_ssa` opens one; nested compiles restore the outer via the token. | n/a |
| `current_identity_book` | `() -> book` | Returns the active book, OR silently creates a `detached=True` book so "a page write is never a hard error just because nothing called begin_identity_book". A detached book is never dumped. | n/a -- a silent sink |
| `end_identity_book` | `(token) -> book` | Detaches; restores previous via token. | n/a |
| `identity_book(module)` | `(module) -> book` | Module's `metadata["identity_book"]` wins; else caches the current book onto the module. After the compile closes, callers without the module get a fresh EMPTY detached book. | n/a |

Also public and unguarded: `page.cells`, `page.columns`, `page.stamps`,
`page.scopes`, `page.clock` are plain dataclass fields.  `git grep` finds no
external mutation of them today; `fortran_c_shell.py` reads `cells.get(...)`
directly at least once (argument_binding by callsite column).

Which primitives can write a fact with NO reference to a source row:
ALL of them.  `set`, `revise`, `concord`, `__setitem__`, `__delitem__`,
`setdefault`, `bind_alias`, `mint_scope`.  The page API has no notion of an
edge; the only "edge" in the substrate is the shared clock (temporal order),
which says WHEN, never FROM WHAT.  The master list's Source/Record/Consumed
test can therefore be satisfied by a write that records no cause; BOOK
status there is not evidence of an edge.

Detached pages: `IdentityPage("planning_value_concordance")` is constructed
outside any book in `precompile_to_ssa.py` (one site, when no page is
passed in) and in tests.  Such a page has its own clock, so its stamps are
not comparable with the book's.

External primitive call sites (scale only; Part B and the companion
censuses own the detail): `.concord(` fortran_c_shell 34, precompile_to_ssa
8, glsl_deployment_strategy 6, topological_reducer 6, tensor_ssa_lowering
3, transformation_priority 2, loop_composer 2, ssa_call_input_adapters 1,
ir_identities 1.  `.revise(` transmogrifier/ssa 10, topological_reducer 6,
precompile_to_ssa 2, loop_composer 2, transformation_priority 1.
`.bind_alias(` precompile_to_ssa 4, fortran_c_shell 3.  `.mapping(`
fortran_c_shell 8, python_special_cases 1, python_identity_programs 1,
graph_express2 1, topological_reducer 1.  `.set(` on a page: at least 14
multi-line sites in fortran_c_shell plus wasm_html_shell 2,
tensor_ssa_lowering 2, ssa_call_input_adapters 1 (single-line `page.set(`
sites are hidden among unrelated `.set(` calls and were not separated by
this census).  `current_identity_book()` is called from 16 files.
69 distinct page-name literals exist across `src` (plus two variable-named
sites in `topological_reducer.py`, variable-named descriptor/member pages
in `transmogrifier/ssa.py`, and the `shape.{name}` family).

## 2. Helpers that wrap the primitives

EDGE column: YES = the written fact or row names the exact source row(s)
AND the stage it derives from; PARTIAL = names a source id or stage but not
both / not the source page-row; NO = a bare fact.
FALLBACK column: can the helper record a default or an absence as if it
were a decision.
SHADOW column: what the helper writes OUTSIDE the book that restates or
decides identity (see section 3 for the consolidated list).

| helper | pages written | row key shape | fact shape | EDGE? | fallback-as-fact? | shadow ledgers | readers (outside this module) |
|---|---|---|---|---|---|---|---|
| `record_shape_transformation` | `shape_transformation_concordance` (edge), `shape_transformation_dependents` (reverse index), `shape_transformation_state` (projection) | edge: `(target_scope, target_id, source_scope, source_id, stage, operation, role)`; dependents: `((source_scope, source_id), edge_row)`; state: `(target_scope, target_id)` | edge: `(source_state, target_state)`, each state the 5-tuple `(shape, dtype, rank, metadata_state, sequence_row_shape)` or None; dependents: `True` at column 0; state: `("resolved", target_state, edge_row)` | YES -- the state fact carries the edge row; the edge row carries source identity + stage + operation + role. Caveat: `source_scope/source_id` are free-form graph identities (`("ProgramABI:<identity>", (identity, field))`, `("return", callee)`, `("sequence_column", seq, col)`), not a row on another page. A `source_state=None` edge "carries no derivation claim". | Partly: `shape_transformation_state` normalizes a missing dtype to `"unknown"` and a missing metadata_state to `"static"` inside the recorded tuple. `"unknown"` is treated as no-claim downstream; `"static"` is not. | none | fortran_c_shell, glsl_deployment_strategy, tensor_ssa_lowering (+ the two `publish_*` below) |
| `withdraw_superseded_shape_derivations` | (via the three invalidate_* helpers) `proven_shape`, `sequence_row_layout_concordance`, `shape_transformation_state` | see those | `("invalidated", source_id, reason)` | PARTIAL: names the upstream source id and a `reason` (= stage string), not the source row/page. It WALKS `dependents` (the cause chain exists) but does not write the traversed edge into the withdrawal fact. | No (only withdraws). | none | transmogrifier/ssa |
| `invalidate_shape_transformation` | `shape_transformation_state` | `(scope, value_id)` | `("invalidated", source_id, reason)` | PARTIAL | No | none | internal only |
| `commit_sequence_row_layout` | `sequence_row_layout_concordance` | `(scope, int(sequence_id))` | `(column_shapes, column_dtypes, source)`; `source` a free string | PARTIAL: `source` is a stage label, no source row. Raises on shape disagreement; merges `"unknown"` dtypes with known ones. | Yes: `"unknown"` dtype columns are WRITTEN as facts (canonicalized from None/""); a later known dtype silently fills them. | none | fortran_c_shell, glsl_deployment_strategy, tests |
| `invalidate_sequence_row_layout` | `sequence_row_layout_concordance` | `(scope, int(sequence_id))` | `("invalidated", source_id, reason)` | PARTIAL | No | none | internal only |
| `concord_sequence_row_dtypes` | `sequence_row_dtype_concordance` | `(scope, int(sequence_id))` for EVERY id in `claims` | `(resolved_dtypes, source)` | PARTIAL: `source` label only. Equates several sequence ids to one row layout but the fact does not say which sibling ids it was resolved with. | Yes: `"unknown"` written as a column dtype fact; `next(iter(known), "unknown")`. | none | precompile_to_ssa, tests |
| `commit_sequence_contract` | `sequence_contract_concordance` | `(scope, int(sequence_id))` | `(policy, column_count, writable, source)` | PARTIAL: `source` label. `writable` OR-merged with the incumbent (monotone). | Mild: `writable=False` is recorded when the caller has no proof either way. | none | fortran_c_shell, precompile_to_ssa, tests |
| `declare_loop_scope` | `loop_scope` | `(function, loop_node_id, "boundary")` and `(function, loop_node_id, row_key)`; `row_key` = `outer_id` or `(outer_id, source_bindings or ("entry", ordinal))` | boundary: `(header, latch, exit)` at col 0; rebind row: col OUTER=outer_id, CARRIED=carried_id, INNER=inner_id, col 3 `("graph", graph_outer, graph_inner)`, col 4 `("bindings", source_bindings)` | PARTIAL: the columns ARE the generation edge outer->carried->inner and the graph ids are named, but no stage; raw `page.set`, so a re-declaration overwrites the same cells silently. | Yes: `("entry", ordinal)` is a key minted when no source bindings exist. | none | precompile_to_ssa, tests, tools/compiler_probes |
| `rebind_loop_scope_inner` | `loop_scope_inner_transition` | `(function, loop_node_id, declared_inner)` | `(resident_inner, reason)` | PARTIAL: names the declared id and a reason string; not the `loop_scope` row/column it transitions. | No | none | precompile_to_ssa, tests |
| `record_proven_shape` | `proven_shape` | `(function, int(value_id))` | `("proven", extents, dtype)` or `("conflicting", extents, dtype)`; column = causal `level` | NO: no source, no stage. Level is a depth integer the caller chooses. Empty extents refused (documented). | Yes: dtype defaults to `"float64"` when None and is written; `level=None` reuses the deepest column. | none | topological_reducer, fortran_c_shell, glsl_deployment_strategy, tensor_ssa_lowering, tests, probes (11 files) |
| `invalidate_proven_shape` | `proven_shape` (+ `shape_transformation_state`) | `(function, int(value_id))` | `("invalidated", source_id, reason)` | PARTIAL | No | none | glsl_deployment_strategy |
| `concord_loop_scope_latch_residents` | via `rebind_loop_scope_inner` | as above | `(resident, "completed_module_latch_projection")` | PARTIAL (fixed stage string; the Phi instruction that proved it is not named) | No | `phi_argument.accounting["loop_scope_inner_transition"]`, `module.metadata["loop_scope_inner_reconciliations"]` | fortran_c_shell, tests |
| `concord_compiler_frame_formals` | `formal_actual_concordance` (`concord`), `formal_storage_resolution` (`set`) | actual: `(callee, formal_id, caller, block, instr_index, position)`; resolution: `(function, formal_id)` | actual: `int(actual.id)` at col 0; resolution: `("compiler_frame_storage", ((caller, actual_id), ...))` | actual page: near-YES (row is a full callsite coordinate; no stage). resolution: PARTIAL (names caller+actual id, not the `formal_actual_concordance` rows it read). | No (skips when evidence is missing; the skip is unrecorded). | `formal.accounting["compiler_frame_storage"/"compiler_frame_sources"]`, `function.metadata["storage_formals"]`, `module.metadata["compiler_frame_formal_reconciliations"]` | fortran_c_shell, tests |
| `concord_program_abi_frame_transitions` | `program_abi_frame_transition` | `(function, formal_id)` | `(program_abi_record, field, retired)`; `retired` = `((key, old_value), ...)` | PARTIAL: before/after accounting in the fact; no stage, no source row. Concord semantics done by hand (`latest` then `set`, raise on differ). | No | rewrites `formal.accounting` (drops provisional keys), `function.metadata["storage_formals"]`, `module.metadata["program_abi_frame_transitions"]` | fortran_c_shell, tests |
| `materializing_binding_kind` | `argument_binding_resolution` | `(callee, formal_id, source_id)` | `("caller_storage", proposed_kind, "materializing_binding_kind")` | PARTIAL: names the deciding function, not the `argument_binding` row/column (callsite) that proved `caller_storage`. | The default branch returns `str(kind)` and writes NOTHING: the default is not recorded as fact (good) and not recorded as unresolved (gap). | none | fortran_c_shell, tests |
| `publish_program_abi_graph_identities` | book: `record_shape_transformation` (stage `program_abi_graph_publication`, op `getattr`, role = field name) | | | Book half: YES (edge). Graph half: NO -- a fixed-point over `parents` copies identity strings onto nodes; no book row, no edge. The receipt is a size/`id()` tuple, not a fact. | Yes: `str(field.get("dtype") or "float64")`; rank defaulted from shape length. | graph node `attributes["program_abi_record_identity" / "program_abi_sequence_row_identity" / "program_abi_indexed_value_identity"]`; `graph.graph["program_abi_identity_publication_receipt"]` | fortran_c_shell, glsl_deployment_strategy |
| `publish_projected_iterable_layouts` | book: `record_shape_transformation` (stage `projected_iterable_layout`, ops `call_result` / `sequence_row_column`) | | | Book half: YES. SSA half: NO (dtype/shape/accounting rewrite of every occurrence with no book row). | Yes: `str(result_dtype or "unknown")`; `descriptor.column_shapes or ((),)` assumes scalar columns. | `value.shape`, `value.dtype`, `value.accounting["sequence_id"/"sequence_length_value_id"/"tensor_metadata_state"/"sequence_row_shape"/"program_abi_rank"]`, `function.metadata["projected_iterable_layout_receipts"]` | fortran_c_shell |
| `concordant_alias_bindings` / `resolved_concordant_alias_bindings` | read only (`planning_value_concordance` + caller ledgers); raise on disagreement | | | reads | n/a; but their INPUT ledgers are the shadow ledgers of master list A6.3 (`function.metadata["value_aliases"]`, `output_identity_aliases`) | none | fortran_c_shell, precompile_to_ssa, tests |
| `loop_scope_declarations`, `committed_*`, `concordant_shape_transformation_state`, `proven_shape_of`, `proven_shape_contract_of`, `shape_store_report`, `render_identity_book`, `render_row`, `row_value_id`, `authored_function_name`, `descriptor_from_shape_transformation_state`, `shape_transformation_state` | read / presentation | | | | | | many (probes, shell, glsl) |

Counts for this module: 17 public defs that write to the book (the first
16 writer rows plus `mint_scope`).  Of those: 3 edge-carrying in the sense
the goal requires (`record_shape_transformation`; the two `publish_*`
functions only through it, and only for their book half); 10 PARTIAL (a
stage label or a source id but never a source page-row); 3 bare
(`record_proven_shape`, `declare_loop_scope`'s boundary row, `mint_scope`);
1 that decides without recording (`materializing_binding_kind`'s default
branch).

### `record_shape_transformation` as the model

Three pages, one write:

1. Edge page `shape_transformation_concordance`.
   Row = `(target_scope, target_id, source_scope, source_id, stage,
   operation, role)`.  The row IS the edge: target, source, stage, operator.
   Fact = `(source_state, target_state)`: the operator's input and output.
   Column = next global column on that page (append-only, "compile time").
   Written only if `latest(row) != fact` (an identical repeat is not
   re-appended).

2. Dependents page `shape_transformation_dependents`.
   Row = `((source_scope, source_id), edge_row)`, fact `True`, column 0.
   Pure reverse index: `scope_rows((source_scope, source_id))` answers
   "what was derived from this identity" without a scan.  This is what
   makes invalidation a graph walk.

3. State page `shape_transformation_state`.
   Row = `(target_scope, target_id)`.
   Fact = `("resolved", target_state, edge_row)`: the current projection AND
   the edge that produced it.  `revise`d (history kept).  When the projection
   changed, `withdraw_superseded_shape_derivations` walks dependents and
   writes `("invalidated", source_id, reason)` onto downstream state /
   proven / layout rows.

What generalizes: (edge row keyed by target+source+stage+operator) +
(reverse index keyed by source) + (state row whose fact points back at its
edge).  What does NOT yet generalize: the source side is a free-form
identity, not a row on another page; and the withdrawal writes
`source_id` + `reason` rather than the new unresolved state with the exact
edge that caused it.

The only other edge-shaped page in the system is `identity_transition`
(master list A3a/A3c: `("move", ...)`, `("retire", ...)`, `("fork", source,
cause)`, `("merge", resolved, cause)`), written in `topological_reducer.py`
and `precompile_to_ssa.py`.  It carries the operator and a cause string;
it does not carry a source page-row either.

## 3. Shadow ledgers written by this module

Places where a helper in this module writes identity facts beside the book
(restating a book fact, or deciding without one).  Every one of these is a
second authority a reader may consult instead of the page.

| shadow ledger | written by | book page it shadows |
|---|---|---|
| `formal.accounting["compiler_frame_storage"]`, `["compiler_frame_sources"]` | `concord_compiler_frame_formals` | `formal_storage_resolution` |
| `function.metadata["storage_formals"]` (append / filter) | `concord_compiler_frame_formals`, `concord_program_abi_frame_transitions` | `formal_storage_resolution`, `program_abi_frame_transition` |
| `formal.accounting` with provisional keys deleted | `concord_program_abi_frame_transitions` | `program_abi_frame_transition` (the `retired` tuple) |
| `module.metadata["compiler_frame_formal_reconciliations"]`, `["program_abi_frame_transitions"]`, `["loop_scope_inner_reconciliations"]` | the three `concord_*` module-seam functions | receipts restating page rows |
| `phi_argument.accounting["loop_scope_inner_transition"]` | `concord_loop_scope_latch_residents` | `loop_scope_inner_transition` |
| graph node `attributes["program_abi_record_identity"]`, `["program_abi_sequence_row_identity"]`, `["program_abi_indexed_value_identity"]` | `publish_program_abi_graph_identities` | NO page -- these identities exist only on the graph |
| `graph.graph["program_abi_identity_publication_receipt"]` (size + `id()` tuple) | `publish_program_abi_graph_identities` | none; an idempotence token |
| `value.shape`, `value.dtype`, `value.accounting[...]` on every SSA occurrence | `publish_projected_iterable_layouts` | `shape_transformation_*` (only the shape part is on the book; dtype and the sequence accounting are graph-only) |
| `function.metadata["projected_iterable_layout_receipts"]` | `publish_projected_iterable_layouts` | receipts |
| `module.metadata["identity_book"]` | `identity_book(module)` | the book handle itself; after `end_identity_book` this cache is the ONLY route to the compile's facts |
| detached `IdentityPage("planning_value_concordance")` | `precompile_to_ssa.py` (caller of this module's class) | `planning_value_concordance` on the book, with an incomparable clock |
| detached `IdentityBook(detached=True)` | `current_identity_book()` when no compile is open | the real book; writes land here and are never dumped |

Inputs this module READS from shadow ledgers rather than pages (audit and
alias helpers): `function.metadata["value_aliases"]`,
`["output_identity_aliases"]`, `["parameter_names"]`, `["storage_formals"]`,
`["closure_formals"]`, `["parameter_member_formals"]`, `formal.accounting[*]`,
`module.sequence_tables`, `module.record_tables`, `module.struct_table`,
`module.union_table`.  Master list A6.3 names `value_aliases` as open; the
others are the same class.

## 4. Id minting

`id_space.py`: an id is `serial | flags`.  Flags are independent bits:
`MINTED` (1<<61, "the compiler invented this value; it names nothing the
source declared"), `SHARED` (1<<60, more than one call frame holds it),
`TENSOR` (1<<59), `HISTORY` (1<<58), `HISTORY_COUNT` (1<<57).  Serial field
is 57 bits.  `compose(serial, flags)` validates ranges; `with_flag` adds a
bit; `is_legacy` = no flags (ProcessGraph `id()` addresses, raw
tensor-identity tokens, pre-migration monotonic values).  `describe`/`label`
render `minted|shared#<serial>` for the concordance.

`monotonic_ids.py`: `MonotonicIdSource.mint()` returns
`compose(next_serial, MINTED)` under a lock; `mint_block(n)`; `peek()`.
`GLOBAL_MONOTONIC_IDS` is a process-wide singleton (serial starts at 1e9,
kept "for the serial"; the flag is what distinguishes it now).

Call sites of `GLOBAL_MONOTONIC_IDS.mint*` per file: fortran_c_shell 66,
ssa_optional_values 8, precompile_to_ssa 7, ir_identities 7, ctypes_layout 3,
ssa_record_return_state 3, ssa_call_input_adapters 3, and one each in
tensor_ssa_lowering, ssa_primitive_lowering, ir_sequence_tables,
ir_indexing, hierarchical_plan, deployment_ssa_binding.  About 100 sites.

`with_flag(..., SHARED)` has no caller in `src` today (only its definition
and a docstring mention in `transmogrifier/ssa.py`); the SHARED bit is
described, not applied.

What a "spontaneous novel identity" is today: any `GLOBAL_MONOTONIC_IDS.mint()`
result.  The `MINTED` flag marks it.  Nothing at mint time records WHY: the
source takes no cause, writes no page, names no stage.  The only causal
information a minted id ever acquires is whatever a downstream writer
happens to put in a fact (`bind_alias(scope, minted, resident)`, a
`loop_scope_inner_transition` fact, an `identity_transition` fork).  The
reverse question, "which transform produced this minted id", has no page to
answer it.

`mint_scope(label)` is the book-side analogue for scopes: `scope_registry`
row `(label, serial)` fact `True` (master list A1, BOOK).  It records
existence and label, not cause.  Callers: fortran_c_shell 3
(`record_field_access`, `record_storage_alias`, `scheduled_call_argument`),
topological_reducer 2 (the fork site also writes `identity_transition`
`(forked, "scope") -> ("fork", source_scope, cause)` -- the one place a
scope's origin IS recorded as an edge; master list A3c), transmogrifier/ssa
1, transformation_priority 1.

Ids and scope serials are ephemeral per run.  A fact that stores a bare id
as its "source" is therefore meaningful only inside the same book; an edge
must be reconstructible from the book's own rows, which is what the one-api
design has to guarantee.

## 5. The audit: `CorrelationTable` + `concordance_report`

Build: one `ValueRow` per `(function, value_id)` from a FINISHED module.
Claims (`Claim(kind, key, source)`) are gathered from: `formal.accounting`
(`program_abi_field`, `program_abi_parameter`, `program_abi_keyed_owner`,
`linked_call_frame_storage`, `compiler_frame_storage`,
`projected_row_source_id`, `ssa_layout_kind`), `function.metadata`
(`parameter_names`, `storage_formals`, `closure_formals`,
`parameter_member_formals`, `value_aliases`, `output_identity_aliases`),
`module.sequence_tables` (roles handle/columnN/length/capacity/status/
live_flags), `module.record_tables`.  Claim kinds: abi-field, abi-parameter,
keyed-part, frame-storage, compiler-frame-storage, projected-row, parameter,
storage-formal, closure-formal, member-formal, alias-of, output-identity-of,
layout-type, sequence-member, record-field.  Every claim's `source` is a
shadow ledger name, not a page.

Finding kinds (module docstring plus the per-page checks):
multiple-definition, unaccounted-formal, conflicting-storage-claims,
descriptor-member-unknown, descriptor-member-shared,
helper-operand-outside-descriptor, duplicate-storage-across-call,
use-not-dominated, alias-target-missing, alias-not-concorded,
source-field-identity-disagreement, callable-identity-disagreement,
source-parameter-identity-disagreement,
source-precision-boundary-disagreement,
source-precision-operator-disagreement, operator-result-type-disagreement,
planning-alias-transition-disagreement, layout-type-unknown,
layout-member-unknown, layout-redeclaration, layout-derivation-invalidated,
binding-kind-disagreement, key-column-not-integral,
table-member-disagreement (master list A1), operand-position-orphan (master
list A3a), plus the loop-scope, stale-carried-read, undefined-operand,
tensor-reduction-domain, source-numeric-*, source-sequence-mutation,
source-callsite-activation, source-precision-region and
post-ssa-numeric-identity families.

A "finding" is one concrete disagreement between two records about one
`(function, value_id)`, or one page row whose own history is non-monotone
or discontinuous.  The audit "writes nothing back".

Book pages the audit READS (via `module.metadata["identity_book"]`):
`argument_binding`, `argument_binding_resolution`,
`callable_identity_concordance`, `layout_state`, `layout_supersession`
(through the struct/union tables' own book), `loop_scope`,
`loop_scope_inner_transition` (via `loop_scope_declarations`),
`operand_position_orphan`, `operator_result_type_concordance`,
`output_identity_concordance`, `planning_alias_transition_concordance`,
`planning_value_concordance`, `record_descriptor`, `record_member`,
`sequence_descriptor`, `sequence_member`, `struct_descriptor`,
`struct_member`, `union_descriptor`, `union_member`,
`source_callsite_activation_concordance`,
`source_field_identity_concordance`, `source_numeric_component_concordance`,
`source_numeric_intrinsic_concordance`,
`source_numeric_operator_specialization_concordance`,
`source_numeric_type_dependency_concordance`,
`source_parameter_identity_concordance`,
`source_precision_boundary_concordance`,
`source_precision_operator_concordance`,
`source_precision_region_concordance`,
`source_sequence_mutation_concordance`,
`tensor_reduction_domain_concordance`.  (32 pages.)

Pages written somewhere in `src` that the audit does NOT read (facts there
are unaudited): `shape_transformation_concordance`,
`shape_transformation_dependents`, `shape_transformation_state`,
`proven_shape`, `sequence_row_layout_concordance`,
`sequence_row_dtype_concordance`, `sequence_contract_concordance`,
`sequence_column_claims`, `formal_actual_concordance`,
`formal_storage_resolution`, `program_abi_frame_transition`,
`scope_registry`, `identity_transition`, `transformation_event`,
`transformation_decision`, `transformation_rejection`,
`lexical_read_binding`, `loop_carried_binding`, `loop_carried_entry`,
`loop_entry_state`, `loop_region_membership`, `loop_result_port_binding`,
`while_carried_test`, `formal_shape`, `value_shape`, `shape.node`,
`shape.linked`, `shape.ssa`, `formal_literal`, `proven_literal`,
`formal_parity`, `item_operand`, `consumer_operand`,
`call_argument_operand`, `callsite_argument`, `call_edge`, `call_record`,
`call_record_pair_concordance`, `callsite_return_specialization`,
`callsite_projection_specialization`, `record_parameter_value`,
`record_parameter_row_handle`, `record_field_demand`, `record_field_write`,
`record_forwarding_edge`, `record_storage_alias`,
`scheduled_call_argument`, `member_formals`, `frame_lease_link`,
`region_feed_consumer`, `region_capture_binding`, `region_value_dtype`,
`control_uniform_dtype`, `cross_function_references`,
`alias_application_concordance`, `aggregate_ledger`,
`scalar_kernel_operand_concordance`, `tensor_shape_enrichment`,
`ssa_call_shape`, `source_value_class_concordance`.  Roughly 60 pages,
including every A2 and A3 page the master list marks BOOK except
`identity_transition`'s orphan consequence.  `shape_store_report` reads the
`shape.*` + `proven_shape` family but is probe-only, not part of
`findings`.

Structural limits: the audit keys by `(function, value_id)`, so it can only
compare claims about one id; it cannot follow an edge from a minted id to
its origin because no page records one; and every page check is hand
written per page (no generic "does this fact name its source" check can
exist, because the substrate has no source slot to check).

## 6. Where this census differs from the master list

Stated per its own rules; none of these contradict a Part A status under
the Source/Record/Consumed test, but several show that test is not the
edge rule.

1. Rule "Scopes are minted by the compile's book ... never by process
   counters" (master list, working rules).  Scopes: true.  VALUE IDS: the
   process-wide `GLOBAL_MONOTONIC_IDS` counter mints every compiler-invented
   value at about 100 sites with no book write.  The master list has no row
   for value-id minting; it is the largest unrecorded identity source.
2. Rule "A missing row raises; no value-id fallback where the book is
   expected."  Inside this module the fallbacks are: `current_identity_book`
   silently creating a detached sink book; `PageMapping.setdefault` writing
   the default as a fact; `record_proven_shape` writing `"float64"` for an
   unknown dtype; `commit_sequence_row_layout` / `concord_sequence_row_dtypes`
   writing `"unknown"` columns; `shape_transformation_state` writing
   `"static"` for a missing metadata state; `declare_loop_scope` minting
   `("entry", ordinal)` keys.  None raises.
3. Rule "Every new record is taught to the audit."  About 60 of ~69 pages
   have no finding that reads them (section 5), including every A2 linker
   page and every A3 read/loop/binding page.  A1's
   `table-member-disagreement` and A3a's `operand-position-orphan` are the
   exceptions and are correctly described.
4. A1 `scope_registry` BOOK: agreed under the three-part test; but its fact
   is `True`, so it records existence, not the transform that forked or
   opened the scope.  Only the reducer's fork site (A3c) writes the cause,
   and on a different page.
5. A2 pages written via `PageMapping`: agreed BOOK; note that
   `__setitem__` skips a write when the value is unchanged and `__delitem__`
   writes `None` with no cause, so those pages hold state, not edges.
6. A3a `identity_transition` is described as the operand-position edge
   page.  Agreed; it is the second edge-shaped page besides
   `shape_transformation_concordance`, and neither names a source page-row.
7. Part B's static scan defines "touches the book" as any `.page`/`.concord`
   /`.latest` call.  By that definition all 17 writers here are BOOK; by the
   edge rule 3 are, 10 are partial, 4 are not.  The two definitions should
   not be conflated when the master list is next revised.

## 7. Verdict

Primitives that must be closed (made non-public or routed through the one
api): `IdentityPage.set`, `revise`, `concord`, `bind_alias`,
`PageMapping.__setitem__`/`__delitem__`/`setdefault`, `IdentityBook.page`
(free page creation is how ~69 unregistered row shapes came to exist),
`IdentityBook.mint_scope`, and `current_identity_book`'s detached-book
fallback.  Direct construction `IdentityPage("...")` outside a book must
also close.  `GLOBAL_MONOTONIC_IDS.mint()` must go through the same api so
a minted id is born with its transform edge.

Helpers that already satisfy the edge rule (target+source+stage+operator in
the row, state fact points at its edge, reverse index for invalidation):
`record_shape_transformation` and, through it, the book half of
`publish_program_abi_graph_identities` and
`publish_projected_iterable_layouts`.  Caveat: their source side is a graph
identity, not a book row.

Helpers that do not: everything else that writes.  Most load-bearing bare
or partial writers: `record_proven_shape` (11 caller files, no source, no
stage, defaults dtype), `bind_alias` (the planning identity authority; fact
is a resident id with no page/row/stage), `declare_loop_scope` /
`rebind_loop_scope_inner` (loop generations by raw `set`, synthesized
`("entry", ordinal)` keys).  `PageMapping.setdefault` and the `"unknown"` /
`"float64"` / `"static"` defaults are where absence becomes fact.

Minimal shape of the single api (not implemented; names are placeholders):

```
post(
    page: str,                      # must be a registered page (row shape declared once)
    row: tuple,                     # scope-first, validated against the page's declared shape
    fact: Any,
    *,
    stage: str,                     # the pass making the statement
    derived_from: tuple[Ref, ...] = (),   # Ref = (page, row, column) -- exact source cells
    minted: Mint | None = None,     # Mint = (transform: str, operands: tuple[Ref, ...])
    mode: "concord" | "revise",
) -> Ref
```

Admission rule: exactly one of `derived_from` (non-empty) or `minted` must
be given; otherwise refuse.  `minted` asserts spontaneous novel identity: it
allocates the id (`compose(serial, MINTED | flags)`) and writes the
`identity_transition`-style edge `(row) -> ("mint", transform, operands)`
in the same clock tick, so later stages can re-identify the value if
topology changes.  `derived_from` writes the fact AND one edge row per
source cell on a shared edge page `(target_page, target_row, source_page,
source_row, source_column, stage)` plus the reverse index (the three-page
shape of `record_shape_transformation`, generalized), and the state fact is
`("resolved", fact, edge_ref)`.  A caller that has only a default writes
`post(..., fact=UNRESOLVED(reason), derived_from=(whatever it did read,))`,
never the default; readers treat `UNRESOLVED` as absence.  `mint_scope`
becomes `post(page="scope_registry", ..., minted=("fork" | "open",
operands))`.  `PageMapping` survives only as a read view; its
`__setitem__`/`setdefault` go away.  Invalidation becomes `post(...,
fact=UNRESOLVED("superseded"), derived_from=(the changed source cell,))` so
a withdrawal is itself an edge, not an erasure.  The shadow ledgers of
section 3 become read views of their pages or are deleted.

The audit then needs one generic finding instead of a check per page: a
fact on any registered page with neither an inbound edge nor a mint record
is `unsourced-fact`; a `MINTED` id with no mint record is
`unsourced-identity`.
