# Concordance edges, lane A (2026-10-03)

Three decisions committed today without an edge, each to become a posted row:

1. `ir_indexing._propagate_scalar_dtypes` (ce587b82): Const keeps its declared
   integer width; only inferred integers widen to int64.
2. `fortran_c_shell` `note_shape` (51b4cebe): copies stating different shapes
   flag the authored row "polymorphic"; the completed-module seam skips it.
3. `ssa_record_return_state.scalar_return_field_versions` (408155a7): a
   version recovered without a book record is admitted by a Const /
   carried-Phi rule.

## 2026-10-03 baseline (HEAD + other lanes' uncommitted symbolic_* edits)

`python tools/audit_identity_concordance.py` (system Python 3.11; the
`.venv` lacks `yaml`). Findings / unsourced facts / unsourced identities /
`[unsourced-fact]` page groups:

| case | findings | unsourced facts | unsourced ids | unsourced-fact groups |
|---|---|---|---|---|
| view | 0 | 4083 | 0 | 57 |
| toplevel | 1 | 2831 | 10 | 46 |
| energy | 0 | 3800 | 10 | 48 |
| controller | 1 | 4546 | 1 | 53 |
| controller_untyped | 5 | 4431 | 1 | 52 |
| mapping | 0 | 197 | 0 | 15 |
| oscillator | 0 | 2952 | 0 | 29 |

Baseline gates (same tree, before lane A edits): record_return_site_identity
14 passed / 1 xfailed; ssa_record_return_state 20 passed; native_scalar_loss_adjoint
2 passed; graph_reverse VJP 1 passed; linear motion 1 passed.

Note: the audit's `unsourced: N fact(s)` total is not deterministic run to run
(view 4083 -> 4022 with no change on its pages; the swing is in
`loop_result_reconciliation`).  Compare per page
(`scratchpad/audit_groups.py` prints `_unsourced_worklist` groups).

## Decision 1 (2026-10-03): `scalar_integer_width`

Row `(function_scope_of(function), ssa_id)`, fact
`IntegerWidthFact(decision, source_dtype, dtype)`, stage
`scalar_dtype_settlement`, mode CONCORD (the pass runs at each section and
again at the final module seam; re-posts are the same fact).

- KEPT_DECLARED: DERIVED(the Const's `ssa_value` cell).  The cell's
  `SSAValueFact.dtype` must equal the instruction's declared width; a
  disagreement raises (measured: 26 Consts in linear motion + scalar loss,
  all agree).
- WIDENED: NOVEL(`inferred_integer_widen`, (the value's `ssa_value` cell,)).
- No `ssa_value` cell: the decision keeps the old behaviour and the row is
  `Unsourced(const_declared_width_not_on_book)` (kept) or
  `Unsourced(widened_value_not_on_book)`.

Finding: the uncelled declared Consts are the repository kernels' `i32`
`llvm_literal` Consts (`binary_value`, `unary_double`, `broadcast_double`,
`reduce_dim_double`, ...: 80 in linear motion), plus 8 `int` literals in
`training_motion__forward_loss_backward`.  The LLVM repository importer
(`src/common/tensors/accelerator_backends/llvm_repository_ssa.py`
`_register_constant`, backend lane) mints them with no `ssa_value` row.
Making the width decision visible therefore adds unsourced cells on the new
page (energy 62, view 52) until the importer posts its Consts' identities.
A first draft widened uncelled Consts instead (reader treats absence as
"not declared"); linear motion still passed, but that silently returns 80
kernel literals to the pre-ce587b82 width, so it was not kept.

## Decision 2 (2026-10-03): `copy_value_shape` + `value_shape_polymorphism`

`fortran_c_shell._class_surface_ssa_program`, the ABI settlement's
`note_shape`.  The private `shape_row_writers` (keyed by `id(owner)`) and
`cross_copy_polymorphic` set are gone.

- Each statement is the stating copy's own row on `copy_value_shape`:
  row `(copy scope, value_id)`, fact `CopyShapeFact(label, shape, dtype,
  storage)`, stage `linked_value_abi_settlement`, mode REVISE.  Copy scope
  = the graph's `lexical_read_scope` (cells on `canonical_value`); a
  graph-native reverse (`scalar_loss_join_reverse`, lane B) has none, so its
  `ingestion_value_scope` (cells on `ingestion_value`).  Neither: raise.
  DERIVED from the value's identity cell in that copy and, for a statement
  carried over a call edge (`source=(caller graph, caller id)` at the three
  propagation sites), the caller's statement cell, else the caller value's
  identity cell.  No cell at all: `Unsourced(shape_statement_source_not_on_book)`.
- Polymorphism: row `(authored function, value_id)` on
  `value_shape_polymorphism`, fact `ShapePolymorphismFact(shapes)`, DERIVED
  from the disagreeing statement cells, REVISE (grows only when a new shape
  joins).  Cross-copy: a new statement whose shape differs from another
  copy's latest statement.  Within-copy (two callsites of one copy, the older
  `linked_value_abi_polymorphism` path): the callee copy's standing
  statement + the new caller's statement.
- The completed-module seam skips a value iff the polymorphism row exists;
  the "polymorphic" label check is gone.  `value_shape` still gets its
  "polymorphic" label, written after the post, for
  `tensor_ssa_lowering._shape_polymorphic_function` (not lane A's file:
  it should read `value_shape_polymorphism` next).
- Behaviour note: after a within-copy polymorphism is posted, later concrete
  statements for that authored row are no longer written to `value_shape`
  (before, only the cross-copy case was suppressed, so a later `merged
  from` statement could overwrite the label and get materialized).

Measured: linear motion posts 34 statements (all DERIVED) and one
polymorphism, `('unbroadcast', 0)` shapes ((2, 3), (3, 2)), derived from
`copy_value_shape` rows under `lexical_reads:unbroadcast|fork` 35 and 36:
the 51b4cebe case, now on the book.  Linear motion + scalar loss 2/2 pass.

## Decision 3 (2026-10-03): return-site versions found without a book record

`ssa_record_return_state.scalar_return_field_versions`, `lookup`.

- Scoped graph (`lexical_read_scope` present), version not published (no
  `ssa_field_version` cell in `read`): `publish_joined_version` walks back
  from the SSA value's `ssa_value` cell along `edges_into` and `mint_of`
  operands (8 steps) to each read `return_site_field_state`'s
  `FieldState.value` cell.  Every state joins: post `ssa_field_version`
  row `(scope, field-state cell)` = value id, DERIVED(field-state cell,
  value cell), stage `record_return_version`, CONCORD; the selection is then
  a published version (its `record_return_field_selection` row derives from
  it as usual).  Any state fails to join:
  `Unresolved(version_not_on_book)` on the selection row (reason appended to
  `RETURN_VERSION_REASONS`), the fallback is kept.  The Const /
  carried-Phi admission is gone from this path.
- OPEN, held for the lead: a graph with NO reduction scope still uses the
  Const / carried-Phi rule, because no row can be keyed (plan 70 section 6,
  "decides as it always did and the book records nothing").  Only the
  book-less synthetic graphs in `tests/test_ssa_record_return_state.py`
  reach it; removing the rule fails 8 of its 20 tests by design (they assert
  receipt-view selections with no book).  Either those tests get book
  fixtures or the rule goes and they are rewritten as refusals.
- Measured (temporary counter, removed): in record_return_site_identity +
  scalar loss, every admission reached is already published (121 lookups:
  42 Const, 37 Load, 42 formal-at-entry); the 7 `record_return_version`
  `ssa_field_version` posts all come from `observed_formal`.  The new join
  path is not exercised by any gate; the audit cases reach no admission.

## Gates after all three (2026-10-03)

Audit findings per case (before -> after): view 0->0, toplevel 1->1,
energy 0->0, controller 1->1, controller_untyped 5->5, mapping 0->0,
oscillator 0->0.  Unsourced facts: view 4083->4074 (noise, see above),
toplevel 2831->2833, energy 3800->3862, controller 4546->4604,
controller_untyped 4431->4489, mapping 197->197, oscillator 2952->2952.
Every increase is `scalar_integer_width` cells (2 / 62 / 58 / 58, view 52):
the kernel `llvm_literal` Consts.  `copy_value_shape` and
`value_shape_polymorphism` add zero unsourced cells.

Tests: record_return_site_identity 14 passed / 1 xfailed (one run in
between showed 6 failed + 8 errors in 45 s while other lanes were editing the
tree; two reruns green, not chased); ssa_record_return_state 20 passed;
native_scalar_loss_adjoint 2 passed; graph-reverse VJP 1 passed; linear
motion 1 passed.

## Coordinator round (2026-10-03, later): sources fixed, rule removed, label dropped

1. **Repository-kernel identities.**
   - `llvm_repository_ssa._FunctionImporter.fresh` records each value's
     (position, kind, LLVM spelling) at import.
   - `import_function` leaves `metadata["llvm_repository_kernel"] =
     (sha256 of the kernel's LLVM text, origins)`.  This is provenance only;
     no decision reads it.
   - The import is `lru_cache`d process-wide (`c_backend_repository_ssa_reference`),
     so the posts happen per book in `post_repository_kernel_identity`.  It
     is called by `_propagate_scalar_dtypes` for every function before any
     width decision; it no-ops for non-kernels and for a kernel already
     posted in this book.  It posts:
     - `repository_kernel_definition` row `("llvm_repository", kernel)`,
       NOVEL(INGEST_SOURCE), fact = digest;
     - one `repository_kernel_value` row `(root row, position)` per value,
       DERIVED(root);
     - the value's `ssa_value` row DERIVED(that cell).
   - KEPT_DECLARED for every kernel literal now derives from a real cell.
   - The link site (`tensor_ssa_lowering` ~5786,
     `module.functions[name] = function`) would be the more natural caller.
     Not edited: that file is the name-arm lane's.
2. **The 2 remaining toplevel cells were frame-linker mints.**
   - `_class_surface_ssa_program`'s aggregate unpack minted the output-index
     `Const int` and its element address with bare ids.
   - Both are now `mint_compiler_value_id(caller, FRAME_SCAFFOLD,
     (callee output's ssa_value cell,))`.
   - toplevel unsourced identities drop 10 -> 6.
3. **Scope-less rule removed.**
   - A version without an `ssa_field_version` cell is admitted only by the
     book join (`publish_joined_version`); otherwise
     `Unresolved(version_not_on_book)` and the fallback.
   - The formal check still applies to joined versions; only versions the
     book published before the lookup skip it.
   - `tests/test_ssa_record_return_state.py`:
     - an autouse fixture begins a book per test;
     - `post_return_site` posts the site's `source_span` root, the
       `canonical_value` cells, `reducer_field_state`,
       `return_site_field_state` and the versions' `ssa_value` rows;
     - return edges carry `return_site_cell`.
   - Test 1's last assertion (an edge naming no site among equal-slot
     sites) drops the stamp.  Test 2 restamps the edge to the other site.
     The loop test stamps its new return edge.
   - Two new tests: without a book the receipt is not a selection; a joined
     receipt publishes its `ssa_field_version`.  22 passed.
4. **Label dropped.**
   - `note_shape` takes `callsites_disagree=True` (the cross-callsite site)
     instead of a "polymorphic" fact label.
   - A proven-polymorphic row gets no further `value_shape` entries.
   - `tensor_ssa_lowering._shape_polymorphic_function` reads only
     `value_shape_polymorphism`.  It no longer reads
     `linked_value_abi_polymorphism` (every such proof also posts the
     row), nor the "two shapes in one row's history" heuristic.  Its blanket
     `except Exception: return False` is gone.

Gates:
- Audit findings 0,1,0,1,5,0,0 (unchanged).
- Unsourced facts per case:

| case | after | baseline |
|---|---|---|
| view | 4022 | 4083 |
| toplevel | 2831 | 2831 |
| energy | 3800 | 3800 |
| controller | 4546 | 4546 |
| controller_untyped | 4431 | 4431 |
| mapping | 197 | 197 |
| oscillator | 2952 | 2952 |

- Unsourced-fact groups 57/46/48/53/52/15/29 = baseline.
- Unsourced identities: toplevel 6 (baseline 10), the rest equal.
- Linear motion: 356 KEPT (all DERIVED), 12 WIDENED (all NOVEL), 0
  unsourced width rows.
- Tests: ssa_record_return_state 22 passed; record_return_site_identity
  14 passed / 1 xfailed; native_scalar_loss_adjoint 2 passed; graph-reverse
  VJP passed; linear motion passed.
