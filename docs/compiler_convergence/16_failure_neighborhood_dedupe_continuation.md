# Failure neighborhoods, pursuit dedupe, batch sort: continuation

Date: 2026-10-16 (work done 2026-10-15 night through 2026-10-16 morning)

This is the continuation receipt after
[15_provenance_sort_glyph_bootstrap_continuation.md](15_provenance_sort_glyph_bootstrap_continuation.md).
It records what was built, every bootstrap run in order with its evidence,
what is verified and what is not, and the decisions that are still open.

## Instrument: failures now carry their concordance neighborhood

Before this work a refused post printed one row, `recorded=A, proposed=B`, and
the failed identity log was forced to `summary` by the bootstrap tool (1 KB of
page counts while the book held ~390k edges).  Now:

- `IdentityBook.post` (`identity_concordance.py`) is a thin wrapper over the
  unchanged body `_post`.  Any `ValueError` leaving it keeps its type and
  message and gains a `FailureClaim` on `error.concordance_claims`: page, row,
  mode, stage, proposed fact, and seed cells (the row's incumbent cell plus the
  proposal's Derived sources / Novel operands).  Nothing is walked at raise
  time, because several callers catch `ConcordanceRefusal` as control flow.
  The type is deliberately NOT changed: those callers would swallow
  disagreements.
- `IdentityBook.neighborhood(seeds, radius=2, max_nodes=40, fanout=6)` diffuses
  one unit of mass over the book's own edges in both directions
  (`edges_into`, `edges_out_of`, `mint_of`).  Each hop passes half a cell's mass
  equally to the `fanout` newest neighbors by clock stamp and reports how many
  it left out (`+N neighbor(s) not shown`).  `CellNeighborhood.lines()` renders
  cells (`n0 [seed incumbent] mass=...`) and edges (`n3 --stage--> n0`).
- `annotate_failure_with_neighborhood(error, book)` walks the exception chain,
  renders every claim's neighborhood, and adds it ONCE as an exception note.
- The compile `finally` in `fortran_c_shell.py` (the same place that already
  calls `_dump_identity_book_log`) runs it for a failed compile.  The note
  reaches `compartment.json`'s traceback, and the lines also go into the failed
  log through `extra_lines`, so the bootstrap's `summary` level no longer
  blanks them.
- `_unsourced_findings` now lists up to three exact rows per group
  (`; e.g. row | row`).  The sort dtype refusal names both dtypes and both
  sequence ids.

Tests: `tests/test_failure_neighborhood.py` (3).  Failures raised outside
`post` (audit `Finding`s, reducer errors) carry no claims yet; they would have
to declare seeds the same way, not have them recovered from message text.

## Fixes made from the concordance

1. **Optional-glyph carry key (bootstrap run 2).**  `source_optional_glyph_carry`
   rows were keyed `(value_class_scope, phi)`, the shared authored scope, but
   since `6d3450ea` the fact embeds copy-local glyph cells
   (`presence_source`).  Two retained copies posted the same key with different
   cells.  The row is now keyed by the copy's canonical glyph scope
   (`graph["source_glyph_sequence_scope"]`); `topological_reducer.py` discovers
   inherited carries there and `glsl_deployment_strategy.
   _post_source_optional_glyph_carries` scans both scopes.
   `test_two_arm_return_phi_publishes_optional_glyph_contract` pinned the
   authored-scope key; it now finds the carry through the phi's
   `source_optional_glyph_carry_row` attribute and asserts the glyph scope.
   REVIEW THAT ASSERTION CHANGE.
2. **Pursuit dedupe (your decision: do not lay material on material).**
   `graph_express2.py`, source pursuit.  `exact_existing` matched only
   definitions that pursuit itself had discovered (`_python_source_identity`).
   Definitions from the submitted source carry only the owner visitor's
   `(module, qualname)` stamp, so a call resolving to the live object
   re-ingested the same authored function via `inspect.getsource`
   (relative lines: `ClassDef line=1`).  The predicate now uses
   `_source_definition_identity(definition) == identity`, and the lookup also
   tries the target's own name (a `cls(...)` call spells `cls`, its target is
   `Table`).  Measured by the user's own bootstrap run afterwards:
   re-ingested definitions 175 -> 65, compilation units 478 -> 296,
   compilation-units stage 5m31s -> 2m41s, blocking failures 4 -> 2, and the
   `proven_shape` raw-row blockers (units 003/049/050) disappeared.
   Open risk: a rebuilt copy used to carry the live module's bindings; they
   are now merged onto the authored definition.  If name-resolution/globals
   failures appear in functions of the same module, look here first.
3. **Batch-wide sort** (`lower_sequence_sort`, `ir_sequence_tables.py`).
   Replaced the scalar bottom-up merge (data-dependent branches per element,
   one mapping lookup per COMPARISON) with: a keys pass (one lookup per row), 
   `log2 n` merge passes each a flat loop of independent iterations that
   finds every element's output slot by a fixed-trip binary search using
   `Select` (stable; `reverse` keeps source order among equals), and a final
   batched row gather + copy-back.  It sorts a permutation, so the interface
   gained an `order` workspace: int64 permutation buffers `(a, b)` plus the
   mapped key per row when keys come from a mapping.  The only caller
   (`precompile_to_ssa.py`, sort mutation) allocates it.  Metadata algorithm
   is `batch_rank_merge_sort`.  Verified: a pure-Python model of the exact
   index arithmetic matches Python's stable sort (4000 random cases per
   direction, ties included) and all 45 tests in
   `tests/test_ir_sequence_tables.py` pass (two updated to the new interface).
   NOT verified: native execution.  `tools/benchmark_ssa_sort.py` compiles and
   executes and needs approval.

## Bootstrap runs

Instrument: `python -u -m tools.bootstrap_compiler_whole_program --provenance-only`
(default hard-fail admission).  Failure evidence is in the compartment receipt
under `build/<output>/products/provenance-core/<version>/compartment.json`
(`error`, `traceback` with the neighborhood note) and per-unit
`growth/product/units/unit_NNN/failure.json`.  `build/` is git-ignored.

| run | result |
| --- | --- |
| 1, 133 s | topology: `lexical_read_binding` disagreement `recorded='fact' proposed='FRAME_BINDING'` (consumer id was a memory address).  Not reproduced in isolation; did not recur.  Probably the duplicate-parse problem. |
| 2, 184 s | topology: `source_optional_glyph_carry` disagreement, `CorrelationTable._graph_work_publication_findings`.  Fixed (item 1). |
| 3, 538 s | PAST topology and compilation units.  10 of 478 units admitted, 4 blocking: units 003/050/049 `proven_shape` raw rows at the dependency-link check (`project_compilation_product.py:6517`); unit 033 sort dtype. |
| 4, 199 s | topology: `source_literal_value` disagreement for `CorrelationTable._function_findings` value 94 (`'defined '` vs `'multiple-definition'`): two parses of one function.  Fixed by item 2. |
| user run after item 2 | 296 units, 2 blocking failures: unit 026 `IdentityPage._fork_of_scope` partial; unit 031 `IdentityPage._reindex` sort dtype. |

## Open items

1. **Unit 031, `IdentityPage._reindex`**, `columns.sort(key=self.column_positions.__getitem__)`.
   Refusal: sort row key dtype `'unknown'` vs mapping key `int64`.  This is NOT
   a missing dtype claim.  The graph (replayed read-only) shows value 2,
   `columns`, as a parentless `Input` with `binding_kind='loop'`, the element
   of `For` over `values(GetAttr row_columns)`; `row_columns` is
   `dict[Any, list[int]]`.  It is not in `sequence_declarations` (15 are
   declared, not 2), has no resolved schema, and
   `_sequence_column_dtype_contracts` (`fortran_c_shell.py:13883`) only types
   declared sequences.  The nested-table survey
   (`fortran_c_shell.py` ~8240-8275) only admits an iterated base whose
   `binding_kind` is `"parameter"`.  And `lower_sequence_sort` rejects
   `child_table_pool`, so sorting lists nested in a field's dict is an
   unsupported FEATURE: nested child-list iteration plus sort over child lists.
   Decision needed: build it, or say where nested field-dict iteration should
   come from.  Do not rewrite the authored source of `_reindex`.
2. **Unit 026, `IdentityPage._fork_of_scope`**, partial: shortfalls
   `membership-sequence` / `call-result-unavailable` at node 23, from
   `name in READ_THROUGH_EXCLUDED_PAGES or name in book.registry.private_pages`
   (membership against a module-level collection and an attribute collection).
   Already listed as partial in doc 15.
3. **`record_proven_shape`** writes through a raw `page.set` on an undeclared
   page (`identity_concordance.py` ~7342); the census
   (`docs/concordance_census/00_book_primitives_and_helpers.md:129`) lists it as
   an unsourced writer with 11 caller files.  It stopped blocking after the
   dedupe but only 6 units were admitted.  If it returns: migrate to `post`
   (declare the page; derive from the shape scope's registry cell plus the
   value's node cell; decide whether `level` becomes part of the fact,
   `("proven", extents, dtype, level)`, or `post` gains an explicit column).
   Readers: `proven_shape_of`, `proven_shape_contract_of`,
   `site_bundle.py:1817`, `glsl_deployment_strategy.py:18346`.
4. **Native verification of the new sort**: `tools/benchmark_ssa_sort.py`
   (compile + execute; needs approval).
5. **Memory**: a pytest session containing `tests/test_ast_parent_ingestion.py`
   ran past 10 minutes and a rerun held over 20 GB without compacting and was
   killed.  Unattributed (that module, an invented test contract, or the
   dedupe edit).  The module is SKIPPED by a module-level `pytestmark` with the
   reason in the file, by the user's instruction; remove it once understood.
6. The two older failing tests seen once and never baselined:
   `test_identity_book_spill.py::test_every_declared_page_is_classified_and_the_table_is_consistent`
   (14 declared pages missing from `page_lifecycle`) and
   `test_concordance_audit_contracts.py::test_audit_has_no_undefined_values_or_unposted_minted_ids[energy|toplevel]`
   (`incompatible physical call inputs`).

## Recipes and hazards

- **Single-unit replay (seconds)**, only while the frozen plan's toolchain is
  unchanged: `python -u -m tools.compile_project_catalogue
  --resolved-process-graph <growth>/resolved-process-graph.pkl
  --process-graph-plan <growth>/inputs/plan.json --planned-unit N --output
  <scratch> --linked-planned-unit '{"unit_index": D, "root":
  "<growth>/product/units/unit_DDD"}'` (dependencies from the unit's
  `dependency_units`).  Use Windows-style paths (`cygpath -m`); `/c/...` is
  mangled by Python.  ANY edit to a toolchain file afterwards makes it refuse
  (`compiler toolchain changed after ProcessGraph planning`), so edit-then-
  replay needs a fresh bootstrap.  This session's edits to
  `ir_sequence_tables.py` / `precompile_to_ssa.py` mean the next check is a
  fresh run.
- Wrapping `lower_sequence_sort` / `_sequence_column_dtype_contracts` with a
  read-only printer in a runner script is how the graph around value 2 was
  read; nothing in the repository was patched to do it.
- Working-copy files are CRLF.  Edit with the Edit tool, or preserve CRLF in
  scripts (a script that wrote LF was normalized back, but do not rely on it).
- Do not build ad-hoc `ExtractionContract`s (`with_sources`,
  `with_execution_file`) in tests or probes: that declares modules authored and
  changes what pursuit may ingest.  An invented one in a dedupe test was
  killed by the user and removed.
- Clear `build/compiler-logs`, `build/compiler-partitions` (and the output
  directory) before each whole-program observation; one compile at a time.

## Uncommitted before this commit

Everything above was uncommitted until this document's commit.  Nothing was
pushed.
