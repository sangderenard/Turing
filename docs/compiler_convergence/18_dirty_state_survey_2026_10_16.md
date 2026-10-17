# Dirty-state survey, 2026-10-16 evening

Taken at 18:53-18:56 CDT on `main`, after commit `f503ba22` (failure
neighborhoods, pursuit dedupe, batch sort; doc 16).  Everything below was found
uncommitted in `turing/` by another worker's session and is committed here as a
WIP checkpoint.  This survey did not author any of it.  Read the changes only
by stat, by the worker's own doc 17, and by targeted diffs of the items that
are authority files or are not described in doc 17; the two large files
(`fortran_c_shell.py` +4462, `precompile_to_ssa.py` +2686) were NOT read.

## The tree does not import (read this first)

`import src.transmogrifier.ssa` fails:

```text
src/transmogrifier/ssa.py:829  unions: Dict[int, SSAValueUnionDescriptor] | None = None
NameError: name 'SSAValueUnionDescriptor' is not defined
```

`SSAValueUnionTable` (class at line 824) annotates with
`SSAValueUnionDescriptor`, which is defined at line 2173 (`SSAValueUnionStorage`
2102, `SSAValueUnionAlternative` 2114).  The module has no
`from __future__ import annotations`, so the annotation is evaluated when the
class body runs.  `ssa.py` was edited at 18:48 and `precompile_to_ssa.py` at
18:49, so this is most likely mid-edit.  Everything that imports
`identity_concordance` (which imports `ssa`) fails to import, which is why the
test gate could not run here: `tests/test_identity_book_spill.py` fails at
collection.  The fix is a forward-reference change (quote those annotations at
lines 829, 856 and 860, or move the class); it was not made because the file is
under active edit by someone else.  Doc 17's reported gates (132 / 193 / 345
passed) predate this state and were not reproducible now.

## What changed, by area

Counts are working tree vs `f503ba22`: 63 modified files (+15,234/-1,364) and
4 untracked.

1. **Memory-graph takeover** (doc 17, first section).  `lower_ast_source_to_ssa`
   now creates a materialized `ProcessGraph` whose topology is owned by
   `BitTensorMemoryGraph`; `NetworkxEmulation` no longer subclasses or contains
   `networkx.DiGraph`.  Files: `graph_express2.py`, `memory_graph/memory_graph.py`,
   `helpers/bit_tensor_memory.py`, `helpers/networkx_emulation.py`,
   `tests/transmogrifier/test_memory_graph_integer_allocation.py`.  Doc 17 states
   the topology RSS rose (about 3.12 GB vs 1.68 GB) and no compression win is
   established.
2. **Unit 026** (`IdentityPage._fork_of_scope`): set/frozenset field receipts are
   admitted; ProgramABI record-field receipts index nested receiver classes and
   set storage.
3. **Unit 031** (`IdentityPage._reindex`): the nested mapping-value sort is
   repaired along the exact graph path `mapping -> values() -> For target` using
   the reducer's existing `mapping_value_aggregate_kind` / `mapping_value_dtype`
   receipt.  This is the feature doc 16 left open (item 1).  The regression uses
   the unannotated target name `resident` on purpose.  The keyed-descriptor
   replacement now rebinds members before deleting the provisional descriptor.
4. **`proven_shape` is a declared page and goes through `post`** (doc 17, third
   run): publications, revisions and invalidations derive from the exact
   shape-state cell.  This closes doc 16 open item 3 as migration debt.
   Loop-continuation rewire receipts now flow into shape invalidation; authored
   `Any`/`object` boundaries use `opaque_ref`; resident row loads claim the
   provisional identity; tuple stream publication carries
   `aggregate_leaf_value_ids`.
5. **Rehydrated glyph references / typed receiver** (doc 17, fourth run):
   validation resolves a retained row through the active book and compares the
   full reference key; `_replace_cells(page: IdentityPage, ...)` carries its
   exact type so `{page.name}` takes the resident string-token path.  Spill
   lifecycle gained `proven_shape` plus fourteen previously unclassified pages
   (`page_lifecycle.py`), which is the failure doc 16 listed as open item 6.
6. **Runtime `!r` no longer calls CPython.**  This reverses a design reviewed
   earlier in this session.  `glyph_sequence_format.py` loses `runtime_repr_utf8`
   and gains `quoted_string_glyph_tokens` (an authored, lossless quoted
   spelling); doc 15's paragraph on the CPython boundary was rewritten to match.
   Unknown scalars stay unresolved instead of going to a host formatter.
7. **Authority-file changes: the extraction contract.**  NOT in doc 17.
   `extraction_contracts/program_extraction.yaml`:
   * removes rule `unresolved-formatted-repr-utf8-result` (the CPython admission
     above);
   * changes `native-os-replace-none-result` from the `cpython-c-api` ABI to the
     brokered `turing-file-broker-v1` (profiles `[c, fortran]`);
   * adds rules `native-os-remove-none-result`, `native-os-getpid-scalar-result`
     (`turing-system-v1`), `native-os-environ-get-text-result`
     (`os._Environ.get`, text arguments as unicode code points) and
     `native-lzma-text-writer` (`lzma.open`, `turing-liblzma-v1`).
   `extraction_contract.py`: under `require_full_native`, any `use_native`
   whose ABI is `cpython-c-api` or whose `shell_profiles` include `python` /
   `cpython-c` is now rejected (`full_native_requires_non_python_abi`), whether
   or not the callable is classified as a native extension.  Supporting changes:
   `shell_external_references.py` (+274), `cpython_runtime.py`,
   `tests/test_shell_external_references.py`, `tests/test_cpython_runtime.py`
   (25 lines removed).  The declared native ABIs `turing-file-broker-v1`,
   `turing-system-v1` and `turing-liblzma-v1` are new vocabulary; whether
   their runtime sides exist was not checked.
8. **Build-product retention that deletes directories.**  NOT in doc 17.  New
   `src/compiler/build_product_retention.py` (+`tests/test_build_product_retention.py`)
   with a byte-range lease (`compiler_product_lease`) and three pruners:
   `prune_stale_fingerprint_products` (keeps the active 64-hex product under
   `products/<boundary>/`, removes older ones with `shutil.rmtree`),
   `prune_stale_orbital_profile_products`, and
   `prune_abandoned_product_temporaries` (`.tmp*` files, `temporary-tiles|cache|spill`
   directories).  `whole_program_compiler_bootstrap.py` calls the fingerprint and
   temporary pruners at the start of a bootstrap, and
   `tools/compiler_probes/profile_memory_n2.py` uses it.  Consequence for
   debugging: a fresh bootstrap now removes the PREVIOUS fingerprint's product
   directory automatically, including the failure receipts that doc 16's replay
   recipe depends on.  Copy a failed run's `growth/` aside before starting
   another.
9. **Value unions** (new SSA type, NOT in doc 17).  `ssa.py` gains
   `SSAValueUnionStorage`, `SSAValueUnionAlternative`, `SSAValueUnionDescriptor`,
   `SSAValueUnionTable`, `SSAValueUnionFrameMember`; a new concordance page whose
   fact is an `SSAValueUnionDescriptor | None` (REVISE) in
   `concordance_declarations.py`; users in `precompile_to_ssa.py` and
   `fortran_c_shell.py`; test `tests/test_value_union_abi.py`.  This is the
   in-flight work that currently breaks the import.
10. Other: `ssa_self_check.py` (+114), `ssa_fortran_backend.py` (+223),
    `ssa_object_context.py` (+148), `ssa_result_type_resolution.py`,
    `ssa_reachability.py`, `ssa_reference_evaluator.py`, `hierarchical_plan.py`,
    `graph_work.py`, `loop_composer.py`, `control_source.py`,
    `resident_sequence_materialization.py`, `project_compilation_product.py`,
    `ir_sequence_tables.py` (+212; not diffed, so it is unknown whether the
    batch-wide sort from doc 16 was touched), and the matching tests
    (`test_glyph_sequence_provenance.py` +746, `test_sequence_contract_concordance.py`
    +625, `test_abstract_tensor_topological_reducer.py` +369).

## What this survey ran

* `py_compile` over all 63 modified and new `.py` files: ok (syntax only).
* `git diff --check`: clean.
* The new and own-area tests could not be collected because of the import break
  above (`tests/test_identity_book_spill.py` errors at collection with the
  NameError); nothing was run that would pass or fail on its merits.
* No bootstrap was run.

## Open / next

1. Fix the `SSAValueUnionDescriptor` forward reference, then rerun the focused
   gate in doc 17 and `tests/test_value_union_abi.py`,
   `tests/test_build_product_retention.py`, `tests/test_failure_neighborhood.py`,
   `tests/test_ir_sequence_tables.py`.
2. Decide the contract changes (item 7) deliberately: they remove a capability
   documented in doc 15 and add three native ABIs.
3. Decide whether automatic product pruning (item 8) should run on every
   bootstrap, given it deletes earlier failure receipts.
4. Items from doc 16 still open: native execution of the batch-wide sort
   (`tools/benchmark_ssa_sort.py`) and the unattributed 20 GB pytest session
   (`tests/test_ast_parent_ingestion.py` is skipped).
5. A fresh user-owned provenance bootstrap is the next authoritative
   observation (doc 17); this tree cannot run one until it imports.

## Follow-up, 19:00: import fixed; the union feature still fails its tests

The user's own bootstrap run on the checkpoint died at the import break above
(`bootstrap_compiler_concordance.py` -> `fortran_c_shell` -> ... -> `ssa.py:829`).
Fixed by quoting the five forward references to `SSAValueUnionDescriptor`
(`SSAValueUnionTable.__init__`, `register` parameter and return, `by_id`,
`SSAValueUnionFrameMember`).  The whole compiler import chain and the
bootstrap tool now import.  The new `tests/transmogrifier/test_memory_graph_integer_allocation.py`
also used `pytest` without importing it; fixed.

With imports working, the focused gate (the doc 17 modules plus the new ones)
still fails, and every failure traces to the unfinished value-union feature, not
to the memory-graph, unit 026/031 or retention work (those suites pass):

* `ValueError: type ABI alternative requires physical storage`
  (`concordance_declarations.py:3048`): some producer constructs a type-ABI
  alternative with an empty `storage`.  Raised in 10 tests of
  `test_sequence_contract_concordance.py` (it failed 10 of 37) and in
  `test_ir_sequence_tables.py` (2).
* `AttributeError: Can't pickle local object 'SSAValueUnionTable.__init__.<locals>.<lambda>'`
  (`fortran_c_shell.py:36215`): `SSAValueUnionTable` passes `_BookRows` a closure,
  so the value-union table cannot be pickled with the module metadata.  3 tests
  of `test_native_reference_boundary.py`.
* The lifecycle audit
  (`test_identity_book_spill.py::test_every_declared_page_is_classified_and_the_table_is_consistent`)
  fails again: `class_subtype`, `union_alternative_state`, `value_union_descriptor`
  and `value_union_member` are declared but absent from `PAGE_LIFECYCLE`.

Not touched: these are design decisions of the value-union feature (what an
alternative's storage is for each producer; how the table is made picklable;
when the new pages may be released).
