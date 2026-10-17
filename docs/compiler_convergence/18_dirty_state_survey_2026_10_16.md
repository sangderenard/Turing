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

## Follow-up, 19:05: source closure now fails on eight authored classes

The user's next run (`--provenance-only --output build/compiler-partitions`, tree
`d9cc04ba`) passed the import, then failed stage 2/11 (source closure, 1m15s):
`RuntimeError: AST parent ingestion did not reach a source-definition fixed point`
for constructor calls to `PageMapping`, `_PageTable`, `LiveScopeProjection`,
`ProjectedScopeRead`, `CellNeighborhood`, `NeighborhoodNode`, `Unresolved` and
`Stage`, all `missing_source_parent`, all defined in the submitted
`identity_concordance.py`.  The same call sites existed in the tree that passed
this stage earlier, so reachability, not new code, changed.

What the log and code show: the final linking pass in
`_expand_unresolved_ast_parents` resolves a Name call to a class only through
`target_definitions[identity]` (filled while pursuit walks reachable call sites)
or `definitions_by_source_identity` (only definitions pursuit discovered, keyed
by `_python_source_identity`).  `lexical_definition` matches functions only.  The
submitted source's own `ClassDef`s were never consulted, so a class whose every
call site sits in a method pursuit did not activate stayed unresolved (`Ref`
links; `Stage` at line 6088 fails even though `Stage(...)` at 4305 is the same
class).  Before pursuit dedupe a re-ingested duplicate happened to be registered
by some reachable site.

Change: the linking pass now also fills `definitions_by_source_identity` from the
stamped authored identity of every definition in the tree (`setdefault`; a
discovered definition keeps its identity).  UNVERIFIED: a probe module with the
same shape (a dataclass whose only construction is in an uncalled method) passes
with and without the change, so it does not reproduce the failure and this fix is
a hypothesis about the mechanism, not a demonstrated cure.  The authoritative
check is the closure stage of the real bootstrap (about 75 s).  If it still
fails, the open question is why pursuit no longer activates the methods that
construct these classes; compare `graph.G.graph["activated"]`-style activation
state against the earlier passing run.

## Follow-up, 19:40: pursuit holds definitions in the concordance

Principle (the user's): the concordance is the registry.  Putting material up
and pulling it down are asks of the book, with an unbroken chain of provenance;
code that keeps its own table beside the book is a regression, and the fix is to
route it through the book.  The two stopgaps above (a stamp-reading widening of
`exact_existing` in pursuit, and a `setdefault` pass in AST-parent linking) were
that regression and are removed.

What replaces them (`graph_express2.py`, `concordance_declarations.py`,
`page_lifecycle.py`):

* New page `source_definition_concordance` `(module, qualname) -> node kind`.  A
  definition is HELD when the book has its row.  The row is posted once,
  DERIVED from the definition's own `source_span` cell (`path == ()`), stage
  `PURSUIT`.  Lifecycle: CLOSURE, like the activation page.
* `target_definitions`, pursuit's identity -> definition table, is now a
  `_HeldDefinitions`: setting an entry is the one put-up path (it posts the row);
  `in`, `get` and `[]` ask the book first.  The node handle it keeps only lets
  pursuit reach the AST object the row stands for; it is never consulted without
  the row.  A second, different node for a held identity is refused
  ("duplicate material"); a property getter and setter, one qualified name,
  hold the first (`hold_authored`).
* The submitted text's own definitions are held when pursuit starts, with their
  seeded bindings, so the existing `identity in target_definitions` branch answers
  for them.  Authored classes also get their constructors demanded when a call
  first constructs them (`activate_construction_methods`), as pursuit already did
  for classes it discovered; the class's span is the demand.
* The final linking pass resolves through `target_definitions` as before, which
  now includes every authored definition.  `definitions_by_source_identity` and
  `definitions_by_name` remain as the old notebooks for discovered definitions;
  `unavailable_identities`, `activated_definitions` and `latent_call_ids` are
  still process-local although the book has an activation page.  Those are the
  next routings, not done.

Verification: `tests/test_held_definitions.py` (4): row + provenance edge to the
`source_span` cell, the book (not the handle) decides, duplicate refusal, property
pair.  A probe module compiled with the real contract in the bootstrap's shape (a
class built only in an uncalled method, `cls(...)`, a property pair) finished
without failure, discovered nothing from the live object, left nothing
unresolved, and showed a book row for every authored definition.  NOT verified:
the real closure stage.  Also failing in the linking suite, and not part of this
change: the value-union errors above, and an `UnboundLocalError` for
`program_abi_receipt` at `fortran_c_shell.py:49725` (a line added with
`_typed_parameter_record_receipts`).

## Follow-up, 20:00: four narrow agents, one review, and what is left

Four Sonnet agents were fanned out (no bootstrap, no contract edits, no repo-wide
runs, no commits).  Results, verified against their diffs and the focused gate:

* **Union table picklable + lifecycle** (done).  `SSAValueUnionTable` gained
  `__reduce__`/`__setstate__` mirroring `SSARecordTable` and `_SSALayoutTable`
  (a pickle carries the owner label and descriptors; `__init__` rebuilds the
  `_BookRows` and the member-claims callback on the book current at load).
  `class_subtype`, `union_alternative_state`, `value_union_descriptor` and
  `value_union_member` are classified in `page_lifecycle.py`.  The lifecycle
  audit and `tests/test_native_reference_boundary.py` (8) pass.
* **`program_abi_receipt` UnboundLocalError** (done).  `_lower_ast_source_to_ssa_impl`
  read the receipt in its first loop over `graph.function_table` (via
  `_typed_parameter_record_receipts`) before it was built; the construction moved
  ahead of that loop, unchanged.
* **Empty union-alternative storage** (partly done).  `_physical_type_arms` in
  `fortran_c_shell.py` now takes `storage` from data the compiler already holds: a
  ProgramABI record receipt for the class identity (`"record"`), else the value
  node's own `program_abi_storage/dtype/rank`, else an authored scalar parameter
  (`"scalar"`).  `__post_init__` is unchanged.  `tests/test_ir_sequence_tables.py`
  47 pass; `tests/test_sequence_contract_concordance.py` 34 pass, 3 fail (was 10).
  OPEN DESIGN QUESTION for the value-union author: the remaining cases have no
  authored physical declaration at all: Python `Constant` literals of
  `builtins.int/bool/float/str`, a user-class parameter with no record receipt
  (`page: IdentityPage`), and a `GetAttr` of a dict-typed record field.  (a)
  declare the compiler's own representation per class (scalar for a numeric
  literal, sequence for `str`, record for a user class), or (b) do not build a
  per-class alternative for an exact, non-union value, which also relaxes the
  "one physical alternative per class" check near `fortran_c_shell.py:19385`.
  Not decided; nothing written for these.
* **Cold review of `b06202e1`** found three real defects in the held-definition
  change, all fixed: (1) installing bindings on a held authored class walked and
  overwrote every method's own bindings (`self`/`cls`, locals), so a construction
  now only activates an authored class; (2) `__setitem__` raised when a class
  discovery re-registered a method already held from discovery, so it now refuses
  only a copy laid over an AUTHORED holder; (3) the later authored definition of a
  qualified name is the live one (`hold_authored` replaces), and `@overload` stubs
  are not held.  `tests/test_held_definitions.py`: 6 pass.  The reviewer also
  noted a nested function's live qualname contains `<locals>`, so it still misses
  the held lookup and falls to discovery as before; unchanged, not a regression.

Baseline discipline (a short-path worktree at `6d3450ea`, `NODUS_TENSOR_CORE`
unset, now removed): `tests/test_process_graph_function_linking.py` already failed
23 tests there (the `KeyError`s on callee names such as
`write_only_record_field_inout__update`, `span_cross__root` set mismatch, and the
rest).  At the checkpoint tree it failed 57; the 34 extra are the value-union
errors above.  Nothing that failed at `6d3450ea` passes now, and nothing new
beyond the union errors was found.  That file is not a clean gate.
