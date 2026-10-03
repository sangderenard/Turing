# Continuation: symbolic cache provenance (item 1), external identity rows (item 2)

Base: turing a7f57c1e. Uncommitted. Not committed by this lane.

## 2026-10-03: item 1, symbolic cache provenance: DONE

Flaw: `sympy_dual_ir_cache` (used by `symbolic_equation_compiler`) keyed the
`dual-ir` and `symbolic-program` layers on `_pipeline_implementation()`, a
digest of the symbolic modules and a few callables. An entry built by another
compiler tree was served as current. That hid the phasing regression
(CONTINUATION_phasing_regression.md).

Fix (`src/compiler/sympy_dual_ir_cache.py`; the key is unchanged):
- The layers that hold compiler output (`PROVENANCED_LAYERS` = `dual-ir`,
  `symbolic-program`) now store `CompiledPayload(value, compiler)`.
  `compiler` is the piece cache's own `PieceCompilerRecord`
  (`native_law_kernels.route_compiler_record()`, taken when the build
  finishes). This reuses the piece mechanism; there is no second one.
- Value and record are one pickle, published by the store's single
  `os.replace`. A reader never pairs a value with another build's record.
- On load, `piece_staleness(payload)` runs. An old payload with no record is
  `<no compiler record>`, which counts as stale.
- A stale entry is rebuilt by default. To serve it instead, pass
  `SympyDualIRCache(serve_stale=True)` or set
  `TURING_SYMPY_DUAL_IR_SERVE_STALE=1`. Serving prints a loud stderr line.
- Rows go on the active book. Pages are declared in
  `concordance_declarations.py` (appended section), and the facts are the
  piece cache's `PieceBuildFact` / `PieceStalenessFact`.
  - `symbolic_cache_staleness` (layer, cache identity): decision `rebuilt` /
    `served` / `rebuild_failed: <Error>`, the changed modules and the recorded
    digest. It is a NOVEL(`symbolic_cache_stale_check`) root.
  - `symbolic_cache_build` (layer, cache identity): the record of the value
    returned. It is DERIVED from the staleness row when it is that row's
    rebuild, otherwise NOVEL(`symbolic_cache_build`).
  - CONCORD while the fact repeats, REVISE when it changes.
- `solved-equations` is not provenanced. It holds a SymPy solve, keyed on the
  producer's own files, and contains no compiler output.

Consequence: every existing `dual-ir` entry is `<no compiler record>` and
rebuilds once. Like the piece cache, an entry is also stale after any edit
to a `src.*` module the building process had loaded, which over-reports by
design.

Staleness demo (scratchpad `symcache/stale_demo.py`, fresh cache dir, about 10 s):
- first: miss, `symbolic_cache_build` col 0, NOVEL root.
- second: hit, same build row, no new cell.
- tampered `src.compiler.ssa_python_materializer` digest: rebuilt. Staleness
  col 0 (`rebuilt`, names that module). The build row gains an edge from
  staleness col 0.
- tampered `symbolic_process_graph` with serve_stale: hit. Staleness col 1
  (`served`) and the "SERVING STALE" line.
- opt-in off again, tampered entry still on disk: rebuilt. Staleness col 2;
  the build row gains an edge from col 2.

Gates:
- tests/test_sympy_dual_ir_cache.py + test_symbolic_structural_constants.py:
  7 passed.
- test_orbital_transfer_compile.py: 12 passed (268 s).
- probe_external_function.py: all ok, 0.000e+00 in both lanes.
- audit: 0,1,0,1,5,0,0, unchanged.

### Other on-disk compile caches with the same flaw (not fixed)
SAME FLAW (no compiler identity at all):
- `native_law_kernels.py:610` `_cache_key` (key = version + backend + batch +
  stage source) and `:641-649` (`kernel.pkl` + DLL). Loads with no compiler
  check, and the write is not atomic.
- `opportunistic_pipeline.py:158-160` / `:190-211` key, `:178-185` write
  (`pipeline-artifacts`). Provider and input digest only; direct write.
- `project_compilation_product.py:6801-6826`, `:6989-6996`: seed regions
  are reused when the authored-source sha matches, whatever compiler built
  them.
- `vehicle_validator_simulation.py:382-399` write, `:413-428` load:
  `source_sha256` only; direct writes.
- `build_math_cache.py:40-104` (`math_cache`, consumer
  `fused_program_wasm_backend.py:3455`): checks no `wasm_math_tables`
  revision.

PARTIAL (a subset of the compiler, or mtime):
- `kernel_bank.py:447-481` key, `:659-681` write: newest mtime over
  `src/compiler` only, one-second resolution.
- `host_code_modules.py:496-513`, `:540-556`: 6 named modules, plus a
  legacy allow-list at `:41`.
- `symbolic_fluid_direct_control.py:96-107`; the check is in
  `symbolic_fluid_native_runtime.py:488-546`: mtime over `src/compiler`.
- `perforated_network_llvm.py:110-121`: stat fingerprints of two trees.
- `project_compilation_product.py:62-135`: a 42-file toolchain list. Its
  `.tmp` names are shared across writers.
- `aot_compile.py:704-718`, `:751-771`: named callables plus a manual
  version bump. This is the reference flaw's shape.
- `sympy_dual_ir_cache` `solved-equations`: producer files only.

Shared store note: `AOTCheckpointStore.store` (`aot_checkpoint.py:197-220`)
names its temp files by pid only. Two threads in one process writing the
same phase collide. Not changed here; it is a shared file.

## 2026-10-03: item 2, external identity rows: STOPPED, one question

Measured (scratchpad `symcache/unsourced_orbital.py`, the real
`compile_sympy_equations` + `piece_from_law` route and contracts, 9 laws):
- The lowered modules have 234 unsourced identities.
- 228 of them are in the external leaves (`external_*` and their
  `__planned_region_0`): 3 or 4 minted ids per leaf function (Const, Call,
  GetElementPtr, region scalars).
- The other 6 are Consts in two law roots: the `0` arguments of `r_i(0)` in
  `initial_condition_lhs` and `orbital_transfer_raw`.

Cause (observed): each leaf is lowered by its own `lower_ast_source_to_ssa`
call (`declare_external`). That call opens a fresh book
(`fortran_c_shell.py` `begin_identity_book()`), and the leaf's own book has
every mint edge: 89 mint rows, 0 unsourced identities. The law's compile
opens another fresh book, links the leaf module in through
`linked_repository_ssa`, and gets none of the leaf's mint edges. The leaf
ids are not unsourced where they were minted. They are unsourced in the
law's book, because the books are separate.

Same wall for (a) and (b): `IdentityBook.post` requires every source Ref to
be a cell on the SAME book (`_source_stamp`). The symbolic rows live on the
ambient book: `external_function`, `external_callsite`, the equations. The
leaf and the law each live on their own fresh compile book, and the
emission rows go on the law module's book. A NOVEL-from-the-declaration
post, or a DERIVED-from-the-callsite post, therefore cannot be written on
the book where the leaf ids and the lowered calls are judged. No
symbolic -> lowering edge exists today for any law; the emission chain ends
at the AST `source_span`.

Question: should the law's lowering, and each leaf's, run on the caller's
ambient book? That would be `lower_ast_source_to_ssa` resuming
`current_identity_book()` (an opt-in such as `identity_book=` passed to
`begin_identity_book`) instead of opening a fresh one. With it, the symbolic
rows, the leaf mints and the law's emission share one book; (a) and (b) are
direct posts; and the leaf's link into the law keeps its edges. Without it,
(a) and (b) need a cross-book reference that does not exist. The change is
in `fortran_c_shell.py` (the name-arm lane's file), so it is not mine to
make.

Gate added for item 1: probe_orbital_craft_binding.py failures 0 (LLVM and
C), rc 0.
