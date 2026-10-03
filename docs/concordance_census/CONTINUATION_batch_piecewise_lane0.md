# CONTINUATION: batch-4 Piecewise lane-0 garbage (propellant-supply piece)

Defect handed over from CONTINUATION_llvm_dt_system_spans.md: the batch-4
`orbital_craft_propellant_supply_t6` piece returned garbage in lane 0 when
its Piecewise condition was false; batch 1 correct.

## 2026-10-03 -- repro, cause, fix, proof

- Repro (scratchpad `pw_lane0/repro.py T`, ~10-20 s incl. a fresh compile
  into a scratch cache dir): `equation_piece` of
  `propellant_supply_rhs(T)` at batch 4 and batch 1; lane 0 throttles 0
  (demand 0 -> condition false), lanes 1-3 random.
  - T=1 before: batch-1 per lane `[1, 1, 0.6067, 1]`; batch 4
    `[0.0, 1, 1, 0.6067]`.
  - T=6 before: batch-1 `[1, 0.2411, 0.1201, 0.2204]`; batch 4
    `[8.14e-312, 1, 0.2411, 0.1201]`.
  - Every lane wrong, not only lane 0: what looked like a one-lane shift is
    the stack next to a one-element buffer (below).
- Emitted LLVM (read only, b4 `.ll`): the region computes the mask column
  `value.25` (4) and the clamp arm `value.30` (4); the entry then does
  `load double value.25[0]` -> `br i1` -> `phi ptr [value.30, if_true],
  [value.8, if_false]` -> `memcpy(out, phi, 32 bytes)`.  `value.8` is the
  constant arm `1.0`, an `alloca double, i64 1`: lanes 1..3 of the copied
  buffer have no writer (the stack is read), and when lane 0 is true every
  lane gets the clamp arm whatever its own condition.
- Stage source (`symbolic_abstract_tensor_source`) showed why:
  `t25 = t21 if t24 else t0`.  The law SSA's `Select` (elementwise) was
  materialized as Python's conditional expression, i.e. one branch for the
  whole column; the lowering spelled that as control flow on the mask's
  first element.
- ROOT CAUSE: `src/compiler/ssa_python_materializer.py:439-445`
  (`_BodyMaterializer`, `Select` case) emitted `(a if mask else b)` in the
  tensor vocabulary too.
- FIX (same file, the Select case): with `tensor_vocabulary` and a mask that
  is not provably constant, `Select` is spelled as the catalogued static
  `AbstractTensor.where(mask, a, b)` (form read from `TENSOR_CALL_FORMS`,
  required to be the class form).  Scalar vocabulary and constant masks
  unchanged.  Stage now reads `t25 = AbstractTensor.where(t24, t21, t0)`;
  the b4 IR calls `where_double(mask, clamp, filled_const, out, 4)` with the
  constant arm materialized by `fill_double` into a 4-element buffer, i.e.
  every output lane now has a producer (the no-writer value is gone through
  the existing where lowering, whose shape/dtype contract is already posted:
  tensor_ssa_lowering where broadcast + 095a3c0c where_double literal arms).
- Stale cache: the piece cache key is the equations, not the compiler.
  Scanned every cached `.ll` (engine_toy/artifacts/llvm_pieces,
  engine_toy/__llvm_lawcache__, turing/artifacts/llvm_pieces*) for the
  scalar-branch pattern (`if_merge:`) -- exactly one hit, the b4
  `orbital_craft_propellant_supply_t6` artifact; moved (not deleted) to
  scratchpad `pw_lane0/stale/`.  It rebuilt on the parity run; new IR has 0
  `if_merge`, 1 `where_double`.
- Proof (repro, T=1 and T=6, fresh compiles):
  - 100 standalone runs bit-identical, all lanes bitwise equal to batch 1.
  - instantiated, in-place OFF: 100 runs bit-identical, match batch 1.
  - instantiated, in-place ON (`in_place=('propellant_supply_next',)`,
    out span pre-poisoned with NaN each run): 100 runs bit-identical,
    match batch 1.
  - C lane: not applicable (`piece_from_law` / `_lower_law` are LLVM only).
  - spans parity harness with the `("orbital/batched", "propellant_supply")`
    skip REMOVED (scratchpad `pw_lane0/parity_noskip.py`), two runs:
    9 scenarios, 7524 field-rounds (was 7428 with the skip), 0 differences.
- Gates: engine_toy tests/test_orbital_jumper.py 11 passed (41 s);
  tests/test_orbital_craft_machine.py 13 passed (24 s); turing
  tests/dt_system/test_llvm_dt_system.py 2 passed;
  tools/audit_identity_concordance.py findings 0,1,0,1,5,0,0 (= baseline).
- OPEN (not fixed, other owners' file): the LLVM backend still accepts a
  conditional branch whose condition is a multi-element column
  (`ssa_llvm_backend.py:2711`, `load_as(..., "i1")` reads element 0) and a
  phi merging a (4,) and a () buffer copied at the larger extent.  Authored
  `a if column else b` is ambiguous in Python; that lowering should refuse
  (shortfall) rather than take lane 0 and read past the smaller arm.

## 2026-10-03 -- follow-up: multi-element branch conditions refused loudly

- Probe (scratchpad `pw_lane0/branch_probe.py`, seconds): authored
  `t3 = t2 if t1 else 1.0` with `t1 = x > 0.0`, batch contract 4 and 1,
  emitted to LLVM and C.  Before: b4 LLVM complete=True, C 0 shortfalls,
  the entry's CondBr condition `%3` has shape (4,) (b1: (1,)).  Both lanes
  read element 0 (LLVM `load_as(..., "i1")`; C `scalar_operand` ->
  `*((T *)(home))`).
- Fix: `ssa_llvm_backend.py` CondBr (right after the unknown-target check)
  and `ssa_c_backend.py` CondBr: a condition whose declared shape is all
  integer extents with product > 1 is a named shortfall:
  "conditional branch on a multi-element condition %3 shape (4,)
  (4 elements) -> if_true/if_false: a branch takes one truth value;
  elementwise selection is where" (C adds `in <fn>`).  Symbolic extents are
  not refused (not provably multi-element).
- After: b4 LLVM complete=False with that reason; C 1 shortfall with that
  reason; b1 unchanged (complete, 0).
- Gates: supply repro T=6 fresh compile, 100 runs bit-identical and equal
  to batch 1 (standalone, in-place off, in-place on); engine_toy
  test_orbital_jumper 11, craft_machine 13, actuation 8, game 7, tracker 9,
  plan 9 passed (collocation not run: another lane's uncommitted WIP);
  turing tests/dt_system/test_llvm_dt_system.py 2 passed; audit findings
  0,1,0,1,5,0,0 (= baseline).
- Cache scan (scratchpad `pw_lane0/scan_pieces.py`, read-only): all 100
  cached `.piece` files under engine_toy/artifacts/llvm_pieces,
  engine_toy/__llvm_lawcache__, turing/artifacts/llvm_pieces*: 0 CondBr on a
  multi-element condition in their SSA.  8 batch-1 `.ll` still have the
  `if_merge` shape (supply t1/t3/t4/t5/t6, orbital_game_phasing, machine
  slew, machine supply); at b1 the condition is (1,) and the merge loads
  and stores ONE double -- correct, not moved.  They will respell as
  `where` only when their cache entry is rebuilt.
- PROPOSAL (not implemented): stale artifacts survive compiler fixes because
  `_llvm_piece_cache_key` hashes schema + id + batch + equations only.
  Precedent: `kernel_bank.KernelBank._compiler_fingerprint` (newest mtime
  over src/compiler) is part of its variant key.  Proposed:
  1. Compiler identity = sha256 over the CONTENT (not mtime: concurrent
     edits and checkouts churn mtimes) of the turing `src.*` modules that
     participate in the route -- the `src.*` entries of `sys.modules` after
     `compile_sympy_equations` + `piece_from_law` return (the import closure
     of the route), sorted by module name.
  2. Record it, don't key on it: the `.piece` carries
     `compiler_digest` + the participating module list, and the build posts
     one book row (piece_id, batch, equations key, compiler digest, module
     digests) so a cached artifact's provenance is an edge, not an
     attribute.  Keying on it directly would invalidate every piece on every
     compiler edit (several lanes edit the compiler concurrently).
  3. On load, a digest mismatch is a named staleness: rebuild by default,
     or (opt-in) serve the old artifact with a logged row naming which
     participating modules changed.  The per-module digests make that
     answer "ssa_python_materializer.py changed", which is also what tells a
     reader whether a given fix can matter to a given piece.

## 2026-10-03 -- piece compiler record, staleness, lock, atomic publish

- Record (turing `src/compiler/native_law_kernels.py`): `PieceCompilerRecord`
  (combined sha256 + `(module, relpath, sha256)` for every `src.*` module
  loaded when the build finished -- 291 in an engine_toy process; a
  superset can only over-report staleness).  `route_compiler_record()`,
  `piece_staleness(piece)` (changed module names; `<no compiler record>`
  for pieces built before records).  File digests cached per process by
  path, verdicts cached per record: first check ~35 ms (hashes the files),
  then ~0.01 ms.  `LLVMPiece.compiler` field (old pickles load with None).
  `native_package.piece_from_law` stamps it on every piece it builds.
- Book: declared here (idempotent, this lane's names): pages `piece_build`
  (PieceBuildFact: digest + module digests) and `piece_staleness`
  (PieceStalenessFact: decision rebuilt|served|rebuild_failed: <Error>,
  changed modules, recorded digest), row (piece, batch, equations key),
  stage `piece_cache`, transforms `piece_build`/`piece_stale_check` (roots).
  `post_piece_book` posts on its own book and writes it beside the piece
  (`<id>.book.log`, `<id>.stale.book.log`).  A rebuild's build row is
  DERIVED from its staleness row (edge on the book).
- Cache (engine_toy `equation_piece`): key unchanged (equations).  Load ->
  `piece_staleness`; fresh = served; stale = rebuilt by default;
  `serve_stale=True` / `ENGINE_TOY_PIECE_SERVE_STALE=1` serves it with a
  `served` row and a loud line.
- Lock + atomic publish (coordinator, after the game lane's MemoryError /
  linker "Permission denied"): builds run under `_PieceLock`
  (`<id>.lock`, msvcrt byte lock / flock, 0.25 s polling, timeout
  `ENGINE_TOY_PIECE_LOCK_TIMEOUT` default 3600 s, TimeoutError naming the
  lock and the holder pid).  After the lock, the index is re-checked (a
  waiter takes the winner's piece).  The build goes to
  `.build-<pid>-<rand>/`, renamed to immutable `v-<sha256(dll)[:16]>/`
  (a loaded DLL is never overwritten), the piece is re-pointed there,
  saved to a temp index, loaded back AND its DLL loaded, then
  `os.replace`d over `<id>.piece`.  On failure: the build dir is removed,
  the old index stays, a `rebuild_failed` row is posted, the error raises.
- Demos (scratchpad pw_lane0/): stale_demo.py (build -> fresh load ->
  tampered digest names `src.compiler.ssa_python_materializer` -> rebuilt,
  book has staleness row + derived build row -> serve_stale serves with a
  `served` row); race.py (two processes, synchronized start, same piece,
  empty cache: both rc 0, both load the same `v-a1995ad3f34fbbad` DLL,
  identical outputs, one version dir, no .build leftovers);
  fail_keep.py (injected build failure: raised, index bytes unchanged,
  no leftovers, `rebuild_failed: RuntimeError` row; a process holding the
  old DLL keeps running it bit-identically while a rebuild publishes a new
  version dir).
- The eight b1 if_merge pieces were rebuilt through this path (as
  `<no compiler record>` legacy) during the suites; scan_index.py: supply
  t1/t3/t4/t5/t6, orbital_game_phasing, machine slew, machine supply all
  now call `where_double`, 0 `if_merge`.
- Gates: jumper 11, craft_machine 13, actuation 8, tracker 9, plan 9
  passed; game 4 failed / 3 passed (test_phasing x3, test_click...) -- the
  rebuilt orbital_game_phasing exposes a compiler regression in the
  uncommitted tree, being bisected by another lane (coordinator);
  test_llvm_dt_system 2 passed; audit 0,1,0,1,5,0,0.
- Observed cost: with several lanes editing the compiler, pieces go stale
  within minutes (one probe piece was stale on 6 modules ~10 min after its
  build), so suites rebuild their pieces (jumper 245-284 s vs 41 s cached).
  Version dirs accumulate (no collection yet); old legacy DLLs remain in the
  key dirs.
