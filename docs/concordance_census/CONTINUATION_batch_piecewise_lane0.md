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
