# CONTINUATION: orbital_game_phasing returns t_wait 0

## 2026-10-03 phasing regression: found, fixed, proven

### Symptom
`phasing(MU, 7e6, 9e6, t_hohmann, pi, 2.2, 0.0)` built against the current
tree returned `t_wait = 0`, `phase = 0` (lead_angle 0.5088 still right).
The expected t_wait is 4995.3 s.

### The brief's premise was wrong
HEAD 822a753b WITHOUT the uncommitted edits is already wrong (t_wait 0).
No uncommitted lane caused this; the constant_role change is not involved.
The early "correct at HEAD" reading came from a stale symbolic dual-IR cache.
That cache is keyed on the symbolic modules only, so a bisect has to set
`TURING_DISABLE_SYMPY_DUAL_IR_CACHE=1` plus a fresh
`TURING_ACCELERATOR_CACHE_DIR`, `TURING_HOST_SSA_CACHE` and
`TURING_LAW_NATIVE_CACHE`, and call `compile_sympy_equations` +
`piece_from_law` with a temp directory.

Commit bisect with fresh caches:
- de609156 (06:17): 4995.296, correct. The old cached piece was built at
  06:25.
- aa5f1aac ("plain ints become floats"): -3024539.87, wrong. Same at
  0519095d, ce587b82 and a2020e0b.
- 822a753b (Select is where()): 0.0, wrong. Main tree: 0.0.

### Chain (observed)
1. The stage source is correct:
   `t31 = AbstractTensor.where(t30, t29, t28)` with `t29 = 1.0`, then
   `t32 = t25 * t31` and `t36 = t32 % t35`, where `t35 = 2*pi`.
2. `precompile_to_ssa.lower_control_sections_to_ssa`, integer-index
   inference: `integer_ops` held `"Mod"`/`"FloorDiv"` as unconditional
   integer seeds. `t36 = t32 % t35` marked t32, t35 and t36 as integer. The
   arithmetic propagation then spread int64 through the Mul/Add cone. The
   region metadata rows for 32 (t11), 34 (t25) and 38 (t31) became int64.
   This was traced with a scratch settrace of `region_value_meta`.
3. tensor_ssa_lowering planned the sign() Select (planned_region_2) with an
   int64 result. Its 1.0 arm was `fill_double`'d and then `fptosi`'d to
   i64. `where_double` read the i64 1 as the double 4.9e-324, and the
   result was cast back to i64 as 0. So sign = 0, Mod(0) = 0 and t_wait = 0.
4. Before 822a753b the same int64 label sat on a Phi of float arms
   (`a if mask else b`). The result was garbage: -1024 rad, which gives
   t_wait -3.02e6.
5. Before aa5f1aac the sign arms were integer literals. The int64 label
   was then accidentally true, so the piece was correct.

This rule is latent and old (4f14a372c, 2026-08-13). aa5f1aac exposed it,
and the where() change changed the wrong value it produced.

### Fix (main tree, src/compiler/precompile_to_ssa.py, about line 14602)
`Mod`/`FloorDiv` (and their lowercase spellings) moved from `integer_ops`
to `arithmetic_ops`. They are now integer only when integer evidence
reaches them: the result is an index, or all operands are integer. Python
`%` and `//` on floats are float. Values with a declared integer dtype are
still collected separately, from the instruction dtypes. The bitwise ops
stay unconditional. The comment above the rule names the program, the
wrong value and the rule.

No role/dtype ROW was changed. This inference writes `region_value_meta`
and posts no row; that was already true before this fix, and the fix does
not add or remove any posting. Open: the integer-index inference should
itself post its decision. It is not on the book.

### Proof (main tree, with the other lanes' edits present)
- Repro, fresh caches: lead 0.50877, t_wait 4995.295986578293,
  phase 5.384967102082877.
- turing tests/test_symbolic_structural_constants.py: 2 passed.
- turing tests/test_orbital_transfer_compile.py: 12 passed (258 s).
- engine_toy tests/test_orbital_game.py: 7 passed (343 s). The shared
  piece cache rebuilt the stale pieces through the staleness check.
- tools/audit_identity_concordance.py: findings 0,1,0,1,5,0,0, unchanged.
- The worktree at Temp\wtr was removed.

Phased shots: rerun the orbital_game shot command. The current
engine_toy/shots/orbital_game_machine_*.png were rendered with the bad
piece.
