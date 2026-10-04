# CONTINUATION: AbstractTensor linalg.solve, current measured state (2026-10-04)

Read-only measurement. No existing file edited. One new probe:
`tools/compiler_probes/probe_solve_c_lane.py {c|llvm} {2x2|3x3tie}`.

## 1. Eager gate

- `python -m pytest tests/test_linalg_solve_unbatched.py -q -x` -> 4 passed, 1.64 s (wall 4 s).
- `python tools/compiler_probes/probe_solve_eager.py` (3 s):
  2x2 0.0; 2x2 pivot 0.0; 3x3 8.9e-16; 4x4 5.6e-17; 5x5 8.9e-16;
  3x3 tied same sign 0.0; tied opp sign 0.0; three-way tie 0.0;
  2x2 rhs matrix 1.1e-16; 6x6 1.8e-13; 4x4 near-singular 0.0.
  The "batched 2x(3x3)" line RAISED ValueError, but from NumPy's reference
  call inside the probe (`np.linalg.solve(batch_a(2,3,3), batch_b(2,3))` is
  read by NumPy 2 as a (3,)-rhs-matrix mismatch), not from AbstractTensor.
  Probe-side artifact; not a solver result.

## 2. LLVM lane, 2x2 (probe_solve_numeric.py)

Command: `python -X faulthandler -u tools/compiler_probes/probe_solve_numeric.py`
(13:50:49, ~1.5 min). Lowering completed (`ssa-program complete; functions=131
exports=41`, +6.2 s). The probe then DIED in its own diagnostics, before
emission: `TypeError: unhashable type: 'SSAValue'` at
`src/compiler/identity_concordance.py:oscillating_rows` (line 3369,
`len(set(facts))`), called from probe line 164. No emission, no clang, no
numeric result from this probe. (49 oscillating `proven_shape` rows hold
SSAValue facts.) Not edited, per rules.

## 3. New probe: tools/compiler_probes/probe_solve_c_lane.py {c|llvm} {2x2|3x3tie}

Same entry/contract as probe_solve_numeric.py, no diagnostics, SENTINEL
-12345.0 poisoning. 3x3tie = [[1,2,3],[1,0,1],[0,1,4]], rhs [1,2,3]
(|pivot| tie in column 0). All runs stopped at the emitter's shortfall gate;
nothing was compiled or executed.

| lane | system | lower | result |
|---|---|---|---|
| LLVM | 2x2 | 68 s | 1 shortfall: `cumsum` in `..._first_occurrence__specialized_*__planned_region_0`, "operation has no repository LLVM emission" |
| LLVM | 3x3tie | 30 s | same 1 shortfall (same function/op) |
| C | 2x2 | 42-57 s | 3 shortfalls: `cumsum` "no module-lane C spelling" (same function); 2x `call_result_contract` caller `solve` value 32 retained bool/(2,2) storage float64 vs callee `_lu_decompose_inplace` value 105 proposed bool/bool |
| C | 3x3tie | 128 s | same 3 shortfalls, shapes (3,3) |

First failure location (observed, not fixed):
- LLVM: `src/compiler/ssa_llvm_backend.py` ~line 4157 (the fall-through
  `LLVMEmissionShortfall(..., "operation has no repository LLVM emission")`).
  `cumsum` IS in the op table (line 381 -> `cumsum_dim_double`), yet the op
  inside `_first_occurrence` planned_region_0 reaches the fall-through.
  Source: `src/common/tensors/linalg.py:185`
  `mask * (mask.cumsum(dim=-1) == 1).cast_like(like)` (note: linalg.py has
  another agent's uncommitted edits).
- C: `src/compiler/ssa_c_backend.py:5242` (`no module-lane C spelling`, cumsum
  has no C spelling) and `:2021` (`call_result_contract` from module metadata
  `call_result_type_conflicts`; same check in `ssa_self_check.py:489`).

The 2026-09-22 note's state (zero shortfalls, NaN output) is no longer the
state: the lane now stops at emission on cumsum.
