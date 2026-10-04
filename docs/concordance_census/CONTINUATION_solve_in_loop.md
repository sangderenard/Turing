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

## 2026-10-04 (later): the cumsum shortfall, root cause and fix; solve now emits and runs

### How the `cumsum` arrived (read-only hook on the lowered module)
In `_first_occurrence__specialized_*__planned_region_0` it is op `cumsum`, attributes
`{dim:-1, region:0, tensor_candidate:'cumsum'}` (NO `tensor_operation`), operands
`(1, shape (), float64)` and the int64 dim, result `(5, shape ())`. The op name WAS
recognized; the lowering (`tensor_ssa_lowering.py` `operation == "cumsum" and source.shape`)
declined it because its source shape was `()`, and a candidate-only op posts no lowering
shortfall (`candidate_only`), so it fell to the LLVM emitter's fall-through.
Why `()`: `mask` is `(candidates == largest).cast_like(M) * eligible`. The shape of every
`cast_like` result was never derived:
1. `_tensor_descriptor_rule` has no rule for `cast_like` (only `_DTYPE_CAST_OPERATIONS`,
   which does not contain it). The book shows `_pivot_mask` rows 15/32/33/34 only
   `invalidated` (`call-result-specialization-changed`), never re-proven.
2. In callsite-specialized graphs the catalogue descriptor (graph-domain default `()`) was kept
   for `cast_like`/`cumsum` (the "re-evaluate from local operands" list lacked them), and
   `_value_shape_dtype` (region `value_shapes`) had no entry for them either.
3. `x.to_dtype(U.get_dtype())` kept the OPERAND's dtype (bool): `get_dtype()` is a call node, not a
   literal. This is the `permutation` bool/(2,2) of the 2026-10-04 C-lane `call_result_contract`
   conflict. Once shapes flowed, it surfaced as "3 incompatible physical call inputs".
Also: the emitter's honest "cast_like over span needs its extents" refusal (ssa_llvm_backend ~3779)
had been hiding behind a silent one-element `cast_like` while the shape was `()`.

### Fix (src/compiler/glsl_deployment_strategy.py unless noted)
- `_tensor_descriptor_rule`: new `cast_like` rule (shape of the VALUE operand, dtype of the
  REFERENCE operand's descriptor); `_literal_dtype` now names `get_dtype()` of a tensor from that
  tensor's descriptor.
- `_tensor_descriptor` `specialized_operator` set and `_tensor_descriptor_rule` re-evaluation list:
  added `cast_like`, `cumsum`, `cumprod`.
- `_value_shape_dtype` (region value_shapes): added `cast_like`, `cumsum`, `cumprod` to the
  descriptor-settled list (NOT the broadcast list: cast_like's reference would give (2,2)).
- tensor_ssa_lowering.py (view alias, ~3818): an `unsqueeze` alias of a source whose extents are
  not settled yet is flagged `ssa_storage_view` (unsqueeze differs from its source by construction).
- fortran_c_shell.py (~42431, concorded source/SSA shape check): a region-call argument
  (`ssa_region_feed`, nonempty shape) is the formal's view of the fed storage, so the owner's
  concordance shape is not its shape: skipped (was a hard ValueError for
  `forward_substitute` value 56: ssa (2,1) vs concordance (2,)).
No spelling table touched; no new rows posted (no op was renamed/respelled; the descriptor
queries already record `graph_tensor_descriptor` transformation edges per answer).

### Measured
`probe_solve_c_lane.py llvm 2x2`: lower 25.7 s, LLVM_SHORTFALLS 0, built 16.1 s, run OK,
sentinel untouched 0 / NaN 0, PRODUCED [1, 0] vs EXPECTED [0.1, 0.6], max abs err 9.0e-01.
`3x3tie`: lower 79.8 s, shortfalls 0, PRODUCED [1, 0, 3] vs [0.8333, -1.6667, 1.1667], max err 1.833.
Wrong numbers, no NaN, nothing unwritten. Stopped there per instructions (not chased).
Gates: tests/test_compiled_linalg.py::test_the_compiled_eigh_matches_numpy passed (23 s);
tests/test_c_backend_llvm_ssa.py::test_direct_tape_lowering_binds_cumsum_to_real_translated_kernel passed.

## 2026-10-04 (bisect against eager): first wrong stage = scalar loop index read as double

Measure only; no existing file edited. New probe: `tools/compiler_probes/probe_solve_bisect.py {2x2|3x3tie} STAGE...`
(same contract/entry as probe_solve_c_lane.py; stage = authored helper code from linalg.py, one intermediate
returned as the output, compared with the same source run eagerly; env PROBE_NOPOISON=1 keeps non-output buffers
unpoisoned, PROBE_DUMP_IR=1 writes build/sb_<stage>.ll). Each run 17-25 s. Note: the probe poisons every non-fed
buffer with -12345.0 by default, so unfed loop-slot reads show as the sentinel.
linalg.py working-tree diff (git diff src/common/tensors/linalg.py) sha256 = 4aad2c7467a5b8ce8b3c9f1fde08a28871674717582d315f5a2a3ccade27c41f
(6 insertions, 26 deletions: `to_dtype(dtype)` -> `cast_like(like)` in _first_occurrence/_one_hot_axis/_pivot_mask/_masked_pivot_rows,
and the duplicate `_first_occurrence` removed).

Bisection order (2x2 [[4,1],[2,3]], rhs [1,2]); all values native vs eager:
- lu_U (full loop)       DIFFERS native [nan nan nan nan]  eager [4 1 0.5 2.5]      <- middle probe
- masked_rows_U (k=0)    MATCH [4 1 2 3]
- perm_after_pivot0      MATCH [1 0 0 1]
- pivot_mask(k=0)        MATCH [1 0];  pivot_value0 MATCH [4];  factor0 MATCH [0 0.5]
- u_after_update0        MATCH [4 1 0.5 2.5]
- u_unrolled_k1 (hand-unrolled k=0 then k=1, literal k)   MATCH [4 1 0.5 2.5]
- the `for k in range(n)` loop version, same body: lu_P native [1 0 0 0] vs eager [1 0 0 1]; lu_sign -1 vs +1;
  pb [1 0] vs [1 2]; y_forward [1 0] vs [1 1.5]; x_back [1 0] vs [0.1 0.6]  (x_back reproduces the probe_solve_c_lane answer).
- last iteration inside the loop: loop_pivot native [0 0] eager [0 1]; loop_hot (= _one_hot_axis(U,n,k)) [0 0] vs [0 1];
  loop_parity -1 vs 1; loop_pivot_row/value/factor NaN; loop_after (index > k) MATCH.
FIRST MISMATCH: the loop's second iteration (k=1), at `_one_hot_axis(..., k)` (linalg.py:206 `== position`) and the
`index >= k` in `_pivot_mask` (linalg.py:237). Last matching: identical body with literal k (k=0, and unrolled k=1).
The pivot mask is all zero at k=1, so masked_pivot_rows writes a zero row to U and P (sign -1), the division by the
zero pivot gives NaN in U, and the answer is the rhs row 0 only. 3x3tie not re-run (2x2 reproduces the symptom).

Read-only IR (build/sb_x_back.ll): `__ssa_..._one_hot_axis__..._planned_region_0` and `..._pivot_mask__..._planned_region_0`
do `%s = load double, ptr %arg.1` for the scalar `position`/`k` and `fcmp oeq/oge` against it, with NO sitofp. The loop
passes its k slot: initial slot `alloca i32` (store i32 0), latch writes `store i64 k+1` into a different i64 slot
(`%value.531`), header loads it as i32. So k=1 is read as the double whose bits are the i64/i32 1 = 4.9e-324 (never == 1.0;
k=0 reads 0.0 by luck of zero neighbouring bytes). Contrast: the inline site in the same loop (`index > k`,
lu_decompose_inplace__planned_region_10) HAS `load i32; sitofp i32 to double` and works (loop_after MATCH). So the
defect is the dtype of the scalar formal at the callee-region boundary: the callee's region treats `position`/`k`
as float64 while the caller's carried slot is int32/int64 and the call passes a pointer with no conversion.
Hypothesis (not verified at the SSA level): the callee's untyped scalar parameter is settled as float64 (it is compared against
a float64 tensor; the specialization hash 0598a65ac1c3 is shared by _axis_index and _one_hot_axis, keyed on `like` only), so its
region formal is float64 even though the loop index is int. With a literal k at the call site (unrolled), the Const is
materialised as a double and everything matches, which is why every non-loop stage matches.
The failing lines (linalg.py:206, :237) are inside the other agent's diff only in their `.cast_like` tail; the `== position`
and `>= k` comparisons themselves are unchanged by it. Not fixed.

## 2026-10-04 (loop-index dtype fix): the formal was defaulted float64; fixed at the descriptor

Repro (seconds, no clang for `ssa`): `python tools/compiler_probes/probe_loop_index_dtype.py {ssa|run}`.
`for k in range(count): total = total + pick(t, k) * (k + 1)`, `pick = (t == k).cast_like(t)`, t=[0,1,2].
Before: native [1,0,0] vs [1,2,3]; callee region_0 took value 0 (`k`) as float64 while the function-level
Call fed the caller's `int` Phi; no Cast. (A literal `range(3)` unrolls to literal k per copy and hides it.)

Decision (read-only sys.settrace hook on `_value_shape_dtype.declared`, glsl_deployment_strategy.py ~3394):
`dtype = tensor.get("dtype") or specialization.dtype or node.dtype or domain.dtype or "float64"`. The callee
formal Input `k` has none of the first four, so `or "float64"` settles it. Why no actual reaches it:
`_callsite_specialized_shell_type` builds `tensor_descriptors` from `_tensor_descriptor(caller, actual)`; the
caller's loop target (Input, `binding_kind=loop`, For `iterator_kind=arithmetic_sequence`) answered None, so only
`t` was passed (hook: `descs {'t': ...}`). An ABI int scalar (`count`) already answered {shape (), int64} and works.

Fix (glsl_deployment_strategy.py, `_tensor_descriptor_rule` + `_tensor_descriptor`):
- `_arithmetic_loop_target_owner` (~19333) names the single-target `range` For of an Input; the rule (~19925) answers
  `{shape (), dtype "int", rank 0}` ("int" = the loop Phi's SSA dtype). It rides the existing channel:
  `_apply_callsite_tensor_descriptors` stamps the copy's formal, `formal_shape` rows, `_value_shape_dtype` reads it.
- Edge: in `_tensor_descriptor` (~19532) the answer's `graph_tensor_descriptor` shape transformation for such a
  target comes from the For node (`loop_target` role, `source_cells=(node_identity_cell(For),)`), not a
  `descriptor_root` (unsourced). Oscillator audit unsourced 2820 -> 2818.
- Conversion at the comparison: the existing scalar-kernel Cast (tensor_ssa_lowering.py ~5545, `binary_scalar_double`)
  now posts a `kernel_input_conversion` row (function scope, block, consumer result, operand position) ->
  (src id, converted id, src dtype, float64, callee), mode REVISE, stage TENSOR_SSA_LOWERING, Derived from the
  scalar's `ssa_value` cell; `Unsourced(KERNEL_OPERAND_NOT_ON_BOOK)` (new reason, concordance_declarations.py:886)
  when the book holds none. `_post_scalar_kernel_input_conversion` at tensor_ssa_lowering.py:1948.
Result: callee region_0 now has `Cast int -> float64` before `binary_scalar_double`; repro native [1,2,3], err 0.

Solve after fix: `probe_solve_c_lane.py llvm 2x2` PRODUCED [1,2] vs [0.1,0.6] (max err 1.4); `3x3tie` PRODUCED
[1,2,3] vs [0.833,-1.667,1.167] (max err 3.667). NEW stage state (probe_solve_bisect.py 2x2): lu_U, lu_P, lu_sign,
loop_hot, loop_after, loop_parity, pivot_mask, pb all MATCH eager (LU is now correct). FIRST WRONG STAGE NOW:
y_forward (`_forward_substitute`, linalg.py:318): native [1,2] = Pb unchanged vs eager [1,1.5]; inputs LU and Pb
match, so the `for i in range(n)` y update (`y * (1 - hot_i) + hot_i * solved`) is not applied. Not chased
(different mechanism: the emitted IR has no remaining scalar `load double` of an index). loop_pivot (a non-carried
loop temp returned after the loop) still reads [0,0] vs [0,1]; lu_U is correct so it is the post-loop read of a
per-iteration temp, not LU. Gates: test_native_scalar_loss_adjoint 2 passed; test_ssa_record_return_state 22 passed;
test_record_return_site_identity 14 passed 1 xfailed; test_symbolic_structural_constants 2 passed;
test_compiled_linalg eigh passed; audit findings 0,1,0,1,5,0,0 unchanged. NOTE: C: drive hit 0 bytes free mid-run
(an "IO failure on output stream" in one pytest compile); freed this session's own build/ dirs; rerun passed.

## 2026-10-04 (y_forward): an evaporated loop's carried name lost its loop-exit version

Repro (seconds): `python -u tools/compiler_probes/probe_loop_update_lost.py {c|g|b|a}`; `c` is
`y = b.clone(); for i in range(2): y = y + 1; return y` (native [1,2] vs eager [3,4]); `a` is the real
`_forward_substitute`. Not specific to substitution: any literal/static `range` loop carrying one name.
Cause (read-only trace): the reducer's canonical `identity_table['y']` is (3,6,6) (initial, body binding,
`authored=False` loop-exit binding = the body's `updated`). `loop_composer.evaporate_unrolled_loops` unrolls the loop
into clones (ids 10,13), redirects the consumers of `updated` to the last clone (`final_value`) but never says the
NAME moved: `updated` (6) leaves the graph, the table filter at the end of the function drops it, history = (3,),
`named_output_histories` (precompile_to_ssa.py ~11097) resolves `y` -> 3 = the pre-loop value, and the Ret carries
(3, 13) with the authored output first. Native returned the input unchanged.
Fix (src/compiler/loop_composer.py, evaporate_unrolled_loops): `carried_finals[updated] = final_value` where the
redirect is already computed; before the table filter, a `name_binding` ALIAS row (existing
`_post_identity_table_mutation(aliases=..., stage=LOOP_COMPOSER)`: REVISE, Derived from the previous cell + the final
copy's identity cell) revises the evaporated version to the final copy, and the dict follows.
Measured: probe c/g/b/a all MATCH (err 0). probe_solve_bisect 2x2 y_forward [1,1.5], x_back [0.1,0.6] MATCH;
probe_solve_c_lane llvm 2x2 and 3x3tie MATCH NumPy (max abs err 0.0 both). Gates: scalar_loss_adjoint 2 passed;
ssa_record_return_state 22 passed; record_return_site_identity 14 passed 1 xfailed; compiled eigh passed;
test_precompile_to_ssa 13 failed 91 passed (baseline 13); audit unsourced 2818 (unchanged).
