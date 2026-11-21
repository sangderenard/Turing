# 2026-11-13: explicit native pointer null is not Python None or a cast

## Defect and repair

An explicitly scalar ptr Const(value=0) produced invalid LLVM `store ptr 0`.
C already supports this native null convention. LLVM now emits `ptr null` only
for exact integer zero with an actual scalar ptr/rank0/nonarena SSA_VALUE
physical receipt. Python None, nonzero integers, float/string coercions and
missing scalar receipts refuse. This does not introduce integer-to-pointer
casts. The physical-receipt helper is shared with the prior Load snapshot fix.

## Verification

* New null cases plus pointer snapshot/owner move tests:17passed.
* Exact backend baseline for the same tests:10passed,7failed; candidate restored
  byte-for-byte afterward. C zero/null behavior, Python None refusal and all
  pointer snapshot/control/address cases are unchanged.
* Existing heap+inplace suites with those tests:44passed,2failed,3errors. The
  incumbent failures remain raw sequence_column_claims fixture writes and
  untyped Indexed dtype refusals in total/pair_update; no new existing failure.
* Mapping audit:28rows,2functions,zero findings/unsourced,latch CLOSED.
* Native C/LLVM zero tests clear a pointer cell and retain its earlier snapshot
  for multiple bit patterns; no sentinel is dereferenced or released.

## Limits

This establishes the primitive needed to clear a transferred owner. It does
not authorize an owner escape or implement public record-result projection.
Ranked/address constants and arbitrary pointer-valued integers are not added.
The separate original C raw allocator fixture limitation remains as recorded
in the pointer-snapshot hazard; no C allocator success is claimed here.
