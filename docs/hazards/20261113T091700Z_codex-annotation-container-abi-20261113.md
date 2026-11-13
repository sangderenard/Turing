# 2026-11-13: native scalar pointer loads snapshot before later stores

## Defect and repair

The LLVM module emitter treated a ptr Load result as an address alias even
when the SSA_VALUE physical receipt explicitly declared a scalar pointer value.
Clearing the original owner cell before using that snapshot reread the cleared
cell, losing the allocation owner head. Pin the loaded pointer only when its
actual physical receipt says ptr/scalar/rank0, its shape is empty and it is not
an arena. Ranked/address and missing-receipt cases retain existing semantics.

## Verification

* Native pointer snapshot tests cover C and LLVM, pointer and opaque-ref control.
  The ptr case is not implemented by substituting an opaque reference.
* Native allocation lifetime test uses the established LLVM allocator/releaser
  and either LLVM or C pointer move: sizes1,7,2,0,33; source owner clears;
  releasing source does not free transferred data; destination release clears;
  repeated destination release is safe. Zero-size arrays are not dereferenced.
* Negative tests preserve missing-receipt and shape1 address semantics.
* New tests plus existing ownedheap and inplace suites: candidate35passed,
  2failed,3errors; exact19line-hunk baseline33passed,4failed,3errors. Only the
  new LLVM pointer snapshot and LLVM real allocation/move cases improve.
  Original backend restored byte-for-byte after baseline.
* Remaining incumbent failures: raw write to sequence_column_claims fixture;
  untyped Indexed source in total and pair_update (three fixture errors).

## Explicit limits

An attempted C allocator version of the raw-address fixture faulted BEFORE the
move at first host dereference. Its raw returned-pointer host ABI is not the
LLVM fixture's ABI. The final C test therefore tests C move only using the
LLVM allocator/releaser, not C allocator support. No record-result projection,
public owner transfer receipt, string-list decoding or returned-record success
is claimed. Ptr Const0 still emits invalid LLVM integer pointer syntax and is a
separate issue; these tests obtain null from a genuinely zero-initialized owner
cell, not an invented inttoptr cast.
