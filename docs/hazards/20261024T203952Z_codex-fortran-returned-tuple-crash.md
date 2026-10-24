2026-10-24 TITLE (MEASURED, where): 
2026-10-24 FORTRAN RETURNED LIST HEAP VIEW (MEASURED, Ubuntu24 WSL):

Observed on fresh main 79b86a3d through the canonical
lower_ast_source_to_ssa entry: c547's compact(n) returns a list inside a
tuple. Native Fortran returns 0 for n=0, then segfaults at n=1. GDB places
SIGSEGV in the generated root subroutine. The root's region Call passes heap
cell id 4, whose accounting says ptr, heap_element_dtype=int64; the Fortran
emitter has it in _heap_views. The callee grows that cell's arena. The
generated root pointed t4_v at the pre-call null arena, called compact,
and read t4_v(1) without refreshing it. _region_call emitted no refresh;
ordinary _ssa_call already emitted one. The new c622 program reproduces the
same crash on untouched main at n=1.

Inferred from those exact records and generated statements: the aggregate
region-call spelling bypassed the existing heap-view refresh rule. The fix
refreshes views for the Call's exact heap input cells immediately after the
region call. It changes no source ABI or heap ownership rule.

Focused C/LLVM/Fortran corpus after the fix: c547, c548, c549 and c622 all
PASS, including n=0 and the first growing append. The full typed corpus,
FUZZ_EXT fuzz, identity concordance audit, and paired Linux gate remain
pending in this receipt.
