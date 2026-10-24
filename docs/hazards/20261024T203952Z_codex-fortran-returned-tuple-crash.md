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
PASS, including n=0 and the first growing append. The existing Fortran
native-execution contract suite passed 7 tests. The identity concordance
mapping audit reports 26 rows, 0 findings, 0 unsourced on both trees.

The full typed corpus finished 388 programs and 1,164 C/LLVM/Fortran status
lines. Against the prior complete 387-program baseline, only c547-c549
changed from Fortran RUN-CRASH -11 to PASS; the three new c622 backend statuses
are PASS. The remaining 9 nonpass backend entries across 8 programs are c22,
c367, c398, c532, c536, c537, c538 and c550. All eight reproduce by the
same checker on untouched fresh main 79b86a3d. FUZZ_EXT=3 seed 76 with LLVM
and Fortran finished 30/30 PASS.

The authoritative Ubuntu24 gate ran against branch 7d6b0dfd and a fresh
79b86a3d main worktree after installing requirements.txt, llvmlite and
ziglang. Every part log ends in a pytest summary. Failed/passed counts are
aa 77/310, ab 7/255, ac 26/475, rec 20/255 on both trees, with identical
failing names. Both trees report one name beyond the committed ac baseline:
test_precompile_to_ssa.py::test_numerical_lowering_routes_scatter_through_real_llvm_algorithm
expects index_assign_double where merged #926 now emits scatter_sum_double.
This branch adds no failing name relative to fresh main; the committed ac
baseline therefore still needs separate attention.
