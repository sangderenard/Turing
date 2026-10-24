2026-10-24 E28 FORTRAN RANK-2 PACKED ABI (MEASURED, Ubuntu24 WSL):
At main c8bf22db, a sanctioned lower_ast_source_to_ssa program reading
x[0,1] from [[1,7,3],[4,2,9]] returned 7.0 in C and LLVM but 3.0 in
Fortran. x[1,0] returned 4.0 in C/LLVM but 7.0 in Fortran. The emitted
Fortran dummy was t0(extent_2, extent_3), and the packed C entry handed it
the unpermuted row-major pointer. The executable shell already used
native_shell_abi._fortran_storage_index for this exact boundary.

The packed library entry now uses that same mapping to copy rank-2-or-higher
inputs into Fortran storage order and outputs back into the published host
order. The mapping accepts runtime extent expressions, with positive extent,
size overflow, allocation, and pointer checks before the native call.
Corpus c411 and c412 are in tests/fortran_span_corpus and are run through
the sanctioned lower_ast_source_to_ssa helper by
tests/test_runtime_sized_spans.py. Both failed on untouched main for the
intended wrong value; both pass on this branch across C, LLVM, and Fortran.
The full runtime-sized-spans test file passed 239 tests. The identity
concordance audit remained CLOSED before and after: 26 rows, 2 functions,
0 findings, 0 unsourced. FUZZ_EXT=1 with LLVM and Fortran passed 30/30 seeds.

The full typed corpus completed 387 programs across C, LLVM, and Fortran.
Its 22 nonpass program names were all present on untouched main: 17 were
rechecked by the same typed checker, c550 and c552 were rechecked separately
by that checker, and direct native children on main exited -11 for c547,
c548, and c549. The c398 LLVM diagnostic minted a different SSA id while
retaining the same incomplete reason. No new failing name appeared.

The Ubuntu24 Linux gate ran against this branch and a fresh main worktree
with the required dependencies. Every part log ended in a pytest summary.
Parts aa/ab/ac/rec respectively had 77/7/25/20 failing tests on both trees;
their failing names matched exactly. A synthetic failed pytest command was
reported as NO PYTEST SUMMARY and failed the gate, verifying the vacuous-pass
guard. The gate had no new failing name.
