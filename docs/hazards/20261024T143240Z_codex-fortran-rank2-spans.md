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
The full runtime-sized-spans test file passed 239 tests. The full typed
corpus, fuzz, post-change concordance audit, and Linux gate remain pending
at this draft receipt.
