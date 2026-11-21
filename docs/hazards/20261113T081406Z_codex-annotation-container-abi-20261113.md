# LLVM consumes existing public sequence capacity and length contracts

The existing source-proved bytes-slice contract was implemented in the C wrapper but not LLVM. LLVM could emit an artifact that ran with zero input descriptor lengths and returned empty slices. It now consumes the existing public input and output descriptor relations: initialize input lengths/capacities, allocate caller-owned result capacity from the exact input buffer, and expose the separately written logical result lengths. Repeated execution retains the full backing arenas while publishing views of the current lengths. No filtered result is assigned an input's logical length.

The runtime-result ABI audit admits an exact LLVM entry/description alongside C and validates the corresponding extent integer type (LLVM int32_t, C int64_t). Existing descriptor/source/port provenance checks remain unchanged. LLVM emits API_CONTRACT and runtime ABI receipts from its wrapper/formal cells; changing a result length relation is refused.

Verification: native bytes slices cover empty inputs, clipped and negative offsets, byte values0/127/128/255, repeat calls, shrinking then growing outputs with one pointer table, retained arrays after execution object release, and mismatched capacity override/tampered ABI rejection. Full concordance passes for the test. Expanded six-file suite:11 passed,2 failed,3 errors; identical HEAD LLVM baseline:10 passed,3 failed,3 errors. Only the new test is repaired, no candidate-only failure. Mapping audit:28 rows across2 functions,zero findings/unsourced,latch CLOSED.

This is the existing bytes contract only. It does not establish a new comprehension bound, returned-record ownership, or make GraphSolver runnable. Those remain separate work. No push, merge, or deployment.
