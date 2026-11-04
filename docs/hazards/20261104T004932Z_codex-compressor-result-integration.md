2026-11-04 Tensor result runtime ABI audit (MEASURED, isolated compressor-results worktree):

The existing C v2 RUNTIME_ABI receipt now audits output capacity and computed
extent indices through `public_tensor_capacity_sources`, in addition to the
existing bytes slice relation. The artifact must directly cite the exact
TENSOR_OUTPUT_CAPACITY_BOUND and TENSOR_METADATA_RESULT_BINDING cells and the
four root SSA_VALUE cells for output, input, shape and count. The returned
tensor's incumbent declared rank selects the logical length count port for
rank 1 or the shape vector port for rank greater than 1. Missing ports,
wrong indices, unproved bound/binding, or a rank disagreement fail audit.
The source root is selected by those cells, not an exported entry name.

Bounded public guarded topk with ProgramABI input shape [3] emits C v2
capacity output15→input1 and length output15→computed count; runtime ABI
audit finds zero issues. The same source with shape [3,4] and axis0 initially
refused two TENSOR_OUTPUT_CAPACITY_BOUND findings: the writer computed count
as Mul(Min(k,3), Const4), while its audit accepted only a direct Cast(Min).
Boundary's exact Mul/product audit correction made that rank2 source lower
and emit complete C v2 with capacity output15→input1 and logical shape
output15→the computed shape vector; runtime ABI audit finds zero issues.
The rank2 refusal was not a missing capacity writer.

`tests/test_tensor_runtime_abi_audit.py`: 2 passed in 9.98 seconds, including
forged capacity, length and shape index negatives. Runtime extent and
sequence capacity tests: 27 passed in 5.13 seconds. Mapping audit: 28 rows,
zero findings/unsourced, CLOSED. No full compressor or rank2 native execution
claim follows from this bounded ABI proof.
