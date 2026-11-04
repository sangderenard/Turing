2026-11-04 Dynamic arange count and computed metadata transport (MEASURED)

Isolated checkout: C:/Users/alber/.codex/worktrees/compressor-results/turing.
Bounded sanctioned repro: `$env:PYTHONPATH=(Get-Location).Path; python -u
C:\Users\alber\AppData\Local\Temp\probe_dynamic_arange.py`. Source returns
`AbstractTensor.arange(bit_count, cls=type(lengths))`; ProgramABI binds
`bit_count` as int64 scalar and `lengths` as int64 rank-one span. The original
8.98-second lower completed SSA with region arange result5 semantic int64,
rank-one span, physical dtype unset. C emission reported arange with no
module-lane spelling and a public result with no extent origin. Its one
region operand is the exact int64 scalar formal0. The authored call has
SOURCE_CALL_PARAMETER_DEFAULT refs for end=None, step=1, dtype=None.

Tensor lowering now clamps the bound to nonnegative int64,
uses a numeric-contract checked int64-to-int32 Cast, maps overflow to a safe
zero count before allocation/kernel, and produces dynamic shape/rank/count
SSA values for create_arange_int64. The first rewrite reached a final audit
refusal: the callee region's count/shape/rank IDs were minted there but the
root aggregate call projected only result5, so copied accounting named three
IDs absent in root. The common `publish_computed_region_result_metadata`
writer now appends those three exact callee values to the region Ret, adds
their positional aggregate outputs and caller Loads, and posts
TENSOR_METADATA_RESULT_BINDING. The call metadata reader maps through that
page to distinct caller IDs. The new audit verifies the exact Ret, Call,
GetElementPtr/Load positions, accounting IDs, and cited SSA_VALUE identities.
The bounded source now completes final SSA audit. Its only C emission
shortfall is the public return buffer's computed extent allocation origin;
native arange parity is not yet claimed. The root C/runtime work owns that
remaining public ABI handoff.

A separate exact writer defect was observed with SSAValue.__setattr__ hooks:
the new checked Cast output3960 is minted int32, then
`ir_indexing._propagate_scalar_dtypes` rewrites it to int64. That pass treated
every narrow non-Const/non-Load integer as inferred, even a Cast with explicit
target_dtype int32. It now keeps a Cast's declared target width under its
existing SCALAR_INTEGER_WIDTH.KEPT_DECLARED receipt and checks the SSA_VALUE
row. Focused `tests/test_scalar_merge_dtypes.py::test_explicit_integer_cast_keeps_declared_narrow_target`
passed in 1.08s. The bounded arange repro after this fix still refuses the
three missing cross-frame metadata IDs until the output bridge above is
installed. `tests/test_dynamic_arange_metadata.py` then passed in 5.48s,
including an output-binding tamper that the new audit rejects. Mapping audit:
28 rows, zero findings, zero unsourced, CLOSED. A neighboring test suite had
16 failures; representative broadcast and unbooked scalar-merge failures
were reproduced with the same trace on untouched canonical and are not
attributed to this change.

Private scalar consumer follow-up (current repair checkout):
`$env:PYTHONPATH=(Get-Location).Path; python -u
C:\Users\alber\AppData\Local\Temp\probe_dynamic_arange_scalar_float.py`
uses the authored chain `arange(bit_count).to_dtype("float64").sum().item()`
under the int64 scalar input contract. An 8.0-second bounded run now reaches
`shape_transformation_concordance` for root value8 / `sum_double` formal1240
and refuses a REVISE with no changed source (prior row revision10315). Native
scalar parity for counts 0, 1, 7, negative, and oversized is unmeasured.
The int64 `.sum().item()` variant first refused an unrelated native
`sum_double` / `cast_double_to_int_values` operand contract; it did not
validate arange execution.

The first wrong metadata writer was captured by
`probe_dynamic_arange_account_writes2.py` (8.55 seconds). Region0 arange
result6 initially held native metadata IDs 3996/3998/3995. Region1
`propagate_repository_ssa_call_metadata.enrich` at line2839 then wrote root
projection IDs 4003/4006/4009 onto that same SSAValue object because
`scoped_values` groups source-owner planned regions and includes foreign
operand objects. Later `settle_call_result_types._publish_type` only mirrored
the already changed accounting; it was not the first writer. The metadata-key
write now targets function-defined formals/results rather than a foreign
operand reference. The current scalar consumer reaches the earlier
sum_double shape row refusal above, so this follow-up is not complete.
The exact direct arange output test and static topk native regression pass
together (2 passed, 9.44 seconds). The static topk regression was separately
fixed by limiting `callee_output_ids` metadata interpretation to Calls
stamped by TENSOR_METADATA_RESULT_BINDING; the exact static test passed.

The current sum-double shape refusal was measured further with scratch
`probe_dynamic_shape_writer.py` (8.02 seconds) and
`probe_dynamic_shape_append.py` (~33 seconds). The exact same accepted
`sum_double` formal1240 `SSA_VALUE` column0 and root value8 column0 support
three `ssa_metadata_transport` edges whose target rank is 1, followed by a
target rank 0 edge with no newer cause. The rank-1 occurrences carry
`program_abi_storage=span` and `program_abi_rank=1`. The rank-0 occurrence
is the region2 formal and Call arg0, with `ssa_storage_alias=8` and
`ssa_region_feed=(2,0)` but no rank key. `scoped_values` groups the four
objects under the same source-owner value ID and asks one shape row to
represent both physical one-cell output storage and the scalar feed. The
first missing proof is a distinct physical-storage versus semantic-result
shape identity for that call edge. No semantic rank or native parity is
claimed from the accepted kernel pointer formal alone.

Current authored scalar-consumer SSA now succeeds. The source is the same
`arange(bit_count).to_dtype("float64").sum().item()` and the bounded command
above completes in 7.54 seconds. `_known_count` had treated each typed
unresolved display shape `()` as a one-element scalar; generated region1
cast and region2 sum therefore used literal count 1. The tensor descriptor's
unresolved state now prevents that static-count claim. The existing
`ensure_dynamic` writer mints count/shape/rank extent SSA with posted int32
physical facts and the exact source's posted rank; the cast's output
descriptor retains those shape-preserving input extents. Region1 and region2
now use their local element-count extent values. A rank-1 `SSA_VALUE` row and
dynamic descriptor on region2 input8 settle the previously discordant
region-feed occurrence at shape-edge publication; its accepted kernel
pointer alone is not treated as tensor geometry. The computed-result bridge
posts two TENSOR_METADATA_RESULT_BINDING rows; the next region calls receive
metadata actuals/formals.

TENSOR_METADATA_CALL_BINDING initially keyed the mutable instruction index.
In a bounded hook, rows 47/60 named live Calls at first publication but
named GEPs after a later tensor-lowering invocation shifted those Calls to
53/62. The existing booked aggregate Call result IDs remain stable. The
page now keys by that exact SSA result, and both writers cite its SSA_VALUE
Ref; the audit checks the unique live Call and paired actual/formal operands.
Its focused moved-index and forged-result test passed in 5.79 seconds.
The C private extent reader requires that input page plus the computed
result binding's exact shape/rank/count projection for the same tensor, then
aliases the local extent result to its formal. It does not search for a
public storage origin. Bounded C emission is now complete with zero
shortfalls (~7.96 seconds); `tests/test_dynamic_arange_metadata.py` passes
2 tests in 9.23 seconds. The unrelated `_read_header` public input regression
caused by overbroad local-occurrence filtering was repaired at enrichment:
only three frame-local metadata IDs are filtered. Its exact public test
passes in 6.55 seconds.

Native scalar parity is RED. A first native attempt returned 0.0 for counts
0, 1, 7, and -3. Generated C at
`C:\Users\alber\AppData\Local\Temp\arange_native_tx6xyxev\dynamic_arange_probe__root.c`
declares `int64_t callout6_0` and `double callout8_0` (lines 68 and 71),
then passes their addresses as dynamic arange/cast output buffers (lines
127-129). Count greater than one can write beyond these one-cell outputs;
native execution was stopped. The missing physical producer is caller-owned
dynamic capacity for private computed span outputs. No successful native
arange parity or overflow-trap result is claimed.

Storage/ownership follow-up: bounded
`C:\Users\alber\AppData\Local\Temp\probe_dynamic_arange_storage.py`
(7.58 seconds) shows region0 output6 and region1 output8 have dynamic
SSATensorDescriptor rows and SSAStorageRequirement.dynamic=True with
element_count=None. Root has no tensor descriptor for either output;
its SSA occurrences are span rank1 with projected count IDs, but
module_storage_requirements(root) reports dynamic=False and no count.
The C aggregate output allocator (`ssa_c_backend.py` around 3677) then
uses the empty display shape and one scalar cell. LLVM's existing
runtime-span allocator refuses without a measurable/public extent;
Fortran/C/LLVM share explicit `ssa_heap_allocation.heap_allocate` for
sequence-owned pointer cells. A sequence descriptor names its
heap_owner_value_id and carries owner with column pointer cells through
helper formals. SSATensorDescriptor has data_value_id and dynamic extent
IDs but no heap owner/pointer-cell relationship. `heap_allocate` produces
a separate ptr-typed carrier; reusing the current int64 tensor aggregate
slot would make C Load interpret pointer bits as int64. No existing tensor
return bridge between logical tensor ID, ptr carrier, and owner was found.
The next implementation requires an exact tensor ownership/return record
before the incumbent heap allocator can safely be used here.
