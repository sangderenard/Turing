2026-10-26 TITLE (MEASURED, where): 
# Caller-owned mapping contexts and explicit fixed child bounds

The no-allocation source fixture declares `Index.table: dict[int, list[int]]`.
Its annotation previously produced no field receipt, so the root had no
descriptor while a late callee-record copy referred to one. The existing
annotation receipt writer now retains that nested list descriptor reference.
The source contract is DYNAMIC: the test supplies declared public pointer
cells and the existing context owner. It does not reinterpret fixed arrays
as owned pointer cells or invent a source capacity policy.

C's API contract already declared those public cells as `ptr`, but the layout
header accepted only numeric storage classes. The header now also prints the
canonical pointer layout from dtype_layout, preserving prior numeric enum IDs.
It does not describe pointer storage as int64. C, LLVM and Fortran each passed
six persistent transactions: missing before allocation, repeated same-key
append above 2**53, different-key isolation, missing after allocation, and
reading the earlier key again. Source concordance remains zero findings.

Fixed arena coverage uses the incumbent ControlProgram sequence lowering with
an explicit FIXED descriptor. The descriptor's parameter-ownership writer had
omitted child pool members; those exact members now derive the same parameter
ABI fact. The setdefault writer also explicitly narrowed the int64 row handle
to generic int/int32 before an int64 table-store call. Lookup, new handle and
result Phi now use the declared parent handle column dtype. The unit runs the
existing constant hoisting, precision and physical-call adaptation transactions.

`python -m pytest -q tests/test_fixed_child_pool_bounds.py -x --tb=short`
completed: 9 passed in 46.34s. Three valid native executions wrote exactly one
row. Six isolated undersized outer/pool cases rejected with validation75.
C/LLVM preserved every supplied arena and counter; the lookup status starts
at its declared absence status0. Fortran ERROR STOP75 runs in subprocesses.

`python -m pytest -q tests/test_mapping_child_sequence_views.py -x --tb=short`
completed: 15 passed in 44.06s, including the original six bounds checks now
using the explicit fixed helper, source context persistence on all backends,
and injected dominance/selection audit failures. This is a bounded native
receipt; no whole-core lowering or persistent String Core service was run by
this lane. The separately observed root count-formal identity baseline remains
under the integration lane's investigation.
