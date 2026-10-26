# Lane E parked report (2026-10-24)

## E26: C module drops Precision-typed formals (UNMERGED, ungated)
`e26_c_precision_formals.patch` (beside this file) is a commit made on main
51330ad8. `is_structural_abi_value` no longer counts Precision parameters as
record descriptors, and the C `prepare_execution` refuses a feed that has no
buffer. With the patch, the C buffer order for the sin core matches LLVM's
32 slots, and the new test fails before the change and passes after it.
The precision thread found the root cause upstream, in
`_typed_parameter_record_receipts`. Apply their fix first. If E26 no longer
reproduces, keep only the unknown-feed refusal.

## E28: Fortran rank-2 spans read row-major memory as column-major (silent WRONG, unfixed)
Fortran declares the dummy as `t0(extent_2, extent_3)`, but C passes the
buffer row-major. So `def f(x, y): return x[0, 1]` with x = [[1,7,3],[4,2,9]]
and static shape (2, 3) returns 3.0 on Fortran; C and LLVM return 7.0.
`x[1, 2]` is wrong the same way. With runtime-sized rank 2, Fortran refuses
loudly instead ("feed 0 has 6 elements; ABI needs 2").
Inside the Fortran lane, arrays follow logical == Python, and its reshape
code converts with `order=` to keep that true. Only the host boundary is
wrong, and every bind(C) function, regions included, sits on it.
Proposed fix (decision pending with Albert): a row-major ABI. Declare
rank>=2 dummies with reversed extents, copy them in to Python-order locals
with `reshape(..., order=[n..1])`, and copy outputs back out at every exit.
Fortran-to-Fortran array actuals of rank>=2 must then pass the raw form.
Corpus numbers c411 and c412 are reserved. The repro uses
tests/test_runtime_sized_spans.py `_lower`/`_run` with rank 2, shape (2, 3).
