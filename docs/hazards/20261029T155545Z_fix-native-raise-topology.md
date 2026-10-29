2026-10-29 TITLE (MEASURED, where): 
# 2026-10-29 Protected native call failure edge

The retained Try's exact deployment region/source Call pair now emits a
native status take immediately after the region call and before reading its
outputs. SSA routes Python-semantic statuses to the matching handler through
the concorded class topology. An uncaught status is reported outward. Native
width limitations have no exception-class row, so an authored handler cannot
catch one as if Python had raised. The native status-to-class mapping is
posted from static class definition cells into `NATIVE_STATUS_CLASS`; the
exception audit checks each class and provenance edge.

The C and LLVM module emitters now support status take and a dynamic status
report. Their `int(float)` conversion reports NaN as ValueError, infinity
as OverflowError, and finite values beyond the int64 slot as a compiler
width limitation before any undefined native conversion. The failed result
is inert and the status branch runs before an output load.

Measured native execution: the two-backend protected-cast test passed for
2.4, NaN, infinity, and 1e30; the three native C corpus cases c1166-c1168
passed. Combined targeted run: 5 passed, 651 deselected. `py_compile` and
`git diff --check` passed. The focused test's `CorrelationTable` audit had
zero findings. The broad `view` audit still has the earlier selected-return
reachability refusal and is not green.

The bounded public compile of the real codec passed the protected-call
failure-edge refusal and next stopped at an unaccounted formal for source
Slice node 75 (`:limit`, dispatch metadata). That is a distinct compiler
identity problem. The codec artifact is not compiled. Finalizers, handler
entry value merges, nested Try dispatch, and generic protected call fault
sources remain unfinished. A separate focused protected helper-call probe
still refuses because its callee Raise lacks a concorded class definition.
