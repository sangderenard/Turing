# 2026-11-13: C native sequence pointer cells are not payload extents

## Change

The C root wrapper can now allocate one native pointer cell for an exact
DYNAMIC sequence column. The reader joins the current posted descriptor,
its direct canonical/contract declaration, the scalar native ptr SSA value,
and the owner/length/capacity descriptor mints for this arena. Missing,
foreign, or ambiguous evidence refuses. The allocation declarations cite
the proof cells. Logical payload storage requirements remain dynamic;
this does not set the list length or its capacity to one.

## Verification

* New native and proof-negative suite: 7 passed in 6.15 seconds.
* Exact same suite with only the C wrapper change removed: 6 passed,
  1 failed in 4.71 seconds. The native dynamic-list case refused private
  root storage before the change. Candidate bytes restored exactly.
* Native C uses two independently growing lists at sizes 0, 1, 17, 4097,
  repeated per execution. Their scalar result matches Python. The payload
  storage requirements remain dynamic before and after emission.
* Wrong column, foreign owner, absent descriptor, nonscalar native shape,
  wrong native type, and ambiguous descriptor membership all refuse.
* Actual GraphSolver compile_nodes C emission advances from four private
  pointer-cell extent refusals (96, 104, 105, 106) to zero shortfalls.

## Limits

The actual GraphSolver artifact was neither compiled nor executed. Zero
emission shortfalls do not establish safe lifetime for its selected owned
record return. A separate public ownership escape guard and eventual owner
transfer protocol are required before that result may run. This change
proves a pointer cell's storage width, not local payload ownership, public
escape authorization, or support for arbitrary linked/adopted descriptors.
