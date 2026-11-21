# 2026-11-13: C refuses unpublished owned sequence escapes

## Change

The C wrapper now mirrors LLVM's existing return guard: a compiler-owned
DYNAMIC sequence with an owner cannot be published as a normal pointer-cell
array unless the exact public-owned or bounded-output contract admits it.
Borrowed spans retain their existing public ABI. Incomplete artifacts cannot
be compiled by the normal C artifact API.

## Verification

* New C guard/native tests, pointer-cell tests, public-owned C tests and
  LLVM-owned tests: 13 passed in 40.57 seconds.
* Same suite with only this C guard removed: 12 passed, 1 failed in 34.76
  seconds. Only the selected-owned-record refusal test changes. Candidate
  backend bytes restored exactly after the comparison.
* New negative test checks the selected owned record result refuses before
  compilation and that no native build directory is created.
* Direct owned bytes concatenation and borrowed bytes alias execute natively
  in C on empty, binary, and longer inputs with repeated executions.
* The previous four-fragment C bytes-concat private-root-storage failure now
  passes its existing native test because the preceding pointer-cell proof
  supplies the exact private arena-handle storage. This is a verified change,
  not an assumed unchanged baseline failure.
* Actual GraphSolver C emission now has exactly the intended return shortfall:
  owned sequence 100 has no validated public ownership escape contract.
  Its former four pointer-cell extent refusals are absent.

## Limits

The actual GraphSolver artifact was not compiled or run. Selected record
owner movement, retained host materialization and release still require their
own proven protocol. During the baseline a separate module's read-only
public-entry validation functions were refactored; repository call-site
inspection confirmed those functions are not invoked by this test suite.
Its existing entry publisher and authority writer were unchanged.
