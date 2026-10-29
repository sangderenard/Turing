2026-10-29 SOURCE TRY ADMISSION (MEASURED, Windows, main 65755b6a)

Program: `try: y = int(x); except ValueError: y = 2; return y` through
`lower_ast_source_to_ssa`.  On untouched main the source Try, protected Call
and handler each had an `INGESTION_VALUE` cell, but the only `CONTROL_BLOCK`
was a statement region.  SSA emitted Cast and Ret with no exceptional branch.
The C artifact reported complete with zero shortfalls: x=2.4 returned 2;
x=NaN returned -9223372036854775808 instead of entering the handler and
returning 2.  A bounded loop slice of the real octet conversion handler
already refused at loop composition with `Raise` as blocker; the non-loop
case exposed the missing artifact-admission check.

Rule: structural precompile posts an unresolved `SOURCE_EXCEPTION_ADMISSION`
row derived from the exact retained source Try cell, then refuses with a
`source-exception-control-owner` boundary.  The audit reads the new page and
reports any unresolved row on a completed module.  No control-block reference
is accepted as proof of executable exceptional semantics.  The generated
resource-scope Try is identified by its existing explicit rewrite marker and
is outside this authored-Try gate.

Focused public-source tests: 3 passed (authored catch refusal, authored
finally refusal, plain cast unchanged).  A focused failed compilation's book
had one admission row, with `Unresolved(source_exception_control_unlowered)`
and exactly the source Try as its provenance cause.  The broader
`test_precompile_to_ssa.py` plus `test_orphaned_loop_refusal.py` gate was
134 passed / 13 failed on both the change and untouched main, with identical
failure names.  The context/resource gate was 8 passed / 20 failed on both,
also with identical failure names.  `tools/audit_identity_concordance.py view`
remained red before and after at `selected return has no exact source call
reachability authority`.  No native exception execution is claimed.
