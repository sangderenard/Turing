2026-10-29 PROTECTED CALL STATUS ROUTES (MEASURED, Windows worktree)

SOURCE_EXCEPTION_STATUS_ROUTE previously covered only an authored Raise
with a statically resolved class. A protected Call can report several
native exception codes, and a reraise carries a runtime class. The source
topology now posts a route for each concorded catchable native status for
Call and reraise transfers. An authored Raise still posts its one exact
class status. Each route cites the source transfer, native class cell,
ordered examined handler cells, selected handler, and unwind owners.
The audit checks every route and completeness over the status class table.

For `try: y = int(x); except ValueError: y = 2`, the source Call's NaN
status 97 selects the handler, while infinity status 98 has no matching
handler. The source exception and control ownership suites passed 18/18
in 3.11 s. The broad concordance audit remains red at the preexisting
selected-return reachability refusal. Native SSA does not yet dispatch
these routes; authored Try admission remains closed.
