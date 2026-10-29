2026-10-29 FIRST SOURCE TRY SSA STATUS CHECKPOINT (MEASURED, Windows worktree)

The bounded public source `try: y = int(x); except ValueError: y = 2`
still refuses admission. Capturing its planned control program and lowering
that program directly now emits a status take after the sole protected
region, a nonzero predicate, and failure/resume SSA successors. A new
SOURCE_EXCEPTION_SSA_CHECKPOINT fact cites the deployed source site,
status value, predicate and both blocks; the audit checks the minted
status source, predicate operand cell set, block ownership and provenance.
The Try's exact CONTROL_BLOCK cell owns both checkpoint blocks. The first
version created them under a marker StatementBlock and recorded
SSA_BLOCK_OWNER_UNROUTED; passing the enclosing Try cell to the existing
block writer corrected that identity.

The failure path re-reports the taken code and enters the existing native
fault exit. `native_try_dispatch` remains an explicit lowering shortfall:
handlers, joins and pending finalizer outcomes are not yet executable.
The public gate remains closed. The focused exception, control ownership
and native status suites passed 25/25 in 7.77 s. The broad concordance
audit remains red at the preexisting selected-return reachability refusal.
