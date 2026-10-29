2026-10-29 TRY CONTINUATION JOIN OWNERSHIP (MEASURED, Windows worktree)

The bounded public-source catch compiles to a source Try with y assigned
from the protected int conversion on success and a folded constant 2 in its
handler. The reducer already posts one `SOURCE_EXCEPTION_CONTINUATION_PHI`
with exact binding cells for those two arms. Before this change, the live
`NATIVE_TRY` control fact carried only the handler class value; the join
had no control owner.

The source Try constructor now follows the Phi's ingestion cell through
the original and forked `CANONICAL_VALUE` edges and requires the result in
the forked graph. `NativeTryBlock` carries the Phi fact cell, and its
`CONTROL_BLOCK` fact derives from that same cell. The bounded control fact
now has the handler class canonical cell and one continuation Phi fact cell.
This records the join; it does not yet execute its values or exceptional
predecessors. A folded handler assignment still needs an exact execution
slot, and the full pending-outcome SSA route remains absent.

The combined source exception and control ownership tests passed 16/16 in
3.73 s. `tools/audit_identity_concordance.py view` remained red before and
after at the existing selected-return reachability refusal. `git diff
--check` passed.
