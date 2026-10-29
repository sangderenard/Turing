2026-10-29 TRY PROTECTED TRANSFER OWNERSHIP (MEASURED, Windows worktree)

The live source Try control previously carried a handler class and a
continuation Phi but no exceptional transfer site. It now reads
`SOURCE_EXCEPTION_TRANSFER` cells whose exact catch-owner source cell maps
through the canonical edges to this forked Try. Its `CONTROL_BLOCK.sites`
field cites those cells as provenance. A transfer with a disagreeing owner
refuses; no source position or class-name match substitutes for the edge.

The nested bounded case has two authored `raise E(...)` statements. Each is
two separate fallible operations: construction of E (a Call) and the Raise
after construction. The inner Try owns two transfer cells for its body;
the outer Try owns two from the inner handler's outward route. A first test
assumption of one site per Raise failed [2,2] versus [1,1]; the corrected
assertion checks both cells in both control facts. This is source ownership,
not proof that each call's status or the raised status is dispatched by SSA.

The combined source exception and ownership suites passed 16/16 in 3.83 s.
The concordance audit remained red before and after at the preexisting
selected-return reachability refusal. `git diff --check` passed.
