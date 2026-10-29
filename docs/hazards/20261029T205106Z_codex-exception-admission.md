2026-10-29 LIVE TRY CONTROL CONSTRUCTION (MEASURED, Windows worktree)

The precompile gate previously refused authored Try before its function shell
had deployment regions. A bounded hook observed the module shell at that
point had zero subgraphs; its planned `root` shell had one region. The gate
still posts every `SOURCE_EXCEPTION_ADMISSION` obligation immediately, but
now lets the matching function shell build one source Try control tree before
it refuses. No SSA exception route executes past that refusal.

The first constructor version looked up a handler class in the fork's read
scope. The fact was under the reducer's original ingestion scope, so all six
admission tests failed with `source Try control has no canonical value scope`
when the module shell was incorrectly selected. The function-shell guard
corrected that selection. A bounded book probe then showed the exact chain:
the `ValueError` occurrence's ingestion cell points to canonical value 2 in
the original function scope, and the fork's canonical value derives from
that original canonical cell. The constructor now walks only those recorded
canonical edges and requires one target value. The same chain maps the Try
and handler source cells to forked control IDs 4 and 3.

The live bounded case posts one `NATIVE_TRY` control block with one carried
handler class cell, while `SOURCE_EXCEPTION_ADMISSION` stays unresolved.
The combined source exception and ownership tests passed 16/16 in 3.69 s.
`tools/audit_identity_concordance.py view` remained red before and after at
the preexisting selected-return reachability refusal; `git diff --check`
passed. A handler assignment of a folded constant still has no region block,
and the control tree does not yet carry that arm's value production. Native
C/LLVM exception execution is not claimed.
