2026-11-12 MERGE OF THE SOURCE-ONLY / UNANNOTATED-ADMISSION BRANCH INTO MAIN, PREPARED FOR REVIEW (MEASURED in worktree C:/dev/Powershell/turing-divider-merge, branch claude/divider-unstable-merge):

What: codex/divider-unstable-permission-20261111 (4 commits, fabc2d6a..5abf84b8; docs/SITE_UNSTABLE_SOURCE_2026-11-12.md)
merged with main at aac13f4a (49 commits). The codex branch and its worktree C:/Users/alber/turing-divider-unstable-20261111
were not touched; the merge is on a new branch registered to the claude account.

One conflict, examples/meowdoku/authz.py (both sides added a capability). Resolved as the union: CAPS holds
compile_unstable and core_git; the edit tier excludes both; operator keeps main's core_git; admin holds every capability
except compile_unstable, which stays an explicit policy-file grant only (the branch's rule). server.py, source_divider.py
and tests/test_meowdoku_authz.py merged without conflict.

Measured:
- The branch's ten test files on the merged tree: 192 passed (35 s).
- main's tests next to the merged files (branch_gate, core_git, divider_annotation_probe, relay, source_divider):
  49 passed on the merged tree (130 s) and 49 passed on main (118 s).
- Two environment facts, not regressions: the meowdoku server import needs the speaktome checkout beside the turing
  checkout (or SPEAKTOME_REPO), and divider_nodus_bridge finds Nodus tools only at <checkout>/../nodus/tools (no override):
  run from a worktree under C:/dev/Powershell, not under Temp. The unmodified branch head no longer imports against the
  current speaktome ("requires SpeakToMe REST API_VERSION 2"); the merge takes main's rest_api and does.

Not done: no service restart; the live service does not load this code until the change lands and the stack restarts.
The source-only mode stays off: it needs `allow_unstable_source: true` in the SiteCompiler start configuration and an
explicit compile_unstable role grant; neither is set by this change. The full test suite was not run.
