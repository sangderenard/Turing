2026-10-29 SOURCE TRY CONTROL OWNERSHIP (MEASURED, Windows worktree)

The live planner previously correlated only loop and conditional control
programs. Its `NativeTryBlock` type had no program constructor, and the
overlay's region-marker walk and nested embedding did not traverse its four
authored arm kinds. The nesting helper now accepts prospective Try programs
and records exact AST sites for body, each handler, else, and finalbody.
The overlay now visits Try arm markers, reports each arm as a distinct marker
scope, and embeds a nested region-owning control in its one source arm,
refusing a region set spanning multiple arms.

Two bounded structural tests passed (1.51 s and 1.33 s). They check the
authored arm sites and a nested conditional embedded in the Try body without
moving the handler marker. The complete ownership test file then passed
9/9 in 1.54 s, then 9/9 again in 1.66 s after adding marker-scope paths.
These tests do not establish a live Try constructor or native
SSA exception control. `tools/audit_identity_concordance.py view` was red
before and after the edit at the existing selected-return reachability
refusal. `git diff --check` passed.
