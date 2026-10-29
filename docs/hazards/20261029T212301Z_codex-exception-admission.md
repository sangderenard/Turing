2026-10-29 STANDALONE RAISE ADMISSION (MEASURED, Windows worktree)

The bounded public source `def root(x: int) -> int: raise ValueError('bad')`
previously lowered as a complete module. Its SSA recorded native validation
code 3, and `validation_fault(3)` produced RuntimeError rather than the
authored ValueError. The source reducer had already posted an exact
SOURCE_EXCEPTION_TRANSFER for the Raise and a static ValueError declaration,
but no SOURCE_EXCEPTION_ADMISSION row existed without an enclosing Try.

Structural precompile now posts an unresolved admission row for each retained
authored Raise or reraise whose source transfer matches the graph's exact
Raise node. The final precompile gate names these source cells separately
from Try owners. The bounded program now refuses before SSA emission, and
its status-route record still cites ValueError with no selected handler.
The admission audit describes either kind of retained exceptional control.

The combined source exception, native status and control ownership suites
passed 24/24 in 8.18 s. The broader concordance audit remained red at the
preexisting selected-return reachability refusal. Three loop suite tests
failed; all three test names also failed on untouched canonical Turing at
C:/dev/Powershell/turing. Two had the same unsettled dtype refusal on both
checkouts. The guard-raise case had an unsettled dtype refusal on canonical
Turing and the new explicit Raise admission refusal here. `git diff --check`
passed. No authored Raise is yet admitted as executable native control.
