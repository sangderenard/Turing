# 2026-10-29 Finalizer needs an exceptional lexical-value join

Probe: `root(x: float, n: int)` loops over a Try whose body assigns
`z = int(x)`, whose conversion handler raises TypeError, and whose
finalbody assigns `z = z + 1`. The clean branch refuses the authored
Try at loop composition with `try-exception-entry-state-not-composed`.

The source exception transfer for protected Call 15 records the entry
binding `z=3`. The handler-entry receipt also selects `z=3`. The graph's
finalbody BinOp 16 has a `lhs` edge from Call 15, the *successful* `int(x)`
result. The reducer's Try handling merges only the successful body and
continuing handlers before reducing finalbody; this handler terminates, so
the successful `z=15` remains the only environment used for that read.
Thus the graph asks the exceptional finalizer to read a value produced only
by the success edge.

An attempted SSA finalizer branch reached the final concordance audit and
was refused twice: `native_try_exception_finally#0 Call reads a value
defined at [('native_try_call_success', 1)]`, for the direct raise and
unmatched protected-call edges. All attempted source edits were reversed;
no native finalizer behavior was shipped. The needed next change is a
source-owned finalbody entry merge of each name from its successful and
exceptional predecessor values, with an SSA Phi or edge-selected binding
that the finalbody's region formal reads. The finalizer's outgoing status
must then preserve or replace the pending exception according to whether
finalbody itself raises.

The same probe confirms why merely lifting the loop-composer gate is
incorrect. The `SOURCE_EXCEPTION_TRANSFER` already identifies the
finalizer chain; the missing piece is the finalbody value identity and
its executable SSA join. No codec artifact was produced in this probe.
