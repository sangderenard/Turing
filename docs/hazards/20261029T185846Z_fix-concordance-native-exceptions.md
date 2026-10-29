2026-10-29 TITLE (MEASURED, where): 
2026-10-29 EXCEPTION BRANCH NOT ACCEPTED (NO COMPILER REPAIR MEASURED)

## Decision

Do not use `fix/concordance-native-exceptions` as a working exception-handling implementation or merge draft PR #1194. The branch has not established correct compilation, native execution, or even a measured source-to-SSA exception-topology invariant. It is a record of an attempted representation, not evidence that `try`, `except`, `raise`, or `finally` now lower correctly.

## Exact branch state

- Base: Turing `main` at `bfa5c73c`. Head before this report: `8f06253d`. Draft PR: https://github.com/sangderenard/Turing/pull/1194.
- Three implementation commits: `2a62ce94`, `50da5dde`, `8f06253d`. They add 287 lines across six compiler files relative to that base.
- The source exception fact stores scoped ProcessGraph node references, operand-edge pairs and roles. It is posted immediately after source graph construction. The SSA module receives a separate view after final loop-scope reconciliation.
- No exceptional CFG edge, handler dispatch, `finally` transfer, per-try state struct, native backend implementation, or fail-closed check for unlowered exception paths was added. The reducer's existing `before = dict(environment)` remains.
- No targeted native compile, concordance-audit run, codec compile, or regression test was completed on this branch. `git diff --check` was the only reported check. The claim that the graph contains every relevant `try` has not been measured.
- Earlier PR #1166 was closed and its implementation reverted before this branch. Its former native results are not results for #1194.

## Why the work is not usable

The work did not start from a bounded reproduction of the compiler refusal or the first wrong source/control/SSA record. Edits were made before the control-flow boundary and identity trail were established. The proposed representation changed repeatedly in response to corrections: a per-function graph, an invented graph abstraction, a whole-source graph, and then a late SSA view. The first graph placement was after reduction even though reduction removes nodes; that placement had to be reversed. None of these versions was compiled and inspected through the sanctioned entry.

The assistant also conflated an actual lowering refusal with the existing `ValidationBlock` path. In current source, that path records an error code and branches to normal continuation; C and LLVM runners read the code after the native entry returns. That is a concrete semantic defect for a catchable authored `raise`, but it is not proof that a compiler refusal was detected and ignored. The assistant first called the lowering problem a hard failure, then described this separate runtime mechanism, without naming an observed lowering error. Those statements did not provide a reliable progress report.

The new concordance audit checks the graph fact's node provenance and internal shape, then checks that the SSA view repeats that fact. It does not establish that the graph captures all authored exception relationships, that its edges become executable exceptional successors, or that any backend preserves Python's control behavior. A valid audit of this limited fact would therefore not certify exception lowering.

## Continuation boundary

Treat #1194 as unverified scratch. A usable continuation starts from Turing main or another clean integrated checkout, identifies the exact first failing compile through the public `lower_ast_source_to_ssa` entry, and records the failing source node, concordance cells, writer, consumer, and refusal. The first compiler invariant to establish is fail-closed admission: every retained authored exceptional transfer must have a proven SSA exceptional path or a declared native-backend mapping; otherwise compilation refuses before artifact emission. Only then can per-try state and handler/finalizer methods be specified against live SSA and loop-scope cells. Do not report a working codec or exception system until native compilation and execution have been observed.

Unrelated untracked `.claude/`, `e HEAD`, and `run_windows_battery.ps1` were present in the checkout and were not changed by this work.
