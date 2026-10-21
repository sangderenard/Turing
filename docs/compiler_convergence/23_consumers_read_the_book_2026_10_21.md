# Continuation 2026-10-21: consumers read the book, producers post

Canonical repository `C:/dev/Powershell/turing`, branch `main`. This report
covers the commits from the unverified checkpoint b4bdb2c1 (the previous
agent's structural provenance core, committed as found) through wave 3.
No bootstrap and no native build were run in this span; every claim is a
test gate, a seconds-long probe on real source, or a baseline on a clean
worktree of the checkpoint (b4bdb2c1) or its parent (65b4fc06).

## Method

Narrow Sonnet agents on rails: read-only fact-finders that trace a failure
to the two records that disagree and the writer, and correctors that edit
only the named region and verify on their own node ids plus a file gate.
Every failure was classified against the clean worktrees before being
called new or old. The user's two laws governed every edit: nothing fixes
generated content after the fact, and every transformation is posted on
the concordance by its producer while every consumer reads the posted row
for what it is about to use.

## Measured

| tree | gate files | passed | failed |
|---|---|---|---|
| checkpoint b4bdb2c1, the original 12-file set | 12 | 464 | 35 |
| after wave 1 (f5c9c7c0), 23 files | 23 | 584 | 21 |
| after wave 2 (9bd1951a), 25 files | 25 | 599 | 17 |
| after wave 3, 28 files | 28 | 609 | 12 |

The twelve remaining failures are all pre-existing: the four
audit-contract fixtures that stop on deep gates (report 21), five
introduced by the checkpoint commit 2b94890e before any of this work
(the classmethod receiver test, three `test_ir_sequence_tables`
constructor tests, the unpickleable-artifact test), and three older
(`test_energy_limit_exact_helper_has_closed_optional_and_structural_abi`,
`test_pruning_updates_caller_and_callee_without_exempting_live_formals`,
`test_full_native_extension_requires_declared_non_python_abi[None-reject]`).

The four real-program probes used throughout (a `while True` function
returning `None` or a pair; its tail-return variant; a caller projecting
the optional result; whole-value use of the result as a tuple dict key)
now lower end to end through `lower_ast_source_to_ssa`. On the checkpoint
they stopped in result-type settlement.

## What the checkpoint had right and what it lacked

The previous agent's core is the mechanism the overview calls for: imported
cells that retain a dependency's owning book, live derivations that record
what each view read so stale views are detected, structural views over
layouts and descriptors, boundary transport receipts, and return layouts
and payload-presence pairs on the book instead of in metadata. Its
readers were made strict before its producers and the test fixtures were
made to post, which is where the 25 new failures came from. None was a
wrong value; every one was a reader refusing a row nobody had posted, or a
fixture handing a pass metadata the pipeline posts on the book.

## Causes closed, by producer or consumer

Fixtures. Ten tests built metadata by hand; they now post their SSA values
and layout rows through the existing helpers inside a fresh compile book
per test. Six precompile tests, twelve result-type tests, three
native-receipt tests and four loop-scope tests likewise. Two assertions
were stale against the pass's intended behaviour and were rewritten: an
equal-priority disagreement refuses, and a caller derives from an accepted
component's imported proof.

Consumers that matched by proxy. The fold step resolved a value's class to
a contract record by name tail; it now uses the shell's exact identity
rule. `semantic_output_ids` metadata was read as the authority for a
function's outputs while the `FUNCTION_OUTPUT` page holds the physical
Ret slots; a `SEMANTIC_OUTPUT` page now carries the reducer's declaration,
the three passes that re-mint an output id revise it, and the metadata is
a view. The finish step compared the book's answer with its live dicts and
took the dict; the rows are the authority and a disagreement refuses. The
result-storage lease looked an accepted callee's cells up on the ambient
book; it reads the owner book and cites through the import receipt.
Result-type settlement keyed a region formal's source on a cut-time
capture id; it resolves through the posted planning-value alias. A planned
region read its parent's optional receipts through a shared scope string;
a callee is record-covered only by a layout row naming an output it
returns.

Producers that did not post. The `Const(None)` rewrite, 34 record-ABI
minter sites, the sequence descriptor and mutation dtype fills, constant
materialization, late source literals, structural none outputs, the final
`Const(None)` sweep, field-slot receiver and load values, four call-type
fixed-point writes, LLVM intrinsic declarations, precompile-built
numerical and region functions, and formals appended by structural
recovery all post or revise their `ssa_value` rows at creation. Nested
functions get `PARAMETER_ANNOTATION` rows; a class fact is posted under
its canonical identity. The loop-scope declaration lost its raw-write
fallback and the two call sites that swallowed its refusal. The
`proven_shape` writer no longer re-posts a repeat or flips a recorded
conflict.

The physical-storage record. `SSAValueFact` carries `physical_dtype`,
`storage`, `rank`, `arena` and `device`, defaulted and carried forward on
every rebuild; the structural view, transport and audit read them, and the
audit reports `ssa-physical-unposted` as the migration worklist. No
producer posts the fields yet; that is stage three, and it is what turns
the ledger's "declared physical" ranking from an accounting label into a
posted fact, which is the fix for the stale-formal shape the tire recorded.

## Open items

- Physical storage stage three: producers post the fields (optional
  presence, conversion casts, pointer inference, keyed parts, settlement,
  physical Phi, returned-record dtypes, tensor lowering), then readers
  switch key by key behind the audit worklist; the rebuild site at
  `fortran_c_shell.py` ~27803 must carry the fields forward first.
- A caller/callee reshape disagreement seen by the `proven_shape` trace:
  module B drops a reshape view and calls a region with `(8,)` while its
  formal is `(8,1,1)`; the recorded conflict is true and its writer is
  untraced.
- `glsl_deployment_strategy.py` ~1696 and ~16933 read `SEMANTIC_OUTPUT`
  by scope string without checking the output is one the declaration
  returns; the region aliasing may reach them.
- A loop with no `source_loop_node_id` is skipped at declaration (the
  existing rebind precedent); hand-built fixtures should declare their
  loops so it can refuse.
- The four audit-contract fixtures, the five checkpoint-introduced
  failures and the three older ones, as listed above.
- The user's bootstrap has not been run on any of these commits.

## Prompt history

> using focused sonnet agents that are on rails for just the mechanical execution of fact finding or correction, orchestrate agents in making real progress to these ends
