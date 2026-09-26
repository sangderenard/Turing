# Regulation through the concordance — report (2026-09-25 / 26)

The identity book (`src/compiler/identity_concordance.py`) is being made the
single authority for identity, durable from source reduction to emission.
The working rules, the page registry and the open list live in
[`CONCORDANCE_MASTER_LIST.md`](CONCORDANCE_MASTER_LIST.md); this report says
what was fixed, how, how it was verified, and what is known to remain.

## Principles applied

- Any id confusion is a failure to use the concordance.  Identities are
  joined through book rows, never by matching value ids or pairs of ids.
- One base fact per identity.  Other pages record structure (which
  operand, which entry), never a restatement of the fact.
- SSA structures keep their interfaces; their storage is the book.
- Scopes are minted by the compile's own book, in causal order.
- A missing row raises.  New records are taught to the audit.

## What was fixed, and how

### 1. SSA tables and call records store into the book

`SSARecordTable`, `SSASequenceTable` and the new `SSACallTable`
(`src/transmogrifier/ssa.py`) keep their interfaces (`records`,
`sequences`, `register`, `by_id`, list-like call records), but every
descriptor and call record is a book row with full history.  Member pages
(`record_member`, `sequence_member`) list, per value, every claim a live
descriptor makes on it; the call-linking resolver (`_linked_caller_member`)
reads those instead of scanning descriptors.  `TransformationLedger`
(frame ledger) stores its decisions and events on the book.  Scopes come
from `IdentityBook.mint_scope` (page `scope_registry`).

### 2. Linker working state moved onto the book

`record_field_demands`, `record_field_writes`, the parameter-by-value map,
the forwarding edges, `record_storage_aliases` and the scheduled call
argument map are book pages (via `IdentityPage.mapping`).  The scheduled
argument map was keyed by caller value id, which collapsed two arguments
holding one value; it is now keyed by the callee formal each argument binds.
A dead snapshot (`callee_callers`) was removed.

### 3. Loop-carried miscompiles: one value held by two bindings

The compiler identified a loop-carried variable by a value id — the value it
starts from or updates to.  When two bindings share that value
(`second = value`, `total = w`), reads of one became reads of the other.
Each of these compiled and computed the wrong answer (or never terminated):

| Program shape | Wrong result before |
|---|---|
| loop reads a parameter that also seeds a carried binding | `value=1, limit=3`: 4.0 instead of 3.0 |
| a zero-trip loop returning that carried binding | returned the other binding |
| call argument inside the loop | 6.0 instead of 4.0 |
| one value passed as two call operands | 8.0 instead of 4.0 |
| while predicate through a region | never terminated |
| direct while predicate | 1.0 instead of 3.0 |
| two bindings sharing an update, returned as a tuple | wrong exit values |

How it is fixed — bookkeeping with a deterministic causal order:

- The reducer records, for every read, which binding it read
  (`lexical_read_binding`, keyed by consumer node, operand role and
  ordinal; also AugAssign targets and bare `return name` roots).  This is
  the one base fact.
- `loop_composer` records which bindings each carried entry is
  (`loop_carried_binding`) and which binding each loop-result port continues
  (`loop_result_port_binding`); continuation rewiring moves only reads of the
  port's own binding.
- The planner and lowering record only structure: which operand a region
  feed or call argument is (`consumer_operand`, `call_argument_operand`,
  `region_feed_consumer`).
- The lowering records each loop's entry state on the book
  (`loop_entry_state`: carried bindings and pre-loop value per initial id)
  and keeps only its position.  One resolver (`_resolve_read`) answers every
  loop read: the loop that carries the read binding gives its header or
  latch value; a loop carrying none of the read bindings passes through to
  the pre-loop value; a mix or an unattributed read raises.  Region feeds,
  call arguments, break-bound ports and the while latch all go through it.
- Loop exits join port -> binding -> carried entry; carried Phis and
  updates are keyed by carried-entry position, not value id.

### 4. Type identity

- `ir_indexing`: a declared scalar `Const` dtype is kept; the literal's
  Python spelling only decides when nothing is declared.  (A declared
  float64 absent payload for `Metrics.advanced_dt` was retyped int64, so the
  callee returned int64 into a float64 caller slot.)
- `precompile_to_ssa`: an undeclared authored literal is typed by its own
  Python type (`return True` had been declared float64).

### 5. Python materializer and the translation scorecard

- A region called under the aggregate convention returns its aggregate as a
  tuple, checked against what the callers project; loop-exit phis are bound;
  output contracts read `callee_output_ids` under the aggregate convention.
- `tools/translation_scorecard.py` passes the program contract.  It had been
  refused by the compiler on every journey (0/19); it is back to 18/19.

### 6. Tests fixed on sight

- 43 compiler calls in 19 test files passed no extraction contract (the
  compiler refuses those); they now pass the repository contract.
- Stale expectations updated: hard-coded region ordinal, descriptor `rank`,
  specialized callee names, the materializer's pre-`Ret` region convention,
  a closed metadata gap, the removed workspace formal, the renamed frame
  ledger rule, and a zero-trip expectation that had encoded the old
  miscompile (the authored program is now the reference).
- New guard: `tests/test_loop_carried_producers.py::
  test_one_value_held_by_two_bindings_computes_the_authored_answer`
  (7 programs; all fail at the previous HEAD, all pass now).

## Verification

Baselines are a clean worktree of the previous HEAD plus the day's earlier
uncommitted work, with the same test files.

| Check | Before | After |
|---|---|---|
| Focused regression set (17 files) | 42 failed | 19 failed, 343 passed, none new |
| `tests/test_process_graph_function_linking.py` | 21 failed | 7 failed |
| `tools/translation_scorecard.py` | 0/19 | 18/19 |
| `tests/test_ssa_python_materializer.py` | 18 failed | 47/47 |
| Concordance audit (6 cases) | 0/2/0/1/5/0 findings | identical |
| Shared-value guard tests | 7 fail | 7 pass |

No long lowering (dt system, woodshop) was run.

## Known to remain

In order (details in `CONCORDANCE_MASTER_LIST.md`, A6):

1. Direct predicate expressions: `lower_control_expression` still resolves a
   bare value leaf by id.  Needs operand rows and a repro.
2. The lowering environment itself: `external_values` is still a private
   value-keyed dict, with loop reads carved out of it.  The line: binding
   state at each causal point on the book, `external_values` a view of it.
3. Private linker maps: `frame_ledgers`, `result_storage_bindings_by_call`,
   `constructor_instance_pools`, `call_anchor_value_ids`, and the
   `metadata["value_aliases"]` snapshot.
4. Name matchers: the annotation fallback in `_graph_sequence_record_abi`
   (and the demand check) matching `identity.rsplit(".", 1)[-1]`; the
   positional result window; numeral result limbs found by string key.
5. Emitted parameter order differs from the authored declaration.
6. Linker decisions that read the book but record nothing (resolver,
   discovery, reconcile, ownership, demand growth, pruning).
7. The 19 remaining failures in the focused set, including two assertions
   written for the pre-structured id space (`max id < 1e9`) whose intent
   needs the author, and `x or {}` over a mapping folding to an untyped
   return.
8. The Part B survey: 138 pipeline functions that read identity without the
   book.
