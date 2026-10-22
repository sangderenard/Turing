# Takeover prompt: finishing the Turing compiler as one product

This is the prompt for whoever takes over compiler work. It may be one agent
or several agents working in parallel. It is written to be copied, unchanged,
to the root of the coordination repository (`nogodsnomasters`). It lives here,
at the root of Turing, because that is where the code it describes lives.

You will be handed two results with it: a full **bootstrap** run and a
**managed balloon tire** run. Neither has been run against the current `main`.
Treat them as the first evidence you read, not as a to-do list.

Read all of this before you run anything. Then read `AGENTS.md` (its last two
sections, "Agents: you will not out-design what is already here" and "The
Kalto Engineer", are binding) and the top of `TEST_BASELINE_AND_HAZARDS.md`.

---

## 1. What you are taking over

The Turing compiler reads authored Python and turns it into native code that
must reproduce the Python exactly: same value, same type, to the ULP. The
pipeline, in order:

1. The AST is read into a **process graph**.
2. The planner carves the graph into regions and call closures, and
   specializes callees per callsite.
3. A control program is built and lowered to **repository SSA**
   (`precompile_to_ssa.py`).
4. Call frames are linked across functions (`fortran_c_shell.py`, about
   60,000 lines).
5. Backends emit **C**, **LLVM** and **Fortran**.

Every value has exactly one identity. Every pass that decides something about
a value posts that decision on the **identity book** (the concordance). The
concordance audit (`src/compiler/identity_concordance.py`, driven by
`tools/audit_identity_concordance.py`) checks the records against each other.

There is one public entry: `src.compiler.fortran_c_shell.lower_ast_source_to_ssa`.

There are two whole-program targets:
- **The bootstrap.** The compiler compiles its own sources, unit by unit,
  through the provenance core (`tools/bootstrap_compiler_whole_program.py`;
  status via `tools/compiler_bootstrap_status.py`).
- **The managed balloon tire.** A real physics program with a managed-dt
  controller, compiled to native and validated against its Python oracle
  (`tools/build_balloon_tire_native.py --managed-dt`, then
  `tools/validate_balloon_tire_managed_native.py`).

"Finished" means both of these pass. It also means everything an ordinary
author writes either compiles correctly or is refused by name, and never
miscompiles. That second half is what this prompt is mostly about.

---

## 2. The state you inherit (2026-10-22)

`main` is at **af315b9**. Every part of the gate shows no new failures.

**Corpus.** `tests/native_semantics_corpus`: 159 programs (c01 to c210, with
gaps). At 5e99a7c (code identical to `main`'s for the corpus) every program
returned Python's value and type, with 0 concordance findings. The corpus is
the language contract: each program is a past bug, minimized, with its inputs
in a trailing `# INPUTS` comment.

**Gate.** `tools/gate/`. Failure counts are aa 89, ab 7, ac 26, rec 20. Those
are known baseline failures, listed by name in `tools/gate/baseline_*.txt`.
A gate passes when it adds **no new names**. It is not judged by the counts.

**Fuzzer.** `tools/differential_fuzz.py`. Seeds 2 to 9 produced no silent
wrong answers, but each seed found and fixed several before reaching that
point. The newest tier (`FUZZ_EXT=3`: helpers that mutate and read a caller's
list) found five families in its first two seeds.

**Bootstrap.** The last measured full run was 23/330 units admitted
(`docs/compiler_convergence/31`). After that, a source-only integration round
went unexecuted (`32`). Then this round's repairs went in (`33`). The run you
are handed is the first measurement of all of that together.

**Tire.** The last runs failed at the transport check:
`Metrics.error_present`, a declared span (15,) leased as (). See `32`.

Receipts, newest first:
- `TEST_BASELINE_AND_HAZARDS.md`, top entries;
- `docs/compiler_convergence/33_identity_repairs_and_bootstrap_chain_2026_10_22.md`.
  Read all of its sections, then its "Open frontier", which lists what is
  known broken or refused and why.

---

## 3. How to hold this work

You are not here to close a ticket. You are here to make a crude, deep machine
into a finished product. Two failures dominate this repository's history.

**Narrow chasing.** An agent fixes the first red line of a bootstrap receipt,
reruns for hours, fixes the next one, and the same identity was lost by the
same pass forty times. Before you fix anything, group the failures by the
record that is wrong. Fix the pass that wrote it, once, for the whole family.
Then write a corpus program that would have caught it.

**Turning loud into silent.** A refusal ("no proven capacity", "operand has
no value", "incompatible physical call inputs") is the compiler telling the
truth. A change that makes a refusal disappear without supplying the missing
record produces silent wrong answers, which are worse than refusals. It
happened four times in the last round; it was caught each time only because
the corpus and fuzzer compare values against Python. The rule:

> Whenever a refusal turns into an answer, run that program against Python
> before you believe it. Then run the whole corpus.

Work broadly and carefully. Ask what the machine is missing, not what the
receipt asks for. For example: "the planner splits a list argument into
per-element values" became the general rule "an aggregate is bound by member
only when neither side changes it and the callee reads it only through
constant-index projections". That one rule closed seven programs and three
latent bugs nobody had seen yet.

Keep observed, inferred and unknown apart in everything you write.

---

## 4. Your first pass over the two results

Do not fix anything during this pass. Produce a **triage table** and commit it
as `docs/compiler_convergence/34_<date>_triage.md`.

For the bootstrap:
1. List every unit that is not admitted, with its stage and its first
   refusal, verbatim.
2. Group the units by the **record that disagrees**, not by the message text.
   Different messages can come from one lost identity, and one message can
   have three causes.
3. For each group, name:
   - the two records that disagree;
   - the value they disagree about;
   - your best hypothesis for the pass that wrote the wrong one, labelled
     *inferred*.
4. Check every admitted unit against the receipts for **false acceptance**.
   Unit 027 was once "admitted" while exporting only one member of an
   optional pair. An admitted unit with a suspicious export is a finding.

For the tire:
1. Find the first refusal or first numeric divergence from the oracle (the
   tire has a dual oracle; see `30`).
2. Say which region, call edge or value it is at.
3. Say whether it belongs to a bootstrap group. It often does: the tire and
   the bootstrap share most passes.

Then rank the groups by **blast radius** (units blocked), times **depth** (how
early the pass is), times **whether it can produce silent wrong answers**.
Silent-capable groups come first, whatever their size.

---

## 5. Instruments, and how to use them

### 5.1 Make a small repro first, always

A real fault reproduces in seconds on a slice of real source under the real
contract. Lift the failing function into a scratch file, keep its types,
wrap a global-scope program in one function, and lower it through the public
entry:

```python
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
module, outputs, exports = lower_ast_source_to_ssa(
    source, "root", name="x",
    extraction_contract="extraction_contracts/program_extraction.yaml",
    progress=lambda _: None)
```

To get C: `src.compiler.ssa_c_backend.emit_ssa_to_c(module, exports[0])`.

### 5.2 Value and type against Python: `tools/native_corpus_typed_check.py DIR`

This runs every `c*.py` in DIR through lowering, C emission and a native call
for each `# INPUTS` row. It compares value and Python type, and prints the
concordance finding count. It also applies the native product's own
root-input refusal (`_unexplained_root_argument_ids`) and reports it as
PRODUCT-REFUSES. `emit_ssa_to_c` alone does not check that refusal, so a
program can look green there and still be refused by the product.

Statuses: PASS, FAIL (wrong value or type), LOWER-FAIL, EMIT-INCOMPLETE,
PRODUCT-REFUSES.

Run the full corpus on any change to a shared pass. It takes about 40
minutes and runs in the background:
`python -u tools/native_corpus_typed_check.py tests/native_semantics_corpus > corpus.log`.
`TURING_ROOT=<worktree>` points it at another tree.

### 5.3 The differential fuzzer: `tools/differential_fuzz.py OUT SEED COUNT`

It generates programs from composable snippets, compiles each, and compares it
with Python. Each native run happens in a forked child, so a segfault is a
result (SEGV), not a crash of the run. Tiers are `FUZZ_EXT` unset, 1, 2 and 3;
each widens the grammar. Re-run single files with
`tools/differential_fuzz_rerun.py`.

Statuses: PASS, WRONG, LOUD, TYPE, FINDING.

- **WRONG** and **TYPE** are emergencies. Stop and minimize.
- **LOUD** results are clustered: every LOUD cluster is a frontier item, and
  it either becomes a lowering or a named refusal in doc 33/34's frontier.
- **Grow the grammar every time you close a family.** Add the snippets that
  would have found it. The fuzzer is how this repository finds what nobody
  thought to write. A snippet is a pair: setup lines with `{v}` suffixes, and
  an integer expression. See `EXTENDED3` for the newest shape.

### 5.4 The concordance audit: `tools/audit_identity_concordance.py`

Run it before and after a change. When you invent a new record, teach the
audit to read it. A finding is a disagreement between two records about one
value: treat every one as a bug, never as noise.

### 5.5 Watching the writer (never patch-and-see)

The compiler is deterministic, so every wrong value has a writer you can
watch. These four probes found every fault in the last round. Write them as
scratch scripts outside the repo. Each takes a few lines.

- **Trap an attribute write.** Patch `SSAValue.__setattr__`, filter by `id`
  and by field (`dtype`, `shape`), and print `traceback.extract_stack(limit=6)`
  when it changes. This finds who turned an int64 into `'int'`, or a `ptr`
  cell into shape (1,).
- **Trap a book post.** Patch the identity book class's `post` (the class in
  `identity_concordance` that has `post`, `page` and `latest_ref`), filter by
  page name and row, and print the fact and the stack. This finds the
  revision that undid another pass's decision.
- **Trap a graph mutation.** Patch `remove_node` on
  `src/transmogrifier/graph/memory_graph/helpers/networkx_emulation.py:NetworkxEmulation`
  and on `networkx.DiGraph`. This finds the pruner that removed a node a
  record still names.
- **Read locals at one line.** `sys.settrace` with a per-frame line tracer.
  Filter on `co_filename` and line number, and `eval` an expression against
  `f_locals`. This reads a variable deep inside a 2,000-line nested function
  without editing it.

You also get far with one-function hooks: wrap a module-level helper (for
example `_settle_linked_dynamic_sequences`, `_static_sequence_capacity_survey`,
`_linked_caller_member`, `function_value_structure`, `_publish_type`), print
its inputs and outputs, and lower the repro. When the helper is imported
inside the shell function, patch it on both modules.

### 5.6 The gate: `tools/gate/run_gate.sh <worktree> <logdir>`

There are four pytest parts. Each is compared with its baseline, and only new
failures are printed. Rules:
- Run it on a **clean worktree** at the commit under test
  (`git worktree add <dir> <sha>`). Never use a stash, and never check out
  one path.
- When something fails, run it on the parent commit's worktree **before**
  you say whose fault it is.
- Part aa takes about 20 minutes. Read **every part's result** before you
  move `main`. The last round broke this rule once and briefly shipped a
  regression.

---

## 6. Rules this round established (do not relearn them)

Each rule is stated in a receipt comment above the code that enforces it.
`grep` for the program in quotes to find it.

1. **One definition of "changed".** Use `_mutated_sequence_node_ids`
   (`glsl_deployment_strategy.py`). It covers retained mutation records,
   method mutations, indexed-store chains and call escapes. Its
   `include_call_escapes=False` mode covers a graph's own effects. Every
   reader that would treat a collection's initializer as its contents
   consults it: the structural fold, region captures, callsite
   specialization (`_callsite_argument_is_initializer`) and the member split
   (`_aggregate_binds_by_member`).
2. **One definition of "escapes into a call".** Use
   `_call_escaped_argument_edges` / `_call_escaped_argument_ids`
   (`resident_sequence_materialization.py`).
3. **A list passed to a call is one storage.** It is split into member
   values only when neither side changes it and the callee reads only
   constant-index projections. The plan builder reads the specialization's
   decision; it does not recompute it on the specialized copy.
4. **Capacity is settled across call edges**
   (`_settle_linked_dynamic_sequences`). A component that grows anywhere is
   DYNAMIC everywhere. An escaped local list whose escapes are all exact,
   non-growing bindings keeps its own row bound
   (`linked_bounded_sequence_ids`). A keyed store into a dict that isn't the
   function's own display seeds growth.
5. **An in-place effect orders later reads.** This includes a call that
   receives a mutable collection (`_sequence_effect_read_edges`). A node that
   a retained mutation names as its effect is never dead metadata.
6. **Births.** A list or set display is born lexically: one clear, then one
   append or add per member. A heap column is a `ptr` cell, and its element
   type lives in `heap_element_dtype`. A row load never demands the element
   dtype of a heap column. Sequence helpers declare their heap column formals
   `storage="reference"`. The owner keeps the span view that planned regions
   reduce through.
7. **Python `int` is `int64`.** `'int'` is a provisional class spelling.
   `_planned_scalar_dtypes` is the scalar typing rule, including a
   parameter's settled linked-ABI class. Structural recovery uses it as well.
8. **A call whose result is a caller formal reads it; it does not redefine
   it.** This is the planner's identity forwarding of a returned formal.
9. **Placement has one rule.** `_control_block_position` is the only one, and
   regions are placed by member completion.
10. **The DYNAMIC protocol inserts, grows on "full", and retries.** A fixed
    fill inside it reports "full" instead of raising (`raise_when_full`). A
    FIXED fill that runs out of capacity raises code 75.

---

## 7. Lanes for distributed work

The work divides into lanes that touch mostly different code. Give each agent
one lane, one branch and one worktree. Lanes meet only at `main` (section 8).

**Lane A: bootstrap families.** Work from the triage table, in rank order.
For each family:
1. minimize to a scratch repro;
2. trap the writer;
3. fix it at the pass that lost the record;
4. add a corpus program;
5. run the corpus;
6. hand the user the one-line bootstrap command when a rerun is warranted.

You do not launch the multi-hour bootstrap yourself unless the user says to
(`AGENTS.md`, "How you work").

**Lane B: managed tire.** Same method. The tire's divergences are often
ABI-shape facts (spans versus scalars, record fields) across call edges.
Compare `docs/compiler_convergence/30` to `32`.

**Lane C: language frontier.** Turn LOUD clusters into lowerings, using
existing mechanisms only. Current list (doc 33, "Open frontier"):
- multi-element replication `[a, b] * k`;
- membership in a folded literal inside a loop;
- standalone slices;
- dict `del`/`pop`;
- `d.get` in a loop;
- nested list stores;
- early return from a loop over a list parameter;
- strings as values;
- variadic returns;
- structural-aggregate reductions (`any((1 > n, 2 > n))`).

Each item says where the record is missing. Every new lowering gets a
corpus program and a fuzz snippet.

**Lane D: defaults and fallbacks.** This is the user's standing order: find
code that does not use the concordance to make its decision. Concrete
leads:
- `ssa_c_backend.py`, the public wrapper: private root storage with no
  storage requirement is sized `else 1`. Nothing reaches it today, but a
  default is not a proof. Make it a named refusal and see what the corpus
  says.
- Planner region inputs default to float64 when nothing proves a type.
  About 3,000 hits across the corpus were measured once; later stages
  retype the ones that reach results. Count them again, and drive the
  number toward zero by supplying the proofs.
- `"int"` versus `"int64"` spellings flowing into physical checks.
- Every `except Exception: pass` in a decision path.
- Every place that matches by name, substring or position (`grep` for
  `binding_name ==`, `.endswith(`, `[0]` on argument lists).

Inventory first, mechanically, with counts. Then fix by category.

**Lane E: backend parity.** The corpus runs through C. LLVM and Fortran are
the more developed lanes (`AGENTS.md`). Make the typed check run the corpus
through each backend, and treat a disagreement between backends as a
finding. Never treat the interpreter lane as the reference.

**Lane F: product.** This covers:
- wiring the typed corpus check and a short fuzz seed into
  `tests/test_native_semantics_corpus.py`, or a sibling test;
- keeping receipts current;
- shrinking the gate's baseline lists, by fixing tests or by finding the
  record that says a test is obsolete (never by skipping or deleting it);
- measuring compile time per corpus program.

Run several lanes at once only when they edit different functions.
`fortran_c_shell.py` and `glsl_deployment_strategy.py` are large; two agents
editing the same nested function will conflict. Before starting, claim
functions in the lane's doc section.

---

## 8. Getting changes to `main` (non-negotiable)

1. Work in a worktree on your lane branch. Commit small, each commit with its
   receipt comments.
2. Before proposing a commit for `main`, all of these must hold:
   - the focused tests for the files you touched pass;
   - the full typed corpus check has 0 non-PASS, compared with the previous
     corpus run's PASS set;
   - a fuzz seed in your lane's tier has 0 WRONG and 0 TYPE;
   - the gate has no new names in **all four** parts.
3. Fast-forward `main` only to a commit that passed all of that. Never move
   `main` while a gate part is still running. Never move it to a commit whose
   successor fixes it (a fix gets its own gate).
4. Every round ends with:
   - a top entry in `TEST_BASELINE_AND_HAZARDS.md`, with what was measured,
     at which commits, and what is not green;
   - a section in the round's `docs/compiler_convergence/` document, with
     each fault's program, wrong record and rule, and the updated frontier.
5. Commit trailers follow the repository's convention. Never put model
   identifiers in commits.

---

## 9. What a finding, a status and a question look like

- **A finding** names:
  - the program (minimal);
  - the two records that disagree, and the value;
  - the pass that wrote the wrong record, with the stack you watched;
  - the fix at that pass;
  - the corpus program that now holds it.
- **A status** says what is green, what is red, what is running and what is
  next, in that order and in short sentences.
- **A question to the user** is only for a decision that is genuinely theirs:
  launching a multi-hour run, choosing between two semantics Python itself
  leaves open, or accepting a refusal as permanent. Never ask permission to
  do the work.

---

## 10. Never

- Never add a tolerance, a sort, a fallback, a default extent or a "repair"
  pass where a record should be consulted.
- Never mask by function name, spelling or position.
- Never turn a loud refusal into an answer you have not compared with Python.
- Never skip, quarantine or delete a failing test to get green.
- Never baseline with a stash or a path checkout.
- Never claim a run is clean because its log is short. Read the end of it.
- Never move `main` past an unread gate part.
- Never write a stand-in next to an existing system to avoid learning it.

---

## 11. Where things are

| Need | Path |
| --- | --- |
| Binding rules | `AGENTS.md` (last two sections) |
| Measured history and hazards | `TEST_BASELINE_AND_HAZARDS.md` |
| Per-round design receipts | `docs/compiler_convergence/NN_*.md` (33 is the latest) |
| Language contract | `tests/native_semantics_corpus/` (+ `tests/test_native_semantics_corpus.py`) |
| Typed corpus check | `tools/native_corpus_typed_check.py DIR` |
| Differential fuzzer | `tools/differential_fuzz.py OUT SEED COUNT`, `tools/differential_fuzz_rerun.py FILES` |
| Gate | `tools/gate/run_gate.sh TREE LOGS`, lists and baselines in `tools/gate/` |
| Concordance audit | `tools/audit_identity_concordance.py` |
| Bootstrap | `tools/bootstrap_compiler_whole_program.py`, `tools/compiler_bootstrap_status.py` |
| Tire | `tools/build_balloon_tire_native.py --managed-dt --backend c`, `tools/validate_balloon_tire_managed_native.py`, `tools/compare_balloon_tire_oracles.py` |
| Public entry | `src/compiler/fortran_c_shell.py: lower_ast_source_to_ssa` |
| Planner and specialization | `src/compiler/glsl_deployment_strategy.py` |
| SSA builder | `src/compiler/precompile_to_ssa.py` |
| Sequence lowering | `src/compiler/ir_sequence_tables.py`, `src/compiler/resident_sequence_materialization.py` |
| Concordance | `src/compiler/identity_concordance.py`, `src/compiler/concordance_declarations.py`, `src/compiler/structural_concordance.py` |
| C backend | `src/compiler/ssa_c_backend.py` |

Start with section 4. The results you were handed are the first evidence.
They are not the plan.
