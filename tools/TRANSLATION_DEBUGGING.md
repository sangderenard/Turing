# Why isn't this translating? — a decision tree

A translation runs Python → ProcessGraph → dual IR → repository SSA →
backend (Fortran / LLVM / C shell) → runtime buffers. A defect anywhere
surfaces at the **end**, as a compiler error or a wrong number. The
expensive mistake — the one that has actually cost days here — is guessing
which stage owns it and reading code at the wrong altitude.

So don't guess. Answer the questions below in order. Each one is cheap,
each one either clears a stage or names it, and each one says what it does
**not** prove.

---

## The rule that matters most

> **A measurement you cannot take is not a zero.**

Every tool here refuses to invent a value it cannot observe. This is not
fastidiousness: `_read()` used to return `0.0` for any value id absent from
the artifact's public buffer ABI, internal allocas are absent from it, and
so probing one returned a zero indistinguishable from a measured zero.
That fabricated evidence produced a full day of confident, wrong diagnosis.

If a tool says **NOT OBSERVABLE**, you have learned nothing about that
value. Do not reason from it.

---

## Q0 — Does it compile at all?

| answer | go to |
|---|---|
| No — the compiler raises | **Q1** |
| Yes, but a number is wrong | **Q0b**, then **Q4** |
| Yes, but it crashes at runtime | **Q3** then **Q4** |

---

## Q0b — Read the program back as Python

Do this **before** Q4 when the compiler is happy and a number is wrong. It is
cheap, it needs no ids, and it answers a question the other tools cannot:
*what did the compiler actually build?*

```python
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_python_materializer import materialize_ir_module, to_source

module, outputs, exports = lower_ast_source_to_ssa(SOURCE, "entry", name="probe")
emitted, skipped = materialize_ir_module(module)
print(skipped)          # functions that could NOT be rendered, and why
print(to_source(emitted))
```

The emitted module is real Python: readable, runnable, and diffable against
what you wrote. Run it on the same inputs and compare against the authored
function directly — that comparison is the only check that catches the defect
class below, because none of them raises.

Two limits, stated so a clear result is not over-read:

* Only single-block functions and the five-block counted loop are rendered.
  Anything else appears in `skipped` with a reason. A `skipped` entry is not
  evidence about correctness, only about coverage.
* A lowering that RAISES returns no module, so the program you most want to
  read is the one you cannot. That is the open prerequisite: the lowering
  needs to hand back its IR alongside its shortfalls.

`python tools/translation_scorecard.py` runs this over a graded corpus and
prints how far each complexity level gets. Use it to place your program: if a
level below yours already stops, that is your bug and it is already known.

**To see WHERE two runs part, not just whether:** the instrumented shell
records every SSA value assignment as it executes, and the emitted local
names ARE the value ids (`t7` is value 7) — no correlation table.

```python
from src.compiler.python_shell import compile_python_shell

shell = compile_python_shell(module, "probe__train")
run = shell.run(w=2.0, n=3)
run.trace                          # every tN, in order, once per iteration
run.last_value("probe__train", 7)  # refuses if never assigned — not a zero
run.first_divergence(other_run)    # the first entry where two runs part
```

This is the honest version of the source-level probe the "Do not do this"
section forbids: it records AFTER lowering, so the observed program is
bit-for-bit the compiled one and observation cannot perturb it. It also hosts
retained Python callables via `bindings=`, and `shell.write(path)` emits the
whole thing as a standalone runnable `.py`.

**Static checks over the same product, before running anything:**

```python
from src.compiler.ssa_self_check import run_all, suspicious_loop_invariant_formals

run_all(module)                            # violations: formal parity,
                                           # id()-scale ids, disagreeing
                                           # region output contracts
suspicious_loop_invariant_formals(module)  # CANDIDATES for the frozen-carried
                                           # check — not convictions
```

The candidate list is separate on purpose: a formal consumed inside a loop
body is either a legitimate loop-invariant input or a carried value the
planner dropped, and the SSA does not record which the author meant. That gap
is the thing to fix — stamping the authored carried set into
`Function.metadata` would make the frozen-carried defect statically decidable.

---

## Q1 — Does the SSA claim to be complete?

```bash
python tools/diagnose_translation.py --stages 1
```

* **Shortfalls reported** → the lowering already knows it failed. Read the
  shortfall text; it names the domain (`ssa-sequence`, `ssa-table`, …) and
  the location. Nothing downstream is meaningful until it clears. **Stop
  here.**
* **Complete** → the SSA believes it is whole. Continue.

Does not prove the SSA is *correct* — only that no pass reported giving up.

---

## Q2 — Is the SSA well-formed?

```bash
python tools/diagnose_translation.py --stages 2
```

Three independent checks:

**2a — Duplicate producers.** Is one id produced by two *different*
`SSAValue` objects?

> **The distinction that costs hours:** an id that is both a formal
> argument and an instruction result is **fine** when both are the *same
> object* — that is deliberate in-place cell reuse, and it is fast. All 14
> such ids in `symbolic_fluid_advance` are exactly that. A **true**
> collision is two *distinct* objects sharing one id; then whichever
> renders last in the backend's id-keyed pointer cache silently wins.

**2b — Phi arity.** Does every phi's incoming-block count match its
operand count?

**2c — Dangling operands.** Is every operand a formal, or produced, or a
known constant?

* **Any FAIL** → the defect is at or above `precompile_to_ssa.py`. The
  backend is faithfully rendering a broken graph; do not read backend code
  yet. **Go to Q6.**
* **All clear** → continue.

---

## Q3 — Is in-place fusion safe here?

```bash
python tools/diagnose_translation.py --stages 3
```

In-place is **wanted**: an out-param that aliases a feed avoids a copy.
This check exists to keep it, not to discourage it. It reports fusions as
expected and flags only the two ways fusion is genuinely unsafe:

* **Two different outputs sharing one pointer** — the second write
  destroys the first.
* **A fused formal read *after* being written** inside the callee — the
  result then depends on instruction order rather than on the program.

* **FAIL** → the *fusion decision* is wrong, not the arithmetic. **Go to
  Q6.**
* **Clear** → the fusions are safe. Continue.

---

## Q4 — Can you even see the value you are blaming?

```bash
python tools/diagnose_translation.py --ids 141,116,47 --stages 4
```

* **NOT OBSERVABLE** → it is an internal alloca with no public buffer.
  You cannot read it. Any earlier conclusion drawn from reading it is void.
  Pick a value that *is* observable and downstream of your suspect, and
  reason from that instead.
* **Observable** → readings from it are real evidence. Continue.

In code: `adv.observable(id)` before `adv._read(id)`, or
`adv._read(id, required=True)` to make the mistake impossible.

---

## Q5 — What actually reaches the wrong value?

```bash
python tools/diagnose_translation.py --ids 141,117 --stages 5
```

This runs the **dye trace** (`influence_field.field_from_ssa`, which *is*
wired to lowered SSA — ~8k transports over the advance function). It
answers the question static IR reading answers only expensively: *what
reaches this, from where, and how much survived.*

Read it **comparatively, never absolutely**:

| reading | means |
|---|---|
| dominant `dynamic` | genuinely runtime-fed |
| dominant `baked` | compile-time-constant influence |
| `recurrent` | arrived through a loop-carried edge — state, not derivation |

> **Calibration warning, learned the hard way:** "dominantly baked" is
> *not* a defect signal in this program. `dt`/`dx`/`gravity`/limits feed
> nearly everything, and the known-**correct** `max_wave_speed` reads
> dominantly baked. An earlier version of this tool flagged it and would
> have sent you down a blind alley.

What *does* carry signal:

* A value whose profile differs from a **comparable value known to work**.
  (Pass both ids; the tool diffs them for you.)
* Two supposedly-**independent** values with *identical* weights — they
  share an influence path. The tool flags these automatically. Note the
  built-in caveat it prints: two outputs of the *same* region call are one
  node in the def-use view, so identical weights there are expected.

---

## Q5b — Watch the value directly (when it isn't observable)

Q4 says an internal value cannot be read. **Watch makes it readable**, without
touching the program:

```python
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
art = emit_ssa_function_to_llvm(module, name, watch=(116, 47, 136))
# art.watched            -> ids now in the public buffer ABI
# art.watch_shortfalls   -> ((id, reason), ...) for any that could not be
```

A watch appends an output slot and copies a value the program **already
computed**. It adds no arithmetic, reorders nothing, renames nothing, and
with `watch=()` the emitted IR is byte-identical. Verified on this program:
`mass_err`, `max_vel`, `dt_limit`, the final state and the `ok` flag are all
bit-identical with watches on and off.

This is the sanctioned alternative to the thing you must not do (below).
A source-level probe shifts value ids; a watch cannot.

**Loop-carried accumulators are watchable**, including the phis. A phi's
storage is a register that does not dominate the return, so a watch on one
gets a *shadow slot* in the entry frame, updated wherever the phi executes.
It therefore reads **the converged value at loop exit** — which is what you
want from an accumulator.

A watch that cannot be honoured is reported, never dropped. A watch that
silently vanished would read as "nothing to see here", which is precisely
the false reassurance this whole mechanism exists to end.

---

## Q5c — Which LAYER owns it? (the routing question)

This is the question that decides where to read code, and getting it wrong
costs days. Run the SSA itself and see which side it agrees with:

```bash
python tools/differential_matrix.py          # oracle / ssa / llvm / fortran
```

```python
from src.compiler.ssa_reference_evaluator import (
    SSAReferenceEvaluator, bind_program_abi_arguments,
)
arguments, unbound = bind_program_abi_arguments(
    function, record=state,
    named={"dt": 0.2, "height_count": 4, "width_count": 4},
    functions=module.functions,      # needed to find parameters BY NAME
)
SSAReferenceEvaluator(module).run(name, arguments)
```

| result | means |
|---|---|
| ssa == oracle, backend differs | **emission** changed the meaning |
| ssa == backend, oracle differs | **lowering/planning** changed it |
| all three differ | the evaluator is not calibrated for this shape — fix that first |
| ssa == oracle == backend | the defect is in something none of them observes |

Worked example, and the reason this step exists: the fluid traversal read
`ssa vs oracle = 0.0` and `llvm vs oracle = 6.66e-03`, which put the
defect in LLVM **emission** — after days of searching the planner and the
AST/SymPy inlet on the assumption it was upstream.

**The evaluator is only usable while its calibration passes.**
`tests/test_ssa_reference_evaluator.py` calibrates it on a pure function,
on a synthetic traversal with hand-computable truth, and on the real
traversal against the authored oracle. If those fail, no routing claim
from it means anything, and the tests say so in their failure messages.

---

## Q5d — Emission owns it. Which INSTRUCTION?

Once Q5c says emission, narrow inside it. Do these in order; each one
throws away a layer, and the later steps are worthless without the
earlier ones.

```bash
python tools/bisect_emission.py            # first divergence, program order
```

**1. Get a series, not an endpoint.** Both sides record per-iteration
rings, and they deliberately record the same thing so they are
comparable:

```python
artifact = emit_ssa_function_to_llvm(module, name, watch=ids, history=24)
ring, count = history_ids(value_id)     # read execution.buffers[ring]

evaluator = SSAReferenceEvaluator(module, history=ids)
evaluator.run(name, arguments)          # evaluator.history[value_id]
```

A final value proves an accumulator ended wrong; only the series says
which iteration spoiled it. `bisect_emission.py` orders by static block
order, so for a loop-carried phi it names the accumulator rather than the
cause — the series is what resolves that. **Check the cadences match**
(some value both sides agree on, ringed at the same length) before
trusting any disagreement between them.

**2. Find the boundary where inputs agree and outputs do not.** Ring a
call's arguments and its results. Arguments agreeing per-iteration while
results disagree localises the defect inside that callee and rules out
everything upstream of it.

**3. Read the pattern of WHICH outputs are wrong.** In the case above the
wrong ones were exactly the time-integrated quantities and every pure
diagnostic was right. That shape pointed at arithmetic and away from the
neighbour gathering, before any code was read.

**4. Reproduce it standalone.** Feed the callee its exact inputs on fresh
buffers, outside the loop:

```python
execution = prepare_artifact_execution(native, {formal_id: value})
```

If it still diverges, the loop, the gathering and in-place aliasing are
all excluded at once. If it does NOT, the defect is in the surroundings
and step 5 would have chased a ghost.

**5. Fingerprint a pure reproducer.** Perturb each input, compare
sensitivities on both sides. Inputs the artifact ignores, or a
coefficient equal to the SUM of several the SSA has separately, say the
operands are collapsing.

**6. Count opcodes before believing an instruction was dropped.** Compare
the SSA's operation census against the emitted body's, and count EVERY
spelling — `mul` as well as `fmul`, `icmp`/`select` as well as `maxnum`.
A shortfall that reconciles exactly once the other spellings are counted
means nothing was lost; it means something is being emitted in the wrong
domain, which is a different and much quieter bug.

Two limits worth knowing before they mislead you:

* `watch=` reaches the ROOT function's frame only. A region-local id is a
  different numbering space and is reported, not silently dropped.
* the single-block emission path publishes its own value set, and a
  Ret-less region as root publishes nothing — so a region cannot be
  bisected by making it the root. Reach it through the smallest caller
  that has a `Ret`.

---

## Q5e — The whole-program native executable runs but a buffer is wrong

Everything above assumes one function, reached through `watch=`/LLVM. A
standalone Fortran/C-shell executable is a different shape of problem:
dozens of `bind(C)` subroutines calling each other by POSITION, each
level with its own local id numbering, and the question is no longer
"what is this SSA value" but **"does this specific C-visible buffer ever
get written, anywhere in the call graph reachable from here"**.

Reading that by eye does not scale — a real case here required following
one buffer through five nested `call` statements across hundreds of
positional arguments each. `tools/trace_fortran_alias.py` does it
mechanically:

```bash
python tools/trace_fortran_alias.py path/to/generated.f90 \
    --entry the_bind_c_entry_name --source state.height
```

Given an entry point and a dotted source name from the `.api.yaml`
contract beside the `.f90`, it finds the entry's own formal for that
source, then for every `call` statement in the entry's body checks
whether the SAME actual token is among the arguments; if so it resolves
the CALLEE's formal at that position (by index, not by name — two
functions' local numbering never has to agree), reports that formal's
declared Fortran `intent` and whether the callee's own body writes an
array element into it, and recurses. The output is the whole chain in
one table: subroutine, formal, intent, write-or-not, at every hop.

**What it is good for:** proving a buffer is never written anywhere
reachable (a `no local write` at every hop, all the way down, is a real
finding — not a "not observable"), and finding exactly which hop a write
happens at when one exists.

**What it does NOT do, and do not read it as if it did:** it follows one
TOKEN. A value can arrive at a callee under MULTIPLE different formal
names (five spatial neighbour views of `state.height` is a real example
in this tree — `t54, t56, t58, t60, t45, t52, t122, t23, t27` are all
"state.height" at one function, and the tracer only follows whichever one
the caller happened to forward under the traced token). A clean trace of
one token is not a clean trace of the value's every occurrence.

> **Always verify against the api contract, never against hand-parsed
> Fortran text.** A prior pass in this exact investigation counted 367
> formals on a subroutine by splitting a declaration line on commas with
> `line[line.index('('):line.rindex(')')]` — `rindex(')')` found the
> LAST `)` on the line, which sits inside the trailing
> `bind(C, name="...")` clause, so the split captured two fragments of
> that string as if they were extra formal names. The real count,
> read from the `.api.yaml` contract (which is generated data, not
> re-derived by eye), was 366 — matching the call site exactly. The
> wrong count was recorded as a finding and had to be corrected later.
> `trace_fortran_alias.py` reads the contract for source-name resolution
> and a real regex-based parser for the rest, specifically so this
> mistake cannot recur.

---

## Q6 — Which layer disagrees with which?

```bash
python tools/correlate_compile.py 141,116,47
```

One fresh compile, then per id, side by side:

* the repository SSA in the **advance function** (producer and consumers),
* the repository SSA in **every planned region** that touches it,
* every **LLVM IR** line mentioning its register,
* the **runtime value**, or an explicit `NOT OBSERVABLE`.

This is the tool for *"the SSA says X but the IR says Y"*. Use it when a
stage above named a suspect and you need to see where the layers stop
agreeing.

---

## Id selection — every tool, same syntax

```bash
--ids 141                     # one
--ids 141,47                  # a list
--ids 141 47                  # spaces work too
--ids 100-120                 # inclusive range
--ids 100-120,141,187..193    # mixed; a..b is the same as a-b
```

`tools/correlate_compile.py` takes the same forms positionally:

```bash
python tools/correlate_compile.py 100-120,141
```

Ranges are capped at 4096 ids per token, and an unreadable token is a hard
error rather than a silent skip — a typo that quietly narrows your search
is worse than one that stops you.

Stages are selectable the same way:

```bash
python tools/diagnose_translation.py --stages 2,3
python tools/diagnose_translation.py --stages 1-3 --ids 141
```

---

## Finding the id to ask about

```python
fn.metadata["value_names"]        # (name, id) for authored locals
fn.metadata["carried_port_values"]  # loop-carried port -> phi
adv.artifact.buffer_order         # exactly what is observable
```

A value's *authored name* is the honest starting point: `value_names` maps
`mass_error`, `max_wave_speed`, `next_mass` to their ids in the advance
function's own numbering. Region-local ids are a **different numbering
space** — an id that means one thing in `advance` means something
unrelated inside `planned_region_0`. `correlate_compile.py` shows both, and
labels which is which, precisely because conflating them is easy.

---

## Silent miscompilations already paid for

These compile with **no shortfall**, run, and return the wrong number. No
stage of the compiler reports anything. Each one has a signature you can check
directly rather than rediscovering it; all three were found by Q0b.

**Only the first loop-carried value is carried — FIXED, signature kept.** A
loop carrying two names froze the second at its ENTRY value for every
iteration. Root cause: the reducer's lexical environment materializes
parameters lazily at first read, so a parameter first touched INSIDE the loop
was absent from the pre-loop snapshot and could never be discovered as
carried state. `topological_reducer` now materializes every parameter the
loop reads before the snapshot.

*Signature (if it recurs):* compute what the program WOULD return with the
suspect value held constant, and compare. A bit-exact match convicts it. The
chain to check, in order: `loop_carried_bindings` on the graph's loop node
(the reducer's discovery), `loop.carried_aliases` reaching
`_ControlSSABuilder.lower_loop` (the plan), the carried phis in the emitted
header (the SSA). The stage whose count first disagrees with the source owns
it — in the original defect the reducer's own discovery was already short.

**A value can be both a formal and a region output.** A one-parameter source
compiles to a two-parameter function whose extra formal is also an output of a
planned region, consumed by a call scheduled BEFORE the one producing it. The
use-before-def is invisible precisely because the value is also a formal, so
it is never unbound.

*Signature:* the emitted function has more parameters than the source, and the
extra one is unnamed. Compare `len(function.args)` against
`metadata["parameter_names"]`.

**A loop body that only forwards a call result is elided entirely.** The body
block emits nothing but `Br`, so the carried update has no producer and the
lowering raises `loop_carried`.

*Signature:* `next_w = update(w)` where `w` is the carried value. Binding
through one more operation (`stepped = update(w); next_w = stepped * 1.0`)
restores it. A call on some OTHER value is fine; it is the round trip that
fails. See `tests/test_loop_carried_producers.py`.

---

## Backend-specific traps already paid for

**gfortran fails silently.** Invoked by absolute path with its own `bin`
off `PATH`, it exits non-zero with *no diagnostic at all* — it spawns
`f951`, which cannot load `libgmp`/`libmpfr`. `ssa_fortran_backend.py`
puts the toolchain on `PATH` for the child; do the same by hand:

```bash
PATH="/c/msys64/mingw64/bin:$PATH" gfortran -fsyntax-only isolated.f90
```

**A huge generated file truncates the real error.** Isolating just the
calling subroutine and its callee into a small file makes gfortran report
the actual mismatch instead of a summary.

**Git Bash `/tmp` is invisible to Windows Python.** Write scratch files to
a real path under `build/` instead; a redirect that "worked" and a Python
open that "cannot find the file" is this, not a race.

---

## Do not do this

**Do not add a source-level probe to observe a value.** Writing
`state.some_field = suspect + 0.0` to make a value observable *shifts value
ids and can rebind the very thing being measured*. It has already produced
a wrong conclusion in this codebase — twice. It also perturbs unrelated
consumers: a probe added to `max_wave_speed` corrupted a different
consumer of that same reduction.

Prefer reading an **existing** public id. If you must add a probe, verify
with `correlate_compile.py` that the ids still mean what you think
**before** trusting its readout.

---

## When every stage clears and it is still wrong

That is a real state, and it is where this program currently sits for
`mass_err`. It means the defect is not in shortfalls, SSA well-formedness,
in-place fusion, observability, or influence topology. Record what you
ruled out **with the check that ruled it out** — a hypothesis closed
without evidence gets re-opened by the next person, and re-walking a
cleared branch is the single largest time sink in this work.

`tools/HANDOFF_fluid_c_shell.md` is that record for the current defect,
including a correction of a claim that was wrong and load-bearing. Keep it
honest in the same way: an overturned finding is more valuable written
down than quietly deleted.

---

# The tool log: what was reached for, in what order, and what it bought

One real hunt, end to end. The order matters more than any individual
tool — each answer narrowed what the next tool had to look at.

| # | Reached for | Question it answered | What it bought |
|---|---|---|---|
| 1 | `git log` / reading the handoff | what was already known | avoided re-deriving four fixed defects |
| 2 | `emit_module` + `gfortran` | does it build at all | 9 errors → a concrete, finite list |
| 3 | Isolating caller+callee into one small `.f90` | what is the REAL diagnostic | full-file compiles truncate the error; isolation revealed it |
| 4 | Reading the pickled SSA directly | does the IR say what I think | disproved three plausible theories in minutes |
| 5 | `correlate_compile.py` (built here) | which layer disagrees | SSA vs LLVM vs runtime, side by side, per id |
| 6 | `diagnose_translation.py` (built here) | which STAGE owns it | routing, instead of reading the wrong altitude |
| 7 | `influence_field` dye (already existed) | what reaches this value | closed the "are they aliased" question |
| 8 | `watch=` (built here) | what IS this value at runtime | turned NOT OBSERVABLE into a measurement |
| 9 | `history=N` (built here) | which ITERATION went wrong | the series, not just the final value — this cracked it |
| 10 | `differential_translation.py` (built here) | is the translation faithful | an independent oracle; found a second, larger defect |
| 11 | `ssa_reference_evaluator.py` (built here) | lowering's fault or emission's | **the routing answer** — SSA matched the oracle exactly, so emission owns it |
| 12 | `differential_matrix.py` (built here) | all representations at once | one table; backend-vs-backend needs no oracle |
| 13 | `bisect_emission.py` (built here) | which VALUE emission got wrong | first divergence in program order; for a carried phi it names the accumulator, so pair it with 14 |
| 14 | `history=` on BOTH sides (extended here) | which ITERATION, on either side | rings every watched value, not only phis, and the evaluator rings the same way — arguments agreeing per-iteration while outputs did not is what localised the defect to one callee |
| 15 | sensitivity fingerprint (ad hoc) | which INPUTS the artifact actually uses | perturb each input of a pure reproducer; the artifact used 3 of 9, and one coefficient equalled the sum of the 6 it lost |
| 16 | opcode census (ad hoc) | is an instruction missing, or in the wrong domain | `fmul` 118 + `mul` 22 = SSA's 140 `Mul`. Nothing missing — 22 were integer. **This named the bug.** |
| 17 | `--trace` on `compile_fortran_module_c_shell` (built here) | per-launch timing/status inside a standalone executable | compile-time-only ring buffer; an untraced artifact has none of this code in it at all |
| 18 | `trace_fortran_alias.py` (built here) | does a buffer get written ANYWHERE reachable in a whole-program native build | follows one token through nested `call` statements by position; proved `state.height` is never written across 5 hops, and where `state.next_height` is |

**What each stage cost when skipped.** Steps 8 and 9 existed only after
days of reasoning about structure. Every one of those days would have been
saved by asking "what is this value, actually?" on day one — which was
impossible, because the answer was `NOT OBSERVABLE` and the tool returned
`0.0` instead of saying so.

**The shape of the whole hunt**, which is the transferable part:

1. A symptom (`mass_err = 0.0`) that turned out to be **correct behaviour**.
2. A "ground truth" (`1.68e-04`) that turned out to be **computed from the
   bug itself**.
3. Structural checks that all passed, repeatedly, because the defect was
   not structural.
4. A measurement that could not be taken, silently reported as a zero.
5. The real defect — a whole-array copy moving one element — found only
   once values became observable, and found *next to* where everyone was
   looking, not inside it.

If a hunt here is going badly, the odds strongly favour one of those five
over an exotic compiler bug.

---

# Field notes: counterintuitive things that actually happened

These are not hypotheticals. Each cost real time in this tree, each looked
like something else first, and each is the reason a check above exists.
They are written down because the *shape* of these mistakes transfers even
when the specific bug does not.

### 1. A tool that answers `0.0` for "I cannot see that"

`_read()` returned `0.0` for any value absent from the public buffer ABI.
Internal allocas are absent from it. So probing an internal accumulator
returned a zero **indistinguishable from a measured zero** — and a
computation that was genuinely producing zero was the thing under
investigation.

I reported "the subtraction itself computes zero" as an established fact
and reasoned onward from it for hours. It was the probe, not the program.

> **The lesson is not "check for None".** It is that a diagnostic which
> degrades gracefully is a diagnostic that lies. Every tool here now
> refuses: `observable()` asks, `_read(required=True)` raises, and the
> correlate tool prints `NOT OBSERVABLE` where a number would go.

### 2. The heuristic that flagged the healthy value

Stage 5 originally flagged "dominantly BAKED" influence as suspicious. Its
first run flagged `max_wave_speed` — which is **correct**. Baked dominance
is simply the norm here, because `dt`/`dx`/`gravity` feed nearly
everything.

Had it shipped, it would have sent the next reader to a known-good value
with an authoritative-looking `[FAIL]` beside it.

> Test a new check against something you *know* is healthy before you trust
> it on something you suspect. A check is a claim; claims get verified.

### 3. Two things sharing an id, where one is fine and one is fatal

14 values in the advance function are **both a formal argument and an
instruction result**. That looks exactly like the id-collision class of bug
that had already been found twice in this session.

It is not a bug. `arg97 is load97` — they are the *same object*. The
argument cell is deliberately reused as that value's storage, which is
in-place and fast. A real collision is two *distinct* objects on one id.

> The check that matters is `is`, not `==`, and not "same number". Stage 2
> encodes this distinction precisely because the number alone sent me
> chasing a non-bug through the emitter.

### 4. A probe that corrupted the thing it measured

To observe an accumulator, I added `state.some_field = accumulator + 0.0`
to the authored source. This **shifted value ids** and rebound a different
consumer of that same reduction — a previously-correct value started
reading `0.0`.

The reading it produced was then used as evidence. Twice.

> This is why `watch=` exists at the emitter instead. Never make a program
> observable by editing it when you can make it observable by *emitting one
> extra copy of what it already computed*.

### 5. The compiler enforcing a rule that saved the tool

The first version of `watch` appended the value to the function's outputs
and copied it at the return. For a loop-carried phi, LLVM rejected the
module: *"Instruction does not dominate all uses"*.

That refusal was **correct and valuable**. A phi's storage is a register
selected per-edge; it genuinely is not readable at the return. The naive
fix (skip phis) would have made accumulators — the most interesting values
to watch — permanently unwatchable. The real fix was a shadow slot in the
entry frame.

The second attempt then hit *"PHI nodes not grouped at top of basic
block!"*, because the capture was emitted between two phis. Also correct,
also load-bearing: the copies now defer to the end of the phi group.

> A verifier rejecting your instrumentation is information about the IR's
> real structure, not an obstacle. Both errors taught the mechanism
> something it needed to know.

### 6. A cached artifact that made a fix look like a no-op

Loading a stale `control_repository_ssa.pkl` produced a failure in an
*unrelated* subsystem, several layers from the change under test. The
pickle held a program lowered by a compiler that no longer existed.

> Now enforced in code: `_cache_is_stale()` compares the pickle against
> every compiler source and re-lowers rather than trusting it. A spurious
> 30-second recompile costs a coffee; a silently stale artifact costs a day
> *and* produces confident wrong conclusions along the way.

### 7. Chasing a symptom that was correct behaviour

`mass_err` read exactly `0.0` against an expected `1.68e-04`. I treated that
as the defect and pursued it for a very long time — aliasing, CFG, phi
propagation, influence topology, backend registers. Every one came back
clean, which should itself have been the signal.

The per-iteration watch showed the two accumulators taking *genuinely
different* summands (`0.997309…` vs `1.0…`) and converging on the same
total. That is a **mass-conserving scheme working correctly**. `mass_err =
0.0` was the right answer all along.

The real defect was one layer away and had never been examined: the
whole-array copy `state.height = state.next_height + 0.0` updated **1 cell
of 16**. My "expected" `1.68e-04` had been computed from `state.height`
*after* the call — i.e. from the corrupted copy. **The number I was
treating as ground truth was itself produced by the bug.**

> When every hypothesis about a symptom clears, question the symptom.
> And check where your reference value came from: a "truth" derived from
> the same run as the observation is not independent of it.

### 8. The check that would have taken seconds

The four broken values carry `program_abi_rank: 2` and
`program_abi_storage: "span"` in accounting, while their `.shape` is `()`.
The emitter's `_value_element_count` reads `.shape`, gets 1, and emits a
scalar load/store where a whole-array copy belongs.

The instructive part is what makes this *hard to see*: rank-with-empty-shape
is the **normal** representation for a dynamically-sized array here, and the
Fortran backend handles it correctly through the extents vector. So the
disagreement is not itself a defect — a naive check on it reports 35 values
and buries the 4 that matter. The hazard is specifically a rank>0 value
standing as a **region-call output**, where the return-copy sizes itself
statically.

Stage 2 now checks exactly that, and names them:

```
[FAIL] 4 region-call OUTPUT(s) declare rank>0 but carry an empty .shape
       id 122 (field height, rank 2) out of planned_region_4
       id 126 (field momentum_x, rank 2) out of planned_region_4
       ...
```

> A check that fires on everything teaches nothing. Narrow it to the
> configuration that actually causes harm, and say what the harm is.

### 9. A zero that made the compiler look guilty

The SSA evaluator reproduced neither the artifact nor the oracle, and its
traversal looked like it was collapsing neighbour reads: every cell's
`height_next` came out exactly equal to its own centre height, and the
mass accumulator summed the OLD grid. That is precisely what a compiler
dropping its flux terms would look like, and it was investigated as such.

The cause was in the instrument. `dt` carries no per-argument accounting,
so the binder matched it by dtype-and-rank, found **two** float64 scalar
candidates, correctly refused the ambiguity — and then let it fall through
to a scratch fill that bound it to **0.0**.

Nothing crashed. Every other value stayed plausible: neighbour reads were
being resolved perfectly the whole time, `wave_speed` was exactly right,
the loop visited all sixteen cells and read each one correctly. With
`dt = 0`, `h + dt*(fluxes)` is just `h`, for any fluxes at all.

Fixing the binding changed `ssa vs oracle` from "wrong everywhere" to
**exactly 0.0**, and with it the whole conclusion: the SSA was correct all
along and the defect is in emission.

> **A default is a claim.** `unbound` and `zero` are different statements,
> and code that turns the first into the second manufactures evidence. The
> binder now reports what it could not bind, and identifies a parameter the
> way the runtime does — through the callee formal it feeds, by name.
>
> Note the shape: this is the SAME defect as field note 1, in a new place,
> committed by someone who had just written field note 1. Silent defaults
> are that easy to reintroduce.

### 10. The bug that deleted terms instead of failing

The one the whole differential apparatus was built to find, kept here in
full because the ROUTE is more reusable than the patch.

A binary scalar operation took its arithmetic domain from `args[0]` alone.
The lowering spells negation as `Mul(int -1, x)`, so every negation of a
float ran as an **integer** multiply and the float operand was reached by
`fptosi`. Fractional values truncated to zero. Nothing raised, nothing was
dropped from the instruction stream, and the emitted code verified clean —
whole terms simply evaluated to nothing. The compiled fluid step ignored
six of its twenty-eight inputs, never saw a tracer go negative, and so
accepted a frame the authored mathematics rejects.

The sequence that found it, each step discarding a layer:

1. **Route first.** The SSA matched the authored oracle exactly, the
   artifact did not (Q5c). Emission owns it. Two earlier hunts through the
   planner and the AST inlet were spent on layers that were innocent.
2. **Series, not endpoints.** The first divergence in program order was a
   loop-carried phi — which says an accumulator ended wrong, never which
   iteration spoiled it. Ringing both sides per iteration showed all 28
   call arguments agreeing while four outputs did not.
3. **Read the pattern before reading code.** The four wrong outputs were
   exactly the time-integrated quantities; every pure diagnostic of the
   current state was right. That is a shape, and it pointed at arithmetic
   rather than at the neighbour gathering.
4. **Shrink to a reproducer.** It reproduced standalone on fresh buffers —
   which killed the loop, the gathering, and in-place aliasing at once.
5. **Fingerprint the reproducer.** A pure function of 28 scalars: perturb
   each input, compare sensitivities. The artifact depended on three inputs
   where the SSA depended on nine, and its one shared coefficient equalled
   the SUM of the six it had lost — which is what a collapsed domain looks
   like from the outside.
6. **Count opcodes before believing anything is missing.** SSA `Mul` 140
   against emitted `fmul` 118 looks like 22 dropped instructions. It was
   not: `fmul` 118 + `mul` 22 = 140. Everything reconciled. Twenty-two
   multiplications were simply integer ones, and that named the defect.

Two confident hypotheses were killed on the way, both by measurement:
that the call site permuted its arguments (checked against the emitted IR
by script, not by eye — reading a 28-pointer call by eye had "confirmed"
it, wrongly), and that six formals were unwired (`buffer_order` was
complete and duplicate-free).

> **Promote, never demote.** Widening an integer operand is exact;
> narrowing a float discards its value. A conversion that loses data in
> service of a type rule will not announce itself — it returns a plausible
> number. This is field note 1's disease in the type system: a silent
> default, wearing a cast.

### 11. What all of these have in common

Most were **the instrument lying, not the program**. The program under
investigation was deterministic and consistent the entire time; what varied
was the quality of the observation.

When a result is baffling, suspect the measurement before the mechanism —
and prefer an instrument that refuses to answer over one that answers
plausibly.

The compressed version, for anyone starting a hunt here:

1. **Can I observe this at all?** (`observable()` — if not, `watch=`.)
2. **Where did my reference value come from?** If the same run produced
   both, it is not independent evidence.
3. **Is the symptom actually wrong?** Correct-but-surprising behaviour has
   eaten more time in this tree than any real bug.
4. **Does a known-good value pass my new check?** If not, the check is
   broken, not the value.
5. **Same number or same object?** `is`, not `==`.
6. **Is my artifact current?** A stale lowering makes a fix look like a
   no-op.
7. **Which LAYER owns it?** (Q5c.) Answer this before reading code, not
   after. Two of the longest hunts here were spent in the wrong layer.
8. **Did anything default silently?** An unbound input that became a zero
   will implicate the compiler for a bug in your harness — and it will
   look exactly like a real one.

---

# Companion documents still to be written

This document covers **one span** of the pipeline: repository SSA through
the backends, for a program that already lowered. Three neighbouring spans
deserve the same treatment and do not have it yet. These are notes on what
each will need — not the documents themselves.

The house style to keep, because it is what made this one useful:

* **Ordered questions, not a feature list.** Each answer clears a stage or
  names one. A reader in trouble is not browsing.
* **Every check states what it does NOT prove.** A check that overclaims
  sends people down blind alleys, which is worse than no check.
* **Refuse to fabricate.** No silent defaults for unobservable things.
* **Record calibration failures in the document.** The
  "dominantly-baked is not a defect signal" warning is worth more than the
  check it qualifies.
* **Every command copy-pasteable and actually run before shipping.**

---

## A. Code/math ingestion → dual IR

Covers: source text and SymPy → `ProcessGraph` → topology reduction →
hierarchy plan → dual IR. Everything *before* SSA exists, where the
authored program is still recognisable and identity is still being decided.

Notes toward it:

* **Node identity is the whole subject.** `ensure_node` keys nodes on
  `id(node)` — the *Python object address*. Object lifetime therefore
  affects graph identity, and a freed temporary's address can be reused.
  Any questionnaire here starts with "is this one node or two, and why".
* **`deduplicate_node` merges on `label` + `type` alone** — no scope, no
  version, no source span. It is guarded to skip `ast.AST` nodes (the
  guard's comment records that merging every `GetAttr` once collapsed
  `x.detach().tanh()` into a self-cycle), but it still applies to non-AST
  structural objects, notably SymPy expressions. "Did two things merge that
  shouldn't have" needs a first-class check.
* **Auto-aliasing at capture.** `result is parent_value` records a
  `value_aliases` entry — correct and necessary, and `max()` returning one
  of its own arguments makes it fire in ordinary code. Needs a way to dump
  the alias chain for a value and ask "is this an alias, and of what".
* **Extraction contract decisions are policy, not accident.** `parent_include`
  labels but does not gate pursuit; a "why was this function ingested / not
  ingested" question belongs here, answered from the receipts
  (`extraction_action`, `extraction_rule`, `extraction_identity`) that are
  already attached to every node.
* **Tooling that likely needs building:** a node-provenance dumper (source
  span → node → what it became), and an alias-chain resolver. The dye trace
  already has `field_from_process_graph` and `field_from_dual_ir` adapters,
  so influence is available at this altitude *now* — the SSA one turned out
  to be wired and unused, and these two probably are too.

## B. The binary head / machine-state operation

Covers: decoding native binaries into repository SSA — the read head,
`InstructionSpec` vocabulary, scalar decoder, machine-state dialect, and
the legalisation that turns retained machine state into ordinary SSA.

Notes toward it:

* **The authority hierarchy is the first thing to state.** The scalar
  `InstructionSpec` vocabulary is the source of truth; the tensor read-head
  is a *derived* write/reversibility accelerator, not a rival decoder. New
  ISAs get added at the spec + scalar-decoder layer. A reader who does not
  know this will "fix" the wrong decoder.
* **The completeness facts are already recorded and should drive the
  tree:** `host_repository_ssa_complete`, `host_machine_state_complete`,
  `host_ssa_blockers`, `host_ssa_hard_blockers`,
  `host_ssa_legalization_shortfalls`, `host_ssa_unresolved_dependencies`.
  The questionnaire is largely "which of these is false, and what does that
  one mean".
* **Blocker vs hard blocker is the load-bearing distinction** — one is
  "not yet", the other is "not this way". `select_host_implementation`
  already encodes the decision; the document should explain the choice it
  makes rather than restate the flags.
* **Retained machine state is legitimate**, the same way in-place fusion is
  legitimate here: the question is never "is there machine state" but "is
  it legalised or is it leaking into a public ABI".
* **Reversibility is the sharp diagnostic** — a decode that cannot be
  re-encoded to the same bytes has lost something, and that check is
  mechanical.

## C. Backends (Fortran / C shell / LLVM / SPIR-V / WASM)

Covers: repository SSA → emitted target text → linked artifact. Partly
covered above for Fortran and LLVM; the *shared* structure deserves its own
document as more backends land.

Notes toward it:

* **The recurring bug class, across every backend, is one identity meaning
  two things** — the same disease as the ingestion layer, one layer down.
  Every backend keeps an id-keyed pointer/register cache, so two distinct
  values on one id means whichever renders last silently wins. A shared
  "id-keyed cache soundness" check would serve all of them.
* **Positional binding is the second recurring class.** `zip(caller_args,
  callee_formals)` truncates silently on a length mismatch;
  `zip(dataclass_fields, return_record)` mispaired every `Metrics` field
  here. Any backend that pairs by position needs a check that the two
  sequences are the same length *and* the same thing.
* **Fixed-point emission loops need their convergence stated.**
  `emit_module` runs five (`ranks`, `scalars`, `array-contracts`, `dtypes`,
  `mutation`) plus two bounded re-emission loops, and "did this converge,
  and what does it mean if it did not" should be answerable without reading
  the loop.
* **Per-backend traps belong in one table**, in the spirit of the gfortran
  silent-failure note above: each is cheap to write down and expensive to
  rediscover.
* **`emit_module` cannot fix what SSA got wrong.** Its loops only unify
  *metadata* on an already-fixed, positionally-paired `Call.args` tuple —
  it can never change how many arguments a call passes. The document should
  say this plainly so structural defects are chased upstream instead of in
  the emitter.
