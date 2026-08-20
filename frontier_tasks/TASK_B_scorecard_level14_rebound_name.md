# Task B — scorecard level 14: a rebound name feeds a call its own future

Read `README.md` in this directory first. Difficulty: medium. Your primary
deliverable is a NAMED SEAM (which pass produces the wrong binding, with
evidence); a fix is the stretch goal, only if it is small and the gate stays
green.

## The symptom

`timeout 120 python tools/translation_scorecard.py` shows level 14 stopped
at MATERIALIZE:

```
14  a parameter name rebound and shadowed  STOP MATERIALIZE  not materialized: ['sc14__train']
```

The materializer's refusal is honest — the real defect is upstream in the
lowered SSA. The authored program (see `JOURNEYS` level 14 in
`tools/translation_scorecard.py`):

```python
def train(x, a):
    x = x * 2.0
    a = helper(a) + x
    x = a - x
    return x + a
```

## The measured defect (verified 2026-08-19 — trust but re-derive cheaply)

Lower it and dump the SSA with this probe (takes ~30 s):

```python
import sys, warnings
sys.path.insert(0, r"C:\dev\Powershell\turing")
SOURCE = (
    "def helper(a):\n    return a * 1.0\n\n"
    "def train(x, a):\n"
    "    x = x * 2.0\n"
    "    a = helper(a) + x\n"
    "    x = a - x\n"
    "    return x + a\n"
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    module, outputs, exports = lower_ast_source_to_ssa(SOURCE, "train", name="sc14")
fn = module.functions["sc14__train"]
for block in fn.blocks.values():
    for ins in block.instrs:
        rid = f"t{int(ins.res.id)}" if ins.res is not None else "-"
        args = ",".join(f"t{int(a.id)}" for a in (ins.args or ()))
        print(f"{rid} = {ins.op}({args}) {dict(ins.attributes or {})}")
```

You will see, inside `sc14__train`:

```
t4 = Call(t5)   callee='sc14__helper'
t11 = Call(t4,t3)  callee='sc14__train__planned_region_1' output_ids=(7, 5, 6)
...
t5 = Load(...)  source_output_id=5     <- t5 is only PRODUCED here, AFTER t4 consumed it
```

The authored call is `helper(a)` where `a` is the FORMAL (value id 1). The
lowering instead feeds the call `t5` — a LATER version of the rebound name
`a` (region 1's own output). The call consumes its own downstream. This is
the stale-identity family: **call-argument resolution binds a rebound
name's final version instead of the version live at the callsite.**

## Where to look (ordered by prior evidence, not certainty)

* The identity history for a name lives in the per-function graph's
  `identity_table` (name -> tuple of value ids, lexical order). For this
  program it is `{'x': (...), 'a': (...)}` — print it from the probe via
  the deployment internals if needed, or find where callsite arguments
  are resolved against it.
* The same family was fixed twice for `while` loops in commits `86fb257`
  and `3ee307b` (read them: `git show 86fb257`, `git show 3ee307b`) —
  publication/recovery keyed by raw graph id could not see which VERSION
  a consumer meant. Your seam is the call-ARGUMENT side of that story.
* Call argument binding for source-linked calls happens during hierarchy
  planning / call linking (`plan_callsite_id`, `source_linked` attributes
  on the Call) — grep `src/compiler/fortran_c_shell.py` and
  `src/compiler/hierarchical_plan.py` for where a callsite's argument
  value ids are chosen from the identity table.

## Method

Instrument (in a throwaway probe, monkeypatch is fine there) the function
that selects the argument value id for callsite 4, and print: the authored
argument name, the identity history it consulted, and the version it chose.
The defect is the CHOICE RULE — name it precisely ("chooses history[-1]
instead of the version preceding the callsite's lexical position", or
whatever you actually observe).

## Definition of done

* Minimum: a `## FINDINGS` section in this file naming the exact function
  and line that makes the wrong choice, with the probe output proving it.
* Stretch: a fix. Levels 0–13 and 15–17 all PASS today and several
  exercise calls with renamed/shadowed arguments (10–13) — they are your
  safety net. Gate green (68 passed) + scorecard: level 14 must not
  regress anything else; if level 14 reaches PASSED, move its pin in
  `tests/test_translation_scorecard.py` (EXPECTED[14]) in the same
  commit, as the file's own doctrine instructs.
* Do NOT touch the materializer to work around this; its refusal is
  correct.

## FINDINGS 2026-08-19

**The exact seam, nailed with direct evidence.**
`src/compiler/fortran_c_shell.py`, inside `_class_surface_ssa_program`,
the `caller_aliases` construction (~line 6036, right before
`exact_bindings`):

```python
caller_aliases: dict[int, int] = {}
for history in (caller_graph.graph.get("identity_table") or {}).values():
    canonical = next((
        int(value_id) for value_id in reversed(history)
        if any(int(value.id) == int(value_id) for value in all_functions[caller_symbol].args)
        or any(instruction.res is not None and int(instruction.res.id) == int(value_id)
               for block in all_functions[caller_symbol].blocks.values()
               for instruction in block.instrs)
    ), None)
    if canonical is not None:
        for value_id in history:
            caller_aliases[int(value_id)] = int(canonical)
```

For level 14, `caller identity_table['a'] = (1, 5)` (1 = the formal `a0`,
5 = `a1`, produced later by `a = helper(a) + x`). This code walks
`reversed((1, 5)) = (5, 1)` and returns the FIRST value already
materialized in the caller's own lowered SSA function. Verified directly
against the real lowered `sc14__train`: **both 1 and 5 are already
produced** (1 is a formal arg, 5 is `t5 = Load(...)`), so the search
stops at 5 immediately — `canonical = 5` — and then EVERY id in the
history, including 1, gets overwritten: `caller_aliases[1] = 5`,
`caller_aliases[5] = 5`. The callsite for `helper(a)` had already
correctly resolved its raw argument to caller value 1 (confirmed by
instrumenting `_build_shell_hierarchy_plan`: `call_parents =
((1, 'arg:0'),)`, correctly built from position 0) — `caller_aliases`
is what then corrupts it to 5 on the way to `exact_bindings`
(~line 6054: `exact_bindings[callee] = caller_aliases.get(caller,
caller)`).

**Root cause, precisely**: this loop collapses an ENTIRE name's identity
history onto ONE canonical id, chosen by "latest identity that happens
to already be materialized" — with no awareness that a plain (non-loop)
rebinding produces several genuinely DIFFERENT, independently correct
SSA values at different lexical points. Collapsing them all to the
latest is exactly the same defect family already fixed twice for while
loops (`86fb257`, `3ee307b`) and level 17's while latch (`7d1fd43`) —
except those were about which value a RETURN publishes; this is the
CALL-ARGUMENT side of the same "identity history collapsed onto its
final value" story, exactly as `README.md`'s hint predicted.

**A fix was attempted and REVERTED — this is a real finding, not just a
diagnosis.** The obvious minimal fix (only redirect an identity that is
NOT itself already materialized; leave an already-produced identity
mapped to itself) makes level 14 PASS cleanly — verified, full SSA dump
showed `t4 = Call(t1)` (the formal, correctly, in valid dependency
order). But it **regressed levels 8 and 9** (loop, call anchored by one
op / loop, carried value through a call) from PASSED to EQUIVALENT-fail
(`worst disagreement 1.852e-01`). Diagnosed why: in
`train(w, n): total = update(w); for _ in range(n): stepped =
update(w); ...`, BOTH the pre-loop call and the in-loop call to
`update(w)` resolve their raw argument to the SAME caller value id (0,
`w`'s formal) at the `_build_shell_hierarchy_plan` stage — the
process-graph's flat (pre-loop-composition) representation does not
yet distinguish "w before the loop" from "w this iteration". The OLD
collapse-to-latest-producible behavior was doing real, if crude, work
here: it uniformly redirected EVERY raw resolution of `w` (0, or the
loop's own internal carried ids 7/8) to the same representative id (7),
which is what let the in-loop call end up consuming something
loop-correct rather than the stale pre-loop formal. My value-scoped fix
correctly left the pre-loop call's `0` as `0` (right) but ALSO left the
in-loop call's raw `0` as `0` (wrong — it needed to reach the loop's
carried representative instead), because `caller_aliases` has no
callsite-position context at all: it is a per-NAME map, not a
per-USAGE-SITE map, so it cannot know that a call site raw-resolved to
`0` is lexically INSIDE a loop that rebinds the name and therefore
needs the carried value, versus lexically BEFORE any rebinding and
correctly wanting the formal.

**What the real fix needs** (not attempted — this is the handoff for
whoever picks it up next): callsite-position awareness threaded into
this resolution, not a name-level table. Two plausible directions,
neither explored: (a) have `_build_shell_hierarchy_plan`'s own
positional/keyword argument matching (the `call_parents` /
`source_position` logic already read in this session, ~line 1451-1522
of `glsl_deployment_strategy.py`) resolve to the CORRECT identity in
the first place, so `caller_aliases` never needs to override an
already-correct resolution — investigate why level 14's Stage 1 gets
the right raw id (1) while level 8's in-loop call's Stage 1 apparently
does NOT reach the loop-carried id directly; or (b) make
`caller_aliases`'s remap conditional on whether the CALLSITE lies
inside the same loop body that owns the later rebinding (using
`enclosing_loop_ids`, already threaded through `SSACallRecord`
elsewhere in this same file) rather than blindly collapsing by name.

**Reverted cleanly** — `git diff` against the committed baseline is
empty for this file; scorecard confirmed back at 17/19, level 14 STOP
MATERIALIZE as before. No fix committed; this FINDINGS section is the
deliverable, per the task's own sanctioned minimum.
