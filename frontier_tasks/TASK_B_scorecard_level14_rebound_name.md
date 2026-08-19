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
