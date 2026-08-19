# Task A — audit the remaining constant-presence trap sites

Read `README.md` in this directory first. Scope: three read sites in two
files. Difficulty: low. Everything you need is in this file.

## The defect class (already fixed twice — you are finishing the sweep)

Every ProcessGraph node is created with a `constant=None` key
(`src/transmogrifier/graph/graph_express2.py`, the `add_node` call around
line 2683). Therefore the test `if "constant" in data:` is TRUE for
essentially every node, and code that then returns `data["constant"]`
reports the literal `None` for arbitrary computations. That lie is how the
compiler used to delete every live `if` arm.

The fix pattern is established and committed — read it before writing
anything:

```
git show d50ba56 -- src/compiler/glsl_deployment_strategy.py src/compiler/shell_reference_tables.py
```

The rule it implements: **a `None` payload counts as a literal only when
the node itself is declared constant** (`type` in
`{"Constant", "Const", "const"}` or `op == "const"`); a non-None payload
still counts unconditionally (span-dissolver constants rely on that).

## The three sites to audit (measure FIRST — some may be harmless)

1. `src/compiler/loop_composer.py` — `_constant` (line ~1573). Returns
   `(True, data["constant"])` on key presence. Used for loop trip counts;
   `_trip_count` requires `isinstance(..., int)` so a `None` lie is
   probably filtered downstream — VERIFY that claim by reading every
   caller of `_constant` in that file before deciding no change is
   needed. If any caller treats `(True, None)` as a real literal, apply
   the d50ba56 pattern.
2. `src/compiler/fortran_c_shell.py` — `literal_value` (line ~3868).
   Checks `attributes["value"]` first, then `"constant" in data`. Read
   its callers: does a non-constant node reaching it return
   `_copy_literal_payload(None)` where the caller distinguishes
   None-meaning-absent from None-meaning-literal? If yes, apply the
   pattern.
3. `src/compiler/fortran_c_shell.py` — the default-literal read around
   line ~6067 (`elif "constant" in node:` inside the `default_literals`
   loop) and the similar check at ~6493. Read the gate ABOVE each: if the
   loop already `continue`s for non-constant node types, the site is
   safe — record that; if not, apply the pattern.

`src/compiler/symbolic_equation_compiler.py` (~line 132) was already
checked: its reads sit inside a type-gated block. Do not change it.

## How to measure

For each site, write a throwaway probe in a scratch directory (NOT the
repo): lower a small program with `lower_ast_source_to_ssa` and print what
the site's function returns for a non-constant node. A ready-made program
that exercises loops and conditionals:

```python
import sys, warnings
sys.path.insert(0, r"C:\dev\Powershell\turing")
SOURCE = (
    "def helper(a):\n    return a * 1.0\n\n"
    "def train(w, n):\n"
    "    total = helper(w)\n"
    "    for k in range(n):\n"
    "        if total > 2.0:\n"
    "            total = total * 0.5\n"
    "        else:\n"
    "            total = total + 1.0\n"
    "    return total\n"
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    module, outputs, exports = lower_ast_source_to_ssa(SOURCE, "train", name="probe")
```

Monkeypatching a function to log its inputs/outputs is fine INSIDE a
throwaway probe script; never leave a monkeypatch in repo code.

## Definition of done

* Each of the three sites has a verdict recorded in this file under
  `## FINDINGS`: "safe because <gate>, unchanged" or "fixed, pattern
  applied" — with the probe evidence.
* Any code change passes the full gate (README rule 3, expect 68
  passed + scorecard 17/19).
* One commit per fixed site (or one commit for the audit if all safe),
  narrative message, plus the FINDINGS update in this file.
