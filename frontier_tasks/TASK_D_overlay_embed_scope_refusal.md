# Task D — make the control overlay's cross-scope failure name itself

Read `README.md` in this directory first. Difficulty: low. Pure
data-structure work in one file, testable in isolation, no expensive runs.

## Background (two sentences)

When `re._compile` lowers, its big if/elif cascade's ~50 scheduled regions
sit in THREE different control scopes of the flat schedule (top level plus
two inner loop bodies), because the main token loop is blocked from
composing (that blocker is separate, `Raise`, and NOT this task). The
conditional overlay then inserts one full copy of the cascade into EACH
scope, and a downstream guard fails the whole lowering with the confusing
message `conditional control duplicated scheduled regions in '_compile':
{3: 3, 4: 3, ...}`.

## The mechanism (read it: `src/compiler/control_source.py`)

`overlay_scheduled_control` → nested `embed(block, nested_root,
nested_regions)` (around line 1289). In the `SequenceBlock` branch, the
`inserted` flag that enforces "only the first lexical marker receives the
nested control" is **local to each SequenceBlock**. When `embed` recurses
into a `LoopBlock`/`WhileBlock`/`ConditionalBlock` body, that body's own
SequenceBlock gets a fresh flag — so every scope containing at least one
consumed marker appends its own copy of `nested_root`.

## What to build

NOT a silent de-duplication: inserting the control at only one of the
scopes would be semantically wrong (loop-body work hoisted out of its
loop), and the repo forbids trading a loud failure for a quiet wrong
program. Instead:

1. Thread an insertion-count (or first-insertion) state through the
   `embed` recursion so it can DETECT that a nested control's markers span
   more than one insertion scope.
2. On detection, raise a `ValueError` that names the situation the way
   this codebase names things — e.g. *"nested control's regions span N
   sequence scopes of the parent (markers at top level and inside
   LoopBlock ...); the schedule and the conditional compartments disagree
   about where these regions live"* — listing the region indices per
   scope. This replaces the downstream duplicate-count crash with the
   actual cause, one layer earlier.
3. Healthy programs must be byte-identical in behavior: single-scope
   embedding is untouched.

## Tests (write them first)

`overlay_scheduled_control` and the block dataclasses
(`SequenceBlock`, `StatementBlock`, `LoopBlock`, `ConditionalBlock`,
`ControlProgram`) are constructible by hand — no compiler pipeline needed.
Look at how `tests/` builds control blocks (grep for
`overlay_scheduled_control` and `StatementBlock((f"__scheduled_region_`).
Add a new test file `tests/test_overlay_scope_refusal.py` with at least:

* a healthy case: flat schedule with a LoopBlock containing ALL of a
  conditional's markers → conditional nests inside the loop, no error,
  every marker appears exactly once (walk the composed root and count);
* the defect case: conditional markers split between top level and a
  LoopBlock body → your new ValueError, message naming both scopes;
* a two-conditional nesting case using `known_nesting` (mirror an
  existing passing shape) to prove nesting hints still work.

Run: `timeout 120 python -m pytest tests/test_overlay_scope_refusal.py -q`
then the full gate (README rule 3, expect 68 + your new tests).

## Regression sentinel you must check

The Shoal fluid program and the whole scorecard must still lower: the gate
covers this (`test_symbolic_fluid_native_runtime` does a fresh lowering,
the scorecard pins levels 0–18). If ANY gate member starts hitting your new
ValueError, a healthy program has multi-scope markers you did not expect —
that is a FINDING, not something to special-case around. Record it in this
file and stop.

## Definition of done

* New tests + the refusal, gate green, one commit.
* `## FINDINGS` note here: run
  `timeout 400 python tools/compile_re_probe.py 2>&1 | tail -5`
  once (takes ~2.5 min) and paste the new error text, confirming re now
  fails with the NAMED cause instead of the duplicate-count message.

## FINDINGS 2026-08-19

Done. `_marker_scope_paths` (module-level, mirrors `embed`'s exact
per-block-type insertion granularity) walks a candidate nested control's
region set through the parent's current `root` before `embed` runs, in
`overlay_scheduled_control`'s `nested_root`. Raises a `ValueError` naming
every distinct scope path and the region indices found there, instead of
letting `embed` insert once per scope and failing downstream on an opaque
duplicate-count mismatch.

Tests: `tests/test_overlay_scope_refusal.py`, 4 cases — healthy
loop-contained conditional (no error, exactly-once marker counts), the
cross-scope defect (refuses, message matches `"sequence scopes"`), message
content (both scope labels and both region-index lists present), and the
pre-existing `known_nesting` equal-region-set shape (still composes,
regression check against the existing `test_overlay_uses_known_nesting_...`
test in `tests/test_control_source.py`). One trap hit while writing the
defect-case test: giving `outer` and `conditional` IDENTICAL region sets
degenerates into the pre-existing, unrelated "maximal control blocks
overlap without containment" guard (equal sets aren't a strict subset, so
neither is ever seen as nested in the other) — fixed by giving `outer` one
extra exclusive region so containment is strict, which is also what real
programs look like.

Gate: 68 baseline + 4 new = **72 passed**
(`test_precompile_to_ssa`, `test_symbolic_fluid_native_runtime`,
`test_symbolic_fluid_direct_backends`, `test_abstract_tensor_indexing`,
`test_ssa_fusion_regions`, `test_region_kernel_dedup`,
`test_translation_scorecard`, `test_overlay_scope_refusal`). Also ran the
full `tests/test_control_source.py` since the shared module changed:
**18 passed**, no regressions.

Confirmation run, `tools/compile_re_probe.py` (full traceback, tail):

```
ValueError: nested control's regions span 3 sequence scopes of the parent
(control index 29, regions (3, 4, ..., 53)):
  top: regions [3, 4, ..., 36, 38, ..., 52]
  top > loop(iteration_317): regions [37]
  top > loop(iteration_322): regions [53]
```

This is MORE precise than the task's two-scope example: three scopes, and
it pinpoints exactly which two regions (37 and 53) are the strays sitting
inside two DIFFERENT small loops while the other ~48 regions of the same
cascade sit at top level — a strong lead for whoever picks up the `Raise`
loop-composition blocker next (those two loops are almost certainly the
`while.1`-shaped constructs the blocked main token loop should have
absorbed had it composed).

No unrelated files touched; the pre-existing untracked changes to
`src/common/tensors/*` present at task start were left alone and had
cleared by the time of the confirmation run (another session's).
