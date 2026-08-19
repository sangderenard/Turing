# Task C — scorecard level 18: `any()` over a generator never renders

Read `README.md` in this directory first. Difficulty: medium. Primary
deliverable: a named seam with evidence; fix only if small.

## The symptom

`timeout 120 python tools/translation_scorecard.py`:

```
18  any() over a generator predicate  STOP MATERIALIZE  not materialized: ['sc18__train', 'sc18__train__planned_region_2']
```

Authored program (level 18 in `tools/translation_scorecard.py`):

```python
def train(w, n):
    hit = any(w * 0.5 ** k < 0.1 for k in range(n))
    return helper(w) + hit
```

## The measured refusals (verified 2026-08-19)

Materializing the lowered module reports two distinct reasons:

```
sc18__train__planned_region_2: no Python form for 'any'. Add it deliberately...
sc18__train: GetElementPtr %t25 addresses by computed path rather than by aggregate index
```

Meaning: the generator's reduction survived lowering as a single opaque
`any` opcode inside a planned region, instead of being decomposed into a
loop with an OR-reduction (or early-exit) over the predicate. The second
refusal (computed-path GetElementPtr in the caller) is likely downstream
fallout of the first — verify rather than assume.

Reproduce with the same probe pattern as Task B (swap in this SOURCE, name
"sc18", and also dump `sc18__train__planned_region_2`).

## Context that constrains the fix

* The repository's doctrine (recorded in project memory and in
  `HANDOFF_tensor_op_ssa_modules.md`): **tensor/reduction ops resolve to
  primitives by ingesting reference list-math into SSA — hand-rolled op
  decompositions were tried and were WRONG.** So do not hand-write an
  `any` lowering inside a backend. Find where reductions like `max`
  acquire their loop decomposition (level 4–9 loops and the fluid model's
  `max(...)` reductions all lower today — locate that path and see why
  `any` is not in its vocabulary).
* The materializer is deliberately vocabulary-limited to what
  `ssa_llvm_backend`'s likeness tables contain (read the module docstring
  of `src/compiler/ssa_python_materializer.py`). `any` reaching it AT ALL
  is the defect; giving the materializer an `any` spelling would mask a
  lowering gap — do not do that.
* The generator machinery has a diagnostic:
  `tools/dump_comprehension_graph.py` (named in
  `HANDOFF_SHOAL_AND_RE_TARGETS.md` §6c as the pairing for this stall).

## Method

1. Dump the region's SSA and confirm exactly one `any` instruction and
   what its operands are (a sequence? a carried value?).
2. Trace where the `any` node comes from: reducer
   (`src/common/tensors/topological_reducer.py` — search how
   `ast.GeneratorExp` and `any`/`all` calls are reduced) vs planning.
3. Compare against a construct that WORKS: `max_wave_speed =
   max(max_wave_speed, wave_speed)` in the fluid program, or an authored
   `sum(...)`/loop-accumulator from scorecard levels 4–7. The difference
   between the working reduction's path and `any`'s path IS the seam.

## Definition of done

* Minimum: `## FINDINGS` here naming where `any` should have been
  decomposed and why it was not (with the graph/SSA evidence), plus
  whether the caller's computed-path GetElementPtr is fallout or a second
  independent defect.
* Stretch: the decomposition, through the same path existing reductions
  use. Gate green (README rule 3); if level 18 passes, move
  `EXPECTED[18]` in `tests/test_translation_scorecard.py` in the same
  commit. Note `any` over a generator must SHORT-CIRCUIT semantically —
  if the decomposition you find evaluates all `n` terms, that is fine for
  this pure probe but say so explicitly in the commit message (it is a
  semantic difference under side effects).
