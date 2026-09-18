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

## FINDINGS 2026-08-19

**The task's own starting hypothesis is wrong, and the SSA dump proves
it directly.** Reproduced and dumped the full lowered module (all
functions, not just the two refusing ones). `sc18__train` already
contains a genuine, correctly-composed loop:

```
loop_header: t19 = Phi(...); CondBr(t19 < n)
loop_body:   t22 = Call(region_0)  # k*0.5**k < 0.1, the predicate
             t9  = Load(...)        # this iteration's bool result
             t25 = GetElementPtr(t16, t19) {binding: collection_publication}
             Store(t9, t25)         # publish into a materialized sequence
loop_exit:   t29 = Call(region_2, t16)   # region_2: t13 = any(t16)
```

**The generator's `for k in range(n)` part is not undecomposed — it is
already a real loop**, materializing each predicate result into a
sequence (value 16) via the standard "collection publication" pattern
(the same mechanism a `list(...)` comprehension would use). Nothing
here is opaque or missing. The only two things the materializer refuses
are (a) region_2's single `any(t16)` call over the now-complete
sequence, and (b) the computed-index `GetElementPtr(t16, t19)` inside
the loop body that publishes into it.

**These two refusals are INDEPENDENT, not cause-and-effect** — the
task's own "likely downstream fallout... verify rather than assume" is
answered: they sit in different functions entirely (`loop_body` inside
`sc18__train` itself vs. `sc18__train__planned_region_2`), so the
second is not fallout of the first. Both are separate occurrences of
the exact same missing capability.

**That capability is not a compiler decomposition gap — it is that the
Python round-trip MATERIALIZER (not a real backend) has zero
sequence/tensor-value representation at all**, and `any`/`all` are not
scalar opcodes in the first place:

* `src/compiler/ssa_llvm_backend.py`: `_BINARY` (line 24) and `_UNARY`
  (line 66) are the SCALAR likeness tables the materializer's own audit
  imports and checks against (`ssa_python_materializer.py`:
  `from .ssa_llvm_backend import _BINARY as _LIKENESS_BINARY, _UNARY as
  _LIKENESS_UNARY`). `any`/`all` are NOT in either.
* They live instead in `_TENSOR` (line 128), a completely separate
  table for array/tensor operations, both mapped to a real backend
  kernel symbol: `"any": "reduce_dim_double"`, same family as `"sum":
  "sum_double"`, `"max"/"min"/"prod": "reduce_dim_double"` — i.e. `any`
  is architecturally a TENSOR REDUCTION, exactly like `sum`/`max`, not
  an unimplemented scalar op.
* Every REAL backend already handles it: `ssa_fortran_backend.py`
  spells it directly as Fortran's native intrinsic (`"any":
  "any({0})"`, `"all": "all({0})"`); SPIR-V and WASM both list
  `any`/`all` in their own reduction vocabularies
  (`ssa_spirv_backend.py:193`, `wasm_class_modules.py:66`).
  `process_graph_autograd.py` and `tensor_ssa_lowering.py` both
  enumerate `any`/`all` alongside `sum`/`prod`/`min`/`max` as one
  reduction family throughout.

Because `_TENSOR`'s vocabulary is invisible to the materializer's
audit, `any` never reaches `UNIMPLEMENTED` (the set of ops the
materializer KNOWS about but declares out of scope) — it falls all the
way through to the generic "no Python form for 'any'. Add it
deliberately" catch-all, which is the correct behavior for a genuinely
UNKNOWN-to-this-file operation, just not evidence of a missing
decomposition anywhere in the compiler.

**Consequently the task's explicit warning ("giving the materializer an
`any` spelling would mask a lowering gap — do not do that") was reasoned
from a premise this measurement disproves — there is no lowering gap to
mask.** The real fix, if pursued, is teaching the materializer an
entirely new CAPABILITY it has never had: representing a materialized
sequence as a Python list, spelling a computed-index
`GetElementPtr`/`Store` as `some_list[i] = value`, and spelling
`any`/`all`/`sum`/`prod`/`min`/`max` over that list as the literal
Python builtin/reduction they already exactly mean (unlike a scalar op
of ambiguous bit width, `any([...])` has no interpretation to guess —
it is Python's own `any`). That is a genuine feature addition to the
round-trip diagnostic tool (sequence values, computed addressing, and
the whole reduction family at once, since they all share the same gap),
not a small, single-opcode fix — out of scope for this task's stretch
goal as originally framed. Not attempted here; flagging for a
dedicated task ("give the Python materializer sequence/collection
support") rather than folding it into this one.

No code changed. Scorecard unchanged at 17/19 with level 18 still
correctly stopped at MATERIALIZE — but now for the accurate reason.
