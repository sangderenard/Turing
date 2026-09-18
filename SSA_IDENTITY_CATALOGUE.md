# The identity catalogue

Corollary to `HANDOFF_SSA_IDENTITIES_AND_DEPLOYMENT.md`. This is the itemised
analysis: which identities the compiler applies today, which it is missing,
what each is worth, and — the part with no home at all — which *facts* the SSA
has nowhere to record, without which several identities cannot be applied
safely at any price.

**How to read the numbers.** Everything under "measured" was taken this session
and is reproducible with `tools/bench_native_step.py` or a static read of the
emitted IR. Everything under "projected" is arithmetic on those measurements
and has **not** been observed. Do not quote a projection as a result.

---

## 0. The reference workload

Every figure below is anchored to one measured case: the per-cell kernel of the
symbolic fluid step, `symbolic_fluid_control__symbolic_fluid_step__planned_region_0`.

| fact | value | how known |
|---|---|---|
| kernel size | 290 instructions, 28 arguments | measured |
| op mix | 140 Mul, 91 Add, 24 Pow, 16 Max, 10 Abs, 9 Const | measured |
| divides / sqrts as instructions | **zero** — SymPy canonicalises both into `Pow` | measured |
| `Pow` exponents | 10× `2`, 6× `-1`, 6× `0.5`, 2× `-2`; **all compile-time constants** | measured |
| constant pool | `-4, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 2` | measured |
| cost | 1683 ns/cell, flat 24²→256², marshalling ≈ 0 | measured |
| emitted IR | 25 `@llvm.pow.f64` sites, 0 `noalias`, 0 fast-math, no `target triple` | measured |
| wrapper overhead | 290 `GetElementPtr`+`Load` per cell; `Ret` consumes 11 | measured |

A useful frame: of 290 instructions, **266 are single-cycle-class arithmetic**
(Mul/Add/Max/Abs/Const) and **24 are transcendental calls**. The 24 dominate by
roughly two orders of magnitude. Any identity work that does not touch them is
rounding error.

---

## 1. Present — what the compiler actually applies

Shorter than it looks, and two of the entries are dead code.

### 1.1 Applied, in the forward path

| identity | where | note |
|---|---|---|
| variadic `min`/`max` → binary fold chain | `hierarchical_plan.py:284` | genuine identity; keeps evaluation order visible |
| `clamp(x,lo,hi)` → `Max` then `Min` | `hierarchical_plan.py:305` | decomposition, one authored op to two primitives |
| scalar-vs-tensor op spelling | `TENSOR_OPERATION_SCALAR_SPELLING` | selects the scalar op when result and all operands are rank-0 |
| operator-authoritative dtypes | `plan_region_to_ssa_instrs` | comparisons → `bool`, `len`/`extent` → `int`, GEP indices → `int` |
| byte-string idiom folding | `ir_byte_idioms.fold_byte_string_idioms` | real fold, but byte-domain only, and on `FusedProgram` |
| string words → fnv1a tokens | `ir_string_interning` | real fold, string-domain only |

That is the complete list. **Not one of them touches floating-point
arithmetic.** The `ir_*.py` family reads like an optimizer directory and is not
one: `ir_indexing`, `ir_container_ops`, `ir_sequence_tables`, `ir_string_ops`
are *lowerings* — subscripts to addresses, containers to tables, strings to
tokens. Useful, necessary, and orthogonal to this document.

### 1.2 Present but never called — the two dead facilities

This is the finding that should sting.

| facility | location | callers |
|---|---|---|
| `aggressively_simplify_expression`, `aggressively_simplify_process_relations` | `symbolic_process_graph.py:396`, `:1142` | **zero, anywhere in the repository, including tests** |
| `quotient_common_subexpressions` (a CSE) | `machine_code_lifting.py:5132` | one test, `test_machine_code_lifting_roundtrip.py`, in the *decompilation* lane |

Both are exported in their modules' `__all__`. Neither runs on any forward
compile. A SymPy-level simplifier and a CSE were both written and neither was
wired to anything that compiles a program.

**Before building anything new, decide what happens to these two.** The
SymPy-level simplifier operates before SSA exists, on process relations, and is
a different tool from an SSA identity pass — it can restructure expressions
(factoring, common denominators) in ways an SSA peephole cannot. The CSE is
closer to directly reusable. Reimplementing either without reading it first
would repeat this session's characteristic mistake.

### 1.3 Present downstream, and only for one backend

LLVM's `-O2` performs constant folding, algebraic identities, CSE, DCE and
`pow(x,2)`→`x*x` on its own. **This is the compiler's entire optimizer today,
it is borrowed, and it is available to exactly one of seven backends.** SPIR-V,
WASM, WebGPU, C, Fortran and WebGL receive whatever the SSA hands them. That
asymmetry is the single strongest argument for putting identities at the SSA
level: it is not a speedup for the LLVM lane, it is the *only* optimizer the
other six will ever have.

---

## 2. Missing and critical

Ordered by measured value on the reference workload.

### 2.1 Constant-exponent `Pow` strength reduction — **LANDED 2026-08-19**

`ir_identities.reduce_constant_exponent_pow`, wired at the single `IRModule`
finalization point in `fortran_c_shell` (it must run AFTER region carving and
value pruning — run earlier it orphans exponent constants that recovered
output ledgers still name; journey 3 caught exactly that). Measured:

| policy | ns/cell | gates |
|---|---|---|
| baseline | 1683 | — |
| exact set (default): `2`, `-1` | ~900 (**1.85×**) | scorecard 10/10 at 0.0e+00 |
| `TURING_POW_INEXACT=1` adds `0.5`, `-0.5`, `-2` | ~480 (**3.5×**) | scorecard 10/10 at 0.0e+00; fluid `mass_err <= 1e-15` held |

The measured mass-error delta the numerics decision asked for: **within the
flagship's own 1e-15 assertion, the inexact set is indistinguishable.** The
original analysis follows.

| exponent | count/cell | lowering | exactness |
|---|---|---|---|
| `2` | 10 | `x*x` | exact |
| `-1` | 6 | `fdiv 1.0, x` | exact |
| `0.5` | 6 | `sqrt x` | differs only at `-0.0` and `-inf` |
| `-2` | 2 | `fdiv 1.0, (x*x)` | one extra rounding vs a single `pow` |

`-O2` already folds the ten `x**2`. The other fourteen survive because
`pow(x,-1)`→`fdiv` and `pow(x,0.5)`→`sqrt` need `afn`. *(projected: removing
all fourteen leaves ~266 cheap ops, i.e. tens of ns/cell rather than 1683 —
call it 15–30×, unobserved.)*

Implementation size: **small**. A table keyed on a `Const` operand, plus the
`Neg`/`Div`/`Sqrt` ops the backends already have.

Exactness is the live policy question. The exact half (`2`, `-1`) is 16 of the
24 and needs no permission from anybody. The inexact half (`0.5`, `-2`) is
worth having and should be argued with a measured mass-error delta, not a
principle.

### 2.2 Dead value elimination — the aggregate unpack

The step wrapper emits **290 `GetElementPtr`+`Load` pairs per cell** and its
`Ret` uses 11. LLVM may kill the other 279 if it can prove the aggregate is a
local `alloca`; across a call boundary with no aliasing facts, it may not. The
other six backends have no chance at all.

Implementation size: **small**, and it is the pass most likely to expose other
bugs, because anything it deletes was something an earlier stage thought was
live. Run it last.

### 2.3 CSE

The stencil reads centre and four neighbours for four fields, and each flux
divides by the same `h`. After 2.1 turns `h**-1` into a reciprocal, the
identical reciprocals become syntactically identical and collapse. *Before*
2.1, they are `Pow` calls the backend cannot merge. **This is the clearest
argument for the reduction ordering: CSE is worth little before strength
reduction and a lot after.**

Note `quotient_common_subexpressions` already exists (§1.2). Read it first.

### 2.4 Index-arithmetic strength reduction

The advance computes `(row-1) % height_count` and `(column±1) % width_count`.
The column pair is **two integer modulos per cell** against a runtime divisor —
20–40 cycles each on x86. Because `0 <= column < width_count` is an invariant of
the loop, both reduce to a compare-and-wrap.

*(projected: ~20–40 ns/cell — invisible today under 1683 ns, dominant once
2.1 lands.)* Do not do this before 2.1; it would be unmeasurable.

### 2.5 FMA formation — audited 2026-08-19, blocked by aliasing, not permission

140 Mul and 91 Add per cell, and the module names no target, so the backend
emits baseline SSE2 with no FMA.

**Audit result (measured).** An opt-in switch now exists —
`TURING_FMA_CONTRACT=1` puts the `contract` flag on every emitted
`fadd`/`fsub`/`fmul` (254 sites on the reference kernel, single chokepoint
`ssa_llvm_backend.scalar_likeness`) and adds `-march=native` to the zig cc
invocation. With both granted: **zero `vfmadd` in the compiled assembly and a
~2% perf delta.** The reason is structural: every float op in the emitted IR
reads its operands from memory slots (`%load.*` — there is not one textual
producer→consumer register chain in the kernel), and without `noalias` LLVM
cannot forward the slot stores to their loads across other stores, so
multiply→add chains never form in registers and there is nothing to contract.
The assembly shows 506 memory loads against ~250 float ops.

**Superseded same day:** the register chains were recovered WITHOUT `noalias`,
by construction instead of by analysis — a slot-keyed same-block register
cache in `_emit_repository_call_module` (see the handoff addendum). Loads
evaporate at emission (817 → 222 on the reference kernel); every store still
happens, so pooled in-place slot semantics are untouched. With the chains in
registers, `TURING_FMA_CONTRACT=1` now forms 17 `vfmadd` where it formed
zero. Default policy: ~280 ns/cell (6×); inexact: ~150–195 (~10×); +contract:
~140–160. The original analysis below stands as the record of why the
memory-form IR blocked everything.

**So the FMA dependency chain was: P2 `noalias` derivation → store-to-load
forwarding → register chains → contraction.** The switch is correct, off by
default (an fma rounds once where mul+add round twice, so contracted results
differ bitwise from every other backend), and becomes valuable exactly when
P2 lands. Expressibility elsewhere, for an eventual SSA-level `MulAdd`:
LLVM (`contract`/`llvm.fmuladd`), C (`fma()`), SPIR-V (GLSL.std.450 `Fma`)
and WGSL (`fma()`) can all say it; WASM (scalar) and Fortran cannot except
through toolchain contraction flags.

---

## 3. Missing and trivial

Each is a few lines, each is exactly result-preserving, and none of them will
show up on the fluid benchmark — the kernel is already dense. **Their value is
not speed on this workload; it is that they are the substrate every later stage
assumes.** Constant folding that does not fold `x*1` leaves literals uncollapsed
that strength reduction then fails to recognise.

| identity | form | exact |
|---|---|---|
| multiplicative unit | `x*1`, `1*x` → `x` | yes |
| additive unit | `x+0`, `0+x`, `x-0` → `x` | yes |
| annihilator | `x*0` → `0` | **no** for `x = NaN`/`inf`; gate on a finiteness fact (§5) |
| self-subtraction | `x-x` → `0` | **no** for `x = inf`/`NaN`; same gate |
| division unit | `x/1` → `x` | yes |
| power units | `x**1` → `x`, `x**0` → `1` | `x**0` differs at `NaN` |
| double negation | `-(-x)` → `x` | yes |
| idempotent extrema | `Max(x,x)`, `Min(x,x)` → `x` | yes |
| negate-and-add | `Add(x, Neg(y))` → `Sub(x,y)` | yes |
| identity cast | `Cast(x, dtype)` where source dtype == target | yes |
| constant deduplication | two `Const` with equal value and dtype → one | yes |
| constant folding | all-`Const` operands → one `Const` | yes, modulo rounding mode |

The `x*0` and `x-x` entries are the instructive ones: **the obvious identities
are the unsafe ones.** Both are wrong under IEEE unless something guarantees the
operand is finite, and today nothing can express that guarantee. Which is §5.

---

## 4. Potential — structural identities that unlock other stages

These are not peephole rewrites. Each is a recognition that changes what a later
stage is permitted to do.

### 4.1 Associative-reduction recognition — **unlocks parallelism**

`acc = acc + x` and `acc = max(acc, x)` are associative and commutative, so they
do not serialize a loop; they split into per-lane partials and one final
combine. The current `parallel_candidate` gate vetoes on *any* carried binding.

The fluid loop carries five of them: `previous_mass`, `next_mass`,
`max_wave_speed`, and two violation maxima. **Every one is an associative
reduction.** This is why it must land with the dependence test in P3 and not
after: fix the unrolling coupling alone, and the loop is rejected again on the
next conjunct.

Floating-point addition is not associative, so a lane split changes the sum's
rounding. The fluid asserts `mass_err <= 1e-15` on a quantity computed by
exactly this accumulator, so this identity has a numerically visible
consequence and needs the same measured-delta treatment as §2.1.

### 4.2 Loop-invariant code motion

Prerequisite for §2.4 and generally, and it needs the loop structure the
deployment work is already touching.

### 4.3 In-place slot recognition → aliasing facts

The carried-slot design — one slot per carried value, read at the top, written
before the latch — is *intentional* (`project-frame-storage-aliasing-regulation`:
regulate, never prune). Recognising it is what licenses `noalias` on everything
that is **not** such a slot. Blanket `noalias` would be a lie; derived
`noalias` is P2's actual content.

### 4.4 Cast-chain collapse

`Cast(Cast(x, a), b)` → `Cast(x, b)` when the intermediate is not narrowing.
Cheap, and the `tensor`→`Cast` normalization in `plan_region_to_ssa_instrs`
manufactures these.

---

## 5. The identities we have nowhere to record

**This is the gap the title of this document points at, and it is not a missing
pass — it is a missing datum.**

Half of §3 and both numerics questions in §2 come down to facts about a value
that are true, knowable at compile time, and unrepresentable in the IR:

| fact | who knows it | what it licenses |
|---|---|---|
| `x` is finite (no `NaN`, no `inf`) | the clamp/bound structure of the source | `x*0` → `0`, `x-x` → `0` |
| `x > 0` | `state.height` is clamped to `minimum_height = 1e-4` | `x**0.5` → `sqrt` **exactly**, because the `-0.0`/`-inf` cases cannot arise |
| `x != 0` | same | `x**-1` → `fdiv` with no division-by-zero path |
| these two buffers are distinct | the storage-formal ABI's carried-slot table | `noalias`, and therefore vectorization |
| this accumulator is an associative reduction | the loop's own update shape | lane splitting |

Note what the second row buys: **the inexact identities become exact when the
domain is known.** `pow(x,0.5)` and `sqrt(x)` differ only at `-0.0` and `-inf`.
If the IR could carry "this value is strictly positive", the substitution is
bit-exact and the whole numerics policy argument in §2.1 evaporates for that
case. Today the choice is between a global fast-math sledgehammer and leaving
six libm calls per cell, because there is no way to say the one true thing that
would make the rewrite safe.

**`SSAValue` already has the slot.** `transmogrifier/ssa.py:54` —
`accounting: Dict[str, Any]`, already carrying `program_abi_field`,
`program_abi_rank` and friends, already threaded through `dataclasses.replace`
in `fortran_c_shell.py`. A value-fact namespace inside it costs no new plumbing.

The proposal, and it should be settled before `ir_identities.py` is written
rather than retrofitted:

1. Reserve an `accounting["facts"]` set with a small closed vocabulary:
   `finite`, `positive`, `nonzero`, `integral`.
2. Facts are **asserted only where they are proven**, never inferred from names
   or observed runtime values — the same discipline `assign_hierarchy_ids`
   already documents for identity.
3. Facts **propagate through identities**, and the pass that consumes a fact
   records which one it used, so a wrong rewrite names its own premise.
4. An identity whose exactness depends on a fact **refuses to fire without it**,
   rather than firing under a global flag.

That last point is the whole design. It is the difference between "we turned on
fast-math and the mass error moved" and "this value is positive, therefore this
rewrite is exact, and here is the assertion that proves it."

---

## 6. Ordering, and why it is the design

```
constant folding
  └─> algebraic identities (§3)
        └─> strength reduction (§2.1, §2.4)
              └─> CSE (§2.3)
                    └─> dead value elimination (§2.2)
```

Each arrow is a real dependency, not a preference:

* identities before strength reduction — `x**(1+1)` must become `x**2` before
  the exponent table can see a `2`;
* strength reduction before CSE — six `Pow(h,-1)` are six opaque calls; six
  `fdiv 1.0, h` are one subexpression;
* everything before DCE — DCE collects what the others orphan, and running it
  early hides the evidence of what the others did.

Run to a fixed point, bounded, and log the round count. A pass that keeps
finding work is either correct and useful or oscillating between two rewrites;
the count tells you which.

---

## 7. Verification protocol

Non-negotiable, because an identity pass that is subtly wrong produces exactly
the class of silent miscompilation `TRANSLATION_DEBUGGING.md` catalogues.

1. **The scorecard, all 10 journeys**, before and after — the frontier is a fact
   under test.
2. **The reference evaluator** on the same SSA — an independent executor is what
   distinguishes "the rewrite is wrong" from "the backend is wrong".
3. **Per-identity round-trip**: the round-trip materializer reads the compiled
   program back as Python. An identity that changes what that reads back has
   changed the program, whatever the numbers say.
4. **`tools/bench_native_step.py`** before and after, so each identity's
   contribution is attributable instead of a lump.
5. For any identity in the inexact set, **report the mass-error delta** on the
   fluid flagship. That number is the argument.

---

## 8. Summary table

| item | § | value on the reference workload | implementation | exact |
|---|---|---|---|---|
| constant-exponent `Pow` | 2.1 | **landed**: 1683→900 exact, →480 opt-in | small | 16 of 24 yes |
| dead aggregate unpack | 2.2 | 279 dead loads/cell | small | yes |
| CSE | 2.3 | large, but only after 2.1 | medium (one exists) | yes |
| index modulo → wrap | 2.4 | ~20–40 ns/cell, projected | small | yes |
| FMA formation | 2.5 | **audited**: 0 fuse until P2 `noalias` | switch landed | no — opt-in |
| trivial identities | 3 | ~none here; substrate for the rest | trivial | mostly |
| associative reductions | 4.1 | unlocks all parallelism | medium | no — measure |
| value facts | 5 | makes the inexact set exact | small, design-first | n/a |

**If only one thing gets done: §2.1.** It is small, 16 of its 24 cases need no
policy decision, and it is the entire measured cost of the flagship kernel.

**If only one thing gets designed: §5.** Every other decision on this list
becomes easier once a value can carry a proven fact, and harder to retrofit once
identities exist that assume it cannot.
