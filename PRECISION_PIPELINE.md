# The precision pipeline

State as of 2026-08-23. Claims marked MEASURED were run; the rest are
structural facts about the code. The distinction matters here because
several confident diagnoses during this work were wrong and only running
them caught it — each of those is recorded below rather than quietly fixed.

## What it does

Declare a width in ordinary Python and the arithmetic under it becomes
double-double, compiled, on a real backend:

```python
def cancel(a: Precision[2], b: Precision[2], out, n):
    for i in range(n):
        out[i] = (a[i] + b[i]) - a[i]
    return out
```

MEASURED, compiled to Fortran and run natively — the same source, one
annotation apart:

| | result |
|---|---|
| ordinary | `[0.0, 4.0, 0.0, 0.0]` |
| `Precision[2]` | `[1.0, 3.7, 1e-16, 5.0]` |
| exact `b` | `[1.0, 3.7, 1e-16, 5.0]` |

Ordinary arithmetic loses `b` completely in three of four. That single
annotation is the whole difference.

## The path a program takes

1. **Recognition** — `topological_reducer._operand_precision` reads the
   declaration off the AST and `_qualified_handler` renames the operation to
   `precision_mul` and siblings. Declarations come from annotated
   assignments (`type_annotations`) and from parameters
   (`function_parameter_annotations`). Indexed operands are read through to
   the name they index, so `a[i] * b[i]` counts.

2. **Carrying** — `ir_identities.carry_precision_through_ssa` puts the limb
   count on the SSA value and renames consumers, to a fixed point. Without
   this, precision stops at the first undeclared temporary.

3. **Identities** — `reduce_precision_operations` fires four of the eight in
   `PRECISION_IDENTITIES`: per-limb negation, sub-as-add-of-neg,
   power-of-two scaling, and one renormalisation per chain. MEASURED: a
   three-addition chain carries one renormalisation instead of three.

4. **Section marking** — `mark_precision_sections` stamps every instruction
   in a section, because the operator is about to be expanded away and the
   *boundary* has to outlive it.

5. **Lowering** — `lower_precision_operations` expands every `precision_*`
   into ordinary `Mul`/`Add`/`Sub`/`Neg`/`Fma`. Afterwards none remains.

6. **Emission** — an ordinary backend compile.

## The two decisions that mattered

**A precision value is one SSA value per limb, not one value with a limb
axis.** This was the thing that blocked progress longest, and the answer was
already on disk: the working `two_product` kernel writes its two limbs to
two arrays. Per-limb values survive because each limb is then an ordinary
scalar every backend can already hold, load and store; the channel-shaped
alternative requires a destination to understand an aggregate before doing
arithmetic on one, and none do.

**The operator is scaffolding; the section is load-bearing.** `precision_mul`
exists to be recognised, propagated and reduced against, and is expanded
before any backend sees it — which is why no backend implements one. But the
expansion is plain arithmetic indistinguishable from anyone else's, so the
section attribute carries the boundary in the operator's place. Stamping
instructions rather than recording a range is what makes it durable: an
expansion inheriting its source's attributes stays marked, and a pass that
moves an instruction cannot move it out of its section.

## `Fma` across the backends

`Fma(x, y, z)` is `x * y + z` under exactly one rounding — the operation, not
a licence to fuse. MEASURED against an exact rational oracle (`Fraction`
arithmetic, not another float computation that could agree by sharing a
mistake):

| lane | spelling | result |
|---|---|---|
| C | `fma()` (C99) | 0 / 2000 mismatches |
| LLVM | `@llvm.fma.f64` | 0 / 500 |
| Fortran | `ieee_fma()` (F2018) | 0 / 2000 |
| WASM | `mul` + `add` | refused before emission |

`BACKEND_PRECISION_CAPABILITIES` is what keeps that unified front honest:
**emitting is not meeting.** WebAssembly has no fma instruction, so it
expands and declares nothing, and a section containing an `Fma` is refused
there before emission rather than discovered afterwards in a residual that
came back zero. Ordinary code wanting the accuracy still compiles on all
four; only code whose correctness depends on the single rounding is turned
away.

The obligations are deliberately not uniform in kind. `FMA_MANDATORY` and
`SECTION_ISOLATION` are requirements a destination meets or must refuse;
`LANE_STAGING` is permission, so a backend ignoring it conforms rather than
fails.

## Findings that corrected earlier claims

**`Fma` needed no new primitive and no compiler tag.** Recorded twice as
blocked on a missing primitive. Wrong twice: LLVM's `contract` flag was
already per-instruction, and adding the intrinsic was a table entry. It
works because a named intrinsic with IEEE semantics has no algebraic pattern
to fold — we replaced the fragile thing rather than protecting it.

**Optimizers did not delete the dual.** Asserted repeatedly as established.
MEASURED and false for this toolchain: gfortran returned the exact `two_sum`
residual at default, `-O2` and `-ffast-math` alike, on inputs that genuinely
lose bits, and `t11 - (t11 - t7)` survived into the emitted Fortran intact.
The likely reason is structural — one SSA value per statement through named
temporaries means no simplifier sees the residual whole. The hazard is real
in principle (it is why `NoContraction`, `precise` and `FP_CONTRACT` exist),
so treat a destination that inlines harder as untested, not safe.

**Parameter annotations were already collected.** `_TypeAnnotator` has no
`ast.arg` visitor, but `build_from_ast` publishes
`function_parameter_annotations`. Patching the annotator would have built a
second path to existing data. Measuring before editing caught it.

**Indexed operands were silently unrecognised.** `a[i] * b[i]` lowered as
ordinary arithmetic because the check accepted only a bare `ast.Name`. The
declaration survived on scalars and was lost by every kernel that walks an
array — which is all of them.

**A "wrong answer" was my own test harness, twice.** Fortran appeared to
compute zero; I had called the loop-body region with `n` where it expected
`i`. A plain-multiply control caught it — a multiply cannot be blamed on
rounding. Later the two_product outputs looked wrong because I passed `p`
and `e` swapped.

## Coefficient capture — a standing constraint

Coefficients are captured AND used at four limbs; the operating width is
then the least that meets the target, chosen per core rather than fixed.

MEASURED on the `sin` core's nine structured coefficients:

| capture | worst relative error |
|---|---|
| 1 limb (what the kernels marshal today) | 7.837e-17 |
| 4 limbs | 1.236e-65 |

Against a half-ulp budget of 1.11e-16, single-double capture spends about a
third of the budget before any arithmetic runs. The compiled core measures
correctly rounded despite that, not because of it.

Capture width and operating width are separate decisions. Capture is free —
it happens once at build time from exact rationals, via
`limb_decomposition(value, 4)` — while operating width is paid per
operation, so it should be searched upward from the smallest rather than
chosen comfortably.

**Blocked on an ABI gap.** A `Precision[n]` parameter still arrives as ONE
scalar, while the lowering represents a precision value as n separate
scalars. Four-limb coefficients therefore cannot be passed in yet: it needs
a declared `Precision[n]` parameter to become n formals, which is the same
decision the lowering already made internally and has not been extended to
the boundary.

## The signal cores, compiled

MEASURED through LLVM against exact rational evaluation of the same
polynomial, 25 points per core inside its own reduced interval. Worst
relative error, expressed in ulp (half an ulp = correctly rounded):

| core | parity | ordinary | Precision[2] |
|---|---|---|---|
| sin | odd | 0.57 | 0.49 |
| cos | even | 0.46 | 0.32 |
| exp | none | 0.53 | 0.44 |
| atan | odd | 0.50 | 0.46 |
| tanh | odd | 0.44 | 0.41 |
| sinh | odd | **0.76** | 0.42 |
| asin | odd | **0.73** | 0.34 |
| sec | even | 0.47 | 0.47 |

`sinh` and `asin` are NOT correctly rounded as ordinary arithmetic -- they
exceed half an ulp and therefore return the wrong double for some arguments
-- and both are correctly rounded at two limbs. That is the case the whole
pipeline exists for: precision is not shaving an already-good number, it is
the difference between right and wrong.

All 19 materialised cores use only `+` and `*`; none divides. The untested
`Div` expansion is therefore not on the signal pack's path at all.

`expm1` and `log1p` have structure `factored` and are not measured above:
the comparison harness only models plain Horner with a parity multiply, so
those figures would be its error rather than theirs.

## What remains

- **Neither pass is wired into a compilation path.** They are called
  explicitly. They go in as one transparent swap; ordering is carry → reduce
  → mark → lower.
- **Four identities are unfired.** `exact_identity_element` needs
  use-rewriting (the machinery `x**1` also wants);
  `sterbenz_cancellation` needs a proven range (catalogue section 5);
  `exact_accumulation_over_long_chain` and `two_product_kernel` need the
  bank.
- **The width rule is fixed-width, not exact.** "Most limbs decides" gives
  `p2` for `p2 × p2`; Shewchuk's exact bound is `2mn` for a product and
  `m + n` for a sum. The current rule is the double-double preset; the
  bit-exact preset needs the other.
- **`Div` is untested** — the one closed operation whose expansion is
  iterative rather than a fixed transformation.
- **Isolation is blunt away from LLVM.** LLVM withholds `contract` per
  instruction. C emits `FP_CONTRACT OFF` for the whole translation unit;
  Fortran has no mechanism and honestly declares none.
- **A cleaner isolation exists and is unbuilt.** Each region is already
  minted as its own function on every backend
  (`..._planned_region_0` is a Fortran `subroutine` and an LLVM `define`),
  so a function-level attribute would replace per-instruction stamping with
  one portable marker — provided the region is not per-element, which is
  unmeasured.
- **Lane closures exist and nothing produces one for precision.** `Deploy`
  and `Join` are registered operators and `deployment_ssa_binding` proves
  lane independence before binding. Limbs are not independent in general —
  `two_sum` propagates error between them — but per-limb negation and
  power-of-two scaling are exactly the cases where they are.

## Loose ends worth closing

- `_TypeAnnotator`'s docstring claims to be where an `int` parameter's
  declared type comes from. It has no `ast.arg` visitor. It misled me.
- `graph_has_tensor_operation` (`node_special_cases.py:718`) reads as
  authoritative and is called by nothing. I reasoned from it before checking.
