# Precision pipeline — state and deferred work

Rewritten 2026-08-23 after a sweep. Each item says what is true, where, and
how it was established. Items marked MEASURED were run, not reasoned about —
several of today's confident diagnoses were wrong and were caught only by
running them, so the distinction is load-bearing.

## What remains — smaller than it looked

An earlier draft of this file called the limb channel an open design problem
and the lowering "not yet made". That was wrong, and the evidence against it
was already on disk.

**The error-free transformations already compile, end to end.** MEASURED:
`TWO_PRODUCT_SOURCE` — ordinary authored Python — through the canonical
source compiler to SSA, lowered to Fortran, built by gfortran, returns the
exact primal AND the exact residual, 0 of 2000 mismatches each, residual
nonzero on every sample. It used Dekker splitting and no `Fma` at all, and
`t11 - (t11 - t7)` — the algebraically-foldable half — survived into the
emitted Fortran untouched.

So no destination needs to know what a precision operation is, and the limb
channel needs no invention: the kernel takes one array per limb (`p` and
`e`), which IS the channel layout materialised. That question was answered
by writing the function.

**What is actually left** is to connect `precision_mul` to that already-
working body — inline it, or emit a call to it. Until then a program that
declares precision still stops at the backend: Fortran says
`! UNSUPPORTED precision_mul`, LLVM reports
`operation has no repository LLVM emission`. Both refuse rather than emit
something wrong, which is the API behaving correctly; they are simply not
yet pointed at the body that works.

The remaining judgement is inline versus call, which is a cost question
(call overhead per element against code growth), not a correctness one.

## Working, and measured

* **Declarations reach SSA.** `Precision[2]` on a parameter or an annotated
  assignment produces `precision_mul` / `precision_add` / `precision_sub`.
* **Indexed operands work.** `a[i] * b[i]` on a declared `a` is recognised.
  Broken until today, and silently: every array-walking kernel lost its
  precision and lowered as ordinary arithmetic.
* **Precision propagates.** `carry_precision_through_ssa` puts the limb
  count on the value as a last-dimension channel plus an `accounting` fact,
  then renames consumers. A four-operation chain that kept two keeps four.
* **Identities fire.** Four of eight: per-limb negation, sub-as-add-of-neg,
  power-of-two scaling, one renormalisation per chain. MEASURED: a
  three-addition chain carries one renormalisation instead of three.
* **`Fma` on four lanes.** MEASURED against an exact rational oracle —
  C 0/2000, LLVM 0/500, Fortran 0/2000 mismatches. WASM expands to multiply
  and add, declares no capability, and is refused a section containing an
  `Fma` before emission.
* **Ordinary code is unaffected.** MEASURED: no precision operation appears
  in a program that declares none; LLVM and Fortran emit it cleanly. The C
  lane refuses loops (`direct scalar C requires one entry block`), a
  pre-existing structural limit of that backend, not a regression.

## Held back on purpose

**Neither SSA pass is wired into a compilation path.** They go in as one
transparent swap. Ordering when they do: carry before reduce; the
specific-name pass after planning.

## Corrected today — do not re-derive these

**The dual is not reliably folded away by optimizers.** Asserted here and in
several commit messages as established. MEASURED and it is not: gfortran
returned the exact `two_sum` residual at default, `-O2` and `-ffast-math`
alike, on inputs that genuinely lose bits. Lowering one SSA value per
statement through named temporaries appears to be why — no simplifier sees
the residual as a single expression. The hazard is real in principle (it is
why `NoContraction`, `precise` and `FP_CONTRACT` exist), but treat a
destination that inlines harder as untested, not as safe.

**`Fma` needed no new primitive and no compiler tag.** Two earlier claims,
both wrong. It is a named intrinsic with IEEE semantics, so nothing folds
it; Fortran passes while declaring no isolation at all.

**Parameter annotations were already collected.** `_TypeAnnotator` has no
`ast.arg` visitor, but `build_from_ast` publishes
`function_parameter_annotations`. Patching the annotator would have built a
second path to data that already existed.

## Blocked on machinery that does not exist

* **`exact_identity_element`** needs use-rewriting — the same machinery
  `x**1` needs in the `Pow` reduction. Should arrive with it.
* **`sterbenz_cancellation`** needs a proven range (catalogue section 5's
  fact slot). Highest-value blocked identity; must never fire on a guess.
* **`exact_accumulation_over_long_chain`** needs a banked superaccumulator.
* **The specific-name pass** — `PRECISION_OPERATOR_NAMES` (105 names) has no
  consumer. Needs operating width from the fused batch and return width from
  the actual consumer.

## Known-shallow, by choice

* **The width rule is fixed-width, not exact.** "Most limbs decides" gives
  `p2` for `p2 × p2`; Shewchuk's exact bound is `p8` (`2mn`), and `p4` for a
  sum (`m + n`). The current rule is the double-double preset; the bit-exact
  preset needs the other, and only one is implemented.
* **`r` always equals `p`.** No collapsing variant, since the return width
  is the consumer's property and no consumer has asked for one.
* **Fortran claims `FMA_MANDATORY`, not `SECTION_ISOLATION`.** It forbids
  reassociating a parenthesised expression but cannot withdraw contraction
  specifically. It passed the measurement anyway — which shows the hazard
  was not triggered, not that it cannot be.
* **`Div` untested.** The one closed operation whose expansion is iterative
  rather than a fixed transformation.

## Loose ends

* **`_TypeAnnotator`'s docstring is false.** It claims to be where an `int`
  parameter's integer-ness comes from; it has no `ast.arg` visitor. It
  misled me today and will mislead the next reader.
* **`graph_has_tensor_operation` is dead code** that reads as authoritative
  (`node_special_cases.py:718`). Nothing calls it. I reasoned from it before
  checking.
* **Lane closures exist and nothing produces one for precision.** `Deploy`
  and `Join` are registered operators and `deployment_ssa_binding` PROVES
  lane independence before binding. Only `loop_composer` builds a region.
  Limbs are not independent in general — `two_sum` propagates error between
  them — but the operations stamped `precision_form` (per-limb negation,
  power-of-two scaling) are exactly those where they are.
