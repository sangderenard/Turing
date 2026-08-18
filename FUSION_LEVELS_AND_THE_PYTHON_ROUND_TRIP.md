# Fusion levels, the Python round trip, and the stateful network

Written 2026-08-18. This is a direction note, not a design document: it records
an intention and the order of the steps, so the through-line survives being put
down. What exists today is marked; everything else is explicitly not built.

## The goal at the end of it

A stateful network — a training loop that carries parameters and optimizer
state across iterations — authored in ordinary Python, compiled, and *checkable*.
Not "it ran and the loss went down", but: you can see what the compiler did to
your program, because the compiler can hand it back to you as Python.

## Why this route, and not more debugging

Every failure this pipeline produced today was silent and plausible, not loud:

* a tape lowering reporting COMPLETE over zero instructions, because "no
  shortfalls" is vacuously true of an empty program;
* a captured training step with a frozen gradient — bit-exact against an eager
  loop for two steps, then walking off to a loss of 8131;
* a capture returning a real-but-empty program with zero shortfalls;
* a loop body silently reduced to `['Br']`, reported as an internal value id
  with no producer.

None of these raised. Each was found by comparing against something independent
and noticing a number that was wrong. That is not a debugging strategy that
scales, and it is the argument for both halves of this note: a **level** makes
what you asked for falsifiable, and a **round trip** makes what you got
readable.

## Step 0 — done: the vocabulary and its refusals

`src/compiler/fusion_levels.py`. Four levels, most preserving first, each with
a testable invariant rather than a description:

    PRESERVE   every authored operator appears in the IR, in source order
    NO_FUSION  no emitted step carries more than one authored operator
    REGIONS    region boundaries survive; interiors unconstrained
    FUSED      no guarantee

`REGIONS` and `FUSED` are what `precompile_only` True/False already select.
`PRESERVE` and `NO_FUSION` are declared and **raise**, naming what is missing.
They deliberately do not fall back to `REGIONS`: a caller who asks for no
collapsing and silently receives collapsing has been handed exactly the kind of
plausible wrong answer this vocabulary exists to prevent.

No pipeline stage moved. The boolean still works and still means what it meant.

## Step 1 — next, and small: the region-call ABI in the materializer

`src/compiler/ssa_python_materializer.py` already turns single-block SSA back
into runnable Python, verified against the symbolic fluid step to 2.1e-17. Run
against a real lowered loop, the **region bodies already round-trip**:

    rt__update__planned_region_0   [Const, Mul, Sub]  ->  t1 = 0.05
                                                          t2 = t1 * t0
                                                          t3 = t0 - t2

What refuses is the *wrapper*, and not for control flow — it is single-block.
It invokes a region with `Call` + `GetElementPtr` + `Load`, and the materializer
has no Python form for `GetElementPtr`. That is missing vocabulary, about the
size of the eleven backward rules added earlier today.

Doing it is worth more than the round trip alone: it renders the region calling
convention **as Python**, which is the most direct way to inspect an ABI that
has not been pinned down.

## Step 2 — bounded, not open-ended: counted-loop reconstruction

The handoff called CFG-to-structured-control an open seam, and in general it is.
The shape this compiler emits is not general:

    entry        [Const, Const, Br]
    loop_header  [Phi, Phi, Lt, CondBr]
    loop_body    [Call, ..., Load, Br]
    loop_latch   [Add, Br]
    loop_exit    [Call, Ret]

The header is phi-phi-compare-branch, the latch is the increment, and the phi
carries `incoming_blocks` and `source_name`. Recognising *that* pattern and
emitting a `while` with carried names is a targeted reconstructor. It is not
the general problem and should not be allowed to become it.

## Step 3 — what the two halves buy together

With `PRESERVE` and a round trip, the pair closes:

* `PRESERVE` -> Python must equal the source that was authored. That is not a
  description of the level, it is a **test** of it.
* Every level below is then a *diff*: exactly which operators the next rung
  ate, in readable Python.

Today the equivalent question — "what did the compiler do to my program?" — was
answered by patching a shortfall constructor to read stack frames.

The immediate payoff is the bug already characterised. The failing loop body is
`['Br']`; materialised it reads as

    for iteration_4 in range(...):
        pass          # the work is visibly gone

which turns `carried update value 5 has no producer` into something you can see
and diff against what you wrote.

## Prerequisite, worth doing regardless

The failing case *raises* instead of returning its module, so today only the
working one can be materialised. Any inspect-the-IR workflow needs the lowering
to hand back its module alongside its shortfalls. "Give me the IR even when it
is wrong" is the precondition for all of the above.

## Two open questions, recorded rather than quietly decided

**One ladder or two axes?** Merging adjacent operators and eliminating them
outright are different operations. The ladder is sound only if each level
strictly disables everything above it plus more. This is not hypothetical: the
`loop_carried` failure is region *elision* — elimination — so `NO_FUSION` would
not touch it and only `PRESERVE` reaches it. Settle it by checking whether
`NO_FUSION` output is always a subset of `PRESERVE` output for the same source.

**`precompile_only` controls two unrelated things.** Per `aot_compile`'s
docstring, `emit_glsl` is gated purely by it. So it means both "how much
collapsing" and "which artifacts are emitted", and any level mapped onto it
inherits the conflation — `PRESERVE` would silently also mean "no GLSL". If the
boolean is ever replaced, separating emission is most of the value.

## What is NOT claimed

* No pipeline stage has been changed by this note or by step 0.
* `PRESERVE` and `NO_FUSION` do not work. They raise.
* The route from `REGIONS`-with-round-trip to a compiled stateful training loop
  is not proven. The loop-carried round trip does not lower yet; the authoring
  workaround (bind through one extra operation) is known but untested on a real
  Adam update, which may hit a second limit once the first is worked around.
