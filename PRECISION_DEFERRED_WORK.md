# Precision pipeline — deferred work

Written 2026-08-23, while the focus was the identity table. Everything here
was found and left alone deliberately. Nothing in this list is a mystery;
each item says what is wrong and where.

## Held back on purpose

**Neither SSA pass is wired into any compilation path.**
`carry_precision_through_ssa` and `reduce_precision_operations`
(`src/compiler/ir_identities.py`) both work when called and are called by
nothing. This is deliberate — they go in as one transparent swap. When they
do, the placement question is real: the carry pass must run before the
reduction phase, and the specific-name pass must run after planning.

## Recognition gaps

**Parameter annotations: fixed, but the annotator is still misleading.**
`def f(a: Precision[2])` used to compile silently as ordinary arithmetic.
The cause was real but the obvious remedy was wrong: `_TypeAnnotator`
(`node_special_cases.py:455`) genuinely has no visitor for `ast.arg`, so
parameter declarations never reach `type_annotations` -- but `build_from_ast`
already unparses every one of them into `function_parameter_annotations`.
`_operand_precision` now reads that instead, rather than teaching the
annotator a second route to a fact already collected.

Two things remain. `_TypeAnnotator`'s docstring still claims it is "where an
`int` parameter's integer-ness comes from", which is not true and will
mislead the next reader exactly as it misled me. And because the reduction
walk does not carry which function it is inside, a parameter name is only
honoured when every function declaring it agrees; where two disagree the
operand reads as ordinary. That is a miss rather than a wrong width, which
is the right way round, but it is a real limit worth removing by carrying
the enclosing function identity.

**`precision_element` is `None` at ingestion.**
`Precision[2]` gives limbs but no element type, so the specific-name lookup
— keyed `(operation, element, limbs)` — cannot resolve until the carry pass
fills it from the SSA value's dtype. Names therefore cannot be made specific
without running the carry pass first. Not a defect, but an ordering
constraint that is currently implicit.

**`Div` is untested.**
`precision_div` is generated and named but no test has exercised it.
Division is the one closed operation whose expansion algorithm is iterative
rather than a fixed transformation, so it is the least likely to behave like
the other four.

## Blocked on machinery that does not exist yet

**`exact_identity_element` needs use-rewriting.**
Dropping `add(x, 0)` means pointing every consumer at `x`.
`src/compiler/ir_identities.py` avoids use-rewriting on purpose — it is
stated as the reason `x**1` is missing from the `Pow` reduction. Both want
the same machinery and should arrive together.

**`sterbenz_cancellation` needs a proven range.**
Catalogue section 5's fact slot (`SSAValue.accounting`). This is the
highest-value blocked identity: it collapses a whole expansion to one
ordinary subtraction, and it is what took `expm1` to 1.75 ulp with no core.
It must never fire on a guess — where it is wrong it is wrong silently and
by the entire residual.

**`Fma` exists in SSA; no backend implements it.**
Recorded here earlier as blocked on a missing primitive, which was wrong
twice over. It was never needed for LLVM (whose `contract` flag is attached
per instruction at `ssa_llvm_backend.py:459`, over exactly `{Add, Sub,
Mul}`), and it was not hard to add. `ir_identities.FMA` is now a real
operation -- `x * y + z` under exactly one rounding -- produced by
`contract_multiply_add_to_fma`, gated on the work contract's
`contract_multiply_add`. What remains is one spelling per backend: C's
`fma`, GLSL's and WGSL's `fma`, LLVM's `llvm.fma`.

Naming it beats relying on the flag: a flag PERMITS fusion, so whether it
happens depends on the toolchain, the named target and the optimizer, and
the six backends with no `-O2` behind them never see it. An instruction
either is an `Fma` or is not.

Still owed: the rewrite is exact at a precision dual (`a * b - fl(a * b)`
IS the residual) and inexact everywhere else, so it should fire
unconditionally there and on licence elsewhere. That needs the dual to be
marked, which nothing does yet -- so today it rides the general licence and
will not fire under `prove`.

**The superaccumulator kernel is unwritten.**
`exact_accumulation_over_long_chain` names it; nothing provides it.

**The specific-name pass is unwritten.**
`PRECISION_OPERATOR_NAMES` (105 names) has no consumer. It needs a pass that
runs after planning and computes, per surviving operation: the operating
width from the fused batch, and the return width from the actual consumer.

**Lane closures exist; nothing produces one for precision.**
SSA already has them, and more rigorously than a declaration would: `Deploy`
and `Join` are registered operators (Deploy fans DOWN to lanes, Join fans UP
from them), `deployment_ssa_binding.py` gives Deploy an `ssa.deploy_token`
result and Join the lane live-outs as operands, and before binding it PROVES
the independence the region record claims -- no value defined in one lane
consumed by a sibling -- reporting and refusing rather than silently
repairing. `ControlDeploymentRegion` is explicitly backend-neutral and grants
permission rather than obligation, with serial always valid, which is exactly
"up to this many lanes, architecture decides".

Only `loop_composer.py:967` builds one, so lanes come from loops and nowhere
else. The precision connection is real but narrow: limbs are NOT independent
in general -- two_sum's error propagates limb to limb, which is the mechanism
-- but the operations stamped `precision_form` are exactly the ones where
they are. Negation flips each limb's sign and power-of-two scaling shifts
each limb's exponent; neither reads a neighbour. Those two qualify for lane
regions over the limb channel, and the binding pass would prove it rather
than believe it.

## Correctness work not yet done

**The prove-vs-fast comparison has never been run.**
This is the acceptance test for the whole design, and `two_product` is now a
concrete subject for it: its dual is exactly the expression an optimizer
deletes (`ca - (ca - av)` is algebraically `av`). Compile it under `prove`
and under `fast`; bit-identical means the protection holds, any difference
at all means it does not. Expect it to FAIL today — see the next item.

**The primal's contraction flag is not withheld.**
`ssa_llvm_backend.py:459` decides `contract` per instruction from a global
contract boolean. A precision primal must not fuse with its own producers,
or Knuth's precondition breaks and the dual computes its residual against
the wrong operand. One added condition; not added.

**Nothing implements the precision operations.**
No destination has a spelling for `precision_mul` and friends. They are
carried intact to a backend that will report a shortfall. This was the
agreed sequence, but it means nothing end-to-end runs yet.

## Known-shallow, by choice

**The width rule is fixed-width, not exact.**
"Most limbs decides" gives `p2` for `p2 × p2`. Shewchuk's exact bounds are
`m + n` limbs for a sum and `2mn` for a product, so the exact answer is
`p8`. The current rule is the double-double preset — renormalise back to the
working width every step — which is a legitimate mode, not a bug. But the
bit-exact preset needs the other rule, and only one is implemented.

**`r` always equals `p`.**
Every generated name returns precision at its operating width. No collapsing
variant (`mul_f64_p2_r1`) exists. Left ungenerated until a real consumer
demands one, since the return width is the consumer's property.

## Dead code found along the way

**`graph_has_tensor_operation` is called by nothing.**
`src/transmogrifier/graph/node_special_cases.py:718`. Its docstring
describes the qualification test for kernel reduction and the
"no algebraic fusion" lowering path, and reads as authoritative. It is not
wired to anything. Either it lost its caller or it never had one — worth
resolving before someone else reasons from it as I did.
