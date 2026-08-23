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

**Parameter annotations are invisible.**
`def f(a: Precision[2])` is not captured. `_TypeAnnotator`
(`src/transmogrifier/graph/node_special_cases.py:455`) has `visit_AnnAssign`
and `visit_ClassDef` and nothing for `ast.arg`, so only annotated
*assignments* in a body are recorded. Its own docstring claims it is "where
an `int` parameter's integer-ness comes from", which is not true today. Any
program declaring precision on a parameter silently compiles as ordinary
arithmetic. This is the most likely thing to bite a real caller.

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

**The fused `two_product` needs a flag, not an `fma` primitive.**
Recorded here earlier as blocked on a missing primitive. That was wrong.
Two routes already exist and neither needs SSA to spell `fma`:

* `ssa_llvm_backend.py:459` attaches LLVM's `contract` flag PER INSTRUCTION,
  and `_CONTRACT_ELIGIBLE` is `{Add, Sub, Mul}` -- exactly the operations
  the dual uses. SSA presents the lane structure (`Mul` then `Sub`) and
  LLVM forms the fma itself.
* `_BUILTIN_TARGETS` in `backend_intrinsics.py` is keyed
  `(backend, semantic_family)` and holds `blas.gemm` today. A
  `precision.two_product` family plus a target per backend is the same
  node-swap protocol, and GLSL/WGSL/C each have a native `fma()` to name.

The real work is that ONE flag serves both directions: withheld at the
primal (where contraction breaks Knuth's precondition) and granted at the
dual (where contraction IS the identity). `contraction_across_the_primal`
in the refusal table and `two_product_kernel` in the identity table are the
same mechanism seen from two sides.

**The superaccumulator kernel is unwritten.**
`exact_accumulation_over_long_chain` names it; nothing provides it.

**The specific-name pass is unwritten.**
`PRECISION_OPERATOR_NAMES` (105 names) has no consumer. It needs a pass that
runs after planning and computes, per surviving operation: the operating
width from the fused batch, and the return width from the actual consumer.

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
