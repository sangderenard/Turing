# Decision: remove the scalar-expression deferral (2026-09-30)

Status: decided by the user in conversation, 2026-09-30, implemented and
verified as far as the "Verified" section below states. Read "Not verified"
before relying on it.

## What was removed

In `src/compiler/glsl_deployment_strategy.py`, the dispatch-metadata
classifier (`_is_dispatch_metadata_node_impl`) contained a rule named
`static_scalar_expression`. A BinOp, UnaryOp or Compare whose operands were all
scalars was classified as dispatch metadata: coordinator bookkeeping, left out
of every region, expected to be evaluated in place by whichever consumer read
it. A scalar operand is a constant, an accessor, or an Input marked
`value_kind == "scalar"`.

The rule arrived on 2026-07-28 in commit 87a867ea ("Enforce complete demo
execution inside AST root"). No comment or message stated its purpose beyond
sitting in a list of Python-syntax constructs kept out of numerical regions.
Its likely purpose, inferred and not documented: avoid a shader dispatch for a
lone `i + 1`. That cost exists only on a shader backend.

## Why it was wrong

The reducer marks a parameter scalar when it is annotated
`bool/bytes/complex/float/int/str` or has a scalar default (`dt=0.5`). For
such a parameter the rule deferred every all-scalar expression to a later
phase, `_graph_control_expression` in `src/compiler/fortran_c_shell.py`, which
rebuilds an expression tree from the node's graph parents at the point a
consumer needs it. Only some consumers call it (branch predicates, scalar field
writes). Its own docstring warns that re-deriving a value from its parents
gives one source value a second physical definition and, for a loop-carried
update, a stale one; region-owned values are consumed through the region's
publication instead. A value whose consumer never calls that phase is
computed by nobody. Two shapes broke:

- A returned scalar expression: `def f(k: int): return k + 1`.
- A scalar expression feeding a tensor operation:
  `def f(x, dt: float): return x * (dt * 0.5)`.

The planner made no region, the control program was empty, the result had no
producer, the parameter stayed in the ABI unused and unnamed, and the
full-native gate rejected the function with an unaccounted formal. The same
functions without the annotation compiled. They are the same program.

It was also a concordance defect. The decision to defer lived only in a
private per-graph cache of the classifier. No row on the identity book said
"this value is evaluated by its consumers", so a consumer that could not
evaluate it had no way to learn it owed the computation. A morph with no
edge, which is the shape of every fault in this repository.

## The chain, as observed (2026-09-30)

1. The reducer marks the Input `value_kind: scalar` from the annotation.
2. The classifier reports the Add as dispatch metadata (plain form: not).
3. The planner's executable-node set is empty, so no dispatch subgraph and no
   region exists.
4. The hierarchy plan has zero region items and the control program has no
   blocks; the function's result has a name history but no producer.
5. The parameter is declared-only and unnamed; the formal-parity check rejects
   it.

## Decision

A scalar expression is ordinary numerical work. It resolves at compile time
when its operands are known (constant folding is another pass) and otherwise
is computed once, by the region that owns it, as soon as its operands are
available, in the planner's dependency order. Every consumer reads that one
published value. Nothing is re-derived per consumer and nothing is deferred to
the latest use. The rule and its helper were deleted, and a breadcrumb comment
stands where they were.

If a scalar op turns out to cost a needless dispatch on a GPU backend, fix
that in the deployment profile of that backend, with the producer recorded on
the identity book. Do not restore a deferral that no row records.

## Verified

All on 2026-09-30.

- Compile-only matrix, rule on versus off: 14 programs (scalar-only chains,
  shared value used twice, conditional, loop bound, scalar into a tensor op,
  scalar default). Off: all compile and each annotated form matches its plain
  twin in rows and operations. On: 7 annotated or defaulted forms rejected.
- `tools/compiler_probes/probe_annotated_scalar_parameter.py` passes.
- `tools/compiler_probes/probe_scalar_native_correctness.py`: 14 programs
  emitted as C, compiled, run, and compared with CPython on the same source.
  All values equal, annotated and plain.
- `tools/audit_identity_concordance.py`: identical rows and findings on all
  six of its cases with the rule on and off.

## Not verified

- The native dt-system lowering over the drift piece and the whole-program
  Woodshop compile. Both use scalar annotations and defaults widely; whether
  more regions appear, or anything shifts, is unmeasured until they run.
- Per-consumer recomputation was not inspected instruction by instruction.
  The shared-value program compiled with the same operation count as its
  plain twin, which is consistent with computing it once.
- GPU backends. No shader path was exercised.

## Related findings, not changed by this decision

- The audit tool reports seven `operand-never-written` findings in its
  `toplevel`, `controller` and `controller_untyped` cases, identical with the
  rule on or off. Same symptom as this defect, so likely another unrecorded
  deferral. Untraced.
- A loop returning an integer accumulator produces a float64 result natively
  where CPython returns an int. The plain twin does the same, so it predates
  this change.
- While validating, the dt-system contract was found to declare the new
  `rollback` and `rollback_threshold_multiplier` record fields with a
  `python_type` key that record fields do not accept, from commit bc34adba,
  which had not been validated by a compile. Fixed alongside this change.

## Where to look

- Breadcrumb comment: `src/compiler/glsl_deployment_strategy.py`, in the
  dispatch-metadata classifier where the rule was.
- Repro probes: `tools/compiler_probes/probe_annotated_scalar_parameter.py`,
  `tools/compiler_probes/probe_scalar_native_correctness.py`.
- History: commit 87a867ea introduced the rule; the commit that removes it
  carries this document.
