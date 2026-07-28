# Mandelbrot ProcessGraph mistake record

Date: 2026-07-28

Status: the implementation described below was removed before this document
was committed. The tracked tree is back at the preceding committed state.

## What went wrong

Work began on multi-frame Mandelbrot recording before the existing
ProcessGraph-to-AbstractTensor execution path had been audited closely enough.
When the first attempted shape did not pass through the compiler, the work
started compensating inside the demo and then modifying general compiler code
in response to symptoms produced by those compensations.

That was backwards. The demo should expose general compiler limitations. It
must not invent a second execution model to get around them.

The removed work included:

- A duplicate copy of the complete one-frame solve and JPEG pipeline.
- A demo-specific `frame_parameters` protocol made from eight-element tuples.
- Manual `AbstractTensor.tensor(...)` calls inserted to compensate for values
  reaching a callable through the wrong execution path.
- An explicit Mandelbrot clamp argument inserted to avoid resolving the
  function's existing static default.
- A switch from ordinary recursive AST-parent ingestion to a
  Mandelbrot-specific closure collector.
- Changes to loop ownership and conditional execution made without first
  proving their behavior on small, backend-independent ProcessGraphs.
- A change that preferred recursively generated function shells over retained
  Python callables without first defining the intended callee-resolution
  contract.
- Ad hoc propagation of Python module globals into child shells.
- Changes to the animated demo's feed and return conventions that were shaped
  around the unfinished multi-frame attempt.

None of these changes remains in tracked source.

## The key diagnostic mistake

The final observed failure was:

```text
AttributeError: 'bool' object has no attribute 'logical_not'
```

This was initially encountered while forcing the full compression path through
the new loop. It is not evidence that the GLSL backend needs a special Boolean
hack. It shows that a scalar Python Boolean operation was admitted to a
numerical region whose operator table assumes AbstractTensor operands.

That distinction matters:

- Tensor logical negation belongs to AbstractTensor and therefore to the
  selected backend.
- Structural scalar `not`, including guards concerning shapes or ordinary
  Python objects, belongs to scalar evaluation, constant folding, or compiled
  control flow.
- The planner needs enough value-domain information to distinguish those
  cases. A demo-specific fallback would only hide that missing distinction.

## Process errors to avoid repeating

1. Do not change the demo's data contract to accommodate a compiler failure.
2. Do not duplicate an AbstractTensor algorithm inside a root function merely
   to make its AST lexical descendants easier to identify.
3. Do not manually wrap values in a concrete path when normal graph execution
   is supposed to select and preserve the backend.
4. Do not turn unresolved static values into literals at a call site.
5. Do not alter general control-flow scheduling without isolated ProcessGraph
   tests covering loops, branches, loop-carried values, and nested calls.
6. Do not call a Python implementation when a resolved ProcessGraph function
   is supposed to be the semantic implementation, unless the boundary is
   explicitly classified as external.
7. Do not claim multi-frame fusion when the implementation is a Python loop
   issuing one frame at a time.
8. Do not proceed from a failing demo stack trace directly to a backend patch.
   Trace the value and operator through reduction, planning, deep compilation,
   AbstractTensor dispatch, and backend dispatch first.

## What was learned

The repository already has the important architectural center:

- ProcessGraph retains graph structure and control information.
- `operator_defs.py` provides an AbstractTensor execution vocabulary.
- `GraphDeepCompiler` emits a transient Python callable from graph operators.
- Executing that callable under `AbstractTensor.use_backend(...)` gives the
  familiar Torch-like model: ordinary tensor instructions execute through the
  selected backend.
- Forward capture can record a numerical execution path and lower it to a
  backend-neutral fused program.

The immediate problem is therefore not "make the demo understand tensors."
It is to make planning preserve the distinction between tensor computation,
scalar computation, structural control, static data, and external effects
while continuing to route tensor computation through the existing
AbstractTensor vocabulary.
