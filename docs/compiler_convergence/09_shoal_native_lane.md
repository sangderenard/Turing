# Task 9 — Shoal native publication and OOB investigation

## Observed code state

- `symbolic_fluid_model.py` publishes equation metrics and state fields, while
  `load_symbolic_fluid_managed_functions()` returns `dt_next` from the Python
  `run_superstep` controller around a compiled native advance function.
- `tests/test_symbolic_fluid_native_runtime.py` currently passes a 4x4 managed
  controller/native-advance path and asserts a valid `dt_next`.
- `native_voxel_fluid.py` demonstrates a separate whole-frame C-shell route
  with explicit `dt_out = dt_next + 0.0` and feedback to `dt_initial`.
- No current source test or marker reproduces the reported out-of-bounds int64
  read.  It must be treated as an unverified symptom until captured.

## Work sequence

1. Identify the exact Shoal executable/command that exhibits the read and run
   it in a contained subprocess with a small grid and trace enabled.
2. Record the failing value's deterministic token chain, formal dtype/shape,
   caller binding, allocation extent, and native trace site.
3. Add a focused failing test at the earliest reproducible layer (SSA binding,
   LLVM emission, C shell, or publication/feedback).
4. Repair that layer and separately decide whether `dt_next` belongs inside the
   compiled frame or remains a managed-controller publication.  Do not conflate
   a scalar feedback omission with an int64 addressing defect.

## Acceptance

- The OOB symptom has a deterministic reproducer and disappears under native
  instrumentation.
- State fields, metrics, and `dt_next` publish through declared output/feedback
  records with verified dtype and extent.
- The existing rollback/mass-conservation test stays green.
