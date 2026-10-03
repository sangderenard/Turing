# Session continuation report, 2026-10-03 (orbital demo push)

The session ended on a credit limit. Everything was committed and pushed as is,
including work from agents still in flight (the step-4 planner and item 4 may
be partial). Each lane has its own `CONTINUATION_*.md` in this folder.

## Landed (turing)

- **Compiled collocation Jacobian matches SymPy to 4.138e-16** (`095a3c0c`,
  `0519095d`, `ccba325c`, `51b4cebe`, `ce587b82`):
  - plan_callsites now plans adjoint calls that have no AST;
  - the eps marker is retired;
  - backward-rule argument_roles;
  - rank-0 formal shapes are published;
  - CALLSITE_RETURN_MEMBER posts;
  - reverse_compile_book, one book per reverse compile;
  - constants keep their declared integer width.

  Green: graph-reverse VJP, native_scalar_loss_adjoint 2/2, linear motion.
- **Orbital item 3** (`c03ae4e6`): undefined sympy functions compile as
  runtime-slot externals in C and LLVM. Orbital benchmark is at 8 passed /
  4 xfailed.
- **Uploaded patches:**
  - `854e145c`: scatter precision;
  - `408155a7`: return-site identity, ported onto our mechanisms.
- **llvm_dt_system** (this commit):
  - each state runs its own program, `state.program` (`bind_program`);
  - pieces are owned per state, so pointers are bound once;
  - outputs are written in place;
  - eager state lives in one contiguous span;
  - publication rows are slice writes.

  Parity is bit-identical over 7,428 field-rounds, and a station frame takes
  1.55 s → 1.0 s.

## Landed (engine_toy, root repo)

The craft is a machine (gimballed main / brake / RCS, mass properties from
the machine reduction). Further work landed in:
- the role-based allocator, at 5–15 ms per call;
- the mean-step kick;
- the BIND dt contract;
- propellant-mass pricing;
- the tracker: wrench seam, hysteresis re-plan, slew lead;
- the game (step 6);
- the collocation planner (step 4, WIP).

Machine transfer: 0.91 m / 0.034 m/s, biprop 0.997x ideal.

## Open, in priority order

1. **CONCORDANCE AUDIT (user, absolute): "nothing is valid in any way that
   doesn't go through the concordance leaving edges".** These decisions were
   committed with no edges:
   - ir_indexing int-width (`ce587b82`);
   - note_shape "polymorphic" flag (`51b4cebe`);
   - argument_roles as attributes (`0519095d`);
   - Unsourced adjoint rows for unscoped forward graphs, and sympy make_node
     operand edges not through _set_operands (`de609156`);
   - the cached backward-rule graph rebuilt without edges to the prior book
     (`51b4cebe`);
   - the port's Const/carried-Phi fallback (`408155a7`);
   - item 4's structural-constant role (in flight).

   Post each one as a row with its edge, and gate on
   tools/audit_identity_concordance.py.
2. **Native dt compile stalls** at the dt_system_over → run_superstep
   callsite. It is a deep recursion in glsl_deployment_strategy.py ending in
   a linear history scan in identity_concordance.py. It also reproduces
   without the spans change. Not yet determined whether it is slow or
   spinning (see CONTINUATION_llvm_dt_system_spans.md).
3. **Name-arm alias fix** (precompile_to_ssa.py, committed as-is here,
   UNVERIFIED): the IndexedStore identity must become a posted book cell with
   an edge from its source, per the rule above.
4. **Item 4:** Derivative of externals uses the declared derivative slot; the
   Integral needs get_tensor lowering. Fix the aa5f1aac floatified axis
   regression first.
5. **Step 4 planner:** converge, fly, and act as the tracker's re-planner (see
   CONTINUATION_orbital_step4_collocation.md).
6. **Game:** switch to the MachineCraft plus batched stations. It is unblocked
   now that per-state binding is fixed.
7. **Smaller items:**
   - the batch-4 propellant-supply Piecewise writes garbage into lane 0;
   - the per-participant publication fields: option 1 is a whole-field
     store, option 2 is a declared publication array (user's choice);
   - the inspect.signature cost per call in AbstractTensor;
   - the ArgumentBindingFact pickling bug.

## Rules in force

- A compile at about 30 min is a suspected loop: use faulthandler dumps,
  compare them, then kill and trace.
- Never hand-edit compiler output.
- One build at a time.
- Use opus agents.
