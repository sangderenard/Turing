# The carried-reduction seam in compiled symbolic_fluid_advance

The compiled advance (5c774c7's loader) runs the traversal at 1.24 us/cell
(13x over the removed exec path), conserves grid mass to 5e-6, and moves the
wavefront at physical speed.  Its five loop-carried scalar reductions do not
compute, and the anatomy is now exact -- measured, not inferred.

## Evidence, all against build/sfdc-defaults

1. **Discovery is complete.**  Both loops (37 column, 38 row) report all five
   carried bindings:
   `(max_height_violation, 21, 118) (max_tracer_violation, 22, 119)
    (max_wave_speed, 20, 117) (next_mass, 19, 116) (previous_mass, 18, 47)`.

2. **The updates are computed.**  All five updated values are produced in
   `loop_body.1` as region_3 aggregate loads: 47, 116, 117, 118, 119.

3. **Three bindings have no Phi and zero consumers.**  117/118/119 are dead
   loads.  Only the masses formed carried Phis (%178: 19->116, %179: 18->47)
   at both loop levels.

4. **No continuation reads any carried result.**  Post-loop:
   - `region_6` (mass numerator) feeds (165, 166) -- the SEED identities of
     next/previous mass (`value_names`: next_mass=165, previous_mass=166);
   - `region_7` (mass_error) feeds (137, 140) -- bare anonymous arguments;
   - `Ret` returns %164 (state.last_wave_speed) which is an argument no
     instruction ever stores to, and %151 (dt_limit) computed in loop_exit
     from the unwritten wave speed.

   So even the two wired Phis have unread exits.  At runtime: max_vel 0,
   dt_limit inf, mass_err NaN, while the grid state itself is correct.

## The defect in one sentence

Carried bindings are discovered and their updates computed, but the
continuation identity -- "after the loop, this NAME means the carried
result" -- is severed for all five: the planner's regions captured the
pre-loop identities as feeds, and rewire_continuation (which replaces the
UPDATED id in consumers) never touches consumers that hold the INITIAL or
seed identity.

## Where to look

- `loop_composer.py` carried-port materialization: `rewire_continuation`
  replaces `updated -> LoopResult` only.  Consumers holding the seed id
  (mass_error's authored parents resolved to 165/166) are untouched.  The
  identity_table's post-loop entry for these names must become the port, and
  the planner's feed capture must see it -- check the ordering of
  materialize_retained_loop_ports against region feed capture in
  `strategize_shell_deployment` (glsl_deployment_strategy.py:17313 region).
- Why 117/118/119 formed no Phi while 116/47 did: the carried lowering in
  precompile_to_ssa (carried_phis / "carried update value has no producer"
  shortfall at ~3162) -- the max updates' identities may alias through the
  Max fold differently than the Add updates.
- Related known-failing test (same territory):
  `test_ir_sequence_tables::test_compiled_retained_loop_mutates_caller_sequence_record`.

## Reproduce

```bash
python -m src.compiler.symbolic_fluid_direct_control --worker --output build/x
python - <<'PY'
import pickle, sys; sys.path.insert(0, ".")
from src.compiler.symbolic_fluid_native_runtime import compile_native_symbolic_fluid_advance
from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState
advance = compile_native_symbolic_fluid_advance("build/x")
ok, metrics = advance(SymbolicFluidGridState.initial(8, 8), 1e-3)
print(metrics.mass_err, metrics.max_vel, metrics.dt_limit)  # nan 0.0 inf
PY
```

## Acceptance, with the precision boundary stated

The coriolis path runs through the LUT trig solver (error bound ~8.3e-10),
so hold the compiled reductions to POLYNOMIAL-ERROR scale, not machine
epsilon: mass_err small at ~1e-9 scale, max_vel near sqrt(g*h) to the same
tolerance.  If the boundary itself becomes the blocker, the LUTs can be
baked to double or the `continuous` (convergence) trig mode selected --
both lanes already exist.

When those numbers land, the dt controller's rejection logic runs on real
values and `examples/symbolic_fluid_live.py` runs compiled end to end.


## Narrowed further (session 2026-08-16, commit after 9dea735)

Probed with the relaxed filter live:

- 47/116 (the mass Adds): `_is_dispatch_metadata_node` False, executable,
  dispatched to regions 2/3.  Their whole chain now works end to end.
- 117/118/119 (the max updates): they are builtin ``Call`` nodes
  (type=Call, static_python_reference='max'), so the dispatch classifier
  calls them coordinator metadata -- never executable, never in a dispatch
  region.  In earlier builds the HIERARCHY planner still emitted their
  PlanLines inside planned_region_3 (outputs 116,117,118,119); with the
  aliases admitted, the seeds become carried boundaries and the hierarchy
  planner drops the max lines too -- no region produces OR feeds 117-119,
  and the honest loop_carried shortfall fails the build.

THE FIX LIVES IN REGION MEMBERSHIP for intrinsic builtin calls whose
consumers are carried continuations: ``max(carried, x)`` is numeric work
exactly like ``carried + x`` (which classifies executable as an Add).  Two
candidate shapes: teach the dispatch classifier that an intrinsic
extraction (python_identity_program with a pure operator step, as float
already is) over scalar operands is numeric work, not coordination; or
keep the hierarchy planner's membership stable under carried boundaries.
The first is the honest one -- it removes the asymmetry between ``+`` and
``max`` instead of patching around it.


## Attempted classifier whitelist (reverted; keep the lesson)

Whitelisting ``Call + extraction_action=intrinsic + tensor in
abstract_tensor_funcs + all operands is_scalar_value`` did NOT admit the
maxes: ``is_scalar_value`` is syntactic (consts, scalar Inputs, scalar
expressions) and the max's operands are a carried current value and a
region output -- neither satisfies it.  The scalarity evidence lives in
region value META (rank-0 dtype float64), not syntax.  Next attempt:
gate on the operands' region_value_meta / planner domain rank instead of
is_scalar_value, or admit intrinsic extractions unconditionally when the
consumer is a carried continuation (the port set names them exactly).
Both build-failure modes are honest now -- the loop_carried shortfall
names the value the moment membership drops it.


## Minimal executable repro decodes the last wrongness (session 3)

8-line nested sum+max (tools: lower nested2__root, count=4) compiles with
zero shortfalls and returns total=24, peak=3 (expect 120, 15).  Decode:
24 = sum(row*4) with col==0; 3 = max(col) with row==0.  The two carried
chains live in different regions, each holding its own copy of
float(row*4+col); each copy binds ONE loop's induction correctly and reads
the OTHER loop's induction from its seed cell.  Cross-loop induction feeds
into inner-body regions are the remaining defect -- everything else in the
carried chain (phis, ports, order, linked calls, phi forward refs) is now
verified correct by this same program.  Fix where induction values bind to
region feeds (precompile loop lowering, external_values[target]=induction
around the projected-binding block); the outer induction consumed by an
inner-body region must resolve to the OUTER phi value, not the start cell.


## Session 4: every ingredient proven; one diff remains

Executable proofs, all exact:
- nested sum+max: (120, 15)
- linked callee with selective outputs (root->helper): 8.0
- linked callee inside nested carried loop: (4,1)/(18,2)
- THE REAL sympy stencil, linked, in a nested carried max loop:
  peak = 3.132091952673165 = sqrt(9.81) EXACT (build/miniwave-native)
- the stencil standalone from the fluid closure: wave_speed exact

Only the full advance still zeros its wave/violation chain.  The remaining
delta between miniwave (works) and advance (zeros) is span-fed stencil
inputs (state.height[row,col] via multi-axis GEP/extents) and in/out
record fields.  Both emit calls to the SAME callee: diff the two emitted
call sites -- the `call void @__ssa_...symbolic_fluid_step(...)` operand
list and each one's result_ptrs/aggregate members -- in
build/miniwave-native vs the advance emission.  The mismatch in that diff
IS the defect.
