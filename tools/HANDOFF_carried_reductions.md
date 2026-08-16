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
