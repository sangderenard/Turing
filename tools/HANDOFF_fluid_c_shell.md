# Handoff: fluid program through the canonical AOT/Fortran/C-shell chain

## Status: the C-shell chain BUILDS AND RUNS

```bash
python -m src.compiler.symbolic_fluid_direct_control --output build/sfdc
python tools/build_fluid_c_shell.py build/sfdc/control_repository_ssa.pkl build/fluid-c-shell
# -> built: build/fluid-c-shell/symbolic_fluid_frame_shell.exe
```

The whole `symbolic_fluid_frame` program -- dt controller included -- now goes
through `lower_ast_source_to_ssa` -> `emit_module` ->
`compile_fortran_module_c_shell` to a real gfortran-linked executable, with no
bespoke Python harness anywhere in the chain. This closes the directive the
previous handoff was written against.

`lower_ast_source_to_ssa` -> `_class_surface_ssa_program` is the ONE valid
path. Do not add a second whole-program lowering entry point.

## What was fixed to get here

Four root causes, each found by tracing real data (the emitted `.f90`, the
lowered SSA pickle, the generated LLVM IR) rather than by reading code and
guessing, and each verified by an actual gfortran compile. Error count went
9 -> 1 -> 0.

1. **Sequence column dtype width.** `_canonical_sequence_dtype`
   (`precompile_to_ssa.py`) widens every sub-int64 integer dtype to int64 at
   the sequence descriptor's sole construction point.

2. **Cross-shell structural schema disagreement.** Each shell (method) lowers
   with its own `_ControlSSABuilder`, which independently infers a shared
   sequence's shape from that one shell's local evidence. A sequence id is a
   GLOBAL identity (traces to one shared ProcessGraph node's `value_id`,
   `fortran_c_shell.py:1267`), so two shells could build structurally
   different descriptors for the same sequence -- showing up as a caller
   passing one SSA value to two formal positions the callee expects distinct.
   New `resolve_sequence_schemas` (`precompile_to_ssa.py`) surveys every
   shell's raw declarations BEFORE the lowering loop and hands every shell one
   resolved schema. Deliberately order-independent: no "whichever shell
   lowered first wins". A genuine cross-shell conflict is now a loud
   `SSALoweringShortfall`, not a silent pick.

3. **`length_address`/`capacity` hardcoded to int32 in two independent
   places** -- `_storage_values` (`ir_sequence_tables.py`, the helper's own
   formals) and `_sequence_descriptor` (`precompile_to_ssa.py`, the caller
   side). They agreed with each other but not with a caller-supplied
   int64 keyed-instance-field length. Both are int64 now. The
   `_keyed_helper_dtypes` table (`fortran_c_shell.py`) had the same int32
   entries for the length/capacity positions and was corrected to match.

4. **SSA id collision on aggregate call results.** `returns_aggregate` adopted
   `int(record.callsite_id)` (an AST node id) as its result value id without
   checking that id was free in the function's own numbering. It collided with
   a required-source-value already produced by an aggregate unpack, giving two
   different instructions one SSA identity. The later freshening pass cannot
   repair this class -- it renames a colliding `.res` in place but never
   rewrites other instructions referencing the old id by number -- so the fix
   is at the allocation site: allocate a fresh id when the callsite id is
   already produced.

Also fixed: `tests/test_symbolic_fluid_native_runtime.py` called
`load_symbolic_fluid_managed_functions(native)` when commit `5c774c7` changed
that parameter from a callable to a `build_directory`. It now passes
`tmp_path`. The test had been failing with a `TypeError` before reaching any
compiler code.

## Two more real defects found and fixed, in the native-runtime Python binder

These are NOT in the Fortran/LLVM emission at all -- they are in
`symbolic_fluid_native_runtime.py`, the hand-written Python shim that reads
compiled results back out by name. Both were found by direct execution-buffer
inspection (compile, run, print `.accounting` and actual returned values),
not by reading code and guessing.

5. **`state.last_wave_speed`/`last_height_violation`/`last_tracer_violation`
   never wrote back.** `symbolic_fluid_dt.py`'s source used a bare alias
   (`state.last_wave_speed = max_wave_speed`) for a loop-carried reduction.
   A bare alias produces no real instruction, so the compiler had nothing to
   route through `carried_port_values` the way the `Ret` path already does --
   the ABI argument stayed `intent(in)` and the runtime read back its own
   seed. The array fields three lines above already work around this with a
   `+ 0.0` forced-copy idiom (`state.height = state.next_height + 0.0`);
   applying the same idiom to the three scalar fields fixed them identically.
   Fixed in `symbolic_fluid_dt.py`.

6. **`Metrics(...)`'s return record was bound to field names by raw
   position** -- `zip(dataclasses.fields(Metrics), record[1:])` in
   `symbolic_fluid_native_runtime.py`'s `_bind`. The dataclass's declared
   field order has no relation to the actual return record's order (that
   order reflects which fields ended up literals vs defaults vs
   region-call results, an internal compilation detail, confirmed by direct
   inspection to be `[div_inf, mass_err, osc_flag, stiff_flag, sim_frame,
   proc_ms, dt_limit, hard_failure, max_vel, max_flux]` -- nothing like
   declaration order). `dt_limit`'s real value was landing under `max_vel`'s
   name; `div_inf`'s literal 0.0 was landing under `max_vel` too. This only
   ever "worked" for `max_vel` by coincidence of the offsets at the time --
   fixing defect 5 above changed the record's shape slightly and broke even
   that coincidence, which is what surfaced this.

   Replaced with name-based resolution using each component's own
   accounting instead: a defaulted field names itself directly
   (`program_abi_default`), a literal keyword names itself directly
   (`program_abi_constructor_literal`), and a computed field resolves via
   the producing region's own captured-operand names (a planner region
   keeps the caller's own identity for anything it CAPTURES rather than
   computes itself, so a reduction's non-self-computed operand -- e.g.
   `max(carried, wave_speed)`'s `wave_speed` -- still carries its outer
   name even though the reduction's own result does not). Also gave
   `dt_limit`'s previously-inline expression a local name
   (`dt_stable_limit`) so it resolves the same way as `mass_error`. A hard
   `RuntimeError` now fires if a required field can't be named, instead of
   silently mispairing.

Verified together: `max_vel`, `max_flux`, `dt_limit`, and all three
`state.last_*` fields now read correct values matching independently
computed ground truth (1.010970, 1.010970, 0.111279, matching a fresh
NumPy-computed max wave speed and CFL limit for the same state).

## What is NOT done: `mass_err` reads exactly `0.0`

`tests/test_symbolic_fluid_native_runtime.py::test_native_sympy_fluid_step_rejects_rolls_back_and_lands_on_frame`
now compiles and runs (~19-21s) and fails on one remaining physics assertion:
attempt 1 at `dt=0.2` is accepted where the reference expects it rejected,
because `metrics.mass_err` comes back `0.0` against a true `1.680060e-04`
(computed independently from grid state before/after in the test harness).
This is the ONLY remaining blocker for this test -- confirmed by reading
`step_with_dt_control_used` in `dt_controller.py`: the accept/reject decision
never consults `dt_limit`/CFL at all, only `mass_err > mass_max`, `div_inf`,
and named `error_channels`; `dt_limit` being correct now (see above) does not
matter to this test.

**Ruled out:** the name-binding bug above. `mass_err`'s id resolves correctly
by name now (confirmed: `by_kwarg["mass_err"]` points at the right SSA value,
same mechanism that fixed `max_vel`/`dt_limit`). The runtime genuinely reads
`0.0` out of that value's buffer.

**Traced so far, with exact LLVM IR references** (from
`compile_native_symbolic_fluid_advance(...).artifact.llvm_ir`):

- The repository SSA is structurally sound: region_6's call passes
  `next_mass`/`previous_mass` at the correctly-named positions
  (`value_names` confirms), region_6's own `Sub[171, 172]` subtracts them in
  the right order, and its `Div` correctly produces the id `_bind` now
  correctly labels `mass_err`.
- The LLVM control flow places the call correctly: `loop_exit:` (after the
  outer loop, not inside it) contains
  `call ... @...planned_region_6(..., ptr %phi.192, ptr %phi.193, ...)`.
- `%phi.192`/`%phi.193` are defined in `loop_header:` as
  `phi ptr [ %value.187/%value.188, %entry ], [ %phi.207/%phi.208, %loop_latch ]`
  -- i.e. LLVM expects `%phi.207`/`%phi.208` (the accumulated per-iteration
  results) to be available wherever control reaches `%loop_latch`.
- But `loop_latch:`'s own visible instructions only update the induction
  variable (`%value.195` feeding `%phi.194`) -- `%phi.207`/`%phi.208` are
  never mentioned there. They are defined instead inside the INNER loop nest
  (`%phi.207 = phi ptr [ %phi.192, %loop_body ], [ %value.116, %loop_latch.1 ]`,
  i.e. the inner loop's own accumulation, which must reach `%loop_latch`
  through the inner loop's own exit block by dominance, not by a literal
  reference inside `%loop_latch:` itself).

**Next step, not yet done:** trace the inner loop's own exit block (find
where `%loop_header.1`/`%loop_body.1`/`%loop_latch.1` branch back out to the
outer `%loop_latch`) and confirm whether `%phi.207`/`%phi.208` genuinely
dominate that branch with their fully-accumulated value, or whether the
inner loop exits too early / branches around them. This is a real LLVM CFG
question, not a naming or binding question -- don't re-check `_bind` or the
repository SSA again, both are confirmed correct for this value.

**Diagnostic trap already hit, worth avoiding:** probing `next_mass`/
`previous_mass` by adding a NEW `state.field = next_mass + 0.0`-style
expression to the source is unsafe as a diagnostic -- it was exactly this
kind of extra reference that, applied to `max_wave_speed` while debugging
defect 5, corrupted a DIFFERENT existing consumer of the same reduction
before the name-based `_bind` fix went in. Any new probe that adds a source
expression should be verified not to change surrounding values before
trusting its own readout.

Useful commands, all cheap:

```python
# per-instruction SSA dump of the advance function
import pickle
module, outputs, exports = pickle.load(open("build/sfdc/control_repository_ssa.pkl","rb"))
fn = module.functions["symbolic_fluid_control__symbolic_fluid_advance"]

# the generated LLVM IR as text, and the module_functions dict _bind uses
from src.compiler.symbolic_fluid_native_runtime import compile_native_symbolic_fluid_advance
adv = compile_native_symbolic_fluid_advance(build_directory)
adv.artifact.llvm_ir       # str
adv._module_functions       # dict[str, Function], one entry per planned_region_N
```

The `.f90` is also worth reading directly; isolating just a caller and its
callee into a small file makes gfortran report the real diagnostic instead of
a truncated one:

```bash
PATH="/c/msys64/mingw64/bin:$PATH" gfortran -fsyntax-only isolated.f90
```

(gfortran invoked by absolute path with its own `bin` off PATH fails silently
with no diagnostic -- `ssa_fortran_backend.py:4958` puts it on PATH for the
child, and you must do the same by hand.)
