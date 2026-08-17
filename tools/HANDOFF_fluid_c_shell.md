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

## RESOLVED: `mass_err = 0.0` was CORRECT. The real defect is the array copy.

Found with the value watch (`emit_ssa_function_to_llvm(..., watch=..., history=N)`),
which made the previously-unobservable accumulators readable per iteration.

**The measurement that settled it.** Watching the two carried accumulators
with `history=24` gave their per-iteration series:

```
next_mass: 0.0, 0.997309, 1.993802, 2.997286, 3.999977, ... 16.017591
prev_mass: 0.0, 1.0,      2.0,      3.0,      4.0,      ... 16.017591
```

The summands genuinely differ (`0.997309` vs `1.0`), so the two
accumulators are NOT aliased and never were -- and they converge to the
same total because **the scheme conserves mass**. `next_mass ==
prev_mass` is the physically correct result, so `mass_err = 0.0` is the
correct answer and was never the bug.

(The 20 samples for a 16-cell grid are 4 outer iterations x (4 inner + 1
exit-check re-entry); the repeated values in the series are the exit
checks, not lost work.)

**What is actually broken.** `state.next_height` is computed correctly in
all 16 cells (`sum = 16.017591076142672`, matching the old sum to 4 ulp --
mass conserved). But after `advance` returns:

```
cells where state.height != state.next_height: 15 of 16
```

Only cell `[0,0]` is updated. So `state.height = state.next_height + 0.0`
-- the whole-array copy -- copies **one element instead of the array**.

**Root cause, confirmed.** That copy is `planned_region_4`, and every one
of its values is shaped `()`:

```
region_4 formals: [(120, 'float64', ()), (124, ...), ...]   # all scalar
  122 = Add[120, 121]   res_shape=()
  126 = Add[124, 125]   res_shape=()
  130 = Add[128, 129]   res_shape=()
  134 = Add[132, 133]   res_shape=()
```

The emitter's copy path branches on `_value_element_count(output)`: count
1 emits a scalar load/store, otherwise a `memcpy`. With `()` shape the
count is 1, so a 16-element array copy is emitted as a single scalar copy.
The arrays lost their shape somewhere before region_4 was built -- that is
the next thing to find, and it is an *extent/shape propagation* question,
not an aliasing or identity one.

**Where to look for the shape loss (owner's steer, recorded before it is
lost).** Do not assume the loss happens near region_4. It may originate far
upstream, at the AST/SymPy process-graph *inlet* -- in the special-cases
and schema-replacement machinery that conforms nodes on the way in
(`node_special_cases`, `python_special_cases`, `interpret_special_case`,
and the tensor-intrinsic recognition around them). A tensor intrinsic
misrecognised there would produce exactly this: a whole-array operation
that arrives downstream already shaped as a scalar, with every later stage
faithfully preserving that mistake. Start there before instrumenting
region_4 itself.

This also explains the failing test without any appeal to `mass_err`: the
grid never actually advances (15/16 cells keep their old values), so the
controller sees a nearly-static field, nothing violates a bound, and
`dt=0.2` is accepted where the reference rejects it.

**Correcting the record:** the "truth" value of `1.680060e-04` used
throughout this handoff as the expected `mass_err` was itself derived from
`state.height` *after* the call -- i.e. from the corrupted copy. It was
never the right target. Two separate wrong conclusions came from trusting
it; see the field notes in `tools/TRANSLATION_DEBUGGING.md`.

## Superseded: the old `mass_err` investigation

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

**Every hypothesis testable via static IR reading has now been checked and
ruled out.** In order:

1. **Binding.** Confirmed correct independently of everything else --
   `by_kwarg["mass_err"]` resolves to id 141 by name, and reading it via
   `adv._read(141)` after a normal (unmodified) run genuinely returns
   `0.0`. This is not a `_read`/lookup artifact: `adv._read(147)`
   (`dt_limit`, resolved by the exact same mechanism) correctly returns
   `0.111279` in the same call.

   **CORRECTION -- an earlier version of this handoff claimed
   `adv._read(136)` showed the raw `Sub` result was itself `0.0`, and
   treated that as proof the zero originates at the subtraction. THAT
   CLAIM WAS WRONG and must not be relied on.** Ids 136/137/140/116/47/
   187/188 are NOT in `artifact.buffer_order` -- they are internal
   allocas with no public buffer. `_read` used to return its `0.0`
   fallback for any such id, which is indistinguishable from a real
   measured zero. Every "internal accumulator reads 0.0" statement
   derived that way was an artifact of the probe, not an observation.
   `_read` now takes `required=True` to raise instead, and
   `adv.observable(id)` reports whether an id is readable at all --
   check it before treating any read as evidence.

2. **Aliasing.** Every alloca involved is textually distinct and confirmed
   via `grep` on the LLVM IR text: `%value.47`/`%value.116` (the two
   accumulators' own output slots) and `%value.187`/`%value.188` (their
   seeds) are four separate `alloca double` instructions, each appearing
   exactly once, hoisted to function entry (not re-allocated per
   iteration). They cannot be the same memory.

3. **CFG placement.** `%phi.192`/`%phi.193` are defined in `loop_header:`
   as `phi ptr [ %value.187/%value.188, %entry ], [ %phi.207/%phi.208,
   %loop_latch ]`. The inner loop's own exit block, `loop_exit.1:`,
   unconditionally branches straight to `%loop_latch`
   (`loop_exit.1: br label %loop_latch`, nothing else in that block), and
   `%phi.207`/`%phi.208` are defined in `loop_header.1:` -- the inner
   loop's header, which dominates every path to `loop_exit.1` including
   the one that exits the inner loop. This is valid, ordinary SSA
   dominance for a nested accumulator; LLVM would have rejected the IR at
   verification if it were not.

4. **Carried-port grouping in the repository SSA** (before LLVM, in
   `precompile_to_ssa.py`'s own loop lowering). Dumped every
   `binding: loop_carried` Phi in the function: `next_mass`'s pair is
   `(initial=19, updated=116)`, `previous_mass`'s is
   `(initial=18, updated=47)` -- genuinely distinct groups, no accidental
   key collision between the two reductions.

5. **The analogous, working case.** `max_wave_speed` -- computed and
   carried the same way, also consumed after the outer loop -- reads
   correctly (see the fixed defects above). The one structural difference:
   `max_wave_speed` is computed entirely inside ONE region call
   (`planned_region_3`'s own `Max` instruction). `next_mass` and
   `previous_mass` are each accumulated by a DIFFERENT region call
   (`planned_region_3` and `planned_region_2` respectively) within the
   SAME shared inner loop body. This is the one remaining structural
   difference between the working and broken cases, and is the most
   promising lead for whoever continues this -- but it is a lead, not a
   confirmed cause; nothing above proves that a two-separate-regions
   pattern is where the defect lives, only that it's the one thing left
   unruled-out.

6. **Also checked and NOT the cause** (recorded so nobody re-walks these):
   - `deduplicate_node` in `graph_express2.py` merges graph nodes by
     `label`+`type` alone, with no scope/version discrimination. Real
     sharpness, but it is explicitly guarded to skip `ast.AST` nodes
     (`ensure_node`, "label-based deduplication must be reserved for
     non-AST structural objects"), so the authored Python accumulators
     are not subject to it.
   - Nothing overwrites `mass_err`'s output slot: the advance function
     contains exactly ONE write to `%out.2` (the region_6 call itself).
   - `planned_region_6`'s own emitted body is correct: `fsub arg.4,
     arg.5` -> `fabs` -> `fdiv`, stored to its `%out.0`, and the caller
     maps that to the right public slot. If its inputs were right, its
     output would be right.
   - 14 ids (every `symbolic_fluid_step` output: `height_next`,
     `wave_speed`, ...) are BOTH a formal arg and an instruction result.
     This looks alarming and is NOT a bug: `arg97 is load97` -- they are
     the SAME SSAValue object, so the argument cell intentionally doubles
     as that value's storage, and the freshening pass skips it correctly
     by its own `canonical[old_id] is result` rule.

**What this means for continuing:** the defect is not in naming/binding,
not in memory aliasing, not in any SSA/CFG structure readable from the IR
text, and not in region_6 itself. The remaining measurement that DOES
carry signal, and the best thread to pull: an earlier probe that
temporarily wrote `next_mass` and `previous_mass` into
`state.last_height_violation`/`state.last_tracer_violation` (ids 157/160,
which ARE public buffers, so that probe was observable and valid) showed
BOTH reading `16.0175910761` -- exactly the sum of the OLD heights.
`previous_mass` is correct at that value; `next_mass` should have been
`16.0149000240`. If that probe is trustworthy, `next_mass` is accumulating
the old height rather than `height_next`, and the question becomes what
region_3's `Add[19, 97]` actually receives for feed 97 at runtime. The
caveat that kept this from being conclusive: changing the source to add
the probe shifts ids, so the probe's own binding needs re-verifying before
its readout is trusted -- do that first (confirm via
`adv.observable(...)` + the correlate tool that the probe ids still mean
what you think) rather than assuming either way.

`tools/correlate_compile.py` exists for exactly this: one fresh compile,
then per-id, side by side, the repository SSA (advance + every region),
the LLVM IR lines, and the runtime value with an explicit NOT OBSERVABLE
marker instead of a misleading zero.

**Diagnostic trap already hit, worth avoiding:** probing `next_mass`/
`previous_mass` by adding a NEW `state.field = next_mass + 0.0`-style
expression to the source is unsafe as a diagnostic -- it was exactly this
kind of extra reference that, applied to `max_wave_speed` while debugging
defect 5, corrupted a DIFFERENT existing consumer of the same reduction
before the name-based `_bind` fix went in. An early probe of this exact
shape (`state.last_height_violation = next_mass + 0.0`, `state.last_tracer_violation
= previous_mass + 0.0`) showed both reading identically -- that result was
NOT trusted as evidence for this reason, and correctly so: reading `Sub`'s
own operands directly via `adv._read()` (no source change at all, see point
1 above) is what actually confirmed the zero originates at the
subtraction. Prefer `adv._read(internal_id)` after a normal run over adding
any new source-level expression.

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
