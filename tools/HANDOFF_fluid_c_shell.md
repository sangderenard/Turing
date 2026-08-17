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

## What is NOT done: the LLVM lane's loop-carried reductions

`tests/test_symbolic_fluid_native_runtime.py::test_native_sympy_fluid_step_rejects_rolls_back_and_lands_on_frame`
now compiles and runs (~21 s) and fails on a PHYSICS assertion, not a build
error: attempt 1 at `dt=0.2` is accepted where the reference expects it
rejected, because `metrics.mass_err` comes back `0.0`.

This is the pre-existing defect commit `5c774c7` documented ("the loop-carried
scalar reductions ... never update across the traversal"). It is NOT a
regression from the work above.

Measured, not assumed:

| quantity | native | truth |
|---|---|---|
| `metrics.mass_err` | `0.0` | `1.680060e-04` (computed from grid state before/after) |
| `metrics.max_vel` | `1.0109701635896629` | `1.0042038251280456` (`max sqrt(g*h)`) -- **correct** |
| `state.last_wave_speed` | `0.0` | should equal `max_vel` |
| `state.last_height_violation` | `0.0` | (same write-back path) |

Note `g ~= 1.0` in this program, not 9.81 -- so `max_vel` being ~1.01 is
CORRECT, and the max reductions genuinely work. An earlier reading of this
table as "max_vel is wrong" was based on assuming g=9.81; measure before
concluding.

### The SSA is correct -- verified, not assumed

Do not go looking for a missing reduction in the IR. All five are present and
correctly wired in `build/*/control_repository_ssa.pkl`:

- `planned_region_2`: `47 = Add[18, 46]` -- `previous_mass += height[r,c]`
- `planned_region_3`: `116 = Add[19, 97]` (`next_mass`), `117/118/119 = Max`
  (`max_wave_speed`, `max_height_violation`, `max_tracer_violation`)
- inner phis `198..202` carry each one, outer phis `183..187` carry them across
  the outer loop, `carried_port_values` in function metadata maps port ->
  carried phi
- every region-call out-param slot index matches the declared output order
  (checked exhaustively; all OK)
- `planned_region_6` computes `136 = Sub[165, 166]` (`next_mass -
  previous_mass`), `137 = Abs`, `141 = Div` -- and the call passes `%phi.186`
  and `%phi.187`, two distinct pointers

### Where the defect actually lives

In the LLVM emission (`ssa_llvm_backend.py`), not the IR. Carried scalars are
emitted as `phi ptr` -- pointers to fixed allocas -- e.g.

```llvm
%phi.202 = phi ptr [ %phi.187, %loop_body ], [ %value.47, %loop_latch.1 ]
```

Region calls use an out-param convention (`region_2` takes 8 in-params + 7
out-params) and the in/out pairing is correct at the call site.

The sharpest lead: **the same reduction publishes correctly through the `Ret`
path but lands as `0.0` through the `state.last_* = ...` store path.**
`max_vel` (Ret) is right while `state.last_wave_speed` (store) is zero, from
one identical `max_wave_speed` value. Start there -- it isolates the failure to
the write-back/publication of carried values rather than to the reductions
themselves, and it likely explains `mass_err` too (`Sub` of two accumulators
yielding exactly `0.0` is the signature of both operands reading a slot that
was never written back, not of two nearly-equal sums).

Useful probes, all cheap:

```python
# per-instruction SSA dump of the advance function
import pickle
module, outputs, exports = pickle.load(open("build/sfdc/control_repository_ssa.pkl","rb"))
fn = module.functions["symbolic_fluid_control__symbolic_fluid_advance"]

# the generated LLVM IR as text
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
art = emit_ssa_function_to_llvm(module, "symbolic_fluid_control__symbolic_fluid_advance")
art.llvm_ir  # str
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
