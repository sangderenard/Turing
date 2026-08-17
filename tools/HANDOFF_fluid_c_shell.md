# Handoff: fluid program through the canonical AOT/Fortran/C-shell chain

## The directive this satisfies

The dt-managed fluid simulation must run as ONE compiled program, entered
through the AOT compiler, with no bespoke Python runner around the dt
controller. The canonical chain is:

```
lower_ast_source_to_ssa (fortran_c_shell.py)   <- the ONLY sanctioned
                                                   whole-source compiler
      -> repository SSA (module, outputs, exports)
emit_module (ssa_fortran_backend.py)           -> Fortran module + api
compile_fortran_module_c_shell (fortran_c_shell.py)
      -> basic C-shell executable, no custom runner
```

`symbolic_fluid_direct_control.py` already invokes `lower_ast_source_to_ssa`
against `SYMBOLIC_FLUID_DT_SOURCE` (the frame program with the managed dt
controller inlined) using the extraction contract at
`extraction_contracts/program_extraction.yaml`. That part was already
correct. The gap was downstream: the Fortran backend could not emit the
whole program, so nothing downstream of it existed. That gap is now closed.

## What was fixed (all in `src/compiler/ssa_fortran_backend.py`)

1. **`Log`/`Exp` intrinsics were unregistered** (only lowercase `log`/`exp`
   existed). One-line addition to the intrinsic table.

2. **The core defect: region-call aggregate binding assumed a caller
   consumes every declared output, in the declared native slot order.**
   `run_superstep`'s frame call only projects 3 of 13 declared outputs. The
   old `_region_call` matched `len(ordered_outputs)` against
   `callee_output_count` and refused to emit anything else — a hard
   shortfall for any partial consumer. Fixed: the callee's FULL declared
   record is always bound at the call. Consumed positions bind the caller's
   projection value; unconsumed scalar positions bind a per-callsite
   discard cell (`discard_c{callsite}_p{slot}`, declared locally, never
   read again). An unconsumed ARRAY slot is not discard-able yet and stays
   a loud shortfall (none occurred in this program).

3. **`aggregate_index` indexes the RAW return record (position space,
   repeats intact), not the canonical native slot list** (which is that
   record deduplicated by SSA id). `symbolic_fluid_advance` repeats id 704
   at positions 1 and 2 of its record. The old code treated
   `aggregate_index` as a direct slot index, silently misaligning every
   position after a repeat. Fixed in two places that both had this bug:
   - `_region_call` (the emission-time call-argument binder): now
     translates position -> slot through
     `callee_output_records` (new: the pre-dedup record, threaded through
     `emit_function`/`emit_module` alongside the existing deduped
     `callee_outputs`).
   - The cross-function dtype-propagation loop inside `emit_module`
     (~line 4341): had the *identical* bug — same fix, same translation.

4. **Cross-function scalar dtype resolution didn't reach call operands.**
   `_coerce_call_operand` was comparing against the callee's *raw* SSA
   formal (often untyped), not the type the callee's own emission actually
   resolves and declares. Added `argument_dtypes`/`output_dtypes` to
   `FortranSubroutine`, threaded them through the existing extent
   fixed-point loop in `emit_module` (now iterates to a joint fixed point
   over extents AND resolved dtypes), and added `_resolved_formal()` so
   both `_region_call` and `_ssa_call` coerce against the callee's
   resolved type. Where a projection's own resolved type differs from the
   callee's declared output slot type (e.g. a `bool` scalar the caller
   treats as `float64`), a typed bridge cell plus a postlude conversion
   line binds the call correctly.

Result: `emit_module` on the full `symbolic_fluid_frame` program (fresh
lowering, contract-driven, dt controller included) reports
**`complete: True` — zero shortfalls.** This was verified directly against
a freshly-relowered pickle (the stale one predated the last three lowering
commits on this branch).

## What's NOT done yet

`tools/build_fluid_c_shell.py` (new) drives
`emit_module` -> `compile_fortran_module_c_shell` end to end, resolving
every ABI input by the api contract's own dotted source names
(`state.height`, `targets.cfl`, `controller.Kp`, ...) against real feed
objects, and derives workspace/keyed-container extents from the grid size.
It gets to `gfortran` and gets down to **one remaining class of error**:

```
Error: Type mismatch in argument 't571' at (1); passed INTEGER(8) to INTEGER(4)
```

in `ssa_sequence_104_lookup_or_default` (a real emitted SSA function, the
sequence/keyed-table lookup helper from `precompile_to_ssa.py`'s
`_emit_table_lookup`). The callee declares its id/index array formals as
`integer(c_int32_t)`; the caller's actual array argument (`t750`, an
index/key array feeding two positions of that call) resolves to
`integer(c_int64_t)` in the caller's own local type inference. This is an
**array-typed operand**, so it doesn't go through the scalar
`_coerce_call_operand` path fixed above (arrays are passed by reference —
`_call_operand(..., array_expected=True)` never coerces). This looks like
the caller and callee independently inferring different index-array
widths for the same keyed-sequence machinery, not a new bug class — it is
adjacent to (but distinct from) the keyed-mapping container work already
done ([[project-decoder-string-subsystem]] /
`project-decoder-build-region-115-heap` in memory). Four occurrences, all
the same subroutine/positions.

### Next step

Read `_FunctionEmitter`'s array-dtype resolution for `ssa_sequence_*`
formals (search `array_base_ids`, `dynamic_array_ranks`, and how sequence
tables assign id/index array element dtype — likely in
`precompile_to_ssa.py`'s sequence-table emission, or the tensor table
machinery `SSATensorTable`) to find where the caller and callee diverge on
whether a keyed-sequence index array is i32 or i64, and unify it — the
same way `argument_dtypes`/`output_dtypes` unified SCALAR formal types in
this session. Likely fix shape: extend the `callee_argument_dtypes`
fixed-point (already threaded through `emit_module`) to also gate ARRAY
operand element-type coercion in `_call_operand`, not just scalar
`_coerce_call_operand` — or find the single source of truth for these
array element types and stop them diverging in the first place (preferred
per repo convention: fix the seam that produces two different answers,
don't paper over it with a cast).

Once that clears, `tools/build_fluid_c_shell.py` should produce a running
executable; verify its 3-frame native output against the proven
Python-controller run recorded in this session (`dt_next` ->
0.0086 -> 0.02656, `rejected=0`, `mass_err` ~1e-16/frame), then point
`examples/symbolic_fluid_live.py` at the built executable and delete the
custom `NativeSymbolicFluidAdvance`/frame-adapter classes in
`symbolic_fluid_native_runtime.py` — those are exactly the bespoke-harness
shape the user rejected; this C-shell executable is what replaces them.

## Files touched

- `src/compiler/ssa_fortran_backend.py` — the four fixes above.
- `tools/build_fluid_c_shell.py` (new) — driver: relowered-pickle ->
  Fortran module -> C-shell executable, contract-based input resolution.
- `tools/HANDOFF_fluid_c_shell.md` (this file).

## How to reproduce / continue

```bash
# Fresh lowering (already contract-driven, ~10 min):
python -m src.compiler.symbolic_fluid_direct_control --output build/sfdc-cshell

# Verify zero Fortran shortfalls:
python - <<'EOF'
import pickle
from src.compiler.ssa_fortran_backend import emit_module
with open("build/sfdc-cshell/control_repository_ssa.pkl", "rb") as f:
    module, outputs, exports = pickle.load(f)
fm = emit_module(module, name="symbolic_fluid", outputs=outputs)
print("complete:", fm.complete)
EOF

# Attempt the C-shell build (currently fails on the int32/int64 array
# mismatch documented above):
python tools/build_fluid_c_shell.py
```
