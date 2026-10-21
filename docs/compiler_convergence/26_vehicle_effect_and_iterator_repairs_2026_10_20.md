# Vehicle effect and iterator repairs — 2026-10-20

This continues reports 24 and 25 in the canonical repository
`C:/dev/Powershell/turing`.  The worktree already contained the formal-consumer
provenance work described by report 23.  No vehicle source was changed.

## Outcome

Three deterministic compiler failures were repaired at their writers or exact
consumers.

1. Callee-formal pruning no longer mutates a book-backed `SSACallTable`
   through its list view.  It revises each caller's whole `CALL_RECORD` row
   from the incumbent row cell and the exact `PRUNED_CALLEE_FORMAL` decision
   cells.
2. A projected native-union sequence column no longer indexes absent producer
   storage and raises `KeyError`.  The consumer now refuses with
   `ssa-sequence/missing-union-payload-column` before creating a tag column.
3. Output-less numerical regions now publish their exact resident-memory
   effects.  `IndexedStore` result identities are recorded as
   `effect_value_ids`; scalar address lowering carries each identity onto the
   resulting `GetElementPtr`/`Store` transaction as
   `source_effect_value_id`; product accounting accepts the effect only when
   either representation is physically present.  Indexed operations are also
   audited as address consumers, so a shapeless base remains a named
   `unresolved-address-base-contracts` refusal.

These are general identity rules.  None contains a vehicle function name,
source-text match, default storage, tolerance, or alternate execution path.

## Real-source movement

### `_outer_rate_hz`

Command:

```powershell
python -u -m tools.compile_project_catalogue --source tools/run_vehicle_native_assembly.py --entry _outer_rate_hz --output build/vehicle_fix_trial_outer_rate_20261020 --jobs 1
```

The former raw `call_record` write exception is gone.  The unit is partial,
with all seven numerical source regions complete.  Its next exact frontier is
the unmaterialized `math.ceil`/`math.log2` native boundary plus the authored
`args.rate_hz` object projection.

### `_PythonVehicleMaterial.visual_snapshot`

Command:

```powershell
python -u -m tools.compile_project_catalogue --source tools/run_vehicle_native_assembly.py --entry _PythonVehicleMaterial.visual_snapshot --output build/vehicle_fix_trial_visual_snapshot_20261020 --jobs 1
```

The former `KeyError(20)` is gone.  The unit now stops with a structured
`ssa-lowering-incomplete` receipt: `source.items()` has a union value contract
but no published resident payload column, and the conditional return slot is
unresolved on `if_true.1`.

### Vehicle mutation methods

The effect repair was checked on five real methods:

- `set_tire_fidelity_mode`
- `set_tire_gas_charge`
- `set_tire_assembly`
- `set_pillar_pose`
- `set_roller_anchor`

Their output-less store regions no longer fail merely because mutation has no
copied return slot.  Both retained `IndexedStore` and lowered `Store` forms
carry the same effect identity.  The remaining incomplete store regions name
the actual boundary: the nested `self.feeds[...].data` capture has no physical
rank/storage contract.  The isolated method catalogue has no declared record
ABI for `_PythonVehicleMaterial`, so treating that value as a native span would
be an invention.

Representative receipts:

- `build/vehicle_fix_trial_set_tire_effect_20261020/manifest.json`
- `build/vehicle_effect_receipt_set_tire_assembly_20261020/manifest.json`
- `build/vehicle_effect_receipt_set_pillar_pose_20261020/manifest.json`

## Bootstrap cross-check

Unit 081's source was rerun directly:

```powershell
python -u -m tools.compile_project_catalogue --source src/compiler/ssa_object_operations.py --entry _result_layout --output build/bootstrap_unit081_result_layout_trial_20261020 --jobs 1
```

It remains partial, but the receipt partitions the problem cleanly.  Two
scalar predicate regions compile.  The unresolved surface is the compiler
object ABI (`function.blocks.values()`, `b.instrs`, `i.args`, `i.op`), the
tuple-valued layout, the generator-driven `any`, and the unavailable
`next(iter(layouts))` result.  The vehicle nested-object frontier and unit 081
therefore agree: object traversal needs its existing record/sequence ABI at
the producer; it cannot be admitted as scalar dispatch metadata.

No full bootstrap was launched.  The repository guidance says to use a small
real slice first and to reserve the long bootstrap for after the generator
feature is implemented.

## Validation

Focused new regressions: 7 passed.  They cover book-backed call pruning,
missing union payload refusal, retained and address-lowered store effects, and
unranked address refusal.

Four-file compiler gate:

```powershell
python -m pytest -q tests/test_precompile_to_ssa.py tests/test_project_compilation_product.py tests/test_native_call_input_receipts.py tests/test_native_formal_provenance.py
```

Result: 245 passed and 6 failed in 19.45 seconds.  The six failures are the
already-documented fixture faults at the current baseline: one source-less
`FunctionTable.declare`, two source-less propagated-frame-tail fixtures, two
raw `argument_binding` writes, and the old source-less pruning fixture.  The
new book-backed pruning regression passes.

`python -u tools/audit_identity_concordance.py mapping` reports 26 rows across
two functions, zero findings, and zero unsourced facts.  The audit tool's
`--pickle` path does not unwrap the compiler's `(module, outputs, exports)`
artifact tuple and raises `AttributeError`; that tool defect was observed but
not changed here.

Unwrapping that tuple read-only and passing its module to the same audit gives
the expected remaining producer gaps: `_outer_rate_hz` has three
`unaccounted-formal` findings (values 2, 5 and 8), while `set_tire_assembly`
has one (value 6, the nested feed data base).  Both report zero unsourced facts
and a closed latch.

`git diff --check` is clean apart from the repository's existing LF/CRLF
warnings.

## Next exact frontier

The next repair should start at the producer of object storage, not at region
finalization:

1. trace how a declared compiler object (`Function`, `BasicBlock`, `Instr`) is
   supposed to publish its fields and sequence members to unit 081;
2. join the vehicle constructor's `feeds` record and each feed's `data` span to
   the same existing record/sequence projection machinery when compiling the
   project closure;
3. only then rerun the bounded vehicle creep and normal bootstrap.
