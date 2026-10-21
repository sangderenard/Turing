# Vehicle class-fragment and constructor-formal repairs — 2026-10-20

This continues reports 24–26 in the canonical repository
`C:/dev/Powershell/turing`. No vehicle source was changed.

## Outcome

Two general compiler defects were repaired from strict vehicle trials.

1. Recursive source pursuit could retain a declaration-only field shell and
   an executable view of the same source `ClassDef`. `_map_ir_from_ast`
   published both views as independent class objects, and strict class
   navigation then refused the duplicate identities. Declaration-only shells
   now carry an explicit fragment receipt. Map-IR ingestion joins only a
   marked fragment with another view having the same exact class identity,
   source span, bases, permissions, and concordant member records. Two full
   definitions under one identity still raise.
2. A constructor parameter used only by `self.field = parameter` acquired its
   SSA use during receiver-slot injection, after the control builder had
   published `parameter_names`. The store source therefore survived as an
   anonymous public formal. Field-slot injection now receives the exact
   version-zero `(name, value_id)` pairs from the ProcessGraph identity table,
   publishes those names on the rebuilt public signature, and writes the same
   identity onto each formal's `program_abi_parameter` accounting. Conflicting
   names or ids raise.

Neither repair deduplicates by spelling, reconstructs parameter position, or
adds a fallback execution path.

## Strict whole-vehicle planning

The pre-repair command stopped in topology reduction with duplicate class
identities for `STController`, `PlacementTransform`, and `PlacementPayload`.
The competing records shared their exact canonical identity and
`class_node_id`; the Placement records split an executable `to_data` view from
an attribute-only declaration shell.

After the fragment join:

```powershell
python -u tools/lower_vehicle_validator_program.py --plan-only --output build/vehicle_class_fragment_join_20261020
```

Result: `planned` in 83.18 seconds. Topology reduction completed and published
125 deterministic compilation units. The resolved graph contains 30 class
objects with 30 unique class identities. The three formerly duplicated
classes each occur once. Receipt:
`build/vehicle_class_fragment_join_20261020/receipt.json`.

## Constructor-formal trial

Compiling resolved unit 000 before the second repair produced a partial
`_DuallyDTState.__init__`: values `2, 4, 0, 1, 5` were public Store sources
with empty accounting. The resolved ProcessGraph named those same identities
exactly as `material`, `vehicle_in`, `contact_in`, `fixture_in`, and
`vehicle_out`.

After the repair:

```powershell
python -u -m tools.compile_project_catalogue --source tools/run_vehicle_native_assembly.py --entry _DuallyDTState.__init__ --output build/vehicle_fix_trial_dtstate_init_20261020 --jobs 1
```

Result: one linkable unit, zero partial, failed, or blocked units. The final
concordance detector reports zero findings. Its public signature is exactly:

```text
self=3, material=2, vehicle_in=4, contact_in=0, fixture_in=1, vehicle_out=5
```

Manifest:
`build/vehicle_fix_trial_dtstate_init_20261020/manifest.json`.

## Tests

Focused fragment and constructor tests pass 8/8. The combined five-file gate
reported 259 passed and 8 failed in 18.21 seconds. Six failures are the
previously documented source-less/raw-write fixture faults from report 26.
The two additional failures are older shell-deployment fixtures which were not
part of report 26's four-file gate: one has no exact source-pursuit activation
root and one expects a recursive activation receipt that is absent. They were
not modified or waived in this work. `git diff --check` reports only the
existing LF/CRLF warnings.

## Next trials

The successful 125-unit plan is a planning receipt, not a native vehicle
build. It predates the constructor-formal repair and must not be reused across
the changed compiler fingerprint. Regenerate the plan before crawling units.
The already-isolated broad frontiers remain:

- `_outer_rate_hz`: `math.ceil`, `math.log2`, and `args.rate_hz` projection;
- `_PythonVehicleMaterial.visual_snapshot`: the producer-side missing union
  payload column and unresolved true-arm return;
- nested `self.feeds[...].data` stores: missing record/rank storage ABI;
- bootstrap `_result_layout`: structural `Function.blocks -> BasicBlock.instrs
  -> Instr.args` record/sequence ABI plus generator consumers.

