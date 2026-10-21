# Vehicle validator runtime and bootstrap-creep receipt — 2026-10-20

## Outcome

The combined pneumatic simulation and vehicle assembly rig is
`tools/run_vehicle_native_assembly.py` with `--assembly-profile dually-axle`.
It still runs correctly on this machine, but the authoritative fine-tire Python
lane is not interactive.  A bounded one-lane stage completed and rendered a
valid final frame.  The default eight-lane run was stopped after measurement
because each reported substep took about six seconds.

The existing compiler-owned project bootstrap creep can be applied directly to
this source.  No vehicle-specific wrapper or alternate splitting mechanism was
added.  In the bounded run it found 12 authored units, completed attempts for
10, produced eight complete source-region SSA artifacts inside eight partial
units, installed no verified native products, and did not reach the root
`_run_dually_python_profile` unit.

## Machine

- CPU: AMD Ryzen 9 7900X, 12 cores / 24 logical processors
- RAM: 33,377,648,640 bytes
- GPUs: NVIDIA RTX 3060, NVIDIA GTX 1660 SUPER, AMD integrated graphics
- Python: 3.11.7 at
  `C:/Users/alber/AppData/Local/Programs/Python/Python311/python.exe`

## Runtime observations

The default eight-lane headless command was:

```powershell
python -u tools/run_vehicle_native_assembly.py build/vehicle_validator_new_machine_20261020 --assembly-profile dually-axle --python-material --headless-frame build/vehicle_validator_new_machine_20261020/final-frame.png
```

At heartbeat 20 it reported:

- simulation time: `0.000582644 s`
- heartbeat wall time: `127.267 s`
- `tick_s_per_substep`: `6.08575`
- viewer wait: `0.2131 s/substep`
- tire critical dt: `8.257994149817198e-05`
- maximum omega: `24218.956367803585`

The run was stopped after about 4.5 minutes.  This was a bounded diagnostic,
not a simulator failure.

The successful one-lane bounded preview command was:

```powershell
python -u tools/run_vehicle_native_assembly.py build/vehicle_validator_preview_20261020 --assembly-profile dually-axle --python-material --headless-frame build/vehicle_validator_preview_20261020/final-frame.png --lanes 1 --dually-stage-seconds 0.00003 --stop-after-stage position-four-wheel-units-on-eventual-installation-pillars
```

It exited zero after roughly three minutes.  The stage accepted 31 of 31
attempts with no rejection.  The final simulation time was `0.0009765625 s`,
the minimum 1/1024-second stage envelope.  The rendered HUD reported four
fine-mesh tires at 101.3 kPa, maximum error `6.4e-8 m`, and no violated rule.

No equivalent prior-machine wall-clock receipt was found, so this run does not
support an honest old-machine/new-machine speedup ratio.

## Strict whole-entry planning

This command exercised the vehicle validator's existing strict full-native
overlay:

```powershell
python -u tools/lower_vehicle_validator_program.py --plan-only --output build/vehicle_validator_creep_20261020/planning
```

It exited 1 after 38.823 seconds during topology reduction, before emitting a
unit plan:

```text
ValueError: class navigation contains duplicate class identities: ('src.common.dt_system.dt_controller.STController', 'src.compiler.abstract_ui_placement.PlacementPayload', 'src.compiler.abstract_ui_placement.PlacementTransform')
```

Receipt:
`build/vehicle_validator_creep_20261020/planning/receipt.json`

Diagnostic log:
`build/compiler-logs/compile_447nx7k1.log`

## Recursive project creep

The existing automatic source splitter was run as a module so repository
imports resolved normally:

```powershell
python -u -m tools.compile_project_catalogue --source tools/run_vehicle_native_assembly.py --entry _run_dually_python_profile --output build/vehicle_validator_project_creep_20261020 --creep-project --jobs 4 --max-total-gb 20 --worker-reserve-gb 4 --max-worker-gb 8 --max-bootstrap-rounds 4
```

This uses `compile_project_bootstrap_creep`, which splits authored calls and
then recursively applies process-graph creep.  It uses the general project
extraction contract and installs only verified native results; it is not proof
of a full-native vehicle executable under the stricter vehicle overlay.

The bounded round-zero state in
`build/vehicle_validator_project_creep_20261020/round_000/progress.json` was:

- 12 authored units total
- 10 completed attempts
- 8 partial units
- 2 failed units
- 8 complete source-region SSA artifacts
- 9 incomplete source regions
- 0 verified native installations
- 1 running unit at the cutoff
- 1 pending root unit

Partial units:

- `_DuallyDTState.__init__`
- `_PythonCall.__init__`
- `_PythonVehicleMaterial.set_pillar_pose`
- `_PythonVehicleMaterial.set_roller_anchor`
- `_PythonVehicleMaterial.set_tire_assembly`
- `_PythonVehicleMaterial.set_tire_fidelity_mode`
- `_PythonVehicleMaterial.set_tire_gas_charge`
- `_array`

Hard failures observed before the cutoff:

- `_outer_rate_hz`: raw write to the `call_record` page was refused; the
  diagnostic says the fact must be written through `IdentityBook.post` with
  the cells from which it was read.
- `_PythonVehicleMaterial.visual_snapshot`: failure payload `20`.  This is an
  observed payload only; its writer was not traced in this diagnostic.

`_PythonVehicleMaterial.__init__` remained in source closure for 606 seconds.
It consumed about 2.50 GB peak private memory and about 686 MB resident memory.
The closure crossed more than 1,400 work items and reached compiler-memory and
graph infrastructure such as `StructView`, `ExternalCallsiteFact`, and
`IntegerRegionAllocator`.  The parent command was then interrupted deliberately
to bound the diagnostic.  Its resulting `[Errno 22] Invalid argument` failure
file is interruption fallout, not an independently observed compiler defect.

The root `_run_dually_python_profile` unit was still pending and never started.
Because round zero did not finish, no round manifest or top-level final manifest
was written.  The durable live receipt is
`build/vehicle_validator_project_creep_20261020/round_000/progress.json`.

## New-problem catalogue

1. The fine deformable-tire Python lane remains several seconds per substep on
   this machine; reducing lane count does not make it interactive.
2. Strict whole-entry planning stops on duplicate class identities before it
   can emit a recursive unit plan.
3. Project splitting works and preserves useful partial SSA, but none of the
   attempted units became a verified native installation.
4. `_PythonVehicleMaterial.__init__` has an effectively unbounded source
   closure at the current authored split boundary.
5. `_outer_rate_hz` violates the IdentityBook write path during SSA lowering.
6. `visual_snapshot` reports the opaque failure payload `20`; its provenance is
   still unknown.
7. The root vehicle profile remains unmeasured because the constructor attempt
   monopolized the last active worker until the diagnostic cutoff.

No source code was changed during this measurement.  The next compiler work is
to trace the two exact identity failures and the constructor closure boundary;
changing the vehicle runner would only conceal those compiler-owned defects.
