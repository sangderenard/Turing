# Vehicle validator to game handoff

Date: 2026-08-29

> **2026-09-01 continuation:** The validator, base game, and integrated
> validator-game now have separate web/native status accounting. The current
> artifact inventory, compiler-dispatch work, mixed scalar/record ABI
> troubleshooting, diagnostics, and next release gates are recorded in
> [`../CONTINUATION_REPORT_VALIDATOR_GAME_DEPLOYMENT_2026-09-01.md`](../CONTINUATION_REPORT_VALIDATOR_GAME_DEPLOYMENT_2026-09-01.md).
> Treat that report as the current deployment status; this document remains
> the product/physics design authority.

## Purpose

The native rig is the construction and validation machine for vehicles that
will enter the game.  It is not a second vehicle implementation.  The next
development phase is to iterate one complete vehicle through the native
assembler until it passes, preserve the exact model/topology/parameter
identity of that result, and then make the game execute that same forward
state transition economically at 120 Hz with three deterministic substeps.
The validator may be slow, deeply instrumented, and exact.  Runtime reduction
is the responsibility of the later game-reducer machine, never a reason to
weaken the validator.

## Current truth

- 2026-09-01 compiler recovery: the complete managed eight-lane/four-wheel
  tire plus repository dt controller now lowers to repository SSA with 106
  functions, zero unresolved calls, and zero undefined operands. A fresh
  integrated O0 validator bundle is at
  `build/vehicle_validator_managed_tire_compiler_fixed_o0`. Its
  `vehicle_game_kernels.dll` compiled at O0, then the existing assembly runner
  passed `clamp-pan`, `engine-pan`, and `engine` and entered `transmission`.
  The run was deliberately interrupted there because correctness had been
  demonstrated and the O0 validator was slow; no performance work was begun.
  Re-run it with:
  `python tools/run_vehicle_native_assembly.py build/vehicle_validator_managed_tire_compiler_fixed_o0 --summary-only`.
- There is not yet a completely passing vehicle, but the same visible native
  run now passes stages 1--18, including gravity admission, rolling start, full
  outfitting/balance, and the strict leveling capture. Gravity admission holds
  6,145 consecutive quiet samples; rolling start holds 2,561, catches
  combustion from the hub wrench, and selects neutral after catch.
- Leveling now consumes a compiler-emitted 13-channel implicit observation
  bank: four tire ground-response forces, four hub pose errors, four tire
  pressures, and body vertical velocity. The bank has finite bandwidth and
  range but explicitly contributes zero mass, wrench, compliance, gap, harness,
  or optics. The controller consumes observed values only; truth and residuals
  remain separately reported. The stage passes an absolute per-corner demand
  of +/-0.5 mm with four live tire responses and full observed support.
- The first stage-19 attempt proved hub reconnect and both differential wrench
  ports but exposed two validator defects: an absolute 0.5 rad/s open-hub cutoff
  rejected a measured 0.623/33.732 rad/s isolation result, and only 0.8 s was
  reserved to settle after the last driven phase. Isolation is now measured as
  a hub/differential speed ratio (required below 5%), the final quarter is a
  braked settle, and the ratio is written into the report. The functional gate
  passes at 2.45%; whole-machine quiescence after excitation remains the current
  stage-19 failure and is reported rather than hidden.
- Native telemetry is now a resumable scientific checkpoint. The runner can
  restore vehicle, output, contact, fixture, and full balloon-membrane state,
  retain the previously passed report prefix, and restart at a named stage.
  The native DLL now has the missing tire-state restore ABI. Failed late stages
  no longer discard the assembled state or prevent subsequent data-only and
  release stages from running.
- The first complete 21-stage report contains 19 passing stage records. The
  destructive pull reaches its 60 s observation ceiling at 8,000 N m per axle,
  47.822 kW peak clutch slip power, 358.062 K peak clutch temperature, 5.495%
  wear, 6.44e-8 glaze, and 0.9999945 minimum health; it correctly remains a
  data-only pass. Its progressive release record becomes non-finite after the
  clamps clear, but it is diagnostic rather than a clean qualification result:
  the restored state was already post-stage-19, stage 19 was replayed, the full
  destructive pull followed, and each clamp was then removed as a discrete
  constraint clear. No spring fracture can be inferred. A clean release must
  resume from the new exact post-stage-18 checkpoint and ramp each clamp's
  stiffness/force authority continuously to zero before detaching it.
- The native assembler, graph telemetry, balloon-skin reference kernels,
  topology-matched tire network candidate, and compiled local material law are
  executable.  The focused suite passes on this state.
- The complete native graph does not yet own every persistent member state and
  invoke the compiled material transition on every applicable edge.  The
  wheel-end constraint/mass system and general equipment load path are also
  incomplete.
- The browser page is stale relative to this work.  Updating or publishing it
  before a native vehicle passes would conceal the actual remaining failures.

This is the required iteration discipline: identify the measured quantity and
the physical intent of its assembly stage before changing either vehicle design
or acceptance policy.

## One-source compilation contract

The model is the equation.  Runtime JSON values are parameters to that model;
they must remain editable and must not be frozen into the equation.

The authoritative inputs are:

1. graph topology and stable state/parameter ABI;
2. equation/process definitions for each graph component and coupling;
3. vehicle/loadout JSON parameters, including material and geometry choices;
4. model-selector identifiers for legitimate alternative equations;
5. precision and scheduler policy chosen by the target emitter.

From those inputs the compiler produces multiple artifacts:

- a native scientific forward program, normally double precision and able to
  use two-limb precision for diagnostic/reference work;
- a native static inverse program composed with AbstractNN Adam for offline
  parameter search, never a tape and never a browser training build;
- a browser forward program using the same topology, equation branches, state
  order, parameter order, units, and transition semantics at lower precision;
- optional learned appendage programs whose weights are separate runtime data
  and whose authority is bounded by exact-reference trials.

Every artifact must carry hashes of topology, equation ABI, ordered state ABI,
ordered parameter ABI, and precision policy.  A vehicle is not transferable if
those identities cannot be compared at load time.

## Validator iteration loop

Use the rig as an automated design/build process, never by teleporting a fully
assembled vehicle into an unconstrained scene:

1. Select the vehicle/loadout JSON and solve engine-pan and occupant packaging.
2. Fit the fused frame envelope, then place the axles under the computed mass
   distribution within declared wheel-placement and clearance bounds.
3. Clamp the pan at the four frame references and install drivetrain members
   with their actual mass, inertia, backlash, wear, and attachment states.
4. Balance with legal braze-on weights where requested; report any required
   remedy rather than silently moving structural body mass.
5. Install wheel pillars, knuckles, bearings, brakes, halfshafts, linkages,
   springs, bump stops, tires, and lines in a controlled sequence.  Explore
   armature range while the frame is restrained.
6. Establish tire/roller CCD crossing, then pressure-load calibration.  Inflate
   from the declared gas state and record membrane, bead, hub, and roller
   reactions before gravity is admitted.
7. Relax the complete unpowered vehicle under gravity.  Require contact
   retention, passive energy behavior, finite state, bounded penetration,
   acceptable vibration decay, and no unintended plasticity or fracture.
8. Exercise steering, service/differential brakes, hub locks, ignition/start,
   a two-of-three rolling start, and the selected transfer ranges.
9. Install optional body and accessories, recompute mass/COM/inertia and every
   attachment wrench, rebalance if allowed, and repeat the relevant gates.
10. Run data-only destructive pulls and traction trials.  Record stall, clutch
    heat/glaze/wear, torsional failure, member plasticity/fracture, and contact
    loss without redefining those trials as pass/fail.
11. Optimize editable JSON parameters against explicit metrics with the native
    inverse/Adam path.  Rebuild only when topology, equation, or ABI changes;
    parameter changes alone must not require symbolic derivation.
12. Repeat until a named configuration passes, then save the validated runtime
    artifact and its complete reference report.

## Minimum pass gates for the first vehicle

- Four tire skins remain coupled to their rims/hubs and retain terrain contact
  under gross vehicle weight without unilateral free energy or tunnelling.
- Stationary total mechanical energy has only declared input and dissipation
  paths.  Gyroscopic, contact, hub, and drivetrain reactions are applied once.
- The frame and wheel-end graph remain connected unless a material transition
  explicitly fractures an edge.  Plastic set and work hardening are persistent.
- Suspension rests at a declared ride-height solution with leveling initially
  off, complementary bump stops, and bounded oscillation.
- Service and differential brakes can arrest their declared inertias; hub
  lockers float or lock at the bearing and reconnect only within slip limits.
- Ignition and starter are separate actions.  Starter and rolling-start paths
  can turn the crank and catch combustion without equating ignition with RPM.
- Steering works at ECU/servo rates when powered and at bounded human-force
  rates after ECU, power, or servo failure.
- Equipment installation updates mass, COM, inertia, clearances, structural
  reactions, electrical/hydraulic/pneumatic demand, and presentation state.
- The report is finite, reproducible under its deterministic seed, and names
  every failed gate with the responsible state channels and top offenders.

## Runtime artifact passed to the game

The validator should emit a compact vehicle package containing:

- validated vehicle/loadout JSON parameters;
- topology/equation/state/parameter ABI hashes;
- ordered initial state and legitimate model-selector IDs;
- mass, COM, principal inertia, attachment, and clearance summaries derived
  from the parameters for quick validation, not as replacement authority;
- reference trajectories and tolerances for short startup/contact regressions;
- optional GPU surrogate weights, normalization, validation loss, novelty
  bounds, exact-trial cadence, and fallback/work-share policy;
- presentation mappings from physical nodes/edges to meshes, lights, audio
  emitters, damage visuals, and the in-world validator machine.

No cached derived value may remain authoritative after a relevant parameter is
edited.  Either recompute it through the shared graph or invalidate the package.

Qualification tolerances are versioned separately from both equation and
vehicle parameters in
`configs/vehicles/qualification/producer-neutral-v1.json`.  The initial spec
defines hard invariants, observation budgets, normalized stationarity limits,
and development/play product classes.  Producers may select a stricter declared
spec; they may not weaken an invariant invisibly or change the physical model
to make a metric pass.  Reports identify the spec and should expose measured
value, tolerance, and margin for every gate.

Product class primarily selects an excitation envelope: terrain/bump height,
spatial wavelength band, cross-slope, traversal speed, gross-mass fraction, and
cycles per band.  Those are inputs to the trial, not looser definitions of
energy conservation or contact correctness.  A producer can declare additional
profiles for commuter, race, agricultural, military, or competition use and
must retain the exact profile with the result.

The producer-facing design objective is an n-axis polar chart backed by named
physical measurements.  Each axis declares its unit, reference range, whether
higher/lower/a target value is better, its desired value, and optimization
weight.  The inverse solver minimizes those normalized residuals by changing
permitted JSON parameters.  Hard invariants and fit/safety constraints are
reported as margins and can never be traded away by a favorable weighted
score.  The chart reports achieved value and uncertainty as well as the compact
display score, so two producers can compare results under the same trial spec.

## Game execution at 120 Hz x 3

The game schedule is one fixed 1/120-second outer tick containing exactly three
ordered 1/360-second physics substeps.  Wall time decides how many outer ticks
to run; it never changes `dt`.  The native validator may oversample the same
transition for scientific trials, but oversampling must not introduce a
different constitutive law or contact branch convention.

Runtime economy should come from compilation and bounded replacement:

- keep the validated graph topology and state layout compact and contiguous;
- compile selected engine/model branches into hard dispatch paths while keeping
  their values in runtime JSON;
- batch four tire appendages and other topology-identical regions;
- use the GPU tire graph surrogate only inside its validated domain, with
  periodic exact membrane trials and immediate exact fallback for novelty,
  damage, plasticity, contact changes, or excessive loss;
- multirate only slow electrical, hydraulic, pneumatic, thermal, controller,
  and presentation observations while accumulating their conserved exchanges
  at the fixed physics boundary;
- render/interpolate from committed physics state; never feed presentation
  averaging back into physical state;
- use native reference traces to quantify drift at 120 x 3 before accepting a
  reduced path for play.

The first performance target is therefore correctness at 120 x 3 for one baked
topology with editable parameters, followed by profiling.  A cheaper equation
is a selectable model, not an accidental divergence between native and web.

## Leveling and falling control

The leveling controller uses an exact four-corner Hadamard identity to move
between corner residuals and heave, roll, pitch, and diagonal cross-weight.
Runtime calibration gains scale those modes before reconstruction.  The main
actuator state is limited by manifold pressure, piston area, efficiency, shared
flow, force reserve, travel, and coarse rate.  Inside the trim-entry band, one
corner per tick receives a bounded short-stroke high-speed preload correction;
selection rotates deterministically so every corner receives equal opportunity.

Alignment actuators are not vertical trim actuators.  They own toe/camber
geometry and a sacrificial series-relief path, so high-current vertical commands
would couple leveling to alignment and breakage.  Final-millimetre control is a
separate series coilover/preload-collar actuator.

Loss of support disables force and cross-weight hunting.  The runtime policy is
an explicit selector with three choices:

1. Hold current geometry.  Preserve coarse/trim state and make no unsupported
   load claim.  Use this when terrain prediction is unavailable or unreliable.
2. Symmetric landing-ready droop.  Move all corners toward a declared safe
   droop at the unloaded placement-rate limit.  This is the default.
3. Terrain-conformal predicted placement.  Move each corner toward a separately
   predicted contact offset.  This requires trustworthy swept terrain probes
   and falls back when prediction confidence is insufficient.

Internal suspension motion cannot change the vehicle COM trajectory in free
fall.  Its legitimate purposes are preparing contact geometry, managing wheel
and body attitude exchange through accounted reactions, and reducing landing
impulse.  It must not manufacture a leveling force against absent ground.

## Relationship to the game world

The validator/assembler is an object inside the existing recursive Living Data
Map game site, not a second validator or game webpage.  Its lifecycle is
explicit: it receives craft material, holds and qualifies the craft while
projecting the construction/line shader, publishes the qualification report,
then releases the exact qualified vehicle entity.  At release the presentation
changes from the projected build lines to the finished car, and that same
entity becomes available to the existing inventory/vehicle-slot pickup path so
the player can grab it and drive away.

The projected subject is `MechanicalCreatureWorld`. Its `initial_vehicle` and
`validator_rig` fields become ordinary inspectable rooms. The world property,
room discovery reference, inventory item, and vehicle slot name the same
initial-car identity; none of those surfaces receives a synthetic copy.
`MechanicalCreatureWorld.tick(tick, dt, subdt, substeps)` assigns one explicit
time envelope to the rig, validator, and car. The website is one caller. A
standalone native host may drive the same entry point in autonomous mode, but
the rig contains no timer or scheduler.

The current browser qualification physics and report logic can be retained, but
its blocking startup overlay is transitional.  Moving it into the rig changes
ownership and presentation, not physics: the game starts normally, the rig is
a world object, and the class tick remains the sole in-game time assignment.
The existing worker executes the assigned envelope rather than independently
owning game time. The rig presentation owns cameras, staging, reports, controls, and
visual explanation; it does not own a second vehicle transition.  Native and
web members import the same compiler-owned vehicle module assembly and exchange
only declared state, parameters, commands, and telemetry through its ABI.

## Deliberately deferred physical variants

- Keep the lower suspension link in the closed force graph, but allow its
  fabricated member geometry to be curved, boxed, or triangulated for ground
  clearance while preserving authoritative endpoints, mass, and beam response.
- Add direct coilover, pull-rod/cantilever, solid-axle/leaf-spring, and long
  stadium-truck multi-shock assemblies as selectable graph topologies. Do not
  emulate them by presentation-only linkage changes.
- Do not expose SPIN or BRACE steering commands on the current one-rack-per-axle
  topology. Honest versions require split racks or independent powered corner
  actuators attached to the real tie rods/knuckles; pivot spin additionally
  requires reversible per-wheel torque authority.

## Immediate work order

1. Use the saved stage checkpoint to iterate the differential-wrench recovery
   interval until its functional proof and post-excitation quiescence both pass;
   do not replay stages 1--18 or weaken the 5% isolation criterion.
2. Preserve the completed destructive-pull characterization. Run clean release
   from the exact post-stage-18 checkpoint, with continuous clamp-force ramp-out
   before topological detach, and isolate the first non-finite balloon/contact
   state if it recurs. Non-finite state remains globally fatal.
3. Make persistent per-edge state part of the native graph tick and invoke the
   compiled material/bushing transition for every applicable structural edge.
4. Put wheel bearing/knuckle/rotor/caliper, hub-lock, and differential rotor
   inertia into that authoritative graph rather than a lumped wrench boundary.
5. Add a general loadout JSON path and qualify body/engine/clutch/accessory
   variants through the same assembler rather than special cases.
6. Materialize the full vehicle objective into static reverse Python/
   AbstractTensor, compose Adam, widen the native training path as requested,
   and prove parameter updates remain JSON data.
7. Emit the validated runtime package and add native-versus-browser trace tests
   at 120 Hz x 3 before rebuilding the Mechanical Creature page. Do not add
   deliberate presentation sleeps; visual pacing must remain a negligible skim
   over real incremental compiler/validator milestones.
8. Only after those comparisons pass, update the game page and publish it from
   the root publication repository. The native report is the gate for that bake,
   not an unrelated presentation copy.

## First handoff task

Run the refreshed bundle through leveling capture and inspect the four pose
residual trajectories and correction-state margins.  Once leveling passes,
continue without changing configuration through differential-wrench proof,
destructive characterization, and progressive release.  Preserve the existing
`native_assembly_report_pre_level_feedback.json` as the comparison baseline.
After the first complete pass, begin the persistent per-edge material/bushing
integration in `vehicle_native_graph_tick`; that remains the shortest path from
a passing construction programme to a validator whose damage result means the
same thing for the native rig and the game.

## Fresh bundle and retained evidence

- Bundle: `build/vehicle_native_validator_20260829`
- Standard kernel: `vehicle_game_kernels.dll`
- Persistent two-limb kernel: `double_double/vehicle_game_kernels_dd.dll`
- Visible scientific programs: `vehicle_scientific_viewer.exe` and
  `vehicle_scientific_viewer_batch.exe`
- Pre-feedback stage-18 baseline:
  `native_assembly_report_pre_level_feedback.json`
- Qualification policy:
  `configs/vehicles/qualification/producer-neutral-v1.json`

The bundle was freshly lowered once from the current canonical source.  A later
resume regenerated and hashed the compiler-owned leveling controller, relinked
the retained canonical C with its current auxiliary ABI, and rebuilt both
viewers.  The manifest verifies pose-error inputs, previous-correction inputs,
and next-correction outputs.
# 2026-08-29 authority correction: Python/AbstractTensor only

The next native or web build must enter through
`src.compiler.vehicle_python_compilation`.  Its authoritative chain is:

`ordinary Python/AbstractTensor -> ProcessGraph -> repository SSA -> target emitter`.

The active native build tool is `tools/build_vehicle_native_teaser.py`; it no
longer imports `vehicle_balloon_tire_native` or the handwritten native tick
shell.  Those older C-rendering modules are retained only as inactive behavior
references while parity is checked.  They must not be linked into a product.

The replacement graph vectorizes batch, wheel, membrane vertex, face, damage
edge, rig point, contact surface, and XYZ axes.  Only the causal tire substep
recurrence remains sequential.  Existing SymPy equations are linked directly
as ProcessGraph functions; they are not reimplemented or interpreted.

Do not start the web build until the combined Python graph has completed its
first compiler shortfall pass and the emitted native artifact has passed clamp
release without NaN or lost ground support.
