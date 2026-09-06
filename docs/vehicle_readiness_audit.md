# Mechanical Creature vehicle readiness audit

Date: 2026-08-29

## Delta since the 2026-08-28 audit

The project is substantially closer to a useful native validator, but it has
not yet qualified one vehicle.  The present distinction is important:

- The native assembler now executes a 21-stage physical installation sequence
  rather than merely describing five conceptual stages.  It begins with a
  clamped pan, installs and observes progressively larger graph regions, admits
  gravity only after CCD crossing plus calibrated pressure load, and retains
  reports and binary telemetry for every attempted stage.
- The latest exact visible run passes stages 1--18. Gravity admission retains
  all four balloon skins on their rollers and records 6,145 consecutive quiet
  samples. Rolling start catches combustion from an external hub wrench and
  returns to neutral. The strict leveling capture passes an absolute +/-0.5 mm
  corner-pose requirement with four live ground-response observations.
- The authoritative tire-local learned candidate is now a topology-matched
  graph operator over the balloon membrane edge list.  Its physical graph is
  cyclic; its executable forward/backward process is a DAG.  The older
  convolution experiment is retained only as non-authoritative archaeology.
- A shared-SSA compiled member-material kernel now executes elastic response,
  plastic set, work hardening, and irreversible fracture.  It passes native
  behavioral tests, but has not yet been applied as the state transition for
  every structural edge in the complete native vehicle graph.
- Body packaging, minimum cab geometry, configurable bed length, chassis fit,
  axle-under-COM wheelbase placement, composite/nonlinear spring selection,
  and complementary bump stops are now runtime-parameterized construction
  inputs.  The frame/body structural graph remains fused; there is no movable
  body-mass sled.
- The scientific viewer source now reveals parts and force paths by assembly
  stage and draws live force vectors.  A fresh native bundle is required before
  those source changes can be observed.

Focused evidence includes 19 passing tests across native assembly and native
deployment/viewer generation after adding the resume ABI. Earlier focused
material and tire-local graph tests remain useful evidence, but no broad suite
was rerun in this final pass. This narrows the next blocking task to the
post-excitation stability of stage 19 and completing the final release report;
it does not yet justify publication.

The first corrected-spec run then proved 12,288 consecutive quiet samples under
gravity but exposed a second policy mismatch: locked pan clamps carried most
gross weight while the hidden live-contact rule demanded 15% of full corner
weight at every tire.  All tires still had active membrane contact, positive
79--205 N roller loads, and 16--33 Pa pressure increases.  The clamped stage now
uses a spec-declared small positive contact/load threshold; the later release
stage retains its separate requirement that tires carry at least 80% of total
vehicle weight.  Reports now distinguish stability from sensor-gate failure.

The leveling path now has a compiler-emitted implicit observation bank instead
of reading a stale host channel. It observes four tire forces, four hub pose
errors, four pressures, and body vertical velocity with finite bandwidth and
range. It is explicitly massless and applies no mechanical wrench, compliance,
gap, harness, or optics. Truth, observation, and residual telemetry remain
separate. A duplicated obsolete final acceptance block was removed so the one
declared observed-ground-response gate is authoritative.

Stage 19 now proves the functional differential/hub path: open-hub wheel speed
is 2.45% of driven differential speed, both ports turn, reconnect occurs at
zero slip, and locked hubs drive the wheels. It remains red because the whole
machine does not return to the generic quiescence envelope after excitation;
that failure is retained. The validator no longer stops there: telemetry can
restore the complete vehicle/contact/fixture/balloon state at a named stage,
and late failures no longer discard later destructive and release evidence.

The resumed run completed all 21 stage records. Nineteen pass. The 60-second
destructive pull terminates at its observation ceiling rather than stall or
rupture: 8,000 N m maximum accessory load per axle, 47.822 kW peak clutch slip
power, 358.062 K peak clutch temperature, 5.495% wear, negligible glaze, and
0.9999945 minimum health. The subsequent release record becomes non-finite, but
it is not a clean release qualification: the saved telemetry was post-stage-19,
stage 19 was replayed, the destructive pull followed, and individual clamps
were removed discontinuously. It proves that this carried-over state diverges;
it does not prove spring fracture or isolate the cause. The runner now writes
an exact post-stage checkpoint for every stage and rejects accidental same-stage
replay. Clean release must resume from stage 18 and continuously ramp clamp
authority to zero before detachment.

Leveling has since advanced from that minimal feedback law to a compiled
pressure/flow-aware controller.  It decomposes measured corner error into
heave, roll, pitch, and diagonal cross-weight, applies runtime calibration
gains, bounds coarse correction by hydraulic force and shared flow, and performs
short-stroke high-rate trim one corner at a time in deterministic round-robin
order.  Vehicle JSON now declares piston area, manifold pressure, flow,
efficiency, force reserve, coarse rate, trim rate, trim stroke, and trim-entry
error.  Calibration capture has a 20-second minimum instead of leaving the
final target only two seconds.

The same 53-input/33-output kernel makes unsupported behavior explicit.  Force
and cross-weight hunting are disabled as support vanishes; selectable policies
hold current geometry, move symmetrically to landing-ready droop, or follow
per-corner predicted terrain offsets at a bounded unloaded rate.  Behavioral C
tests prove pressure/rate bounds, frozen force hunting in a fall, all three
policy directions, and single-corner round-robin trim.  A complete rig rerun is
still needed to measure actual leveling convergence and later release behavior.

## Decision

The situation is clear enough to continue targeted development, but it is not
ready for the long page build, publication, or a claim of native/game parity.
Several important systems are executable and tested. Several others are only
data contracts, JavaScript host mechanics, diagnostic native equipment, or
source-presence tests. Those distinctions are listed explicitly below.

The generated Mechanical Creature page predates the latest equation, inverse,
attachment, ballast, bumper, drag, and precision changes. Nothing from this
audit has been published.

## Non-negotiable compilation contract

- The model is the equation.
- JSON values are runtime parameters. They are not constants and are not
  frozen, folded, or renamed as model coefficients.
- Equation constants such as pi may be represented in multiple limbs.
- Reverse mode is a static ProcessGraph transformation. It does not use a tape.
- The reverse equation is materialized as Python/AbstractTensor, composed with
  AbstractNN Adam, widened to two limbs, and then AOT-compiled to native C.
- Browser Wasm/WebGPU is forward-only. It does not contain Adam or training.
- Forward CCD chooses a contact branch. Reverse derivatives may refine a fixed
  branch, but do not replace contact discovery or validation.
- The native rig and game must consume the same equation and parameter ABI.
  A presentation renderer may differ; a reduced physics model may not.

## Evidence-backed status

| System | Current status | What is actually executable | Remaining gap |
|---|---|---|---|
| Fixed time and substeps | Integrated | Worker advances the vehicle at fixed 120 Hz with three fixed substeps; wall time schedules work but does not enlarge physics `dt`. | Scientific managed-time substitution is documented but not the browser's active scheduler. |
| Spawn and qualification | Integrated browser gate | The page withholds the living map for 20 simulated seconds, keeps the engine off until 10 seconds, starts it at 10 seconds, records epsilon without gating on it, and checks contact dropout, tire penetration, energy creation, oscillation, tilt, and finite state. Leveling defaults off. | The gate observes browser snapshots and has not been rerun against a newly built page containing this audit's changes. |
| Tire constitutive law | Integrated equation | The shared SymPy contact equation uses toroidal pneumatic volume and pressure, a pressure-derived patch, combined-slip Coulomb/Stribeck limiting, and dynamic longitudinal/lateral sidewall states. Its radial response is now a fixed three-stage implicit-midpoint mode: undamped steps conserve the local quadratic energy exactly and carcass loss removes an explicit nonnegative amount. The reaction is its impulse over `dt`, never positional rejection. | The gas law is locally linearized over one contact step for the passive modal solve; a future discrete-gradient nonlinear gas solve could remove that approximation. |
| Browser tire/terrain crossing | Analytic local-plane path; finite-branch work remains | Resident GPU contact already evaluates the concentric torus active-arc inequality and closed-form penetration moment. The scalar Wasm host now uses the same continuous torus/local-plane geometry and swept ring-support crossing instead of the 5x3 radial maximum-penetration candidate. Both remain unilateral and apply no post-step wheel rejection. | Finite terrain-triangle active-set selection is still host geometry rather than part of the compiled integral operator. Edge/vertex torus branches need the same analytic treatment before claiming a general soft-torus/triangle collider. |
| Native torus contact | Integrated native kernel | The symbolic torus/plane boundary integral lowers completely to C and LLVM and is called by the native roller/terrain shell. | Active finite-triangle face/edge/vertex branch selection remains outside the integral kernel. |
| Suspension quiescence | Executable targeted test | A 20-second engine-off, three-substep, gross-mass test passes and checks passive energy, contact retention, penetration, and oscillations. | It exercises the canonical transition/contact path, not the complete deformable mechanical graph. |
| Wheel-end graph | Partial and blocking | Hub, five-axis bearing, knuckle, upright, rotor, caliper, steering arm, halfshaft joints, wishbones, and coilover are explicit connected graph parts; unsprung mass includes upright, caliper, rotor, wheel, tire, and the coilover fraction. The worker solves their displayed constraint positions. | The canonical chassis kernel still consumes a lumped wheel-end wrench. Bearing/knuckle constraint dynamics are not compiled as the authoritative mass matrix. |
| Differential brakes | Functional torque, incomplete inertia | Front/rear differential brakes apply a bounded implicit dissipative shaft constraint and publish reaction torque; outrigger deployment locks all wheel brakes. | The differential brake rotor declares polar inertia but that inertia is not yet coupled into the driveline mass matrix. Source currently labels this as a future pass. |
| Structural plasticity and fracture | Compiled local kernel; whole-graph integration blocking | Worker state tracks member elastic/plastic strain, changed rest lengths, fracture/open edges, coilover damage, halfshaft fracture, shell mounts, and per-junction bushing dissipation. A shared-SSA native member kernel now behaviorally proves elastic response, plastic set, work hardening, and irreversible fracture. | The compiled material transition is not yet invoked for every edge by the authoritative native vehicle graph. Load distribution and parts of the browser position solve remain host-authored mechanics. |
| Bushings | Partial | Every applicable graph edge carries a static parameterized six-axis Kelvin-Voigt pack; worker telemetry integrates linear/angular dissipation. | Bushing forces are not yet all reduced through a compiled graph dynamics solve; some are telemetry/dissipation over the worker position solution. |
| Vehicle mass accounting | Integrated for current selectable parts | Base parts, live fuel, selected body/ammunition, engine, clutch, wheel part, bumpers, attachment bosses, hangers, and requested ballast update total mass, COM, and principal inertia through runtime state and/or configuration rebuilds. | The native rig does not yet accept an equipment-selection JSON and rebuild the full outfitted configuration before solving. Some graph nodes still say their masses are already included in the lumped base mass rather than owning independent generalized coordinates. |
| Generic wrench attachments | Partial | Four persistent frame-corner bosses have admitted force/moment envelopes, breakable six-axis mounts, mass, COM/inertia accounting, and are used by bumpers and ballast hangers. | This is not yet a truly arbitrary any-graph-point braze-on constructor. Body, turret, and engine mounts still use their own authored mount families. |
| Heavy bumpers | Partial | Front/rear tube assemblies have density-derived mass, explicit endpoints, preload, travel, compression/rebound damping, force limits, graph load paths, and worker terrain contact through their shock law. | Their dynamics remain in the browser worker rather than the shared compiled mechanical equation/native rig. |
| Ballast hangers | Integrated configuration/graph | Per-corner requested mass is converted by material density into physical block volume/height; impossible clearance is rejected; hanger and block masses enter COM/inertia; graph nodes and load paths are real. | Native equipment loading is not yet wired to select these values from a rig JSON. |
| Aerodynamic drag vectors | Integrated equation | Three runtime body-frame unit vectors carry independent `Cd` and reference area and contribute `-0.5*rho*Cd*A*abs(v dot n)*(v dot n)*n` after yaw rotation. Defaults cover longitudinal, lateral, and vertical drag. Unit-vector validation and force evaluation pass. | Roll/pitch rotation of the aerodynamic frame is not yet included; current compiled transform uses yaw. |
| Engine/clutch/transmission selection | Partial | Durable engine records include Jeep I6, commuter I4, Merlin/Packard V12, industrial diesel, electric/servo and others; clutch, transmission, transfer ranges, chassis/oil-pan fit, body packaging, axle-under-COM placement, and live parameter updates exist without per-profile vehicle-kernel prebakes. Composite/nonlinear spring and bump-stop laws share the vehicle equation. | Several architecture/workshop fields are records rather than a cylinder-pressure/crankshaft physical subgraph. The old shop engine, ship engine, CVT, telescoping universal drive shaft, and chain/belt test parts are not complete authoritative components. |
| Fuel, ignition, starter, alternator, battery | Partial | Runtime fuel mass/consumption, incompatible-fuel derating/damage, ignition dispatch, starter torque/load, alternator generation, battery energy, electrical loads, and engine-on state after qualification exist. Timing feeds engine audio. | The engine profile's mixture-control record admits that manifold/charge flow is not yet torque-authoritative; torque remains the compiled BMEP curve with compatibility/timing scales. |
| Steering | Partial | Front/rear racks, pinions, proportioner, tie rods, knuckles, servo and mechanical fallback are explicit; worker computes velocity-sensitive ECU rate or assisted/manual human-force rates when ECU/power/coupling fails. | Authoritative wheel-end dynamics are still lumped at the canonical kernel boundary, so the graph topology is ahead of its compiled mass/constraint solve. |
| TC, ABS, TILT, cruise, governor | Integrated control equations/worker state | Persistent slip/utilization filters drive TC/ABS; TILT uses pitch/COM logic, governor intervention, and rear differential brake; cruise solves throttle against registered speed; governor is runtime-settable. | Behavioral tests are thinner than the implementation: many current tests assert source presence rather than running closed-loop scenarios. |
| Reverse-throttle bug | Implemented guard | Direction changes brake against motion, suppress throttle during reversal, and clear feathered throttle state. | Needs a behavioral regression that performs forward-to-reverse and proves throttle cannot remain pegged. |
| Lights and horn | Integrated browser systems | Head/tail/brake lights consume electrical power, wiring damage can disable circuits, headlights enter the renderer light set, and horn audio is generated. | Native scientific viewer does not model the complete electrical/lighting presentation. |
| Engine audio DSP | Integrated browser audio | The Wasm engine PCM path uses firing/timing telemetry, dry signal, first octave-down shadow, dry-shadow product, a second octave-down shadow of that product, bounded mix, and the Spectral Analyzer-derived stateful FIR mid/high rolloff. | Audio is intentionally observational and is not the mechanical simulation clock. |
| Hydraulics, pneumatics, leveling, alignment | Partial | Leveling defaults off; worker-owned slow actuators, pose presets, manual wheel mode, tire pressure regulation, compressor/pump loads, link-length modifiers, and stationary/full-time alignment calibration exist. | Much of the mechanical realization is a worker host solve. It is not in native shared-SSA parity. |
| Turret body, armor, recoil, ammo, outriggers | Partial | Five turrets, armor mass, ammo mass/volume capacity, crosshair targeting, friendly-fire rejection, fire takeover, individual recoil impulses, four hydraulic outriggers, terrain anchoring, brake interlock, and graph edges/UI controls exist. | Turret targeting/fire tests largely inspect JavaScript presence. Recoil is applied by the worker, not the compiled graph equation. The native rig cannot yet outfit and solve this body. |
| Native scientific rig | Executable resumable staged validator; still blocking | The native builder compiles the canonical vehicle transition, balloon/contact kernels, roller fixture, graph tick, GLSL diagnostic renderer, and optional persistent two-limb versions. The assembler emits all 21 stage records and a separate exact post-stage checkpoint after every stage. Vehicle/contact/fixture/full membrane state can be restored only at the matching next stage unless replay is explicit. Nineteen stages pass; stage 19's drivetrain-specific proof also passes. | Stage 19 fails post-excitation whole-machine quiescence. The observed release divergence is confounded by post-stage-19 replay, the destructive pull, and discontinuous clamp clears; clean release from stage 18 remains required. Whole-graph constraint/material parity and complete arbitrary equipment loading remain incomplete. |
| Inverse/Adam path | Executable small proof, not vehicle-ready | A static ProcessGraph reverse is materialized into AbstractTensor Python without a tape, composed with functional Adam, widened to two limbs, and now executes a real scalar update. The two-limb type now has a variable real-power operator using limb-aware log/exp range reduction. | No canonical vehicle objective JSON has been accepted through full reverse + two-limb AOT C. A separate repository-SSA/LLVM reverse route still has an aggregate-region lowering shortfall after an earlier fixed-point bug was removed. |
| Rig outfitting stages | Executable core sequence; incomplete outfitting | The native tool executes 21 installation/calibration stages from clamped pan through destructive/data trials. Each stage has an installed component set, mass, clamp reactions, stability observations, and retained telemetry. | Equipment selection is not yet a general rig input that reconstructs arbitrary body/engine/clutch/accessory loadouts. Later powered and destructive stages remain unreachable until gravity admission passes. |
| Build progress | Partial | Page build reports eight vehicle lowering milestones with elapsed time and writes atomically; the qualification page reports physical simulated-time progress. | Deep backend passes still do not publish incremental operator/region counts, so one milestone can remain quiet for minutes. |
| Publication | Not ready | The correct publishing repository is the root `nogodsnomasters` repository, not Turing except for the known mistaken copy. | Current source is dirty, generated HTML is stale, focused browser validation has not run, and no audited artifact has been copied, committed, or pushed. |

## Tests that currently provide strong evidence

- Verified the torus active-arc closed form against 4,096-interval direct
  quadrature and verified the three-stage radial mode's exact discrete energy
  identity with zero and positive carcass loss.
- Lowered the modified shared contact equation through native C, scalar Wasm,
  and vectorized WebGPU; checked emitted worker JavaScript syntax with Node.
- Executed two-limb variable real power across the precision surface.
- Executed static reverse plus two-limb Adam on a small non-neural equation.
- Executed the 20-second engine-off gross-weight suspension quiescence test.
- Compiled the modified canonical vehicle equation after adding drag vectors.
- Validated density-sized ballast, impossible-fit rejection, bumper shocks, and
  generic corner attachment graph paths.
- Compiled and ran native passive/locked fixture tests and two-limb C lane
  preservation in the native deployment suite during this audit.

## Tests that are currently weaker than their names may imply

Many vehicle UI, turret, hydraulic, electrical, steering, damage, and native
tests assert that a data field or JavaScript source fragment exists. Those are
useful ABI guards, but they do not prove closed-loop physical behavior. This is
especially important for wheel-end coupling, reverse-throttle recovery,
outfitted mass changes, turret recoil, outrigger anchoring, lighting failure,
and plastic/fracture energy behavior.

## Required order before a long page build

1. Move finite-triangle face/edge/vertex active-set selection into the shared
   torus contact operator. Browser GPU and scalar paths now use a continuous
   torus/local-plane arc, but general triangle-boundary contact is not complete.
2. Put wheel bearing/knuckle/rotor/caliper and differential-brake rotor inertia
   into the authoritative compiled mass/constraint equation rather than a
   lumped wheel-end contract.
3. Move mechanical-graph constraint, bushing, plastic-set, fracture, recoil,
   bumper and outrigger force updates behind the shared compiler input/ABI so
   native and browser execute the same graph math.
4. Give the native rig a real equipment-selection JSON path that rebuilds and
   validates mass, COM, inertia, clearance and attachment wrenches at each
   outfitting stage.
5. Run the canonical vehicle objective through static reverse, two-limb Adam,
   and native C AOT; do not involve browser Wasm in training.
6. Add behavioral scenarios for direction reversal, TILT/differential brake,
   powered/manual steering failure, body/ammo/recoil, outriggers, lights, and
   damage passivity.
7. Run the focused native, worker, vehicle, audio, and page suites; build one
   fresh page; verify the qualification gate and drive it in a real browser.
8. Only then copy the validated artifact to the root publication repository,
   commit, and push.
# Direction correction: compiled balloon-skin tire (2026-08-28)

The analytic torus/plane contact arc is no longer the intended tire collision
authority.  It is retained only as historical/transitional code while the
vehicle is moved to `compiled-balloon-skin-v1`.  The replacement is a closed,
outward-wound triangle membrane with per-vertex mass, StVK face energy,
Kelvin strain-rate dissipation, enclosed-volume polytropic gas pressure, and
equal/opposite bead-to-rim wrenches.  Hard terrain contact is evaluated from
the deformed skin state, not from the torus used to generate its rest mesh.

The compiler-owned equations and topology ABI are in
`src/compiler/vehicle_balloon_tire.py`.  Until its mesh assembly/integrator is
the worker and native authority, the old radial/toroidal path remains a known
deployment blocker and must not be described as the completed tire model.

`src/compiler/vehicle_tire_force_network.py` now defines a complementary
batched deployment surrogate. It consumes periodic membrane and aligned terrain
histories through finite-difference orders, rest geometry, explicit gas
pressure/volume/temperature, and 48 canonical vehicle-state channels, then
predicts the six-axis bead/rim hub wrench plus three thermodynamic consistency
lanes. Its
forward and functional Adam programs compile on AbstractTensor; symbolic/layer
derivatives feed Adam through explicit gradient slots, so a stored backward
graph is optional. The scientific membrane/contact graph remains the teacher
and acceptance oracle.

`src/compiler/vehicle_tire_force_workshare.py` supplies the exact-first
deployment transition. Its loss-to-alpha equation compiles through shared SSA
to C and Wasm. Runtime alpha is also a deterministic reference-work budget:
low validated loss buys sparse network-only steps, but a bounded periodic trial
interval, rapid high-loss recovery, and immediate plastic/contact-novelty
override keep the scientific rig in authority wherever the surrogate is least
trustworthy.

`src/compiler/vehicle_tire_authority.py` content-addresses topology, ordered
state, physical normalization, exact kernels, GPU candidates, work sharing,
and multiplayer reconciliation into one manifest. The learned path is GPU-only;
native LLVM is its parity oracle. The linear candidate is one tiled resident
GEMM with JSON parameters. The inflated reference mesh now has a conservative
construction-prestress lane that cancels reference gas pressure exactly,
removing the prior unbalanced startup inflation impulse.
