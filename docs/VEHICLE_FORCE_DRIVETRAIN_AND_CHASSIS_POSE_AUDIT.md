# Vehicle force drivetrain and chassis pose audit

Date: 2026-08-25

## Verdict

The earlier vehicle felt better than its mechanics deserved. Its contact
kernel received longitudinal slip as `vehicle speed - commanded target speed`,
while scalar Wasm independently accelerated toward the same target and
steering directly accelerated yaw. That was a speed controller dressed as tire
physics. It could not faithfully distinguish gas-pedal torque, wheel spin,
static grip, kinetic sliding, or steering-generated yaw.

The vehicle now advances in this order:

```text
pedals → configured axle torque → four wheel angular velocities
       → contact-point velocity and tire slip
       → four compiled pressure/load/friction contact lanes (WebGPU)
       → summed force and r×F torque
       → compiled linear + roll/pitch/yaw chassis step (Wasm)
       → one complete world snapshot
```

There is no desired chassis speed in this chain. The configured maximum wheel
angular speed is a numerical/engine safety limit, not a road-speed command.
Front steering changes the two front contact tangent bases. Yaw therefore
comes from the tire forces and their moment arms.

Live control verification exposed a handedness mismatch: positive UI strafe
means camera-right, while positive yaw rotates the chassis heading toward
world-right, but a positive front tangent angle produced the opposite `r×F`
yaw moment. The front-axle steering angle is now negated at the control/physics
boundary. Only the first two named contact lanes (`front_left`, `front_right`)
receive that steering angle.

A later audit found another false input at this boundary: all four contact
lanes derived compression from chassis-center height alone. Roll and pitch did
not load individual corners, and an inverted chassis could still find terrain
beneath its wheel X/Z coordinates. The compiled equations were real, but that
host measurement made their result physically meaningless.

Each corner now measures terrain distance along chassis-local suspension-down.
The ray must point into the terrain normal (`-down·normal > 0.18`), the surface
must lie forward along that ray, and the resulting geometric compression must
be positive. This per-corner compression is the explicit input to both the
parallel WGSL contact lane and scalar Wasm chassis transition. Roll and pitch
therefore load and unload different corners. Once inverted, wheel support is
zero: tires may spin internally, but they cannot publish normal, friction, or
drive force to the chassis.

## State and geometry

The worker is still the single authoritative owner of world physics. Its
snapshot ABI now retains position, linear velocity, roll/pitch/yaw, three
angular velocities, four suspension compressions, and four wheel angular
velocities. A tick cannot overlap the next tick and cannot publish until the
selected contact stage and chassis Wasm stage have joined. WebGPU's promise API
is therefore an implementation boundary, not a second simulation schedule.

The four suspension hard points are published as a stick-and-ball chassis:
four force-bearing nodes, four perimeter members, and two diagonal members.
The members are rigid distance constraints; compliance lives in the four
suspension springs. This deliberately follows the useful force-pair and
spring/damper contract in `src/common/dt_system/classic_mechanics` without
making the vehicle chassis a numerically fragile floppy body. Contact force at
each node contributes both chassis force and `attachment × force` torque.
The members publish steel material properties, but the solver interprets them
as the rigid limit rather than integrating enormous explicit steel spring
frequencies at 120 Hz.

The suspension uses the directional pneumatic-damper shape already established
by Turing's `PneumaticDamperEngine`: 3200 N·s/m in compression, 4100 N·s/m in
rebound, and 0.96 pneumatic efficiency. Chassis angular damping is 4.2 s⁻¹.
The compiled spring law clamps measured compression rate to the configured
1.25 m/s before applying the asymmetric gas damping. This prevents initial
contact acquisition from turning a larger damper coefficient into an enormous
one-tick launch impulse.

The scale audit places wheel centers at ±0.56 m around a ±0.42 m body and uses
a 1.24 m axle spacing. The complete 0.18 m tire width therefore clears the body
rather than intersecting it. Suspension travel is 0.20 m—below the 0.32 m tire
diameter—instead of the former 0.34 m. Coil stiffness is 26 kN/m before the
instantaneous wishbone motion-ratio reduction. The higher asymmetric damping
puts the 620 kg chassis near a controlled off-road response rather than the
former long-travel, underdamped buoyancy.

Every upper and lower pickup now has a visible rigid mount edge to its adjacent
lower-frame node. Each coilover tower connects into the roll-cage lower node,
and every cage lower node connects back to the triangulated frame. Those are
declared load paths in the same mechanical graph, not decorative bars.

## Slip-derivative traction and brake control

Each wheel retains its previous longitudinal slip. The compiled chassis/wheel
step measures `max(0, (|slip|-|previous slip|)/dt)` and combines that growth
with current friction utilization. Above the configured 0.92 yellow-band
target, or when slip magnitude is rapidly increasing, it progressively scales
that wheel's drive torque and brake torque. Throttle and ABS have independent
intervention gains; both retain a configured 0.08 minimum authority. The
controller does not assign velocity and does not manufacture grip—the tire
kernel still clips force using pressure/load-sensitive Coulomb limits.

The recycled snapshot carries four traction scales and four ABS scales, and
the contact HUD publishes both beside every patch. Green means reserve grip,
yellow is the intended controlled boundary, and red remains genuine kinetic
sliding rather than a controller-selected mode.

The renderer now applies roll, pitch, and yaw to the generated chassis mesh.
The mounted HUD retains the four colored pneumatic contact patches and adds a
stick/ball structure view whose node colors report grip/limit/slide/air and
whose member weight/color reports paired suspension compression.

## Motion cues, chase camera, depth, and wheel realization

The car JSON now also parameterizes presentation without giving presentation
physics authority. Upward-facing world surfaces receive a procedural,
world-space tiled grid with a stronger major line every four tiles. Because the
pattern is anchored to world coordinates rather than the camera or mesh UVs,
its optical flow conveys speed and remains continuous across generated slabs
and analytic gradient solids.

Mounted vehicles select a conventional trailing camera. It reads authoritative
chassis position, yaw, pitch, and velocity, then applies separate exponential
position and facing responses. Speed adds a small configurable pullback and the
camera is kept above the shared analytic ground surfaces. This spring is a
camera filter only: it writes no body pose, force, contact, or control channel.
The neutral camera ray is constrained to intersect the chassis/look-ahead
target. Chassis yaw advances the base camera heading; pointer or right-stick
yaw/pitch is retained as a relative free-look offset, so steering cannot make
the vehicle drive out of its own camera. A raised blue cabin over the amber
chassis provides a visible fallback silhouette independent of wheel shading.

WebGL2 now performs a half-viewport `DEPTH_COMPONENT24` camera prepass in
addition to the existing five-layer light-space shadow depth array. The camera
depth texture is retained as an explicit renderer resource for later
depth-aware effects; it does not participate in collision or masquerade as the
compiled contact kernel.

Four round cylinder wheels, four steel frame members, and four suspension
struts now live in the guaranteed main scene mesh, alongside the body and
cabin. This corrects a mount-scope error that referenced an undefined `config`
before any vehicle geometry could be appended. Steering and suspension change
wheel transforms; authoritative angular velocity advances alternating tread
bands without changing the round wheel silhouette. Struts change length and
use grip/limit/slide colors. These parts remain non-colliding presentation
geometry; tire contact continues to come exclusively from the compiled
contact-patch lanes.

The wheel presentation now uses the same signed longitudinal convention as
the worker and chassis graph: positive local X is the front axle. The physics
kernel was already applying steering to `front_left` and `front_right`; only
the rendered local centers had inverted front and rear, which made correct
front-axle steering look like rear steering.

## Compiled torque graph and mechanical cutaway

The former authored forward/reverse torque constants have been removed. The
strict vehicle JSON now specifies displacement, BMEP, combustion/clutch/
driveline efficiencies, forward and reverse gear ratios, final-drive ratio,
engine rotating inertia, engine mass, and engine position/orientation. The
symbolic vehicle program derives indicated four-stroke torque and publishes an
explicit path:

`engine -> clutch -> selected gear -> final drive -> front/rear differential -> half-shafts`

Front and rear differential torques use the configured axle split, and each
wheel lane still applies its own half-shaft fraction and traction-control
scale. The program also publishes engine acceleration torque and angular
acceleration. Engine orientation defines its crank axis in the chassis frame.
World acceleration and external contact torque are projected into that frame;
engine offset and mass produce a local mount moment through `r x (-m a)`.
Crank and mount reactions are then added directly to chassis roll/pitch/yaw in
the compiled Wasm transition, while tire forces remain the external force path.

The torque graph is data in the vehicle model, not merely a diagram. Its live
channels cross the worker snapshot ABI to both an instrument panel and the
renderer. The presentation is now a mechanical cutaway: low floorpan, driver
seat, steel roll cage, engine, clutch, transmission, center driveshaft, front
and rear differentials, half-shafts, suspension struts, and wheels. Powertrain
component colors respond to their own compiled torque channel. Presentation
geometry remains non-authoritative and non-colliding.

The terrain renderer combines a strong world-anchored grid with per-cell
checker and height coloring so slope, speed, trench depth, and berm curvature
remain readable. Hotbar slot 10 carries a depth-map tool: sculpt mode lowers
with primary and raises with secondary; middle mode relaxes samples toward the
authored middle height with primary and grows the world texture/brush scale
with secondary. Every edit rebuilds the same sampled collision field. The
vehicle monitor also carries a `RIGHT CAR` control that lifts the body, retains
yaw, clears roll/pitch and angular velocity, and sends recovery through the
authoritative worker.

Steep sampled cells exposed a contact-reach defect: the worker discarded a
surface above `bodyY + 0.08` even when it remained inside the suspension's much
larger declared reach. Surface acceptance now uses the full reach. After each
compiled pose transition, four fresh wheel samples enforce the mechanical
suspension travel stop derived from linkage geometry. This prevents a fast
wheel from crossing a terrain cell beneath maximum compression while retaining
spring force, pitch/roll torque, and ordinary crest separation.

The visual chassis no longer submits the floorpan or driver-seat blocks. Frame
and cage members are silver, suspension links remain yellow, and tires and all
drivetrain members are black. The engine is represented by crank/bank rails;
the rest of the powertrain is clutch/gear shafts, driveshaft, differential and
half-shaft members. Engine and transmission mount crossmembers visibly suspend
that graph from the frame. These members read compiled torque channels, while
the equivalent crank/mount reactions remain authoritative in the chassis-local
Wasm torque reduction.

Mounting transfers the Springtail item out of inventory and into world custody
at the actor's current horizontal position and sampled support height. A
clickable top-down marker follows the active or parked vehicle; `V` mounts or
dismounts it. Selecting a tool no longer ejects the driver. Vehicle steering
continues to use the movement axis, while pointer/right-stick yaw and pitch
remain a free-look and tool-aim channel. Projectile tools originate at the
chassis camera hard point and inherit chassis velocity rather than firing from
the trailing camera.

The dynamic vehicle upload is deliberately not a scene transaction. It updates
the GPU vertex buffer at 30 Hz without republishing DOM geometry, rebuilding
portal topology, or resending static colliders to the physics worker. This
fixes the mount-time regression where presentation animation repeatedly paid
the full static-world synchronization cost.

The page's animation callback is also now failure-preserving. Vehicle stepping,
the chase camera, map marker, and wheel shader report their stage and error in
the viewport readout; a wheel-presentation fault disables that draw only, and
the outer callback always schedules the next frame. A mount-specific exception
can therefore no longer impersonate a performance freeze by silently killing
the sole animation loop.

## Sampled off-road surface

The courtyard no longer contains the former gradient test wedge. It now holds
a 49 by 33 sampled terrain grid whose inner elliptical band is a depressed mud
trench and whose adjacent outer band is a tall berm/hill. Two triangles per
cell define both the visible mesh and the piecewise-planar height/gradient
sampler. There is still no ramp or vehicle-terrain physics mode.

Inside the sampled terrain domain, this surface replaces the flat world-floor
candidate so the trench is a real depression rather than a drawing hidden by
an invisible floor. The shared platformer/projectile/vehicle/rigid-body contact
ABI consumes its height and normal. Gravity, contact pressure, friction,
wheel torque, and chassis pose decide whether the car climbs the berm, follows
the trench, slips, leaves the surface, or lands.

The shared surface contract also publishes an 8 m thick world-bottom rejection
volume from `y=0` to `y=-8`. This is an emergency safety manifold, not another
floor material or a vehicle-only contact mode. After an ordinary compiled body
step, the common worker rejects any body contained in the volume—or any swept
step that crossed completely through it—back to its shape-specific support
boundary and removes only inward vertical velocity. This closes the former
vehicle failure where the analytic floor sampler stopped accepting the floor
after the chassis had already fallen below its contact reach.

## Compilation boundary

The vehicle now expands its compact JSON dimensions into one explicit
`abstract-ui-mechanical-wrench-graph-v1`. It has 60-plus named nodes and edges
covering the triangulated lower frame, roll cage, four suspension corners,
steering tie rods, tire patches, engine, clutch, transmission, prop shaft,
differentials, CV half-shafts, and six-axis powertrain mounts. Every node owns
both a force and a moment vector. Edges declare rigid-distance, rigid-offset,
spring-damper, steering-link, pneumatic-contact, torque-shaft, CV-shaft, or
six-axis-mount laws.

Each suspension corner is a graph, not a vertical presentation post: paired
upper chassis pickups converge on an upper spherical ball joint, paired lower
pickups converge on a lower ball joint, a fixed-length upright connects them,
the hub is constrained to the upright, the coilover connects the lower joint
to its chassis hard point, and the contact patch sits below the hub. The host
constraint projection solves those linked node positions from the one retained
corner travel coordinate. Both tire location and visible link endpoints consume
that same solution. Contact force enters at the solved patch, so the compiled
chassis torque is the patch's actual `r × F`; coilover axis angle supplies a
bounded instantaneous motion ratio to both the parallel WGSL contact lane and
the scalar Wasm spring result.

This is a reduced-coordinate constrained mechanism, not a second independently
integrated collection of loose point masses. That distinction is intentional:
it preserves the lockstep rigid chassis while making the suspension geometry
causal. A future compliant-frame mode may give selected graph nodes their own
mass and integrate member strain, but it must reduce the same node wrenches and
must not create a second asynchronous vehicle clock.

## Conserved mass, center of gravity, and design spring loads

The configured 620 kg is now one conserved ledger. It is not 620 kg plus the
declared parts: 142 kg belongs to the engine, 58 kg to the transmission, 18 and
20 kg to the front and rear differentials, and each corner has a distinct 14 kg
wheel/hub/brake assembly plus a 12 kg pneumatic tire. The remaining 278 kg
belongs to the frame, cage, driver, and unmodeled equipment. The Python compiler
derives a local center of mass of approximately `[-0.033, 0.072, 0] m` and
roll/pitch/yaw inertia with the
parallel-axis theorem. Those derived inverse inertias are the Wasm chassis
parameters; the former uniform-box estimate is gone.

Contact-patch torque arms are measured from that derived center of mass. The
rearward engine position therefore produces the expected 47.34/52.66 front/rear
static axle split through ordinary contact moments rather than a hidden steering
or suspension correction. The JSON's configured 47.4/52.6 split agrees within
0.001. The mechanical graph publishes each corner's supported kilograms, static
newtons, linkage motion ratio, and design spring compression. The mounted HUD
shows design kg/kN beside the live spring kN, compression, patch area, and tire
utilization, making an overloaded or unloaded spring directly inspectable.

The wheel presentation consumes the same solved hub state and JSON dimensions as
contact physics. Rim radius belongs to `wheels`; tire radius, pressure, and width
belong to `tires`. The generated mesh publishes distinct semantic spans for a
0.42 m tall by 0.24 m wide balloon carcass/sidewall/tread and for its 0.23 m
six-spoke wheel, hub, and brake rotor. Wheel and thick-annulus tire masses derive
0.847 kg m2 of rotational inertia per corner, which the compiled wheel angular
equation consumes. The old presentation-only full-width axle bars were removed;
the graph's four differential-to-hub CV edges are the sole half-shaft geometry.
This adds no presentation-only wheel pose or second clock.

Contact area remains the compiled pressure/load result. The host derives the
visible footprint width from 65–85% of declared tread width and length as area
divided by that width, yielding the short, wide footprint of a balloon tire.
The configured lateral and longitudinal carcass stiffness remain independent
inputs to the friction ellipse.

The roll cage now participates in ground contact. Each emitted cage node and
each cage-member midpoint samples the same declared terrain surface as the tires.
Penetration and point velocity produce a bounded normal spring/damper force plus static/kinetic Coulomb friction;
the force and its center-of-mass `r × F` moment enter the lockstep chassis inputs
before Wasm integration. A bounded post-step projection catches numerical
tunneling but does not supply propulsion. Consequently an inverted car rests and
rolls on its cage instead of sinking into the terrain and gliding on a hidden
horizontal proxy.

The powertrain reaction audit also removed a double count that could make the
car look gimbal-mounted. The prior equation applied full combustion torque to
the chassis after most of that torque had already crossed the clutch and become
wheel/contact force, and its default crank axis disagreed with the longitudinal
engine-to-transmission graph. The mount now reacts only the crank's net angular
acceleration torque, along the graph's longitudinal shaft. Component inertial
mount loads use specific acceleration (acceleration minus gravity), so a freely
falling vehicle does not invent engine-mount load while a supported/accelerating
vehicle does transmit the appropriate force and moment into the frame.

## Transmission state and crawler policy

The transmission is a worker-owned state machine in the same 120 Hz lockstep
loop as tire contact and chassis pose. Its JSON schedule has six forward ratios:
first is a genuine `6.4:1` crawler, second is `3.1:1`, and automatic mode starts
in second. Reverse has its own `5.6:1` ratio. Gear, displayed gear (including
reverse), and automatic/manual mode cross the presentation boundary in one
snapshot record.

Automatic shifting is load-aware rather than a speed-only ladder. The worker
estimates available wheel torque from indicated engine torque, the candidate
gear, final drive, clutch efficiency, and driveline efficiency. It compares
that with the previous lockstep contact-patch longitudinal force reduced to
wheel torque. Upshifts require the next gear to retain configured torque
reserve. Higher gears downshift for insufficient reserve or low shaft speed.
Second enters crawler first only under substantial throttle, low road speed,
and inadequate torque reserve; ordinary starts therefore remain in second.
`AUTO`, gear-down, and gear-up buttons write commands to the worker. A manual
gear command selects manual mode; `AUTO` returns authority to the load-aware
algorithm without starting a parallel controller.

## Initial driving presentation and compact controls telemetry

The vehicle slot declares Springtail as the initial mounted vehicle at player
spawn. The browser host removes its inventory item through the same mounting
operation used later, enters the existing full-viewport shader layout, retains
`V` dismount and tool/free-look control, and registers the vehicle in the normal
lockstep world worker. Browser security forbids script-initiated fullscreen
before user activation, so the first non-Escape key or pointer gesture requests
true browser fullscreen; the page is already full-viewport before that gesture.

Mounted telemetry defaults to a compact four-corner overlay. Every wheel shows
contact mode, live kN, friction utilization, and separate TC and ABS intervention
percentages with proportional color bars. Intervention is `1 - command scale`:
0% means no controller reduction, while 92% means the configured 8% minimum
torque/brake command remains. `STATS` expands the same overlay to retain total
mass/CG, design kg and kN per spring, live compression, physical footprint
dimensions, chassis-load diagram, and drivetrain torque graph without occupying
that screen area during ordinary driving.

- JSON strictly validates mass distribution, geometry, tire pressure and
  friction, pneumatic suspension coefficients, engine and driveline parameters,
  slip-derivative traction/ABS parameters, and controls.
- SymPy equations compile through the repository ProcessGraph/SSA path to the
  four-lane WGSL contact kernel.
- SymPy chassis/wheel equations compile through repository SSA to scalar Wasm.
- Python builds the vehicle slot, stick/ball structure, deployment plan,
  snapshot ABI, diagnostic model, and final HTML model.
- JavaScript remains the browser/worker host adapter and currently owns surface
  sampling, GPU buffer submission/readback, force reduction, and presentation.

## Spectral Analyzer BVH isolation

`src/compiler/abstract_ui_bvh.py` isolates Spectral Analyzer's deterministic
median-centroid triangle BVH layout (`lo.xyz,left; hi.xyz,right; start,count`)
as a small Python compiler utility with stable ordering and provenance. It does
not import the renderer or optical material catalog. This is the preprocessing
contract for arbitrary-mesh contact, but the current live vehicle still samples
sampled/analytic height fields and generated box tops: BVH traversal and closest-feature
contact generation are not yet wired into the live WebGPU kernel.

## Remaining risks

1. GPU outputs are mapped back to the CPU every physics tick. It is lockstep
   correct but may not be a performance win. A GPU-resident reduction stage and
   buffered readback should be measured before adding more asynchronous work.
2. Arbitrary triangle meshes need compiled BVH traversal, closest point or
   swept contact generation, stable feature IDs, and physics material lookup.
3. The chassis solid-contact fallback is still a circular horizontal proxy.
   A vehicle-ready oriented manifold needs multiple contacts, friction, and
   restitution without replacing the tire patches.
4. Euler pose is sufficient for ordinary crawling and jumping, but unrestricted
   tumbling eventually needs quaternion integration and normalization.
5. The no-WebGPU path spins wheels and integrates gravity through compiled Wasm
   but does not yet run the same tire-contact equations locally. It must compile
   the contact SSA to a scalar/batched Wasm fallback before that path is called
   physics-equivalent.
6. The camera depth texture is produced and retained, but no post-process or
   material currently consumes it. Depth-aware spray, contact shadow, and
   occlusion cues should be separate renderer passes rather than physics rules.
7. The gearbox has a worker-owned six-ratio shift state, but clutch slip, an
   engine-speed torque curve, differential speed constraints, and limited-slip
   behavior remain future laws for the same compiled graph.
8. The graph currently reduces contact and powertrain paths to the equivalent
   chassis wrench; it does not yet publish axial force for every individual
   wishbone and cage member. Adding member-load telemetry should solve graph
   reactions from the same constraint Jacobian, not infer colors from spring
   compression.
