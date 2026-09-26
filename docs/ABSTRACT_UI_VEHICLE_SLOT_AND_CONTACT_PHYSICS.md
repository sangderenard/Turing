# AbstractUI vehicle slot and shared contact physics

Date: 2026-08-25

## Decision

A player may own one neutral `vehicle-slot`. Mounting a vehicle changes how the
same movement inputs are interpreted; it does not turn the car into a tool or
put terrain traversal behind a car-name branch. The first occupant is the JSON
configured `Springtail` car in hotbar position 9.

The source configuration is `configs/vehicles/fun_car.json`. It carries mass
and corner mass fractions, chassis and wheel attachment geometry, suspension,
tire inflation pressure, patch bounds, static and kinetic friction, load
sensitivity, longitudinal/lateral stiffness, controls, fixed-step policy, and
presentation roles. The canonical JSON digest is embedded in the page model.

## General solid-contact contract

There is no ramp physics category. A wedge, rolling hill, or flat slab may
publish `abstract-ui-contact-surfaces-v1`, the same height/gradient contact
contract consumed by platformer bodies, projectiles, rigid bodies, and vehicle
patches. Shape reach and separation determine candidate contacts; stable runtime
part ID breaks equal-height ties. The gradient supplies a normal, while weight,
gravity, penetration, damping, and material friction determine the impulse.
Leaving the surface domain simply removes that impulse and preserves momentum.

The courtyard demo is a 49 by 33 sampled mud-oval height field with a trench
and adjacent berm. It is not detected as a ramp or by vehicle kind. Its mesh
triangles and piecewise-planar contact sampler share the same samples. A future
triangle/BVH sampler must produce the same ABI: height, gradient/normal,
identity, separation/reachability, and generation.

## Baked parallel contact kernel

`symbolic_wheel_contact_equations()` is a seven-output SymPy equation set. It
lowers through the canonical ProcessGraph and repository SSA at float32, then
through the existing WebGPU backend into WGSL. The launch is baked for four
wheel lanes in one `(4, 1, 1)` workgroup.

Each lane independently calculates:

- suspension compression and compression rate;
- spring/damper normal load;
- pneumatic contact area `clamp(load / pressure, patch_min, patch_max)`;
- pressure- and load-sensitive friction limits;
- longitudinal and lateral requested force;
- a continuous transition from the static limit to kinetic Coulomb friction;
- force at the chassis attachment and its cross-product torque.

The seven output buffers are chassis force XYZ, chassis torque XYZ, and contact
area. Together with the packed feed buffer this intentionally fits WebGPU's
portable eight-storage-binding floor. Wheel reaction force is defined as the
exact negative of chassis force rather than spending three more bindings on
duplicated values. A chassis reduction sums the four paired-force records.

The existing 120 Hz world-physics worker is the single tick coordinator. It
selects mesh contacts, submits WebGPU storage buffers, waits at an explicit
stage barrier, reduces the returned forces, invokes the compiled scalar Wasm
chassis transition, and only then publishes the shared pose snapshot. A
`tickInFlight` guard prevents overlapping async GPU submissions from becoming
overlapping physics ticks. When WebGPU results are live, their contact forces
replace the scalar spring contribution and provide tire traction and attachment
torque; the scalar Wasm contact law is the lockstep fallback.

The preferred path creates the compute pipeline inside that worker. Browsers
that expose WebGPU only on the page use a narrow JavaScript bridge instead: the
worker sends one four-lane contact packet, the page dispatches the same baked
WGSL, and the worker waits for its result before completing that tick. This is
coordination across JavaScript, Wasm, and shader stages, not a second async
simulation schedule. A bounded timeout selects the Wasm fallback rather than
allowing the world clock to fork or a late GPU result to mutate a later tick.

The vehicle also has a JSON-configured rigid-body contact layer for chassis
collisions with solid world objects. It resolves penetration, normal velocity,
restitution, and static/kinetic tangential friction after the tire-patch stage
and before snapshot publication. This layer is distinct from the tire law:
tires remain four pressure-sensitive suspension contacts; chassis impacts are
ordinary solid-body contacts.

While mounted, a generated diagnostic schematic displays all four tire patches
and chassis springs. Patch area controls ellipse size; green/amber/red/gray
encode static grip, friction limit, kinetic slide, and separation. Each corner
also reports area, load, friction demand, and suspension compression.

## Audit and remaining risks

This improves the compilation boundary but does not finish it:

- WebGPU dispatch/readback and chassis reduction are still handwritten worker
  orchestration and are honestly classified as bespoke semantic runtime.
- The current surface adapter samples declared analytic height fields and flat
  collider tops. Arbitrary mesh
  support requires a compiled triangle/BVH contact stage behind the same ABI.
- The initial tire law is deterministic and continuous, but not a Pacejka
  empirical fit. Temperature, carcass relaxation, camber, differential speed
  constraints/limited slip, anti-roll coupling, and aerodynamic downforce
  remain explicit future parameters/stages. Wheel angular inertia and the
  open front/rear differential torque path are compiled today.
- GPU readback is suitable for proving lockstep coordination, not the final
  fast path. The next deployment should keep wheel results GPU-resident and run
  the chassis reduction as a second compute stage behind the same tick barrier.
- Vehicle state and mount/dismount transitions must eventually enter the
  authoritative edit/event reducer identified by the broader Living Data Map
  audit.

The architectural acceptance rule is that a new vehicle configuration may
change JSON and be precompiled into the same kernel/package interfaces without
adding a browser branch for its name or kind.
