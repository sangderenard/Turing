2026-11-09 STUDIO NATIVE PRODUCT BOUNDARY (MEASURED, canonical Windows,
base f998b80f, source only; supersedes the earlier foundation status only for
the bounded work described here):

fountain3d.make_law now has optional environment and dt parameters. An
environment replaces the old chamber walls/SDF and containment diagnostics,
not overlays them. It calls the incumbent wall closure and retains the same
particle neighbor walk, force, friction, spin, integration and diagnostics.
Default arguments preserve the old ABI and operation order. A separate bounded
comparison against HEAD's actual _contact and make_law ASTs held every output
bit-identical for two contact-rich steps in flat, SDF and hinged-body modes.

studio_environment.py admits exact room bounds, independent closed convex
parts and filled convex planar sheets. All authored vertex coordinates remain
runtime scalar inputs; triangle ordering, part/body IDs and material IDs are
retained. Optical materials are not converted into friction. Exact rational
predicates over the actual float32 vertices reject tiny retained folds,
nonconvexity, zero-volume closed shells, open boundaries and inconsistent
winding. Native float32 contact-area underflow/overflow is rejected too.
Repeated compound body IDs refuse until union ownership is implemented.

studio_physics_build.py explicitly emits a genuine complete product through
the existing SSA/WebGPU compiler and optionally the existing graph_wasm route.
There is no request-time build, server import, startup hook or publication.
The two-triangle tabletop plus six room planes was emitted with the full
native neighbor/contact/integrator law and complete named state ABI. An actual
emitted manifest passed unchanged ComputeRuntime.sync/load/reset, with a
device transport double. This is not GPU kernel execution. The manifest
retains the classifier verdict/cost model and source topology/frame/digest.

The shared frame/pose module web/marble_studio_physics.js uses the existing
marble_studio_scale.js transform. Native and Studio axes are both Y-up, with
one physical metres conversion and a radius-independent translation above the
native parked-origin/grid-floor sentinel. Geometry never fits itself to a
particle extent. Fixed rigs remain fixed while marble radius changes. Starts
carry position, orientation, velocity, spin, age and physical radius. An exact
pauseAtDisplayedState result returns the same pose, without camera reframing.
The old toy's quarter-turn coordinate conversion is not used here. A bounded
one-second replay helper scales its native-step budget with product dt, up to
5760 steps; other games retain the existing default 360-step allowance.

The physical-size range required a timestep contract, not a gravity/room
scaling shortcut. physical_step selects power-of-two smaller fixed steps and
native spring stiffness for at most 1% static gravitational compression,
retaining the incumbent omega*dt allowance. Metres and seconds stay real;
gravity is 9.81 m/s^2. Both native friction impulse limits receive the same dt,
as do particle/body integration, spin, age, noise, drag and swelling. Manifest,
WASM law, source digest and settling sample timing agree. Equilibrium probes
at radii 0.1, 1, 10 and 200 mm pass. This remains compliant DEM, not hard-body
CCD or a proof of every fast impact/multiple-contact configuration.

Studio products are capped at 12 active marbles, the incumbent exact neighbor
walk capacity. For a very large room/radius ratio, only the SEARCH cell size
is coarsened to preserve exact float32 cell keys. Every potential particle
candidate still fits the bounded walk; radius/positions/forces do not change.
A three-particle coarse/fine-grid parity check retains identical next-state
columns and no dropped neighbors. Larger active counts are explicitly refused.

Final gate: 35 passed in 38.55 seconds:
  Python311 -m pytest -q tests/test_meowdoku_studio_environment.py
    tests/test_meowdoku_studio_contacts.py
    tests/test_meowdoku_studio_convex_contacts.py
    tests/test_meowdoku_compute_state_js.py --tb=short
Node shared-frame/full-pose, physical angular integration, preview motion and
replay-layer checks also pass. Course worker's 10 exact-topology/grade cases
pass separately. Independent review found three admission/manifest issues
(missing cost_model, zero-volume shells, tolerated folds), all fixed and
verified. The final step-propagation review found no additional blocker.

RELEASE GATE REMAINS CLOSED. This is not complete user-visible integration:
- Studio startSim still selects its old toy product. No new imports, handlers,
  save schema, selector, server endpoint, or live/photo pose wiring were added.
- Whole-scene authored-part extraction/admission still must connect the exact
  room/table/gobo/backdrop/fixture snapshot. Arbitrary nonconvex housings and
  compound courses cannot be relabelled as independent convex proxy bodies.
- Planar ownership supports the safe face case, but nonconvex concave edges,
  folds, compound-solid inside/exit semantics and broadphase mesh capacity
  remain unresolved. The authored courses keep collisionReady=false.
- Scalar vertex ABI/unrolled geometry is proven only at bounded topology.
  Full-stage dispatch/memory/performance and native GPU execution are untested.
- Actual static/physics/preview/photo/live transitions, motion-control axis
  adaptation, fast impacts and optional reciprocal Pluck player-body coupling
  need integration tests before activation. Existing Pluck clipping is not a
  reciprocal rigid-body solver.

No active GPU job, server restart, default build, export, publication, commit
or push was performed. Editing fountain3d.py changes its whole-source build
digest and will invalidate default product caches on a future build. Keep this
source batch separate from the already-frozen visual release until deliberate
integration and execution checks are complete.
