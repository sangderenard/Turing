# Vehicle model compilation contract

## Model and parameters

The model is the authored vehicle equation. Its named inputs are the parameters
supplied by JSON. Compiling the model must preserve those parameter
slots; a vehicle or engine profile assigns parameter values and does not create
a second equation or hide the values as compiler literals.

Equation constants such as π may be represented in two binary64 limbs. They
are constants in the equation, not JSON parameters. JSON parameter values stay
live at every native and browser ABI boundary and are never constant-folded.

Native simulation/training and browser simulation share the model equation and
the JSON parameter ABI. The native scientific path additionally compiles the
model's inverse graph and AbstractNN Adam in Python/AbstractTensor form, wraps
that complete callable in two-limb precision, and only then AOT-compiles it to
C. Browser WASM is forward-only and contains no inverse graph, Adam update, or
optimizer state.

The native rig begins as an unequipped structural mockup, but that is only its
first qualification stage. Installing a body, engine, clutch, transmission,
fuel/electrical system, armor, weapon, accessory, bumper, or ballast changes
the runtime parameter record and must recompute mass, center of mass, inertia,
clearance, collision geometry, and attachment wrenches before the same equation
is solved again. Passing bare quiescence never qualifies an outfitted vehicle.

Aerodynamic resistance is also part of that runtime ABI. Each configured body
supplies unit directions, drag coefficients, and reference areas; the equation
adds `-0.5*rho*Cd*A*abs(v dot n)*(v dot n)*n` for every direction. A body choice
may change those parameters without changing or recompiling the equation.

The tire is a closed, pressurized balloon-skin graph. The torus formula may
generate its compile-static rest mesh, but is neither its runtime collision
shape nor its force law. Every membrane vertex carries mass; every face uses
the same compiled hyperelastic strain energy and strain-rate dissipation;
closed-skin volume drives polytropic gas pressure; and both bead rings return
equal/opposite force and moment to the rim. Hard-surface contact is a
unilateral impulse at the deformed skin's actual crossing, never positional
rejection. Skin material, pressure, bead, mass, and contact values remain live
JSON parameters at every backend ABI.

## Collision and the backward transform

The backward transform is a local collision refinement tool, not the global
collision authority. Forward continuous collision detection discovers the
candidate skin-vertex/skin-edge versus hard-triangle branches and verifies the
accepted result. For a fixed branch, the inverse graph supplies exact
derivatives of signed distance, time-of-impact, membrane, and constraint
residuals. A Newton/SQP or complementarity solve uses those derivatives in
two-limb arithmetic. If the active triangle, feature, friction regime, or
separation state changes, the forward detector selects a new branch and the
solve restarts. Adam tunes persistent model parameters; it does not resolve
individual collision events.

## Learned tire deployment operator

The scientific balloon-skin graph remains the teacher and validation
authority. A deployment surrogate may replace only its declared boundary:
the complete periodic membrane/terrain field and canonical whole-vehicle state
map to the six-axis rim/hub wrench plus auxiliary pressure, volume-ratio, and
gas-temperature reconstruction lanes. The exact polytropic gas kernel remains
state authority; auxiliary thermodynamic error is a novelty signal. The input
includes temporal finite-difference orders zero through two, the rest skin,
explicit pressure/volume/temperature state, and 48 named vehicle values. Every
channel has a published physical normalization scale. All four live wheels
occupy the batch axis.

Learned tire deployment is GPU-only. A native CPU lowering exists solely as a
numerical parity oracle and fallback. Model selection first tests an augmented
full-state linear map, deployed as one resident tiled GEMM, then retains the
periodic convolution only if held-out nonlinear reference trials justify its
cost. Learned parameters remain JSON values with explicit dtype, shape, and
content identity; they are never frozen into the equation.

The authored skin is an inflated construction reference, not an unstressed
sheet. A conservative reference-face prestress potential cancels gas pressure
face-for-face at the reference pressure and geometry, preventing a startup
inflation impulse while retaining an auditable energy lane.

The forward network and functional Adam update are AbstractTensor programs.
A stored reverse graph is optional: compiled symbolic teacher derivatives and
AbstractNN layer derivatives may feed Adam's explicit gradient inputs directly.
The surrogate is deployable only while force/moment error, contact branch,
passivity, and held-out transient gates remain within their configured bounds.

Deployment uses an exact-first work-share state. `alpha=1` initially runs the
scientific tire teacher. Fresh periodic teacher trials measure normalized
six-wrench loss; sustained low loss lowers alpha slowly toward a nonzero audit
duty, while elevated loss raises it quickly. An accumulator converts alpha to
deterministic reference-call duty rather than evaluating both paths every
step. Maximum trial spacing prevents indefinite blind inference. Plastic or
damage activity and novel contact branches immediately force exact evaluation
and may restore full reference authority. When both answers exist their convex
mix cannot exceed the two supplied wrench answers merely because authority
changed.

For multiplayer, the compiled balloon teacher is server authority. The GPU
operator supplies client prediction and reconciles at the six-axis hub-wrench
boundary. Fixed stepping, ordered skin checkpoints, authority/model digests,
and exact-trial sequence state are part of the replay contract. Current
floating-point execution is tolerance-deterministic, not falsely claimed to
be bitwise identical across GPU vendors; bitwise rollback requires a later
reproducible transcendental or quantized authority path.
