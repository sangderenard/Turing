# Continuation: orbital craft step 7 (fuel-burn mass loss, attitude)

Owner files: `engine_toy/orbital_jumper.py`, `engine_toy/orbital_actuation.py`,
`engine_toy/tests/test_orbital_jumper.py`, `engine_toy/tests/test_orbital_actuation.py`.
Not touched: `orbital_tracker.py`, `orbital_plan.py`, `orbital_collocation.py`.
Uncommitted.

## 2026-10-03 survey findings

- Catalogue laws available (engine_toy/honorary_engine_equation_catalogue.py):
  N1.3 `dR/dt = R skew(omega_B)` (rotation-matrix attitude), N1.4/N1.5
  (orthogonality, det 1), N1.6 Euler rigid-body equation, N7.2 variable-mass
  momentum, N10.1 point-mass inertia tensor, TS1.2 `F = m_dot c`,
  TS1.4 `c = I_sp g_0`, TS2.1 Tsiolkovsky delta-v. No `tau = r x F` law in the
  catalogue (only dipole torques).
- Machine-sim convention (turing/src/compiler/abstract_ui_vehicles.py):
  `mass_properties()` = residual mass as a uniform box `m(a^2+b^2)/12` plus
  declared components as point masses (diagonal only, principal axes = body
  axes); `inverse_inertia_{roll,pitch,yaw}` columns; torques as
  `cross(position, force)`; rate `w += dt*tau/I` then angle `+= dt*w_next`.
  The angle form is the one memory records as the vehicle energy injector
  (wrong rate axes); its own comment defers "a full quaternion lowering".
  engine_mass_properties.py: N10.1 full tensor from point masses.
- Decision taken: attitude is the catalogue's own state, R (9 columns), per
  N1.3; inertia/torque reuse the vehicle convention (box + point masses,
  diagonal, torque = r x F).

## Plan

Pieces (sequential): actuation -> N4.1 gravity -> N7.2/N1.6 momentum+mass+rate
-> N1.1/N1.3 position+attitude -> thrust cost.

## 2026-10-03 implementation written (not yet compiled)

- orbital_actuation.py: THRUSTER_KINDS (ideal = inf Isp, cold-gas 70 s,
  monopropellant 230 s, bipropellant 310 s), STANDARD_GRAVITY_M_S2 (g_0 of
  TS1.4). Thruster gains mass_kg (point mass); CraftDesign gains
  propellant_kg and body_size_m. inertia_tensor/principal_inertia (box +
  N10.1 point masses; refuses non-principal designs). actuation_matrix(design,
  attitude=None) = R @ B_craft; torque_matrix; propellant_flow_per_throttle.
  Laws: propellant_demand (TS1.2 solved for m_dot, 1/c column),
  propellant_supply_rhs = Piecewise(max(0,min(1,P/(D dt))), D>0; 1) -- the
  exact step average of H(P) at demand D; actuation_force_rhs now world-frame
  (R @ ...) times supply; actuation_torque_rhs = sum r_k x F_k.
- orbital_jumper.py: pieces actuation(t) -> gravity(c, unchanged) ->
  momentum (N7.2 outflow share -flow*p/m, propellant/mass balance, N1.6 with
  diagonal I) -> position (N1.1 + N1.3 by Cayley map) -> cost (delivered
  thrust). Discrete identity: v_new = v + dt F / m_new exactly.
- Symbolic check (no compile): Cayley step orthogonal to 1.1e-16, rotation
  angle 2 atan(dt|w|/2) about w as designed; N1.6 spelled gives the standard
  Euler equations.
- OPEN: attitude publishes nothing to the dt controller. Rotation does not
  bound dt (max_vel is translational; dx is the orbit scale). A spinning craft
  in orbit at dt ~ 22 s would step ~2 rad per substep (orthogonal, but the
  angle is 2 atan(dt w/2), not dt w).

## 2026-10-03 first compile: two findings

1. MY BUG: the actuation piece read the OLD `propellant_supply` column (a
   piece's rhs reads inputs; its own `_next` lands after). Fix: the supply
   becomes its own piece ahead of actuation.
2. COMPILER DEFECT (seconds-long repro): `Piecewise((x, x > 0), (1, True))`
   through `equation_piece` returns 5e-324 (integer 1's bits read as f64)
   when the Integer branch is selected; `(sp.Float(1.0), True)` returns 1.0.
   Repro: equation_piece("repro_pw_const_a", (Eq(y_next, Piecewise((x, x>0),
   (1, True))),)) called with x=-2.

## 2026-10-03 compiler defect FIXED at its identity

- Identity: `symbolic_equation_compiler._numeric_constant` converted SymPy
  Integer/Rational/Float payloads to float but not Python `int`. ProcessGraph
  spells the singletons One/Zero/NegativeOne as Python ints (dumped:
  `NegativeOne -1 <class 'int'>`), so a node the module declares float64
  kept an int payload; the LLVM value-slot store then emitted `store i64 1`
  into a slot read back as double (5e-324; -1 would read as NaN). Zero only
  "worked" because 0's bits are 0.0.
- Fix (turing/src/compiler/symbolic_equation_compiler.py): ints (not bools)
  become floats in `_numeric_constant`. Repro after fix: Piecewise((x, x>0),
  (1|-1|0, True)) at x=-2 -> 1.0, -1.0, 0.0.
- Caveat: pieces already cached in __llvm_lawcache__ compiled before the fix
  keep their artifacts (cache key = equations, not compiler).

## 2026-10-03 supply split into its own piece; actuation tests

- tests/test_orbital_actuation.py: 8 passed (122.7 s incl. compiles). The
  cold-gas fixture in test_mixed_throttles now declares propellant_kg=50.

## 2026-10-03 step-7 tests GREEN

- Empty step left propellant -1.1e-16 (P - dt*(P/(dt D))*D). Mass balance
  now spells the draw as min(P, dt*flow): same quantity, rounding removed.
- tests/test_orbital_jumper.py: 7 passed (31.6 s). Measured:
  - burn (biprop 400 N, 50 s): burned 6.578814277277 kg = F t/(Isp g0)
    6.578814277277 kg; dv 20.079328704 vs TS2.1 20.066078113 (excess
    1.3e-2 <= right-Riemann bound 2.6e-2).
  - empty (cold gas 100 N, 2 kg): propellant 0.0 exactly, supply 0.0,
    delivered impulse 1372.931000000 = c P0; dv 4.6010 vs TS2.1 4.5918.
  - pure torque (couple 2 N m, I_z 41.667): omega_z 0.480000000000000 =
    tau t / I_z; turned 2.4585 rad = sum 2 atan(dt w_n/2) (continuum 2.4);
    |R^T R - I| 8.9e-16.
  - thrust follows attitude: Rz(90) +x thruster -> (0, 250, 0); arbitrary
    R -> R @ B_craft @ u; after the spin, main engine force = R_before @ x.
  - step-1 orbit unchanged: radius 1.79e-3, energy 1.27e-5, 1779 substeps
    (same as ORBITAL_CRAFT_STEP2_CONTINUATION.md).

## 2026-10-03 tracker compatibility + handoff

- tests/test_orbital_tracker.py (read-only run, not edited): 3 passed, 56 s;
  LEO->GEO numbers: final |r - r_ref| 63.1 m, fuel 1.2416 x ideal.
- Tracker change NEEDED to use attitude (not made; other owner):
  orbital_tracker.py:101 `B = actuation_matrix(design)` should be
  `actuation_matrix(design, craft.attitude())` (or `craft.actuation_matrix()`)
  once the craft can rotate; the allocation also does not see the propellant
  supply (an empty tank makes B effectively zero) and has no torque channel
  (`torque_matrix(design)`) to hold attitude.
- OPEN QUESTION for the user: what bounds dt for the attitude? Nothing the
  rotation pieces publish reaches the controller (max_vel is translational).
