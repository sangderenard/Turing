# Continuation: orbital craft as an engine_toy machine

Owner files: `engine_toy/orbital_craft_machine.py` (new),
`engine_toy/orbital_actuation.py`; wiring in `engine_toy/orbital_jumper.py`
(shared with the attitude-dt-bound agent: re-read before every edit).
Not touched: `orbital_tracker.py`, `orbital_game.py`,
`orbital_collocation.py`. Uncommitted.

## 2026-10-03 coordinator finding 1: thrusterless jumper "fails to compile"

- Does NOT reproduce on the current compiler. Fresh-cache compiles of all six
  `orbital_jumper_dt_pieces(0, 0)` pieces (and `(1, 0, batch=4)`) succeed;
  `orbital_craft_actuation_t0` outputs are `raw_force_*` copies and
  `Integer(0)` (torque, flow), the supply is `Integer(1)`. A
  `OrbitalJumper(mass_kg=1000)` round runs (0.7 s, cached pieces).
- The game note quotes the raise at `ssa_llvm_backend.py:6499`; the
  output-render check is at line 5959 today. The note was made against an
  older backend, before `aa5f1aac` (`_numeric_constant`: Python ints from
  ProcessGraph's One/Zero become floats), which makes those outputs double.
- Repros (scratch, seconds each): bare `Integer(0)` output, two zero outputs,
  copy + zero, `w*0 + x` with zeros -- all compile.
- Consequence for the game: stations can be plain `mass_kg` jumpers again.

## 2026-10-03 coordinator request 2: batch > 1

- `OrbitalJumper(..., batch=N)`: every column is one cell per lane, pieces
  built at batch N, one dt state, one shared dt (the dt system folds
  `max_vel` by max and `dt_limit` by min over lanes). Per-lane initial values
  take a leading lane axis (`position_m`/`velocity_m_s` (N, 3), `mass_kg`
  (N,), `attitude` (N, 3, 3)); one value broadcasts. Batch 1 keeps every seam
  shape; batch N readings carry the lane axis.
- Also added the two hooks the machine craft needs (`_inertia_kg_m2`,
  `_dt_pieces`); wiring only.
- Measured (tests/test_orbital_jumper.py, 4 stations, 5 rounds of 300 s):
  batch 4 = 455 substeps, 1.735 s incl. build; 4 singles = 1820 substeps,
  5.847 s incl. builds; worst lane difference 6.5e-9 m at 7e6 m.

## 2026-10-03 DEFECT (llvm_dt_system, not fixed here): states share one program

- Identity: `instantiate_state` calls `bind_pieces`, which writes the state's
  generated `PieceState`/`advance_pieces` into the MODULE globals;
  `dt_system_over` calls the global `advance_pieces`. So every persistent
  state runs the step program (and wall-cost ledger wrap) of the state
  instantiated LAST in the process. The state records `bound_pieces`
  (names, schedule) but not the program it was spelled for.
- Repro (seconds): build a batch-4 thrusterless jumper, then a batch-1
  jumper, then advance the batch-4 one: every lane's mass and momentum
  become lane 0's (`[1000, 2e4, 5, 3.3e5]` -> `[1000, 1000, 1000, 1000]`).
  In tests: `test_interleaved_states_each_run_their_own_program`
  (strict xfail; it will start failing when the defect is fixed).
- Why the game works today: the craft and the stations are all six-axis
  designs (same pieces), so the shared program is the right one; but each
  `substeps` reading counts into the last state's ledger. A batched station
  state next to a thrustered craft WILL run the wrong program.
- Fix at the identity (not made: `llvm_dt_system.py` is its own lowered
  source and the only guard is the `tests/dt_system/test_llvm_dt_system.py`
  lowering, a build): keep the namespace's `advance_pieces` on the state in
  `instantiate_state`, and have the Python driver (`dt_system`) bind that
  state's own program before `dt_system_over` runs, so the lowered entry's
  text is unchanged.

## 2026-10-03 coordinator item: r() returned position and velocity half a step apart

- The integrator is symplectic Euler (kick p by dt F(x_n), drift x by the
  new p/m): read as leapfrog, the momentum column is the HALF-STEP value
  while the position is whole-step. Fix, inside the scheme: (a) the initial
  momentum is seeded half a first substep back, `p0 = m v0 - dt_init/2
  F_grav(x0)`; (b) `r()` returns `(p + dt_last/2 F_grav(x)) / m`, the
  second half kick. F_grav is the COMPILED N4.1 gravity piece called on the
  host at the current position (applied force zeroed). Applied/thrust
  forces are constant per substep, and a left-sum kick of a constant force
  is exact, so they get no half-step term (thrust-only tests unchanged,
  bit-for-bit). With no centers nothing changes.
- Measured (test_circular_orbit_one_period_holds_radius_and_energy):
  radius error 2.06e-3 -> 6.6e-6, specific-energy drift 1.38e-5 -> 4.3e-9,
  1760 substeps. 16 jumper/actuation tests pass, 1 xfail (below).
- Remaining first-order seam: with varying dt the leapfrog kick should use
  (dt_n + dt_{n+1})/2; the piece uses dt_{n+1}. Not changed (it is the law
  in the momentum piece).

## 2026-10-03 coordinator item: dt never grows back after a spin (MEASURED)

- Repro (scratch dt_recover.py, seconds): spin craft (two-couple design),
  10 s rounds: spin up 3 rounds (w_z 1.44, dt_next 0.087 from the attitude
  dt_limit 0.125/|w|), despin 3 rounds to w_z = 0.000000, coast 4 rounds:
  dt_next stays 0.087006, 115 substeps every coast round (first round was
  1 substep). Confirmed: once the attitude bound shrinks dt it never
  recovers. Cause as read: no orbital piece publishes energy_j/power_w, so
  every participant is HOLD (do not grow). The machine-craft pieces publish
  none either. Not fixed: which energy/power the craft should publish
  (translational + rotational kinetic energy against thrust/torque power?)
  is the dt system's exchange-time question, open for the user.

## 2026-10-03 the llvm_dt_system shared-program defect (xfail test)

- Recorded above ("DEFECT (llvm_dt_system, not fixed here)"); the test is
  tests/test_orbital_jumper.py::test_interleaved_states_each_run_their_own_
  program (strict xfail). Seconds-long repro: batch-4 thrusterless jumper,
  then a batch-1 jumper, then advance the batch-4 one: masses
  [1000, 2e4, 5, 3.3e5] -> [1000, 1000, 1000, 1000]. Also: `substeps` of an
  earlier state counts into the LAST state's ledger (the wrap is the last
  state's), so r()'s first-step test uses time_s, not substeps.

## 2026-10-03 coordinator item: the allocate seam

- `MachineCraft.allocate(wrench)` (tracker shape: `.force_n` world,
  `.torque_n_m` craft frame about the CURRENT centre of mass) or
  `allocate(force, torque)`; applies throttles AND gimbal commands itself;
  returns orbital_actuation.Allocation (`.achieved` with `.force_n`/
  `.torque_n_m`, `.throttles`, `.gimbal_rad`, `.force_shortfall_n`,
  `.torque_shortfall_n_m`). Declared `MachineCraft.applies_allocation =
  True`.
- Tracker line to change (orbital_tracker.py, fly(), currently line 808):
  `craft.throttle(command.throttles)` ->
  `if not getattr(craft, "applies_allocation", False): craft.throttle(command.throttles)`

## 2026-10-03 machine pieces: two SymPy blow-ups, both fixed in my laws

- mass properties: the adjugate/determinant of the FULL symbolic tensor
  (charges, moments, 1/M) recursed past the limit inside ProcessGraph
  ingestion (`symbolic_process_graph.ingest_value` -> `make_node` -> str).
  The inverse is now its own piece over the plain `inertia_*` columns the
  mass-properties piece publishes (0.2 s + 0.0 s compiles).
- supply: building the hydrazine tank's `Piecewise((...), (demand > 0))`
  with sixteen nested deadband Piecewise inside `demand` recursed inside
  SymPy itself (`Piecewise` -> `cond.rewrite(ITE)` -> `atoms`), before
  any compiler code. The delivered (deadband) throttle is now a state
  column `thruster{k}_delivered`, written by the slew piece from the
  slewed state; supply/actuation/cost read the column.
- First build of the 19-thruster, 3-tank craft: 96.7 s (actuation 62.2 s,
  slew 11.5, supply 8.1, momentum 8.0, cost 5.4).
- Measured at rest (window 0.05 s): main throttle state 0.1, 0.2, 0.3 (no
  thrust: deadband 0.4), 0.4 -> 1600 N, 0.5 -> 2000 N; mass drop per round
  0.026315 kg = 1600 * 0.05 / (310 g0). The lateral force that appears is
  the craft turning: the CM is off the engine axis, so the untrimmed engine
  torques it.

## 2026-10-03 r() synchronisation: effect on the tracker and game tests (MEASURED)

- Coast check (scratch sync_probe.py; one circular period, 7000 km):
  uniform steps (1.3 s rounds, one substep each): old reading |r-R| 4911.8 m,
  |v_r| 10.58 m/s, energy 1.96e-6; synchronised 6.9 m, 0.0037 m/s,
  9.65e-13. Non-uniform steps (5 s rounds = 3.31 + 1.69 s substeps): old
  10468 m, 22.49 m/s, 8.9e-6; synchronised 2093 m, 6.61 m/s, 7.7e-7.
  The residual with non-uniform steps is the integrator's kick weight:
  variable-step leapfrog kicks by (dt_n + dt_{n+1})/2 F(x_n); the momentum
  piece kicks by dt_{n+1} F(x_n). Fixing that is a law change in the
  momentum piece (a carried previous-dt column), NOT made.
- tests/test_orbital_game.py: 7 passed. tests/test_orbital_tracker.py: 6
  passed, 2 FAILED with the synchronised r(); both pass with the half-step
  terms disabled (scratch baseline_tracker.py patches `_half_step_kick` to
  zero = the old read-out and seeding):
  - test_rotating_craft_slews_its_main_engine_and_arrives: propellant
    165.51 kg = 1.1185x ideal, final 334.5 m / 3.333 m/s (old) ->
    190.11 kg = 1.2848x (asserts < 1.15), 88.8 m / 5.548 m/s (new).
  - test_kick_off_plan_replans_once_and_arrives: final |r|-r2 -333.9 m,
    v_r 3.347 (old) -> -391.8 m, v_r 5.249 (asserts < 5.0) (new).
  Position error fell 4x; velocity error and trim fuel rose. Not
  diagnosed further (tracker-owned); suspects: the tracker's gains/bands
  were set against the old read-out, and the variable-step kick seam above
  (rounds clip the last substep every round).

## 2026-10-03 THE MACHINE CRAFT: built, tests green

Files: engine_toy/orbital_craft_machine.py (new), orbital_actuation.py
(Thruster fields + step-8 laws + allocate_wrench), orbital_jumper.py
(batch, hooks, r() sync), fluids.py (+3 rows), tests/
test_orbital_craft_machine.py (11 tests). Owned suites: 27 passed, 1 xfail.

- Document: turret_production.ProductionGraph wrapped in machines.Machine
  (production_graph). 3 x 4-node 6061 rings, longerons, ring members, bay
  diagonals; docking ring = port_role structural-mount; thrust structure;
  MMH/NTO/N2H4 drum tanks (part_role propellant-tank, fluid, capacity_kg,
  fill_kg; node mass = shell); 40 kg equipment bay on -y; main engine on a
  universal-joint (Cardan) + 2 linear-hydraulic-actuator TVC rams; 2 retro
  + 16 RCS nozzles condensed into their carriers (solver_condensed_into);
  routed fuel-line edges tank->thruster with feed_share (mixture 1.65).
  Thruster records and feeds are READ BACK from node/edge declarations.
- check() == []; wrench_paths: 41 massive bodies, all rigid to
  hull.docking_ring; no thruster path crosses a feed line.
- Mass properties: machine_package.mass_properties on mode_table.
  _charged_nodes(OperatingState(charges_kg=tanks)). Dry 249.88 kg, wet
  945.88 kg; cm full (0.1195, -0.0254, 0.0412) -> drained (MMH 5, NTO 8,
  N2H4 60) (0.2137, -0.0743, 0.1208): 132.7 mm shift; full tensor
  (I_xy ~10 kg m2), so N1.6 runs as a matrix law with I^-1. The piece law
  matches the reduction at a partial fill to 1e-12 (cm) / 1e-11 (tensor).
- Allocation formulation (orbital_actuation.allocate_wrench): variables =
  throttles + each gimbal's two Cardan actuator angles. Cone EXACT as
  cos a cos b >= cos(cone) (n.d = cos a cos b), slew EXACT as a per-actuator
  box, throttle slew/range a box, fuel per tank linear, deadband an
  enumerated on/off choice (2^m, m = 3 here). SLSQP (scipy), warm start =
  current states, objective normalised by its warm value. WHY not SOCP in
  thrust-vector variables: the deadband lower bound is non-convex in |f|,
  the actuator slew is a box in angles not a cone in f, and scipy has no
  SOCP solver (cvxpy/clarabel/ecos absent). WHY not L-BFGS-B: the cone and
  tank budgets are not boxes. The angle problem is bilinear (u * d(a,b)),
  so SLSQP's answer is a local optimum from the warm start; the
  constraints hold exactly (projected after the solve).
- Measured: pure torques met by 2-8 RCS with |F| <= 6.8e-5 N, |tau short|
  <= 7e-6 N m; trim through the cm: gimbal 2.3747 deg vs line 2.3706 (full),
  6.4036 vs 6.4036 (drained), |tau| < 1e-6 N m; 10 kN request: achieved
  (4084.8, -37.0, 94.9) N, shortfall (5915.2, 537.0, -94.9); h=0.1 s from
  rest: main cannot pass its 0.4 deadband (2/s), F_x 88 N (RCS only);
  h=0.5 s: 2998.8 N. dt: throttle state 2/s, lit at 0.20 s, gimbal 10 deg/s
  to 7 deg, |F - law| 4.6e-13 N, |tau - law| 1.1e-13 N m; 20 s burn:
  NTO/MMH drawn 1.650000, propellant = impulse/c to 1e-12; allocate seam:
  delivered |F| 2999.6000 = achieved, torque 4e-7 N m.
- Allocation cost: 0.4-1.1 s per call (8 branches x SLSQP). OPEN.

INVENTED (no existing machinery decided it): the craft's dimensions and
masses; Thruster fields role/cone/gimbal_axis/slews/deadband/feeds; the
Cardan d(a,b) form; delivered-throttle and per-tank supply/flow columns;
reduction-as-linear-law coefficients; the fuel network as routed fuel-line
edges only (fuel_network.FuelNetworkSpec/Runtime are cylinder-admission
plumbing: no rocket admission stage, and emit_fuel_network_graph's
"fuel-supply-line" is not a declared member constraint); fluids rows
hydrazine/MMH/NTO; allocation weights = the tracker's (T_max, tau_max).
Thrust is priced per newton (tracker convention), so swapping main for RCS
thrust is fuel-neutral in the cost; pricing by propellant is a choice left
open.

HOOKS NEEDED (not made; other owners):
- orbital_tracker.fly() line 808: `craft.throttle(command.throttles)` ->
  `if not getattr(craft, "applies_allocation", False): craft.throttle(command.throttles)`.
- orbital_tracker.attitude_torque_demand: `principal_inertia(design)` ->
  the craft's `inertia_tensor()` (full tensor: `I @ (...) + w x (I w)`);
  `_request`'s `torque_matrix(design)` test and `burn_capacity(design,
  ...)` assume CM-relative fixed thrusters: for a machine craft use
  `craft.allocate` (capacity = allocate a large request, read achieved).
- orbital_game.craft_part_geometry: read `craft.thruster_geometry()`
  (mount relative to the CURRENT cm, exhaust at the CURRENT gimbal,
  delivered throttle) when the craft has it; body box: `craft.design.
  body_size_m` is the machine prism.
- Machine craft is batch 1 only (MachineCraft).
