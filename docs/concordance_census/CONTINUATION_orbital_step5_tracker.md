# Continuation: orbital craft step 5 (off-plan switching) + tracker catch-up to step 7

Owner files: `engine_toy/orbital_tracker.py`, `engine_toy/tests/test_orbital_tracker.py`,
and ONLY the attitude dt bound in `engine_toy/orbital_jumper.py`.
Not touched: `orbital_collocation.py`, `orbital_game.py` (other agents).
Uncommitted.

## 2026-10-03 survey findings

- dt mechanism for a piece to bound the step (read, not run):
  `examples/llvm_dt_system.py piece_source`: a piece that has an output named
  `dt_limit` publishes it per participant (`pub_dt_limit`, present flag) and
  in the amalgamated `Metrics.dt_limit` (min over pieces; `+inf` is the
  system's own "no bound" value). `dt_controller.step_with_dt_control_used`
  then takes `dt_next = min(PI proposal, metrics.dt_limit)` on both the
  rollback and no-rollback lanes, and the retry halving respects it. Precedent:
  `tests/dt_system/test_llvm_dt_system.py::test_llvm_piece_steps_under_the_dt_system`.
  The bound applies from the NEXT attempt; the first attempt of the first round
  is the plan's `dt_init`, which OrbitalJumper computes from its own initial
  state (CFL) -- the attitude bound must enter there too.
- `exchange_time_s` (E/P) is the energy-stability slot; it is not the right
  slot for a kinematic angle-per-step bound. `dt_limit` is.
- The game (`orbital_game.py:400`) calls `fly(craft, plan, gains, until_s=,
  round_s=)` once per frame; any switching state must therefore persist
  across fly calls (an optional mode object), and fly's defaults must keep the
  step-3 behaviour (no switching unless a re-planner is injected).
- Collocation notes (step 4) observed: the tracker calls
  `orbital_plan.reference(plan, t)` which only knows HohmannPlan legs.

## 2026-10-03 PART B: attitude publishes `dt_limit` (orbital_jumper.py)

- Position piece (N1.1/N1.3) gains output `dt_limit = attitude_step_max /
  |omega_new|` (column `attitude_step_max`, declared per craft through
  `OrbitalJumper(attitude_step_rad=ATTITUDE_STEP_RAD = 0.125)`; `math.inf`
  declares no bound). |omega| = 0 compiles to +inf (measured), the dt
  system's own "no bound". The initial `dt_init` also takes
  `attitude_step_rad / |omega_0|` (first attempt has no publication behind it).
- One compile of `orbital_craft_position` (~15 s incl. run).
- MEASURED (LEO r=7e6, dx chosen so CFL dt = 22 s, window 22 s, spin
  0.1 rad/s about z, six-axis design; scratch spin22.py):
  - no bound (attitude_step_rad=inf): 2 substeps/window, turned
    1.66596 rad per 22 s window (2 atan(1.1); omega t = 2.2).
  - bound 0.125 rad: 18 substeps/window, dt_limit 1.25 s, turned
    2.197204 rad per window (shortfall 0.13 %, = 18 * 2 atan(1.2222*0.05)).
- Known property, not changed: every orbital piece is HOLD (no exchange
  channel), so dt never grows back once the attitude bound shrinks it; a
  slew leaves the craft at the slew's small dt for the rest of the run.

## 2026-10-03 PART A first cut (superseded the same day, see scope change)

- Built: attitude-dependent B, fuel-aware L-BFGS-B allocation (reserve price
  ~1/P plus the hard round budget met by its KKT multiplier via bisection),
  torque rows with an SO(3) attitude PD. Unit check: cold-gas + biprop on
  one axis, 150 N demand: ample tank -> biprop 1.0, cold 0.4999; 0.05 kg
  tank -> cold 0, biprop 0.760, draw exactly 0.05 kg.
- Rotating main+RCS craft, LEO 7000 -> 8000 km: burn 1 starts with the
  engine 1.78 rad off the demand; slew (w_a 0.2) brings it to 0.13 rad by
  t_burn1+36 s. Final |r - r_ref| 345 m, |v - v_ref| 3.4 m/s, fuel ratio
  1.87: a static residual (|e_r| 344.7 m, |e_v| 3.376 m/s, |F_des| 2.6 N,
  throttles ~1e-4) from t ~ 4000 s on. Cause not yet found.

## 2026-10-03 SCOPE CHANGE (coordinator, from the user)

- The craft becomes an engine_toy machine (machine-sim graph geometry; typed
  navigation/brake/main thrusters with gimbal cones, own feeds, throttle slew).
  A new agent builds the allocation (throttles + gimbal angles for a
  requested force+torque, reporting the ACHIEVED wrench).
- DROPPED from my scope: fuel-aware allocation and the torque_matrix
  attitude allocation. The tracker now produces a DESIRED WRENCH through one
  seam, `craft.allocate(wrench) -> achieved wrench + throttles`, treats the
  achieved wrench as what happens, and feeds the shortfall into the
  off-plan error and the next command. Until the layer lands, a thin adapter
  over orbital_actuation's least squares stands in (one-line swap).
- Parts B and C unchanged. The new agent also edits orbital_jumper.py:
  re-read it right before any edit; keep my edits small and local.

## 2026-10-03 wrench seam built; root cause of the static residual FOUND

- orbital_tracker.py now requests a desired wrench (world force + craft
  torque) through `craft.allocate(wrench) -> Allocation(achieved, throttles)`;
  `LeastSquaresAllocator` (step-3 L-BFGS-B over R@B_craft and torque_matrix)
  stands in; the one line is in `fly`. The achieved wrench is what happens:
  the request's shortfall is carried into the next request (clipped to
  |F_pd|, no windup) and enters eps as |s| h / m of velocity error.
- Attitude: hold first (rate damping); point the strongest-thruster axis at
  the request only when the held attitude leaves a shortfall >
  pointing_tolerance |F| + deadband. (Pointing on every small demand made the
  RCS chase a rotating target: 1.4 rad error, 0.066 rad/s spin, 130 kg RCS.)
- ROOT CAUSE of the steady 334 m / 3.32 m/s residual (and the step-3
  tracker's 63 m at GEO): `OrbitalJumper.r()` returns symplectic Euler's
  STAGGERED pair -- the velocity is half a substep ahead of the position,
  |v_read - v(t)| ~ g dt / 2 (8000 km, dt ~ 1 s: 6.23 * 1 / 2 = 3.1 m/s;
  measured 3.32). The PD settles where w^2 e_r = 2 w e_v, |e_r| = 2 e_v / w
  = 332 m (measured 333.8). Not a tracker defect; the seam reports a
  non-co-temporal (r, v). Not my file to fix (jumper owner / new agent):
  r() could return v - (dt/2) * F/m of the last substep.

## 2026-10-03 coordinator: game-lane findings for the tracker

1. Fuel: 7000->8000 uses 1.71x the rocket-equation ideal; burn 1 1.64x
   (PD spreads it over ~50 s and overshoots); coast ~29 kg / 600 s fighting
   integration drift. Wanted: coast deadband sized to the integrator drift;
   burns along the plan's impulse direction at full throttle, centred on the
   burn time, finite-burn corrected; the PD only trims. Measure vs ideal.
2. Expose on/off-plan state on fly()'s result.
3. Expose per-thruster throttle history / last throttles from fly().

## 2026-10-03 fuel-honest burns + trims (measured, 7000 -> 8000 km, main+RCS craft, rounds 2 s)

- Burns: plan impulses (HohmannPlan leg velocity steps, or plan.impulses())
  flown at full capacity along a fixed direction, dv = impulse - e_v at
  arming, rocket-equation duration, centred on t_b, armed 6/w_a early to
  slew, fire within burn_alignment_rad, closed on DELIVERED dv (achieved
  force / mean mass). PD off during burns.
- Trims: PD only outside a coast band: v_band = max(k |v_ref|, |g| h)
  (twice the reading-stagger bound g dt/2, dt <= h), p_band = max(k |r|,
  |g| h / w) (the PD's own stagger offset); stop at half bands.
- MEASURED (scratch diag.py, rocket-equation ideal 148.0 kg for 486.8 m/s
  at Isp 310): burns 149.0 kg (+0.7 %), trims 13.7 kg (7.4 main + 6.3 RCS),
  coast 0.006 kg; total 162.7 kg = 1.10 x ideal (was 1.71 x on the game's
  PD-only lane; 242 kg on my PD-only run). Burn 1 armed 258 s, slewed
  1.58 rad, start 288.1, done 314; burn 2 3521.5 -> 3546.
- Tried p_band = 2 |g| h / w: WORSE (trim 91 kg): coasting becomes a trim
  limit cycle (e_r 500 -> 1250 m in ~200 s of coast, then a slew + burst).
  The drift between trims is real (r() is seeded co-temporal, so the
  integrator's first half-step shifts the orbit energy); only a co-temporal
  r() fixes both the residual and this. Reverted to |g| h / w.
- Remaining trim cost is post-burn finite-burn error (e_r ~700 m after
  burn 1) plus the stagger equilibrium. "Few percent of finite-burn
  optimum": burns alone yes (+0.7 %); whole flight 10 %, limited by the
  r() stagger (open, jumper owner).

## 2026-10-03 later: fixes found by measurement

- Burn never ended: the last 0.1 m/s sat inside the stand-in's fuel deadband
  (10 N at 1e5 N thrusters). Burns now end when what is left is <= 1.5x the
  deadband force; trims take the rest.
- Re-plan burn sign bug: arming after t_b used dv = step - (v - v_ref(post)).
  Now dv = v_ref - v (+ step while the reference is still pre-burn).
- Six-axis burns: a saturated request let the least squares fire both axes
  full (sideways waste). Capacity is now the exact along-direction maximum
  by LP (scipy linprog, max s: B u = s d, u in box); the burn requests
  min(capacity, m * remaining / h) exactly along its direction.
- Trim requests clipped to that capacity (a saturated request drowned the
  torque rows: the stand-in spent RCS on sideways force, attitude diverged).
- Stand-in allocator: an EMPTY tank makes propellant thrusters achieve
  nothing (was: achieved = B u regardless; the tracker believed 10 kN it
  never got and the run diverged). Partial tank not modelled.
- Bands use the dt system's own continuation step (advance's returned
  dt_next), not the round: game rounds are 10 s, substeps 3.3 s.
- The game calls fly() per frame with no mode: fly now keeps one mode per
  craft (weak map; renewed when a different plan object is passed) so burns
  and switch state persist; `tracker_mode(craft)` exposes it.
- HohmannPlan.reference(t) method added in orbital_plan.py (coordinator hook
  request); the tracker reads every plan through plan.reference(t).

## 2026-10-03 MEASURED (final code)

- tests/test_orbital_tracker.py: 7 passed (144 s). test_orbital_jumper.py
  8 passed + 1 xfail (another agent's new interleaved-states test),
  test_orbital_actuation.py 8 passed, test_orbital_plan.py 9 passed,
  test_orbital_game.py (read-only) 7 passed.
- LEO->GEO six-axis 10 g (step-3 test): fuel 1.0703 x impulsive ideal (step 3
  PD-only: 1.2416); final |r - r_ref| 119.3 m, |v - v_ref| 0.378 m/s,
  |r| - r2 -19.9 m, |v| - v_circ 0.043 m/s.
- Rotating main+RCS craft 7000 -> 8000 km (spin 0.05 rad/s, engine radial):
  burn 1 armed t_b-40 s with the engine 0.465 rad off, fires t_b-10..+12 s,
  worst pointing while firing 0.0116 rad; propellant 165.5 kg = 1.1185 x
  TS2.1 ideal 147.97 kg (burns ~ ideal +0.7 %; the rest trims + RCS);
  final |r - r_ref| 334.5 m, |v - v_ref| 3.33 m/s (= r() stagger, below).
- Kick (300 m/s radial, t_b1 + 300 s, band 0.03/0.005): nominal max eps
  0.0188; one 'off' at 602 s (eps 0.0392), one re-plan (r1 7024488 m ->
  8000 km, burn 1 now), 'on' at 666 s; no chatter; arrives |r| - r2
  -333.9 m (stagger).
- Game (read-only): 1.5279 x rocket ideal (was 1.71); rendezvous 4753 m,
  8.43 m/s (was 4.93 km, 9.5 m/s). The six-axis craft cannot turn (thrusters
  on axis, zero torque) so burns off the craft axes cost (|cos|+|sin|) =
  1.4007 more: its impulsive optimum is 1004.6 kg = 1.358 x; the burns used
  1001.7 kg (at that optimum); trims 128.7 kg (+12.8 %).

## OPEN

1. r() STAGGER (jumper owner / machine agent): symplectic Euler's velocity is
   half a substep off its position; the tracker settles at
   |e_r| = 2 (g dt/2) / w (333 m at 8000 km, dt ~1 s) and trims spend fuel
   on real drift the stagger hides. A co-temporal r() (v + (dt/2) a of the
   last substep) removes both; not my file.
2. Hohmann-from-present as re-planner departs horizontally from a circle:
   mid-transfer (radial velocity ~250 m/s) a 150 m/s kick costs a 652 m/s
   re-plan and empties the tank. The kick test therefore kicks 300 s after
   burn 1. The collocation planner (same callable signature) is the fix.
3. Impulsive plans: a centred finite burn against an impulsive reference
   reaches eps 0.019-0.025 nominally, which floors the off-plan threshold
   (defaults 0.05/0.01; the kick test uses 0.03/0.005).
4. HOLD: no orbital piece publishes an exchange channel, so dt never grows
   back after the attitude bound shrinks it (by reading).
5. Seam question for the machine-layer agent: the tracker calls
   craft.throttle(allocation.throttles) after craft.allocate(wrench). If
   allocate also sets gimbals, it should own the command too.
- Part B now has a test in tests/test_orbital_tracker.py
  (test_attitude_dt_limit_keeps_a_spin_per_step_small): unbounded 1.66596
  rad per 22 s window (2 substeps), bounded 2.197204 rad (18 substeps).
  Tracker file: 8 tests.

## 2026-10-03 after 6b20ed8 (co-temporal r()) and 3d6d74c (mean-step kick)

Coordinator items, in order:
1. Re-derived against the honest r(): the stagger floors are gone; bands are
   coast_position_deadband * |r_ref| and coast_velocity_deadband * |v_ref|,
   defaults 1e-6 / 1e-5 (7 m, 0.075 m/s at 7000 km; the jumper's coast error
   is 6.9 m/orbit). Sweep on the main+RCS transfer (fuel x TS2.1 ideal):
   0/0 1.104, 1e-6/1e-5 1.061, 2e-6/5e-5 1.079, 4e-6/1e-4 1.082,
   1e-5/1e-4 1.186, 1e-5/3e-4 1.214, 3e-5/3e-4 1.278: wider is WORSE.
   Noise source found: burn 2 armed with the engine 2.94 rad off (a trim had
   pointed it away), fired 26 s late -> 6.3 km error -> 28 kg of trims. Fix:
   the slew lead is now the attitude PD's own settling time from the present
   angle ((1+zwt)e^-zwt = alignment/angle) + 2 rounds (was a fixed 6/w_a).
2. fly() skips craft.throttle when craft.applies_allocation; declared on
   OrbitalJumper (False) and MachineCraft (True).
3. attitude_torque_demand takes the full inertia tensor (craft.inertia_tensor();
   OrbitalJumper.inertia_tensor() added = diag of its principal moments).
4. New `Actuation` object in orbital_tracker: allocate (applying), probe
   (machine: apply=False), inertia, can_torque (machine: probe a pure torque),
   capacity (fixed design: the LP; machine: probe the pointing thruster's
   thrust along its axis with zero torque, read the achieved force; cached),
   pointing (machine: the ACHIEVED thrust line, which leans off +x because
   the gimbal cancels the CoM moment).
5. Stand-in prices fuel by propellant mass (fuel_price: T_k/(Isp g0),
   scaled so the most efficient kind's strongest thruster costs 1; ideal-only
   designs keep impulse pricing).
6. Machine-craft test added.

Tracker changes found by the machine run:
- Capacity probe asks for the pointing thruster's own thrust (asking for the
  sum also lit the forward RCS and left no torque authority).
- Burns light within burn_alignment_rad (0.05) and, once lit, burn on within
  burn_release_rad (0.2): pausing every round made fire/pause cycles.
- A trim that must turn more than burn_release_rad requests torque only
  until it is within it (asking for force meanwhile, the machine allocator
  spent it sideways on brakes/RCS and never turned: tumbling trims).

MEASURED (tests/test_orbital_tracker.py 9 passed, 307 s; jumper/actuation/plan
27 passed + 1 xfail; craft_machine 12 passed; game (read-only) 7 passed):
- rotating main+RCS: 159.77 kg = 1.0797 x ideal (limit 1.15; was 1.28 on
  the honest r() before re-deriving); final |r - r_ref| 2.1 m, 0.034 m/s.
- kick: one off (602 s, eps 0.0386), one on (656 s); final 4.6 m,
  v_r -0.032 m/s (limit 5.0; was 5.25).
- LEO->GEO six-axis: 1.0813 x ideal; final 27.8 m, 0.24 m/s.
- game: rendezvous 18.6 m, 0.27 m/s (was 4753 m, 8.4 m/s); 1.4004 x ideal.
- MACHINE CRAFT 7000 -> 8000 km (2 s rounds, w_a 0.2): arrives |r| - r2
  -0.40 m, |r - r_ref| 0.95 m, 0.035 m/s. Bipropellant 189.3 kg = 1.353 x
  TS2.1 ideal 139.96 kg, plus hydrazine 28.5 kg. Main gimbal up to 0.034
  rad. By phase/role (commanded): burn main 139.6 kg (= ideal), burn RCS
  7.8, burn brake 3.7, trims 73 (main 38, brake 11, RCS 24).

OPEN (allocator lane, orbital_actuation.allocate_wrench):
- With the main engine lit the allocation saturates the RCS: static repro,
  machine craft at rest, allocate(F = 3997 N along +x, tau = 0, round_s 2,
  apply=False): main 1.0, RCS throttle sum 7.94 (of 16), achieved torque
  -0.47 N m about y; in flight it also lights a retro (0.25-0.49) against
  the main. With the RCS spent, a requested pitch torque (+5 N m) came back
  -5 N m while the main fired, so the attitude drifts to the 0.2 rad
  release gate during every burn (burn 1 took 94 s for a 55 s plan), the
  thrust leans up to 0.2 rad, and trims pay 73 kg to clean up. This is the
  1.35 x; the burns themselves cost the ideal.
- Machine allocate costs 0.1-0.4 s/call; the machine test takes ~4 min.
