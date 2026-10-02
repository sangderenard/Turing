# Orbital craft, build step 3: the fixed plan and its live tracker (2026-10-02)

Decisions: `docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md` (binding).
Steps 1-2: `ORBITAL_CRAFT_STEP1_CONTINUATION.md`, `..._STEP2_...`.
Status: built and green. Nothing committed. No jumper/actuation edits.

## Files (root repo, `engine_toy/`)

- `orbital_plan.py` -- Hohmann plan + reference trajectory (section KE laws).
- `orbital_tracker.py` -- on-plan tracker (design 4, ON PLAN only).
- `tests/test_orbital_plan.py` (9), `tests/test_orbital_tracker.py` (3).
  From `engine_toy/`:
  `python -m pytest tests/test_orbital_plan.py tests/test_orbital_tracker.py -q`
  -> 12 passed, 33 s warm (the flight is ~26 s). Steps 1-2: 11 passed.

## Public surfaces

`orbital_plan`: `HohmannPlan` (frozen: mu, r1, r2, phase, t_burn1,
t_burn2, dv1, dv2, a_transfer, e_transfer, transfer_time; `ideal_delta_v`,
`legs()`), `hohmann_plan(mu, r1, r2, *, t_burn1, phase)`,
`reference(plan, t) -> (r, v)`, `kepler_residual(plan, t)`,
`two_body_acceleration(mu, positions)`, `KEPLER_LAWS` (eq_KE1_1..7,
eq_KE2_1..13, also module globals), `KEPLER_SCALE` (a `LawScale`).
Imports nothing from the jumper.

`orbital_tracker`: `TrackingGains` (frozen: natural_frequency_rad_s=0.02,
damping_ratio=1, fuel_weight=1e-4), `allocate_throttles(design, F, w)`,
`tracking_command(plan, design, gains, t, r, v, m) -> TrackingCommand`,
`fly(craft, plan, gains, until_s=, round_s=) -> FlightReport`. Touches the
craft only through `r()`, `throttle()`, `advance()`, `design`, `mass_kg`,
`time_s`, `fuel_impulse_n_s`.

## Laws and provenance

All derived from `src/transmogrifier/orbital.py` `Orbit` (imported, unchanged):
apsides of `orbital_radius_theta` (theta = 0, pi) give a_t, e_t; `vis_viva`
(positive root) gives both burns; `kepler_equation` at E = pi over the mean
motion gives the transfer time; the Newton stage is `kepler_equation`'s
residual over its E-derivative; r(t) is `orbital_radius_theta` at the true
anomaly (cos_nu, sin_nu from E -- no atan2); v(t) is d position/dE times
dE/dt = n / (1 - e cos E) (Kepler's equation differentiated in time);
two-body acceleration is -grad of `Orbit.hamiltonian`. The one law the
original does not state: mean motion n = sqrt(mu/a^3) (eq_KE1_6).

Pieces (`equation_piece`, composition by substitution only):
`orbital_plan_hohmann` (batch 1), `orbital_plan_mean_anomaly`,
`orbital_plan_kepler_stage`, `orbital_plan_state`, `orbital_plan_two_body`
(batch `REFERENCE_BATCH` = 64; longer t is chunked, short t padded).

## Kepler stage

ONE stage law (eq_KE2_2, iterate symbols E_k -> E_k1, N11 convention); the
consumer applies it `KEPLER_STAGES = 6` times from the declared starter
E_0 = M. Measured: compiled max residual 2.4e-16 over 2048 samples of all
three legs (e = 0.715); numpy grid 4.4e-16 for e <= 0.85, 2.3e-12 at 0.9.
`KEPLER_SCALE` predicate e <= 0.85; `hohmann_plan` refuses beyond it
(r2/r1 > 12.3); `reference` raises if residual > 1e-12.

## Plan results (LEO 7000 km -> 42164 km, mu 3.986004418e14)

dv1 2336.7958 m/s (rel err 0), dv2 1433.9315 (4.8e-16), T 19178.154 s
(1.9e-16); vis-viva on the transfer 8e-15; |r(t2)| = r2 to 1e-16; v equals
the finite-difference derivative of r to 3e-10. Lowering transfer
(r2 < r1) uses the same laws (negative dvs).

## Tracker method

Computed-acceleration PD: `a_des = [g(r_ref) - g(r)] - w^2 e_r - 2 zeta w e_v`
(feedforward = the reference's two-body acceleration minus what gravity
already gives the craft); `F_des = m a_des`. Allocation is the controller
cost (decision 7): `J(u) = |B u - F_des|^2/(2 T_max^2) + w_f sum T_k u_k / T_max`
over the throttle box, B = `actuation_matrix(design)` as dF/du; convex box
QP, L-BFGS-B from the clipped least-squares start. Deadband = w_f * T_max;
opposed thrusters never fire together. Throttles held per round (ZOH).

## Tracking result (test)

Six-axis jumper 1000 kg, 100 kN per thruster, round 10 s, dx 50 km,
t_burn1 = 300 s, flown to t2 + 3000 s: 2248 rounds, 8991 substeps.
Final |r - r_ref| 63 m, |v - v_ref| 0.37 m/s; |r| - r2 = -56 m,
|v| - v_circ = -0.003 m/s, v_r = 0.37 m/s. Fuel 4.682e6 N*s = 1.242 x
ideal m(|dv1| + |dv2|) = 3.771e6. Tolerances asserted: 200 m, 1 m/s,
0.05 m/s speed, fuel ratio in [1, 1.3).

- The steady 0.37 m/s is radial and equals g dt/2 at GEO (dt ~3.3 s):
  symplectic Euler's momentum is staggered half a step from position. At
  dx 10 km it is 0.05 m/s (and 11 m position), wall 130 s.
- Fuel overhead is finite-burn lag behind the impulsive reference (the PD
  then recovers the position lag) plus the six-axis L1 geometry: 20 kN
  thrusters measured 1.74 x ideal, 100 kN 1.24 x. Fixing that is the
  planner's job (step 4: finite burns in the collocation plan).

## The catalogue move (exact)

1. Cut `kepler_laws` into `honorary_engine_equation_catalogue.py` as
   `def _expand_kepler():` under a new header `# 7d. KEPLER -- two-body
   orbits and the Hohmann transfer` after the Woodshop section, with
   `globals().update(_expand_kepler())`. Its body is unchanged except
   `orbit=Orbit` becomes a local `from src.transmogrifier.orbital import
   Orbit` (the catalogue needs turing on sys.path for that one import --
   the 3-line guard `orbital_plan` carries) and `sp.symbols("...")` ->
   `sp.symbols('...')` so `raw_token_report` sees the tokens.
2. `ENGINE_PREFIXES['KE'] = ('Kepler', None)`.
3. `_law_scales()['kepler_fixed_stage'] = LawScale(... laws=("KE1", "KE2") ...)`
   with `KEPLER_SCALE`'s predicates and source.
4. In `orbital_plan`: `KEPLER_LAWS = honorary._discover_equations()['Kepler']`,
   `KEPLER_SCALE = honorary.LAW_SCALES['kepler_fixed_stage']`; delete
   `kepler_laws` and its `Orbit` import. Nothing else changes; the piece
   cache keys are srepr-based, so the pieces are reused.

## Step 4 plugs in here

The collocation planner replaces `hohmann_plan` with a plan of the same
shape (legs + finite burns); `tracking_command` needs only `reference` and
the plan's `mu`. Off-plan switching (step 5) fires on
`TrackingCommand.position_error_m` / `velocity_error_m_s`.
