# Orbital craft, build step 2: the actuation matrix (2026-10-02)

Decisions: `docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md` (binding).
Step 1: `docs/ORBITAL_CRAFT_STEP1_CONTINUATION.md`.
Status: step 2 built and green. Nothing committed.

## Files (root repo, `engine_toy/`)

- `orbital_actuation.py` (new) -- `Thruster`, `CraftDesign`,
  `actuation_matrix`, `clamp_throttles`, `six_axis_jumper`, the law
  (`thrust_magnitude`, `actuation_force_rhs`), `thruster_columns`.
- `orbital_jumper.py` (edited) -- actuation piece first in the round,
  `throttle()` seam, `raw_force_*` columns, per-thruster fuel.
- `tests/test_orbital_actuation.py` (new). Run from `engine_toy/`:
  `python -m pytest tests/test_orbital_jumper.py tests/test_orbital_actuation.py -q`
  -> 11 passed (13.5 s warm; ~60 s with cold piece compiles).

## The law

Catalogue `eq_TS1_2` (Tsiolkovsky, `F_thrust = m_dot * c`). No new catalogue
entry. The throttle is DECLARED as the fraction of max mass flow:
`m_dot = clamp(u, u_min, u_max) * max_thrust / c`; `c` cancels (checked, as
G cancels in step 1's N4.1). Per axis:
`applied_force_a = sum_k direction_k,a * TS1.2_k + raw_force_a`, i.e.
`B @ clamp(u) + raw`, `B[:, k] = max_thrust_k * direction_k`, attitude =
identity until decision 9. The design enters as columns
(`thruster{k}_max_thrust`, `_direction_{x,y,z}`, `_throttle_min/max`), so
one compiled piece (`orbital_jumper_actuation_t{n}`) serves every design
with n thrusters. `Thruster.position_m` and `.kind` are declared but unread
until step 7 (torque r x F; Isp per kind via `eq_TS1_4`).

Jumper (`six_axis_jumper(T, m)`, order +x,-x,+y,-y,+z,-z; mounts on the
line of action, zero torque):

    B = T * [[1,-1, 0, 0, 0, 0],
             [0, 0, 1,-1, 0, 0],
             [0, 0, 0, 0, 1,-1]]

## Seam rule

`throttle(u)` sets `thruster{k}_throttle`; `F(f)` sets `raw_force_*`. They
SUPERPOSE (neither wins): throttle drives the piece, `F` is the raw
override kept for tests and non-thruster forces. Throttles zero -> step 1
exactly. `r()` unchanged. `applied_force()` reads what the last substep
integrated; `commanded_force()` is the host-side `B @ clamp(u) + raw`.

## Fuel (decision 7)

`fuel_impulse` name kept = total impulse (N*s) =
`sum_k clamp(u_k) * max_thrust_k * dt + |raw_force| * dt` (the raw part is
still the original `force_cost_integral` integrand). New per-thruster
`thruster{k}_impulse`. Opposed thrusters cost fuel at zero net force
(tested).

## Measured

- +x burn, 100 N, 1000 kg, no gravity, 100 s: x error 1.6612 m (bound
  4.47 m), 314 substeps -- identical to step 1's F() test today.
- Step-1 circular orbit today: radius 1.79e-3, energy 1.27e-5, 1779
  substeps, vs the step-1 doc's 2224. Cause not measured; the actuation
  piece publishes no metric and writes zero force there, and commit
  5f47485 (dt ratchet fix) landed after that doc's numbers.

## Step 3 plugs in here

A live tracker of a fixed plan reads `r()` and the plan's reference, solves
for `u` against `actuation_matrix(design)` (B is the control Jacobian
dF/du; with clamp bounds it is a box-constrained allocation), and calls
`throttle(u)` once per round. The fuel term of its cost is
`sum_k u_k * max_thrust_k` (per-thruster columns already exist).
