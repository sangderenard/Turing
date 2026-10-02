# Orbital craft, build step 1: jumper behind the r()/F() seam (2026-10-02)

Decisions: `docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md` (binding).
Status: step 1 built and green. Nothing committed.

## Files

- `engine_toy/orbital_jumper.py` (root repo) -- `orbital_jumper_dt_pieces`,
  `GravityCenter`, `OrbitalJumper` (seam `r()` / `F()`, `advance()`).
- `engine_toy/tests/test_orbital_jumper.py` -- engine_toy's tests folder is
  the convention for engine_toy modules. Run from `engine_toy/`:
  `python -m pytest tests/test_orbital_jumper.py -q` (3 passed, ~12 s).

## What it is

Woodshop pattern (`woodshop.newton_world_dt_pieces`), four `equation_piece`s
in one `RoundNode` (sequential), instantiated once (`instantiate_system`),
advanced with `advance_round`. Eager dt system only; no `lowered_system`.

| piece | law | writes |
|---|---|---|
| N4.1 gravity | `eq_N4_1` vector form, summed over N centers, + `applied_force_*` | `force_*` |
| N1.2 momentum | `eq_N1_2`, symplectic Euler | `momentum_*` |
| N1.1 position | `eq_N1_1` | `position_*`, publishes `max_vel` |
| thrust cost | original `force_cost_integral` integrand, ds -> dt | `fuel_impulse` (N*s) |

Columns: `mass` (constant until step 7), `position_*`, `momentum_*`,
`force_*`, `applied_force_*` (the seam), `fuel_impulse`,
`center{k}_x/y/z`, `center{k}_mu`.

Provenance: `eq_N4_1` with `|x_j - x_i|` read as the Euclidean distance and
`m_j = mu/G` (G cancels) equals the original
`OrbitalTransfer.symbolic_transfer_spline` `F_grav` times mass -- asserted
numerically in the test (rel 1e-13). The test's energy is the original's
`total_energy_expression`. The cost integrand is imported unchanged.

## Measured

- Circular orbit, mu = 3.986e14, R = 7000 km, one period (5828.5 s) in 20
  rounds, dx = 50 km, cfl 0.5: max radius error 2.06e-3 R, max specific
  energy drift 1.38e-5, 2224 substeps, mean dt 2.62 s (CFL 0.5 dx/v =
  3.31 s; the PI controller ranges ~0.67..6.6 s, capped by dx/max_vel).
- Constant F, no gravity, 100 s: x error 2.12 m against the first-order
  symplectic-Euler bound (a/2) t dx/|v0| = 4.47 m; velocity and fuel exact.

## Finding: HOLD ratchet with `energy_exchange_fraction`

With Woodshop's `Targets(..., energy_exchange_fraction=0.2)` and pieces that
publish no energy channel, every participant is `HOLD`, so
`exchange_time_bound` caps the next dt at the current one; the window's
landing remainder (a short final substep) then ratchets dt down every round
(3.3 s -> 3.7e-4 s in six 291 s rounds, then a window failed to land). The
jumper declares no fraction. Woodshop does not hit this only because its
window is one substep. Not changed in Woodshop or the dt system.

## Step 2 plugs in here

The actuation matrix (control signals -> force) is a new piece placed before
"N4.1 gravity" that writes `applied_force_*_next` from control-signal
columns (valves, throttles), or `F()` is fed from it. Nothing downstream of
`applied_force_*` changes; `fuel_impulse` becomes propellant mass when step
7 adds a specific impulse per thruster kind and makes `mass` a written
column.
