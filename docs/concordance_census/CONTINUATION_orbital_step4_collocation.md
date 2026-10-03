# Continuation: orbital craft step 4, the collocation planner (condensed)

Full 420-line working log: `git show 17e69599:docs/concordance_census/CONTINUATION_orbital_step4_collocation.md`.
Code: `engine_toy/orbital_collocation.py`, tests `engine_toy/tests/test_orbital_collocation.py`
(9 passed at hand-off). Design: `docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md`
(decisions 4, 7; build step 4). Landed in engine_toy commits 77f4496 and 3afe283.

## Status

The planner works. A nominal LEO 7000 -> 8000 km transfer plans in ~0.6 s at
Hohmann fuel (1.0000 x) and 1.0010 x Hohmann time. A live re-plan after a
300 m/s kick takes 1-2 s (was 205 s with SLSQP on the full transcription).
Tracker flights of the plans arrive within ~5-16 m and ~0.03 m/s of target.

## Architecture

- Transcription: N nodes (N=40), node state = momentum(3), position(3),
  propellant mass(1). A slice is `substeps` = 16 kick-drift-kick steps of the
  jumper's own laws (N4.1, TS1.2/1.4 flow, N7.2, N1.1), supply 1. The slice
  Jacobian is the chain of each step's compiled Jacobian (forward accumulation;
  no extra compile). Per-slice error 3.6 m vs 256 sub-steps (one 85 s step was
  994 m off, which made the plan unflyable: 1.55 x fuel).
- Laws: jumper's builders + `eq_KE1_3` vis-viva arrival rows + decision-7 cost
  (alpha*I/T-style time term, beta, kappa barrier, fuel budget). Planning design
  at a declared attitude (default identity); thrusters priced by
  `orbital_tracker.fuel_price`'s rule. Hohmann warm start.
- Jacobians: `compile_reverse_rows` = ONE explicit-seed fused reverse motion per
  row set inside `reverse_compile_book`, one native run per seed. Scalar arena
  reused (`prepare_artifact_execution` once); re-preparing per call cost 241 ms
  vs 0.76 ms.
- Solver (default): `_Condensed` / `solve_structured`. Burn slices own a
  duration and per-thruster impulse w = u*dt (throttle box as linear rows);
  coast slices form arcs of equal slices sharing one duration; nodes follow by
  forward substitution (defects zero by construction). SLSQP then runs over
  (n_u+1)*burns + arcs variables (15 for 2 burns) and 5 arrival rows. The burn
  structure grows by the switching function (primer vector, one adjoint sweep
  through the stored Jacobians); multipliers by min-norm least squares.
- Old path kept as the measured reference: `plan_transfer(method="SLSQP")`
  (full transcription + least-squares polish of defects when infeasible).
- `CollocationPlan`: `reference(t)`, `impulses()`; `collocation_replanner(problem, craft=)`; live re-plan warms from the plan's remainder (`remainder_warm_start`). `fixed_first=True` exists, default off (never feasible).
- Tracker protocol: thrusting slices read impulsively (coast from node k to the
  slice midpoint, back from node k+1; impulses() = that step at the midpoint).
  Slices under 1 m/s (`THRUST_THRESHOLD_M_S`) are coasts. Ramped references and
  16 sub-step impulses per slice both fail with the tracker (11 km / 130 km off).

## Key numbers

| item | value |
|---|---|
| compile, arrival rows (5 x 7 wrt) | 96 s; 0.19 ms per call |
| compile, slice rows 6 thrusters (7 rows x 27 wrt) | 874 s; 0.76 ms per call (7 seeds) |
| compiled slice Jacobian in a solve | 24-46 ms (chain of substeps) |
| nominal, condensed | 16 it, 0.6 s, fuel 1.0000 x H, T 1.0010 x |
| machine proxy (six 4 kN biprop axes) | 21 it, 0.8 s, 486.2 m/s (H 486.8); flight 182.0 kg vs 139.9 planned (1.30 x; tracker's Hohmann machine test reads 1.38 x) |


Live re-plan from the plan's remainder (300 m/s kick): 1.2-2.0 s at kicks 302/600/1500 s, dv 409.5/453.2/536.2 vs Hohmann-from-present 690/813/918; a cold start costs the same dv in ~2 s.
Kick flight (kick at 600 s): one re-plan, 451.3 m/s, final |r - r_ref| 4.4 m.

## Cost-weight sensitivity (condensed solver; ratios to Hohmann)

Defaults: alpha = T_Hohmann/budget; beta = alpha*I_H/T_H^2 (dJ/dT = 0 at the Hohmann warm start); kappa 1e-3; budget 2e6 N s.

| weights | nominal fuel | nominal T | kick dv (m/s) |
|---|---|---|---|
| default | 1.0000 | 1.0010 | 411.1 |
| alpha x10 | 1.020 | 1.919 | 385.0 |
| beta x10 | 1.831 | 0.662 | 622.2 |
| kappa 1e-5..1e-1, budget 1.2/1.05 x H | 1.0000 | 1.001 | 411 |

Only the alpha:beta ratio matters; it sets the trip length. kappa and the
budget do nothing until the tank nears empty (barrier is a safety rail).
Defaults stay: scale-free (J ~ 0.5-1 for any craft), Hohmann is a stationary
point in time so the planner deviates only for real gains, and they sit at the
knee (x3 either way buys < 6 % fuel or < 7 % time for 35-75 % of the other).
Re-plans recompute weights from the present (budget = propellant left); freezing them saves <0.2 % fuel and lengthens trips 12-35 %.

## Root causes found

- Why SLSQP on the full transcription stalled (not scaling: Jacobian cond 79):
  (1) ~35 exactly flat directions (how a coast arc is split among slices only
  changes discretization) plus the bilinear throttle x duration valley in every
  slice; (2) the compiled reverse of the throttle clamp Min(Max(u,lo),hi) reads
  a tie as the average of both sides, so on the box edge (u=0 coasts, u=1 full
  burns, where SLSQP iterates live) every throttle column is HALF its inside
  value (-10043 vs -20086). `_Transcription.inside` handles (2) but alone did
  not fix SLSQP. Condensing over a burn structure removes both. Ruled out: trust-constr+SR1, tied coast durations.
- Compiler walls from the first lane, all fixed: W1 grads absent from
  buffer_order (42a689a2 plan_callsites; bisected to 534a4941, glsl_deployment_
  strategy `_propagate_callsite_tensor_specializations`/`publish_return_members`
  rework); W2 pow backward lacked LLVM emission (eps marker, per-callsite
  folds); W3 second compile in one process refused (`reverse_compile_book`);
  W2/W3 landed in turing 095a3c0c..51b4cebe. Also fixed here: `binary_value`
  reference-copy collision (fortran_c_shell records
  `module_metadata["tensor_ssa_reference"]`, `lower_training_motion_to_
  repository_ssa` links from it; committed).
- MachineCraft.propellant_kg fixed (no piece reads propellant_mass; dt keeps only read columns). Same class, NOT fixed: `MachineCraft.propellant_supply`.

## Open items

- ROW CACHE: compiled rows persist in `%TEMP%\orbital_collocation_rows`, keyed
  on the laws' srepr plus `perforated_network_llvm._compiler_fingerprint`
  (laws only). They were built before several compiler fixes. Rebuild once on
  current HEAD (about 15 min compile) and make the cache check
  `PieceCompilerRecord` so a stale compile cannot be served.
- Tracker threshold: the spinning craft triggers a re-plan at t=12 s (eps 0.0318
  > 0.03 during the first burn; 0.5 s re-plan, same plan). Tracker behaviour,
  not the planner's.
- `ENGINE_TOY_PIECE_SERVE_STALE=1` was used for all runs (other lanes' uncommitted edits mark pieces stale; a rebuild while another process holds the DLL fails lld-link "Permission denied"). Re-confirm without it.
- INVENTED defaults (not from a source): supply 1 in the slice law; substeps 16;
  N=40; THRUST_THRESHOLD_M_S = 1; polish budget 1000 nfev (old path); the weight
  rule for alpha/beta/kappa and budget 2e6 N s; `pointing_proxy` planning design
  (six +/- world-axis thrusters at 1e4 N for the jumper kick case).

## Traps

- Do not trust a plan's low fuel until it is flown or run at 256 sub-steps: the
  early "0.998 x Hohmann" was discretization error.
- Do not re-split a slice into sub-step impulses for the tracker, and do not
  hand it a ramped reference over a long partial-throttle slice.
- Do not add a second box row duplicating the w>=0 bound: SLSQP stops at
  iteration 1 on a promoted zero-impulse burn. Do not read SLSQP's multipliers
  on the two in-plane-degenerate plane rows (arbitrary; spurious switch).
- Compiles take 8-15 min and 1+ GB; never launch one without being asked.
