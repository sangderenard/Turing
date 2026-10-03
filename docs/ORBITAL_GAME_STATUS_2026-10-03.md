# Orbital game: challenges and remaining work (2026-10-03)

## Current integration work on `wip/2026-10-03-inflight`

The user has prioritized correct dt-managed flight before planner, rendering,
and performance work. The original game used the existing dt outer coordinator
and rollback but authored its own leapfrog/Euler/Cayley updates. The correction
now constructs coupled updates by calling the actual library `RK4Integrator`;
the existing PieceState owns all physical and stage columns. Full production
flight acceptance is still pending. Detailed trace and measured gates:
[`concordance_census/ORBITAL_DT_METRICS_2026-10-03.md`](concordance_census/ORBITAL_DT_METRICS_2026-10-03.md).

Verified Turing changes: appended metric-channel transport (`3202f8ba`),
participant energy contracts taking precedence over the legacy aggregate
fallback (`8ab64810`), and output-only state registration (`302dcbc3`). The
small coupled native rotor gate passes with actual rejection and rollback;
it does not establish complete craft/game acceptance or concordance closure.

The tracker no longer predicts dt substeps. Its exact native throttle-ramp
observable passes three focused tests; flight progress now reads the applied
delta-v integrated by the same library stages. The frame-comparison gate
remains pending. Collocation adapter edits and prior screenshots remain
unverified as game acceptance. Gravity is still exerted only by the declared
fixed centers; craft and stations have no mutual gravity or backreaction.

The remainder of this report records the inherited state and historical
measurements, which must not be treated as results for the new integration.

Scope: only the orbital game (`engine_toy/orbital_game.py`, its drawing, its tests, and what it depends on).
For everything else see `docs/concordance_census/CONTINUATION_SESSION_2026-10-03.md`.
Goal stated by the user: a Kerbal-Space-Program-grade orbital game, eventually as a web demo. The design
decisions behind it are in `docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md` (decision 6: click a mass, transfer
to it, thrusters animated; decision 7: alpha/beta fuel-rate and time cost, steep run-out penalty).

## 1. What exists

- **Run it** (from `C:\dev\Powershell\engine_toy`): `python orbital_game.py`. Click a station or press 1-9 to go there;
  `+`/`-` change time warp (default 60 s per frame). Headless: `python orbital_game.py --click 0 --frames 24 --every 60
  --burn-shots --out shots/orbital_game_machine`. Other flags: `--warp`, `--planner {hohmann,collocation}`,
  `--station-dx` (default 5e3 m; 5e4 is about 5x faster but the stations drift ~46 km).
- **Code:** `orbital_game.py` (996 lines), `tests/test_orbital_game.py` (178 lines, 7 tests, ~3 min).
  Commits on `nogodsnomasters`: 2f3666d (first game), 607a3b0 (machine craft, batched stations).
- **How it works:** the craft is a `MachineCraft` (engine_toy machine: gimballed 4 kN main, 2 brake, 16 RCS,
  per-tank propellant, centre of mass from the machine reduction). The stations are ONE batched
  `OrbitalJumper(batch=N)` of massless bodies on circular prograde orbits around one gravity centre. A click builds a
  PHASED Hohmann transfer: the wait-and-lead computation is a set of compiled laws (`eq_PH1_1..5` on `eq_KE1_6`,
  via `equation_piece`); co-orbital targets are refused. Each frame calls `orbital_tracker.fly` for the craft and
  advances the stations over the same window. The screen shows the plan, the flown path, fuel per tank, main gimbal
  angle, firing groups, ON/OFF PLAN and the re-plan count (from `fly()`'s report).
- **Drawing reads declared parts:** `craft_drawing` takes mounts and exhaust directions from
  `craft.thruster_geometry()`, so the main plume tilts with the gimbal and the centre-of-mass marker moves as tanks
  drain (its shift is drawn 10x because it is millimetres). 2D pygame projection; the engine_toy GL path is not used.

## 2. Measured state (before the later fixes noted in section 3; re-measure)

| Quantity | Value |
|---|---|
| Frame cost, one 120 s frame during the phasing wait | 3.02 s before, 1.65 s after (stations 1.03 s, craft ~0.6 s = 60 two-second rounds with allocation) |
| Kestrel rendezvous (7000 -> 9000 km, wait 1h23m, transfer 59m), 60 s frames | 76.4 m, 1.30 m/s, measured 300 s after arrival time |
| Same, 120 s frames | 200.8 m, 3.03 m/s |
| After the 300 s mark | 42.4 m / 0.69 m/s at +100 s, 11.2 m / 0.15 m/s at +200 s, 4.3 m / 0.04 m/s at +300 s, then a 1-9 m deadband limit cycle |
| Fuel, Kestrel, main engine | 244.5 kg biprop = 1.021x the rocket-equation ideal (239.5 kg); hydrazine 24.1 kg |
| 8000 km test, craft + second lane | 450.1 m / 4.01 m/s, biprop 1.113x ideal, ON PLAN; stations 1.25 m / 0.10 m off their circles |
| Construction (pieces cached) | 14.9 s |

## 3. Challenges, by kind

**A. Correctness**
1. `--planner collocation` is BROKEN. `_collocation_planner()` imports `orbital_collocation.collocation_plan`, a name
   guessed before the planner existed. The real API is `plan_transfer(problem, t0, position, velocity, ...)`,
   `CollocationProblem`, `CollocationPlan` and `collocation_replanner(problem, craft=...)`. The game also does not
   wire the collocation re-planner (a comment at the replanner assignment says so), so no ON/OFF PLAN with
   collocation, and `PLAN_SWEEP_RAD["collocation"] = pi` was set by assumption.
2. The phasing piece regression (wait 0 instead of 4995.3 s) is FIXED in turing a7f57c1e (`Mod`/`FloorDiv` were an
   unconditional integer seed). It hid behind a stale symbolic cache. After it, the game tests were 7/7; they have
   not been re-run on the final tree.
3. Untested behaviour: clicking a second target in mid-transfer. Design intent is "always the total trip from the
   current moment"; nothing in the tests covers it.

**B. Flight quality**
1. Burns close at round granularity: the main engine cannot throttle below 0.4, so one 2 s round is ~3.5 m/s; burn 1
   overdelivered by 1.37 m/s, which the tracker then cleans up (the 76 m / 1.3 m/s above).
2. `fly()` clips its last round to the caller's frame boundary, so frame length moves the round grid against the burn
   times: 60 s and 120 s frames give different results. The game's result depends on the warp setting.
3. `ARRIVAL_SETTLE_S = 300 s` came from the old 100 kN craft; the machine craft needs ~600 s to settle. This is a
   measurement choice left for the user.
4. Attitude is the user's live complaint ("it's having a hard time"). The craft has only RCS for torque; the main
   engine's centre-of-mass lean (2.37 degrees) is handled by the gimbal. Reaction wheels are the planned fix (see
   dependencies); their edits are unverified.
5. Rendezvous ends at 76 m and 1.3 m/s: close, not matched. There is no terminal approach or docking phase.
6. Fuel is not yet a game mechanic: no HUD for the alpha/beta cost, no run-out failure state, no total-propellant ratio
   in the HUD (hydrazine 24 kg of attitude work is invisible in the 1.021x figure).

**C. Performance (the wall to live play)**
- 1.65 s of wall time per 120 s game frame is a slideshow. Two costs dominate: (1) Python bookkeeping around the dt
  system, ~16 ms per substep of eager AbstractTensor copies around microseconds of native work (profiled before the
  spans work; one station frame went 1.55 s -> 1.0 s); (2) the allocator and tracker run in Python every 2 s round
  (5-15 ms per `allocate`, SciPy SLSQP).
- The structural fix is the dt-managed native compile of the whole dt system, which is NOT yet full-native (see the
  continuation report). Until then, `--station-dx 5e4` and larger warp are the only levers.
- Piece cache churn: any edit to a loaded `src.*` module marks pieces stale; they rebuild (minutes) on the next
  start. A lock and atomic publish landed (95e1e8d) after two concurrent-build faults.

**D. Rendering**
- The 24 screenshots in `engine_toy/shots/orbital_game_machine_*.png` were rendered with the BROKEN phasing piece
  (burn 1 at t=0, OFF PLAN, no arrival). Re-render after wheels land.
- The engine_toy GL render of a machine writes no PNG and exits 0 (open since September); the game uses a 2D projection.

**E. Realism gaps against the KSP goal**
- One gravity centre; no second body, no sphere-of-influence handoff, no atmosphere. (The craft-binding probe does
  use Earth + Moon, so the laws support it; the game does not.)
- Coplanar circular stations; Hohmann transfers only. The Hohmann re-planner is poor mid-transfer (it assumes a
  circular start: a 150 m/s kick became a 652 m/s re-plan). Collocation fixes this (411 m/s) but is not wired in.
- No maneuver-node UI, trajectory prediction line, camera control, sound, or failure/score loop.

**F. Web demo (not yet investigated; inferred from the code)**
- The loop is Python end to end: tracker, allocator (SciPy SLSQP), planner (SciPy SLSQP), pygame. Only the compiled
  laws could run in a browser. A WASM lane exists (`ssa_wasm_backend`, `compile_graph_reverse_to_wasm`), but the
  dt-system native compile that it would need is not done, and the allocator/planner need either Pyodide or a
  compiled solver. Rendering would move to canvas/WebGL.

## 4. Remaining work, in priority order

1. **Unbreak `--planner collocation`.** Write a `TransferOrder -> CollocationProblem` adapter, wire
   `collocation_replanner`, set `PLAN_SWEEP_RAD` from the plan instead of assuming pi. Accept: a click with
   `--planner collocation` arrives, with ON/OFF PLAN shown, and a test covers it (kick mid-transfer -> one re-plan).
2. **Re-run the game tests on the final tree, then re-render the shots** (command in section 1) once the wheels
   lane is verified. Accept: Kestrel arrives phased; shots show burn 1 at the planned time.
3. **Tracker hooks** (owner: tracker): end the burn at the exact cutoff, spool-down aware, RCS for the residual;
   make `fly(t0->t2)` equal `fly(t0->t1)` then `fly(t1->t2)`. Accept: Kestrel result identical at 60 s and 120 s
   frames; burn-1 overdelivery < 0.1 m/s. (The lane was stopped: with no gravity a cut costs 1.3 mm / 2.7e-4 m/s;
   the gravity case and the test are unwritten.)
4. **Settle window and success criterion:** choose from data (decision below); add a rendezvous result readout.
5. **Reaction wheels + dt metrics** (owner: craft; stopped, unverified on `wip/2026-10-03-inflight`): verify, then
   confirm the game's hydrazine drops from 24 kg and the attitude holds during burns.
6. **Frame time:** set a target and attack the two costs in section 3C (compiled dt system; allocate only when the
   wrench changes; lower allocation rate during coasts).
7. **Game loop:** mid-transfer re-targeting, terminal approach and speed matching, fuel/cost HUD with the run-out
   barrier, failure and score states.
8. **Realism:** second gravity body and SOI, non-coplanar targets, prediction line, camera and zoom, sound.
9. **Web path:** decide the stack (decision below), then port in this order: compiled dt system, solvers, renderer.

## 5. Dependencies

- Craft files (`orbital_actuation.py`, `orbital_craft_machine.py`, `orbital_jumper.py`) and `orbital_tracker.py`
  are owned by other lanes; the game only reads them through `craft.thruster_geometry()`, `craft.allocate`,
  `tracker_mode` and `fly()`'s report. Needed hooks: items 1 and 3 above.
- Wheels and tracker cutoff work are UNVERIFIED on `wip/2026-10-03-inflight` (root repo `ed4fa0a`).
- Whole-dt-system native compile (turing; see the continuation report, section 2).

## 6. Decisions for the user

1. Frame-time target and the warp policy (what wall-clock cost per game second is acceptable).
2. What counts as arrival: current 300 s measurement point, the 600 s the machine craft needs, or a matched-speed
   criterion.
3. Web target: Pyodide (SciPy in the browser) versus compiled solvers (QP/SLSQP replacements) versus a thin server.
4. Scope of "KSP-grade": second body and SOI, docking, and non-coplanar targets: which are in the first demo.
5. Default planner (Hohmann or collocation) and whether a 1-2 s re-plan may block a frame.

## 7. Verification checklist before calling the game "working"

`python -m pytest tests/test_orbital_game.py` (7 tests, ~3 min; piece rebuilds add minutes after compiler edits),
then the headless command in section 1 at 60 s and at 120 s frames, then read `orbital_game_machine_*.png`.
Never compile with `extraction_contract=None`; one build at a time.
