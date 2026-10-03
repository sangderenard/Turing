# CONTINUATION: orbital craft step 6, the live game

Owner files: `engine_toy/orbital_game.py`, `engine_toy/tests/test_orbital_game.py`.
Do not edit the other `orbital_*.py` (other agents own them).

## 2026-10-03 findings

- Viewer pattern reused: `engine_toy/turret_demo.py` (pygame OPENGL window,
  `gl_text.TextLayer`, a plan-view pygame Surface uploaded with
  `TextLayer.draw_surface`, headless `--frames` mode = `pygame.HIDDEN` +
  `glReadPixels` -> PNG). No `shots/` directory exists; turret_demo saves
  `{out}_{NN}.png` in cwd.
- Live loop: per frame, `orbital_tracker.fly(craft, plan, gains,
  until_s=t+frame, round_s=10)` (fly is the tracker's own per-round loop);
  targets are thrusterless `OrbitalJumper`s (same N4.1/N1.2/N1.1 pieces),
  advanced with the same window -> lockstep at frame boundaries.
- `fly()` zeroes throttles on return, so plumes read the per-thruster
  impulse deltas (`thruster_impulses_n_s`, the thrust-cost piece output)
  over the frame: frame-average clamped throttle = dI_k / (dt * T_k).
- Tracker exposes no on/off-plan state (decision 4 ON-PLAN only); the game
  shows plan deviation |r - r_ref| (FlightReport.final_position_error_m).
- No phasing in `orbital_plan`; added as laws in orbital_game (eq_PH1_*),
  composed with KEPLER_LAWS eq_KE1_6, compiled with `equation_piece`
  (sympy Mod is in the process-graph rule table).
- `orbital_collocation.py` does not exist yet; switch = `planner=` name.
- 2026-10-03: phasing piece `orbital_game_phasing` compiles (sympy Mod +
  sign + Abs through the sanctioned route, ~95 s first build); 4 raising/
  lowering cases put the target at phase+pi at arrival to 1e-9 rad, wait
  in [0, synodic). Co-orbital refused by PHASING_SCALE.
- 2026-10-03: BLOCKED (transient): engine_toy/orbital_actuation.py was
  found truncated to 45 lines mid-docstring ("build steps 2 and 7 ...")
  -- another agent's step-7 rewrite in progress (or the Windows UTF-8
  write truncation). Not touched by me. Waiting for it to parse before
  running the flight test.
- 2026-10-03: orbital_actuation.py now parses (475 lines, step 7: attitude
  R, propellant supply/flow). orbital_jumper.py is NOT yet migrated
  (393 lines): instantiating any OrbitalJumper fails
  `KeyError: 'attitude_xx'` in llvm_dt_system.instantiate_state. The game's
  flight test is blocked on the step-7 agent migrating the jumper; not mine
  to edit. Phasing tests (5) pass independently.
- 2026-10-03: jumper migrated (570 lines). Game switched to the declared
  propellant (bipropellant six-axis, 5000 kg wet / 4000 kg propellant);
  my invented impulse budget removed. NEW FAILURE (not mine): a
  thrusterless OrbitalJumper (mass_kg=..., thruster_count 0) no longer
  compiles: piece `orbital_craft_actuation_t0`: "return: output %t3 cannot
  render as i64; output %t4 cannot render as i64" (ssa_llvm_backend.py:6499)
  -- the t0 actuation outputs are literal integer zeros. Step-1 thrusterless
  tests should hit the same. Game targets therefore declared as stations
  with the six-axis ideal RCS design held at zero throttle (same compiled
  pieces as the craft); revert to mass_kg once t0 compiles.
- 2026-10-03: first flight (7000 -> 8000 km, wait 600 s): phasing exact
  (wait 600.000 s), craft-vs-plan 843 m at t_burn2, but craft-target
  46.5 km. The error is the STATION: its integrated orbit leaves the
  analytic circle by 46 km over 3832 s (|r|-r2 = +2.9 km and growing
  ~11 m/s) at dx = 5e4 m (CFL dt ~3.5 s, symplectic Euler, e ~ w dt/2).
  Also: at t_burn2 the finite arrival burn has not run yet (rel v 239 m/s
  = dv2), so the rendezvous must be measured after the burn settles.
- 2026-10-03: station dx 5e3 m -> station off its analytic circle by
  4.8 km at arrival (first order in dt, as expected). Rendezvous measured
  at t_burn2 + ARRIVAL_SETTLE_S (300 s = 6/w of default gains):
  distance 4929 m, rel speed 9.51 m/s, plan deviation 681 m.
- 2026-10-03: propellant 1267 kg = 1.71 x Tsiolkovsky ideal (740 kg).
  Split: coast before burn 1 = 29 kg/600 s (tracker fighting the craft's
  dx=5e4 integration error vs the analytic circle); burn 1 impulse
  2.02e6 vs m0*dv1 1.24e6 N*s (1.64x: PD velocity response
  v0(1-wt)e^{-wt} reverses sign -> sum|a| ~1.27 v0, plus spread over
  ~50 s). Tracker-owned; reported, not changed.
- 2026-10-03: GAME LAUNCHES headless (python orbital_game.py --click 0
  --frames N --every K). Kestrel (7000 -> 9000 km): wait 1h23m15s,
  transfer 59m20s, rendezvous 0.403 km @ 7.40 m/s; propellant 48.6% left
  (2054 kg used vs Tsiolkovsky ideal ~1266 kg incl. 83 min of coast
  station-keeping against the craft's own integration error).
- 2026-10-03: COST. Per 120 s frame: 4 stations at dx 5e3 = 3.23 s wall,
  craft coast 0.16 s, tracked craft frame ~0.25 s. The stations dominate
  (10x substeps for accuracy). Hook to request (not mine): OrbitalJumper
  with batch > 1 (orbital_jumper_dt_pieces already takes batch) so the
  stations run as ONE dt state; CLI --station-dx trades accuracy for speed.
- 2026-10-03: coordinator direction (machine craft coming in
  orbital_craft_machine.py): drawing now reads ONLY declared parts via
  `craft_part_geometry(craft, throttles)` (Thruster.position_m/direction
  carried by craft.attitude()) and `craft_body_corners(craft)`
  (CraftDesign.body_size_m); map icon + panel inset "craft (top view,
  declared parts)" both drawn by `_draw_craft`. A gimballed machine craft
  plugs its current directions into craft_part_geometry. 2-D projection;
  the machine GL mesh path was NOT used (open defect: GL render of a
  machine writes no PNG and exits 0).
- 2026-10-03: headless burn shot verified (--burn-shots): -x/-y plumes
  leave the +x/+y mounts during the arrival burn. Tests: 7 passed (66 s).
- 2026-10-03: coordinator update (MachineCraft 6b20ed8, batched stations,
  fly() on/off-plan). CHECKED FIRST: tests/test_orbital_jumper.py::
  test_interleaved_states_each_run_their_own_program -> still XFAIL
  (strict), i.e. the llvm_dt_system defect (bind_pieces module global:
  every state runs the program of the LAST instantiated state) is live.
  The requested game has a MachineCraft state + a batch-N station state =
  two different programs in one process -> it bites by construction.
  STOPPED per instruction; no game changes made for this update. Current
  orbital_game.py is unaffected (craft + 4 batch-1 stations all build the
  identical six-thruster/one-center program).

## 2026-10-03 (after a2020e0b) the game on the machine craft

Files: engine_toy/orbital_game.py, engine_toy/tests/test_orbital_game.py
(only). Uncommitted.

- Craft = `MachineCraft([center], orbital_craft(), ...)`; round 2 s and
  `TrackingGains(attitude_frequency_rad_s=0.2)` = the configuration of
  tests/test_orbital_tracker.py::test_machine_craft_transfer_arrives_with_
  gimballed_burns (MACHINE_GAINS, ROUND_S). The old six-axis 100 kN design,
  STATION_DESIGN, craft_part_geometry and craft_body_corners are gone.
- Stations = ONE `OrbitalJumper([center], mass_kg=1e3, batch=N)` (plain
  thrusterless lanes), advanced in ONE round per frame (`stations.advance(
  window)`). Lanes vs their analytic circles after the Kestrel flight:
  1.5 / 0.5 / 0.1 / 0.0 m. The batch-4 propellant-supply lane-0 defect does
  NOT show here (no thrusters, no propellant).
- fly() gets the tracker's own `hohmann_replanner` (planner "hohmann";
  collocation: none wired, see hooks), so `FlightReport.on_plan` /
  `.replans` are live; the game keeps the last report (`game.report`,
  `on_plan()`, `replans()`), draws the plan RED when OFF PLAN, says
  "OFF PLAN: re-plan n", and reads the FLOWN plan (`tracker_mode(craft).
  plan` while its origin is the order's plan) for the curve, the burn
  windows and the arrival time.
- Drawing: `craft_drawing(craft, loaded_cm) -> CraftDrawing`: parts =
  `craft.thruster_geometry()` (mount about the CURRENT CoM, exhaust at the
  CURRENT gimbal state, delivered throttle), all placed about the fixed
  prism centre (machine_package.measure_prism) so the CoM moves; tanks
  filled to their contents; CoM marker (quartered disc) + ring at the
  loaded CoM. The shift is mm on a metre craft, so the inset draws it x10
  (COM_SHIFT_MAGNIFICATION, labelled). Plume length by declared role (main
  1, brake 0.6, RCS 0.35), drawn on top of the body.
- Readout: propellant per tank (kg, % of loaded), main gimbal tilt from the
  gimbal STATE and from the tracker's last `TrackingCommand.gimbal_rad`,
  CoM moved (mm), groups by declared role (main / retro / RCS): firing now
  (delivered throttle) and fired over the frame (impulse delta), tracker
  ON/OFF PLAN + re-plans, wall seconds per frame.
- FRAME TIME (one 120 s frame during the phasing wait, pieces cached,
  median): before (HEAD 0b9ed22: six-axis craft at 10 s rounds + 4 batch-1
  stations chunked by 10 s) 3.02 s; after 1.65 s = stations 1.03 s (one
  batch-4 round at dx 5e3) + craft ~0.6 s (60 rounds of 2 s with
  allocation). Construction 14.9 s (cached).
- RENDEZVOUS, default click (Kestrel, 7000 -> 9000 km; wait 1h23m15s,
  transfer 59m20s), measured at t_burn2 + ARRIVAL_SETTLE_S (300 s):
  * 120 s frames: 200.8 m @ 3.03 m/s; plan deviation 200.7 m; ON PLAN, 0
    re-plans (max eps 4.7e-4); biprop 244.5 kg = 1.021 x TS2.1 ideal
    239.5 kg; hydrazine 24.1 kg; main gimbal up to 2.80 deg; retro never
    lit; CoM moved 20.0 mm. Before (old craft): 26.0 m @ 0.386 m/s.
  * 60 s frames (the default warp, the headless run): 76.4 m @ 1.30 m/s.
  * after the 300 s mark the craft is still trimming: +100 s 42.4 m /
    0.686 m/s, +200 s 11.2 m / 0.151, +300 s 4.3 m / 0.041, then a
    deadband limit cycle 1-9 m / ~0.03 m/s (coast_position_deadband 1e-6
    x 9e6 m = 9 m).
  Cause as read (tracker-owned, not changed): burn 1 closed with
  remaining -1.37 m/s (overdelivered). The 4 kN main cannot throttle below
  its 0.4 deadband: 0.4 x 4000 N x 2 s / 900 kg ~ 3.5 m/s per round, so a
  burn closes at round granularity. fly() also clips its last round to the
  frame boundary, so the frame length moves the round grid against the
  burn times; that is why 60 s and 120 s frames differ. ARRIVAL_SETTLE_S
  (300 s, from the old 100 kN craft) is short for this craft; changing it
  is a measurement choice I left for the user.
- Tests (tests/test_orbital_game.py, 7 passed, ~3 min):
  test_click_flies_the_machine_craft_to_the_target (8000 km probe + a
  second lane: 450.1 m @ 4.01 m/s, biprop 1.113 x ideal, ON PLAN,
  stations 1.25 / 0.10 m off their circles) and
  test_drawing_reads_the_machine_parts_gimbal_and_centre_of_mass
  (geometry = thruster_geometry about the prism centre, main exhaust at
  the tilted gimbal state, CoM moved after a 20 s main burn, tank fills).
- Shared piece cache, two transient faults while other lanes were running
  (not mine, recorded): (1) `LLVMPiece.load` raised MemoryError in
  pickle.load on orbital_machine_actuation (passes on a retry, so read
  as a partly-written .piece); (2) after the catalogue's piece-staleness
  change (equation_piece now rebuilds a cached piece whose compiler sources
  changed), lld-link "Permission denied" writing orbital_craft_actuation_t0
  b1's DLL while tests/test_orbital_jumper.py was rebuilding the same piece
  in another process. The cache has no write lock and no atomic
  rename.

HOOKS NEEDED (other owners):
- orbital_collocation: a re-planner the game can build from (mu, r2) or
  from the order's plan (`collocation_replanner` takes a
  CollocationProblem), so the collocation planner gets ON/OFF PLAN too.
- orbital_tracker: burn close at round granularity vs the main deadband
  (see above); `fly()`'s last round clipped to `until_s` puts the round
  grid on the caller's frame boundaries.
- honorary_engine_equation_catalogue.equation_piece: lock + atomic write
  of the .piece/.dll so concurrent lanes do not read partial pieces or
  collide on the DLL.

## 2026-10-03 12:00 REGRESSION (turing, not mine): phasing piece rebuilt wrong

- The catalogue's new staleness check rebuilt `orbital_game_phasing` (book
  row: decision 'rebuilt', '<no compiler record>') at 12:00 against the
  current turing tree (HEAD 822a753b "elementwise Select ... is where()",
  plus many uncommitted src/compiler edits from other lanes). The new
  piece returns t_wait = 0 and phase = 0 for the Kestrel case
  (lead_angle 0.5088 is still right; expected t_wait 4995.3 s). The Mod /
  sign / Abs part of eq_PH1_4 is what broke.
- tests/test_orbital_game.py phasing: 3 of 4 cases FAIL now (2 passed:
  the lowering case and the co-orbital refusal); they passed at about 11:40
  with the old piece. Repro (seconds): `phasing(MU, 7e6, 9e6,
  hohmann_plan(MU,7e6,9e6).transfer_time, pi, 2.2, 0.0)`.
- The shots in engine_toy/shots/orbital_game_machine_*.png were rendered
  AFTER the rebuild: burn 1 fires at t=0 (unphased), the tracker goes OFF
  PLAN and re-plans once (the shots show the red OFF PLAN state), and there
  is no arrival within the 24 shots. The phased run before the rebuild (the
  numbers above) arrived 76.4 m @ 1.30 m/s at the 60 s warp. To get phased
  shots again: fix the compiler, then rerun the same command.
