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
