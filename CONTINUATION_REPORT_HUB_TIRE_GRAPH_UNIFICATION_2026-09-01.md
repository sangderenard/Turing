# Continuation Report — Hub/Tire Graph Unification & the Road to True Assembly
**Date:** 2026-09-01 · **Branch:** nogodsnomasters (turing/ is a nested git repo; the tree carries heavy concurrent user work — NEVER commit, stash, or checkout-path anything without being asked.)

Predecessor reports: `CONTINUATION_REPORT_VALIDATOR_GAME_DEPLOYMENT_2026-09-01.md` (deployment seam, pooled C emission, record relocation, physical-dtype bake, threaded viewer). This report covers the architecture pivot that follows it.

---

## 1. The directive (the whole point — read this first)

The user's design, in order of issuance, forms one coherent architecture:

1. **Eliminate the wheel object on the graph; the hub replaces it.** The hub carries on *its* graph the tire object, "making them full participants in both the vehicle and tire simulation and no longer double representing the wheel area."
2. **The tire is not a toroidal balloon.** The **inner profile of the wheel — the inside radius of the torus hole — becomes essentially metal**: a rigid barrier owned by the hub, moving with the hub frame. The membrane gets **zoned real physics**: tread, sidewall, and interior each with their own stretch/reaction laws. Place the tire **in shape, in position, at 1 atm**, then **inflate from the wheel** — the metal barrel automatically bounds the balloon at exactly the right size.
3. **Beads are NOT deleted.** They are live **sealed-gasket boundary conditions** against the bead seats. The seal is maintained by pressure itself (inflation seats the bead — the real mechanism). Under the right conditions we **tag events**: bead *burp* (transient vent while momentarily unseated) and *tire failure* (bead past the flange). Damage events, not springs. Do not weld the beads; do not remove bead behavior.
4. **Graph disconnect/reconnect** becomes a first-class capability: wheels are built complete (hub + tire, inflated) **on pillars**, then, as in the original, **the suspension is constructed and solved in place** around them.
5. **End state:** tire prep, inflation, suspension assembly, and vehicle loading work "essentially like they first did, but even better and more accurate, **forcing the wheels until the stress falls off the chassis clamps**." I.e. the vehicle is assembled held by chassis clamps; the wheels are forced into load-bearing position; when the load path fully transfers to the tires the clamp stress goes to zero and the clamps release. That stress handoff is the acceptance criterion for "assembled."
6. **Pneumatic-graph note (future work, user's words, do not implement yet):** auto-air on a wheel must *require* at least one pneumatic tube through the wheel, connected to a pneumatic graph. For our system the line goes back to the **wheel bearing, which holds volume** (real CTIS: an annular sealed chamber between two rotary seals at the bearing — the stationary-to-rotating interface). Supply arrives on the stationary side along/through the knuckle/spindle (not through the axle shaft, barring hollow-shaft military style). Additionally: structural tubes and pneumatic actuator circuits may opt in as **gas-volume/pressure-holding members of the pneumatic graph**. The existing graph skeleton already matches (`pneumatic_rotary_union` → `drilled-hub-air-passage` → `pneumatic_wheel_valve`, `rotates_with=hub`).

## 2. What was DONE this session (Stage A — graph collapse) — landed, tests in flight

All in `src/compiler/abstract_ui_vehicles.py` `_vehicle_mechanical_graph` unless noted:

- **`wheel_rim` and `tire_carcass` nodes deleted** from `points` (they were coincident with `hub` at `[x, hub_y, hub_z]` — pure double representation). `hub` is the single wheel-area node, with an explanatory comment at the site (~line 5788).
- `moving_mass_components`: `"hub": f"wheel_{corner}"` — hub now carries the 68 kg wheel mass.
- `moves_with` knuckle set: `hub` added, dead names removed. Type-chain branches for the dead names removed.
- **Edges**: `hub_to_wheel` **deleted** (its `polar_inertia_kg_m2` was a duplicate — `wheel_bearing` knuckle→hub already carries `config.wheel_rotational_inertia()`); `rotor_mount` re-homed hub→brake_rotor; `tire_reference_frame` **deleted**; pneumatic valve `moves_with` → hub; `drilled-hub-air-passage` `rotates_with` → hub; `pneumatics.tire_valve` target → hub (terminal stays `{prefix}.tire_skin.closed_volume`); **bead edges** (`tire_skin.bead_R_N`, ~line 5923) re-homed `a: hub`.
- **Key discovery:** the tire skin was ALREADY first-class on the graph — 128 `tire_skin.vertex_N` nodes (each `mass_in_total=True` at `tire_mass_kg/128`), membrane faces, bead edges. So **`tire_carcass`'s 14 kg was a genuine double count in the vehicle totals** — the skin vertices already sum to 14 kg. Removing the carcass node is a mass-accounting *correctness fix*, and unsprung totals derived from `component_masses` elsewhere (line ~372) should be audited for the same duplication (NOT yet audited — see hangups).
- **Consumers updated:** `vehicle_native_deployment.py:1340` viewer `wheel_nodes` → `suspension.{corner}.hub`; `abstract_ui_div_map.py:6998` JS rotate list dropped the two names; `tests/test_abstract_ui_vehicles.py` — position assert now on hub + absence asserts for the dead names (line ~346), mass asserts now hub==68 and per-corner skin-vertex sum ≈ 14.0 (line ~1276).
- `wheel_bearing` (knuckle→hub, `rotational-bearing`) is now the designated **separable connect point** for Stage C disconnect/reconnect, and per the pneumatic note it is also the future annular air-chamber volume node. Nothing separable implemented yet — it is only *identified*.

**Verification state:** two targeted tests (`test_silver_upright_mass_is_connected_to_the_real_unsprung_corner_graph`, `test_living_map_has_a_vehicle_slot_not_a_car_specific_control_mode`) were still running in background at close (the living-map test alone runs many minutes; full file took ~7 min for 6 tests). Check task output `bdik6tr19` or re-run those two by name. **No bundle rebuild has been done since the collapse** — the deployed bundle predates all of this.

## 3. Stage B spec (NOT started) — the rigid-rim zoned tire

This is the next major work. Locus: `vehicle_balloon_tire.py` (ABI/params), `vehicle_balloon_tire_program.py` (authored physics — the thing the compiler compiles; DO NOT hand-write C), topology generator, and the bridge in `vehicle_native_deployment.py`.

- **Rim barrel = rigid barrier surface in the hub frame.** Unilateral: membrane/gas cannot penetrate; it closes the pressure volume (bead-to-bead wall contributes its hub-driven area to the volume integral). It is geometry OF the hub object.
- **Beads = gasket contact against bead seats**: unilateral seat constraint + interference/friction preload; pressure force seats them. Monitor seal state → `bead_burp` (vent while unseated beyond tolerance) and bead-past-flange → tire failure tag. This *replaces* the current bead Kelvin–Voigt springs (`bead_stiffness_n_per_m` etc. — currently the stiffest mode in the sim, forcing ~49 kHz substeps). Expect a large legal-dt win, but the dt system remains authoritative — measure, don't assume.
- **Zoned membrane**: per-face material class by section angle (tread band / sidewalls / bead region): distinct stiffness, damping, bending, thickness, from real-construction values in config. Current single StVK+Kelvin law becomes the per-zone parameterization.
- **Init & inflation**: mesh at molded shape, seated on rim, gas charge = 1 atm; inflation is a valve mass-flow process from the wheel (`state_channel tire_pressure_<corner>` already exists; `gas_charge_fraction` already exists in the ABI). The original's inflation scene becomes an actual simulated process.
- The **seat-at-hub bridge initializer** in `vehicle_native_deployment.py` (`balloon_tire_appendage_initialize` constructing the ring at hub pose from `seat_*` index arrays) is IN SOURCE but was never verified — its rebuild was killed. Stage B's 1-atm-seated init supersedes its spirit but the hub-frame placement machinery it built is exactly what Stage B needs.

## 4. Stage C spec (NOT started) — disconnect/reconnect, pillars, clamps

- Graph items must support **disconnect/reconnect at runtime**: build hub+tire assemblies complete on pillars ("THE PILLARS HOLD THE WHEELS, FROM THE START" — this is how the original works; READ the original), then construct the suspension and **solve it in place**.
- **Chassis clamps** hold the vehicle during assembly. Loading = forcing the wheels into position until the measured clamp stress falls to ~0, then release. Clamp members need stress instrumentation; the handoff is the assembly acceptance test and should become a battery stage.
- The 21-stage assembly battery (`native_vehicle_assembly_stages`, `tools/run_vehicle_native_assembly.py`) is the natural home; stages should be reworked to follow this true assembly order.

## 5. Hangups & outstanding issues (honest list)

- **Pre-existing failure in the user's uncommitted WIP:** `test_vehicle_engine_profile_switch_includes_aircraft_and_heavy_diesel_without_runtime_compile` fails with `NameError: 'suspension'` at `abstract_ui_vehicles.py:7741` (chassis_leveling `hydraulic_authority` block, a `+` hunk of user work; no `suspension =` binding in scope — likely wants `config.source["suspension"]`). NOT caused by the collapse; NOT fixed per standing order (don't modify the user's authored Python unasked). Tell the user.
- Stage-A tests were still in flight at close; bundle not rebuilt since the collapse. Next build: `tools/build_vehicle_validator_native.py --contract deploy` (O3) or `--optimization O0` for visible debugging — **O0 WITH POOLING is the debug standard**; never O1.
- **Unsprung-mass audit**: `wheel_mass + tire_mass + ...` at ~line 372 may still count tire mass that the skin vertices now carry — check whether corner unsprung parameters double the 14 kg.
- Viewer node-classification cosmetics: hub renders as hub-type now (former carcass drew tire-colored); harmless, revisit with Stage B visuals.
- Carried from predecessor report: 6 spec tests in `test_ssa_c_aggregate_constants.py` (storage-contract style); LLVM-IR lane 43 named shortfalls; GPU dialect widening + planned-region inlining (the standard: shaders auto-launched where valuable); unnamed captured cells feed-name contract; `material.telemetry` public-ABI restoration (currently a zeroed stub with receipt); `TURING_POOL_TLS` `__declspec(thread)` ignored on mingw → use `_Thread_local`; O3 release build + 21-stage battery on a fixed bundle.

## 6. Ethos & principles (violations of these caused every disaster this arc)

1. **Original sources go through the compiler.** The bespoke-C "original" is REFERENCE ONLY — for behavior, assembly order, and scenes. Never link it as product physics, never gate the dt system with it. This was violated once; the fallout was severe and total reversion was required.
2. **No band-aids, no stencils, no host-side patches.** Fix causes (the rim-wrench NaN was an analytically cancellable expression, not a clamp-to-zero; the hub-0 deadness was record-parameter privatization + a bool bake, not a display issue). When a value looks wrong, find WHERE IT'S BORN.
3. **The dt system is authoritative.** 1024 Hz outer windows are fine only because the managed substepper remains in charge. In-game simplification (token physical responses) is a later, separate decision.
4. **Deployment demands are firm.** Everything deployable deploys (pools now; compute shaders where auto-determined valuable; Fortran/LLVM for code; C as outer shell; video double-buffered non-blocking). Refusals must be named receipts, not silence.
5. **Don't re-run known results** (TEST_BASELINE_AND_HAZARDS.md); don't baseline via stash or checkout-path; show output inline, never `> file`; O0 for debug visibility; expensive turing builds are normal at 10+ min.
6. **Single representation.** The hub/wheel collapse is the pattern: wherever two graph objects shadow one physical thing (mass, inertia, position, volume), one must die. The double-counted tire mass and duplicated polar inertia found here suggest auditing for more.
7. **Ask-before-substitute.** Design changes to authored physics happen only on the user's explicit direction; present diffs for review when history is hot.

## 7. Immediate next steps, in order

1. Confirm the two in-flight Stage-A tests pass (task `bdik6tr19`); run `tests/test_abstract_ui_div_map.py` if it asserts on the JS list.
2. Report the pre-existing `suspension` NameError to the user; fix only on direction.
3. Rebuild O0 pooled bundle; confirm viewer wheels render at hubs on the pillars; then Stage B per §3.
4. Unsprung-mass audit (§5).
5. Stage C per §4, battery rework, clamps-stress acceptance.
