# Orbital craft solver: decisions (user, 2026-10-02)

Status: decisions only; nothing built. The orbital transfer sympy set
(`src/transmogrifier/orbital_transfer.py`) stays the compile-as-written
benchmark (`tests/test_orbital_transfer_compile.py`); this is its
manifestation in the dt system and compiler.

1. **Strictly lockstep.** The craft runs in the lockstep dt system
   (`examples/llvm_dt_system.py`: instantiate_system once, advance_round),
   pieces manifested the Woodshop way (`engine_toy/woodshop.py`
   `newton_world_dt_pieces`): catalogue laws, column spelling, explicit
   time discretization, `equation_piece`. Arc length becomes time.
2. **Craft laws assemble the actuation matrix.** Any thruster and mass design
   plugs in through craft laws that build the matrix from control signals
   (valves, producers, throttles) to forces; a live craft solver produces
   those signals, and they are integrated into actions in lockstep.
3. **Prototype jumper.** A craft-like jumper is acceptable for the
   development stages.
4. **Solver: both, as a pipeline with two states.**
   - ON PLAN: a planner (collocation over residual pieces, needs the
     residual Jacobian -- the live differentiator) holds the whole remaining
     trip; the live controller tracks it.
   - OFF PLAN: when tracking error exceeds a threshold, re-plan. Oscillation
     is allowed while it re-homes.
   - The intention is ALWAYS the total trip from the current moment: every
     re-plan targets the final orbit from the present state (a shrinking
     horizon over the whole remaining transfer), never a short local horizon.
   - Implied: a hysteresis band on the threshold so on/off does not chatter;
     the re-plan's starting guess is the old plan's remainder.
5. **Provenance.** Import from the original orbital module what can be
   imported (the gravity forms, the structure of the set) to show
   development provenance; build what is new in the dt-system and compiler
   region.

## Further decisions (user, same day)

6. **A live game.** There is a fixed flight plan, but the intention is a live
   game: clicking on a mass transfers the craft to it, with the thruster
   activations animated.
7. **Cost.** An alpha/beta mix of total average fuel-consumption rate and
   time to arrive, plus a steep ("big, scary") run-out-of-fuel penalty that
   rises sharply as remaining fuel approaches zero. Deviation from the plan
   belongs in the cost too (user, same day), but in the live controller's
   cost and the off-plan threshold, not the planner's: each re-plan starts
   from the present state, so the planner's deviation is zero by
   construction; the tracking controller weighs plan deviation against fuel
   and the off-plan switch fires on it.
8. **The seam is r() and F().** Whatever the craft becomes, it interfaces
   with the trajectory r() and the force F(). Everything up to that seam can
   be developed now, before any further craft question is answered; the
   early jumper sits behind it.
9. **End goal for the craft.** Mass falls as fuel burns, at a rate set by the
   thruster type (specific impulse per thruster kind); the craft has its own
   machine-sim-style but simplified moment and inertia (attitude dynamics),
   so thrust direction comes from orientation and actuators, not from
   ideal axis forces.

Build order: (1) jumper behind the r()/F() seam as lockstep dt-system
pieces; (2) actuation matrix from control signals to F; (3) live tracking of
a fixed plan; (4) collocation planner with the cost of decision 7;
(5) off-plan switching with hysteresis and warm-started full-trip re-plans;
(6) the game: click a mass, transfer, animated thrusters; (7) fuel-burn mass
loss and attitude dynamics.
