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

Open: duration (fixed vs free), cost form (per-axis fuel vs |F|), mass
(constant first vs rocket equation), first-build order.
